"""Two independent lane marking searches over clip contact sheets."""
from __future__ import annotations

import base64
import concurrent.futures as cf
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import qwen_lane_marking_visual_filter as vision

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)


def make_prompt(rules):
    common = """前面有一张已确认的本类别参考图，参考图只用于理解标线样式，不参与判断结果。判断应比较路面标线与指示装置的特征，不要根据车辆、建筑或背景相似度猜测。
随后是待筛的车载前视图，每张上方有 frame 编号。必须逐张观察，并对每个待筛帧给出明确判断，不能省略任何帧。没有目标特征则 match=false；疑似但模糊、遮挡则 match=false、uncertain=true。
只输出严格 JSON，不要 Markdown。decisions 数组每个帧一项，frame 为真实标签中的整数，match 和 uncertain 为 JSON 布尔值。命中时 reason 必须简短说明实际可见证据，未命中时 reason 可以为空字符串。
{"decisions":[{"frame":100,"match":true,"uncertain":false,"reason":"实际可见证据"}]}
必须使用真实帧号，而不是格式示例中的 100。编号为 0 的帧同样必须判断。
"""
    return {
        "可变车道": """你只负责识别可变导向车道（锯齿车道），重点观察路面两侧边界：白色纵向实线内侧连续、等间距排列向车道内部伸出的短斜线，形成两排锯齿状边界。短斜线可能因透视变得很小，要仔细查看近处车道边界。
看到这种锯齿标线即可命中，不要求存在电子指示牌或任何其他车道类型。两排沿车道边界向内伸出的短斜线，与车道中央的 V 形减速标记、普通直行箭头、导流区的大斜纹和单独虚线不同；后几种不能单独命中。
""" + common,
        "潮汐车道": """你只负责识别潮汐车道。满足下列任一组合即可命中：
A. 路面明确双黄虚线，且对应车道上方有独立红叉/箭头通行指示装置。
B. 同一组车道上方有成组红叉/箭头通行指示装置，车道内沿纵向连续排列多个白色 V 形标记，并伴随黄色边界线。黄色标线磨损时，不要求双黄虚线的每个细节都清楚。
C. 确实读到“潮汐车道”文字。
普通路口的圆形红绿灯、普通左转/直行信号灯不等于车道级红叉/箭头装置。只有 V 形标记、普通箭头、普通护栏或普通车道线不能命中。只按上述潮汐规则识别，不要求任何其他车道类型的标线。
""" + common,
    }


def make_sheet(samples, width, kind="潮汐车道"):
    tiles = []
    top, bottom = (0.35, 0.75) if kind == "可变车道" else (0.04, 0.73)
    cell_height = round(width * (bottom - top) * 9 / 16) + 32
    for frame, path in samples:
        with Image.open(path) as original:
            original.draft("RGB", (width, round(width * 9 / 16)))
            image = original.convert("RGB")
            w, h = image.size
            image = image.crop((0, int(h * top), w, int(h * bottom)))
            image.thumbnail((width, cell_height - 32), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (width, cell_height), "#eeeeee")
        tile.paste(image, ((width - image.width) // 2, 32))
        ImageDraw.Draw(tile).text((8, 4), f"frame {frame}", font=FONT, fill="black")
        tiles.append(tile)
    columns = min(2, len(tiles))
    sheet = Image.new("RGB", (columns * width, math.ceil(len(tiles) / columns) * cell_height), "#cccccc")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * width, (index // columns) * cell_height))
    return sheet


def parse_features(answer, valid_frames):
    raw = str(answer).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(raw)
    decisions = data.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(valid_frames):
        raise ValueError("Response must judge every input frame")
    seen = set()
    for row in decisions:
        if not isinstance(row, dict) or type(row.get("frame")) is not int or row["frame"] not in valid_frames or row["frame"] in seen:
            raise ValueError("Invalid or duplicate frame decision")
        if type(row.get("match")) is not bool or type(row.get("uncertain")) is not bool or not isinstance(row.get("reason"), str):
            raise ValueError("Invalid decision fields")
        if row["match"] and (row["uncertain"] or not row["reason"].strip()):
            raise ValueError("Positive decision must have clear visible evidence")
        seen.add(row["frame"])
    return {"frames": sorted(row["frame"] for row in decisions if row["match"]),
            "uncertain_frames": sorted(row["frame"] for row in decisions if row["uncertain"]),
            "evidence": [{"frame": row["frame"], "reason": row["reason"]} for row in decisions if row["match"]],
            "decisions": decisions}


VARIABLE_VERIFY_PROMPT = """只看这张图的道路近处，请回答路面是否能清楚看到两排白色锯齿标线沿着同一条车道左右两侧向前延伸。锯齿是连续短斜齿从两边向车道内部伸出，不是黄色线、导流区三角斜纹、路口箭头或路边停车位。只要清楚看到两排锯齿就回答 true，别自行增加其他条件。请用严格JSON输出：{"two_rows_white_sawtooth":true或false,"left_side_visible":true或false,"right_side_visible":true或false,"evidence":"一句话描述"}。看不清为false。"""

TIDAL_VERIFY_PROMPT = """独立核查这张道路照片，只报告实际清楚可见的特征，禁止推测。分别观察：1. 对应车道正上方是否有成组独立红叉X/箭头车道通行装置（监控摄像头、路灯、蓝色指路牌、普通圆形红绿灯均不算）；2. 路面是否有双黄虚线；3. 同一组车道内部是否连续排列多个白色V形标记（边界锯齿不算V形）；4. 这些车道是否有黄色边界；5. 是否清楚读到“潮汐车道”四个字。只输出严格JSON，所有字段必填：{"lane_control_signals":false,"double_yellow_dashed":false,"repeated_v_markings":false,"yellow_boundaries":false,"readable_tidal_text":false,"evidence":"具体看到的装置和标线，未看到也如实说明"}。看不清的字段填false。"""


def verify_candidate_frame(path, config, args, label):
    with Image.open(path) as original:
        image = original.convert("RGB")
        width, height = image.size
        if label == "可变车道":
            image = image.crop((int(.27 * width), int(.47 * height), int(.70 * width), int(.73 * height)))
            image = image.resize((1290, 780), Image.Resampling.LANCZOS)
        else:
            image = image.crop((0, int(.04 * height), width, int(.73 * height)))
            image.thumbnail((2048, 1600), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    image_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = {"model": config["EAS_MODEL"], "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": VARIABLE_VERIFY_PROMPT if label == "可变车道" else TIDAL_VERIFY_PROMPT},
    ]}], "stream": False, "temperature": 0, "max_tokens": 250,
        "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(config["EAS_ENDPOINT"] + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": config["EAS_TOKEN"], "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        answer = json.load(response)["choices"][0]["message"]["content"]
    raw = answer.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(raw)
    fields = (("two_rows_white_sawtooth", "left_side_visible", "right_side_visible") if label == "可变车道"
              else ("lane_control_signals", "double_yellow_dashed", "repeated_v_markings", "yellow_boundaries", "readable_tidal_text"))
    if any(type(result.get(key)) is not bool for key in fields) or not isinstance(result.get("evidence"), str):
        raise ValueError("Invalid lane verification")
    result["verified"] = (all(result[key] for key in fields) if label == "可变车道" else
        result["readable_tidal_text"] or (result["lane_control_signals"] and
        (result["double_yellow_dashed"] or result["repeated_v_markings"])))
    return result


def process_clip(clip, config, prompt, args, signature, fixture=None):
    started = time.monotonic()
    row = {"clip_dir": str(clip) if clip else "validation_fixture", "clip_name": clip.name if clip else "validation_fixture", "run_signature": signature, "ok": False}
    try:
        if fixture is not None:
            samples = [(record["frame"], Path(record["path"])) for record in fixture]
        else:
            files = sorted((clip / args.camera).glob("*.jpg"))
            if not files:
                return {**row, "skipped": True, "error": f"No images in {args.camera}", "sample_count": 0}
            indexes = sorted(set(range(0, len(files), args.stride)) | {len(files) - 1})
            samples = [(index, files[index]) for index in indexes]
        row["samples"] = [{"frame": frame, "path": str(path)} for frame, path in samples]
        row["sample_count"] = len(samples)
        row["observed_features"] = {"evidence": []}
        errors = []
        row["tidal_lane"] = False
        row["tidal_frames"] = []
        phases = [("可变车道", "variable")] if getattr(args, "variable_only", False) else [("可变车道", "variable"), ("潮汐车道", "tidal")]
        for label, prefix in phases:
            phase_started = time.monotonic()
            sheet = make_sheet(samples, args.thumb_width, label)
            content = [{"type": "text", "text": f"这是已确认的{label}参考图，不是待筛帧。"},
                       {"type": "image_url", "image_url": {"url": config["reference_images"][label]}}]
            inputs = [(frame, make_sheet([(frame, path)], args.thumb_width, label)) for frame, path in samples] if args.separate_images else [(None, sheet)]
            for frame, picture in inputs:
                buffer = io.BytesIO()
                picture.save(buffer, format="JPEG", quality=94)
                image_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                if frame is not None:
                    content.append({"type": "text", "text": f"以下图片对应 frame {frame}"})
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            content.append({"type": "text", "text": prompt[label] + "\n本次必须判断的待筛帧号：" + json.dumps([frame for frame, _ in samples])})
            payload = {"model": config["EAS_MODEL"], "messages": [{"role": "user", "content": content}], "stream": False, "temperature": 0, "max_tokens": args.max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
            error_text = ""
            row[prefix + "_ok"] = False
            row[prefix + "_frames"] = []
            row[prefix + "_lane"] = False
            for attempt in range(1, args.attempts + 1):
                try:
                    request = urllib.request.Request(config["EAS_ENDPOINT"] + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": config["EAS_TOKEN"], "Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=args.timeout) as response:
                        reply = json.loads(response.read(2_000_000))
                    answer = reply["choices"][0]["message"]["content"]
                    result = parse_features(answer, {frame for frame, _ in samples})
                    if result["frames"]:
                        sample_paths = dict(samples)
                        checks = []
                        for frame in result["frames"]:
                            check = verify_candidate_frame(sample_paths[frame], config, args, label)
                            checks.append({"frame": frame, **check})
                            if check["verified"]:
                                break
                        row[prefix + "_verification"] = checks
                        confirmed = [check for check in checks if check["verified"]]
                        result["frames"] = [check["frame"] for check in confirmed]
                        result["evidence"] = [{"frame": check["frame"], "reason": check["evidence"]} for check in confirmed]
                    row[prefix + "_ok"] = True
                    row[prefix + "_frames"] = result["frames"]
                    row[prefix + "_lane"] = bool(result["frames"])
                    row[prefix + "_raw_answer"] = answer
                    row[prefix + "_usage"] = reply.get("usage")
                    row["observed_features"][prefix] = result
                    row["observed_features"]["evidence"].extend({**entry, "class": label} for entry in result["evidence"])
                    break
                except urllib.error.HTTPError as error:
                    detail = error.read(8_000).decode(errors="replace").replace(config["EAS_TOKEN"], "[REDACTED]")
                    error_text = f"HTTP {error.code}: {detail[:300]}"
                    if error.code in (400, 401, 403, 404):
                        break
                except Exception as error:
                    error_text = f"{type(error).__name__}: {error}".replace(config["EAS_TOKEN"], "[REDACTED]")
                if attempt < args.attempts:
                    time.sleep(min(attempt * 2, 6))
            row[prefix + "_latency_seconds"] = round(time.monotonic() - phase_started, 3)
            if not row[prefix + "_ok"]:
                errors.append(label + ": " + error_text)
        row["ok"] = row["variable_ok"] and (getattr(args, "variable_only", False) or row["tidal_ok"])
        row["uncertain_frames"] = sorted({frame for prefix in ("variable", "tidal") for frame in row["observed_features"].get(prefix, {}).get("uncertain_frames", [])})
        if errors:
            row["error"] = "；".join(errors)
        if row["variable_lane"] or row["tidal_lane"]:
            key = hashlib.sha256(row["clip_dir"].encode()).hexdigest()[:12]
            target = args.out_dir / "contact_sheets" / f"{row['clip_name']}_{key}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(target, quality=94)
            row["contact_sheet"] = str(target)
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}".replace(config["EAS_TOKEN"], "[REDACTED]")
    row["latency_seconds"] = round(time.monotonic() - started, 3)
    return row


def copy_hits(row, args):
    if not args.review_root or not row.get("samples"):
        return
    if row["clip_dir"] == "validation_fixture":
        relative = Path("validation_fixture")
    elif args.root:
        relative = Path(row["clip_dir"]).relative_to(args.root)
    else:
        relative = Path(row["clip_name"])
    frame_paths = {sample["frame"]: Path(sample["path"]) for sample in row["samples"]}
    for label, key in [("可变车道", "variable_frames"), ("潮汐车道", "tidal_frames")]:
        frames = row.get(key, [])
        if not frames:
            continue
        target = args.review_root / label / "clips_with_photos" / relative
        target.mkdir(parents=True, exist_ok=True)
        copied = []
        for frame in frames:
            source = frame_paths[frame]
            dest = target / f"frame_{frame:06d}_{source.name}"
            if not dest.exists():
                shutil.copyfile(source, dest)
            copied.append(str(dest))
        (target / "selection.json").write_text(json.dumps({"class": label, "clip_dir": row["clip_dir"], "frames": frames, "images": copied, "evidence": [entry for entry in row["observed_features"]["evidence"] if entry["frame"] in frames and entry["class"] == label]}, ensure_ascii=False, indent=2))


def export_state(rows, args, metadata, status):
    records = sorted(rows.values(), key=lambda row: row["clip_dir"])
    variable = [row for row in records if row.get("variable_ok") and row.get("variable_lane")]
    tidal = [row for row in records if row.get("tidal_ok") and row.get("tidal_lane")]
    failed = [row for row in records if not row.get("ok") and not row.get("skipped")]
    metadata.update(status=status, processed_clips=len(records), successful_clips=sum(row.get("ok") is True for row in records), skipped_clips=sum(bool(row.get("skipped")) for row in records), failed_clips=len(failed), variable_clips=len(variable), tidal_clips=len(tidal), sampled_images=sum(row.get("sample_count", 0) for row in records), last_update_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    temp = args.out_dir / "progress.json.tmp"
    temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    temp.replace(args.out_dir / "progress.json")
    for label, hits in [("可变车道", variable), ("潮汐车道", tidal)]:
        target = args.review_root / label if args.review_root else args.out_dir / label
        target.mkdir(parents=True, exist_ok=True)
        (target / "clip_names.txt").write_text("".join(row["clip_name"] + "\n" for row in hits))
        (target / "clip_paths.txt").write_text("".join(row["clip_dir"] + "\n" for row in hits))
    (args.out_dir / "failed_clip_paths.txt").write_text("".join(row["clip_dir"] + "\n" for row in failed))
    fields = ["clip_name", "clip_dir", "ok", "skipped", "variable_lane", "tidal_lane", "variable_frames", "tidal_frames", "uncertain_frames", "sample_count", "latency_seconds", "error", "contact_sheet"]
    with (args.out_dir / "results.tsv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({**row, **{key: json.dumps(row.get(key, [])) for key in ("variable_frames", "tidal_frames", "uncertain_frames")}})
    return metadata


def run_batch(args):
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.root:
        args.root = args.root.resolve()
    if args.review_root:
        args.review_root = args.review_root.resolve()
    if args.detach:
        command = [sys.executable, str(vision.PROJECT / "scripts/qwen_lane_marking_visual_filter.py"), *[arg for arg in sys.argv[1:] if arg != "--detach"]]
        with (args.out_dir / "run.log").open("a") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (args.out_dir / "run.pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"pid": process.pid, "log": str(args.out_dir / "run.log"), "output": str(args.out_dir)}, ensure_ascii=False), flush=True)
        return 0
    if args.stride < 1 or args.thumb_width < 128:
        raise ValueError("Invalid stride or thumbnail width")
    lock = (args.out_dir / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = vision.load_configuration()
    rules = args.rules.read_text()
    prompt = make_prompt(rules)
    reference_links = re.findall(r"\[示例图片\]\(([^)]+)\)", rules)
    if len(reference_links) != 2:
        raise ValueError("rule.md must contain the two labeled reference images")
    config["reference_images"] = {label: vision.encode_image(args.rules.parent / link, args.max_side) for label, link in zip(("可变车道", "潮汐车道"), reference_links)}
    signature = hashlib.sha256(json.dumps({"prompt": prompt, "verification_prompt": [VARIABLE_VERIFY_PROMPT, TIDAL_VERIFY_PROMPT], "model": config["EAS_MODEL"], "endpoint": config["EAS_ENDPOINT"], "camera": args.camera, "stride": args.stride, "width": args.thumb_width, "max_tokens": args.max_tokens, "separate_images": args.separate_images, "rules_sha256": hashlib.sha256(rules.encode()).hexdigest(), "version": 6}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    (args.out_dir / "prompt_snapshot.txt").write_text(json.dumps(prompt, ensure_ascii=False, indent=2))
    (args.out_dir / "rules_snapshot.md").write_text(rules)
    fixture = json.loads(args.validation_manifest.read_text()) if args.validation_manifest else None
    clips = [None] if fixture is not None else [Path(line.strip()).resolve() for line in args.clip_list.read_text().splitlines() if line.strip()]
    if args.limit:
        clips = clips[:args.limit]
    if len(clips) != len(set(clips)):
        raise ValueError("Clip manifest contains duplicate source directories")
    rows = {}
    path = args.out_dir / "results.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("run_signature") == signature:
                rows[row["clip_dir"]] = row
    metadata = {"pid": os.getpid(), "model": config["EAS_MODEL"], "input_root": str(args.root) if args.root else None, "total_clips": len(clips), "camera": args.camera, "stride": args.stride, "workers": args.workers, "rules_path": str(args.rules.resolve()), "run_signature": signature, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "review_root": str(args.review_root) if args.review_root else None}
    export_state(rows, args, metadata, "running")
    (args.out_dir / "README.md").write_text("\n".join([
        "# Qwen 两类车道独立视觉筛选", "",
        f"模型：`{config['EAS_MODEL']}`；输入 {len(clips)} 条 clip；规则：`{args.rules}`。",
        "",
        f"使用 `{args.camera}`，每 {args.stride} 帧抽一张并补末帧。每条 clip 分别发送可变车道、潮汐车道两个独立视觉请求；可变车道图片侧重路面，潮汐图片保留上方信号与路面。两个目标独立判断，不要求同一 clip 同时满足两类规则。",
        "",
        "初筛命中后，对候选帧进行不带参考图的独立单帧复核。可变类放大画面中间近处路面，核对同一车道两侧白色锯齿；潮汐类保留完整道路区域，逐项核对车道控制装置、黄色边界和V形标记。每类找到一张确认命中的帧后即保存该帧。该步骤优先减少误报；远处、画面边缘或遮挡的标线可能漏检。",
        "",
        "可变车道：白色实线内侧成排锯齿状短斜线。潮汐车道：双黄虚线与车道级通行信号，或示例中的同组车道红叉/箭头、连续 V 形标记与黄色边界组合，或明确潮汐车道文字。",
        "",
        "两类分别保存到输出根目录的 `可变车道/` 与 `潮汐车道/`，各含 `clip_names.txt`、`clip_paths.txt` 与 `clips_with_photos/`。照片按来源车辆、月份、日期和 clip 层级存放，以保留同名 clip 的不同来源。",
        "",
        "`progress.json` 为实时进度，`results.jsonl` 记录模型观察结果、原始回答及命中帧，`results.tsv` 为结果汇总。采样用于初筛，不代表逐帧检查整段录像。",
        "",
    ]))
    if args.summarize_only:
        export_state(rows, args, metadata, "summarized")
        return 0
    with path.open("a", encoding="utf-8") as output:
        for round_number in range(1, 4):
            pending = [clip for clip in clips if not rows.get(str(clip) if clip else "validation_fixture", {}).get("ok") and not rows.get(str(clip) if clip else "validation_fixture", {}).get("skipped")]
            if not pending:
                break
            print(json.dumps({"round": round_number, "pending": len(pending), "workers": args.workers}, ensure_ascii=False), flush=True)
            with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
                # Bound image reads and queued inference work to active workers.
                iterator = iter(pending)
                active = {}
                for _ in range(min(args.workers, len(pending))):
                    clip = next(iterator)
                    active[pool.submit(process_clip, clip, config, prompt, args, signature, fixture)] = clip
                count = 0
                while active:
                    done, _ = cf.wait(active, return_when=cf.FIRST_COMPLETED)
                    for future in done:
                        clip = active.pop(future)
                        row = future.result()
                        try:
                            copy_hits(row, args)
                        except Exception as error:
                            row["export_error"] = f"{type(error).__name__}: {error}"
                            row["ok"] = False
                            row["error"] = row["export_error"]
                        rows[row["clip_dir"]] = row
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        count += 1
                        if count % 25 == 0 or row.get("variable_lane") or row.get("tidal_lane") or not row.get("ok"):
                            state = export_state(rows, args, metadata, "running")
                            print(json.dumps({key: state[key] for key in ("processed_clips", "total_clips", "variable_clips", "tidal_clips", "failed_clips")}, ensure_ascii=False), flush=True)
                        try:
                            next_clip = next(iterator)
                        except StopIteration:
                            continue
                        active[pool.submit(process_clip, next_clip, config, prompt, args, signature, fixture)] = next_clip
            export_state(rows, args, metadata, "retrying" if round_number < 3 else "running")
    state = export_state(rows, args, metadata, "complete" if not any(not row.get("ok") and not row.get("skipped") for row in rows.values()) else "complete_with_errors")
    print(json.dumps(state, ensure_ascii=False), flush=True)
    return 2 if state["failed_clips"] else 0
