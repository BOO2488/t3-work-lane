#!/usr/bin/env python3
"""Classify candidate road images with the existing Qwen EAS vision service."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import re
import shlex
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
CATEGORIES = ("可变车道", "潮汐车道", "两者都有", "普通道路", "无法确认")
CREDENTIAL_SCRIPTS = (
    PROJECT / "model/run_qwen_generate_df_qa_q35_eas_onefile_semantic_nobehind.sh",
)


def load_configuration():
    names = ("EAS_ENDPOINT", "EAS_TOKEN", "EAS_MODEL")
    config = {name: os.environ[name] for name in names if os.environ.get(name)}
    for path in CREDENTIAL_SCRIPTS:
        if config.get("EAS_ENDPOINT") and config.get("EAS_TOKEN"):
            break
        if not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            match = re.match(r"^\s*export\s+(EAS_ENDPOINT|EAS_TOKEN|EAS_MODEL)\s*=(.*)$", line)
            if not match:
                continue
            parts = shlex.split(match.group(2), comments=True)
            if len(parts) == 1 and "$" not in parts[0] and "`" not in parts[0]:
                config.setdefault(match.group(1), parts[0])
    if not config.get("EAS_ENDPOINT") or not config.get("EAS_TOKEN"):
        raise ValueError("Missing EAS endpoint or credential in existing configuration")
    config.setdefault("EAS_MODEL", "Qwen3.6-35B-A3B")
    config["EAS_ENDPOINT"] = config["EAS_ENDPOINT"].rstrip("/")
    return config


def build_prompt(rules):
    # Keep visual criteria, omit labeled filenames and reference URLs.
    lines = [line for line in rules.splitlines() if not line.startswith(("示例 clip：", "[示例图片]"))]
    clean = re.sub(r"\[[^\]]+\]\(https?://[^)]+\)", "", "\n".join(lines))
    return f"""你是车载前视图道路标识视觉筛选员。只依据实际看到的标线、车道指示设施及清晰可读的文字判断。
根据以下规则把这一张图片分类为：可变车道、潮汐车道、两者都有、普通道路、无法确认。

规则：
{clean}

识别要求：
1. 可变车道重点看白色纵向实线内侧是否有沿车道排列、向内伸出的两排短斜线（锯齿状边界）。
2. 潮汐车道按以下三种证据组合判断，满足任一种即可保留：A. 双黄虚线和对应车道上方的红叉/箭头通行指示装置；B. 对应车道上方有成组红叉/箭头动态通行指示装置，同一组车道内有连续、重复的白色 V 形标记，并伴随黄色边界线；C. 清晰可读的“潮汐车道”文字。不要要求所有典型要素同时可见，黄色边界磨损时也不能仅因为未看清双黄虚线就否定组合 B。
3. 普通虚线、普通直行/转向箭头、路口普通红绿灯、单独的 V 形标记、导流区斜纹、减速标线，均不能单独证明上述两类车道。
4. 双黄虚线是两条平行的间断黄色线，不要把双黄实线或单黄虚线当成双黄虚线。
5. 普通道路表示画面清楚且未见上述两类证据；若关键区域模糊、被遮挡或证据不足，则返回无法确认。不要猜测未看清的文字。
6. 路面出现真实特征即可分类，不要求自车一定正在该车道中。若两类特征分别明确出现，可返回两者都有。
7. 独立报告实际看到的特征，即使你认为是普通道路，也要如实填写 repeated_v_markings 和 signal_and_v_same_lanes。repeated_v_markings 表示车道中央沿行驶方向连续排列多个白色 V 形标记；signal_and_v_same_lanes 表示车道级红叉/箭头指示装置与这些 V 形标记对应同一组车道。readable_lane_text 只写确实读到的车道类型文字，否则为空字符串。最终类别会由程序按这些实际可见特征统一生成。

只输出一个严格 JSON 对象，不要 Markdown。字段：
{{"category":"可变车道/潮汐车道/两者都有/普通道路/无法确认", "white_sawtooth":true或false, "double_yellow_dashed":true或false, "lane_control_signals":true或false, "repeated_v_markings":true或false, "signal_and_v_same_lanes":true或false, "readable_lane_text":"清晰可读的车道类型文字或空字符串", "evidence":["实际可见的证据"], "reason":"简短说明分类依据"}}
"""


def encode_image(path, max_side):
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        width, height = image.size
        image = image.crop((0, int(height * 0.04), width, int(height * 0.73)))
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_answer(answer):
    raw = str(answer).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("category") not in CATEGORIES:
        raise ValueError("Response has no valid category")
    for field in ("white_sawtooth", "double_yellow_dashed", "lane_control_signals", "repeated_v_markings", "signal_and_v_same_lanes"):
        if type(data.get(field)) is not bool:
            raise ValueError(f"Response field {field} must be a JSON boolean")
    if not isinstance(data.get("evidence"), list) or not all(isinstance(item, str) for item in data["evidence"]):
        raise ValueError("Response evidence must be an array of strings")
    if not isinstance(data.get("reason"), str):
        raise ValueError("Response reason must be a string")
    if not isinstance(data.get("readable_lane_text"), str):
        raise ValueError("Response readable_lane_text must be a string")
    return apply_feature_rules(data)


def apply_feature_rules(data):
    data = dict(data)
    variable = data["white_sawtooth"]
    tidal = ("潮汐车道" in data["readable_lane_text"] or
             (data["lane_control_signals"] and data["double_yellow_dashed"]) or
             (data["lane_control_signals"] and data["repeated_v_markings"] and data["signal_and_v_same_lanes"]))
    suggested = data["category"]
    data["model_category"] = suggested
    data["category"] = ("两者都有" if variable and tidal else "可变车道" if variable else
                        "潮汐车道" if tidal else "无法确认" if suggested in ("无法确认", "可变车道", "潮汐车道", "两者都有") else "普通道路")
    if data["category"] != suggested:
        data["model_reason"] = data["reason"]
        data["reason"] = "按 rule.md 中可见特征的组合规则归类；" + "；".join(data["evidence"])
    return data


def process_image(item, config, prompt, args):
    started = time.monotonic()
    record = dict(item)
    last_error = ""
    for attempt in range(1, args.attempts + 1):
        try:
            payload = {
                "model": config["EAS_MODEL"],
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": encode_image(Path(item["image_path"]), args.max_side)}},
                ]}],
                "stream": False,
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request = urllib.request.Request(
                config["EAS_ENDPOINT"] + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Authorization": config["EAS_TOKEN"], "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                result = json.loads(response.read(2_000_000))
            answer = result["choices"][0]["message"]["content"]
            parsed = parse_answer(answer)
            record.update(parsed, ok=True, usage=result.get("usage"), raw_answer=answer)
            record.update(latency_seconds=round(time.monotonic() - started, 3), attempts=attempt)
            return record
        except urllib.error.HTTPError as error:
            detail = error.read(10_000).decode(errors="replace").replace(config["EAS_TOKEN"], "[REDACTED]")
            last_error = f"HTTP {error.code}: {detail[:400]}"
            if error.code in (400, 401, 403, 404):
                break
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}".replace(config["EAS_TOKEN"], "[REDACTED]")
        if attempt < args.attempts:
            time.sleep(min(attempt * 2, 6))
    record.update(ok=False, error=last_error, latency_seconds=round(time.monotonic() - started, 3), attempts=attempt)
    return record


def collect_images(args):
    paths = list(args.image or [])
    if args.root:
        paths.extend(p for p in args.root.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    paths = sorted({p.resolve() for p in paths})
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise ValueError("No candidate images found")
    grouped = {}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        clip = next((parent.name for parent in path.parents if parent.name.endswith("_dlp_pilotGtParser")), path.parent.name)
        key = (clip, digest)
        if key in grouped:
            grouped[key]["duplicate_paths"].append(str(path))
        else:
            grouped[key] = {"clip_name": clip, "image_path": str(path), "image_sha256": digest, "duplicate_paths": []}
    return len(paths), list(grouped.values())


def load_cache(paths, signature):
    cache = {}
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("ok") is True and row.get("run_signature") == signature:
                cache[(row["clip_name"], row["image_sha256"])] = row
    return cache


def export_results(output, results, metadata):
    fields = ["clip_name", "image_path", "ok", "category", "white_sawtooth", "double_yellow_dashed", "lane_control_signals", "evidence", "reason", "error", "latency_seconds"]
    with (output / "results.tsv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({**row, "evidence": "；".join(row.get("evidence", []))})
    variable = sorted({row["clip_name"] for row in results if row.get("ok") and row.get("category") in ("可变车道", "两者都有")})
    tidal = sorted({row["clip_name"] for row in results if row.get("ok") and row.get("category") in ("潮汐车道", "两者都有")})
    review = sorted({row["clip_name"] for row in results if not row.get("ok") or row.get("category") == "无法确认"})
    for name, clips in [("variable_clip_names.txt", variable), ("tidal_clip_names.txt", tidal), ("matched_clip_names.txt", sorted(set(variable + tidal))), ("needs_review_clip_names.txt", review)]:
        (output / name).write_text("\n".join(clips) + ("\n" if clips else ""), encoding="utf-8")
    metadata.update(
        unique_clip_count=len({row["clip_name"] for row in results}),
        completed_images=len(results),
        failed_images=sum(row.get("ok") is not True for row in results),
        category_image_counts=dict(Counter(row.get("category", "请求失败") for row in results)),
        variable_clip_count=len(variable), tidal_clip_count=len(tidal), needs_review_clip_count=len(review),
    )
    (output / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    (output / "README.md").write_text("\n".join([
        "# Qwen 路面标识视觉筛选", "",
        f"模型：`{metadata['model']}`；规则：`{metadata['rules_path']}`。",
        "",
        f"共扫描 {metadata['input_image_count']} 张候选图片，按 clip 名称和图片内容去重后为 {metadata['unique_image_count']} 张；本次已处理 {len(results)} 张。",
        "",
        f"命中可变车道 {len(variable)} 个 clip，潮汐车道 {len(tidal)} 个 clip；分类是模型对候选图片的判断。",
        "",
        "本次逐张识别现有候选图，未检查原始 clip 的所有帧，也不代表整段 101 帧始终具有同类标识。保留无法确认和请求失败的记录，供复核。",
        "",
        "[可变车道名字](variable_clip_names.txt) · [潮汐车道名字](tidal_clip_names.txt) · [合并命中名字](matched_clip_names.txt) · [需要复核](needs_review_clip_names.txt) · [图片结果及证据](results.tsv)",
        "",
        "同一 clip 任一图片明确命中即列入相应清单；含两类证据的结果可以同时列入两张清单。完全相同的重复图片共用一次推理，路径保存在 JSONL 中。",
        "",
    ]), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--image", action="append", type=Path)
    parser.add_argument("--clip-list", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--camera", default="camera/center_camera_fov120")
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--thumb-width", type=int, default=768)
    parser.add_argument("--separate-images", action="store_true")
    parser.add_argument("--review-root", type=Path)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--rules", type=Path, default=PROJECT / "outputs/rule.md")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if min(args.workers, args.timeout, args.attempts, args.max_side, args.max_tokens) < 1 or args.limit < 0:
        parser.error("Numeric options must be positive; limit may be zero")
    if args.clip_list or args.validation_manifest:
        from qwen_lane_marking_clip_batch import run_batch
        return run_batch(args)
    config = load_configuration()
    rules = args.rules.read_text(encoding="utf-8")
    prompt = build_prompt(rules)
    signature = hashlib.sha256(json.dumps({"prompt": prompt, "model": config["EAS_MODEL"], "endpoint": config["EAS_ENDPOINT"], "max_side": args.max_side, "max_tokens": args.max_tokens}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    count, items = collect_images(args)
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt_snapshot.txt").write_text(prompt, encoding="utf-8")
    (output / "rules_snapshot.md").write_text(rules, encoding="utf-8")
    result_path = output / "results.jsonl"
    cache = load_cache([result_path, *args.cache_jsonl], signature)
    results, pending = [], []
    for item in items:
        key = (item["clip_name"], item["image_sha256"])
        if key in cache:
            results.append({**cache[key], **item, "cached": True})
        else:
            pending.append(item)
    metadata = {"model": config["EAS_MODEL"], "endpoint": config["EAS_ENDPOINT"], "rules_path": str(args.rules.resolve()), "run_signature": signature, "input_root": str(args.root.resolve()) if args.root else None, "input_image_count": count, "unique_image_count": len(items), "cached_images": len(results)}
    print(json.dumps({"model": config["EAS_MODEL"], "candidate_images": count, "unique_images": len(items), "cached": len(results), "pending": len(pending), "output": str(output)}, ensure_ascii=False), flush=True)
    with result_path.open("a", encoding="utf-8") as file:
        # Record externally cached entries in the current run as well.
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_image, item, config, prompt, args): item for item in pending}
            for future in concurrent.futures.as_completed(futures):
                row = {**future.result(), "run_signature": signature}
                results.append(row)
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                print(json.dumps({"completed": len(results), "total": len(items), "clip": row["clip_name"], "category": row.get("category"), "ok": row["ok"], "latency_seconds": row["latency_seconds"], "error": row.get("error")}, ensure_ascii=False), flush=True)
    results.sort(key=lambda row: (row["clip_name"], row["image_path"]))
    export_results(output, results, metadata)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return 2 if metadata["failed_images"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
