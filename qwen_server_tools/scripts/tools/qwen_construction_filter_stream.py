#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream T3 DLP clip folders through Qwen-VL/EAS and keep road-construction clips.

The filter is intentionally clip-oriented: sample frames from each clip, ask the
vision model whether road-construction facilities affect the ego vehicle path,
and copy representative front-camera frames for kept clips into the target
layout used by /mnt/wzb11/道路施工-all/<clip_name>/frame_*.jpg.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

QWEN_DIR = Path(__file__).resolve().parents[2] / "model"
if str(QWEN_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_DIR))

from qwen_adapter_aliyun_eas_q35 import QwenVL  # noqa: E402


DEFAULT_ROOT = Path("/mnt/data_clip/t3_origin/dlp_data_analysis")
DEFAULT_EXCLUDE = Path(
    "/mnt/data_clip/t3_origin/dlp_data_analysis/"
    "data_collection/gt_data/pilotGtParser/dongfeng_t3/20260522"
)
DEFAULT_KEEP_DIR = Path("/mnt/wzb11/道路施工-all")
DEFAULT_RESULT_DIR = Path("/mnt/wzb11/mihomo/qwen_construction_filter_results")

CAM_ORDER = [
    ("前视窄角", "center_camera_fov30"),
    ("左前视", "left_front_camera"),
    ("右前视", "right_front_camera"),
    ("左后视", "left_rear_camera"),
    ("右后视", "right_rear_camera"),
]

PROMPT = """你是自动驾驶道路场景筛选员。请根据给定的同一时刻多视角图片，判断该 clip 是否属于“必须保留的城市施工道路场景”。

核心规则：
只保留近中距离内会影响自车正常行驶路径的城市施工道路场景。判断重点不是颜色，也不是画面里单纯出现围挡/锥桶/水马/施工车，而是施工设施是否实际进入或压缩机动车通行空间，是否使自车需要减速、绕行、并线、变道或按临时路径通行。

必须保留：
1. 当前车道或相邻车道被施工围挡、水马、防撞桶、锥桶阵列、施工车辆、施工机械等占用。
2. 原本可通行车道被封闭、压缩、部分占用，或前方近中距离出现封闭端头/施工隔离端头/临时通行缺口。
3. 锥桶/水马/导向设施形成连续边界，迫使车辆改变行驶轨迹。
4. 施工人员、施工机械、工程车辆位于车道内或车道边界内侧，作业区贴近或侵入机动车道。
5. 路中间或当前车道前方横向摆放施工牌、导向牌、限速牌、锥桶、水马，形成明显封闭端头。

必须删除：
1. 只是路边长期围挡，位于人行道、绿化带、建筑红线内或路外区域，机动车道宽度和方向正常。
2. 只是远处背景施工、吊车、楼体、脚手架、建筑工地，不影响近中距离当前行驶路线。
3. 少量孤立锥桶/水马只是路边摆放或停车隔离，没有连续导流边界，也没有造成车道收窄/封闭/绕行/并线。
4. 普通中央护栏、永久隔离栏、普通交通标志、路灯、信号灯、收费站、检查站等普通设施。
5. 普通货车、工程外观车辆、清扫车、养护车单独出现但不占道、不导改。
6. 只有工人或反光背心，但没有车道封闭、占用或导改。

距离要求：
只看自车近中距离。远处背景施工一律不能作为保留依据。

请严格输出 JSON，不要输出 markdown，不要额外解释。格式：
{
  "decision": "keep" 或 "delete",
  "confidence": 0.0 到 1.0,
  "affects_ego_path": true 或 false,
  "evidence_frame": "当前帧编号或时间戳",
  "reason": "一句中文理由，必须说明是否侵入/压缩/导改机动车通行空间"
}
"""


@dataclass
class FrameSample:
    frame_index: int
    timestamp: str
    views: Dict[str, Path]


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def iter_clip_dirs(root: Path, exclude: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        if is_under(p, exclude):
            dirnames[:] = []
            continue
        if "camera" in dirnames and (
            "sensor_temporal_alignment.json" in filenames
            or "sensor_temporal_alignment_15ms.json" in filenames
        ):
            yield p
            dirnames[:] = []


def safe_json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_alignment(clip_dir: Path) -> List[Tuple[str, Dict[str, str]]]:
    for name in ("sensor_temporal_alignment.json", "sensor_temporal_alignment_15ms.json"):
        p = clip_dir / name
        if not p.exists():
            continue
        data = safe_json_load(p)
        out: List[Tuple[str, Dict[str, str]]] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item:
                    ts, mapping = next(iter(item.items()))
                    if isinstance(mapping, dict):
                        out.append((str(ts), mapping))
        return out
    return []


def choose_indices(n: int, max_samples: int, stride: int) -> List[int]:
    if n <= 0:
        return []
    if stride > 0:
        idxs = list(range(0, n, stride))
    else:
        if max_samples <= 1:
            idxs = [n // 2]
        else:
            idxs = sorted({round(i * (n - 1) / (max_samples - 1)) for i in range(max_samples)})
    if max_samples > 0:
        return idxs[:max_samples]
    return idxs


def build_samples(clip_dir: Path, max_samples: int, stride: int) -> List[FrameSample]:
    alignment = load_alignment(clip_dir)
    samples: List[FrameSample] = []
    for idx in choose_indices(len(alignment), max_samples, stride):
        ts, mapping = alignment[idx]
        views: Dict[str, Path] = {}
        for label, cam in CAM_ORDER:
            rel = mapping.get(cam)
            if rel:
                path = clip_dir / rel
                if path.exists():
                    views[label] = path
        if views:
            samples.append(FrameSample(idx, ts, views))
    return samples


def load_images(sample: FrameSample) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    for _label, _cam in CAM_ORDER:
        path = sample.views.get(_label)
        if path and path.exists():
            img = Image.open(path).convert("RGB")
            imgs.append(img)
    return imgs


def build_prompt(sample: FrameSample, clip_name: str) -> str:
    view_lines = []
    used = 0
    for label, _cam in CAM_ORDER:
        if label in sample.views:
            used += 1
            view_lines.append(f"{used}) {label}：<image>")
    return (
        f"CLIP={clip_name}\nFRAME_INDEX={sample.frame_index}\nTIMESTAMP={sample.timestamp}\n"
        "你将看到以下同步视角，前方/左前方/右前方均以自车为参照：\n"
        + "\n".join(view_lines)
        + "\n\n"
        + PROMPT
    )


def parse_model_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {
        "decision": "delete",
        "confidence": 0.0,
        "affects_ego_path": False,
        "evidence_frame": "",
        "reason": f"模型未输出可解析JSON: {raw[:200]}",
        "raw_response": raw,
    }


def make_qwen(args: argparse.Namespace) -> QwenVL:
    qwen = QwenVL(
        backend="aliyun_eas",
        eas_endpoint=args.eas_endpoint,
        eas_token=args.eas_token,
        eas_model_name=args.eas_model,
        eas_timeout=args.eas_timeout,
        eas_verify_ssl=not args.eas_insecure,
        eas_prefer_route=args.eas_prefer_route,
    )
    qwen.set_generation_defaults(max_new_tokens=args.max_new_tokens, top_p=args.top_p, temperature=args.temperature)
    return qwen


def copy_keep_frames(clip_dir: Path, samples: List[FrameSample], keep_dir: Path) -> List[str]:
    out_dir = keep_dir / clip_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for i, sample in enumerate(samples):
        front = sample.views.get("前视窄角")
        if not front:
            continue
        dst = out_dir / f"frame_{sample.frame_index:06d}_{sample.timestamp}.jpg"
        if not dst.exists():
            shutil.copy2(front, dst)
        copied.append(str(dst))
    return copied


def judge_clip(clip_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    samples = build_samples(clip_dir, args.samples_per_clip, args.sample_stride)
    if not samples:
        return {
            "clip": str(clip_dir),
            "clip_name": clip_dir.name,
            "decision": "delete",
            "confidence": 0.0,
            "reason": "没有找到可用相机帧",
            "frame_results": [],
            "copied_files": [],
        }

    qwen = make_qwen(args)
    frame_results: List[Dict[str, Any]] = []
    keep_hits: List[Dict[str, Any]] = []
    for sample in samples:
        images = load_images(sample)
        prompt = build_prompt(sample, clip_dir.name)
        text = qwen.generate_on_images(
            text=prompt,
            images=images,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=False,
        )
        parsed = parse_model_json(text)
        parsed.update({"frame_index": sample.frame_index, "timestamp": sample.timestamp})
        frame_results.append(parsed)
        if str(parsed.get("decision", "")).lower() == "keep" or parsed.get("affects_ego_path") is True:
            keep_hits.append(parsed)
            if args.stop_on_keep:
                break

    decision = "keep" if keep_hits else "delete"
    best = max(
        keep_hits or frame_results,
        key=lambda x: float(x.get("confidence", 0.0) or 0.0),
    )
    copied = copy_keep_frames(clip_dir, samples, args.keep_dir) if decision == "keep" and not args.dry_run else []
    return {
        "clip": str(clip_dir),
        "clip_name": clip_dir.name,
        "decision": decision,
        "confidence": float(best.get("confidence", 0.0) or 0.0),
        "reason": best.get("reason", ""),
        "evidence_frame": best.get("evidence_frame", best.get("frame_index", "")),
        "frame_results": frame_results,
        "copied_files": copied,
    }


class JsonlSink:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())


def load_done(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                clip = obj.get("clip")
                decision = str(obj.get("decision", "")).lower()
                if clip and decision in {"keep", "delete"}:
                    done.add(str(clip))
            except Exception:
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--exclude", type=Path, default=DEFAULT_EXCLUDE)
    parser.add_argument("--keep-dir", type=Path, default=DEFAULT_KEEP_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--samples-per-clip", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--stop-on-keep", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eas-endpoint", default=os.getenv("EAS_ENDPOINT", ""))
    parser.add_argument("--eas-token", default=os.getenv("EAS_TOKEN", ""))
    parser.add_argument("--eas-model", default=os.getenv("EAS_MODEL", "Qwen3.6-35B-A3B"))
    parser.add_argument("--eas-timeout", type=int, default=int(os.getenv("EAS_TIMEOUT", "600")))
    parser.add_argument("--eas-insecure", action="store_true")
    parser.add_argument("--eas-prefer-route", default=os.getenv("EAS_PREFER_ROUTE", "chat_completions"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if not args.eas_endpoint or not args.eas_token:
        raise SystemExit("EAS_ENDPOINT/EAS_TOKEN 不能为空")

    args.result_dir.mkdir(parents=True, exist_ok=True)
    result_jsonl = args.result_dir / "clip_results.jsonl"
    sink = JsonlSink(result_jsonl)
    done = load_done(result_jsonl)

    print(f"[INFO] root={args.root}", flush=True)
    print(f"[INFO] exclude={args.exclude}", flush=True)
    print(f"[INFO] done={len(done)} workers={args.workers}", flush=True)
    print(f"[INFO] keep_dir={args.keep_dir}", flush=True)
    print(f"[INFO] result_jsonl={result_jsonl}", flush=True)
    print("[INFO] streaming clip discovery and Qwen filtering...", flush=True)

    start = time.time()
    submitted = 0
    completed = 0
    keep_count = 0
    delete_count = 0
    error_count = 0
    max_inflight = max(1, int(args.workers) * 2)

    def handle_done(fut, clip):
        nonlocal completed, keep_count, delete_count, error_count
        try:
            res = fut.result()
        except Exception as exc:
            res = {
                "clip": str(clip),
                "clip_name": clip.name,
                "decision": "error",
                "confidence": 0.0,
                "reason": str(exc),
                "frame_results": [],
                "copied_files": [],
            }
        sink.append(res)
        completed += 1
        if res["decision"] == "keep":
            keep_count += 1
        elif res["decision"] == "delete":
            delete_count += 1
        else:
            error_count += 1
        if completed % 20 == 0:
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"[PROGRESS] completed={completed} submitted={submitted} keep={keep_count} "
                f"delete={delete_count} error={error_count} rate={completed/elapsed:.2f} clips/s",
                flush=True,
            )

    with futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        future_map = {}
        for clip in iter_clip_dirs(args.root, args.exclude):
            if str(clip) in done:
                continue
            if args.max_clips > 0 and submitted >= args.max_clips:
                break
            fut = ex.submit(judge_clip, clip, args)
            future_map[fut] = clip
            submitted += 1
            if submitted % 20 == 0:
                print(f"[DISCOVER] submitted={submitted} latest={clip}", flush=True)
            while len(future_map) >= max_inflight:
                done_futs, _ = futures.wait(future_map, return_when=futures.FIRST_COMPLETED)
                for done_fut in done_futs:
                    done_clip = future_map.pop(done_fut)
                    handle_done(done_fut, done_clip)
        print(f"[INFO] discovery finished submitted={submitted}", flush=True)
        for fut in futures.as_completed(list(future_map)):
            clip = future_map[fut]
            handle_done(fut, clip)

    elapsed = max(time.time() - start, 1e-6)
    print(
        f"[SUMMARY] completed={completed} submitted={submitted} keep={keep_count} "
        f"delete={delete_count} error={error_count} rate={completed/elapsed:.2f} clips/s",
        flush=True,
    )

    final_json = args.result_dir / "clip_results.json"
    records = []
    with result_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    final_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    keep_txt = args.result_dir / "keep_clips.txt"
    keep_txt.write_text(
        "\n".join(r["clip"] for r in records if r.get("decision") == "keep") + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] results={final_json} keep_list={keep_txt}")


if __name__ == "__main__":
    main()
