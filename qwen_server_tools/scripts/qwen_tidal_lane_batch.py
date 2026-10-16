#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

AGENT_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(AGENT_DIR))
from qwen_adapter_aliyun_eas_q35 import QwenVL  # noqa: E402

DEFAULT_CAMERA_REL = Path('camera/center_camera_fov30')
PROMPT = """你是车载前视图“潮汐车道/可逆车道”筛选员。图中是同一个行车片段按时间抽取的多张缩略图，每张左上角有 frame 编号。

你的任务是判断哪些 frame 中出现“明确的潮汐车道/可逆车道场景”。不要泛泛识别普通车道、普通公交道、普通导向箭头、普通车道信号灯。

潮汐车道核心特征包括三类，至少需要有明确证据；优先保留两类及以上同时出现的画面：

1. 静态标线（空间标识）
- 路面有大号“潮汐车道”文字。
- 可见双黄虚线、可逆车道边界线、特殊车道边界。
- 可见分时段蓝底/电子指示牌，说明潮汐车道通行方向或时间窗。
- 可见潮汐/可变/可逆车道相关地面箭头或专用标识。

2. 空中信号（动态路权）
- 车道上方龙门架或悬臂杆有车道信号灯。
- 可见绿箭/红叉/X/向下箭头等车道级路权指示。
- 多个车道上方分别显示通行/禁行信号，用于动态分配车道方向。
- 重点是“灯先于线”：信号明确告诉驾驶人该车道当前是否可通行。

3. 物理防线（隔离重构）
- 可见拉链车、移动护栏、铰接式移动隔离墩。
- 可见隔离墩/护栏按车道级排列，明显用于动态调整车道分隔。
- 可见刚性屏障把对向车流隔开，并呈现可移动/可重构特征。

强正例：
- 路面直接喷涂“潮汐车道”，并配有方向箭头或分时指示牌。
- 龙门架上方每条车道有红叉/绿箭信号，指示某车道当前方向/路权。
- 中央隔离带有移动护栏或拉链式隔离设施，用于调整上下行车道数量。

必须排除的误判：
- 普通车道线、普通导向箭头、普通直行/左转/右转箭头。
- 普通公交专用道、普通潮汐交通提示以外的公交/专用车道。
- 普通红绿灯、普通路口信号灯，不是车道级红叉/绿箭。
- 普通中央护栏、隔离栏、桥梁护栏、绿化隔离栏，不具备移动/可重构或潮汐车道语义。
- 普通施工导改、锥桶、水马、道路施工围挡，这不是潮汐车道。
- 远处模糊疑似标志，看不清“潮汐车道/可逆车道/红叉绿箭/移动护栏”等关键证据时判 false。

判断原则：
- 不要只看颜色。重点看是否有“可逆车道/动态路权/车道级重构”的明确语义。
- 若只有普通道路标线或普通指示牌，construction 必须为 false。
- 若画面无法确认是潮汐车道，只能给 false。

只输出严格 JSON，不要 markdown：
{"construction": true/false, "frames": [帧号整数...], "features": ["静态标线"/"空中信号"/"物理防线"], "impact": "说明潮汐车道证据，例如路面文字/红叉绿箭/移动护栏/分时指示", "reason": "一句话说明可见证据"}

如果没有明确潮汐车道证据，construction 必须为 false，frames 为空数组，features 为空数组。"""

_tls = threading.local()


def get_model(args: argparse.Namespace) -> QwenVL:
    model = getattr(_tls, 'model', None)
    if model is None:
        model = QwenVL(
            backend='aliyun_eas',
            eas_endpoint=args.endpoint,
            eas_token=args.token,
            eas_model_name=args.model,
            eas_timeout=args.timeout,
            eas_prefer_route=args.route,
        )
        _tls.model = model
    return model


def iter_clip_dirs(root: Path):
    yield from sorted(p for p in root.rglob('*_dlp_pilotGtParser') if p.is_dir())


def sampled_images(clip_dir: Path, camera_rel: Path, stride: int) -> list[tuple[int, Path]]:
    camera_dir = clip_dir / camera_rel
    if not camera_dir.is_dir():
        return []
    files = sorted(camera_dir.glob('*.jpg'))
    return [(idx, files[idx]) for idx in range(0, len(files), stride)]


def copy_review_frames(args: argparse.Namespace, clip_dir: Path, frames: list[int], frame_to_file: dict[int, str]) -> list[str]:
    if not args.review_dir or not frames:
        return []
    try:
        rel_dir = clip_dir.resolve().relative_to(args.root.resolve())
    except Exception:
        rel_dir = Path(clip_dir.name)
    out_dir = args.review_dir / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for frame in frames:
        fname = frame_to_file.get(frame)
        if not fname:
            continue
        src = clip_dir / args.camera / fname
        if not src.exists():
            continue
        dst = out_dir / f'frame_{frame:06d}_{fname}'
        if not dst.exists():
            shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def make_sheet(samples: list[tuple[int, Path]], thumb_w: int = 512) -> Image.Image:
    thumbs: list[Image.Image] = []
    font = ImageFont.load_default()
    cell_w, cell_h = thumb_w, 330
    for frame, path in samples:
        im = Image.open(path).convert('RGB')
        w, h = im.size
        im = im.crop((0, int(h * 0.06), w, int(h * 0.78)))
        im.thumbnail((cell_w, 288))
        canvas = Image.new('RGB', (cell_w, cell_h), 'white')
        canvas.paste(im, ((cell_w - im.width) // 2, 28))
        d = ImageDraw.Draw(canvas)
        label = f'frame {frame} | {path.name}'
        d.rectangle((0, 0, cell_w, 25), fill=(255, 255, 210))
        d.text((8, 7), label, fill=(0, 0, 0), font=font)
        thumbs.append(canvas)

    cols = 2 if len(thumbs) > 1 else 1
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * cell_w, rows * cell_h), (230, 230, 230))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * cell_w, (i // cols) * cell_h))
    return sheet


def parse_answer(text: str) -> dict[str, Any]:
    raw = str(text).strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    m = re.search(r'\{.*\}', raw, re.S)
    if m:
        raw = m.group(0)
    try:
        data = json.loads(raw)
    except Exception:
        yes = any(x in str(text).lower() for x in ['true', '是', '有施工', '存在施工'])
        return {'construction': yes, 'frames': [], 'features': [], 'reason': str(text)[:240], 'parse_error': True, 'raw': str(text)[:1000]}
    frames = data.get('frames') or []
    clean_frames = []
    for v in frames:
        try:
            clean_frames.append(int(v))
        except Exception:
            pass
    allowed_features = {'静态标线', '空中信号', '物理防线'}
    features = [str(v) for v in (data.get('features') or []) if str(v) in allowed_features]
    return {
        'construction': bool(data.get('construction')),
        'frames': clean_frames,
        'features': features,
        'impact': str(data.get('impact', ''))[:240],
        'reason': str(data.get('reason', ''))[:500],
        'raw': str(text)[:1000],
    }


def process_clip(args: argparse.Namespace, clip_dir: Path) -> dict[str, Any]:
    samples = sampled_images(clip_dir, args.camera, args.stride)
    result: dict[str, Any] = {
        'clip_dir': str(clip_dir),
        'camera': str(args.camera),
        'ok': False,
        'construction': False,
        'frames': [],
        'files': [],
        'reason': '',
        'sample_count': len(samples),
    }
    if not samples:
        result.update(ok=True, reason=f'no {args.camera} jpg samples')
        return result

    sheet = make_sheet(samples)
    model = get_model(args)
    t0 = time.time()
    try:
        answer = model.generate_on_image(
            PROMPT,
            sheet,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        parsed = parse_answer(answer)
        frame_to_file = {frame: path.name for frame, path in samples}
        frames = [f for f in parsed.get('frames', []) if f in frame_to_file]
        construction = bool(parsed.get('construction') and frames and parsed.get('features'))
        review_files = copy_review_frames(args, clip_dir, frames, frame_to_file) if construction else []
        result.update(
            ok=True,
            construction=construction,
            frames=frames,
            files=[frame_to_file[f] for f in frames],
            review_files=review_files,
            features=parsed.get('features', []),
            impact=parsed.get('impact', ''),
            reason=parsed.get('reason', ''),
            raw=parsed.get('raw', ''),
            latency_sec=round(time.time() - t0, 3),
        )
    except Exception as exc:
        result.update(error=repr(exc), latency_sec=round(time.time() - t0, 3))
    return result


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row.get('clip_dir', ''))
            except Exception:
                pass
    return done


def write_json_summary(json_path: Path, jsonl_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if jsonl_path.exists():
        with jsonl_path.open('r', encoding='utf-8') as src:
            for line in src:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    return records


def append_txt_summary(txt_path: Path, jsonl_path: Path, root: Path, stride: int, camera: Path) -> tuple[int, int]:
    total = 0
    hits = 0
    with jsonl_path.open('r', encoding='utf-8') as src, txt_path.open('a', encoding='utf-8') as out:
        out.write('\n批量视觉模型潮汐车道筛选目录: ' + str(root) + '\n')
        out.write(f'模型策略: Qwen-VL contact sheet，每{stride}帧选取一张前视图片 {camera}\n')
        out.write('筛选口径: 潮汐车道/可逆车道；优先保留路面潮汐车道文字、红叉绿箭车道信号、移动护栏/拉链车等明确证据，排除普通车道/普通信号/普通护栏/施工导改。\n')
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if row.get('construction'):
                hits += 1
                out.write(f"\n目录: {row['clip_dir']}\n")
                for frame, fname in zip(row.get('frames', []), row.get('files', [])):
                    out.write(f"潮汐车道场景: 第{frame}帧开始, 文件名: {fname}\n")
                features = row.get('features') or []
                if features:
                    out.write(f"命中特征: {', '.join(features)}\n")
                reason = row.get('reason') or ''
                impact = row.get('impact') or ''
                if impact:
                    out.write(f"通行影响: {impact}\n")
                if reason:
                    out.write(f"模型依据: {reason}\n")
        out.write(f"\n批量视觉模型筛选完成: 共处理 {total} 个目录，发现潮汐车道候选 {hits} 个目录。\n")
    return total, hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--jsonl', type=Path, required=True)
    ap.add_argument('--json', type=Path, default=None)
    ap.add_argument('--txt', type=Path, required=True)
    ap.add_argument('--camera', type=Path, default=DEFAULT_CAMERA_REL)
    ap.add_argument('--review-dir', type=Path, default=None)
    ap.add_argument('--stride', type=int, default=100)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--endpoint', default=os.environ.get('EAS_ENDPOINT', ''))
    ap.add_argument('--token', default=os.environ.get('EAS_TOKEN', ''))
    ap.add_argument('--model', default=os.environ.get('EAS_MODEL', 'Qwen3.6-35B-A3B'))
    ap.add_argument('--route', default=os.environ.get('EAS_PREFER_ROUTE', 'chat_completions'))
    ap.add_argument('--timeout', type=int, default=int(os.environ.get('EAS_TIMEOUT', '600')))
    ap.add_argument('--max-new-tokens', type=int, default=220)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--start', type=int, default=0, help='skip the first N discovered clips before limit/done filtering')
    ap.add_argument('--done-jsonl', type=Path, action='append', default=[], help='extra jsonl result files whose clip_dir entries should be skipped')
    ap.add_argument('--exclude-prefix', type=Path, action='append', default=[], help='skip clips whose path is under this prefix')
    ap.add_argument('--summarize-only', action='store_true')
    args = ap.parse_args()
    if args.json is None:
        args.json = args.jsonl.with_suffix('.json')

    if args.summarize_only:
        write_json_summary(args.json, args.jsonl)
        total, hits = append_txt_summary(args.txt, args.jsonl, args.root, args.stride, args.camera)
        print(f'summarized total={total} hits={hits} json={args.json} txt={args.txt}')
        return 0

    if not args.endpoint or not args.token:
        raise SystemExit('EAS endpoint/token required via args or env')

    clips = list(iter_clip_dirs(args.root))
    if args.exclude_prefix:
        excludes = [str(p.resolve()) for p in args.exclude_prefix]
        clips = [c for c in clips if not any(str(c.resolve()).startswith(prefix.rstrip('/') + '/') or str(c.resolve()) == prefix for prefix in excludes)]
    if args.start:
        clips = clips[args.start:]
    if args.limit:
        clips = clips[: args.limit]
    done = load_done(args.jsonl)
    for done_jsonl in args.done_jsonl:
        done |= load_done(done_jsonl)
    pending = [c for c in clips if str(c) not in done]
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f'total={len(clips)} done={len(done)} pending={len(pending)} workers={args.workers}', flush=True)
    lock = threading.Lock()
    processed = 0
    matched = 0
    with args.jsonl.open('a', encoding='utf-8') as out:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_clip, args, clip): clip for clip in pending}
            for fut in cf.as_completed(futures):
                row = fut.result()
                with lock:
                    out.write(json.dumps(row, ensure_ascii=False) + '\n')
                    out.flush()
                    processed += 1
                    if row.get('construction'):
                        matched += 1
                    if processed % 50 == 0 or row.get('construction'):
                        print(
                            f"processed={processed}/{len(pending)} matched={matched} ok={row.get('ok')} construction={row.get('construction')} latency={row.get('latency_sec')}",
                            flush=True,
                        )

    write_json_summary(args.json, args.jsonl)
    total, hits = append_txt_summary(args.txt, args.jsonl, args.root, args.stride, args.camera)
    print(f'done total={total} hits={hits} jsonl={args.jsonl} json={args.json} txt={args.txt}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
