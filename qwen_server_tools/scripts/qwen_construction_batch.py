#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

AGENT_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(AGENT_DIR))
from qwen_adapter_aliyun_eas_q35 import QwenVL  # noqa: E402

CAMERA_REL = Path('camera/center_camera_fov120')
PROMPT = """你是车载前视图城市施工道路安全防护体系筛选员。图中是同一个行车片段按时间抽取的多张缩略图，每张左上角有 frame 编号。

请按“空间阻断、动线重构、多维预警”三个协同特征，判断哪些 frame 属于城市施工道路核心场景：
1. 刚性封闭（空间阻断）：标准化模块围挡、蓝/绿/金属施工围挡、迎车面防撞桶、水马、硬隔离设施，清晰阻隔施工作业面与通行边界。
2. 柔性引导（动线重构）：等间距斜向布设的反光锥桶阵列、渐变过渡段、车道收窄、多车道向单车道阶梯式合流/分流、临时交通导改。
3. 主动诱导（多维预警）：黄底“前方施工”等前置施工标志牌、红蓝爆闪灯、LED 动态导向箭头屏、夜间/恶劣天气下的强视觉预警。

优先筛出同时具备两类及以上特征的画面；若只有单一特征，必须非常明确且占据道路通行区域，例如连续锥桶阵列、完整围挡封闭、明显施工警示屏。不要把普通店招、普通停放车辆、普通护栏、树木阴影、非施工广告牌、道路中央隔离栏误判为施工。

只输出严格 JSON，不要 markdown：
{"construction": true/false, "frames": [帧号整数...], "features": ["空间阻断"/"动线重构"/"多维预警"], "reason": "一句话说明可见证据"}
如果没有明确施工，frames 为空数组，features 为空数组。"""

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
    yield from sorted(root.glob('*/*/*/*_dlp_pilotGtParser'))


def sampled_images(clip_dir: Path, stride: int) -> list[tuple[int, Path]]:
    camera_dir = clip_dir / CAMERA_REL
    if not camera_dir.is_dir():
        return []
    files = sorted(camera_dir.glob('*.jpg'))
    return [(idx, files[idx]) for idx in range(0, len(files), stride)]


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
    allowed_features = {'空间阻断', '动线重构', '多维预警'}
    features = [str(v) for v in (data.get('features') or []) if str(v) in allowed_features]
    return {
        'construction': bool(data.get('construction')),
        'frames': clean_frames,
        'features': features,
        'reason': str(data.get('reason', ''))[:500],
        'raw': str(text)[:1000],
    }


def process_clip(args: argparse.Namespace, clip_dir: Path) -> dict[str, Any]:
    samples = sampled_images(clip_dir, args.stride)
    result: dict[str, Any] = {
        'clip_dir': str(clip_dir),
        'ok': False,
        'construction': False,
        'frames': [],
        'files': [],
        'reason': '',
        'sample_count': len(samples),
    }
    if not samples:
        result.update(ok=True, reason='no center_camera_fov120 jpg samples')
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
        result.update(
            ok=True,
            construction=bool(parsed.get('construction') and frames and parsed.get('features')),
            frames=frames,
            files=[frame_to_file[f] for f in frames],
            features=parsed.get('features', []),
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


def append_txt_summary(txt_path: Path, jsonl_path: Path, root: Path, stride: int) -> tuple[int, int]:
    total = 0
    hits = 0
    with jsonl_path.open('r', encoding='utf-8') as src, txt_path.open('a', encoding='utf-8') as out:
        out.write('\n批量视觉模型筛选目录: ' + str(root) + '\n')
        out.write(f'模型策略: Qwen-VL contact sheet，每{stride}帧选取一张前视图片 camera/center_camera_fov120\n')
        out.write('筛选口径: 空间阻断、动线重构、多维预警；优先保留两类及以上协同特征，单一特征需非常明确且位于道路通行区域。\n')
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if row.get('construction'):
                hits += 1
                out.write(f"\n目录: {row['clip_dir']}\n")
                for frame, fname in zip(row.get('frames', []), row.get('files', [])):
                    out.write(f"施工场景: 第{frame}帧开始, 文件名: {fname}\n")
                features = row.get('features') or []
                if features:
                    out.write(f"命中特征: {', '.join(features)}\n")
                reason = row.get('reason') or ''
                if reason:
                    out.write(f"模型依据: {reason}\n")
        out.write(f"\n批量视觉模型筛选完成: 共处理 {total} 个目录，发现施工 {hits} 个目录。\n")
    return total, hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--jsonl', type=Path, required=True)
    ap.add_argument('--txt', type=Path, required=True)
    ap.add_argument('--stride', type=int, default=100)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--endpoint', default=os.environ.get('EAS_ENDPOINT', ''))
    ap.add_argument('--token', default=os.environ.get('EAS_TOKEN', ''))
    ap.add_argument('--model', default=os.environ.get('EAS_MODEL', 'Qwen3.6-35B-A3B'))
    ap.add_argument('--route', default=os.environ.get('EAS_PREFER_ROUTE', 'chat_completions'))
    ap.add_argument('--timeout', type=int, default=int(os.environ.get('EAS_TIMEOUT', '600')))
    ap.add_argument('--max-new-tokens', type=int, default=220)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--summarize-only', action='store_true')
    args = ap.parse_args()

    if args.summarize_only:
        total, hits = append_txt_summary(args.txt, args.jsonl, args.root, args.stride)
        print(f'summarized total={total} hits={hits} txt={args.txt}')
        return 0

    if not args.endpoint or not args.token:
        raise SystemExit('EAS endpoint/token required via args or env')

    clips = list(iter_clip_dirs(args.root))
    if args.limit:
        clips = clips[: args.limit]
    done = load_done(args.jsonl)
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

    total, hits = append_txt_summary(args.txt, args.jsonl, args.root, args.stride)
    print(f'done total={total} hits={hits} jsonl={args.jsonl} txt={args.txt}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
