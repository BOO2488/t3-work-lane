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
PROMPT = """你是车载前视图“城市施工道路影响自车行驶”筛选员。图中是同一个行车片段按时间抽取的多张缩略图，每张左上角有 frame 编号。

你的任务不是泛泛识别“施工”，而是判断哪些 frame 中存在“近中距离内会影响当前自车行驶路径的城市道路施工/养护”。

必须保留的强正例：
1. 施工牌、限速牌、导向牌、锥桶、水马、临时护栏等横向摆在路中间或当前车道前方，形成封闭端头。
2. 当前车道或相邻机动车道被施工设施占用、压缩或封闭，自车继续前进需要减速、绕行、并线或变道。
3. 近中距离内存在临时导流路径、车道收窄、车道封闭、施工占道、作业区侵入机动车道。
4. 施工预警牌、LED/发光箭头、黄黑导向板等与当前通行变化同时出现。

必须排除的误判：
1. 路边长期围挡、人行道围挡、建筑红线内围挡、隔离带围挡，但机动车道正常通行。
2. 远处施工、远处围挡、远处塔吊或建筑施工，当前近中距离车道未受影响。
3. 普通护栏、桥梁护栏、绿化隔离栏、中央隔离栏、普通交通标志。
4. 少量孤立锥桶/水马，只是路边摆放、停车隔离或未形成导改。
5. 普通货车、清扫车、工程外观车辆、工人单独出现，但没有施工占道或通行路径变化。
6. 只看到建筑工地、脚手架、塔吊、楼体施工，但没有影响机动车道。

判断原则：
- 不要主要根据颜色判断。颜色只能辅助识别物体。
- 优先判断空间关系：施工设施是否位于自车近中距离、是否进入机动车通行空间、是否改变自车行驶路径。
- 只有“会影响当前车辆行驶”的施工路段才算 true；不影响当前车辆行驶的一律 false。

只输出严格 JSON，不要 markdown：
{"construction": true/false, "frames": [帧号整数...], "features": ["空间阻断"/"动线重构"/"多维预警"], "impact": "占道/收窄/绕行/并线/封闭端头/无影响", "reason": "一句话说明可见证据"}

如果只是路边围挡、远处施工、普通护栏、孤立锥桶/水马，或者不影响当前车辆通行，construction 必须为 false，frames 为空数组，features 为空数组。"""

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
    allowed_features = {'空间阻断', '动线重构', '多维预警'}
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
        out.write('\n批量视觉模型筛选目录: ' + str(root) + '\n')
        out.write(f'模型策略: Qwen-VL contact sheet，每{stride}帧选取一张前视图片 {camera}\n')
        out.write('筛选口径: 只保留近中距离内实际影响自车行驶路径的城市施工道路；路边长期围挡、远处施工、普通护栏、孤立锥桶/水马均排除。\n')
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
                impact = row.get('impact') or ''
                if impact:
                    out.write(f"通行影响: {impact}\n")
                if reason:
                    out.write(f"模型依据: {reason}\n")
        out.write(f"\n批量视觉模型筛选完成: 共处理 {total} 个目录，发现施工 {hits} 个目录。\n")
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
