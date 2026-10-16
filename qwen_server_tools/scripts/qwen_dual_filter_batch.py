#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

AGENT_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(AGENT_DIR))
from qwen_adapter_aliyun_eas_q35 import QwenVL  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from PIL import Image, ImageDraw, ImageFont

from qwen_lane_impact_batch import sampled_images  # noqa: E402

DEFAULT_CAMERA_REL = Path('camera/center_camera_fov30')
_tls = threading.local()

COMBINED_PROMPT = """你是车载前视图严格筛选员。图中是同一个行车片段按时间抽取的多张 center_camera_fov30 前视窄角缩略图，每张左上角有 frame 编号。请一次性完成两个独立判断：

任务A lane_impact：判断是否属于“城市施工道路影响自车行驶路径”。
只保留近中距离内实际进入或压缩机动车通行空间、使自车需要减速/绕行/并线/变道/沿临时路径通行的施工道路场景。
保留证据包括：当前或相邻车道被围挡、水马、防撞桶、锥桶阵列、施工车/机械占用；路中间或当前车道前方有施工牌/导向牌/限速牌/锥桶/水马形成封闭端头；连续导流导致车道收窄、封闭、绕行或并线；作业区侵入机动车道。
必须排除：路边长期围挡但车道正常、远处背景施工、建筑工地/塔吊/脚手架背景、普通护栏/绿化隔离栏/桥梁护栏/普通标志、孤立锥桶/水马、普通车辆/工人但不占道不导改。

任务B tidal_lane：判断是否属于“潮汐车道/可逆车道”。
只保留有明确可逆车道/动态路权/物理重构证据的画面。
保留证据包括：路面写“潮汐车道”；双黄虚线、可逆车道边界、潮汐/可逆车道专用箭头或分时段通行指示牌；龙门架/悬臂杆上方车道级红叉、绿箭、向下箭头；移动护栏、拉链车、铰接式移动隔离墩改变车道分隔。
必须排除：普通车道线、普通导向箭头、普通红绿灯、普通公交专用道、普通护栏/绿化隔离栏/桥梁护栏、普通施工导改、远处模糊疑似设施。

两个任务必须独立判断。施工导改不能算潮汐车道；潮汐车道设施也不能自动算施工。若无法确认，判 false。

只输出严格 JSON，不要 markdown，不要额外解释。格式必须是：
{
  "lane_impact": {
    "construction": true/false,
    "frames": [帧号整数...],
    "features": ["空间阻断"/"动线重构"/"多维预警"],
    "impact": "占道/收窄/绕行/并线/封闭端头/无影响",
    "reason": "一句话说明施工设施是否侵入/压缩/导改机动车通行空间"
  },
  "tidal_lane": {
    "construction": true/false,
    "frames": [帧号整数...],
    "features": ["静态标线"/"空中信号"/"物理防线"],
    "impact": "说明潮汐车道证据，例如路面文字/红叉绿箭/移动护栏/分时指示",
    "reason": "一句话说明可见证据"
  }
}

如果某任务没有明确证据，该任务 construction 必须为 false，frames 和 features 必须为空数组。
"""


def make_compact_sheet(samples: list[tuple[int, Path]], thumb_w: int = 384) -> Image.Image:
    thumbs: list[Image.Image] = []
    font = ImageFont.load_default()
    cell_w, cell_h = thumb_w, 248
    for frame, path in samples:
        im = Image.open(path).convert('RGB')
        w, h = im.size
        im = im.crop((0, int(h * 0.08), w, int(h * 0.76)))
        im.thumbnail((cell_w, 216))
        canvas = Image.new('RGB', (cell_w, cell_h), 'white')
        canvas.paste(im, ((cell_w - im.width) // 2, 26))
        d = ImageDraw.Draw(canvas)
        label = f'frame {frame} | {path.name}'
        d.rectangle((0, 0, cell_w, 23), fill=(255, 255, 210))
        d.text((6, 6), label, fill=(0, 0, 0), font=font)
        thumbs.append(canvas)

    cols = 2 if len(thumbs) > 1 else 1
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * cell_w, rows * cell_h), (230, 230, 230))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * cell_w, (i // cols) * cell_h))
    return sheet


def _extract_json_obj(text: str) -> dict[str, Any]:
    raw = str(text or '').strip()
    if raw.startswith('```'):
        raw = raw.strip('`')
        raw = raw.removeprefix('json').strip()
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find('{')
        end = raw.rfind('}')
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                pass
    return {}


def parse_combined_answer(text: str) -> dict[str, dict[str, Any]]:
    data = _extract_json_obj(text)
    return {
        'lane_impact': data.get('lane_impact') if isinstance(data.get('lane_impact'), dict) else {},
        'tidal_lane': data.get('tidal_lane') if isinstance(data.get('tidal_lane'), dict) else {},
        'raw': str(text or '')[:1200],
    }


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
    for dirpath, dirnames, _filenames in os.walk(root):
        path = Path(dirpath)
        if path.name.endswith('_dlp_pilotGtParser'):
            yield path
            dirnames[:] = []



def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get('ok') is True:
                    done.add(row.get('clip_dir', ''))
            except Exception:
                pass
    return done


def normalize_result(
    parsed: dict[str, Any],
    samples: list[tuple[int, Path]],
    allowed_features: set[str],
    raw: str = '',
) -> dict[str, Any]:
    frame_to_file = {frame: path.name for frame, path in samples}
    frames = []
    for v in parsed.get('frames') or []:
        try:
            frame = int(v)
        except Exception:
            continue
        if frame in frame_to_file:
            frames.append(frame)
    features = [str(v) for v in (parsed.get('features') or []) if str(v) in allowed_features]
    construction = bool(parsed.get('construction') and frames and features)
    return {
        'construction': construction,
        'frames': frames,
        'files': [frame_to_file[f] for f in frames],
        'features': features,
        'impact': str(parsed.get('impact', ''))[:240],
        'reason': str(parsed.get('reason', ''))[:500],
        'raw': raw,
    }


def copy_task_frames(
    args: argparse.Namespace,
    clip_dir: Path,
    output_dir: Path | None,
    frames: list[int],
    files: list[str],
) -> list[str]:
    if not output_dir or not frames:
        return []
    out_dir = output_dir / clip_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for frame, fname in zip(frames, files):
        src = clip_dir / args.camera / fname
        if not src.exists():
            continue
        dst = out_dir / f'frame_{frame:06d}_{fname}'
        if not dst.exists():
            shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def ask_model(model: QwenVL, prompt: str, sheet, args: argparse.Namespace) -> tuple[str, float]:
    t0 = time.time()
    answer = model.generate_on_image(
        prompt,
        sheet,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
    )
    return answer, round(time.time() - t0, 3)


def process_clip(args: argparse.Namespace, clip_dir: Path) -> dict[str, Any]:
    samples = sampled_images(clip_dir, args.camera, args.stride)
    result: dict[str, Any] = {
        'clip_dir': str(clip_dir),
        'camera': str(args.camera),
        'ok': False,
        'sample_count': len(samples),
        'lane_impact': {'construction': False, 'frames': [], 'files': [], 'reason': ''},
        'tidal_lane': {'construction': False, 'frames': [], 'files': [], 'reason': ''},
    }
    if not samples:
        result.update(ok=True, reason=f'no {args.camera} jpg samples')
        return result

    sheet = make_compact_sheet(samples, args.thumb_width)
    model = get_model(args)
    try:
        answer, latency = ask_model(model, COMBINED_PROMPT, sheet, args)
        parsed = parse_combined_answer(answer)
        lane = normalize_result(parsed.get('lane_impact') or {}, samples, {'空间阻断', '动线重构', '多维预警'}, parsed.get('raw', ''))
        lane['latency_sec'] = latency
        lane['image_files'] = copy_task_frames(args, clip_dir, args.work_output_dir, lane['frames'], lane['files']) if lane['construction'] else []

        tidal = normalize_result(parsed.get('tidal_lane') or {}, samples, {'静态标线', '空中信号', '物理防线'}, parsed.get('raw', ''))
        tidal['latency_sec'] = latency
        tidal['image_files'] = copy_task_frames(args, clip_dir, args.lane_output_dir, tidal['frames'], tidal['files']) if tidal['construction'] else []

        result.update(ok=True, lane_impact=lane, tidal_lane=tidal, latency_sec=latency)
    except Exception as exc:
        result.update(error=repr(exc))
    return result


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


def write_txt_summary(txt_path: Path, records: list[dict[str, Any]], root: Path, stride: int, camera: Path) -> tuple[int, int, int]:
    lane_hits = 0
    tidal_hits = 0
    error_count = 0
    with txt_path.open('w', encoding='utf-8') as out:
        out.write('双任务批量视觉模型筛选目录: ' + str(root) + '\n')
        out.write(f'模型策略: 每{stride}帧选取一张前视图片 {camera}，同一 contact sheet 连续判断施工影响和潮汐车道。\n')
        for row in records:
            lane = row.get('lane_impact') or {}
            tidal = row.get('tidal_lane') or {}
            if lane.get('construction'):
                lane_hits += 1
                out.write(f"\n[施工影响] 目录: {row.get('clip_dir')}\n")
                for frame, fname in zip(lane.get('frames', []), lane.get('files', [])):
                    out.write(f'命中帧: 第{frame}帧, 文件名: {fname}\n')
                if lane.get('features'):
                    out.write('命中特征: ' + ', '.join(lane.get('features', [])) + '\n')
                if lane.get('impact'):
                    out.write('通行影响: ' + str(lane.get('impact')) + '\n')
                if lane.get('reason'):
                    out.write('模型依据: ' + str(lane.get('reason')) + '\n')
            if tidal.get('construction'):
                tidal_hits += 1
                out.write(f"\n[潮汐车道] 目录: {row.get('clip_dir')}\n")
                for frame, fname in zip(tidal.get('frames', []), tidal.get('files', [])):
                    out.write(f'命中帧: 第{frame}帧, 文件名: {fname}\n')
                if tidal.get('features'):
                    out.write('命中特征: ' + ', '.join(tidal.get('features', [])) + '\n')
                if tidal.get('impact'):
                    out.write('证据说明: ' + str(tidal.get('impact')) + '\n')
                if tidal.get('reason'):
                    out.write('模型依据: ' + str(tidal.get('reason')) + '\n')
        out.write(f"\n双任务筛选完成: 共处理 {len(records)} 个目录，施工影响候选 {lane_hits} 个，潮汐车道候选 {tidal_hits} 个。\n")
    return len(records), lane_hits, tidal_hits



def write_task_file_outputs(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    task_specs = [
        ('tidal_lane', args.lane_output_dir, 'qwen_tidal_lane', '潮汐车道'),
        ('lane_impact', args.work_output_dir, 'qwen_work_construction', '施工场景'),
    ]
    for key, out_dir, stem, title in task_specs:
        if not out_dir:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for row in records:
            task = row.get(key) or {}
            rows.append({
                'clip_dir': row.get('clip_dir'),
                'camera': row.get('camera'),
                'ok': row.get('ok'),
                'sample_count': row.get('sample_count'),
                **task,
            })
        jsonl_path = out_dir / f'{stem}.jsonl'
        json_path = out_dir / f'{stem}.json'
        txt_path = out_dir / f'{stem}.txt'
        with jsonl_path.open('w', encoding='utf-8') as f:
            for item in rows:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        with txt_path.open('w', encoding='utf-8') as out:
            hits = 0
            out.write(f'{title}筛选输出\n')
            out.write(f'数据根目录: {args.root}\n')
            out.write(f'主视觉参考: {args.camera}\n')
            for item in rows:
                if not item.get('construction'):
                    continue
                hits += 1
                out.write(f"\n目录: {item.get('clip_dir')}\n")
                for frame, fname in zip(item.get('frames', []), item.get('files', [])):
                    out.write(f'命中帧: 第{frame}帧, 文件名: {fname}\n')
                if item.get('features'):
                    out.write('命中特征: ' + ', '.join(item.get('features', [])) + '\n')
                if item.get('impact'):
                    out.write('证据/影响: ' + str(item.get('impact')) + '\n')
                if item.get('reason'):
                    out.write('模型依据: ' + str(item.get('reason')) + '\n')
                if item.get('image_files'):
                    out.write('命中原图:\n')
                    for image_file in item.get('image_files', []):
                        out.write(f'- {image_file}\n')
            out.write(f"\n完成: 共处理 {len(rows)} 个目录，命中 {hits} 个。\n")


def write_error_outputs(records: list[dict[str, Any]], error_jsonl: Path | None) -> tuple[int, Path | None]:
    if error_jsonl is None:
        return 0, None
    errors = [r for r in records if r.get('ok') is not True or r.get('error')]
    error_jsonl.parent.mkdir(parents=True, exist_ok=True)
    error_json = error_jsonl.with_suffix('.json')
    error_txt = error_jsonl.with_suffix('.txt')
    with error_jsonl.open('w', encoding='utf-8') as f:
        for row in errors:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    error_json.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding='utf-8')
    with error_txt.open('w', encoding='utf-8') as f:
        for row in errors:
            f.write(str(row.get('clip_dir', '')) + '\n')
    return len(errors), error_txt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--jsonl', type=Path, required=True)
    ap.add_argument('--json', type=Path, default=None)
    ap.add_argument('--txt', type=Path, required=True)
    ap.add_argument('--error-jsonl', type=Path, default=None)
    ap.add_argument('--camera', type=Path, default=DEFAULT_CAMERA_REL)
    ap.add_argument('--lane-output-dir', type=Path, default=Path('/mnt/wzb11/t3_qwen_filter_project/outputs/lane'))
    ap.add_argument('--work-output-dir', type=Path, default=Path('/mnt/wzb11/t3_qwen_filter_project/outputs/work'))
    ap.add_argument('--stride', type=int, default=200)
    ap.add_argument('--workers', type=int, default=128)
    ap.add_argument('--thumb-width', type=int, default=384)
    ap.add_argument('--endpoint', default=os.environ.get('EAS_ENDPOINT', ''))
    ap.add_argument('--token', default=os.environ.get('EAS_TOKEN', ''))
    ap.add_argument('--model', default=os.environ.get('EAS_MODEL', 'Qwen3.6-35B-A3B'))
    ap.add_argument('--route', default=os.environ.get('EAS_PREFER_ROUTE', 'chat_completions'))
    ap.add_argument('--timeout', type=int, default=int(os.environ.get('EAS_TIMEOUT', '30')))
    ap.add_argument('--max-new-tokens', type=int, default=360)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--done-jsonl', type=Path, action='append', default=[])
    ap.add_argument('--exclude-prefix', type=Path, action='append', default=[])
    ap.add_argument('--summarize-only', action='store_true')
    args = ap.parse_args()
    if args.json is None:
        args.json = args.jsonl.with_suffix('.json')
    if args.error_jsonl is None:
        args.error_jsonl = args.jsonl.with_name(args.jsonl.stem + '_errors.jsonl')

    if args.summarize_only:
        records = write_json_summary(args.json, args.jsonl)
        total, lane_hits, tidal_hits = write_txt_summary(args.txt, records, args.root, args.stride, args.camera)
        write_task_file_outputs(records, args)
        error_total, error_txt = write_error_outputs(records, args.error_jsonl)
        print(f'summarized total={total} lane_hits={lane_hits} tidal_hits={tidal_hits} errors={error_total} error_txt={error_txt} json={args.json} txt={args.txt}')
        return 0

    if not args.endpoint or not args.token:
        raise SystemExit('EAS endpoint/token required via args or env')

    done = load_done(args.jsonl)
    for done_jsonl in args.done_jsonl:
        done |= load_done(done_jsonl)
    excludes = [str(p.resolve()) for p in args.exclude_prefix]
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f'starting stream done={len(done)} workers={args.workers} camera={args.camera}', flush=True)
    lock = threading.Lock()
    processed = 0
    submitted = 0
    seen = 0
    lane_hits = 0
    tidal_hits = 0
    error_count = 0
    start_time = time.time()
    max_inflight = max(1, args.workers * 2)

    def handle_future(fut: cf.Future, out) -> None:
        nonlocal processed, lane_hits, tidal_hits, error_count
        try:
            row = fut.result()
        except Exception as exc:
            row = {
                'clip_dir': '',
                'camera': str(args.camera),
                'ok': False,
                'sample_count': 0,
                'lane_impact': {'construction': False, 'frames': [], 'files': [], 'reason': ''},
                'tidal_lane': {'construction': False, 'frames': [], 'files': [], 'reason': ''},
                'error': repr(exc),
            }
        with lock:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            out.flush()
            processed += 1
            if row.get('error'):
                error_count += 1
            if (row.get('lane_impact') or {}).get('construction'):
                lane_hits += 1
            if (row.get('tidal_lane') or {}).get('construction'):
                tidal_hits += 1
            if processed % 1000 == 0 or processed == submitted:
                elapsed = max(time.time() - start_time, 1e-6)
                print(
                    f'processed={processed}/{submitted} seen={seen} lane_hits={lane_hits} '
                    f'tidal_hits={tidal_hits} errors={error_count} ok={row.get("ok")} '
                    f'rate={processed/elapsed:.3f} clips/s elapsed={elapsed:.1f}s',
                    flush=True,
                )

    with args.jsonl.open('a', encoding='utf-8') as out:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures: dict[cf.Future, Path] = {}
            for clip in iter_clip_dirs(args.root):
                clip_resolved = str(clip.resolve())
                if excludes and any(clip_resolved.startswith(prefix.rstrip('/') + '/') or clip_resolved == prefix for prefix in excludes):
                    continue
                if seen < args.start:
                    seen += 1
                    continue
                seen += 1
                if str(clip) in done:
                    continue
                fut = ex.submit(process_clip, args, clip)
                futures[fut] = clip
                submitted += 1
                if submitted % 1000 == 0:
                    print(f'submitted={submitted} seen={seen} inflight={len(futures)} latest={clip}', flush=True)
                if args.limit and submitted >= args.limit:
                    break
                while len(futures) >= max_inflight:
                    done_futs, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                    for done_fut in done_futs:
                        futures.pop(done_fut, None)
                        handle_future(done_fut, out)
            print(f'discovery_finished seen={seen} submitted={submitted} inflight={len(futures)}', flush=True)
            for fut in cf.as_completed(list(futures)):
                handle_future(fut, out)

    records = write_json_summary(args.json, args.jsonl)
    total, lane_total, tidal_total = write_txt_summary(args.txt, records, args.root, args.stride, args.camera)
    write_task_file_outputs(records, args)
    error_total, error_txt = write_error_outputs(records, args.error_jsonl)
    print(f'done total={total} lane_hits={lane_total} tidal_hits={tidal_total} errors={error_total} error_txt={error_txt} jsonl={args.jsonl} json={args.json} txt={args.txt}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
