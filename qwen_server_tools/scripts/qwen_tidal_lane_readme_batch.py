#!/usr/bin/env python3
"""Run the existing Qwen tidal-lane batch with the strict README criteria."""

from qwen_tidal_lane_batch import *  # noqa: F401,F403
import qwen_tidal_lane_batch as base
import os
import argparse
import json
import re
import sys
from pathlib import Path

_original_iter_clip_dirs = base.iter_clip_dirs
_original_parse_answer = base.parse_answer


base.PROMPT = """你是车载前视图潮汐车道/可逆车道严格筛选员。图中是同一 clip 按时间抽取的多张图片，左上角有 frame 编号。

只有同时满足以下两项，并且两项作用于同一组车道时，才返回 construction=true：
1. 空中动态车道信号：每条车道上方独立布置红叉、绿箭头等车道级通行/禁行信号，能体现车道方向或路权动态变化。
2. 特殊地面标线：受控车道附近有沿行驶方向连续、重复的白色鱼骨状小箭头/V形短线，并可伴随黄色可逆车道边界线。

必须排除：普通路口红绿灯；只有车道级红叉/绿箭但没有连续鱼骨地面标线；只有鱼骨线但没有动态车道信号；单个或普通直行/转向箭头；斑马线；导流区大斜纹；普通车道线；高速公路、收费站、施工导改、普通护栏；远处模糊无法确认的标志。

判断必须基于画面中实际可见证据。不要因为道路宽、存在龙门架或有普通箭头而猜测。无法同时确认两项时必须返回 false。

只输出严格 JSON，不要 markdown：
{"construction": true/false, "frames": [帧号整数...], "features": ["空中信号", "静态标线"], "impact": "描述两类证据如何共同作用于同一组车道", "reason": "一句话说明可见证据"}

若为 false，frames 和 features 必须为空数组。"""


def iter_clip_dirs_from_list(root: Path):
    list_path = os.environ.get('QWEN_CLIP_LIST', '')
    if not list_path:
        yield from _original_iter_clip_dirs(root)
        return
    root_resolved = root.resolve()
    with Path(list_path).open('r', encoding='utf-8') as src:
        for line in src:
            p = Path(line.strip())
            if not line.strip() or not p.is_dir():
                continue
            try:
                p.resolve().relative_to(root_resolved)
            except ValueError:
                continue
            yield p


base.iter_clip_dirs = iter_clip_dirs_from_list


def load_successful_done(path: Path) -> set[str]:
    """Failed requests must be retried on the next run."""
    done = set()
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
                if row.get('ok') is True and row.get('clip_dir'):
                    done.add(row['clip_dir'])
            except (ValueError, TypeError):
                continue
    return done


def parse_strict_answer(answer: str):
    raw = re.sub(r'^```(?:json)?\s*', '', str(answer).strip())
    raw = re.sub(r'\s*```$', '', raw)
    data = json.loads(raw)
    if type(data.get('construction')) is not bool:
        raise ValueError('Qwen construction must be a JSON boolean')
    if not isinstance(data.get('frames'), list) or not isinstance(data.get('features'), list):
        raise ValueError('Qwen frames/features must be JSON arrays')
    if data['construction']:
        if not {'空中信号', '静态标线'}.issubset(data['features']) or not data['frames']:
            raise ValueError('Positive result lacks both required features or frame IDs')
    elif data['frames'] or data['features']:
        raise ValueError('Negative result must have empty frames/features')
    return _original_parse_answer(raw)


base.load_done = load_successful_done
base.parse_answer = parse_strict_answer


def preflight_main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--camera', type=Path, default=Path('camera/center_camera_fov30'))
    ap.add_argument('--stride', type=int, default=100)
    args = ap.parse_args()
    args.endpoint = os.environ.get('EAS_ENDPOINT', '')
    args.token = os.environ.get('EAS_TOKEN', '')
    args.model = os.environ.get('EAS_MODEL', 'Qwen3.6-35B-A3B')
    args.route = os.environ.get('EAS_PREFER_ROUTE', 'chat_completions')
    args.timeout = 30
    args.max_new_tokens = 220
    args.review_dir = None
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if not args.endpoint or not args.token:
        row = {'ok': False, 'error': 'Missing EAS endpoint/token'}
    else:
        row = {'ok': False, 'error': 'No clip with sampled camera JPGs found'}
        for clip in iter_clip_dirs_from_list(args.root):
            if base.sampled_images(clip, args.camera, args.stride):
                try:
                    row = base.process_clip(args, clip)
                except Exception as exc:
                    row = {'ok': False, 'error': repr(exc)}
                break
    args.report.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"preflight ok={row.get('ok')} report={args.report}", flush=True)
    if row.get('ok') is not True:
        print('Qwen request failed; full batch and export were not started.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(preflight_main() if '--preflight-only' in sys.argv else base.main())
