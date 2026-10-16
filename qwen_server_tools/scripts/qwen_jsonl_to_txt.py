#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--jsonl', type=Path, required=True)
    ap.add_argument('--txt', type=Path, required=True)
    ap.add_argument('--total-dirs', type=int, default=10581)
    args=ap.parse_args()
    total=hit=err=0
    rows=[]
    if args.jsonl.exists():
        with args.jsonl.open('r', encoding='utf-8') as src:
            for line in src:
                if not line.strip():
                    continue
                total += 1
                try:
                    row=json.loads(line)
                except Exception:
                    err += 1
                    continue
                if not row.get('ok'):
                    err += 1
                if row.get('construction'):
                    hit += 1
                    rows.append(row)
    tmp=args.txt.with_suffix(args.txt.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        f.write('Qwen视觉模型施工筛选实时结果\n')
        f.write('源目录: /mnt/data_clip/t3_origin/dlp_data_analysis/data_collection/gt_data/pilotGtParser/dongfeng_t3/20260522\n')
        f.write('策略: 每100帧抽一张前视图，模型判断施工场景\n')
        f.write(f'实时统计: 已处理 {total}/{args.total_dirs} 个目录，发现施工 {hit} 个目录，错误 {err} 个。\n')
        f.write('说明: 本文件每30秒自动刷新一次，批处理结束后数值会停止变化。\n\n')
        for row in rows:
            f.write(f"目录: {row.get('clip_dir')}\n")
            for frame, fname in zip(row.get('frames') or [], row.get('files') or []):
                f.write(f"  施工场景: 第{frame}帧开始, 文件名: {fname}\n")
            reason=row.get('reason') or ''
            if reason:
                f.write(f"  模型依据: {reason}\n")
            f.write('\n')
    tmp.replace(args.txt)
    print(f'processed={total} hits={hit} errors={err} txt={args.txt}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
