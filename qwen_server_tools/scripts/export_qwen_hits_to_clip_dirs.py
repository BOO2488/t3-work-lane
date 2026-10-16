#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

CAMERA_REL = Path('camera/center_camera_fov120')
IMG_EXT = {'.jpg', '.jpeg', '.png'}


def iter_rows(jsonl: Path):
    with jsonl.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                yield {'_bad_line': line_no, '_error': repr(exc)}
                continue
            yield row


def main() -> int:
    ap = argparse.ArgumentParser(description='Export Qwen construction hits to clip-name folders.')
    ap.add_argument('--jsonl', type=Path, required=True, help='Qwen batch jsonl output')
    ap.add_argument('--out-dir', type=Path, required=True, help='output root, e.g. /mnt/wzb11/道路施工-all')
    ap.add_argument('--camera', type=Path, default=CAMERA_REL, help='camera path relative to each clip')
    ap.add_argument('--skip-existing-clips', action='store_true', help='leave existing clip folders untouched')
    ap.add_argument('--exclude-clips-dir', type=Path, help='exclude clip names already present in this directory')
    ap.add_argument('--clear', action='store_true', help='clear output directory before exporting')
    ap.add_argument('--copy-all-hit-frames', action='store_true', help='copy all returned frames; default also copies all returned frames')
    args = ap.parse_args()

    if args.clear and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing_clip_names = {p.name for p in args.out_dir.iterdir() if p.is_dir()}
    excluded_clip_names = set()
    if args.exclude_clips_dir:
        excluded_clip_names = {p.name for p in args.exclude_clips_dir.iterdir() if p.is_dir()}

    rows = [r for r in iter_rows(args.jsonl) if not r.get('_bad_line')]
    hit_rows = [r for r in rows if r.get('construction') and r.get('frames') and r.get('files')]

    paths_by_base: dict[str, set[str]] = defaultdict(set)
    for r in hit_rows:
        clip_dir = str(r.get('clip_dir', ''))
        if clip_dir:
            paths_by_base[Path(clip_dir).name].add(clip_dir)

    copied = 0
    skipped_existing = 0
    missing = []
    manifest_rows = []

    for r in hit_rows:
        clip = Path(str(r.get('clip_dir', '')))
        base = clip.name
        if not base:
            continue
        if base in excluded_clip_names:
            continue
        out_name = base
        # If two different source clips share the same basename, add parent hints.
        if len(paths_by_base[base]) > 1:
            parents = [p.name for p in list(clip.parents)[:3] if p.name]
            suffix = '__' + '_'.join(reversed(parents[:2])) if parents else '__dup'
            out_name = base + suffix

        dst_dir = args.out_dir / out_name
        if args.skip_existing_clips and out_name in existing_clip_names:
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)

        frames = r.get('frames') or []
        files = r.get('files') or []
        for frame, fname in zip(frames, files):
            try:
                frame_i = int(frame)
            except Exception:
                frame_i = -1
            src = clip / args.camera / str(fname)
            dst = dst_dir / f'frame_{frame_i:06d}_{fname}'
            if not src.exists():
                missing.append((str(clip), str(frame), str(fname), str(src)))
                continue
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                skipped_existing += 1
            else:
                shutil.copy2(src, dst)
                copied += 1
            manifest_rows.append((str(clip), out_name, str(frame), str(fname), str(src), str(dst)))

    manifest = args.out_dir / 'manifest.tsv'
    with manifest.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['clip_dir', 'output_clip_name', 'frame', 'source_file', 'source_path', 'output_path'])
        w.writerows(manifest_rows)

    missing_path = args.out_dir / 'missing.tsv'
    with missing_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['clip_dir', 'frame', 'source_file', 'expected_source_path'])
        w.writerows(missing)

    clip_dirs = [p for p in args.out_dir.iterdir() if p.is_dir()]
    image_count = sum(
        1
        for d in clip_dirs
        for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXT
    )

    print(f'total_rows={len(rows)}')
    print(f'hit_clips={len(hit_rows)}')
    print(f'copied={copied}')
    print(f'skipped_existing={skipped_existing}')
    print(f'missing={len(missing)}')
    print(f'output_clip_dirs={len(clip_dirs)}')
    print(f'output_images={image_count}')
    print(f'manifest={manifest}')
    print(f'missing_file={missing_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
