"""Inventory original clips and persist exclusion of all existing accepted clips."""
import json
import os
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

PROJECT=Path(__file__).resolve().parents[1]
SOURCE=Path('/mnt/data_clip/t3_origin/dlp_data_analysis/data_collection/gt_data/pilotGtParser/dongfeng_t3')
OUT=PROJECT/'outputs/new_strict_roadwork_20261014'

def dirs(path):
    with os.scandir(path) as it:
        return sorted(Path(e.path) for e in it if e.is_dir(follow_symlinks=False))

def main():
    OUT.mkdir(exist_ok=True)
    old=json.loads((PROJECT/'outputs/work_strict_road_review_20261014/action_plan.json').read_text())['rows']
    accepted=[r for r in old if r['decision']=='keep']
    excluded_names={r['clip_name'] for r in accepted}
    excluded_paths={s['source_clip_dir'] for r in accepted for s in r['sources']}
    exclusions={'clip_names':sorted(excluded_names),'source_paths':sorted(excluded_paths),
                'image_sha256':sorted({r['sha256'] for r in accepted})}
    (OUT/'exclude_existing_52.json').write_text(json.dumps(exclusions,ensure_ascii=False,indent=2)+'\n')
    level=dirs(SOURCE)
    with ThreadPoolExecutor(12) as pool:
        for depth in range(4):
            level=[p for group in pool.map(dirs,level) for p in group]
            print('depth',depth+2,'directories',len(level),flush=True)
    clips=[p for p in level if p.name.endswith('_dlp_pilotGtParser')]
    rows=[]
    for p in clips:
        relative=p.relative_to(SOURCE)
        rows.append({'clip_dir':str(p),'clip_name':p.name,'batch':relative.parts[0],
                     'vehicle':relative.parts[1],
                     'excluded':p.name in excluded_names or str(p) in excluded_paths})
    (OUT/'inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    summary={'source':str(SOURCE),'target_new_clips':672,'existing_excluded_clip_names':len(excluded_names),
             'total_source_paths':len(rows),'excluded_source_paths':sum(r['excluded'] for r in rows),
             'eligible_source_paths':sum(not r['excluded'] for r in rows),
             'batches':dict(Counter(r['batch'] for r in rows)),
             'confirmed_new_clips':0,'status':'inventory_complete'}
    (OUT/'progress.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
