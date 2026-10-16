"""Resumeable variable-lane search with independent deceleration exclusion."""
import argparse,concurrent.futures as cf,hashlib,json,shutil,subprocess,sys,time,urllib.request,fcntl
from pathlib import Path
import qwen_lane_marking_clip_batch as batch
import qwen_lane_marking_visual_filter as vision
import review_variable_vs_deceleration as strict

STRICT_EXTRA = '\n严格保留条件：目标必须位于机动车行驶车道边界。必须清楚看见连续边界实线及附着其上、向同一机动车道内伸出的短斜齿；不能仅凭鱼骨外观判断。独立菱形/平行四边形/矩形白块及纵向减速标线一律排除；人行道、斑马线、人行横道、非机动车道、停车位、导流区、路口虚线与路面文字一律排除。看不清单元轮廓、与实线的连接关系或所属车道，则uncertain，不保留。JSON另外必须输出motor_vehicle_lane、pedestrian_or_cycle_area、geometry_certain三个布尔字段，分别表示确认位于机动车道、位于行人或非机动车区域、确认标线形态与连接关系清楚。'
REVIEW_PROMPT = strict.PROMPT + STRICT_EXTRA

def verify(path,config,args,label):
    payload={'model':config['EAS_MODEL'],'messages':[{'role':'user','content':[
        {'type':'text','text':'同一待审帧的完整道路视图和近处路面放大图，请结合两图确认边界线和白色单元的连接关系'},
        {'type':'image_url','image_url':{'url':vision.encode_image(path,2048)}},
        {'type':'image_url','image_url':{'url':strict.encode(path)}},
        {'type':'text','text':REVIEW_PROMPT}]}],'stream':False,'temperature':0,'max_tokens':500,'chat_template_kwargs':{'enable_thinking':False}}
    request=urllib.request.Request(config['EAS_ENDPOINT']+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':config['EAS_TOKEN'],'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=args.timeout) as response:raw=json.load(response)['choices'][0]['message']['content']
    clean=raw.strip()
    if clean.startswith('```'):clean=clean.split('\n',1)[1].rsplit('```',1)[0].strip()
    d=json.loads(clean)
    if d.get('category') not in ('variable','deceleration','other','uncertain') or any(type(d.get(k)) is not bool for k in ('attached_short_teeth','independent_diamond_blocks')) or not isinstance(d.get('reason'),str):raise ValueError('Invalid shape review response')
    if any(type(d.get(k)) is not bool for k in ('motor_vehicle_lane','pedestrian_or_cycle_area','geometry_certain')):raise ValueError('Missing strict geometry fields')
    accepted=d['category']=='variable' and d['attached_short_teeth'] and not d['independent_diamond_blocks'] and d['motor_vehicle_lane'] and not d['pedestrian_or_cycle_area'] and d['geometry_certain']
    if accepted:
        # Independent shape-only check avoids motor-lane criteria priming the
        # model into treating deceleration blocks as variable-lane teeth.
        payload['messages'][0]['content']=[{'type':'image_url','image_url':{'url':strict.encode(path)}},{'type':'text','text':strict.PROMPT}]
        request=urllib.request.Request(config['EAS_ENDPOINT']+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':config['EAS_TOKEN'],'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=args.timeout) as response:second=json.load(response)['choices'][0]['message']['content'].strip()
        if second.startswith('```'):second=second.split('\n',1)[1].rsplit('```',1)[0].strip()
        check=json.loads(second)
        if check.get('category') not in ('variable','deceleration','other','uncertain') or any(type(check.get(k)) is not bool for k in ('attached_short_teeth','independent_diamond_blocks')):raise ValueError('Invalid independent geometry check')
        d['independent_shape_review']=check
        accepted=check['category']=='variable' and check['attached_short_teeth'] and not check['independent_diamond_blocks']
    return {**d,'verified':accepted,'evidence':d['reason']}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--clip-list',type=Path,required=True);parser.add_argument('--out-dir',type=Path,required=True);parser.add_argument('--photos-dir',type=Path,required=True);parser.add_argument('--workers',type=int,default=32);parser.add_argument('--detach',action='store_true');a=parser.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True);a.photos_dir.mkdir(parents=True,exist_ok=True)
    if a.detach:
        with (a.out_dir/'run.log').open('a') as f:p=subprocess.Popen([sys.executable,__file__,*[x for x in sys.argv[1:] if x!='--detach']],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        (a.out_dir/'run.pid').write_text(str(p.pid));print(json.dumps({'pid':p.pid}));return
    lock=(a.out_dir/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=vision.load_configuration();a.camera='camera/center_camera_fov120';a.stride=100;a.thumb_width=1024;a.separate_images=True;a.max_tokens=800;a.attempts=2;a.timeout=180;a.variable_only=True
    positive=Path('/mnt/data_clip/t3_origin/dlp_data_analysis/data_collection/gt_data/pilotGtParser/dongfeng_t3/20260522/LDP41B960SD165294/2026_04/2026_04_24/2026_04_24_09_47_35_dlp_pilotGtParser/camera/center_camera_fov120/1776995286711394000.jpg')
    config['reference_images']={'可变车道':vision.encode_image(positive,2048)}
    prompt=batch.make_prompt('');prompt['可变车道']+='\n必须排除纵向减速标线：沿每条边界两侧排列的独立菱形或平行四边形白块不是可变车道。只有朝车道内侧、连接在边界实线上的短斜齿才能命中。'
    prompt['可变车道']+='\n只判断机动车行驶车道。排除人行道、斑马线、非机动车道和停车位；独立菱形减速白块不能命中，标线形态看不清则不命中。'
    signature=hashlib.sha256((prompt['可变车道']+REVIEW_PROMPT+'remaining-v4-strict-motor-consensus').encode()).hexdigest();batch.verify_candidate_frame=verify
    (a.out_dir/'prompt_snapshot.json').write_text(json.dumps({'initial':prompt['可变车道'],'review':REVIEW_PROMPT},ensure_ascii=False,indent=2))
    clips=[Path(x) for x in a.clip_list.read_text().splitlines() if x];assert len(set(clips))==len(clips)
    rows={};path=a.out_dir/'results.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                d=json.loads(line)
                if d['run_signature']==signature:
                    rows[d['clip_dir']]=d
                elif d.get('ok') and not d.get('variable_lane') and not d.get('variable_verification'):
                    # A stricter exclusion rule preserves earlier clear negatives;
                    # all previous candidates and reviewed frames are screened again.
                    d['reuse_note']='Earlier initial-screen negative; stricter exclusions preserve this negative.'
                    rows[d['clip_dir']]=d
            except (ValueError,KeyError):pass
    def save(status):
        hits=[d for d in rows.values() if d.get('ok') and d.get('variable_lane')];failed=[d for d in rows.values() if not d.get('ok') and not d.get('skipped')]
        report={'status':status,'total_clips':len(clips),'processed_clips':len(rows),'successful_clips':sum(bool(d.get('ok')) for d in rows.values()),'skipped_clips':sum(bool(d.get('skipped')) for d in rows.values()),'failed_clips':len(failed),'variable_clips':len(hits),'workers':a.workers,'pid':__import__('os').getpid(),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'photos_dir':str(a.photos_dir)}
        tmp=a.out_dir/'progress.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(a.out_dir/'progress.json')
        (a.out_dir/'clip_names.txt').write_text(''.join(d['clip_name']+'\n' for d in hits));(a.out_dir/'clip_paths.txt').write_text(''.join(d['clip_dir']+'\n' for d in hits));(a.out_dir/'failed_clip_paths.txt').write_text(''.join(d['clip_dir']+'\n' for d in failed));return report
    save('running')
    with path.open('a') as f:
        for round_no in range(3):
            pending=[p for p in clips if not rows.get(str(p),{}).get('ok') and not rows.get(str(p),{}).get('skipped')]
            if not pending:break
            with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
                iterator=iter(pending);active={}
                for _ in range(min(a.workers,len(pending))):
                    p=next(iterator);active[pool.submit(batch.process_clip,p,config,prompt,a,signature)]=p
                while active:
                    done,_=cf.wait(active,return_when=cf.FIRST_COMPLETED)
                    for future in done:
                        active.pop(future);d=future.result()
                        if d.get('ok') and d.get('variable_lane'):
                            try:
                                dest=a.photos_dir/d['clip_name'];dest.mkdir(exist_ok=True)
                                for s in d['samples']:
                                    if s['frame'] in d['variable_frames']:
                                        source=Path(s['path']);target=dest/f'first_hit_frame_{s["frame"]:06d}_{source.name}'
                                        if target.exists() and target.read_bytes()!=source.read_bytes():target=dest/f'first_hit_frame_{s["frame"]:06d}_{hashlib.sha256(d["clip_dir"].encode()).hexdigest()[:8]}_{source.name}'
                                        shutil.copyfile(source,target)
                            except Exception as e:d['ok']=False;d['error']=str(e)
                        rows[d['clip_dir']]=d;f.write(json.dumps(d,ensure_ascii=False)+'\n');f.flush()
                        if len(rows)%25==0 or d.get('variable_lane') or not d.get('ok'):print(json.dumps(save('running'),ensure_ascii=False),flush=True)
                        p=next(iterator,None)
                        if p:active[pool.submit(batch.process_clip,p,config,prompt,a,signature)]=p
    print(json.dumps(save('complete' if all(d.get('ok') or d.get('skipped') for d in rows.values()) else 'complete_with_errors'),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
