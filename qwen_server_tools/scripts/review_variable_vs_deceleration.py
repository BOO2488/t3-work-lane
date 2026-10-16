import argparse,base64,concurrent.futures,io,json,shutil,sys,time,urllib.request
from pathlib import Path
from PIL import Image
import qwen_lane_marking_visual_filter as vision

ROOT=Path('/mnt/wzb11/t3_qwen_filter_project/lane-chaixi')
OUT=ROOT/'review_variable_20261013'
POS=Path('/mnt/wzb11/t3_qwen_filter_project/outputs/lane/20260522_fov120/clips_with_photos/2026_04_24_09_47_35_dlp_pilotGtParser/first_hit_frame_000300_1776995286711394000.jpg')
NEG=ROOT/'可变车道/2026_04_24_05_45_33_dlp_pilotGtParser/first_hit_frame_000000_1776980734907707000.jpg'
PROMPT='仔细看近处的每一条白色车道边界。减速标线是独立的白色菱形/平行四边形块，常在同一条边界线左右两侧都有独立白块排列，单元粗而宽，块之间有空隙。可变车道是锯齿边界，每一条边界只有朝车道内侧的细短斜齿，像一把锯子，斜齿与边界线连接，单元细长或三角形，并非独立四边形。重点区分同一条边界两侧的独立菱形块与单侧细齿，别把整个车道左右两条边界混淆。只按真实可见形状判断；不清楚为uncertain。严格JSON：{"category":"variable|deceleration|other|uncertain","attached_short_teeth":false,"independent_diamond_blocks":false,"reason":"描述白色单元的形状、在单条边界的哪一侧"}。'

def encode(p):
    with Image.open(p) as im:
        im=im.convert('RGB');w,h=im.size
        im=im.crop((int(w*.12),int(h*.47),int(w*.88),int(h*.73)))
        im.thumbnail((1800,900))
        buf=io.BytesIO();im.save(buf,'JPEG',quality=95)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validate',action='store_true');a=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    config=vision.load_configuration()
    (OUT/'prompt_snapshot.txt').write_text(PROMPT)
    files=[POS,NEG] if a.validate else sorted((ROOT/'可变车道').glob('*/*.jpg'))
    snapshot=OUT/'input_manifest.json'
    if not a.validate: snapshot.write_text(json.dumps([str(p) for p in files],ensure_ascii=False,indent=2))
    def ask(p):
        content=[]+[{'type':'text','text':'以下是待审图'}, {'type':'image_url','image_url':{'url':encode(p)}},{'type':'text','text':PROMPT}]
        payload={'model':config['EAS_MODEL'],'messages':[{'role':'user','content':content}],'stream':False,'temperature':0,'max_tokens':400,'chat_template_kwargs':{'enable_thinking':False}}
        for attempt in range(3):
            try:
                req=urllib.request.Request(config['EAS_ENDPOINT']+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':config['EAS_TOKEN'],'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=180) as response:raw=json.load(response)['choices'][0]['message']['content']
                clean=raw.strip()
                if clean.startswith('```'):clean=clean.split('\n',1)[1].rsplit('```',1)[0].strip()
                d=json.loads(clean)
                assert d['category'] in ('variable','deceleration','other','uncertain')
                assert type(d['attached_short_teeth']) is bool and type(d['independent_diamond_blocks']) is bool
                if d['category']=='variable' and (not d['attached_short_teeth'] or d['independent_diamond_blocks']):d['category']='uncertain'
                return {'image':str(p),'clip_name':p.parent.name,'ok':True,**d}
            except Exception as e:
                error=str(e).replace(config['EAS_TOKEN'],'[REDACTED]');time.sleep(2)
        return {'image':str(p),'clip_name':p.parent.name,'ok':False,'category':'uncertain','reason':error}
    rows=[]
    target=OUT/('validation.jsonl' if a.validate else 'results.jsonl')
    with target.open('w') as f,concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for d in pool.map(ask,files):
            rows.append(d);f.write(json.dumps(d,ensure_ascii=False)+'\n');f.flush()
            print(json.dumps({'done':len(rows),'total':len(files),'clip':d['clip_name'],'category':d['category']},ensure_ascii=False),flush=True)
    if a.validate:
        assert rows[0]['category']=='variable' and rows[1]['category']=='deceleration',rows
        return
    grouped={}
    for d in rows:grouped.setdefault(d['clip_name'],[]).append(d)
    counts={'variable':0,'deceleration':0,'other':0,'uncertain':0}
    for name,items in grouped.items():
        category='variable' if any(d['category']=='variable' for d in items) else ('uncertain' if any(d['category']=='uncertain' for d in items) else ('deceleration' if any(d['category']=='deceleration' for d in items) else 'other'))
        counts[category]+=1
        if category!='variable':
            dest=OUT/('减速标线' if category=='deceleration' else '待确认' if category=='uncertain' else '其他')/name
            dest.parent.mkdir(exist_ok=True)
            assert not dest.exists()
            (ROOT/'可变车道'/name).rename(dest)
    (OUT/'summary.json').write_text(json.dumps({'total_clips':len(grouped),'counts':counts,'failed_requests':sum(not d['ok'] for d in rows)},ensure_ascii=False,indent=2))
    (OUT/'retained_clip_names.txt').write_text(''.join(name+'\n' for name,items in grouped.items() if any(d['category']=='variable' for d in items)))
    print(json.dumps(counts,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
