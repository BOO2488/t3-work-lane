"""Resumable strict EAS candidate discovery. Never counts model hits as accepted."""
import argparse, base64, hashlib, io, json, re, time, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
from discover_new_roadwork import PROJECT, OUT
from qwen_lane_marking_visual_filter import load_configuration

PROMPT='''逐格查看车载图片，寻找明确的道路施工实体，不是施工提示或交通隔离。
合格：清楚可见的道路破拆开挖/管沟铺设/铺路压实作业/修筑路基/正在建设且未完成的道路桥梁结构。
不合格：只有锥桶、水马、围挡、施工标志；只有机械停放、运输、行驶；只露围挡上方吊臂；路外楼房施工；保洁绿化；正常桥梁。不可凭工程车猜测施工。
每格标有 ID。只列真正清楚看见实际施工作业面或未完成道路结构的格子。主体太远、遮挡、模糊一律 uncertain，不列为 clear。
输出严格 JSON：{"clear":[{"id":整数,"type":"开挖/铺装/管沟/路基/桥梁","evidence":"具体位置和看见的实体"}],"uncertain":true或false,"reason":"简短说明"}。不合格 clear=[]。'''

def sample(row):
    folder=Path(row['clip_dir'])/'camera'
    images=[]
    for camera in ['center_camera_fov30','center_camera_fov120']:
        files=sorted((folder/camera).glob('*.jpg'))
        if not files:continue
        indices=sorted({round(i*(len(files)-1)/5) for i in range(6)})
        offset=len(images)
        images.extend({'id':offset+j,'camera':camera,'frame':idx,'path':str(files[idx])}
                      for j,idx in enumerate(indices))
    return images

def sheet(samples):
    canvas=Image.new('RGB',(768*3,462*((len(samples)+2)//3)),'#dddddd')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    for i,r in enumerate(samples):
        with Image.open(r['path']) as im:
            im=im.convert('RGB'); im.thumbnail((768,432))
            x,y=i%3*768,i//3*462
            canvas.paste(im,(x,y+30))
            draw.text((x+4,y+2),f"ID {r['id']} {r['camera'].replace('center_camera_','')} frame {r['frame']}",font=font,fill='black')
    return canvas

def process(row,config,args):
    result=dict(row,ok=False,stage='model_candidate_only')
    try:
        samples=sample(row)
        if not samples:return dict(result,ok=True,clear=[],reason='no front images',samples=[])
        canvas=sheet(samples)
        buffer=io.BytesIO(); canvas.save(buffer,format='JPEG',quality=90)
        payload={'model':config['EAS_MODEL'],'messages':[{'role':'user','content':[
            {'type':'text','text':PROMPT}, {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()}}]}],
            'temperature':0,'max_tokens':700,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}
        request=urllib.request.Request(config['EAS_ENDPOINT']+'/v1/chat/completions',data=json.dumps(payload).encode(),
            headers={'Authorization':config['EAS_TOKEN'],'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=args.timeout) as response:
            data=json.loads(response.read(2000000))
        answer=data['choices'][0]['message']['content']
        parsed=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',answer.strip()))
        assert isinstance(parsed.get('clear'),list)
        lookup={r['id']:r for r in samples}
        clear=[]
        for item in parsed['clear']:
            assert type(item['id']) is int and item['id'] in lookup
            assert isinstance(item.get('evidence'),str) and item['evidence']
            clear.append(dict(lookup[item['id']],evidence=item['evidence'],type=item.get('type')))
        result.update(ok=True,clear=clear,samples=samples,reason=parsed.get('reason'),uncertain=parsed.get('uncertain'))
        if clear:
            key=hashlib.sha256(row['clip_dir'].encode()).hexdigest()[:16]
            target=OUT/'candidate_sheets'/f'{key}.jpg'
            canvas.save(target,quality=94)
            result['sheet']=str(target)
    except Exception as error:
        result['error']=(type(error).__name__+': '+str(error)).replace(config['EAS_TOKEN'],'[REDACTED]')[:400]
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=16)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--timeout',type=int,default=120)
    args=parser.parse_args()
    config=load_configuration()
    inventory=json.loads((OUT/'inventory.json').read_text())
    priority={p.name for p in (PROJECT/'outputs/work_readme_setsid_20261005_w32').iterdir() if p.is_dir()}
    done=set()
    log=OUT/'model_screen.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            r=json.loads(line)
            if r.get('ok'):done.add(r['clip_dir'])
    eligible=[r for r in inventory if not r['excluded'] and r['clip_dir'] not in done]
    eligible.sort(key=lambda r:(r['clip_name'] not in priority,r['clip_name'],r['vehicle'],r['batch']))
    # Same clip name is reviewed once even if stored in multiple batches/vehicles.
    names=set(); pending=[]
    done_names={Path(p).name for p in done}
    for r in eligible:
        if r['clip_name'] in names or r['clip_name'] in done_names:continue
        names.add(r['clip_name']); pending.append(r)
    pending=pending[:args.limit]
    (OUT/'candidate_sheets').mkdir(exist_ok=True)
    print(json.dumps({'pending_this_run':len(pending),'already_screened':len(done),'workers':args.workers}),flush=True)
    failures=0; candidates=0
    with log.open('a') as f, ThreadPoolExecutor(args.workers) as pool:
        futures=[pool.submit(process,r,config,args) for r in pending]
        for index,future in enumerate(as_completed(futures),1):
            row=future.result();f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            failures+=not row['ok'];candidates+=bool(row.get('clear'))
            print(json.dumps({'processed':index,'candidates_unverified':candidates,'errors':failures,
                              'last_error':row.get('error')},ensure_ascii=False),flush=True)
    print('FINISHED: model candidates require individual visual review before export.',flush=True)

if __name__=='__main__':main()
