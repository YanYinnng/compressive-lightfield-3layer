"""Candidate IO and RGB optical forward; prediction/GT camera construction is not provided."""
from __future__ import annotations
import argparse
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image
HEIGHT,WIDTH=1080,1920
DEFAULT_GEOMETRY={'image_h':1080,'image_w':1920,'screen_w_m':.596736,'layer_xs_m':[.06,0.,-.06],'fov_h_deg':40.,'ipd_m':.06}


def pose_array(value):
    pose=np.asarray(value,dtype=np.float64)
    if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1],atol=1e-8):
        raise ValueError('Expected finite c2w 4x4, last row [0,0,0,1]')
    if not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=2e-5) or abs(np.linalg.det(pose[:3,:3])-1)>2e-5:
        raise ValueError('c2w rotation must be orthonormal with determinant +1')
    return pose


def intrinsic_array(value):
    k=np.asarray(value,dtype=np.float64)
    if k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1],atol=1e-8):
        raise ValueError('Expected finite positive-focal K 3x3, last row [0,0,1]')
    if abs(np.linalg.det(k))<1e-12:raise ValueError('Singular intrinsics')
    return k


def fixed_intrinsics(geometry=DEFAULT_GEOMETRY):
    w,h=geometry['image_w'],geometry['image_h']
    f=w/(2*math.tan(math.radians(geometry['fov_h_deg'])/2))
    return np.array([[f,0.,w/2],[0.,f,h/2],[0.,0.,1.]])


def bilinear(image,u,v):
    h,w=image.shape[:2];x,y=u-.5,v-.5
    support=np.isfinite(x)&np.isfinite(y)&(x>-1)&(x<w)&(y>-1)&(y<h)
    x,y=np.where(support,x,0),np.where(support,y,0)
    x0,y0=np.floor(x).astype(np.int64),np.floor(y).astype(np.int64)
    ax,ay=x-x0,y-y0
    result=np.zeros(x.shape+(3,),dtype=np.float64)
    for ox,oy,weight in ((0,0,(1-ax)*(1-ay)),(1,0,ax*(1-ay)),(0,1,(1-ax)*ay),(1,1,ax*ay)):
        ix,iy=x0+ox,y0+oy
        valid=support&(ix>=0)&(ix<w)&(iy>=0)&(iy<h)
        result+=image[np.clip(iy,0,h-1),np.clip(ix,0,w-1)]*(weight*valid)[...,None]
    return result


def render_numpy(layers,cameras,geometry=DEFAULT_GEOMETRY,tile_rows=64):
    layers=np.asarray(layers)
    h,w=geometry['image_h'],geometry['image_w']
    if layers.shape!=(3,h,w,3):raise ValueError('Expected three HWC RGB patterns')
    if layers.dtype==np.uint8:layers=layers.astype(np.float64)/255
    elif np.issubdtype(layers.dtype,np.floating) and np.isfinite(layers).all() and np.all((layers>=0)&(layers<=1)):
        layers=layers.astype(np.float64,copy=False)
    else:raise ValueError('Use RGB8 or normalized floating RGB')
    sw=geometry['screen_w_m'];sh=sw*h/w
    eyes=np.empty((2,h,w,3),dtype=np.float32);masks=np.empty((2,h,w),dtype=bool)
    u=np.arange(w,dtype=np.float64)+.5
    for eye,camera in enumerate(cameras):
        k=intrinsic_array(camera['K']);pose=pose_array(camera['c2w']);inv=np.linalg.inv(k)
        r,p=pose[:3,:3],pose[:3,3]
        for top in range(0,h,tile_rows):
            bottom=min(top+tile_rows,h);v=np.arange(top,bottom,dtype=np.float64)+.5
            pixels=np.stack(np.broadcast_arrays(u[None,:],v[:,None],np.ones((bottom-top,w))),axis=-1)
            local=pixels@inv.T*np.array([1.,-1.,-1.])
            dirs=local@r.T;dx,dy,dz=np.moveaxis(dirs,-1,0)
            parallel=np.abs(dx)<1e-9
            phase=np.zeros((bottom-top,w,3));valid=np.ones((bottom-top,w),dtype=bool)
            for layer,x in enumerate(geometry['layer_xs_m']):
                t=np.divide(x-p[0],dx,out=np.zeros_like(dx),where=~parallel)
                us=(p[1]+t*dy+sw/2)/sw*w;vs=(sh/2-p[2]-t*dz)/sh*h
                valid&=(~parallel)&(t>0)&(us>=0)&(us<=w)&(vs>=0)&(vs<=h)
                phase+=bilinear(layers[layer],us,vs)
            eyes[eye,top:bottom]=np.sin((np.pi/2)*phase)**2;masks[eye,top:bottom]=valid
    return eyes,masks


def forward_render(patterns_rgb,pose_l,pose_r,K_l=None,K_r=None,tile_rows=64):
    """Return float eye images (2,H,W,3) and strict masks (2,H,W).
    Omit both K for original FOV40; provide both for offaxis.
    K uses right/down/forward; c2w uses right/up/back, view direction -z.
    No sigmoid/Gamma or prediction algorithm is applied.
    """
    layers=np.asarray(patterns_rgb)
    if layers.ndim!=4:raise ValueError('Expected (3,H,W,3)')
    geo=dict(DEFAULT_GEOMETRY,image_h=layers.shape[1],image_w=layers.shape[2])
    if (K_l is None)!=(K_r is None):raise ValueError('Provide both K matrices')
    if K_l is None:K_l=K_r=fixed_intrinsics(geo)
    return render_numpy(layers,[{'K':K_l,'c2w':pose_l},{'K':K_r,'c2w':pose_r}],geo,tile_rows)


def create_predictor(weights):
    """Implement and return predict(left_rgb,right_rgb,original_pose_l,original_pose_r).
    Inputs are HWC uint8 RGB and 4x4 c2w in meters. Return:
    {'patterns':[L1,L2,L3], 'offaxis':{'K_l':3x3,'K_r':3x3,
       'pose_l':4x4,'pose_r':4x4,'patterns':[L1,L2,L3]}}.
    Implement your own pattern method and L2 camera calculation.
    """
    raise NotImplementedError('请自行实现预测与离轴相机计算；--format-demo仅生成占位数据。')


def read_rows(path):
    path=Path(path);text=path.read_text(encoding='utf-8')
    data=[json.loads(s) for s in text.splitlines() if s.strip()] if path.suffix=='.jsonl' else json.loads(text)
    return data['samples'] if isinstance(data,dict) else data


def read_rgb(path):
    with Image.open(path) as image:
        if image.size!=(WIDTH,HEIGHT) or image.mode!='RGB':raise ValueError('Expected 1920x1080 RGB')
        return np.asarray(image).copy()


def save_layers(out,directory,number,layers):
    if len(layers)!=3:raise ValueError('Return three layers per branch')
    (out/directory).mkdir(exist_ok=True);record={}
    for k,value in enumerate(layers,1):
        image=np.asarray(value)
        if image.shape!=(HEIGHT,WIDTH,3) or image.dtype!=np.uint8:raise ValueError('Expected full-resolution uint8 RGB')
        rel=f'{directory}/sample_{number:06d}_layer{k}.png'
        Image.fromarray(image).save(out/rel);record[f'layer{k}']=rel
    return record


def render_saved(index,rows,submission,out):
    source=Path(__file__).resolve().parent
    subprocess.run([sys.executable,'-B',str(source/'check_submission.py'),'--index',str(index),'--submission',str(submission)],check=True)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose an empty output directory')
    out.mkdir(parents=True,exist_ok=True);by_id={r['id']:r for r in read_rows(submission)};outputs=[]
    for number,row in enumerate(rows):
        submitted=by_id[row['id']];record={'id':row['id']}
        for name,branch,pl,pr,kl,kr in (
            ('normal',submitted,row['pose_l'],row['pose_r'],None,None),
            ('offaxis',submitted['offaxis'],submitted['offaxis']['pose_l'],submitted['offaxis']['pose_r'],submitted['offaxis']['K_l'],submitted['offaxis']['K_r'])):
            layers=[read_rgb(submission.parent/branch[f'layer{k}']) for k in (1,2,3)]
            eyes,masks=forward_render(layers,pl,pr,kl,kr);(out/name).mkdir(exist_ok=True);record[name]={}
            for eye,side in enumerate(('left','right')):
                rel=f'{name}/sample_{number:06d}_{side}.png';mask=f'{name}/sample_{number:06d}_{side}_mask.png'
                Image.fromarray(np.floor(np.clip(eyes[eye],0,1)*255+.5).astype(np.uint8)).save(out/rel)
                Image.fromarray(masks[eye].astype(np.uint8)*255).save(out/mask)
                record[name][side]=rel;record[name]['mask_'+side]=mask
        outputs.append(record)
    (out/'eye_images.json').write_text(json.dumps({'samples':outputs},indent=2),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index',required=True);ap.add_argument('--out',required=True);ap.add_argument('--group',default='group_001')
    ap.add_argument('--weights');ap.add_argument('--format-demo',action='store_true');ap.add_argument('--submission')
    args=ap.parse_args();index=Path(args.index).resolve();out=Path(args.out).resolve();rows=read_rows(index)
    if args.submission:
        if args.format_demo or args.weights:ap.error('Use --submission as a separate self-check mode')
        render_saved(index,rows,Path(args.submission).resolve(),out);return
    if args.format_demo and args.weights:ap.error('Format example does not load weights')
    predictor=None if args.format_demo else create_predictor(args.weights)
    if out.exists() and any(out.iterdir()):ap.error('Choose a new empty --out')
    out.mkdir(parents=True,exist_ok=True);outputs=[]
    for number,row in enumerate(rows):
        left,right=read_rgb(index.parent/row['left']),read_rgb(index.parent/row['right'])
        pl,pr=pose_array(row['pose_l']),pose_array(row['pose_r'])
        if args.format_demo:
            placeholders=[np.full((HEIGHT,WIDTH,3),128,dtype=np.uint8) for _ in range(3)]
            # Zero-valued camera templates must be filled in by the candidate.
            result={'patterns':placeholders,'offaxis':{'K_l':np.zeros((3,3)),'K_r':np.zeros((3,3)),'pose_l':np.zeros((4,4)),'pose_r':np.zeros((4,4)),'patterns':placeholders}}
        else:result=predictor(left,right,pl,pr)
        off=result['offaxis'];record=dict(id=row['id'],**save_layers(out,'patterns',number,result['patterns']))
        record['offaxis']={'reference_layer':'L2',**save_layers(out,'offaxis_patterns',number,off['patterns'])}
        for key in ('K_l','K_r'):record['offaxis'][key]=(off[key] if args.format_demo else intrinsic_array(off[key])).tolist()
        for key in ('pose_l','pose_r'):record['offaxis'][key]=(off[key] if args.format_demo else pose_array(off[key])).tolist()
        outputs.append(record)
    (out/'index.json').write_text(json.dumps({'schema_version':'lightfield_submission_v3','group':args.group,'format_demo':args.format_demo,'samples':outputs},ensure_ascii=False,indent=2),encoding='utf-8')
    code=out/'code';code.mkdir();source=Path(__file__).resolve().parent
    for name in ('inference.py','check_submission.py','requirements.txt','LICENSE'):shutil.copy2(source/name,code/name)
    option=' --format-demo' if args.format_demo else ''
    if args.weights:
        target=out/'weights'/Path(args.weights).name;target.parent.mkdir();shutil.copy2(args.weights,target);option+=f' --weights weights/{target.name}'
    command=f'python code/inference.py --index /path/to/test.json --out reproduced --group {args.group}{option}'
    note='离轴K_l/K_r与pose_l/pose_r矩阵全部置0，由考生填写；填写后再校验和前向。\n\n' if args.format_demo else ''
    (code/'README.md').write_text('# 复现\n\n'+note+'~~~bash\n'+command+'\n~~~\n',encoding='utf-8')
    description='格式示范：六张pattern为占位数据。离轴K_l/K_r与pose_l/pose_r矩阵全部置0，由考生填写；填写后再校验和前向。' if args.format_demo else '请填写与自己实际实现一致的方法说明。'
    (out/'REPORT.md').write_text('# 方法说明\n\n'+description+'\n\n本轮100个Lego case只做流程测试，不计成绩。\n\n请说明自己的方法、相机计算、依赖、必要权重和复现命令。\n\n~~~bash\n'+command+'\n~~~\n',encoding='utf-8')
    demo=out/'demo';demo.mkdir();parts=[]
    for row in outputs[:12]:
        parts.append('<h2>'+html.escape(row['id'])+'</h2>')
        for label,branch in (('patterns',row),('offaxis_patterns',row['offaxis'])):
            parts.append('<p>'+label+'</p><section>'+''.join(f'<div>L{k}<img src="../{branch[f"layer{k}"]}"></div>' for k in (1,2,3))+'</section>')
    (demo/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>body{font:16px system-ui;margin:24px}section{display:flex;gap:10px}section div{width:32%}img{width:100%}</style><h1>提交格式示范</h1><p>仅占位数据，不含相机参考答案。</p>'+''.join(parts)+'</html>',encoding='utf-8')
    if args.format_demo:
        print('[template] 离轴内外参全为0占位，由考生填写；填写后再校验和前向。')
    else:
        subprocess.run([sys.executable,'-B',str(source/'check_submission.py'),'--index',str(index),'--submission',str(out/'index.json')],check=True)
    print('[submission]',out/'index.json')


if __name__=='__main__':main()
