"""Validate every generated stereo pair and produce a reproducible file index."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if (ROOT/'python_packages').is_dir():sys.path.insert(0,str(ROOT/'python_packages'))
import numpy as np
from PIL import Image
from lightfield_lab.exam_data import read_manifest, read_rgb8


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset',type=Path)
    args=parser.parse_args();path=args.dataset.resolve()
    started=time.perf_counter()
    manifest,rows=read_manifest(path/'all.json')
    config=json.loads((path/'config.json').read_text())
    done=json.loads((path/'DONE.json').read_text())
    assert done['success'] and len(rows)==done['pairs']
    files={};samples=[];eye_yaw_by_split={}
    for row in rows:
        poses=np.array([row['pose_l'],row['pose_r']])
        for pose in poses:
            local=np.linalg.inv(pose)@np.array([0,0,0,1])
            assert np.max(np.abs(local[:2]))<1e-10 and local[2]<0
        arrays=[]
        for key in ['left','right']:
            image=read_rgb8(path/row[key]);arrays.append(image)
            with (path/row[key]).open('rb') as stream:files[row[key]]=hashlib.file_digest(stream,'sha256').hexdigest()
        assert files[row['left']]!=files[row['right']],row['id']
        samples.append({'id':row['id'],'split':row['split'],'baseline_m':float(np.linalg.norm(poses[0,:3,3]-poses[1,:3,3])),
                        'left_right_mean_abs_difference':float(np.abs(arrays[0].astype(np.float32)-arrays[1]).mean()/255)})
        eye_yaw_by_split.setdefault(row['split'],[]).extend(np.rad2deg(np.arctan2(poses[:,1,3],poses[:,0,3])).tolist())
    split_counts={s:sum(row['split']==s for row in rows) for s in eye_yaw_by_split}
    for split,count in split_counts.items():
        _,part=read_manifest(path/f'{split}.json');assert len(part)==count
        assert {r['id'] for r in part}=={r['id'] for r in rows if r['split']==split}
    ranges={s:[min(values),max(values)] for s,values in eye_yaw_by_split.items()}
    gaps={}
    if 'train' in ranges:
        gaps={'train_to_val_eye_yaw_gap_deg':ranges['val'][0]-ranges['train'][1],
              'test_to_train_eye_yaw_gap_deg':ranges['train'][0]-ranges['test'][1]}
        assert min(gaps.values())>0,'Individual eye ranges overlap across splits.'
    result={'passed':True,'pairs':len(rows),'images':len(files),'split_counts':split_counts,
        'eye_yaw_ranges_deg':ranges,'split_gaps':gaps,'min_stereo_difference':min(s['left_right_mean_abs_difference'] for s in samples),
        'scene_sha256':config['renderer']['source_sha256'],'image_bytes':sum((path/p).stat().st_size for p in files),
        'seconds_audit':time.perf_counter()-started,'samples':samples,'files_sha256':files}
    (path/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['samples','files_sha256']}),flush=True)


if __name__=='__main__':main()
