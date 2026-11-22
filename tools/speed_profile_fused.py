"""Per-stage cost of one sample in the fused submission pipeline (synchronised timers), plus the layering-v5 extras.

python tools/speed_profile_fused.py --root <scene dir with test.json> [--n 24]     (run from the repo root, H100)
Mirrors V3Predictor + ExamPredictor._solve_fused + submit_v3 export; medians over samples 3..n.
"""
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '.')
from PIL import Image
from lightfield_lab import fused
from lightfield_lab.exam_adapter import ExamPredictor, network_initial
from lightfield_lab.exam_geometry import validate_pose
from lightfield_lab.v3_submission import offaxis_camera

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--n', type=int, default=24)
args = parser.parse_args()
rows = json.load(open(args.root/'test.json'))['samples'][:args.n]
pred = ExamPredictor(None, steps=100, lr=.08, solver='fused')
dev = pred.device
T = {}


def lap(name, t0, sync=True):
    if sync:
        torch.cuda.synchronize()
    now = time.perf_counter()
    T.setdefault(name, []).append(now-t0)
    return now


for row in rows:
    t = time.perf_counter()
    imgs = [np.asarray(Image.open(args.root/row[s]).convert('RGB')) for s in ('left', 'right')]
    t = lap('cpu: read+decode 2 PNG', t, False)
    pl, pr = validate_pose(row['pose_l']), validate_pose(row['pose_r'])
    poses = np.stack([pl, pr])
    images = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(dev).float()/255
    t = lap('gpu: upload images', t)
    init = network_initial(pred.model, images, torch.from_numpy(poses).to(dev), pred.screens)
    t = lap('N: network initial', t)
    fs = fused.FusedSolver(poses, images)
    t = lap('N: FusedSolver setup (host geometry, mask, count sync)', t)
    best, score, hist = fs.adam(init, steps=100, lr=.08)
    t = lap('N: Adam 100 + 11 evals', t)
    out_n = torch.floor(best.clamp(0, 1)*255+.5).byte().permute(0, 2, 3, 1).cpu().numpy()
    t = lap('N: quantise + download', t)
    k, cams = offaxis_camera(pl, pr, 'l2_camera')
    t = lap('O: Kooima camera (host)', t)
    fo = fused.FusedSolver(cams, images, k)
    t = lap('O: FusedSolver setup', t)
    g0 = fused.sart(fo, images, 30)
    t = lap('O: SART 30', t)
    best_o, _, _ = fo.adam(g0, steps=100, lr=.08)
    t = lap('O: Adam 100 + 11 evals', t)
    out_o = torch.floor(best_o.clamp(0, 1)*255+.5).byte().permute(0, 2, 3, 1).cpu().numpy()
    t = lap('O: quantise + download', t)
    for a in list(out_n)+list(out_o):
        Image.fromarray(a).save(io.BytesIO(), format='PNG', compress_level=1)
    t = lap('cpu: encode 6 PNG level 1 (serial)', t, False)
    # ---- layering v5 extras (normal branch): weights from an eye-space ownership map, weighted SART, Adam 300
    w_eye = torch.rand(2, 3, 1080, 1920, device=dev)
    t = time.perf_counter()
    fields = torch.cat([w_eye, torch.ones_like(w_eye[:, :1])], 1)
    fs._grids = None
    pulled = fs.pull_to_layers(fields)
    W = torch.stack([pulled[k, :, k].sum(0)/pulled[k, :, 3].sum(0).clamp_min(1e-6) for k in range(3)]).clamp(0, 1)
    t = lap('v5: ownership weights to layers', t)
    fs.set_layer_weights(sart=.1+.9*W)
    g1 = fused.sart(fs, images, 30)
    fs.set_layer_weights()
    t = lap('v5: weighted SART 30', t)
    fs.adam(g1, steps=300, lr=.08)
    t = lap('v5: Adam 300 + 31 evals', t)

print(f'{len(rows)} samples from {args.root}; medians over samples 3..{len(rows)} (ms)')
total = 0.
for name, v in T.items():
    m = float(np.median(v[2:]))*1000
    if not name.startswith('v5'):
        total += m
    print(f'  {name:55s} {m:8.1f}')
print(f'  {"sum of current pipeline stages (serial)":55s} {total:8.1f}')
