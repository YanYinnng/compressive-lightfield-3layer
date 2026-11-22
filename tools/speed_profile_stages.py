"""Stage-by-stage timing of the submission solver (normal + Kooima off-axis) on one GPU.

Run from development_l2cam/. Times are CUDA-synchronised wall clock; use them for ratios only.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '.')
sys.path.insert(0, 'solution')
from PIL import Image
from lightfield_lab.exam_adapter import ExamPredictor, network_initial
from lightfield_lab.exam_geometry import ExamRenderer, HEIGHT, WIDTH, validate_pose
from lightfield_lab.core import image_loss, scores
from lightfield_lab.v3_submission import offaxis_camera
from lightfield_lab.tomographic import sart

dev = torch.device('cuda')
T = {}
def tic():
    torch.cuda.synchronize(); return time.perf_counter()
def toc(name, t0, n=1):
    torch.cuda.synchronize(); T.setdefault(name, []).append((time.perf_counter()-t0)/n)

root = Path('_ta_releases/lego100_practice_v1/testing/test_input/lego')
rows = json.load(open(root/'test.json'))['samples'][:6]
pred = ExamPredictor(None, steps=100, lr=.08, diagnostics=False)
import io
for i, row in enumerate(rows):
    t = tic(); left = np.asarray(Image.open(root/row['left']).convert('RGB')); right = np.asarray(Image.open(root/row['right']).convert('RGB')); toc('io: read 2 PNG (CPU)', t)
    pl, pr = validate_pose(row['pose_l']), validate_pose(row['pose_r'])
    poses = np.stack([pl, pr])
    t = tic(); images = torch.from_numpy(np.stack([left, right])).permute(0, 3, 1, 2).to(dev).float()/255; views = torch.from_numpy(poses).to(dev); toc('h2d images', t)
    # ---------------- normal branch
    t = tic(); init = network_initial(pred.model, images, views, pred.screens); toc('N network init', t)
    t = tic(); r = ExamRenderer(poses, HEIGHT, WIDTH, dev, geometry_backend='torch'); toc('N geometry cache', t)
    t = tic(); s0 = scores(r, r.quantize_patterns(init), images)['psnr_mean_eyes']; toc('N/O initial score (f64)', t)
    it = r.iteration_copy(torch.float32)
    p = init.detach().float().clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=.08)
    for step in range(1, 101):
        t = tic(); opt.zero_grad(set_to_none=True); acc = it.accumulate(p); toc('step: forward sample (grid_sample)', t)
        t = tic(); loss = image_loss(torch.sin(acc*math.pi/2).square(), images, it.geometry_mask); toc('step: sin^2 + masked loss', t)
        t = tic(); loss.backward(); toc('step: backward', t)
        t = tic(); opt.step(); toc('step: Adam', t)
        t = tic()
        with torch.no_grad(): p.clamp_(0, 1)
        toc('step: clamp', t)
        if step % 10 == 0:
            t = tic(); q = r.quantize_patterns(p); sc = scores(r, q, images)['psnr_mean_eyes']; toc('ckpt every10: quantize + f64 score + sync', t)
    t = tic(); out = torch.floor(p.detach().clamp(0, 1)*255+.5).byte().permute(0, 2, 3, 1).cpu().numpy(); toc('d2h uint8 patterns', t)
    t = tic(); scores(r, q, images.double()); r.geometry_mask.sum((1, 2, 3)).cpu().tolist(); toc('N/O final score (f64)', t)
    # ---------------- off-axis branch (Kooima + SART)
    k, cams = offaxis_camera(pl, pr, 'l2_camera')
    t = tic(); ro = ExamRenderer(cams, HEIGHT, WIDTH, dev, geometry_backend='torch', intrinsics=k); toc('O geometry cache', t)
    t = tic(); g = sart(ro, images, iterations=30); toc('O SART 30 iters', t)
    so = scores(ro, ro.quantize_patterns(g), images)['psnr_mean_eyes']
    # ---------------- PNG encode
    for level in (6, 1):
        t = time.perf_counter(); buf = io.BytesIO(); Image.fromarray(out[0]).save(buf, format='PNG', compress_level=level)
        T.setdefault(f'io: PNG encode one layer level {level} (CPU)', []).append(time.perf_counter()-t)
    import cv2
    t = time.perf_counter(); cv2.imencode('.png', out[0], [cv2.IMWRITE_PNG_COMPRESSION, 1]); T.setdefault('io: cv2 PNG encode one layer level 1 (CPU)', []).append(time.perf_counter()-t)
    print(i, f'init {s0:.2f} -> {sc:.2f} dB; sart init {so:.2f}', flush=True)

res = {k: float(np.median(v[1:] if len(v) > 3 else v)) for k, v in T.items()}  # drop warm-up
for k, v in res.items():
    print(f'{k:48s} {v*1000:9.2f} ms')
json.dump(res, open(Path('runs/speed_20261108/profile_4060.json'), 'w'), indent=1)

# kernel-level view of one Adam step
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        loss = image_loss(torch.sin(it.accumulate(p)*math.pi/2).square(), images, it.geometry_mask)
        loss.backward(); opt.step()
        with torch.no_grad(): p.clamp_(0, 1)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=18))
