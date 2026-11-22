"""Independent check of our K / c2w conventions with gsplat renders of the Lego scene.

All candidate cameras share the eye centre, so the off-axis view must equal the standard
view resampled by the ray mapping (exact up to bilinear resampling inside the standard FOV).
Compares gsplat(off-axis camera) with warp(gsplat(standard camera)) per hypothesis.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.exam_data import read_rgb8
from lightfield_lab.intrinsics import standard_intrinsics

root = Path(sys.argv[1])
device = torch.device('cuda')
std = json.loads((root/'standard/test.json').read_text())['samples']
H, W = 1080, 1920
f0, _, cx0, cy0 = standard_intrinsics(H, W)
for model in sys.argv[2:]:
    rows = json.loads((root/model/'test.json').read_text())['samples']
    diffs = []
    for s_row, h_row in list(zip(std, rows))[:int(5)]:
        for eye, side in enumerate(('left', 'right')):
            source = torch.from_numpy(read_rgb8(root/'standard'/s_row[side])).permute(2, 0, 1)[None].double().to(device)/255
            target = torch.from_numpy(read_rgb8(root/model/h_row[side])).permute(2, 0, 1).double().to(device)/255
            fx, fy, cx, cy = h_row['gt_K_fx_fy_cx_cy'][eye]
            rt = torch.tensor(np.array(h_row['gt_pose_' + side[0]])[:3, :3], device=device)
            rs = torch.tensor(np.array(s_row['pose_' + side[0]])[:3, :3], device=device)
            v, u = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float64)+.5, torch.arange(W, device=device, dtype=torch.float64)+.5, indexing='ij')
            local = torch.stack([(u-cx)/fx, -(v-cy)/fy, -torch.ones_like(u)], -1)
            src = local@(rs.T@rt).T
            depth = -src[..., 2]
            us, vs = f0*src[..., 0]/depth+cx0, cy0-f0*src[..., 1]/depth
            inside = (depth > 0)&(us > 1)&(us < W-1)&(vs > 1)&(vs < H-1)
            grid = torch.stack([2*us/W-1, 2*vs/H-1], -1)[None]
            warped = F.grid_sample(source, grid, align_corners=False)[0]
            err = (warped-target).abs().mean(0)[inside]
            diffs.append((float(err.mean())*255, float(inside.double().mean())))
    d = np.array(diffs)
    print(f'{model:16s} mean |gsplat(off-axis) - warp(gsplat(standard))| = {d[:, 0].mean():.2f}/255 over {d[:, 1].mean():.0%} of pixels', flush=True)
