"""Normal branch: network init vs SART init, quantized PSNR vs Adam steps (lr .08). Lego practice, 8 samples."""
import json, math, sys, io
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '.')
from PIL import Image
from lightfield_lab.exam_adapter import ExamPredictor, network_initial
from lightfield_lab.exam_geometry import ExamRenderer, HEIGHT, WIDTH, validate_pose
from lightfield_lab.core import refine
from lightfield_lab.tomographic import sart

dev = torch.device('cuda')
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('_ta_releases/lego100_practice_v1/testing/test_input/lego')
rows = json.load(open(root/'test.json'))['samples']
rows = rows[::max(1, len(rows)//8)][:8]
pred = ExamPredictor(None, steps=100, lr=.08, diagnostics=False)
out = {}
for row in rows:
    imgs = [np.asarray(Image.open(root/row[s]).convert('RGB')) for s in ('left', 'right')]
    poses = np.stack([validate_pose(row['pose_l']), validate_pose(row['pose_r'])])
    images = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(dev).float()/255
    r = ExamRenderer(poses, HEIGHT, WIDTH, dev, geometry_backend='torch')
    inits = {'network': network_initial(pred.model, images, torch.from_numpy(poses).to(dev), pred.screens),
             'sart30': sart(r, images, iterations=30), 'sart10': sart(r, images, iterations=10)}
    for name, init in inits.items():
        _, trace = refine(r, images, init, steps=100, lr=.08, iteration_dtype=torch.float32)
        out.setdefault(name, []).append([h['png_psnr'] for h in trace['history']])
    sizes = []
    q = torch.floor(_.clamp(0, 1)*255+.5).byte().permute(0, 2, 3, 1).cpu().numpy()
    for level in (1, 6):
        b = io.BytesIO(); Image.fromarray(q[0]).save(b, format='PNG', compress_level=level); sizes.append(len(b.getvalue())/1e6)
    print(row['id'], {k: round(v[-1][-1], 2) for k, v in out.items()}, 'png MB l1/l6', [round(s, 2) for s in sizes], flush=True)
for name, v in out.items():
    v = np.array(v)
    print(f'{name:8s} steps 0..100 by 10:', ' '.join(f'{x:5.2f}' for x in v.mean(0)))
