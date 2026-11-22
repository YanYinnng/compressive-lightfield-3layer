"""H100 cost of the EyeReal network initial (normal branch) vs fused SART-30, per sample. Run from the repo root."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '.')
from PIL import Image
from lightfield_lab.exam_adapter import ExamPredictor, network_initial
from lightfield_lab.exam_geometry import validate_pose
from lightfield_lab import fused

root = Path(sys.argv[1])
rows = json.load(open(root/'test.json'))['samples'][:12]
pred = ExamPredictor(None, steps=100, lr=.08, diagnostics=False)
dev = pred.device
t_net, t_sart = [], []
for row in rows:
    imgs = [np.asarray(Image.open(root/row[s]).convert('RGB')) for s in ('left', 'right')]
    poses = np.stack([validate_pose(row['pose_l']), validate_pose(row['pose_r'])])
    images = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(dev).float()/255
    torch.cuda.synchronize(); t = time.perf_counter()
    network_initial(pred.model, images, torch.from_numpy(poses).to(dev), pred.screens)
    torch.cuda.synchronize(); t_net.append(time.perf_counter()-t)
    t = time.perf_counter()
    fused.sart(fused.FusedSolver(poses, images), images, 30)
    torch.cuda.synchronize(); t_sart.append(time.perf_counter()-t)
print(f'[network] {np.median(t_net[2:])*1000:.1f} ms/sample; fused setup+SART-30 {np.median(t_sart[2:])*1000:.1f} ms/sample')
