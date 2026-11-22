"""Audit: do the poses of each benchmark dataset look exactly at O2 (no roll, IPD 0.06)?
Also how far bbox_eye on the supplied rotation is from bbox_eye_exact (rebuilt look-at), in pixels."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.offaxis_models import camera, look_at_o2  # noqa: E402


def angle(a, b):
    # Stable for tiny angles (arccos near 1 turns float32 rounding into ~0.03 deg).
    d = a.T@b
    skew = np.array([d[2, 1]-d[1, 2], d[0, 2]-d[2, 0], d[1, 0]-d[0, 1]])/2
    return np.degrees(np.arctan2(np.linalg.norm(skew), (np.trace(d)-1)/2))


def vector_angle(u, v):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), u@v))


for spec in sys.argv[1:]:
    name, path = spec.split('=', 1)
    rows = json.loads(Path(path).read_text(encoding='utf-8'))['samples']
    rot, axis, roll, ipd, dk = [], [], [], [], []
    for row in rows:
        poses = [np.array(row['pose_l'], float), np.array(row['pose_r'], float)]
        for pose in poses:
            rebuilt = look_at_o2(pose[:3, 3])
            rot.append(angle(rebuilt[:3, :3], pose[:3, :3]))
            to_eye = pose[:3, 3]/np.linalg.norm(pose[:3, 3])
            axis.append(vector_angle(pose[:3, 2], to_eye))
            roll.append(np.degrees(np.arcsin(np.clip(pose[2, 0], -1, 1))))
        ipd.append(np.linalg.norm(poses[0][:3, 3]-poses[1][:3, 3]))
        k_given, _ = camera('bbox_eye', *poses)
        k_exact, _ = camera('bbox_eye_exact', *poses)
        dk.append(np.abs(k_given-k_exact).max())
    q = lambda v: f'median {np.median(v):.3g}  p95 {np.percentile(v, 95):.3g}  max {np.max(v):.3g}'
    print(f'== {name} ({len(rows)} samples)')
    print(f'   rotation vs exact look-at (deg)  {q(rot)}')
    print(f'   back axis vs direction to O2     {q(axis)}')
    print(f'   roll (deg)                       {q(np.abs(roll))}')
    print(f'   IPD (m)                          {q(ipd)}')
    print(f'   max |K(bbox_eye) - K(exact)| px  {q(dk)}')
