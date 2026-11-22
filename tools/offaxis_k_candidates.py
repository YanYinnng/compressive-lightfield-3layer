"""Candidate off-axis intrinsics (M1a lens shift, M1b screen-fitted frustum, M2 kappa) for a few poses.

Poses stay fixed (look-at O2); only K = (fx, fy, cx, cy) changes. Prints valid-area fractions
(rays hitting all three layers) under the base K and each candidate. Exploration only.
"""
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.render_lego_gt import look_at
from lightfield_lab.gaussian_scene import stereo_poses
W, H = 1920, 1080
SW = .596736; SH = SW*H/W; LAYERS = (.06, 0., -.06)
F0 = W/(2*math.tan(math.radians(20)))
K0 = (F0, F0, W/2, H/2)

def valid_fraction(pose, K, h=270, w=480):
    fx, fy, cx, cy = K; s = w/W
    R, o = pose[:3, :3], pose[:3, 3]
    lx = ((np.arange(w)+.5)/s-cx)/fx; ly = -((np.arange(h)+.5)/s-cy)/fy
    d = [R[j, 0]*lx[None]+R[j, 1]*ly[:, None]-R[j, 2] for j in range(3)]
    valid = np.abs(d[0]) > 1e-9
    for x in LAYERS:
        t = (x-o[0])/np.where(valid, d[0], 1)
        u = (o[1]+t*d[1]+SW/2)/SW; v = (SH/2-(o[2]+t*d[2]))/SH
        valid &= (t > 0)&(u >= 0)&(u <= 1)&(v >= 0)&(v <= 1)
    return valid.mean()

def screen_bbox(pose, layer_x=0.):
    """Layer rectangle corners -> camera normalized coords (x right, y up), bbox."""
    corners = np.array([[layer_x, sy*SW/2, sz*SH/2, 1.] for sy in (-1, 1) for sz in (-1, 1)])
    cam = (np.linalg.inv(pose)@corners.T).T[:, :3]
    x, y = cam[:, 0]/-cam[:, 2], cam[:, 1]/-cam[:, 2]
    return x.min(), x.max(), y.min(), y.max()

def k_fit(pose, square=True):
    xmin, xmax, ymin, ymax = screen_bbox(pose)
    fx, fy = W/(xmax-xmin), H/(ymax-ymin)
    if square: fx = fy = min(fx, fy)
    cx = W/2-fx*(xmin+xmax)/2; cy = H/2+fy*(ymin+ymax)/2
    return fx, fy, cx, cy

rows = json.load(open(ROOT/'testing/demo_input/test.json'))['samples']
cases = [('demo_000000 L', np.array(rows[0]['pose_l'])), ('demo_000000 R', np.array(rows[0]['pose_r'])),
         ('yaw38 d0.8 L', stereo_poses(.8, 38, 0)[0]), ('yaw38 d0.8 R', stereo_poses(.8, 38, 0)[1]),
         ('yaw30 elev15 d0.7 L', stereo_poses(.7, 30, 15)[0]), ('d1.35 yaw0 L', stereo_poses(1.35, 0, 0)[0])]
print(f"{'case':22s} {'base valid':>10s} | M1 lens-shift only (f fixed): cx-960, cy-540, valid | M1 fit square: hFOV, cx-960, cy-540, valid")
for name, pose in cases:
    xmin, xmax, ymin, ymax = screen_bbox(pose)
    shift = (W/2-F0*(xmin+xmax)/2, H/2+F0*(ymin+ymax)/2)
    kf = k_fit(pose)
    print(f"{name:22s} {valid_fraction(pose, K0):10.3f} | {shift[0]-960:+7.1f} {shift[1]-540:+7.1f} {valid_fraction(pose, (F0, F0, *shift)):.3f}"
          f" | {2*math.degrees(math.atan(W/2/kf[0])):5.1f}deg {kf[2]-960:+7.1f} {kf[3]-540:+7.1f} {valid_fraction(pose, kf):.3f}")
kappa = 5
print(f'M2 kappa {kappa}deg: principal point shift {F0*math.tan(math.radians(kappa)):.0f} px (nasal, opposite sign per eye)')
