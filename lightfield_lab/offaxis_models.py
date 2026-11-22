"""Candidate off-axis eye cameras (K, c2w) built only from the eye pose and the L2 panel.

All models except l2_camera keep the input c2w (look at O2, no roll) and change K only.
K is zero-skew [fx, fy, cx, cy] in the TA convention (pixel centres at i+0.5, v down).
"Extents" are the L2 corners projected into the fixed camera: a = X/(-Z) (right), b = -Y/(-Z)
(down), so pixel u = fx*a + cx and v = fy*b + cy.

  standard          base model: square pixels, centred principal point, hFOV 40 deg
  shift_eye         focal kept, principal point moved so the L2 extents are centred (per eye)
  shift_cyclopean   shift_eye averaged over the two eyes (submitted 2026-11-07 14:27)
  bbox_eye          asymmetric frustum whose image rectangle is exactly the L2 extents:
                    fx = W/(r-l), fy = H/(b1-b0), cx = -fx*l, cy = -fy*b0 (non-square pixels)
  bbox_square_eye   square pixels, focal so that the extents fit inside, extents centred
  bbox_cyclopean    bbox_eye averaged over the two eyes
  bbox_eye_exact    bbox_eye on the exact look-at-O2, no-roll rotation rebuilt from each eye position
                    (spec 2.1); the supplied float32 poses deviate from it by <= 2.8e-6 deg (rounding)
  l2_camera         Kooima: eye centre kept, image plane = L2 panel (rotation changes)
"""
import numpy as np

from .exam_geometry import HEIGHT, WIDTH, screen_corners, validate_pose
from .intrinsics import l2_camera, standard_intrinsics

MODELS = ('standard', 'shift_eye', 'shift_cyclopean', 'bbox_eye', 'bbox_eye_exact', 'bbox_square_eye', 'bbox_cyclopean', 'l2_camera')
WORLD_UP = np.array([0., 0., 1.])


def look_at_o2(position):
    """Spec 2.1: back axis z = (P - O2)/|P - O2|, right x = k x z/|k x z| (k = world up), up y = z x x."""
    position = np.asarray(position, dtype=np.float64)
    z = position/np.linalg.norm(position)
    x = np.cross(WORLD_UP, z)
    if np.linalg.norm(x) < 1e-9:
        raise ValueError('Look-at construction degenerates when the eye is straight above O2.')
    x /= np.linalg.norm(x)
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = np.column_stack([x, np.cross(z, x), z]), position
    return c2w


def l2_extents(pose, height=HEIGHT, width=WIDTH):
    """(l, r, b0, b1): normalized extents of the projected L2 corners in the fixed camera."""
    pose = validate_pose(pose)
    corners = screen_corners(height=height, width=width)[1].numpy()
    camera = (corners-pose[:3, 3])@pose[:3, :3]  # world -> camera (right/up/back)
    if (camera[:, 2] >= 0).any():
        raise ValueError('L2 corners must be in front of the camera.')
    a, b = camera[:, 0]/-camera[:, 2], -camera[:, 1]/-camera[:, 2]
    return a.min(), a.max(), b.min(), b.max()


def eye_k(model, pose, height=HEIGHT, width=WIDTH):
    f = standard_intrinsics(height, width)[0]
    if model == 'standard':
        return standard_intrinsics(height, width)
    l, r, b0, b1 = l2_extents(pose, height, width)
    if model == 'shift_eye':
        return np.array([f, f, width/2-f*(l+r)/2, height/2-f*(b0+b1)/2])
    if model == 'bbox_eye':
        fx, fy = width/(r-l), height/(b1-b0)
        return np.array([fx, fy, -fx*l, -fy*b0])
    if model == 'bbox_square_eye':
        s = min(width/(r-l), height/(b1-b0))
        return np.array([s, s, width/2-s*(l+r)/2, height/2-s*(b0+b1)/2])
    raise ValueError(model)


def camera(model, pose_l, pose_r, height=HEIGHT, width=WIDTH):
    """Per-eye [fx, fy, cx, cy] (2x4) and c2w (2x4x4) of an off-axis camera model."""
    poses = np.stack([validate_pose(pose_l), validate_pose(pose_r)])
    if model == 'bbox_eye_exact':
        poses = np.stack([look_at_o2(p[:3, 3]) for p in poses])
        return np.stack([eye_k('bbox_eye', p, height, width) for p in poses]), poses
    if model == 'l2_camera':
        pairs = [l2_camera(p, height, width) for p in poses]
        return np.stack([k for k, _ in pairs]), np.stack([c for _, c in pairs])
    if model in ('shift_cyclopean', 'bbox_cyclopean'):
        base = np.stack([eye_k(model.replace('cyclopean', 'eye'), p, height, width) for p in poses])
        return np.broadcast_to(base.mean(0), (2, 4)).copy(), poses
    return np.stack([eye_k(model, p, height, width) for p in poses]), poses


def k_matrix(k):
    fx, fy, cx, cy = k
    return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])


def from_matrices(k_l, k_r):
    """Declared TA 3x3 matrices -> 2x4 (zero skew required)."""
    out = []
    for k in (k_l, k_r):
        k = np.asarray(k, dtype=np.float64)
        if abs(k[0, 1]) > 1e-9 or abs(k[1, 0]) > 1e-9:
            raise ValueError('Only zero-skew K is supported by this evaluator.')
        out.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    return np.array(out)
