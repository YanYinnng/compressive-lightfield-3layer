"""Optional zero-skew pinhole intrinsics; standard TA path stays unchanged."""
import numpy as np
import torch
from torch.nn import functional as F


def standard_intrinsics(height, width):
    f = width/(2*np.tan(np.deg2rad(40.)/2))
    return np.array([f, f, width/2, height/2], dtype=np.float64)


def normalize_intrinsics(value, count, height, width):
    """Return [N,4] fx/fy/cx/cy, or None for exactly standard intrinsics.

    Accept [4], [N,4], [3,3], or [N,3,3]. General skew/distortion is unsupported.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    k = np.asarray(value, dtype=np.float64)
    if k.shape[-2:] == (3, 3):
        if not np.allclose(k[..., 2, :], [0, 0, 1], atol=1e-12) or np.any(k[..., 0, 1] != 0) or np.any(k[..., 1, 0] != 0):
            raise ValueError('Only zero-skew pinhole K is supported.')
        k = np.stack([k[..., 0, 0], k[..., 1, 1], k[..., 0, 2], k[..., 1, 2]], -1)
    if k.shape == (4,):
        k = np.broadcast_to(k, (count, 4)).copy()
    if k.shape != (count, 4) or not np.isfinite(k).all() or (k[:, :2] <= 0).any():
        raise ValueError('Require finite per-view (fx,fy,cx,cy) with positive focal lengths.')
    if np.array_equal(k, np.broadcast_to(standard_intrinsics(height, width), k.shape)):
        return None
    return k


def warp_to_standard(images, intrinsics):
    """Off-axis image -> standard-K network input; returns warped images and support.

    Forward image homography is K_std K_off^-1; grid_sample needs the inverse
    mapping from standard output pixel centres to off-axis source coordinates.
    Rays outside the supplied image are zero-padded and reported in support.
    """
    count, _, height, width = images.shape
    k = normalize_intrinsics(intrinsics, count, height, width)
    if k is None:
        return images, torch.ones(count, 1, height, width, dtype=torch.bool, device=images.device)
    k = torch.as_tensor(k, dtype=images.dtype, device=images.device)
    f, _, cx, cy = standard_intrinsics(height, width)
    y, x = torch.meshgrid(torch.arange(height, device=images.device, dtype=images.dtype)+.5,
                          torch.arange(width, device=images.device, dtype=images.dtype)+.5, indexing='ij')
    u = (x-cx)[None]*k[:, 0, None, None]/f+k[:, 2, None, None]
    v = (y-cy)[None]*k[:, 1, None, None]/f+k[:, 3, None, None]
    grid = torch.stack([2*u/width-1, 2*v/height-1], -1)
    support = ((u >= .5)&(u <= width-.5)&(v >= .5)&(v <= height-.5))[:, None]
    return F.grid_sample(images, grid, align_corners=False), support


SCREEN_TO_CAMERA = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])  # c2w columns: right=+y, up=+z, back=+x


def l2_camera(pose, height=1080, width=1920, screen_width=.596736):
    """Off-axis camera of the reference layer (generalized perspective projection).

    Keeps the eye position; the image plane is the L2 panel itself, so camera pixel
    (u, v) sees L2 pixel (u, v). Returns ([fx, fy, cx, cy], c2w with screen-aligned axes).
    """
    from .exam_geometry import validate_pose
    origin = validate_pose(pose)[:3, 3]
    if origin[0] <= 0:
        raise ValueError('The eye must be in front of L2 (x > 0).')
    scale = width/screen_width  # pixels per metre on the panel
    f = scale*origin[0]
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = SCREEN_TO_CAMERA, origin
    return np.array([f, f, width/2+scale*origin[1], height/2-scale*origin[2]]), c2w


def warp_between(images, k_source, poses_source, poses_target):
    """Resample images of source cameras (K_source, c2w) into standard-K target cameras.

    Cameras share their centres, so each target pixel ray is projected into the
    source camera (a rotation homography). Rays behind the source camera or outside
    its image are zero and reported as unsupported.
    """
    count, _, height, width = images.shape
    k = torch.as_tensor(np.asarray(k_source, dtype=np.float64), dtype=torch.float64, device=images.device)
    rs = torch.as_tensor(np.stack([np.asarray(p, dtype=np.float64)[:3, :3] for p in poses_source]), device=images.device)
    rt = torch.as_tensor(np.stack([np.asarray(p, dtype=np.float64)[:3, :3] for p in poses_target]), device=images.device)
    f, _, cx, cy = standard_intrinsics(height, width)
    y, x = torch.meshgrid(torch.arange(height, device=images.device, dtype=torch.float64)+.5,
                          torch.arange(width, device=images.device, dtype=torch.float64)+.5, indexing='ij')
    local = torch.stack([(x-cx)/f, -(y-cy)/f, -torch.ones_like(x)], -1)  # target camera, right/up/back
    relative = rs.transpose(1, 2)@rt  # target camera -> source camera
    source = torch.einsum('nij,hwj->nhwi', relative, local)
    depth = -source[..., 2]
    front = depth > 1e-9
    depth = torch.where(front, depth, torch.ones_like(depth))
    u = k[:, 0, None, None]*source[..., 0]/depth+k[:, 2, None, None]
    v = k[:, 3, None, None]-k[:, 1, None, None]*source[..., 1]/depth
    support = front&(u >= .5)&(u <= width-.5)&(v >= .5)&(v <= height-.5)
    grid = torch.stack([torch.where(front, 2*u/width-1, torch.full_like(u, 3.)),
                        torch.where(front, 2*v/height-1, torch.full_like(v, 3.))], -1).to(images.dtype)
    return F.grid_sample(images, grid, align_corners=False), support[:, None]


def screen_adaptive_intrinsics(poses, height, width, model='lens_shift'):
    """Design candidate K from layer2 projected bounds, retaining camera rotations.

    This is an image-framing design, not a claim that physical eye intrinsics can
    be inferred uniquely from display geometry. screen_fit fits a bounding box,
    which does not guarantee optimal three-layer valid coverage.
    """
    from .exam_geometry import screen_corners, validate_pose
    if model not in ('lens_shift', 'screen_fit'):
        raise ValueError('model must be lens_shift or screen_fit')
    corners = screen_corners(height=height, width=width)[1].numpy()
    result = []
    for pose in poses:
        pose = validate_pose(pose)
        camera = (np.linalg.inv(pose)@np.column_stack([corners, np.ones(4)]).T).T[:, :3]
        if (camera[:, 2] >= 0).any():
            raise ValueError('Screen corners must be in front of the camera.')
        normalized = np.stack([-camera[:, 0]/camera[:, 2], camera[:, 1]/camera[:, 2]], -1)
        lower, upper = normalized.min(0), normalized.max(0)
        f = standard_intrinsics(height, width)[0]
        if model == 'screen_fit':
            f = min(width/(upper[0]-lower[0]), height/(upper[1]-lower[1]))
        centre = (lower+upper)/2
        result.append([f, f, width/2-f*centre[0], height/2-f*centre[1]])
    return np.array(result)


def forward_render_intrinsics(patterns_rgb, poses, intrinsics):
    """Independent tiled NumPy reference; does not modify the supplied TA code."""
    from .ta_reference import load_reference
    from .exam_geometry import validate_pose
    patterns = np.asarray(patterns_rgb)
    if patterns.ndim != 4 or patterns.shape[0] != 3 or patterns.shape[-1] != 3:
        raise ValueError('Require [3,H,W,3] patterns.')
    patterns = patterns.astype(np.float64)/255 if patterns.dtype == np.uint8 else patterns.astype(np.float64)
    _, h, w, _ = patterns.shape
    poses = [validate_pose(p) for p in poses]
    k = normalize_intrinsics(intrinsics, len(poses), h, w)
    if k is None:
        k = np.broadcast_to(standard_intrinsics(h, w), (len(poses), 4))
    reference = load_reference()
    eyes = np.empty((len(poses), h, w, 3), dtype=np.float32)
    masks = np.empty((len(poses), h, w), dtype=bool)
    for eye, (pose, (fx, fy, cx, cy)) in enumerate(zip(poses, k)):
        r, o = pose[:3, :3], pose[:3, 3]
        x = (np.arange(w)+.5-cx)/fx
        for top in range(0, h, 64):
            bottom = min(h, top+64)
            y = -(np.arange(top, bottom)+.5-cy)/fy
            dx, dy, dz = [r[j, 0]*x[None]+r[j, 1]*y[:, None]-r[j, 2] for j in range(3)]
            parallel = np.abs(dx) < 1e-9
            phase = np.zeros((bottom-top, w, 3))
            valid = ~parallel
            for layer, layer_x in enumerate((.06, 0., -.06)):
                t = np.divide(layer_x-o[0], dx, out=np.zeros_like(dx), where=~parallel)
                u = (o[1]+t*dy+.596736/2)/.596736*w
                v = (.596736*h/w/2-(o[2]+t*dz))/(.596736*h/w)*h
                valid &= (t > 0)&(u >= 0)&(u <= w)&(v >= 0)&(v <= h)
                phase += reference._bilinear_rgb(patterns[layer], u, v)
            eyes[eye, top:bottom] = np.sin(phase*(np.pi/2))**2
            masks[eye, top:bottom] = valid
    return eyes, masks
