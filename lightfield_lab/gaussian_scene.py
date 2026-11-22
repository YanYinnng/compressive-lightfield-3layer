"""Supplied Graphdeco Gaussian PLY -> calibrated pinhole target images.

The scene remains in its native coordinates, including view-dependent SH.
World cameras are transformed into the native scene instead of rotating SH.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
import torch


def load_gaussian_ply(path):
    path = Path(path)
    with path.open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('Expected PLY.')
        fields, count, binary = [], None, False
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError('Incomplete PLY header.')
            line = raw.decode('ascii').strip()
            if line == 'format binary_little_endian 1.0':binary = True
            if line.startswith('element vertex '):count = int(line.split()[-1])
            if line.startswith('property '):
                parts = line.split()
                if parts[1] != 'float':raise ValueError('This loader expects scalar float32 Gaussian fields.')
                fields.append(parts[2])
            if line == 'end_header':break
        offset = stream.tell()
    if not binary or count is None:
        raise ValueError('Expected little-endian Gaussian PLY.')
    dtype = np.dtype([(field, '<f4') for field in fields])
    if path.stat().st_size != offset+count*dtype.itemsize:
        raise ValueError('Unexpected PLY payload size.')
    array = np.memmap(path, mode='r', offset=offset, dtype=dtype, shape=(count,))
    columns = lambda names: np.stack([array[name] for name in names], axis=-1)
    means = columns(['x', 'y', 'z'])
    dc = columns([f'f_dc_{i}' for i in range(3)])[:, None]
    rest_names = sorted((name for name in fields if name.startswith('f_rest_')), key=lambda name: int(name.rsplit('_', 1)[1]))
    rest = columns(rest_names).reshape(count, 3, -1).transpose(0, 2, 1)
    sh = np.concatenate([dc, rest], axis=1)
    degree = math.isqrt(sh.shape[1])-1
    if (degree+1)**2 != sh.shape[1]:raise ValueError('Invalid SH basis count.')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'means': means, 'quats': columns([f'rot_{i}' for i in range(4)]),
            'log_scales': columns([f'scale_{i}' for i in range(3)]),
            'opacity_logits': np.array(array['opacity']), 'sh': sh, 'sh_degree': degree,
            'sha256': digest}


def look_at_origin(position):
    back = np.asarray(position, dtype=np.float64)
    back = back/np.linalg.norm(back)
    right = np.cross([0., 0., 1.], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    view = np.eye(4)
    view[:3, :3] = np.stack([right, up, back], axis=1)
    view[:3, 3] = position
    return view


def stereo_poses(distance_m, yaw_deg, elevation_deg, ipd_m=.06):
    yaw, elevation = np.deg2rad([yaw_deg, elevation_deg])
    midpoint = distance_m*np.array([np.cos(elevation)*np.cos(yaw), np.cos(elevation)*np.sin(yaw), np.sin(elevation)])
    right = look_at_origin(midpoint)[:3, 0]
    return np.stack([look_at_origin(midpoint-ipd_m/2*right), look_at_origin(midpoint+ipd_m/2*right)])


def native_viewmats(poses_world, scale_m, center_native, rotation_scene_to_world=None):
    """World p = scale*A*(native p-center); preserve native Gaussian/SH axes."""
    rotation = np.eye(3) if rotation_scene_to_world is None else np.asarray(rotation_scene_to_world)
    if scale_m <= 0 or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError('Expected positive uniform scale and proper rotation.')
    result = []
    for world in np.asarray(poses_world):
        native = np.eye(4)
        native[:3, :3] = rotation.T@world[:3, :3]
        native[:3, 3] = np.asarray(center_native)+rotation.T@world[:3, 3]/scale_m
        native[:3, 1:3] *= -1  # GL right/up/back -> OpenCV right/down/forward
        result.append(np.linalg.inv(native))
    return np.stack(result)


class GaussianTargetRenderer:
    def __init__(self, path, scale_m=.12, device='cuda', placement=None):
        """placement (optional, e.g. from tools/uco3d_assets.py): {scale_m, rotation_scene_to_world,
        center_native}; default: scale_m, native Z up, centre of the Gaussian-mean bounding box."""
        from gsplat import rasterization
        self.rasterization, self.device = rasterization, device
        source = load_gaussian_ply(path)
        means = source['means']
        if placement is None:
            self.scale_m, self.rotation = scale_m, np.eye(3)
            self.center = (means.min(0)+means.max(0)).astype(np.float64)/2
            choice = 'uniform chosen scale, native Z up, center of complete Gaussian-mean bounding box; not claimed as the demo-image calibration'
        else:
            self.scale_m, self.rotation = float(placement['scale_m']), np.asarray(placement['rotation_scene_to_world'], dtype=np.float64)
            self.center = np.asarray(placement['center_native'], dtype=np.float64)
            choice = 'supplied placement'
        scale_m = self.scale_m
        self.metadata = {'source_sha256': source['sha256'], 'gaussians': len(means), 'sh_degree': source['sh_degree'],
            'scene_transform': {'scale_metres_per_native_unit': scale_m, 'rotation_scene_to_world': self.rotation.tolist(),
                'center_native': self.center.tolist(), 'translation_world_metres': (-scale_m*self.rotation@self.center).tolist(),
                'choice': choice},
            'parameter_domains': 'Graphdeco log scales -> exp once; opacity logits -> sigmoid once; quaternion wxyz normalized; SH numeric-suffix order',
            'colour': 'SH evaluated by gsplat in native view directions, white background, clamp then RGB8 rounding; no Gamma or per-image normalization'}
        tensor = lambda array: torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device=device)
        self.means = tensor(means)
        self.quats = torch.nn.functional.normalize(tensor(source['quats']), dim=-1)
        self.scales = tensor(source['log_scales']).exp()
        self.opacities = tensor(source['opacity_logits']).sigmoid()
        self.sh = tensor(source['sh'])

    @torch.no_grad()
    def render(self, poses_world, height=1080, width=1920, hfov_deg=40, intrinsics=None):
        focal = width/(2*math.tan(math.radians(hfov_deg)/2))
        matrix = torch.tensor([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]], dtype=torch.float32, device=self.device)
        from .intrinsics import normalize_intrinsics
        k = normalize_intrinsics(intrinsics, len(poses_world), height, width)
        camera_k = matrix[None].expand(len(poses_world), -1, -1).contiguous()
        if k is not None:
            camera_k = torch.zeros(len(poses_world), 3, 3, dtype=torch.float32, device=self.device)
            camera_k[:, 0, 0] = torch.as_tensor(k[:, 0], device=self.device)
            camera_k[:, 1, 1] = torch.as_tensor(k[:, 1], device=self.device)
            camera_k[:, 0, 2] = torch.as_tensor(k[:, 2], device=self.device)
            camera_k[:, 1, 2] = torch.as_tensor(k[:, 3], device=self.device)
            camera_k[:, 2, 2] = 1
        matrices = torch.tensor(native_viewmats(poses_world, self.scale_m, self.center, getattr(self, 'rotation', None)), dtype=torch.float32, device=self.device)
        colour, alpha, _ = self.rasterization(self.means, self.quats, self.scales, self.opacities, self.sh,
            matrices, camera_k, width, height,
            sh_degree=self.metadata['sh_degree'], backgrounds=None,
            near_plane=.01/self.scale_m, far_plane=10/self.scale_m, packed=True, rasterize_mode='classic')
        # gsplat 1.5.3 checks packed backgrounds against a shape without cameras.
        # Its black-background RGB is premultiplied; composite white explicitly.
        colour = colour + (1-alpha)
        images = torch.floor(colour.clamp(0, 1)*255+.5).to(torch.uint8).cpu().numpy()
        # The stereo diagnostic needs a pair; a single camera (e.g. a resumed chunk) reports None.
        difference = float(np.abs(images[0].astype(np.float32)-images[1]).mean()/255) if len(images) >= 2 else None
        return images, {'foreground_fraction_each_eye': (alpha[..., 0] > .05).float().mean((1, 2)).cpu().tolist(),
                        'left_right_mean_abs_difference': difference}
