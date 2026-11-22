"""TA submission v3 contract: one predictor returning normal and off-axis branches.

Off-axis branch: extrinsics (input c2w) and input images unchanged; each eye gets a
dynamic K computed from the poses and the L2 screen (lens shift that centres the
projected L2 screen, shared by both eyes by default), and the three
patterns are solved again under those rays.
"""
import numpy as np

from .exam_adapter import ExamPredictor
from .exam_geometry import HEIGHT, WIDTH
from .intrinsics import l2_camera, screen_adaptive_intrinsics

DEFAULTS = {'steps': 100, 'lr': .08, 'offaxis_model': 'l2_camera', 'offaxis_initial': 'sart', 'hard_layering': 0.}


def k_matrix(fx, fy, cx, cy):
    """TA K convention: right/down/forward, pixel centres at i+0.5."""
    return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])


OFFAXIS_MODELS = ('bbox_eye_exact', 'bbox_eye', 'l2_camera', 'bbox_square_eye', 'lens_shift', 'lens_shift_stereo', 'lens_shift_cyclopean', 'screen_fit')


def offaxis_intrinsics(pose_l, pose_r, model=DEFAULTS['offaxis_model']):
    """lens_shift_stereo: per-eye horizontal shift, but one shared cy so the
    reinterpreted pair gains no vertical disparity (binocular epipolar constraint).
    lens_shift_cyclopean: both eyes share the mean shift (convergence stays at O2)."""
    if model not in OFFAXIS_MODELS:
        raise ValueError(f'offaxis model must be one of {OFFAXIS_MODELS}')
    k = screen_adaptive_intrinsics([pose_l, pose_r], HEIGHT, WIDTH, 'screen_fit' if model == 'screen_fit' else 'lens_shift')
    if model == 'lens_shift_stereo':
        k[:, 3] = k[:, 3].mean()
    elif model == 'lens_shift_cyclopean':
        k[:, 2:] = k[:, 2:].mean(0)
    return k


def offaxis_camera(pose_l, pose_r, model=DEFAULTS['offaxis_model']):
    """(per-eye [fx, fy, cx, cy], per-eye c2w) of the declared off-axis cameras.

    l2_camera: eye centres kept, image plane = L2 panel (pixel (u,v) sees L2 pixel (u,v)),
    rotation aligned with the panel. Other models keep the input rotation and only shift K.
    """
    if model == 'l2_camera':
        cameras = [l2_camera(p, HEIGHT, WIDTH) for p in (pose_l, pose_r)]
        return np.stack([c[0] for c in cameras]), np.stack([c[1] for c in cameras])
    if model in ('bbox_eye_exact', 'bbox_eye', 'bbox_square_eye'):  # asymmetric frustum fitted to L2, look-at-O2 rotation
        from .offaxis_models import camera
        return camera(model, pose_l, pose_r)
    return offaxis_intrinsics(pose_l, pose_r, model), np.stack([np.asarray(pose_l, float), np.asarray(pose_r, float)])


class V3Predictor:
    """predict(left, right, pose_l, pose_r) -> TA v3 result dict (uint8 HWC layers)."""

    def __init__(self, weights=None, steps=DEFAULTS['steps'], lr=DEFAULTS['lr'],
                 offaxis_model=DEFAULTS['offaxis_model'], offaxis_initial=DEFAULTS['offaxis_initial'],
                 hard_layering=DEFAULTS['hard_layering'], solver='torch', **options):
        if solver == 'fused' and hard_layering:
            raise ValueError('The fused solver requires --hard-layering 0.')
        self.solver = ExamPredictor(weights, steps=steps, lr=lr, diagnostics=False, solver=solver, **options)
        self.offaxis_model = offaxis_model
        self.offaxis_initial = offaxis_initial
        self.hard_layering = hard_layering
        self.last_scores = None

    def __call__(self, left_rgb, right_rgb, pose_l, pose_r, depths=None):
        """depths: optional precomputed stereo depth for the normal branch's hard layering."""
        pose_l, pose_r = np.asarray(pose_l, dtype=np.float64), np.asarray(pose_r, dtype=np.float64)
        normal = self.solver(left_rgb, right_rgb, pose_l, pose_r, hard_layering=self.hard_layering, depths=depths)
        normal_score, normal_valid = self.solver.last_score, self.solver.last_valid_pixels
        k, cameras = offaxis_camera(pose_l, pose_r, self.offaxis_model)
        rotated = not np.allclose(cameras[:, :3, :3], np.stack([pose_l, pose_r])[:, :3, :3])
        offaxis = self.solver(left_rgb, right_rgb, pose_l, pose_r, intrinsics=k, render_poses=cameras if rotated else None,
                               initial_method=self.offaxis_initial)
        self.last_scores = {'normal': normal_score, 'normal_valid_pixels': normal_valid,
                            'offaxis': self.solver.last_score, 'offaxis_valid_pixels': self.solver.last_valid_pixels,
                            'offaxis_fx_fy_cx_cy': k.tolist(),
                            'solver_normal': self.solver.records[-2], 'solver_offaxis': self.solver.records[-1]}
        return {'patterns': normal,
                'offaxis': {'K_l': k_matrix(*k[0]), 'K_r': k_matrix(*k[1]), 'pose_l': cameras[0], 'pose_r': cameras[1],
                            'patterns': offaxis}}


def make_v3_predictor(weights=None, **options):
    return V3Predictor(weights, **options)
