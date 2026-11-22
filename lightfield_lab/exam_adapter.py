"""EyeReal initialization and per-case refinement under the TA contract."""
from pathlib import Path
import time

import numpy as np
import torch

from .core import ROOT, make_model, refine, scores, synchronize
from .exam_geometry import (HEIGHT, WIDTH, NETWORK_EYE_ORDER, NETWORK_LAYER_ORDER, ExamRenderer,
                            screen_corners, validate_pose)


def network_initial(model, images, views_m, screens_front_to_back, intrinsics=None):
    """TA [left,right] and L1/L2/L3 -> checkpoint [right,left], back-to-front.

    Images stay paired with their poses; only the network input order changes.
    The output is returned in TA front-to-back order.
    """
    if intrinsics is not None:
        from .intrinsics import warp_to_standard
        images, _ = warp_to_standard(images, intrinsics)
    layers = torch.tensor(NETWORK_LAYER_ORDER, device=images.device)
    eyes = torch.tensor(NETWORK_EYE_ORDER, device=images.device)
    with torch.no_grad():
        output = model(images[eyes][None], views_m[eyes][None], screens_front_to_back[layers][None])[0]
    return output[layers].clamp(0, 1).detach()


class ExamPredictor:
    # 'legacy' is the original baseline: float32 network-output parameters with float64
    # sampling. 'float32'/'float64' make parameters, sampling and loss all that precision.
    ITERATION_MODES = {'legacy': None, 'float32': torch.float32, 'float64': torch.float64}

    def __init__(self, weights=None, steps=30, lr=.02, phase_steps=0, anchor_weight=0., device=None, iteration_dtype="float32", geometry_backend="torch", diagnostics=None, quantized_tail=False, solver="torch",
                 layering=None, layering_eps=.3):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.steps, self.lr, self.phase_steps, self.anchor_weight = steps, lr, phase_steps, anchor_weight
        if iteration_dtype not in self.ITERATION_MODES:
            raise ValueError('iteration_dtype must be legacy, float32 or float64.')
        self.iteration_dtype = iteration_dtype
        self.geometry_backend = geometry_backend
        if solver not in ('torch', 'fused'):
            raise ValueError('solver must be torch or fused')
        self.solver_backend = solver
        self.diagnostics = solver == 'torch' if diagnostics is None else diagnostics
        if solver == 'fused':
            if self.device.type != 'cuda':
                raise ValueError('The fused solver requires CUDA.')
            if self.diagnostics or iteration_dtype != 'float32' or phase_steps or anchor_weight or quantized_tail:
                raise ValueError('The fused solver supports float32 intensity refinement without diagnostics, anchors or quantized tail.')
        self.quantized_tail = quantized_tail
        if quantized_tail and steps < 30:
            raise ValueError('quantized_tail requires at least 30 steps.')
        if steps < 0 or not 0 <= phase_steps <= steps or lr <= 0 or anchor_weight < 0:
            raise ValueError('Invalid solver parameters.')
        defaults = [ROOT/'weights/pretrained_model.pth', ROOT.parent/'weights/pretrained_model.pth']
        self.weights_path = Path(weights).resolve() if weights else next((p for p in defaults if p.is_file()), defaults[0])
        if not self.weights_path.is_file():
            raise FileNotFoundError(f'Pretrained weights missing: {self.weights_path}')
        torch.set_num_threads(4)
        started = time.perf_counter()
        self.model = make_model(HEIGHT, WIDTH, self.device, self.weights_path)
        self.screens = screen_corners(self.device)
        if solver == 'fused':
            from . import fused
            fused.backend()
        if layering not in (None, 'none', 'mono'):
            raise ValueError('layering must be none or mono')
        # Layering v5 (normal branch, fused solver): relative-depth ownership from Depth Anything V2 Small and an
        # ownership-weighted SART initial value instead of the network (docs/分层v5_20261108.md).
        self.layering_eps = layering_eps
        self.mono = None
        if layering == 'mono':
            if solver != 'fused':
                raise ValueError('layering mono requires the fused solver')
            from .mono_depth import MonoDepth
            self.mono = MonoDepth(device=self.device)
        synchronize(self.device)
        self.seconds_model_load = time.perf_counter()-started
        self.records = []
        self.deadline, self.remaining = None, 0
        self.last_eyes = self.last_masks = None
        self.export_guess, self.export_measured, self.final_seconds, self.setup_seconds = 3., None, 15., 0.

    def configure_run(self, deadline, sample_count, export_seconds=3., final_seconds=15.):
        """export_seconds: per-sample guess for work outside __call__ until measured;
        final_seconds: once-only packaging reserve after the last sample."""
        if self.solver_backend == 'fused' and deadline is not None:
            raise ValueError('The fused solver requires --time-limit 0.')
        self.deadline, self.remaining = deadline, sample_count
        self.export_guess, self.export_measured, self.final_seconds = export_seconds, None, final_seconds

    def note_export(self, seconds):
        """Measured per-sample work outside __call__ (PNG, scoring, previews)."""
        self.export_measured = max(self.export_measured or 0., seconds)

    def _case_deadline(self):
        # Reserve measured export and setup costs for the remaining samples, then
        # share what is left equally. Soft allocation, not a hard OS watchdog.
        if self.deadline is None:
            return None
        remaining = max(self.remaining, 1)
        export = self.export_guess if self.export_measured is None else self.export_measured
        left = (self.deadline-time.perf_counter()-self.final_seconds-1.25*remaining*export
                -1.25*(remaining-1)*self.setup_seconds)
        return time.perf_counter()+max(0., left/remaining)

    def __call__(self, left_rgb, right_rgb, pose_l, pose_r, intrinsics=None, render_poses=None, initial_method='network',
                 hard_layering=None, depths=None):
        """intrinsics: optional per-eye K (off-axis branch); None keeps the standard FOV40 path.
        render_poses: off-axis c2w with other rotations (same eye centres). The network then sees
        the images resampled into the input (standard-K) cameras; patterns are refined on the
        declared cameras. initial_method 'sart': training-free linearised tomography on the
        declared cameras instead of the network (lightfield_lab.tomographic).
        hard_layering: margin of the hard three-layer contrast constraint (standard branch only;
        depths: per-eye stereo depth, estimated here when not supplied). None/0 disables it."""
        started = time.perf_counter()
        arrays = [np.asarray(left_rgb), np.asarray(right_rgb)]
        if any(x.shape != (HEIGHT, WIDTH, 3) or x.dtype != np.uint8 for x in arrays):
            raise ValueError('Predictor requires two HWC 1080p uint8 RGB arrays.')
        poses = np.stack([validate_pose(pose_l), validate_pose(pose_r)])
        images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(self.device).float()/255
        if initial_method not in ('network', 'sart'):
            raise ValueError('initial_method must be network or sart')
        if initial_method == 'sart':
            initial = None
            if render_poses is not None:
                poses = np.stack([validate_pose(p) for p in render_poses])
        elif render_poses is None and intrinsics is None and self.mono is not None:
            initial = 'ownership'  # weighted SART in _solve_fused; the network is not used
        elif render_poses is None:
            views = torch.from_numpy(poses).to(self.device)
            initial = network_initial(self.model, images, views, self.screens, intrinsics=intrinsics)
        else:
            from .intrinsics import warp_between
            views = torch.from_numpy(poses).to(self.device)
            standard_view, _ = warp_between(images, intrinsics, render_poses, poses)
            initial = network_initial(self.model, standard_view, views, self.screens)
            poses = np.stack([validate_pose(p) for p in render_poses])
        if self.solver_backend == 'fused':
            if hard_layering:
                raise ValueError('The fused solver does not support hard layering.')
            return self._solve_fused(poses, images, intrinsics, initial, started)
        synchronize(self.device)
        seconds_network = time.perf_counter()-started
        began = time.perf_counter()
        renderer = ExamRenderer(poses, HEIGHT, WIDTH, self.device, geometry_backend=self.geometry_backend, intrinsics=intrinsics)
        synchronize(self.device)
        seconds_geometry = time.perf_counter()-began
        if initial is None:
            from .tomographic import sart
            initial = sart(renderer, images, iterations=30)
        initial_score = scores(renderer, renderer.quantize_patterns(initial), images)['psnr_mean_eyes']
        seconds_setup = time.perf_counter()-started
        # A valid network result is retained if the refinement budget is short.
        result, trace = refine(renderer, images, initial, steps=self.steps, lr=self.lr,
                               phase_steps=self.phase_steps, anchor_weight=self.anchor_weight,
                               deadline=self._case_deadline(), iteration_dtype=self.ITERATION_MODES[self.iteration_dtype],
                               quantize_schedule=[(self.steps-29, 0), (self.steps-14, 1)] if self.quantized_tail else ())
        layering = None
        if hard_layering and intrinsics is None and render_poses is None:
            from . import hard_layering as hard
            try:
                if depths is None:
                    depths = hard.estimate_depth(arrays[0], arrays[1], poses)
                result, layering = hard.apply(renderer, images, result, depths, poses, margin=hard_layering)
            except Exception as error:  # the refined result stands on any failure
                layering = {'applied': False, 'error': repr(error)}
        refined = time.perf_counter()
        output = torch.floor(result.clamp(0, 1)*255+.5).byte().permute(0, 2, 3, 1).cpu().numpy()
        with torch.no_grad():
            # Same RGB8 patterns through the cached TA-parity renderer (verified
            # against the NumPy reference to <1e-7); used for fast self-scoring.
            if self.diagnostics:
                self.last_eyes = renderer(result).permute(0, 2, 3, 1).cpu().numpy()
                self.last_masks = renderer.geometry_mask[:, 0].bool().cpu().numpy()
            else:
                self.last_score = scores(renderer, result, images.double())
                self.last_valid_pixels = renderer.geometry_mask.sum((1, 2, 3)).cpu().tolist()
        # Setup before and export after refinement recur for every later sample.
        self.setup_seconds = max(self.setup_seconds, seconds_setup+time.perf_counter()-refined)
        self.records.append({'network_initial_psnr_cached_dB': initial_score, 'seconds_network_and_input': seconds_network,
            'seconds_geometry_cache': seconds_geometry, 'offaxis_intrinsics': intrinsics is not None, 'seconds_predictor_total': time.perf_counter()-started,
            'refinement': trace, 'hard_layering': layering})
        self.remaining = max(0, self.remaining-1)
        return list(output)

    def _solve_fused(self, poses, images, intrinsics, initial, started):
        from . import fused

        prepared = time.perf_counter()
        solver = fused.FusedSolver(poses, images, intrinsics)
        if not (solver.count > 0).all():
            raise ValueError('At least one eye has no valid three-panel intersection.')
        if isinstance(initial, str):  # 'ownership': layering v5
            from .mono_depth import ownership
            weights = ownership(self.mono, solver, images)
            solver.set_layer_weights(sart=self.layering_eps+(1-self.layering_eps)*weights)
            initial = fused.sart(solver, images, iterations=30)
            solver.set_layer_weights()
        elif initial is None:
            initial = fused.sart(solver, images, iterations=30)
        refining = time.perf_counter()
        result, best_score, history = solver.adam(initial, steps=self.steps, lr=self.lr)
        elapsed = time.perf_counter()-refining
        checkpoints = [0, *range(10, self.steps, 10)]
        if self.steps:
            checkpoints.append(self.steps)
        trace = {'steps_run': self.steps, 'steps_requested': self.steps,
                 'selected_step': checkpoints[history.index(best_score)], 'stopped_for_time': False,
                 'iteration_dtype': str(torch.float32), 'seconds_refine_and_checkpoint_scoring': elapsed,
                 'history': [{'step': step, 'png_psnr': score} for step, score in zip(checkpoints, history)]}
        self.last_score = {'psnr_mean_eyes': best_score}
        self.last_valid_pixels = solver.count.tolist()
        output = torch.floor(result.clamp(0, 1)*255+.5).byte().permute(0, 2, 3, 1).cpu().numpy()
        self.records.append({'network_initial_psnr_cached_dB': history[0],
            'seconds_network_and_input': prepared-started, 'seconds_geometry_cache': 0.,
            'seconds_fused_setup_and_sart': refining-prepared, 'offaxis_intrinsics': intrinsics is not None,
            'seconds_predictor_total': time.perf_counter()-started, 'refinement': trace, 'hard_layering': None})
        self.remaining = max(0, self.remaining-1)
        return list(output)
