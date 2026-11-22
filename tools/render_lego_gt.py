"""Render 3D ground truth from the supplied Lego Gaussian PLY for depth/off-axis evaluation.

No training is performed. Output goes to a new directory; lego_v1 is a read-only input.

Sets (all poses are TA c2w metres: +x toward viewer, +y right, +z up, O2 origin):
  eyebox  - selected lego_v1 eye pairs; per eye a sub-aperture grid that keeps the eye's
            rotation and shifts its centre in the camera right/up plane (refocus = image
            shift), plus look-at neighbour rings and a baseline sweep.
  offaxis - gaze offset, head roll, screen-parallel gaze, wide angle and near/far pairs;
            each row also stores the base-model look-at poses from the same eye centres.
  demo    - TA demo poses rendered with this calibration, compared with the demo targets.

The scene transform must match lego_v1 (checked against its config.json).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT/'python_packages').is_dir():
    sys.path.insert(0, str(ROOT/'python_packages'))
# Same JIT extension settings as collect_lego.py, so the cached gsplat build is reused.
os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT/'cache/torch_extensions'))
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0')
os.environ.setdefault('MAX_JOBS', '4')

import numpy as np

from lightfield_lab.exam_geometry import HEIGHT, WIDTH, HFOV_DEG, IPD_M, LAYER_X_M, SCREEN_WIDTH_M, validate_pose
from lightfield_lab.gaussian_scene import look_at_origin, stereo_poses

SUBAPERTURE_OUTER = (11, 2.)  # 11x11 grid, 2 mm spacing: +-10 mm (EyeReal focal experiment scale)
SUBAPERTURE_INNER = (5, 1.)   # 5x5 grid, 1 mm spacing: +-2 mm (pupil scale)
RING_RADII_MM, RING_ANGLES = (10., 20., 40.), 8
BASELINE_SWEEP_CM = (-3., -2., -1., 1., 2., 3., 4., 5., 7., 8., 9.)  # 0 = left eye, 6 = right eye


def look_at(position, target=(0., 0., 0.)):
    """No-roll c2w (right, up, back) looking from position at target."""
    position = np.asarray(position, dtype=np.float64)
    back = position-np.asarray(target, dtype=np.float64)
    back /= np.linalg.norm(back)
    right = np.cross([0., 0., 1.], back)
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, np.cross(back, right), back], axis=1)
    pose[:3, 3] = position
    return pose


def rolled(pose, roll_deg):
    """Rotate a camera about its own optical axis (head roll)."""
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    result = pose.copy()
    right, up = pose[:3, 0], pose[:3, 1]
    result[:3, 0], result[:3, 1] = c*right+s*up, -s*right+c*up
    return result


def shifted(pose, right_mm, up_mm):
    result = pose.copy()
    result[:3, 3] += (right_mm*pose[:3, 0]+up_mm*pose[:3, 1])/1000
    return result


def subaperture_offsets_mm():
    offsets = set()
    for count, spacing in (SUBAPERTURE_OUTER, SUBAPERTURE_INNER):
        half = (count-1)//2
        offsets.update((spacing*i, spacing*j) for i in range(-half, half+1) for j in range(-half, half+1))
    return sorted(offsets, key=lambda o: (round(math.hypot(*o), 6), round(math.atan2(o[1], o[0]), 6)))


def valid_fraction(pose, height=135, width=240):
    """TA hard-mask rule at reduced resolution (fraction of pixels hitting all three panels)."""
    rotation, origin = pose[:3, :3], pose[:3, 3]
    focal = width/(2*np.tan(np.deg2rad(HFOV_DEG)/2))
    x = (np.arange(width)+.5-width/2)/focal
    y = -(np.arange(height)+.5-height/2)/focal
    dx, dy, dz = [rotation[j, 0]*x[None, :]+rotation[j, 1]*y[:, None]-rotation[j, 2] for j in range(3)]
    screen_height = SCREEN_WIDTH_M*height/width
    parallel = np.abs(dx) < 1e-9
    valid = ~parallel
    for layer_x in LAYER_X_M:
        t = np.divide(layer_x-origin[0], dx, out=np.zeros_like(dx), where=~parallel)
        u = (origin[1]+t*dy+SCREEN_WIDTH_M/2)/SCREEN_WIDTH_M*width
        v = (screen_height/2-(origin[2]+t*dz))/screen_height*height
        valid &= (t > 0)&(u >= 0)&(u <= width)&(v >= 0)&(v <= height)
    return float(valid.mean())


def select_eyebox_rows(rows, per_split):
    chosen = []
    for split, count in per_split.items():
        part = [r for r in rows if r['split'] == split]
        if count and part:
            indices = np.unique(np.linspace(0, len(part)-1, count).round().astype(int))
            chosen.extend(part[i] for i in indices)
    return chosen


def eyebox_views(row, offsets):
    """List of (relative stem, pose, kind, params) for one eyebox sample."""
    views = []
    poses = {'left': np.array(row['pose_l']), 'right': np.array(row['pose_r'])}
    for eye, pose in poses.items():
        for a, b in offsets:
            views.append((f'sub_{eye}_r{a:+05.1f}_u{b:+05.1f}', shifted(pose, a, b), 'subaperture',
                          {'eye': eye, 'offset_mm': [a, b], 'rotation': 'same as eye camera'}))
        for radius in RING_RADII_MM:
            for k in range(RING_ANGLES):
                angle = 360*k/RING_ANGLES
                a, b = radius*math.cos(math.radians(angle)), radius*math.sin(math.radians(angle))
                views.append((f'ring_{eye}_{radius:03.0f}mm_{angle:03.0f}deg', look_at(shifted(pose, a, b)[:3, 3]), 'ring',
                              {'eye': eye, 'radius_mm': radius, 'angle_deg': angle, 'rotation': 'look-at O2, no roll'}))
    left, right = poses['left'][:3, 3], poses['right'][:3, 3]
    direction = (right-left)/np.linalg.norm(right-left)
    for s in BASELINE_SWEEP_CM:
        views.append((f'baseline_{s:+04.0f}cm', look_at(left+s/100*direction), 'baseline',
                      {'cm_from_left_toward_right': s, 'rotation': 'look-at O2, no roll'}))
    return views


def offaxis_design():
    """Deterministic off-axis pairs. Each item: (category, params, pose_l, pose_r)."""
    items = []
    def pair_from_midpoint(midpoint, eye_pose_fn):
        head = look_at(midpoint)
        left, right = midpoint-IPD_M/2*head[:3, 0], midpoint+IPD_M/2*head[:3, 0]
        return eye_pose_fn(left), eye_pose_fn(right)
    def midpoint(distance, yaw, elevation):
        y, e = math.radians(yaw), math.radians(elevation)
        return distance*np.array([math.cos(e)*math.cos(y), math.cos(e)*math.sin(y), math.sin(e)])
    # Gaze offset: both eyes fixate a screen-plane point other than O2.
    for yaw in (0., 20.):
        for target in [(0., .1, 0.), (0., -.1, 0.), (0., .2, 0.), (0., 0., .08), (0., 0., -.08), (0., .15, .06)]:
            m = midpoint(.95, yaw, 0.)
            items.append(('gaze_offset', {'midpoint': {'distance_m': .95, 'yaw_deg': yaw, 'elevation_deg': 0.},
                                          'fixation_point_m': list(target)},
                          *pair_from_midpoint(m, lambda p, t=target: look_at(p, t))))
    # Screen-parallel gaze: each eye looks along -x (asymmetric frustum w.r.t. the screen).
    for lateral, height in [(0., 0.), (.12, 0.), (-.12, 0.), (.08, .05), (-.08, -.05)]:
        m = np.array([.95, lateral, height])
        def forward_pose(p):
            pose = np.eye(4); pose[:3, :3] = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]; pose[:3, 3] = p
            return pose
        left, right = m-np.array([0., IPD_M/2, 0.]), m+np.array([0., IPD_M/2, 0.])
        items.append(('screen_parallel_gaze', {'midpoint_m': m.tolist()}, forward_pose(left), forward_pose(right)))
    # Head roll: baseline tilted and each camera rolled about its own optical axis.
    for yaw in (0., 25.):
        for roll in (-20., -10., -5., 5., 10., 20.):
            m = midpoint(.95, yaw, 0.)
            head = look_at(m)
            c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
            axis = c*head[:3, 0]+s*head[:3, 1]
            left, right = m-IPD_M/2*axis, m+IPD_M/2*axis
            items.append(('head_roll', {'midpoint': {'distance_m': .95, 'yaw_deg': yaw}, 'roll_deg': roll},
                          rolled(look_at(left), roll), rolled(look_at(right), roll)))
    # Wide viewing angles beyond the lego_v1 envelope.
    for yaw, elevation, distance in [(40., 0., 1.), (-40., 0., 1.), (45., 0., 1.), (-45., 0., 1.), (50., 0., 1.2),
                                     (-50., 0., 1.2), (0., 15., 1.), (0., -15., 1.), (0., 20., 1.), (30., 15., .8),
                                     (-30., -15., .8), (35., 10., 1.4)]:
        left, right = stereo_poses(distance, yaw, elevation)
        items.append(('wide_angle', {'distance_m': distance, 'yaw_deg': yaw, 'elevation_deg': elevation}, left, right))
    # Near and far viewing distances.
    for distance, yaw in [(.5, 0.), (.55, 10.), (.6, -15.), (1.6, 0.), (1.8, 10.), (2., -10.)]:
        left, right = stereo_poses(distance, yaw, 0.)
        items.append(('near_far', {'distance_m': distance, 'yaw_deg': yaw, 'elevation_deg': 0.}, left, right))
    return items


# Arguments that change how files are produced, not what they contain.
RESUME_NEUTRAL_ARGUMENTS = ('chunk', 'dry_run')


def check_resume_config(path, config):
    """Refuse to mix outputs of different designs in one directory.

    A resume must reproduce the stored configuration apart from timestamps and
    RESUME_NEUTRAL_ARGUMENTS; the original creation time is kept and resumes are listed.
    """
    def content(value):
        value = json.loads(json.dumps(value))
        for key in ('created_utc', 'resumed_utc'):
            value.pop(key, None)
        for key in RESUME_NEUTRAL_ARGUMENTS:
            value.get('arguments', {}).pop(key, None)
        return value
    if not path.exists():
        return config
    existing = json.loads(path.read_text(encoding='utf-8'))
    if content(existing) != content(config):
        old, new = content(existing), content(config)
        changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        raise SystemExit(f'{path} was written by a different configuration (differs in: {changed}). '
                         'Use a new --out, or restore the original arguments to resume.')
    return {**config, 'created_utc': existing['created_utc'],
            'resumed_utc': existing.get('resumed_utc', [])+[config['created_utc']]}


def write_png(path, array, level):
    from PIL import Image
    temporary = path.with_suffix('.png.part')
    Image.fromarray(array).save(temporary, format='PNG', compress_level=level)
    temporary.replace(path)


class Job:
    def __init__(self, args):
        self.args, self.out = args, args.out
        self.renderer = None
        self.pool = ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2)-1)))
        self.pending = []
        self.rendered = self.skipped = 0

    def ensure_renderer(self):
        if self.renderer is None:
            from lightfield_lab.gaussian_scene import GaussianTargetRenderer
            self.renderer = GaussianTargetRenderer(self.args.asset, self.args.scale)
        return self.renderer

    def images(self, items):
        """items: list of (relative path, pose); renders only missing files, in camera chunks."""
        todo = [(rel, pose) for rel, pose in items if not (self.out/rel).is_file()]
        self.skipped += len(items)-len(todo)
        for start in range(0, len(todo), self.args.chunk):
            chunk = todo[start:start+self.args.chunk]
            images, _ = self.ensure_renderer().render(np.stack([p for _, p in chunk]))
            for (rel, _), image in zip(chunk, images):
                (self.out/rel).parent.mkdir(parents=True, exist_ok=True)
                self.pending.append(self.pool.submit(write_png, self.out/rel, image, self.args.png_level))
            self.rendered += len(chunk)
        self.flush()

    def depths(self, items):
        """Expected depth along the camera axis (metres, float16) and alpha (uint8) for each pose."""
        import torch
        from lightfield_lab.gaussian_scene import native_viewmats
        renderer = self.ensure_renderer()
        todo = [(rel, pose) for rel, pose in items if not (self.out/rel).is_file()]
        for start in range(0, len(todo), self.args.chunk):
            chunk = todo[start:start+self.args.chunk]
            focal = WIDTH/(2*math.tan(math.radians(HFOV_DEG)/2))
            k = torch.tensor([[focal, 0, WIDTH/2], [0, focal, HEIGHT/2], [0, 0, 1]], dtype=torch.float32, device=renderer.device)
            mats = torch.tensor(native_viewmats(np.stack([p for _, p in chunk]), renderer.scale_m, renderer.center),
                                dtype=torch.float32, device=renderer.device)
            with torch.no_grad():
                depth, alpha, _ = renderer.rasterization(renderer.means, renderer.quats, renderer.scales, renderer.opacities,
                    renderer.sh, mats, k[None].expand(len(mats), -1, -1).contiguous(), WIDTH, HEIGHT,
                    sh_degree=renderer.metadata['sh_degree'], backgrounds=None, near_plane=.01/renderer.scale_m,
                    far_plane=10/renderer.scale_m, packed=True, rasterize_mode='classic', render_mode='ED')
            depth = (depth[..., 0]*renderer.scale_m).cpu().numpy().astype(np.float16)
            alpha = torch.floor(alpha[..., 0].clamp(0, 1)*255+.5).to(torch.uint8).cpu().numpy()
            for (rel, _), d, a in zip(chunk, depth, alpha):
                path = self.out/rel
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name+'.part')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, depth_m=d, alpha_u8=a)
                temporary.replace(path)

    def flush(self):
        for future in self.pending:
            future.result()
        self.pending = []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT/'datasets/lego_v1')
    parser.add_argument('--asset', type=Path, default=ROOT/'assets/lego_bulldozer.ply')
    parser.add_argument('--scale', type=float, default=.12)
    parser.add_argument('--sets', nargs='+', default=['demo', 'offaxis', 'eyebox'], choices=['demo', 'offaxis', 'eyebox'])
    parser.add_argument('--eyebox-train', type=int, default=8)
    parser.add_argument('--eyebox-val', type=int, default=4)
    parser.add_argument('--eyebox-test', type=int, default=4)
    parser.add_argument('--limit-offsets', type=int, default=None, help='Pilot only: first N sub-aperture offsets')
    parser.add_argument('--limit-offaxis', type=int, default=None, help='Pilot only: first N off-axis pairs')
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--png-level', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true', help='Write design manifests only; no rendering')
    args = parser.parse_args()
    args.out = args.out.resolve()
    started = time.perf_counter()
    source_config = json.loads((args.source/'config.json').read_text())
    transform = source_config['renderer']['scene_transform']
    if not math.isclose(transform['scale_metres_per_native_unit'], args.scale):
        raise ValueError('Scene scale differs from lego_v1; targets would not be consistent.')
    args.out.mkdir(parents=True, exist_ok=True)
    offsets = subaperture_offsets_mm()[:args.limit_offsets]
    config = {'created_utc': datetime.now(timezone.utc).isoformat(), 'training_performed': False,
              'source_dataset': str(args.source.relative_to(ROOT)) if args.source.is_relative_to(ROOT) else str(args.source),
              'scene_transform': transform, 'scale_m': args.scale, 'image': {'width': WIDTH, 'height': HEIGHT,
              'horizontal_fov_degrees': HFOV_DEG, 'principal_point': [WIDTH/2, HEIGHT/2]},
              'pose_convention': 'TA c2w metres; columns right/up/back; camera looks along local -z',
              'subaperture': {'outer_grid': SUBAPERTURE_OUTER, 'inner_grid': SUBAPERTURE_INNER, 'count_per_eye': len(offsets),
                              'rule': 'same rotation as the eye; centre shifted by a*right+b*up (mm). For a focal plane '
                                      'perpendicular to the optical axis at axial distance d, a point maps to pixel shift '
                                      '(-f*a/d, +f*b/d) relative to the centre view (f = W/(2 tan 20deg)).'},
              'rings': {'radii_mm': RING_RADII_MM, 'angles': RING_ANGLES, 'rotation': 'look-at O2, no roll'},
              'baseline_sweep_cm': BASELINE_SWEEP_CM, 'depth_files': 'npz depth_m (float16 expected depth along camera axis, '
              'metres; meaningless where alpha_u8==0) and alpha_u8', 'png_compress_level': args.png_level,
              'sets': args.sets, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    config = check_resume_config(args.out/'config.json', config)
    (args.out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    job = Job(args)
    progress = args.out/'progress.json'
    def report(stage, **extra):
        progress.write_text(json.dumps({'stage': stage, 'rendered_images': job.rendered, 'skipped_existing': job.skipped,
                                        'seconds': time.perf_counter()-started, **extra}, indent=2))
        print(json.dumps({'stage': stage, 'rendered': job.rendered, 'seconds': round(time.perf_counter()-started, 1), **extra}), flush=True)

    if 'demo' in args.sets:
        from PIL import Image
        demo_rows = json.loads((ROOT/'testing/demo_input/test.json').read_text())['samples']
        records = []
        for row in demo_rows:
            items = [(f'demo/{row["id"]}_{eye}.png', validate_pose(row[key])) for eye, key in [('L', 'pose_l'), ('R', 'pose_r')]]
            if not args.dry_run:
                job.images(items)
            record = {'id': row['id']}
            if not args.dry_run:
                for (rel, _), key in zip(items, ['left', 'right']):
                    ours = np.asarray(Image.open(args.out/rel)).astype(np.float64)/255
                    theirs = np.asarray(Image.open(ROOT/'testing/demo_input'/row[key]).convert('RGB')).astype(np.float64)/255
                    mse = float(((ours-theirs)**2).mean())
                    record[key] = {'render': rel, 'psnr_vs_demo_target_dB': -10*math.log10(max(mse, 1e-12)),
                                   'mean_abs_difference': float(np.abs(ours-theirs).mean())}
            records.append(record)
        (args.out/'demo_check.json').write_text(json.dumps({'note': 'High PSNR would mean the TA demo used the same '
            'scene calibration; low PSNR means demo targets need their own calibration.', 'samples': records}, indent=2))
        report('demo')

    if 'offaxis' in args.sets:
        rows = []
        for number, (category, params, left, right) in enumerate(offaxis_design()[:args.limit_offaxis]):
            identifier = f'offaxis_{number:03d}_{category}'
            left, right = validate_pose(left), validate_pose(right)
            assert math.isclose(np.linalg.norm(left[:3, 3]-right[:3, 3]), IPD_M, abs_tol=1e-9)
            base = [look_at_origin(p[:3, 3]) for p in (left, right)]
            fractions = [valid_fraction(left), valid_fraction(right)]
            row = {'id': identifier, 'left': f'offaxis/{identifier}_L.png', 'right': f'offaxis/{identifier}_R.png',
                   'pose_l': left.tolist(), 'pose_r': right.tolist(), 'category': category, 'params': params,
                   'pose_l_base': base[0].tolist(), 'pose_r_base': base[1].tolist(),
                   'depth_l': f'offaxis/{identifier}_L_depth.npz', 'depth_r': f'offaxis/{identifier}_R_depth.npz',
                   'valid_fraction_each_eye': fractions, 'base_rotation_angle_deg_each_eye': [
                       float(np.degrees(np.arccos(np.clip((np.trace(b[:3, :3].T@p[:3, :3])-1)/2, -1, 1))))
                       for b, p in zip(base, (left, right))]}
            rows.append(row)
        usable = [r for r in rows if min(r['valid_fraction_each_eye']) > 0]
        if not args.dry_run:
            job.images([(r[k], np.array(r[p])) for r in usable for k, p in [('left', 'pose_l'), ('right', 'pose_r')]])
            job.depths([(r[k], np.array(r[p])) for r in usable for k, p in [('depth_l', 'pose_l'), ('depth_r', 'pose_r')]])
        (args.out/'offaxis.json').write_text(json.dumps({'note': 'TA manifest fields plus pose_*_base (look-at O2 from '
            'the same eye centres, the base-model assumption). Rows with an eye lacking any valid TA ray are listed in '
            'excluded.', 'samples': usable, 'excluded': [r['id'] for r in rows if r not in usable]}, indent=2))
        report('offaxis', pairs=len(usable))

    if 'eyebox' in args.sets:
        source_rows = json.loads((args.source/'all.json').read_text())['samples']
        chosen = select_eyebox_rows(source_rows, {'train': args.eyebox_train, 'val': args.eyebox_val, 'test': args.eyebox_test})
        records_path = args.out/'eyebox_records.jsonl'
        done = {json.loads(line)['id'] for line in records_path.read_text().splitlines() if line.strip()} if records_path.exists() else set()
        for number, row in enumerate(chosen):
            identifier = f'eyebox_{row["id"]}'
            folder = f'eyebox/{row["id"]}'
            views = eyebox_views(row, offsets)
            if identifier in done or args.dry_run:
                continue
            job.images([(f'{folder}/{stem}.png', pose) for stem, pose, _, _ in views])
            job.depths([(f'{folder}/centre_{eye}_depth.npz', np.array(row[key])) for eye, key in [('left', 'pose_l'), ('right', 'pose_r')]])
            from PIL import Image
            centre_check = {}
            for eye, key in [('left', 'left'), ('right', 'right')]:
                ours = np.asarray(Image.open(args.out/folder/f'sub_{eye}_r+00.0_u+00.0.png')).astype(np.int16)
                theirs = np.asarray(Image.open(args.source/row[key])).astype(np.int16)
                centre_check[eye] = int(np.abs(ours-theirs).max())
            record = {'id': identifier, 'source_id': row['id'], 'split': row['split'], 'sampling': row['sampling'],
                      'left': f'{folder}/sub_left_r+00.0_u+00.0.png', 'right': f'{folder}/sub_right_r+00.0_u+00.0.png',
                      'pose_l': row['pose_l'], 'pose_r': row['pose_r'],
                      'depth_l': f'{folder}/centre_left_depth.npz', 'depth_r': f'{folder}/centre_right_depth.npz',
                      'centre_max_abs_difference_vs_source_u8': centre_check,
                      'views': [{'image': f'{folder}/{stem}.png', 'pose': pose.tolist(), 'kind': kind, **params}
                                for stem, pose, kind, params in views]}
            with records_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            report('eyebox', sample=number+1, of=len(chosen))
        if args.dry_run:
            views = eyebox_views(chosen[0], offsets)
            (args.out/'eyebox_design.json').write_text(json.dumps({'samples': [r['id'] for r in chosen],
                'views_per_sample': len(views), 'example': [{'stem': s, 'kind': k, **p} for s, _, k, p in views]}, indent=2))
        else:
            records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
            (args.out/'eyebox.json').write_text(json.dumps({'note': 'TA manifest fields (centre views) plus all extra '
                'views; see config.json for conventions.', 'samples': records}, indent=2))
    job.flush()
    if not args.dry_run:
        digests = {}
        for path in sorted(args.out.rglob('*')):
            if path.is_file() and path.suffix in ('.png', '.npz'):
                with path.open('rb') as stream:
                    digests[str(path.relative_to(args.out)).replace('\\', '/')] = hashlib.file_digest(stream, 'sha256').hexdigest()
        (args.out/'files.sha256.json').write_text(json.dumps(digests, indent=1))
        result = {'success': True, 'files': len(digests), 'bytes': sum((args.out/p).stat().st_size for p in digests),
                  'rendered_this_run': job.rendered, 'skipped_existing': job.skipped, 'seconds': time.perf_counter()-started,
                  'finished_utc': datetime.now(timezone.utc).isoformat(), 'training_performed': False}
        (args.out/'DONE.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    else:
        report('dry-run complete')


if __name__ == '__main__':
    main()
