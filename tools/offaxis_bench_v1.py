"""Our off-axis benchmark (the reference for all off-axis evaluation from 2026-11-07).

build: Lego Gaussian scene (--asset), or every asset of a tools/uco3d_assets.py folder (--assets,
       --per-asset samples each, asset-level tune/verify split by CRC32), eye positions drawn like the TA test2 design (midpoint distance
       0.6-1.4 m, yaw -30..30 deg, elevation -5..10 deg; IPD 0.06 m along the horizontal right
       axis; every eye exactly looks at O2 with no roll, spec 2.1). Per sample:
         input    standard-camera renders (what the TA gives as left/right)
         gt       off-axis camera bbox_eye_exact (K per eye, pose = the input pose)
         physical renders with the gt camera (a consistent off-axis light field)
eval:  solve the off-axis branch from the input images with a given solver configuration and
       score under the gt camera against (a) the input images (TA rule: reuse, no rerender)
       and (b) the physical renders.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import zlib

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.offaxis_models import camera, look_at_o2  # noqa: E402

GT_MODEL = 'bbox_eye_exact'
INPUT_CAMERA = {'horizontal_fov_deg': 40.0, 'K': [[2637.5783226764374, 0, 960.0], [0, 2637.5783226764374, 540.0], [0, 0, 1]],
                'image_size': [1920, 1080], 'pose_type': 'camera_to_world', 'translation_units': 'meters',
                'world_origin': 'Layer2 center', 'world_axes': {'x': 'toward viewer', 'y': 'screen right', 'z': 'up'}}


def eye_poses(distance, yaw_deg, elevation_deg, ipd=.06):
    yaw, elevation = math.radians(yaw_deg), math.radians(elevation_deg)
    mid = distance*np.array([math.cos(elevation)*math.cos(yaw), math.cos(elevation)*math.sin(yaw), math.sin(elevation)])
    right = look_at_o2(mid)[:3, 0]  # horizontal binocular axis of the head
    return look_at_o2(mid-ipd/2*right), look_at_o2(mid+ipd/2*right)


def cmd_build(args):
    project = ROOT if (ROOT/'python_packages').is_dir() else ROOT.parents[1]
    sys.path.insert(0, str(project/'python_packages'))
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(project/'cache/torch_extensions'))
    os.environ.setdefault('MAX_JOBS', '4')
    from lightfield_lab.gaussian_scene import GaussianTargetRenderer
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'images').mkdir()
    (out/'physical').mkdir()
    if args.assets:  # (asset name, PLY, placement, extra sample fields, sample count, RNG)
        folder = Path(args.assets)
        jobs = [(a['sequence'], folder/f'{a["sequence"]}.ply', a['placement'],
                 {'asset': a['sequence'], 'category': a['category'],
                  'split': 'verify' if zlib.crc32(a['sequence'].encode()) % 3 == 0 else 'tune'},
                 args.per_asset, np.random.default_rng([args.seed, zlib.crc32(a['sequence'].encode())]))
                for a in json.loads((folder/'assets.json').read_text())]
    else:
        jobs = [(None, args.asset or project/'assets/lego_bulldozer.ply', None, {}, args.count, np.random.default_rng(args.seed))]
    samples, metadata = [], {}
    for asset, ply, placement, extra, count, rng in jobs:
        scene = GaussianTargetRenderer(ply, placement=placement)
        metadata[asset or 'lego'] = scene.metadata
        prefix = f'{asset}/' if asset else ''
        for folder_name in ('images', 'physical'):
            (out/folder_name/prefix).mkdir(exist_ok=True)
        for n in range(count):
            distance, yaw, elevation = rng.uniform(.6, 1.4), rng.uniform(-30, 30), rng.uniform(-5, 10)
            pose_l, pose_r = eye_poses(distance, yaw, elevation)
            poses = np.stack([pose_l, pose_r])
            k, gt_poses = camera(GT_MODEL, pose_l, pose_r)
            standard, _ = scene.render(poses)
            physical, info = scene.render(gt_poses, intrinsics=k)
            names = {}
            for folder_name, images in (('images', standard), ('physical', physical)):
                for side, image in zip('LR', images):
                    name = f'{folder_name}/{prefix}offaxis_{n:06d}_{side}.png'
                    Image.fromarray(image).save(out/name, compress_level=1)
                    names[(folder_name, side)] = name
            samples.append({'id': f'{asset}_{n:03d}' if asset else f'offaxis_{n:06d}', **extra,
                            'pose_l': pose_l.tolist(), 'pose_r': pose_r.tolist(),
                            'left': names[('images', 'L')], 'right': names[('images', 'R')],
                            'design': {'distance_m': distance, 'yaw_deg': yaw, 'elevation_deg': elevation},
                            'gt_offaxis': {'model': GT_MODEL, 'K_l': k_mat(k[0]), 'K_r': k_mat(k[1]),
                                           'pose_l': gt_poses[0].tolist(), 'pose_r': gt_poses[1].tolist(),
                                           'physical_left': names[('physical', 'L')], 'physical_right': names[('physical', 'R')],
                                           'physical_foreground': info['foreground_fraction_each_eye']}})
            if len(samples) % 20 == 0:
                print(f'[build] {len(samples)} samples ({asset or "lego"})', flush=True)
        del scene
        torch.cuda.empty_cache()
    (out/'test.json').write_text(json.dumps({
        'phase': 'offaxis_bench_v1', 'counts_toward_grade': False, 'input_camera': INPUT_CAMERA,
        'geometry': {'image_h': 1080, 'image_w': 1920, 'screen_w_m': .596736, 'layer_xs_m': [.06, 0., -.06], 'fov_h_deg': 40., 'ipd_m': .06},
        'offaxis_task': {'reference_layer': 'L2', 'intrinsics': 'dynamic', 'target_policy': 'reuse original left/right RGB without rerendering'},
        'design': {'seed': args.seed, 'distance_m': [.6, 1.4], 'yaw_deg': [-30, 30], 'elevation_deg': [-5, 10], 'look_at': 'O2 exact, no roll'},
        'scene': metadata if args.assets else metadata['lego'], 'samples': samples}, indent=1))


def k_mat(k):
    return [[float(k[0]), 0., float(k[2])], [0., float(k[1]), float(k[3])], [0., 0., 1.]]


def cmd_eval(args):
    from lightfield_lab.core import scores, synchronize
    from lightfield_lab.exam_adapter import ExamPredictor
    from lightfield_lab.exam_data import read_rgb8
    from lightfield_lab.exam_geometry import ExamRenderer, HEIGHT, WIDTH
    device = torch.device('cuda')
    index = Path(args.index)
    base = index.parent
    rows = json.loads(index.read_text())['samples'][:args.limit]
    predictor = ExamPredictor(None, steps=args.steps, lr=args.lr, diagnostics=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    load = lambda names: torch.from_numpy(np.stack([read_rgb8(base/n) for n in names])).permute(0, 3, 1, 2).to(device).double()/255
    records = []
    for row in rows:
        gt = row['gt_offaxis']
        gt_k = np.array([[m[0][0], m[1][1], m[0][2], m[1][2]] for m in (gt['K_l'], gt['K_r'])])
        gt_renderer = ExamRenderer(np.array([gt['pose_l'], gt['pose_r']]), HEIGHT, WIDTH, device, geometry_backend='torch', intrinsics=gt_k)
        reused, physical = load([row['left'], row['right']]), load([gt['physical_left'], gt['physical_right']])
        left, right = read_rgb8(base/row['left']), read_rgb8(base/row['right'])
        record = {'id': row['id'], **row['design'], **{k: row[k] for k in ('asset', 'category', 'split') if k in row}}
        for config in args.configs:
            model, initial = config.split(':')
            synchronize(device)
            started = time.perf_counter()
            if model == 'standard':
                layers = predictor(left, right, row['pose_l'], row['pose_r'], initial_method=initial)
            else:
                k, c2w = camera(model, row['pose_l'], row['pose_r'])
                rotated = not np.allclose(c2w[:, :3, :3], np.array([row['pose_l'], row['pose_r']])[:, :3, :3], atol=1e-12)
                layers = predictor(left, right, row['pose_l'], row['pose_r'], intrinsics=k,
                                   render_poses=c2w if rotated else None, initial_method=initial)
            synchronize(device)
            seconds = time.perf_counter()-started
            patterns = torch.from_numpy(np.stack(layers)).permute(0, 3, 1, 2).to(device).double()/255
            with torch.no_grad():
                record[config] = {'reused': scores(gt_renderer, patterns, reused)['psnr_mean_eyes'],
                                  'physical': scores(gt_renderer, patterns, physical)['psnr_mean_eyes'], 'seconds': seconds}
        records.append(record)
        print(row['id'], ' '.join(f"{c}: {record[c]['reused']:.2f}/{record[c]['physical']:.2f}" for c in args.configs), flush=True)
    summary = {}
    for config in args.configs:
        entry = {}
        for target in ('reused', 'physical'):
            v = np.array([r[config][target] for r in records])
            entry[target] = {'mean': float(v.mean()), 'median': float(np.median(v)), 'worst5_mean': float(np.sort(v)[:max(1, len(v)//20)].mean())}
        entry['seconds_mean'] = float(np.mean([r[config]['seconds'] for r in records]))
        summary[config] = entry
    groups = {}
    for key in ('split', 'asset'):
        values = sorted({r[key] for r in records if key in r})
        groups[key] = {v: {c: float(np.mean([r[c]['reused'] for r in records if r.get(key) == v])) for c in args.configs} for v in values}
    (out/'summary.json').write_text(json.dumps({'gt_model': GT_MODEL, 'n': len(records), 'steps': args.steps, 'lr': args.lr,
                                                'summary': summary, 'groups': groups, 'records': records}, indent=1))
    print(f'GT camera {GT_MODEL}; mean / median / worst-5% PSNR dB; target reused (TA rule) | physical render')
    for config, e in summary.items():
        print(f"  {config:28s} reused {e['reused']['mean']:6.2f} {e['reused']['median']:6.2f} {e['reused']['worst5_mean']:6.2f}"
              f" | physical {e['physical']['mean']:6.2f} {e['physical']['median']:6.2f} {e['physical']['worst5_mean']:6.2f}  [{e['seconds_mean']:.2f}s]")
    for split, means in groups['split'].items():
        print(f'  split {split}: ' + ', '.join(f'{c} {v:.2f}' for c, v in means.items()))
    if groups['asset']:
        first = args.configs[0]
        ranked = sorted(groups['asset'].items(), key=lambda item: item[1][first])
        print(f'  per-asset mean ({first}): worst ' + ', '.join(f'{a} {m[first]:.1f}' for a, m in ranked[:6])
              + ' | best ' + ', '.join(f'{a} {m[first]:.1f}' for a, m in ranked[-3:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', required=True)
    b.add_argument('--count', type=int, default=200)
    b.add_argument('--seed', type=int, default=20261107)
    b.add_argument('--asset', default=None)
    b.add_argument('--assets', default=None, help='folder from tools/uco3d_assets.py convert (assets.json + PLYs)')
    b.add_argument('--per-asset', type=int, default=6)
    e = sub.add_parser('eval')
    e.add_argument('--index', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--limit', type=int, default=None)
    e.add_argument('--steps', type=int, default=100)
    e.add_argument('--lr', type=float, default=.08)
    e.add_argument('--configs', nargs='+', default=['bbox_eye_exact:sart', 'bbox_eye_exact:network', 'standard:network', 'l2_camera:sart'])
    args = parser.parse_args()
    {'build': cmd_build, 'eval': cmd_eval}[args.command](args)


if __name__ == '__main__':
    main()
