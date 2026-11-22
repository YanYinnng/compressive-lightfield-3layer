"""Generate TA-compatible stereo targets from the supplied Lego Gaussian PLY."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT/'python_packages').is_dir():sys.path.insert(0, str(ROOT/'python_packages'))
os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT/'cache/torch_extensions'))
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0')
os.environ.setdefault('MAX_JOBS', '4')

import numpy as np
from PIL import Image
import torch

from lightfield_lab.gaussian_scene import GaussianTargetRenderer, stereo_poses


def design_samples(seed, counts, pilot=False):
    if pilot:
        positions = [(.95, 0, 0), (.75, -20, 8), (.75, 20, -8), (1.25, -32, 0),
                     (1.25, 32, 0), (.65, 0, 10), (1.35, 15, -10), (1., -10, 0)]
        return [('pilot', index, *position) for index, position in enumerate(positions)]
    generator = np.random.default_rng(seed)
    rows = []
    # Eight-degree midpoint gaps also separate individual eye yaw ranges at 6cm IPD.
    for split, limits, count in [('train', (-22., 22.), counts[0]), ('val', (30., 38.), counts[1]), ('test', (-38., -30.), counts[2])]:
        for index in range(count):
            rows.append((split, index, float(generator.uniform(.65, 1.35)),
                         float(generator.uniform(*limits)), float(generator.uniform(-10, 10))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, default=ROOT/'assets/lego_bulldozer.ply')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--train', type=int, default=256)
    parser.add_argument('--val', type=int, default=32)
    parser.add_argument('--test', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--scale', type=float, default=.12)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.train, args.val, args.test) < 0 or args.scale <= 0:parser.error('Invalid counts/scale.')
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        parser.error('Choose a new empty output, or --resume an identical configuration.')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'images').mkdir(exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    renderer = GaussianTargetRenderer(args.asset, args.scale)
    config = {'seed': args.seed, 'counts_requested': [args.train, args.val, args.test], 'pilot': args.pilot,
        'image': {'width': 1920, 'height': 1080, 'mode': 'RGB8 PNG', 'horizontal_fov_degrees': 40., 'principal_point': [960,540]},
        'pose': 'c2w metres, right/up/back columns, no roll, each eye looks at O2', 'ipd_metres': .06,
        'sampling': {'distance_metres': [.65,1.35], 'elevation_degrees': [-10,10],
            'train_yaw_degrees': [-22,22], 'val_yaw_degrees': [30,38], 'test_yaw_degrees': [-38,-30],
            'scope': 'chosen development envelope, not a claim to cover every formal test pose'},
        'renderer': {'name': 'gsplat', 'version': __import__('gsplat').__version__, 'torch': torch.__version__,
                     'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0), **renderer.metadata}}
    config_path = args.out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Resume configuration differs.')
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    manifest = args.out/'records.jsonl'
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()] if manifest.exists() else []
    by_id = {row['id']: row for row in records}
    if len(by_id) != len(records):raise ValueError('Duplicate existing IDs.')
    design = design_samples(args.seed, [args.train,args.val,args.test], args.pilot)
    with manifest.open('a', encoding='utf-8') as output:
        for split, index, distance, yaw, elevation in design:
            identifier = f'{split}_{index:06d}'
            poses = stereo_poses(distance, yaw, elevation)
            if identifier in by_id:
                row = by_id[identifier]
                if not all((args.out/row[key]).is_file() for key in ['left','right']):raise ValueError('Committed row has missing images.')
                if not np.allclose(np.array([row['pose_l'],row['pose_r']]),poses,atol=1e-12):raise ValueError('Resume pose mismatch.')
                continue
            images, diagnostic = renderer.render(poses)
            if diagnostic['left_right_mean_abs_difference'] <= 1e-6 or min(diagnostic['foreground_fraction_each_eye']) < .005:
                raise ValueError(f'Degenerate generated pair: {identifier}: {diagnostic}')
            paths = []
            for eye, label in enumerate(['L','R']):
                relative = f'images/{identifier}_{label}.png'
                path = args.out/relative; temporary = path.with_suffix('.png.part')
                Image.fromarray(images[eye]).save(temporary, format='PNG', compress_level=2)
                temporary.replace(path);paths.append(relative)
            row = {'id': identifier, 'left': paths[0], 'right': paths[1], 'pose_l': poses[0].tolist(), 'pose_r': poses[1].tolist(),
                   'split': split, 'sampling': {'distance_m': distance, 'yaw_deg': yaw, 'elevation_deg': elevation}, 'diagnostic': diagnostic}
            output.write(json.dumps(row)+'\n');output.flush()
            records.append(row);by_id[identifier] = row
            if len(records) % 10 == 0 or len(records)==len(design):
                progress = {'completed_pairs': len(records), 'expected_pairs': len(design), 'seconds': time.perf_counter()-started}
                (args.out/'progress.json').write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
    for split in ['pilot'] if args.pilot else ['train','val','test']:
        (args.out/f'{split}.json').write_text(json.dumps({'samples':[r for r in records if r['split']==split]},indent=2))
    (args.out/'all.json').write_text(json.dumps({'samples':records},indent=2))
    result = {'success': True, 'pairs': len(records), 'images': 2*len(records),
        'split_counts': {s:sum(r['split']==s for r in records) for s in sorted({r['split'] for r in records})},
        'seconds': time.perf_counter()-started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': renderer.metadata['source_sha256'], 'training_performed': False}
    (args.out/'DONE.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
