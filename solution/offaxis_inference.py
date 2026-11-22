"""Research off-axis patterns with declared per-eye K and unchanged input/poses.

Output is an experimental bundle. The separate official off-axis format has not
been supplied; this manifest retains all information needed for later conversion.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.core import make_model, refine, scores, save_images
from lightfield_lab.exam_adapter import network_initial
from lightfield_lab.exam_data import read_manifest, read_rgb8
from lightfield_lab.exam_geometry import ExamRenderer, screen_corners, HEIGHT, WIDTH
from lightfield_lab.intrinsics import normalize_intrinsics, warp_to_standard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--intrinsics', type=Path, required=True, help='JSON samples: [{id,intrinsics:[[fx,fy,cx,cy],...]}]')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT/'weights/pretrained_model.pth')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.08)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--direct-initial', action='store_true', help='Control: bypass intrinsic image warp')
    args = parser.parse_args()
    torch.set_num_threads(4)
    index, rows = read_manifest(args.index)
    declared = json.loads(args.intrinsics.read_text())['samples']
    if len({r['id'] for r in declared}) != len(declared):
        parser.error('Duplicate intrinsic sample IDs')
    declared = {r['id']: r['intrinsics'] for r in declared}
    if set(declared) != {r['id'] for r in rows}:
        parser.error('Intrinsic IDs must exactly match the input manifest')
    for row in rows:
        normalize_intrinsics(declared[row['id']], 2, HEIGHT, WIDTH)
    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    model = make_model(HEIGHT, WIDTH, device, args.weights)
    records = []
    for row in rows:
        started = time.perf_counter()
        images = torch.from_numpy(np.stack([read_rgb8(index.parent/row[k]) for k in ('left', 'right')])).permute(0, 3, 1, 2).to(device).float()/255
        poses = np.array([row['pose_l'], row['pose_r']])
        k = declared[row['id']]
        initial = network_initial(model, images, torch.from_numpy(poses).to(device), screen_corners(device),
                                  intrinsics=None if args.direct_initial else k)
        renderer = ExamRenderer(poses, HEIGHT, WIDTH, device, geometry_backend='torch', intrinsics=k)
        patterns, trace = refine(renderer, images, initial, steps=args.steps, lr=args.lr, iteration_dtype=torch.float32)
        save_images(patterns, args.out/row['id'], 'layer')
        _, support = warp_to_standard(images, k)
        records.append({'id': row['id'], 'intrinsics': k, 'pose_l': row['pose_l'], 'pose_r': row['pose_r'],
                        'patterns': [f'{row["id"]}/layer-{n}.png' for n in (1, 2, 3)],
                        'score_under_declared_K': scores(renderer, patterns, images),
                        'valid_fraction': renderer.geometry_mask.mean((1, 2, 3)).tolist(),
                        'network_warp_support_fraction': float(support.float().mean()),
                        'seconds': time.perf_counter()-started, 'trace': trace})
    (args.out/'offaxis_patterns.json').write_text(json.dumps({'research_format': True, 'input_images_unchanged': True,
        'poses_unchanged': True, 'samples': records}, indent=2))
    print(f'Generated {len(records)} off-axis research samples.')


if __name__ == '__main__':
    main()
