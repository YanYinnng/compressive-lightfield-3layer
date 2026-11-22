"""Two questions about the off-axis drop and the solver.

A. physical: same off-axis camera, two targets. 'reinterpreted' = the standard-camera render reused
   as the off-axis target (TA policy); 'physical' = the scene rendered with the off-axis camera
   (a consistent light field). Baseline 'standard' = standard render, standard camera.
B. solvers: SART + Adam, SART with periodic-branch reassignment, Adam with a branch-loss warm start.
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
from lightfield_lab.core import refine, scores, synchronize  # noqa: E402
from lightfield_lab.exam_data import read_rgb8  # noqa: E402
from lightfield_lab.exam_geometry import ExamRenderer, HEIGHT, WIDTH  # noqa: E402
from lightfield_lab.offaxis_models import camera  # noqa: E402
from lightfield_lab.tomographic import sart  # noqa: E402

device = torch.device('cuda')


def load(base, row):
    return torch.from_numpy(np.stack([read_rgb8(base/row[k]) for k in ('left', 'right')])).permute(0, 3, 1, 2).to(device).float()/255


def solve(renderer, images, method):
    synchronize(device)
    started = time.perf_counter()
    if method == 'sart30+adam100':
        p, _ = refine(renderer, images, sart(renderer, images, 30), steps=100, lr=.08, iteration_dtype=torch.float32)
    elif method == 'sartB30+adam100':
        p, _ = refine(renderer, images, sart(renderer, images, 30, branch_from=10), steps=100, lr=.08, iteration_dtype=torch.float32)
    elif method == 'sartB100':
        p = renderer.quantize_patterns(sart(renderer, images, 100, branch_from=10))
    elif method == 'sart100':
        p = renderer.quantize_patterns(sart(renderer, images, 100))
    elif method == 'sart30+phase50+adam100':
        p, _ = refine(renderer, images, sart(renderer, images, 30), steps=100, lr=.08, phase_steps=50, iteration_dtype=torch.float32)
    else:
        raise ValueError(method)
    synchronize(device)
    with torch.no_grad():
        return scores(renderer, p, images.double())['psnr_mean_eyes'], time.perf_counter()-started


def part_a(args, out):
    root = Path(args.physical)
    std = json.loads((root/'standard/test.json').read_text())['samples'][:args.limit]
    rows = []
    for number, s_row in enumerate(std):
        poses = np.array([s_row['pose_l'], s_row['pose_r']])
        standard_images = load(root/'standard', s_row)
        record = {'id': s_row['id']}
        record['standard'] = solve(ExamRenderer(poses, HEIGHT, WIDTH, device, geometry_backend='torch'), standard_images, 'sart30+adam100')[0]
        for model in args.models:
            m_row = json.loads((root/model/'test.json').read_text())['samples'][number]
            k, c2w = np.array(m_row['gt_K_fx_fy_cx_cy']), np.array([m_row['gt_pose_l'], m_row['gt_pose_r']])
            r = ExamRenderer(c2w, HEIGHT, WIDTH, device, geometry_backend='torch', intrinsics=k)
            record[f'{model}/reinterpreted'] = solve(r, standard_images, 'sart30+adam100')[0]
            record[f'{model}/physical'] = solve(r, load(root/model, m_row), 'sart30+adam100')[0]
        rows.append(record)
        print('A', json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in record.items()}), flush=True)
    keys = [k for k in rows[0] if k != 'id']
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print('A summary (mean PSNR dB, sart30+adam100):', json.dumps({k: round(v, 2) for k, v in summary.items()}), flush=True)
    (out/'part_a.json').write_text(json.dumps({'summary': summary, 'records': rows}, indent=1))


def part_b(args, out):
    methods = ['sart30+adam100', 'sartB30+adam100', 'sart30+phase50+adam100', 'sart100', 'sartB100']
    rows = []
    for spec in args.index:
        name, path = spec.split('=', 1)
        path = Path(path)
        for row in json.loads(path.read_text(encoding='utf-8'))['samples'][:args.limit_b]:
            images = load(path.parent, row)
            poses = np.array([row['pose_l'], row['pose_r']])
            for model in ['standard']+args.models:
                k, c2w = camera(model, *poses)
                r = ExamRenderer(c2w, HEIGHT, WIDTH, device, geometry_backend='torch', intrinsics=None if model == 'standard' else k)
                record = {'set': name, 'id': row['id'], 'model': model}
                for method in methods:
                    record[method] = solve(r, images, method)
                rows.append(record)
                print('B', name, row['id'], model, ' '.join(f'{m}:{record[m][0]:.2f}' for m in methods), flush=True)
    summary = {}
    for key in sorted({(r['set'], r['model']) for r in rows}):
        group = [r for r in rows if (r['set'], r['model']) == key]
        summary['/'.join(key)] = {m: [float(np.mean([g[m][0] for g in group])), float(np.mean([g[m][1] for g in group]))] for m in methods}
    print('B summary [mean PSNR dB, seconds]:')
    for key, value in summary.items():
        print(f'  {key:24s} ' + '  '.join(f'{m}: {v[0]:.2f} [{v[1]:.2f}s]' for m, v in value.items()), flush=True)
    (out/'part_b.json').write_text(json.dumps({'summary': summary, 'records': rows}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical', required=True)
    parser.add_argument('--index', action='append', default=[])
    parser.add_argument('--models', nargs='+', default=['l2_camera', 'bbox_eye'])
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--limit-b', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    part_a(args, args.out)
    part_b(args, args.out)


if __name__ == '__main__':
    main()
