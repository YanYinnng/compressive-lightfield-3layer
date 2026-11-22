"""Measure frozen v3 configurations on a TA-format EyeReal research split."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.exam_geometry import HEIGHT, WIDTH
from lightfield_lab.ray_eval import render_views, ta_grids
from lightfield_lab.v3_submission import make_v3_predictor
from testing_v3.inference import read_rgb, read_rows


CONFIGS = {
    'init': (0, .08, 'lens_shift_cyclopean'),
    'step30': (30, .02, 'lens_shift_cyclopean'),
    'step60': (60, .08, 'lens_shift_cyclopean'),
    'step60_lr04': (60, .04, 'lens_shift_cyclopean'),
    'step80_lr06': (80, .06, 'lens_shift_cyclopean'),
    'step100': (100, .08, 'lens_shift_cyclopean'),
    'step200': (200, .08, 'lens_shift_cyclopean'),
    'lens_shift': (100, .08, 'lens_shift'),
    'lens_shift_stereo': (100, .08, 'lens_shift_stereo'),
    'screen_fit': (100, .08, 'screen_fit'),
}


def pupil_grids(poses, device):
    # Same orientation, offsets along local right; no new target views exist.
    expanded = np.repeat(poses, 3, axis=0)
    expanded[:, :3, 3] += np.tile([0, -.002, .002], 2)[:, None]*expanded[:, :3, 0]
    return ta_grids(expanded, HEIGHT, WIDTH, device, torch.float64)


@torch.no_grad()
def pupil_drift(patterns, grids, masks):
    patterns = torch.as_tensor(np.stack(patterns), device=grids.device).permute(0, 3, 1, 2).double()/255
    eyes = render_views(patterns, grids).reshape(2, 3, 3, HEIGHT, WIDTH)
    masks = masks.reshape(2, 3, HEIGHT, WIDTH)
    support = masks[:, 1:] & masks[:, :1]
    count = support.sum(dim=(-1, -2))
    error = ((eyes[:, 1:]-eyes[:, :1]).square()*support[:, :, None]).sum(dim=(-1, -2, -3))
    rmse = (error/(3*count.clamp_min(1))).sqrt()
    return {'mean_rmse': float(rmse[count > 0].mean()) if count.any() else None,
            'min_common_fraction': float(count.min())/(HEIGHT*WIDTH)}


def summarize(records):
    summary = {}
    for name, group in records.items():
        row = {'samples': len(group), 'sequences': len({r['sequence'] for r in group})}
        for key in ('normal_psnr', 'offaxis_psnr', 'normal_initial_psnr', 'offaxis_initial_psnr',
                    'normal_valid_fraction', 'offaxis_valid_fraction', 'solve_seconds'):
            values = np.array([r[key] for r in group])
            row[key] = {'mean': float(values.mean()), 'p05': float(np.quantile(values, .05)),
                        'p50': float(np.quantile(values, .5)), 'p95': float(np.quantile(values, .95))}
            if key in ('normal_psnr', 'offaxis_psnr'):
                row[key]['worst_5pct_mean'] = float(np.sort(values)[:math.ceil(.05*len(values))].mean())
        drift = [r['pupil_drift']['mean_rmse'] for r in group if r['pupil_drift'] is not None
                 and r['pupil_drift']['mean_rmse'] is not None]
        row['pupil_drift_mean_rmse'] = float(np.mean(drift)) if drift else None
        summary[name] = row
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', nargs='+', choices=CONFIGS, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT/'weights/pretrained_model.pth')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if len(set(args.config)) != len(args.config):
        parser.error('Provide each configuration once.')
    args.out.mkdir(parents=True, exist_ok=False)
    predict = make_v3_predictor(args.weights, device=args.device)
    rows = read_rows(args.index)
    records = defaultdict(list)
    started = time.perf_counter()
    with (args.out/'samples.jsonl').open('w', encoding='utf-8', buffering=1) as stream:
        for number, row in enumerate(rows, 1):
            images = [read_rgb(args.index.parent/row[key]) for key in ('left', 'right')]
            poses = np.asarray([row['pose_l'], row['pose_r']])
            grids, masks = pupil_grids(poses, predict.solver.device)
            for name in args.config:
                steps, lr, predict.offaxis_model = CONFIGS[name]
                predict.solver.steps, predict.solver.lr = steps, lr
                tick = time.perf_counter()
                result = predict(*images, *poses)
                seconds = time.perf_counter()-tick
                scores = predict.last_scores
                actual_steps = [scores[f'solver_{key}']['refinement']['steps_run'] for key in ('normal', 'offaxis')]
                if actual_steps != [steps, steps]:
                    raise ValueError(f'{row["id"]}: expected {steps} steps, got {actual_steps}.')
                record = {'id': row['id'], 'sequence': row['sequence'], 'variant': row['variant'],
                          'config': name, 'solve_seconds': seconds, 'steps_run': actual_steps,
                          'pupil_drift': pupil_drift(result['patterns'], grids, masks)}
                for branch in ('normal', 'offaxis'):
                    record[f'{branch}_psnr'] = scores[branch]['psnr_mean_eyes']
                    record[f'{branch}_initial_psnr'] = scores[f'solver_{branch}']['network_initial_psnr_cached_dB']
                    record[f'{branch}_valid_fraction'] = float(np.mean(scores[f'{branch}_valid_pixels']))/(HEIGHT*WIDTH)
                if not all(math.isfinite(record[k]) for k in ('normal_psnr', 'offaxis_psnr')):
                    raise ValueError(f'Nonfinite score in {row["id"]}: {name}')
                records[name].append(record)
                stream.write(json.dumps(record, allow_nan=False)+'\n')
            print(f'{number}/{len(rows)} {row["id"]}', flush=True)
    by_variant = {variant: summarize({name: [r for r in group if r['variant'] == variant]
                                      for name, group in records.items()})
                  for variant in sorted({row['variant'] for row in rows})}
    summary = {'index': str(args.index), 'configurations': {name: CONFIGS[name] for name in args.config},
               'scope': 'RGB8 and TA masks; solve time excludes input IO and drift probe. Drift is local appearance change without GT.',
               'seconds': time.perf_counter()-started, 'overall': summarize(records), 'by_variant': by_variant}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
