"""Summarize development probes without pooling overlapping sample sets."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
paths = ['public_pose_sweep', 'public_object_sweep', 'public_object_1080p',
         'public_network100_1080p', 'public_random_sweep', 'analytic_sweep', 'analytic_consistency_sweep']
summary = {}
for name in paths:
    path = ROOT/'runs'/name/'summary.json'
    if not path.is_file():
        continue
    report = json.loads(path.read_text(encoding='utf-8'))
    cases = report['cases']
    if not cases:
        continue
    baseline = np.array([c['variants']['network']['psnr_mean_eyes'] for c in cases])
    result = {'case_count': len(cases), 'configuration': report['configuration'], 'variants': {}}
    common = set.intersection(*(set(c['variants']) for c in cases))
    for variant in cases[0]['variants']:
        if variant not in common:
            continue
        rows = [c['variants'][variant] for c in cases]
        gains = np.array([row['psnr_mean_eyes'] for row in rows])-baseline
        fields = ['psnr_mean_eyes', 'upstream_pooled_psnr', 'diagnostic_nonwhite_psnr',
                  'neighbour_psnr_1p5mm', 'neighbour_psnr_6mm', 'pattern_rmse_from_network',
                  'seconds_refine_and_checkpoint_scoring']
        means = {field: float(np.mean([row[field] for row in rows])) for field in fields
                 if all(field in row for row in rows)}
        result['variants'][variant] = {**means, 'minimum_gain_over_initial_dB': float(gains.min()),
            'maximum_gain_over_initial_dB': float(gains.max())}
    summary[name] = result
out = ROOT/'runs/exploration_summary.json'
out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
