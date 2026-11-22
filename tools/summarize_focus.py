"""Keep experiment sets separate and expose peak collapse, not only mean error."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
names = ['focus_existing', 'focus_textured_evaluation', 'focus_sampling_check', 'focus_jitter_check']
results = {}
for name in names:
    path = ROOT/'runs'/name/'summary.json'
    if not path.exists():
        continue
    report = json.loads(path.read_text())
    item = {'case_count': len(report['cases']), 'aggregate': report['aggregate'], 'depth_order': {}}
    for diameter in report['aggregate']:
        methods = report['aggregate'][diameter]
        item['depth_order'][diameter] = {}
        for method in methods:
            triplets = []
            for case in report['cases']:
                for eye in case['eyes']:
                    if eye['diameter_mm'] != float(diameter):
                        continue
                    rois = eye['variants'][method]['rois']
                    if not all(label in rois and rois[label]['peak_observable'] and not rois[label]['gt_peak_at_scan_boundary']
                               for label in ['near', 'middle', 'far']):
                        continue
                    predicted = [rois[label]['peak_distance_cm'] for label in ['near', 'middle', 'far']]
                    truth = [rois[label]['gt_peak_distance_cm'] for label in ['near', 'middle', 'far']]
                    span = max(truth)-min(truth)
                    triplets.append({'case': case['name'], 'eye': eye['eye'], 'predicted_cm': predicted, 'truth_cm': truth,
                        'strict_order_correct': predicted[0] < predicted[1] < predicted[2],
                        'predicted_span_over_gt_span': (max(predicted)-min(predicted))/span if span > 0 else None})
            ratios = [v['predicted_span_over_gt_span'] for v in triplets if v['predicted_span_over_gt_span'] is not None]
            item['depth_order'][diameter][method] = {
                'eligible_triplets': len(triplets), 'strict_order_correct': sum(v['strict_order_correct'] for v in triplets),
                'mean_predicted_span_over_gt_span': float(np.mean(ratios)) if ratios else None,
                'triplets': triplets}
    results[name] = item
out = ROOT/'runs/focus_summary.json'
out.write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
print('Wrote', out)
for name, item in results.items():
    print(name, 'cases:', item['case_count'])
    for diameter, methods in item['depth_order'].items():
        print('pupil', diameter, {k: (v['strict_order_correct'], v['eligible_triplets'], v['mean_predicted_span_over_gt_span'])
                                 for k, v in methods.items()})
