"""Summarize depth proxies per subset without inventing a combined depth grade."""
import argparse
import json
from pathlib import Path

import numpy as np


def mean_or_none(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def focus_summary(rows, field):
    result = {}
    for subset in sorted({r['evaluation_subset'] for r in rows}):
        selected = [r for r in rows if r['evaluation_subset'] == subset]
        methods = sorted({m for r in selected for eye in r[field].values() for m in eye.get('methods', {})})
        result[subset] = {}
        for method in methods:
            result[subset][method] = {}
            for region in ('near', 'middle', 'far'):
                per_case = []
                for row in selected:
                    eye_metrics = [eye['methods'][method]['rois'][region] for eye in row[field].values()
                                   if method in eye.get('methods', {}) and region in eye['methods'][method]['rois']]
                    observed = [v for v in eye_metrics if v['peak_observable']]
                    if observed:
                        case = {key: mean_or_none([v[key] for v in observed]) for key in
                                ('focus_response_nrmse', 'focus_response_gain')}
                        peak_valid = [v for v in observed if not v.get('gt_peak_at_scan_boundary', False)]
                        case['peak_error_diopters'] = mean_or_none([v['peak_error_diopters'] for v in peak_valid])
                        per_case.append(case)
                result[subset][method][region] = {'observable_cases': len(per_case),
                    'peak_cases_without_gt_boundary': sum(v['peak_error_diopters'] is not None for v in per_case),
                    **{key: mean_or_none([v[key] for v in per_case]) for key in
                       ('focus_response_nrmse', 'focus_response_gain', 'peak_error_diopters')}}
            order_by_case = []
            for row in selected:
                comparisons = []
                for eye in row[field].values():
                    metrics = eye.get('methods', {}).get(method, {}).get('rois', {})
                    valid = [v for v in metrics.values() if v['peak_observable'] and not v.get('gt_peak_at_scan_boundary', False)]
                    for i, first in enumerate(valid):
                        for second in valid[i+1:]:
                            truth_delta = first['gt_peak_distance_cm']-second['gt_peak_distance_cm']
                            if abs(truth_delta) > 1e-6:
                                predicted_delta = first['peak_distance_cm']-second['peak_distance_cm']
                                comparisons.append(float(np.sign(truth_delta) == np.sign(predicted_delta)))
                if comparisons:
                    order_by_case.append(float(np.mean(comparisons)))
            result[subset][method]['depth_order'] = {'eligible_cases': len(order_by_case),
                'mean_pair_order_agreement': mean_or_none(order_by_case),
                'rule': 'case mean of eye/ROI pairs with distinct non-boundary GT peaks; prediction ties fail'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--field', choices=['focus', 'eyes'], default='focus')
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.records for line in path.read_text().splitlines() if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate case IDs across record files')
    summary = {'cases': len(rows), 'aggregation': 'equal case weights after averaging available observable eyes; regions separate',
               'missing': 'unobservable/absent ROI excluded with counts, not replaced by zero',
               'official_score': False, 'source_records': [str(p) for p in args.records],
               'focus': focus_summary(rows, args.field)}
    if args.field == 'focus':
        summary['warning'] = 'Image shift-and-add interpolation may bias peak locations; compare direct-ray calibration.'
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({'cases': len(rows), 'output': str(args.out)}))


if __name__ == '__main__':
    main()
