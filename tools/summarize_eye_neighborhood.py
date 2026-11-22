"""Validate the paired eye-neighbourhood run and summarize the frozen split."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

from PIL import Image
from summarize_depth_constraints import REGIONS, focus_case, load_rows, margins, mean_valid


METHODS = ('adam100', 'center_tail', 'oracle_tail', 'stereo_tail')
BOOKSHELF = ('adam100', 'center_tail', 'stereo_angular', 'stereo_phase')


def check_patterns(run, groups):
    for files in groups.values():
        if len(files) != 3:
            raise ValueError('Expected three output layers.')
        for file in files:
            path = run/file
            with path.open('rb') as stream:
                if stream.read(26)[24:26] != bytes((8, 2)):
                    raise ValueError(f'Not RGB8: {path}')
            with Image.open(path) as image:
                if (image.format, image.mode, image.size) != ('PNG', 'RGB', (1920, 1080)):
                    raise ValueError(f'Invalid pattern: {path}')
                image.load()


def bookshelf_summary(run, ids):
    rows = load_rows(run/'records.jsonl')
    if set(rows) != ids:
        raise ValueError('Bookshelf run must cover the predeclared IDs.')
    for row in rows.values():
        if set(row['methods']) != set(BOOKSHELF) or not row['methods']['stereo_angular']['meets_floor']:
            raise ValueError(f"Incomplete or invalid candidate for {row['id']}")
        check_patterns(run, row.get('patterns', {}))
    result = {'n': len(rows), 'depth_proxy_fraction': mean(r['depth_proxy_fraction'] for r in rows.values()),
              'fit_support': mean(r['fit_support'] for r in rows.values())}
    for name in BOOKSHELF:
        values = [r['methods'][name] for r in rows.values()]
        deltas = [m['psnr_dB']-r['methods']['adam100']['psnr_dB'] for r, m in zip(rows.values(), values)]
        result[name] = {field: mean_valid([m[field] for m in values]) for field in
                        ('psnr_dB', 'seconds', 'allocation_mse_stereo_proxy', 'angular_nrmse_stereo_proxy')}
        result[name].update({
            'delta_dB': mean(deltas), 'worst_delta_dB': min(deltas),
            'worst5pct_dB': mean(sorted(m['psnr_dB'] for m in values)[:math.ceil(.05*len(rows))]),
            'passes': sum(m['meets_floor'] for m in values), 'positive_step': sum(m['step'] > 0 for m in values),
            'contrast_margins': [mean_valid(v) for v in zip(*(
                margins({'contrast_matrix_oracle': m['contrast_matrix_stereo_proxy']}) for m in values))],
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--bookshelf', action='store_true')
    args = parser.parse_args()
    ids = set(json.loads((args.run/'ids.json').read_text()))
    if args.bookshelf:
        result = bookshelf_summary(args.run, ids)
        (args.run/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps(result, allow_nan=False))
        return
    rows = load_rows(args.run/'compare_records.jsonl')
    focus = load_rows(args.run/'focus/records.jsonl')
    if set(rows) != ids or set(focus) != ids:
        raise ValueError('Comparison and focus must cover the predeclared IDs.')
    for row in rows.values():
        if set(row['patterns']) != set(METHODS) or set(row['methods']) != set(METHODS):
            raise ValueError(f"Incomplete methods for {row['id']}")
        for name in METHODS[2:]:
            if row['methods'][name]['psnr_dB'] < row['floor_dB']-1e-5:
                raise ValueError(f"PSNR guard failed for {row['id']} {name}")
        check_patterns(args.run, row['patterns'])
        for eye in focus[row['id']]['eyes'].values():
            if set(eye['methods']) != set(METHODS) or eye['height'] != 1080 or eye['pupil_samples'] != 13:
                raise ValueError(f"Focus protocol mismatch for {row['id']}")
    held = [r for r in rows.values() if r['subset'] != 'train_tune']
    result = {'n': len(held), 'ids': [r['id'] for r in held],
              'target_support': {s: mean(r['target_support'][s] for r in held) for s in ('oracle', 'stereo')}}
    for name in METHODS:
        values = [r['methods'][name] for r in held]
        result[name] = {k: mean(m[k] for m in values) for k in
                        ('psnr_dB', 'delta_dB', 'angular_holdout_nrmse', 'seconds_incl_baseline_and_depth',
                         'allocation_mse_oracle')}
        focal = [focus_case(focus[r['id']], name) for r in held]
        result[name].update({
            'used_tail': sum(v['selected_tail_step'] > 0 for v in values),
            'worst_delta_dB': min(v['delta_dB'] for v in values),
            'focal_stack_dB': mean(item[0] for item in focal),
            'focus_nrmse': {k: mean_valid([item[1][k] for item in focal]) for k in REGIONS},
            'focus_eligible': {k: sum(item[1][k] is not None for item in focal) for k in REGIONS},
            'contrast_margins': [mean_valid(v) for v in zip(*(margins(m) for m in values))],
            'paired_wins': {k: sum(m[k] < r['methods']['adam100'][k] for r, m in zip(held, values))
                            for k in ('angular_holdout_nrmse', 'allocation_mse_oracle')},
        })
    (args.run/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
