"""Summarize paired depth experiments and focal metrics on a frozen split."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from PIL import Image


REGIONS = ('near', 'middle', 'far')


def load_rows(path):
    rows = list(map(json.loads, path.read_text().splitlines()))
    indexed = {row['id']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f'Duplicate case IDs in {path}')
    return indexed


def mean_valid(values):
    valid = [v for v in values if v is not None]
    return mean(valid) if valid else None


def margins(method):
    for i, name in enumerate(REGIONS):
        row = method['contrast_matrix_oracle'][name]
        yield row[i]-max(x for j, x in enumerate(row) if j != i) if row else None


def focus_case(row, method):
    eyes = [eye['methods'][method] for eye in row['eyes'].values() if 'methods' in eye]
    response = {}
    for label in REGIONS:
        response[label] = mean_valid([e['rois'].get(label, {}).get('focus_response_nrmse') for e in eyes])
    return mean_valid([e['stack_psnr_mean_dB'] for e in eyes]), response


def summarize(rows, focus):
    result, csv_rows = {}, []
    for method in rows[0]['methods']:
        individual = []
        for row in rows:
            data = row['methods'][method]
            stack, response = focus_case(focus[row['id']], method)
            entry = {'case': row['id'], 'subset': row['subset'], 'method': method,
                     'center_dB': data['psnr_dB'],
                     'delta_dB': data['psnr_dB']-row['methods']['adam100']['psnr_dB'],
                     'meets_floor': int(data['meets_floor']),
                     'allocation_mse': data['allocation_mse_oracle'],
                     'seconds': data['solve_s_incl_depth'],
                     'seconds_with_reference_and_png': data['solve_s_incl_depth_and_png']+data.get('reference_adam_s', 0.),
                     'focal_stack_dB': stack, 'selected_step': data['selected_step'],
                     **{f'{label}_contrast_margin': value for label, value in zip(REGIONS, margins(data))},
                     **{f'{label}_focus_nrmse': value for label, value in response.items()}}
            individual.append(entry)
        csv_rows.extend(individual)
        fields = [k for k in individual[0] if k not in ('case', 'subset', 'method', 'meets_floor')]
        result[method] = {key: mean_valid([v[key] for v in individual]) for key in fields}
        result[method]['passes'] = sum(v['meets_floor'] for v in individual)
        result[method]['eligible_focus_cases'] = {
            label: sum(v[f'{label}_focus_nrmse'] is not None for v in individual) for label in REGIONS}
        result[method]['worst_delta_dB'] = min(v['delta_dB'] for v in individual)
    return result, csv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run
    comparison = load_rows(run/'compare_records.jsonl')
    focus = load_rows(run/'focus'/'records.jsonl')
    if set(focus) != set(comparison):
        raise ValueError('Focus and optimization must have identical case IDs.')
    for row in comparison.values():
        if len(row['methods']) != 6 or set(row['patterns']) != set(row['methods']):
            raise ValueError(f'Incomplete methods for {row["id"]}')
        for files in row['patterns'].values():
            if len(files) != 3:
                raise ValueError('Each method must export three phase patterns.')
            for name in files:
                with (run/name).open('rb') as stream:
                    header = stream.read(26)
                if header[24:26] != bytes((8, 2)):
                    raise ValueError(f'Expected RGB8 PNG: {name}')
                with Image.open(run/name) as image:
                    image.load()
                    if (image.format, image.mode, image.size) != ('PNG', 'RGB', (1920, 1080)):
                        raise ValueError(f'Unexpected pattern format {name}')
        for eye in focus[row['id']]['eyes'].values():
            if 'methods' in eye and (set(eye['methods']) != set(row['methods'])
                                     or eye['height'] != 1080 or eye['pupil_samples'] != 13):
                raise ValueError(f'Focus protocol mismatch for {row["id"]}')
    held = [r for r in comparison.values() if r['subset'] != 'train_tune']
    result, pairs = summarize(held, focus)
    with (run/'held_out_pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=pairs[0])
        writer.writeheader()
        writer.writerows(pairs)
    all_summary = {'held_out_count': len(held), 'held_out': result,
                   'train_verify': summarize([r for r in held if r['subset'] == 'train_verify'], focus)[0],
                   'val_test': summarize([r for r in held if r['subset'] != 'train_verify'], focus)[0],
                   'stereo_mean_coverage': mean(r['stereo_depth']['valid_fraction_of_oracle'] for r in held),
                   'stereo_mean_axial_mae_mm': mean_valid([r['stereo_depth']['axial_mae_mm'] for r in held])}
    (run/'summary.json').write_text(json.dumps(all_summary, indent=2, allow_nan=False))
    for name, values in result.items():
        print(name, json.dumps(values))


if __name__ == '__main__':
    main()
