"""T1-6: external process-to-exit timings, including complete packaging and validation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.exam_data import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lego', type=Path, default=ROOT/'datasets/lego_v1')
    parser.add_argument('--counts', type=int, nargs='+', default=[100, 300])
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, rows = read_manifest(args.lego/'all.json')
    records = []
    result_path = args.out/'timings.json'
    if result_path.exists():
        records = json.loads(result_path.read_text())
    jobs = [('control_full', 8, 30, .02, 'inference.py'), ('control_fast', 8, 30, .02, 'fast_inference.py')]
    jobs += [(f'n{n}_{label}', n, steps, lr, 'fast_inference.py') for n in args.counts
             for label, steps, lr in [('fast', 30, .02), ('balanced', 100, .08), ('precise', 200, .08)]]
    for label, count, steps, lr, entry in jobs:
        if any(r['label'] == label for r in records):
            continue
        if count > len(rows):
            raise ValueError('Count exceeds number of unique input samples.')
        index = args.out/f'input_{count}.json'
        selected = [{**r, **{key: os.path.relpath(manifest.parent/r[key], args.out) for key in ('left', 'right')}} for r in rows[:count]]
        index.write_text(json.dumps({'samples': selected}, indent=2))
        command = [sys.executable, '-B', str(ROOT/'solution'/entry), '--index', str(index), '--out', str(args.out/label),
                   '--group', label, '--steps', str(steps), '--lr', str(lr), '--device', 'cuda']
        tick = time.perf_counter()
        with (args.out/f'{label}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        seconds = time.perf_counter()-tick
        metrics = json.loads((args.out/label/'metrics.json').read_text())
        records.append({'label': label, 'count': count, 'steps': steps, 'lr': lr, 'entrypoint': entry,
                        'whole_process_seconds': seconds, 'within_600s': seconds <= 600,
                        'mean_psnr_dB': metrics['mean_psnr_dB'], 'reported_main_seconds': metrics['seconds_entry_main_through_packaging'],
                        'scope': 'process launch to exit, includes imports, code/weight packaging and format checker'})
        result_path.write_text(json.dumps(records, indent=2))
        print(json.dumps(records[-1]), flush=True)
    (args.out/'complete.json').write_text(json.dumps({'state': 'SUCCEEDED'}))


if __name__ == '__main__':
    main()
