"""Pattern-only pipeline: bounded threaded PNG I/O, complete code/weight packaging.

No previews are generated during this command. The recorded total includes input,
output completion, format checking, and packaging; process startup is measured by
the external benchmark. This entry point does not alter inference.py defaults.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import create_predictor


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--group', default='fast')
    parser.add_argument('--weights')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--lr', type=float, default=.02)
    parser.add_argument('--phase-steps', type=int, default=0)
    parser.add_argument('--quantized-tail', action='store_true')
    parser.add_argument('--anchor-weight', type=float, default=0.)
    parser.add_argument('--iteration-dtype', choices=['legacy', 'float32', 'float64'], default='float32')
    parser.add_argument('--geometry-backend', choices=['numpy', 'torch'], default='torch')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None)
    parser.add_argument('--time-limit', type=float, default=600.)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or not args.group or any(c in args.group for c in '/\\:') or args.group in ('.', '..'):
        parser.error('Require positive workers and a simple group name.')
    predictor = create_predictor(args.weights, steps=args.steps, lr=args.lr, phase_steps=args.phase_steps,
                                 anchor_weight=args.anchor_weight, iteration_dtype=args.iteration_dtype,
                                 geometry_backend=args.geometry_backend, device=args.device, diagnostics=False, quantized_tail=args.quantized_tail)
    from lightfield_lab.exam_data import read_manifest, read_rgb8
    from lightfield_lab.submission_support import finish_delivery
    index, rows = read_manifest(args.index)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Choose a new empty output.')
    (out/'patterns').mkdir(parents=True)
    predictor.configure_run(started+args.time_limit, len(rows), export_seconds=.4, final_seconds=15.)
    args.weights = str(predictor.weights_path)
    def read(row):
        return read_rgb8(index.parent/row['left']), read_rgb8(index.parent/row['right'])
    def write(path, array):
        temporary = path.with_suffix('.png.part')
        Image.fromarray(array).save(temporary, format='PNG', compress_level=1)
        temporary.replace(path)
    outputs, metrics = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        reads = {i: pool.submit(read, row) for i, row in enumerate(rows[:2])}
        pending = []
        for number, row in enumerate(rows):
            tick = time.perf_counter()
            left, right = reads.pop(number).result()
            if number+2 < len(rows):
                reads[number+2] = pool.submit(read, rows[number+2])
            layers = predictor(left, right, row['pose_l'], row['pose_r'])
            record = {'id': row['id']}
            for layer, array in enumerate(layers, 1):
                relative = f'patterns/sample_{number:06d}_layer{layer}.png'
                pending.append(pool.submit(write, out/relative, array))
                record[f'layer{layer}'] = relative
            if len(pending) >= 12:
                for future in pending[:6]:
                    future.result()
                pending = pending[6:]
            outputs.append(record)
            score = predictor.last_score
            metrics.append({'id': row['id'], 'psnr_mean_eyes_dB': score['psnr_mean_eyes'],
                            'psnr_each_eye_dB': score['psnr_each_eye'], 'mse_each_eye': score['mse_each_eye'],
                            'valid_pixels_each_eye': predictor.last_valid_pixels, 'solver': predictor.records[-1],
                            'seconds_sample_io_predict_render_score': time.perf_counter()-tick})
            print(f'[sample] {row["id"]}: {score["psnr_mean_eyes"]:.3f} dB', flush=True)
        for future in pending:
            future.result()
    (out/'patterns.json').write_text(json.dumps({'group': args.group, 'samples': outputs}, indent=2))
    source = Path(__file__).resolve().parent
    (out/'code').mkdir()
    for name in ('inference.py', 'check_submission.py', 'requirements.txt', 'LICENSE'):
        shutil.copy2(source/name, out/'code'/name)
    (out/'weights').mkdir()
    shutil.copy2(predictor.weights_path, out/'weights'/predictor.weights_path.name)
    subprocess.run([sys.executable, '-B', str(source/'check_submission.py'), '--index', str(index),
                    '--submission', str(out/'patterns.json')], check=True)
    args.reference_score, args.entrypoint = False, 'fast_inference.py'
    finish_delivery(out, predictor, metrics, args, started)


if __name__ == '__main__':
    main()
