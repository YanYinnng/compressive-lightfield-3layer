"""Local native-resolution A1/A3 comparison; no hyperparameter selection."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.core import refine, scores, synchronize, save_images
from lightfield_lab.exam_adapter import network_initial
from lightfield_lab.core import make_model
from lightfield_lab.exam_data import read_manifest, read_rgb8
from lightfield_lab.exam_geometry import ExamRenderer, screen_corners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--index', type=Path, nargs='+', default=[ROOT/'testing/demo_input/test.json'])
    parser.add_argument('--max-samples', type=int, default=None, help='Evenly spaced rows per manifest; fixed before results')
    parser.add_argument('--skip-export', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    started = time.perf_counter()
    model = make_model(1080, 1920, device, ROOT/'weights/pretrained_model.pth')
    synchronize(device)
    result = {'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'torch': torch.__version__, 'steps': args.steps, 'lr': .02,
              'model_load_s': time.perf_counter()-started, 'records': [],
              'purpose': 'implementation comparison on public demo; not tuning or a generalization claim'}
    samples = []
    for index_path in args.index:
        index, rows = read_manifest(index_path)
        if args.max_samples is not None:
            if args.max_samples < 1:
                parser.error('--max-samples must be positive')
            rows = [rows[i] for i in np.linspace(0, len(rows)-1, min(args.max_samples, len(rows)), dtype=int)]
        samples.extend((index, row) for row in rows)
    result['manifests'] = [str(p) for p in args.index]
    result['max_samples_per_manifest'] = args.max_samples
    for index, row in samples:
        started = time.perf_counter()
        images = torch.from_numpy(np.stack([read_rgb8(index.parent/row[k]) for k in ('left', 'right')])).permute(0, 3, 1, 2).to(device).float()/255
        synchronize(device)
        read_s = time.perf_counter()-started
        poses = np.array([row['pose_l'], row['pose_r']])
        started = time.perf_counter()
        initial = network_initial(model, images, torch.from_numpy(poses).to(device), screen_corners(device))
        synchronize(device)
        network_s = time.perf_counter()-started
        reference_mask, reference_eyes = None, None
        for name, backend, dtype in [('baseline', 'numpy', None), ('fp32', 'numpy', torch.float32), ('fp32_torch_geometry', 'torch', torch.float32)]:
            started = time.perf_counter()
            renderer = ExamRenderer(poses, 1080, 1920, device, geometry_backend=backend)
            synchronize(device)
            geometry_s = time.perf_counter()-started
            with torch.no_grad():
                mask = renderer.geometry_mask.bool().cpu()
                eyes = renderer(renderer.quantize_patterns(initial)).cpu()
            if reference_mask is None:
                reference_mask, reference_eyes = mask, eyes
            mask_diff = int((mask != reference_mask).sum())
            intensity_diff = float((eyes-reference_eyes).abs().max())
            started = time.perf_counter()
            patterns, trace = refine(renderer, images, initial, steps=args.steps, lr=.02, iteration_dtype=dtype)
            synchronize(device)
            refine_s = time.perf_counter()-started
            started = time.perf_counter()
            score = scores(renderer, patterns, images)
            synchronize(device)
            score_s = time.perf_counter()-started
            started = time.perf_counter()
            if not args.skip_export:
                save_images(patterns, args.out/row['id']/name, 'layer')
            export_s = time.perf_counter()-started
            record = {'id': row['id'], 'mode': name, 'read_inputs_s': read_s, 'network_s': network_s,
                      'geometry_s': geometry_s, 'refine_including_setup_s': refine_s, 'final_score_s': score_s,
                      'export_s': export_s, 'mask_difference': mask_diff, 'max_intensity_difference': intensity_diff,
                      'score': score, 'trace': trace}
            if name == 'baseline':
                baseline_psnr = score['psnr_mean_eyes']
            record['psnr_delta_from_baseline_dB'] = score['psnr_mean_eyes']-baseline_psnr
            record['accepted'] = mask_diff == 0 and abs(record['psnr_delta_from_baseline_dB']) < .05 and intensity_diff <= 1e-7
            result['records'].append(record)
            (args.out/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(json.dumps({k: record[k] for k in ('id', 'mode', 'geometry_s', 'refine_including_setup_s', 'score')}), flush=True)
            del renderer, patterns, eyes, mask
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    result['accepted'] = all(r['accepted'] for r in result['records'])
    (args.out/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not result['accepted']:
        raise SystemExit('A1/A3 comparison acceptance failed; inspect comparison.json')


if __name__ == '__main__':
    main()
