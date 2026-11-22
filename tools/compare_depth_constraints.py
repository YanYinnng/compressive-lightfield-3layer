"""Compare network and geometry starts with oracle and stereo depth on fixed Lego cases."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.core import refine, scores
from lightfield_lab.depth_constraints import PhaseAllocation, stereo_depth
from lightfield_lab.depth_eval import depth_regions
from lightfield_lab.exam_data import read_manifest, read_rgb8
from lightfield_lab.exam_geometry import ExamRenderer, HEIGHT, WIDTH
from tools.overnight_eval import Context, append, evaluation_subset, save_patterns, sync, to_images


WEIGHTS = (.001, .005, .02, .08)
MAX_DROP_DB = .25
SOLVER = {'steps': 100, 'lr': .08, 'iteration_dtype': torch.float32, 'eval_every': 10}
REGIONS = ('near', 'middle', 'far')


def read_depth(manifest, row):
    depths, alphas = [], []
    for key in ('depth_l', 'depth_r'):
        with np.load(manifest.parent/row[key]) as arrays:
            depths.append(arrays['depth_m'].astype(np.float32))
            alphas.append(arrays['alpha_u8'])
    return depths, alphas


def contact_sheet(out, case, paths):
    w, h, left, top = 350, 197, 130, 31
    image = Image.new('RGB', (left+3*w, top+len(paths)*h), '#15191e')
    draw = ImageDraw.Draw(image)
    for layer in range(3):
        draw.text((left+layer*w+8, 9), f'L{layer+1}', fill='white')
    for index, (method, files) in enumerate(paths.items()):
        y = top+index*h
        draw.text((8, y+h//2), method, fill='white')
        for layer, file in enumerate(files):
            with Image.open(out/file) as pattern:
                image.paste(pattern.resize((w, h), Image.Resampling.LANCZOS), (left+layer*w, y))
    image.save(out/'patterns'/case/'comparison.png')


def run_case(context, manifest, row, out, chosen=None):
    tensor = lambda a: torch.as_tensor(np.stack(a), device=context.device)
    sync()
    began = time.perf_counter()
    rgb = np.stack([read_rgb8(manifest.parent/row[key]) for key in ('left', 'right')])
    target = to_images(rgb, context.device)
    poses = np.array([row['pose_l'], row['pose_r']])
    renderer = ExamRenderer(poses, HEIGHT, WIDTH, context.device, geometry_backend='torch')
    sync()
    setup_s = time.perf_counter()-began
    began = time.perf_counter()
    initial = context.initial(target, poses)
    sync()
    network_s = time.perf_counter()-began
    record = {'id': row['id'], 'split': row['split'], 'subset': evaluation_subset(row),
              'setup_s': setup_s, 'network_s': network_s, 'methods': {}, 'patterns': {}}
    began = time.perf_counter()
    baseline, trace = refine(renderer, target, initial, **SOLVER)
    baseline_s = time.perf_counter()-began
    floor = max(checkpoint['png_psnr'] for checkpoint in trace['history'])-MAX_DROP_DB
    record['psnr_floor_dB'] = floor

    def allocation(depths, alphas):
        data = [depth_regions(d, a, p) for d, a, p in zip(depths, alphas, poses)]
        x = tensor([values for _, values in data]).float()
        valid = tensor([(a >= 128)&np.isfinite(d)&(d > 0) for d, a in zip(depths, alphas)])
        return PhaseAllocation(renderer, target, x, valid), {
            name: tensor([m[name] for m, _ in data]) for name in REGIONS}

    began = time.perf_counter()
    depths, alphas = read_depth(manifest, row)
    oracle, rois = allocation(depths, alphas)
    sync()
    oracle_s = time.perf_counter()-began
    began = time.perf_counter()
    estimated = stereo_depth(rgb, poses)
    # Suppress far-away background mismatches. Bounds are relative to each camera.
    trust = [255*((d >= .25)&(d <= 2.)).astype(np.uint8) for d in estimated]
    stereo, _ = allocation(estimated, trust)
    sync()
    stereo_s = time.perf_counter()-began
    shared = (oracle.trusted & stereo.trusted).cpu().numpy()
    record['stereo_depth'] = {
        'valid_fraction_of_oracle': int(shared.sum())/max(1, int(oracle.trusted.sum())),
        'axial_mae_mm': float(np.abs(np.stack(depths)[shared]-np.stack(estimated)[shared]).mean()*1000)
        if shared.any() else None,
        'oracle_valid_fraction': float(oracle.trusted.float().mean()),
        'stereo_valid_fraction': float(stereo.trusted.float().mean()),
    }

    def save(name, patterns, seconds, selected_step, details=None):
        score = scores(renderer, patterns, target)
        result = {'psnr_dB': score['psnr_mean_eyes'], 'psnr_each_eye_dB': score['psnr_each_eye'],
                  'allocation_mse_oracle': float(oracle.loss(patterns)),
                  'contrast_matrix_oracle': oracle.contrast_matrix(patterns, rois),
                  'solve_s_incl_depth': setup_s+seconds, 'selected_step': selected_step,
                  'meets_floor': score['psnr_mean_eyes'] >= floor, **(details or {})}
        result['solve_s_with_reference'] = result['solve_s_incl_depth']+result.get('reference_adam_s', 0.)
        record['methods'][name] = result
        if chosen is not None:
            began = time.perf_counter()
            record['patterns'][name] = save_patterns(out/'patterns'/row['id'], name, patterns, out)
            result['solve_s_incl_depth_and_png'] = result['solve_s_incl_depth']+time.perf_counter()-began

    began = time.perf_counter()
    quantized_initial = renderer.quantize_patterns(initial)
    sync()
    save('network', quantized_initial, network_s+time.perf_counter()-began, 0)
    save('adam100', baseline, network_s+baseline_s, trace['selected_step'])
    for source, prior, depth_s in [('oracle', oracle, oracle_s), ('stereo', stereo, stereo_s)]:
        weights = [chosen[source]] if chosen is not None else WEIGHTS
        began = time.perf_counter()
        geometric = prior.geometric_initial() if chosen is not None else None
        sync()
        geometric_s = time.perf_counter()-began
        starts = [('net', initial, network_s)]
        if geometric is not None:
            starts.append(('geo', geometric, geometric_s))
        for start, seed, init_s in starts:
            for weight in weights:
                # The pure geometric method uses its own PSNR frontier. Keep
                # only RGB8 checkpoints within the current quality window.
                frontier = []
                peak_psnr = float('-inf')
                checkpoint_index = 0

                @torch.no_grad()
                def rank(patterns, psnr):
                    nonlocal peak_psnr, checkpoint_index
                    if start == 'geo':
                        peak_psnr = max(peak_psnr, psnr)
                        frontier[:] = [item for item in frontier if item[1] >= peak_psnr-MAX_DROP_DB]
                        if psnr >= peak_psnr-MAX_DROP_DB:
                            frontier.append((float(prior.loss(patterns)), psnr,
                                             (patterns*255).round().byte(), checkpoint_index*SOLVER['eval_every']))
                        checkpoint_index += 1
                        return psnr
                    if psnr >= floor:
                        return True, -float(prior.loss(patterns)), psnr
                    return False, psnr, 0.

                began = time.perf_counter()
                result, trace = refine(renderer, target, seed, **SOLVER,
                                       pattern_loss=lambda p: weight*prior.loss(p), checkpoint_rank=rank)
                selected_step = trace['selected_step']
                if frontier:
                    _, _, packed, selected_step = min(frontier, key=lambda x: (x[0], -x[1]))
                    result = packed.double()/255
                    sync()
                elapsed = time.perf_counter()-began
                name = f'{source}_{start}' if chosen is not None else f'{source}_{weight:g}'
                save(name, result, depth_s+init_s+elapsed, selected_step,
                     {'weight': weight, 'selection_floor_dB': floor if start == 'net' else peak_psnr-MAX_DROP_DB,
                      'reference_adam_s': baseline_s if start == 'net' else 0.})
    if record['patterns']:
        contact_sheet(out, row['id'], record['patterns'])
    return record


def main():
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT/'weights/pretrained_model.pth')
    parser.add_argument('--stage', choices=('tune', 'compare'), required=True)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, rows = read_manifest(args.gt/'eyebox.json')
    rows = [row for row in rows if (manifest.parent/row['depth_l']).is_file()]
    config = args.out/'selected_weights.json'
    if args.stage == 'tune':
        rows = [row for row in rows if evaluation_subset(row) == 'train_tune']
    chosen = json.loads(config.read_text()) if args.stage == 'compare' else None
    output = args.out/f'{args.stage}_records.jsonl'
    previous = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    finished = {r['id'] for r in previous}
    context = Context(args)
    for row in rows:
        if row['id'] not in finished:
            result = run_case(context, manifest, row, args.out, chosen)
            append(output, result)
            previous.append(result)
            print(row['id'], {k: round(v['psnr_dB'], 3) for k, v in result['methods'].items()}, flush=True)
    if args.stage == 'tune':
        def rank_weight(source, weight):
            methods = [r['methods'][f'{source}_{weight:g}'] for r in previous]
            return sum(m['meets_floor'] for m in methods), -np.mean([m['allocation_mse_oracle'] for m in methods])

        chosen = {source: max(WEIGHTS, key=lambda w: rank_weight(source, w)) for source in ('oracle', 'stereo')}
        config.write_text(json.dumps(chosen, indent=2))
        print('frozen weights', chosen, flush=True)


if __name__ == '__main__':
    main()
