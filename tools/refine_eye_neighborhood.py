"""EyeReal followed by centre refinement and a guarded eye-neighbourhood tail."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.core import refine, scores
from lightfield_lab.depth_constraints import PhaseAllocation, stereo_depth
from lightfield_lab.depth_eval import depth_regions
from lightfield_lab.exam_data import read_manifest, read_rgb8
from lightfield_lab.exam_geometry import ExamRenderer, HEIGHT, WIDTH
from lightfield_lab.eye_neighborhood import AngularTargets, shifted_pose, warp_parallel
from tools.overnight_eval import Context, append, evaluation_subset, save_patterns, sync, to_images

FIT = ((-2, -2), (-2, 2), (2, -2), (2, 2))
HOLDOUT = ((-1, 0), (1, 0), (0, -1), (0, 1))
WEIGHTS = (.05, .2, .8)
STRIDE = 3
DROP_DB = .25
REGIONS = ('near', 'middle', 'far')
TAIL = dict(steps=24, lr=.003, lr_schedule=((13, .001), (21, .0003)),
            eval_every=4, iteration_dtype=torch.float32)


def centres(folder, row):
    rgb = np.stack([read_rgb8(folder/row[k]) for k in ('left', 'right')])
    poses = np.array([row['pose_l'], row['pose_r']])
    return rgb, poses


def cache_depth(rows, manifest, output):
    import cv2

    cv2.setNumThreads(4)
    for row in rows:
        rgb, poses = centres(manifest.parent, row)
        started = time.perf_counter()
        depth = stereo_depth(rgb, poses)
        np.savez_compressed(output/f"{row['id']}.npz", depth=depth,
                            seconds=time.perf_counter()-started)


def targets(folder, row, renderer, target, poses, depths, valid, offsets, measured):
    views, eyes, rgbs, masks = [], [], [], []
    for i, name in enumerate(('left', 'right')):
        samples = {tuple(v['offset_mm']): v for v in row.get('views', ())
                   if v['kind'] == 'subaperture' and v['eye'] == name}
        for offset in offsets:
            pose = shifted_pose(poses[i], offset)
            image, trust = warp_parallel(target[i], depths[i], valid[i], np.asarray(offset)*.001, STRIDE)
            if measured:
                source = samples[offset]
                if not np.allclose(pose, source['pose'], atol=1e-8):
                    raise ValueError(f"Pupil pose mismatch for {row['id']}")
                rgb = read_rgb8(folder/source['image'])[::STRIDE, ::STRIDE].copy()
                image = to_images([rgb], renderer.device)[0]
            views.append(pose)
            eyes.append(i)
            rgbs.append(image)
            masks.append(trust)
    return AngularTargets(renderer, views, eyes, torch.stack(rgbs), target, torch.stack(masks), STRIDE)


def guarded_refine(renderer, image, initial, floor, loss, weight, **settings):
    @torch.no_grad()
    def rank(patterns, psnr):
        if psnr >= floor-1e-6:
            return True, -float(loss(patterns)), psnr
        return False, psnr, 0.

    return refine(renderer, image, initial, pattern_loss=lambda p: weight*loss(p),
                  checkpoint_rank=rank, **settings)


def compare_case(context, manifest, row, out, depth_dir, frozen):
    device, folder = context.device, manifest.parent
    as_tensor = lambda a: torch.as_tensor(np.stack(a), device=device)
    started = time.perf_counter()
    rgb, poses = centres(folder, row)
    image = to_images(rgb, device)
    renderer = ExamRenderer(poses, HEIGHT, WIDTH, device, geometry_backend='torch')
    initial = context.initial(image, poses)
    baseline, _ = refine(renderer, image, initial, steps=100, lr=.08, iteration_dtype=torch.float32)
    baseline_seconds = time.perf_counter()-started
    baseline_psnr = scores(renderer, baseline, image)['psnr_mean_eyes']
    floor = baseline_psnr-DROP_DB
    record = {'id': row['id'], 'subset': evaluation_subset(row), 'floor_dB': floor,
              'methods': {}, 'patterns': {}, 'target_support': {}}

    depths, alphas = [], []
    for key in ('depth_l', 'depth_r'):
        with np.load(folder/row[key]) as arrays:
            depths.append(arrays['depth_m'].astype(np.float32))
            alphas.append(arrays['alpha_u8'])
    oracle = as_tensor(depths)
    valid = (as_tensor(alphas) >= 128) & torch.isfinite(oracle) & (oracle > .25) & (oracle < 2.)
    holdout = targets(folder, row, renderer, image, poses, oracle, valid, HOLDOUT, measured=True)
    depth_data = [depth_regions(d, a, p) for d, a, p in zip(depths, alphas, poses)]
    phase = PhaseAllocation(renderer, image, as_tensor([x for _, x in depth_data]).float(), valid)
    rois = {name: as_tensor([m[name] for m, _ in depth_data]) for name in REGIONS}

    def save(name, patterns, extra_seconds, step):
        psnr = scores(renderer, patterns, image)['psnr_mean_eyes']
        with torch.no_grad():
            nrmse = float((holdout.loss(patterns)/holdout.signal.clamp_min(1e-12)).sqrt())
        record['methods'][name] = {'psnr_dB': psnr, 'delta_dB': psnr-baseline_psnr,
            'angular_holdout_nrmse': nrmse, 'selected_tail_step': step,
            'allocation_mse_oracle': float(phase.loss(patterns)),
            'contrast_matrix_oracle': phase.contrast_matrix(patterns, rois),
            'seconds_incl_baseline_and_depth': baseline_seconds+extra_seconds}
        if frozen is not None:
            record['patterns'][name] = save_patterns(out/'patterns'/row['id'], name, patterns, out)

    save('adam100', baseline, 0., 0)
    if frozen is not None:
        started = time.perf_counter()
        control, trace = refine(renderer, image, baseline, **TAIL)
        save('center_tail', control, time.perf_counter()-started, trace['selected_step'])

    for source in ('oracle', 'stereo'):
        started = time.perf_counter()
        stereo_seconds = 0.
        if source == 'oracle':
            depth, trusted = oracle, valid
        else:
            with np.load(depth_dir/f"{row['id']}.npz") as arrays:
                depth = torch.from_numpy(arrays['depth']).to(device)
                stereo_seconds = float(arrays['seconds'])
            trusted = torch.isfinite(depth) & (depth > .25) & (depth < 2.)
        prior = targets(folder, row, renderer, image, poses, depth, trusted, FIT, measured=source == 'oracle')
        record['target_support'][source] = float(prior.mask.float().mean())
        sync()
        setup_seconds = time.perf_counter()-started+stereo_seconds
        for weight in WEIGHTS if frozen is None else (frozen[source],):
            started = time.perf_counter()
            result, trace = guarded_refine(renderer, image, baseline, floor, prior.loss, weight, **TAIL)
            elapsed = time.perf_counter()-started
            if scores(renderer, result, image)['psnr_mean_eyes'] < floor-1e-5:
                raise RuntimeError('Centre PSNR guard failed.')
            name = f'{source}_{weight:g}' if frozen is None else f'{source}_tail'
            save(name, result, setup_seconds+elapsed, trace['selected_step'])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--depth-dir', type=Path, required=True)
    parser.add_argument('--ids', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT/'weights/pretrained_model.pth')
    parser.add_argument('--stage', choices=('cache-depth', 'tune', 'compare'), required=True)
    args = parser.parse_args()
    manifest, rows = read_manifest(args.gt/'eyebox.json')
    identifiers = set(json.loads(args.ids.read_text()))
    rows = [r for r in rows if r['id'] in identifiers]
    if {r['id'] for r in rows} != identifiers:
        raise ValueError('Some requested IDs are absent from the truth manifest.')
    if args.stage == 'cache-depth':
        args.depth_dir.mkdir(parents=True, exist_ok=True)
        cache_depth(rows, manifest, args.depth_dir)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    config = args.out/'selected_weights.json'
    frozen = json.loads(config.read_text()) if args.stage == 'compare' else None
    if frozen is None:
        rows = [r for r in rows if evaluation_subset(r) == 'train_tune']
    output = args.out/f'{args.stage}_records.jsonl'
    records = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    finished = {r['id'] for r in records}
    context = Context(args)
    for row in rows:
        if row['id'] in finished:
            continue
        record = compare_case(context, manifest, row, args.out, args.depth_dir, frozen)
        append(output, record)
        records.append(record)
        print(row['id'], {k: [round(v['delta_dB'], 3), round(v['angular_holdout_nrmse'], 3),
                             v['selected_tail_step']] for k, v in record['methods'].items()}, flush=True)
    if frozen is None:
        config.write_text(json.dumps({source: min(WEIGHTS, key=lambda w: np.mean([
            r['methods'][f'{source}_{w:g}']['angular_holdout_nrmse'] for r in records]))
            for source in ('oracle', 'stereo')}, indent=2))


if __name__ == '__main__':
    main()
