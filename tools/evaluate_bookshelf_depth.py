"""Run frozen Lego candidates on official RGB-only Bookshelf inputs."""
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
from lightfield_lab.depth_constraints import PhaseAllocation
from lightfield_lab.depth_eval import depth_regions
from lightfield_lab.exam_data import read_manifest
from lightfield_lab.exam_geometry import ExamRenderer, HEIGHT, WIDTH
from tools.compare_depth_constraints import contact_sheet
from tools.overnight_eval import Context, append, save_patterns, sync, to_images
from tools.refine_eye_neighborhood import DROP_DB, FIT, HOLDOUT, REGIONS, TAIL, centres, guarded_refine, targets


def run_case(context, manifest, row, depth_dir, output, weight, preview):
    device = context.device
    started = time.perf_counter()
    rgb, poses = centres(manifest.parent, row)
    image = to_images(rgb, device)
    renderer = ExamRenderer(poses, HEIGHT, WIDTH, device, geometry_backend='torch')
    initial = context.initial(image, poses)
    base_settings = dict(steps=100, lr=.08, iteration_dtype=torch.float32)
    baseline, _ = refine(renderer, image, initial, **base_settings)
    baseline_seconds = time.perf_counter()-started
    floor = scores(renderer, baseline, image)['psnr_mean_eyes']-DROP_DB

    started = time.perf_counter()
    with np.load(depth_dir/f"{row['id']}.npz") as data:
        depths = data['depth']
        cpu_seconds = float(data['seconds'])
    valid = np.isfinite(depths)&(depths > .25)&(depths < 2.)
    geometry = [depth_regions(d, v.astype(np.uint8)*255, p) for d, v, p in zip(depths, valid, poses)]
    tensor = lambda a: torch.as_tensor(np.stack(a), device=device)
    depth, trusted = tensor(depths), tensor(valid)
    phase = PhaseAllocation(renderer, image, tensor([x for _, x in geometry]).float(), trusted)
    rois = {name: tensor([m[name] for m, _ in geometry]) for name in REGIONS}
    sync()
    depth_seconds = time.perf_counter()-started+cpu_seconds
    started = time.perf_counter()
    fit = targets(manifest.parent, row, renderer, image, poses, depth, trusted, FIT, measured=False)
    sync()
    fit_seconds = time.perf_counter()-started
    hold = targets(manifest.parent, row, renderer, image, poses, depth, trusted, HOLDOUT, measured=False)
    record = {'id': row['id'], 'floor_dB': floor, 'depth_proxy_fraction': float(phase.trusted.float().mean()),
              'fit_support': float(fit.mask.float().mean()), 'methods': {}}
    files = {}

    def save(name, patterns, seconds, step):
        psnr = scores(renderer, patterns, image)['psnr_mean_eyes']
        with torch.no_grad():
            angular = float((hold.loss(patterns)/hold.signal).sqrt()) if float(hold.signal) > 1e-10 else None
        record['methods'][name] = {'psnr_dB': psnr, 'meets_floor': psnr >= floor-1e-5,
            'allocation_mse_stereo_proxy': float(phase.loss(patterns)) if bool(phase.trusted.any()) else None,
            'angular_nrmse_stereo_proxy': angular, 'step': step, 'seconds': seconds,
            'contrast_matrix_stereo_proxy': phase.contrast_matrix(patterns, rois)}
        if preview:
            files[name] = save_patterns(output/'patterns'/row['id'], name, patterns, output)

    save('adam100', baseline, baseline_seconds, 0)
    started = time.perf_counter()
    control, trace = refine(renderer, image, baseline, **TAIL)
    save('center_tail', control, baseline_seconds+time.perf_counter()-started, trace['selected_step'])
    started = time.perf_counter()
    angular, trace = guarded_refine(renderer, image, baseline, floor, fit.loss, weight, **TAIL)
    save('stereo_angular', angular, baseline_seconds+depth_seconds+fit_seconds+time.perf_counter()-started,
         trace['selected_step'])
    if not record['methods']['stereo_angular']['meets_floor']:
        raise RuntimeError('Angular selection broke the PSNR guard.')
    started = time.perf_counter()
    constrained, trace = guarded_refine(renderer, image, initial, floor, phase.loss, .001, **base_settings)
    save('stereo_phase', constrained, baseline_seconds+depth_seconds+time.perf_counter()-started,
         trace['selected_step'])
    if preview:
        contact_sheet(output, row['id'], files)
        record['patterns'] = files
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--depth-dir', type=Path, required=True)
    parser.add_argument('--ids', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT/'weights/pretrained_model.pth')
    args = parser.parse_args()
    manifest, rows = read_manifest(args.index)
    ids = json.loads(args.ids.read_text())
    rows_by_id = {r['id']: r for r in rows}
    weight = json.loads(args.selection.read_text())['stereo']
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out/'records.jsonl'
    finished = {json.loads(s)['id'] for s in path.read_text().splitlines()} if path.exists() else set()
    context = Context(args)
    for i, identifier in enumerate(ids):
        if identifier in finished:
            continue
        result = run_case(context, manifest, rows_by_id[identifier], args.depth_dir, args.out, weight,
                          i in (0, len(ids)//2, len(ids)-1))
        append(path, result)
        print(identifier, {k: [round(v['psnr_dB'], 3), v['step']]
                           for k, v in result['methods'].items()}, flush=True)


if __name__ == '__main__':
    main()
