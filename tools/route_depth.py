"""T1-7/8/9: evaluate frozen schedule candidates on Lego eye-box truth."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.core import refine, scores
from lightfield_lab.depth_eval import depth_regions, layer_depth_matrix
from lightfield_lab.exam_geometry import ExamRenderer, HEIGHT, WIDTH
from lightfield_lab.exam_data import read_manifest, read_rgb8
from lightfield_lab.focus import focus_metrics, roi_masks
from lightfield_lab.ray_eval import ta_grids, render_views, masked_psnr_views
from lightfield_lab.refocus import aperture_subset, refocus_views
from tools.overnight_eval import Context, to_images, save_patterns, append, evaluation_subset, summarize_eyebox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--choices', type=Path, required=True)
    parser.add_argument('--gt', type=Path, default=ROOT/'datasets/lego_gt_v1')
    parser.add_argument('--weights', type=Path, default=ROOT/'weights/pretrained_model.pth')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--focus-height', type=int, default=540)
    parser.add_argument('--chunk', type=int, default=4)
    parser.add_argument('--regularizers', action='store_true')
    parser.add_argument('--pupil-only', action='store_true')
    parser.add_argument('--subset', choices=['all', 'train_tune', 'confirmation'], default='all')
    parser.add_argument('--regularizer-choice', type=Path)
    args = parser.parse_args()
    choices = json.loads(args.choices.read_text())
    configs = choices['candidate_configs']
    names = list(dict.fromkeys(['baseline30', *choices['selected'].values()]))
    if args.regularizers:
        selected100 = configs[choices['selected']['100']]
        configs = dict(configs)
        configs.update({
            'anchor001': {**selected100, 'anchor_weight': .001},
            'anchor01': {**selected100, 'anchor_weight': .01},
            'smooth001': {**selected100, 'highpass_weight': .001},
            'quantized_tail': {**selected100, 'quantize_schedule': [[selected100['steps']-29, 0], [selected100['steps']-14, 1]]},
        })
        regularizer_names = ['anchor001', 'anchor01', 'smooth001', 'quantized_tail']
        if args.regularizer_choice:
            selected_name = json.loads(args.regularizer_choice.read_text())['selected']
            regularizer_names = [selected_name] if selected_name is not None else []
        names.extend(regularizer_names)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {'choices': choices, 'focus_height': args.focus_height, 'limit': args.limit,
              'regularizers': args.regularizers, 'pupil_only': args.pupil_only,
              'subset': args.subset, 'method_names': names,
              'pupil': 'regular 1mm lattice, radius 2mm circle, equal weights',
              'focus_truth_resampling': 'area-resized RGB8 truth; LCD rendered natively then area-resized',
              'metrics': 'proxies, not official depth grades'}
    config_path = args.out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit('Resume config mismatch')
    config_path.write_text(json.dumps(config, indent=2))
    manifest, rows = read_manifest(args.gt/'eyebox.json')
    if args.subset == 'train_tune':
        rows = [r for r in rows if evaluation_subset(r) == 'train_tune']
    elif args.subset == 'confirmation':
        rows = [r for r in rows if evaluation_subset(r) != 'train_tune']
    context = Context(args)
    record_path = args.out/'eyebox_records.jsonl'
    records = [json.loads(line) for line in record_path.read_text().splitlines()] if record_path.exists() else []
    finished = {r['id'] for r in records}
    for row in rows[:args.limit]:
        if row['id'] in finished:
            continue
        started = time.perf_counter()
        images = to_images([read_rgb8(manifest.parent/row[k]) for k in ('left', 'right')], context.device)
        poses = np.array([row['pose_l'], row['pose_r']])
        renderer = ExamRenderer(poses, HEIGHT, WIDTH, context.device, geometry_backend='torch')
        initial = context.initial(images, poses)
        methods = {'network_initial': renderer.quantize_patterns(initial)}
        for name in names:
            methods[name], _ = refine(renderer, images, initial, iteration_dtype=torch.float32, **configs[name])
        region_masks, depths = [], []
        for eye, key in enumerate(('depth_l', 'depth_r')):
            with np.load(manifest.parent/row[key]) as data:
                depth, alpha = data['depth_m'].astype(np.float64), data['alpha_u8']
            regions, _ = depth_regions(depth, alpha, poses[eye])
            region_masks.append(regions); depths.append((depth, alpha))
        rois = {label: torch.as_tensor(np.stack([r[label] for r in region_masks]), device=context.device)
                for label in ('near', 'middle', 'far')}
        structure = {name: layer_depth_matrix(patterns, renderer, rois, images) for name, patterns in methods.items()}
        centre = {name: scores(renderer, value, images)['psnr_mean_eyes'] for name, value in methods.items()}
        paths = {name: save_patterns(args.out/'patterns'/row['source_id'], name, value, args.out) for name, value in methods.items()}
        per_view = []
        # Full eye-box by default; a named pupil-only pilot reduces pilot cost.
        evaluated_views = row['views']
        if args.pupil_only:
            evaluated_views = [v for v in evaluated_views if v['kind'] == 'subaperture' and np.hypot(*v['offset_mm']) <= 2.0001]
        for start in range(0, len(evaluated_views), args.chunk):
            views = evaluated_views[start:start+args.chunk]
            target = to_images([read_rgb8(manifest.parent/v['image']) for v in views], context.device)
            grids, mask = ta_grids([v['pose'] for v in views], HEIGHT, WIDTH, context.device, torch.float64)
            values = {}
            with torch.no_grad():
                for name, patterns in methods.items():
                    psnr, fractions = masked_psnr_views(render_views(patterns, grids), target, mask)
                    values[name] = psnr.tolist()
            for j, view in enumerate(views):
                per_view.append({k: v for k, v in view.items() if k != 'pose'} |
                                {'valid_fraction': float(fractions[j]), 'psnr_dB': {name: values[name][j] for name in methods}})
            del grids, target
        focus = {}
        size = (args.focus_height, args.focus_height*16//9)
        for eye, eye_name in enumerate(('left', 'right')):
            sub = [v for v in row['views'] if v['kind'] == 'subaperture' and v['eye'] == eye_name]
            selected = aperture_subset([v['offset_mm'] for v in sub], 1., radius_mm=2., shape='circle')
            sub = [sub[i] for i in selected]
            offsets = np.array([v['offset_mm'] for v in sub])*.001
            truth_images = to_images([read_rgb8(manifest.parent/v['image']) for v in sub], context.device)
            truth_images = F.interpolate(truth_images, size=size, mode='area').double()
            depth, alpha = depths[eye]
            valid_depth = depth[(alpha >= 128)&np.isfinite(depth)&(depth > 0)]
            if len(valid_depth) < 16:
                focus[eye_name] = {'reason': 'insufficient foreground depth'}; continue
            lo, hi = np.quantile(valid_depth, [.05, .95])
            distances = np.linspace(max(.05, lo-.04), hi+.04, 13).tolist()
            truth_stack, truth_valid = refocus_views(truth_images, offsets, distances)
            sharp = F.interpolate(images[eye:eye+1], size=size, mode='area')[0].double()
            low_rois = {label: F.interpolate(rois[label][eye:eye+1, None].float(), size=size, mode='area')[0] > .99 for label in rois}
            eye_results = {}
            for name, patterns in methods.items():
                rendered, valid_masks = [], []
                with torch.no_grad():
                    for start in range(0, len(sub), args.chunk):
                        grids, mask = ta_grids([v['pose'] for v in sub[start:start+args.chunk]], HEIGHT, WIDTH, context.device, torch.float64)
                        rendered.append(F.interpolate(render_views(patterns, grids), size=size, mode='area'))
                        valid_masks.append(F.interpolate(mask[:, None].float(), size=size, mode='area') > .999)
                stack, valid = refocus_views(torch.cat(rendered), offsets, distances, torch.cat(valid_masks))
                common = truth_valid.all(0)&valid.all(0)
                filtered = {label: roi*common for label, roi in low_rois.items() if (roi*common).sum() >= 16}
                eye_results[name] = focus_metrics(stack, truth_stack, sharp, common, filtered, [100*d for d in distances])
            focus[eye_name] = {'distances_m': distances, 'pupil_samples': len(sub), 'methods': eye_results}
        record = {'id': row['id'], 'source_id': row['source_id'], 'split': row['split'], 'evaluation_subset': evaluation_subset(row),
                  'centre_psnr_dB': centre, 'patterns': paths, 'views': per_view, 'focus': focus,
                  'layer_depth': structure, 'seconds': time.perf_counter()-started}
        append(record_path, record); records.append(record)
        summarize_eyebox(record_path, args.out/'eyebox_summary.json')
        print(json.dumps({'id': row['id'], 'centre': centre, 'seconds': record['seconds']}), flush=True)
    (args.out/'complete.json').write_text(json.dumps({'state': 'SUCCEEDED', 'samples': len(records)}))


if __name__ == '__main__':
    main()
