"""Overnight non-training experiments on the evaluation GPU. No network weights are updated.

Stages (each resumable: one JSON line per finished sample, finished samples are skipped):
  offaxis - experiment A/B on lego_gt_v1/offaxis.json: solve with the true poses and with the
            base-model look-at poses from the same eye centres; score both under the true poses.
  eyebox  - neighbour-view PSNR on lego_gt_v1/eyebox.json for several solver settings; the
            exported RGB8 patterns are kept for later focal-stack analysis.
  sweep   - solver convergence on lego_v1 (+ TA demo): float32 Adam iterations, float64
            scoring of 8-bit checkpoints, learning-rate x step grid, random-init control subset.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from lightfield_lab.core import image_loss, make_model, refine, scores
from lightfield_lab.exam_adapter import network_initial
from lightfield_lab.exam_data import read_manifest, read_rgb8
from lightfield_lab.exam_geometry import HEIGHT, WIDTH, ExamRenderer, screen_corners
from lightfield_lab.ray_eval import masked_psnr_views, render_views, ta_grids

CHECKPOINTS = (10, 20, 30, 50, 75, 100, 150, 200, 300)


def sync():
    torch.cuda.synchronize()


def to_images(arrays, device):
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device).float()/255


def adam_fp32(renderer, target, initial, steps, lr, checkpoints=(), anchor_weight=0.):
    """float32 Adam on [0,1] patterns; float64 scoring of quantized checkpoints.

    Returns (best quantized patterns, best psnr, trace). Iteration seconds exclude scoring.
    """
    grid, mask = renderer.grid.float(), renderer.geometry_mask.float()
    eyes = renderer.eyes
    parameter = initial.detach().float().clamp(0, 1).clone().requires_grad_(True)
    anchor = parameter.detach().clone()
    optimizer = torch.optim.Adam([parameter], lr=lr)
    best = renderer.quantize_patterns(parameter)
    best_psnr = scores(renderer, best, target)['psnr_mean_eyes']
    trace, iteration_seconds = [{'step': 0, 'psnr': best_psnr, 'best': best_psnr, 'seconds': 0.}], 0.
    wanted = set(checkpoints) | {steps}
    for step in range(1, steps+1):
        sync(); started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        sources = parameter[:, None].expand(-1, eyes, -1, -1, -1).reshape(-1, 3, HEIGHT, WIDTH)
        sampled = torch.nn.functional.grid_sample(sources, grid, align_corners=False).reshape(3, eyes, 3, HEIGHT, WIDTH)
        loss = image_loss(torch.sin(((sampled[0]+sampled[1])+sampled[2])*math.pi/2).square(), target, mask)
        if anchor_weight:
            loss = loss+anchor_weight*(parameter-anchor).square().mean()
        loss.backward(); optimizer.step()
        with torch.no_grad():
            parameter.clamp_(0, 1)
        sync(); iteration_seconds += time.perf_counter()-started
        if step in wanted:
            candidate = renderer.quantize_patterns(parameter)
            psnr = scores(renderer, candidate, target)['psnr_mean_eyes']
            if psnr > best_psnr:
                best, best_psnr = candidate, psnr
            trace.append({'step': step, 'psnr': psnr, 'best': best_psnr, 'seconds': iteration_seconds})
    return best, best_psnr, trace


class Context:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda')
        torch.set_num_threads(4)
        started = time.perf_counter()
        self.model = make_model(HEIGHT, WIDTH, self.device, args.weights)
        self.screens = screen_corners(self.device)
        self.model_seconds = time.perf_counter()-started

    def initial(self, images, poses):
        return network_initial(self.model, images, torch.from_numpy(np.asarray(poses)).to(self.device), self.screens)


def done_ids(path):
    if not path.exists():
        return set()
    return {json.loads(line)['id'] for line in path.read_text().splitlines() if line.strip()}


def append(path, record):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, allow_nan=True)+'\n')


def save_patterns(folder, name, patterns, base):
    folder.mkdir(parents=True, exist_ok=True)
    arrays = torch.floor(patterns.clamp(0, 1)*255+.5).byte().permute(0, 2, 3, 1).cpu().numpy()
    paths = []
    for k, array in enumerate(arrays, 1):
        path = folder/f'{name}_layer{k}.png'
        Image.fromarray(array).save(path, compress_level=1)
        paths.append(str(path.relative_to(base)).replace('\\', '/'))
    return paths


def stage_offaxis(context, out):
    args = context.args
    manifest, rows = read_manifest(args.gt/'offaxis.json')
    records = out/'offaxis_records.jsonl'
    finished = done_ids(records)
    for row in rows:
        if row['id'] in finished:
            continue
        started = time.perf_counter()
        images = to_images([read_rgb8(manifest.parent/row[k]) for k in ['left', 'right']], context.device)
        true_poses = np.stack([row['pose_l'], row['pose_r']])
        base_poses = np.stack([row['pose_l_base'], row['pose_r_base']])
        true_renderer = ExamRenderer(true_poses, HEIGHT, WIDTH, context.device)
        record = {'id': row['id'], 'category': row['category'], 'params': row['params'],
                  'valid_fraction_each_eye': true_renderer.geometry_mask.mean((1, 2, 3)).tolist(), 'psnr_true_pose_dB': {}}
        for condition, poses in [('true', true_poses), ('base', base_poses)]:
            try:
                solver_renderer = true_renderer if condition == 'true' else ExamRenderer(poses, HEIGHT, WIDTH, context.device)
            except ValueError as error:
                record[f'{condition}_error'] = repr(error); continue
            initial = context.initial(images, poses)
            record['psnr_true_pose_dB'][f'init_{condition}'] = scores(true_renderer, true_renderer.quantize_patterns(initial), images)['psnr_mean_eyes']
            default, _ = refine(solver_renderer, images, initial, steps=30, lr=.02)
            record['psnr_true_pose_dB'][f'default30_{condition}'] = scores(true_renderer, default, images)['psnr_mean_eyes']
            longer, _, _ = adam_fp32(solver_renderer, images, initial, 100, .05)
            record['psnr_true_pose_dB'][f'adam100_lr05_{condition}'] = scores(true_renderer, longer, images)['psnr_mean_eyes']
        record['seconds'] = time.perf_counter()-started
        append(records, record)
        print(json.dumps({'stage': 'offaxis', 'id': row['id'], **record['psnr_true_pose_dB']}), flush=True)
    summarize(records, out/'offaxis_summary.json', group='category')


def stage_eyebox(context, out):
    args = context.args
    manifest, rows = read_manifest(args.gt/'eyebox.json')
    records = out/'eyebox_records.jsonl'
    finished = done_ids(records)
    for row in rows[:args.eyebox_limit]:
        if row['id'] in finished:
            continue
        started = time.perf_counter()
        images = to_images([read_rgb8(manifest.parent/row[k]) for k in ['left', 'right']], context.device)
        poses = np.stack([row['pose_l'], row['pose_r']])
        renderer = ExamRenderer(poses, HEIGHT, WIDTH, context.device)
        initial = context.initial(images, poses)
        methods = {'init': renderer.quantize_patterns(initial)}
        methods['default30'], _ = refine(renderer, images, initial, steps=30, lr=.02)
        methods['adam100_lr05'], _, _ = adam_fp32(renderer, images, initial, 100, .05)
        methods['adam300_lr05'], _, _ = adam_fp32(renderer, images, initial, 300, .05)
        methods['anchor100_lr05_w0.1'], _, _ = adam_fp32(renderer, images, initial, 100, .05, anchor_weight=.1)
        centre = {name: scores(renderer, value, images)['psnr_mean_eyes'] for name, value in methods.items()}
        folder = out/'eyebox_patterns'/row['source_id']
        pattern_paths = {name: save_patterns(folder, name, value, out) for name, value in methods.items()}
        views = row['views']
        per_view = []
        for start in range(0, len(views), args.chunk):
            chunk = views[start:start+args.chunk]
            target = to_images([read_rgb8(manifest.parent/v['image']) for v in chunk], context.device)
            grids, mask = ta_grids([np.array(v['pose']) for v in chunk], HEIGHT, WIDTH, context.device, torch.float64)
            results = {}
            for name, value in methods.items():
                with torch.no_grad():
                    psnr, fraction = masked_psnr_views(render_views(value, grids), target, mask)
                results[name] = psnr.tolist()
            for k, view in enumerate(chunk):
                per_view.append({key: view[key] for key in view if key not in ('pose',)} |
                                {'valid_fraction': float(fraction[k]), 'psnr_dB': {n: results[n][k] for n in methods}})
            del grids, target
        record = {'id': row['id'], 'source_id': row['source_id'], 'split': row['split'], 'centre_psnr_dB': centre,
                  'patterns': pattern_paths, 'views': per_view, 'seconds': time.perf_counter()-started}
        append(records, record)
        print(json.dumps({'stage': 'eyebox', 'id': row['id'], 'centre': centre, 'seconds': record['seconds']}), flush=True)
    summarize_eyebox(records, out/'eyebox_summary.json')


def stage_sweep(context, out):
    args = context.args
    sources = []
    lego_manifest, lego_rows = read_manifest(args.lego/'all.json')
    sources += [(lego_manifest, r) for r in lego_rows]
    demo_manifest, demo_rows = read_manifest(ROOT/'testing/demo_input/test.json')
    sources += [(demo_manifest, r) for r in demo_rows]
    records = out/'sweep_records.jsonl'
    finished = done_ids(records)
    generator = torch.Generator(device='cuda').manual_seed(2026)
    for number, (manifest, row) in enumerate(sources[:args.sweep_limit]):
        if row['id'] in finished:
            continue
        started = time.perf_counter()
        images = to_images([read_rgb8(manifest.parent/row[k]) for k in ['left', 'right']], context.device)
        poses = np.stack([row['pose_l'], row['pose_r']])
        sync(); t = time.perf_counter(); initial = context.initial(images, poses); sync(); network_s = time.perf_counter()-t
        t = time.perf_counter(); renderer = ExamRenderer(poses, HEIGHT, WIDTH, context.device); sync(); geometry_s = time.perf_counter()-t
        record = {'id': row['id'], 'split': row.get('split', 'demo'), 'sampling': row.get('sampling'),
                  'network_s': network_s, 'geometry_cache_s': geometry_s, 'runs': []}
        for lr in args.lrs:
            _, best, trace = adam_fp32(renderer, images, initial, max(CHECKPOINTS), lr, CHECKPOINTS)
            record['runs'].append({'init': 'network', 'lr': lr, 'trace': trace})
        if number % args.random_every == 0:
            random_init = .1+.2*torch.rand(initial.shape, generator=generator, device='cuda')
            _, best, trace = adam_fp32(renderer, images, random_init, max(CHECKPOINTS), .05, CHECKPOINTS)
            record['runs'].append({'init': 'random_0.1_0.3', 'lr': .05, 'trace': trace})
        record['seconds'] = time.perf_counter()-started
        append(records, record)
        print(json.dumps({'stage': 'sweep', 'n': number, 'id': row['id'], 'seconds': round(record['seconds'], 1),
                          'final': {f"{r['init']}@{r['lr']}": round(r['trace'][-1]['best'], 2) for r in record['runs']}}), flush=True)
    summarize_sweep(records, out/'sweep_summary.json')


def summarize(records, path, group):
    rows = [json.loads(line) for line in records.read_text().splitlines() if line.strip()] if records.exists() else []
    groups = {}
    for row in rows:
        groups.setdefault(row[group], []).append(row['psnr_true_pose_dB'])
    summary = {name: {key: float(np.mean([v[key] for v in values if key in v])) for key in sorted(set().union(*values))}
               | {'count': len(values)} for name, values in groups.items()}
    path.write_text(json.dumps({'mean_psnr_dB_by_group': summary, 'records': len(rows)}, indent=2))


def evaluation_subset(row):
    """Stable predeclared partition; independent of results and job completion order."""
    import hashlib
    split = row['split']
    if split != 'train':
        return split
    identifier = row.get('source_id', row['id'])
    return 'train_tune' if hashlib.sha256(identifier.encode()).digest()[0] % 2 == 0 else 'train_verify'


def summarize_eyebox(records, path):
    rows = [json.loads(line) for line in records.read_text().splitlines() if line.strip()] if records.exists() else []
    if not rows:
        return
    def group_summary(rows):
        methods = list(rows[0]['centre_psnr_dB'])
        buckets = {}
        for row in rows:
            for view in row['views']:
                if view['kind'] == 'subaperture':
                    radius = round(math.hypot(*view['offset_mm']), 3)
                    key = 'subaperture_r<=2mm' if radius <= 2.0001 else ('subaperture_r<=5mm' if radius <= 5.0001 else 'subaperture_r>5mm')
                    if radius == 0:
                        key = 'centre'
                elif view['kind'] == 'ring':
                    key = f"ring_{view['radius_mm']:.0f}mm"
                else:
                    key = 'baseline_between_eyes' if 0 < view['cm_from_left_toward_right'] < 6 else 'baseline_outside'
                buckets.setdefault(key, []).append(view['psnr_dB'])
        summary = {key: {m: float(np.nanmean([v[m] for v in values])) for m in methods} | {'views': len(values)}
                   for key, values in sorted(buckets.items())}
        return {'samples': len(rows), 'mean_psnr_dB_by_view_group': summary,
                'centre_psnr_dB_mean': {m: float(np.mean([r['centre_psnr_dB'][m] for r in rows])) for m in methods}}
    splits = sorted({r['split'] for r in rows})
    subsets = sorted({evaluation_subset(r) for r in rows})
    path.write_text(json.dumps({'samples': len(rows),
        'by_split': {s: group_summary([r for r in rows if r['split'] == s]) for s in splits},
        'by_evaluation_subset': {s: group_summary([r for r in rows if evaluation_subset(r) == s]) for s in subsets},
        'partition_rule': 'sha256(source_id or id), first byte even => train_tune; odd => train_verify',
        'selection_subset': 'train_tune'}, indent=2))


def summarize_sweep(records, path):
    """Curves per split, so that settings chosen on one subset can be confirmed on another.

    lego_v1 splits are by eye yaw: train within +-22 deg (no training is done, so it is simply the
    common-angle set), val 30..38 deg, test -38..-30 deg; 'demo' are the two TA inputs.
    """
    rows = [json.loads(line) for line in records.read_text().splitlines() if line.strip()] if records.exists() else []
    def curves_of(subset):
        curves = {}
        for row in subset:
            for run in row['runs']:
                key = f"{run['init']}_lr{run['lr']}"
                for point in run['trace']:
                    curves.setdefault(key, {}).setdefault(point['step'], []).append((point['best'], point['seconds']))
        return {key: {str(step): {'mean_best_psnr_dB': float(np.mean([p for p, _ in values])),
                                  'min_best_psnr_dB': float(np.min([p for p, _ in values])),
                                  'mean_iteration_seconds': float(np.mean([s for _, s in values])), 'samples': len(values)}
                      for step, values in sorted(points.items())} for key, points in curves.items()}
    splits = sorted({row['split'] for row in rows})
    summary = {split: curves_of([row for row in rows if row['split'] == split]) for split in splits}
    path.write_text(json.dumps({'samples': len(rows), 'samples_by_split': {s: sum(r['split'] == s for r in rows) for s in splits},
                                'curves_by_split': summary,
                                'curves_by_evaluation_subset': {s: curves_of([r for r in rows if evaluation_subset(r) == s])
                                    for s in sorted({evaluation_subset(r) for r in rows})},
                                'partition_rule': 'sha256(source_id or id), first byte even => train_tune; odd => train_verify',
                                'note': 'Choose settings on one subset and confirm on a disjoint one; do not pool splits for selection.',
                                'mean_network_s': float(np.mean([r['network_s'] for r in rows])) if rows else None,
                                'mean_geometry_cache_s': float(np.mean([r['geometry_cache_s'] for r in rows])) if rows else None}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stages', nargs='+', default=['offaxis', 'eyebox', 'sweep'], choices=['offaxis', 'eyebox', 'sweep'])
    parser.add_argument('--gt', type=Path, default=ROOT/'datasets/lego_gt_v1')
    parser.add_argument('--lego', type=Path, default=ROOT/'datasets/lego_v1')
    parser.add_argument('--weights', type=Path, default=ROOT/'weights/pretrained_model.pth')
    parser.add_argument('--lrs', type=float, nargs='+', default=[.02, .05, .08])
    parser.add_argument('--random-every', type=int, default=10)
    parser.add_argument('--eyebox-limit', type=int, default=None)
    parser.add_argument('--sweep-limit', type=int, default=None)
    parser.add_argument('--chunk', type=int, default=8)
    args = parser.parse_args()
    args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    context = Context(args)
    status = {'started_utc': datetime.now(timezone.utc).isoformat(), 'training_performed': False,
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'model_load_s': context.model_seconds, 'stages': {}}
    (args.out/'config.json').write_text(json.dumps(status, indent=2))
    failures = 0
    for name in args.stages:
        started = time.perf_counter()
        try:
            {'offaxis': stage_offaxis, 'eyebox': stage_eyebox, 'sweep': stage_sweep}[name](context, args.out)
            status['stages'][name] = {'state': 'SUCCEEDED'}
        except Exception:
            failures += 1
            status['stages'][name] = {'state': 'FAILED', 'error': traceback.format_exc()}
            print(status['stages'][name]['error'], flush=True)
        status['stages'][name]['seconds'] = time.perf_counter()-started
        torch.cuda.empty_cache()
        (args.out/'status.json').write_text(json.dumps(status, indent=2))
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (args.out/'status.json').write_text(json.dumps(status, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
