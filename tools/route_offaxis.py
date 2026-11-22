"""Controlled intrinsic experiments with fixed extrinsics and independent Gaussian targets."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python_packages'))
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT/'cache/torch_extensions'))
os.environ.setdefault('MAX_JOBS', '4')
from lightfield_lab.core import refine, scores
from lightfield_lab.exam_adapter import network_initial
from lightfield_lab.exam_geometry import ExamRenderer, HEIGHT, WIDTH
from lightfield_lab.exam_data import read_manifest
from lightfield_lab.gaussian_scene import GaussianTargetRenderer
from lightfield_lab.intrinsics import standard_intrinsics, warp_to_standard
from tools.overnight_eval import Context, to_images, append, save_patterns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lego', type=Path, default=ROOT/'datasets/lego_v1')
    parser.add_argument('--weights', type=Path, default=ROOT/'weights/pretrained_model.pth')
    parser.add_argument('--asset', type=Path, default=ROOT/'assets/lego_bulldozer.ply')
    parser.add_argument('--per-split', type=int, default=2)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    context = Context(args)
    scene = GaussianTargetRenderer(args.asset)
    base = np.tile(standard_intrinsics(HEIGHT, WIDTH), (2, 1))
    rows = []
    for split in ('train', 'val', 'test'):
        _, group = read_manifest(args.lego/f'{split}.json')
        rows.extend(group[i] for i in np.linspace(0, len(group)-1, min(args.per_split, len(group)), dtype=int))
    (args.out/'config.json').write_text(json.dumps({'steps': args.steps, 'lr': .08,
        'poses_fixed': True, 'note': 'Controlled truth study, not official off-axis submission; no tuning on val/test.',
        'scene_transform': scene.metadata['scene_transform']}, indent=2))
    for row in rows:
        poses = np.array([row['pose_l'], row['pose_r']])
        for shift in (0., 60., 120., 240.):
            k = base.copy(); k[:, 2] += [shift, -shift]; k[:, 3] += .3*shift
            target_rgb, _ = scene.render(poses, intrinsics=k)
            target = to_images(list(target_rgb), context.device)
            true_renderer = ExamRenderer(poses, HEIGHT, WIDTH, context.device, geometry_backend='torch', intrinsics=k)
            variants = [('ignore_K', base, False), ('correct_K_direct_initial', k, False), ('correct_K_warped_initial', k, True)]
            for delta in (1., 5.):
                perturbed = k.copy(); perturbed[:, 2] += delta
                variants.append((f'cx_error_{delta:g}px', perturbed, True))
            perturbed = k.copy(); perturbed[:, :2] *= 1.01
            variants.append(('focal_error_1percent', perturbed, True))
            results = {}
            for name, solver_k, warp in variants:
                initial = network_initial(context.model, target, torch.from_numpy(poses).to(context.device), context.screens,
                                          intrinsics=solver_k if warp else None)
                renderer = ExamRenderer(poses, HEIGHT, WIDTH, context.device, geometry_backend='torch', intrinsics=solver_k)
                patterns, trace = refine(renderer, target, initial, steps=args.steps, lr=.08, iteration_dtype=torch.float32)
                results[name] = {'psnr_under_true_K': scores(true_renderer, patterns, target)['psnr_mean_eyes'],
                                 'solver_valid_fraction': renderer.geometry_mask.mean((1, 2, 3)).tolist(),
                                 'selected_step': trace['selected_step']}
                if name == 'correct_K_warped_initial':
                    results[name]['patterns'] = save_patterns(args.out/'patterns'/row['id'], f'shift{shift:g}', patterns, args.out)
            _, warp_support = warp_to_standard(target, k)
            record = {'id': row['id'], 'split': row['split'], 'shift_px': shift, 'intrinsics_fx_fy_cx_cy': k.tolist(),
                      'pose_l': row['pose_l'], 'pose_r': row['pose_r'],
                      'true_valid_fraction': true_renderer.geometry_mask.mean((1, 2, 3)).tolist(),
                      'network_warp_support_fraction': float(warp_support.float().mean()), 'variants': results}
            append(args.out/'records.jsonl', record)
            print(json.dumps({'id': row['id'], 'shift': shift, 'scores': {n: v['psnr_under_true_K'] for n, v in results.items()}}), flush=True)
    (args.out/'complete.json').write_text(json.dumps({'state': 'SUCCEEDED', 'samples': len(rows)}))


if __name__ == '__main__':
    main()
