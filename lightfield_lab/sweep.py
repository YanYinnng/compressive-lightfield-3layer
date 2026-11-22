"""Bounded multi-case sweep; reports both fitted and held-out neighbouring views."""
import argparse
import io
import json
from pathlib import Path, PurePosixPath
import re
import time
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import torch

from .core import (ROOT, UPSTREAM, CachedRenderer, EyeRealDataset, eye2world_pytroch,
                   make_model, png_values, refine, save_images, scores, scene_dict, synchronize)
from .scenes import analytic_case
from .exam_geometry import ExamRenderer
from .exam_adapter import network_initial


def public_cases(archive_path, count, height, width, device, min_distance=70, validation=False):
    """Pick evenly spaced pairs; never extract untrusted ZIP paths to disk."""
    pattern = re.compile(r'pair(\d+)_x([-+\d.e]+)_y([-+\d.e]+)_z([-+\d.e]+)_([01])\.(?:jpg|png)$', re.I)
    with zipfile.ZipFile(archive_path) as archive:
        pairs = {}
        for name in archive.namelist():
            match = pattern.search(PurePosixPath(name).name)
            if match and '__MACOSX' not in name:
                parent = str(PurePosixPath(name).parent)
                pairs.setdefault((parent, int(match[1])), {})[int(match[5])] = (name, match)
        complete = []
        for key, pair in sorted(pairs.items()):
            distance_range = re.search(r'_R_(\d+)_(\d+)_', key[0])
            if len(pair) == 2 and distance_range and (validation or int(distance_range[1]) >= min_distance):
                complete.append((key, pair))
        chosen = np.linspace(0, len(complete)-1, min(count, len(complete)), dtype=int) if complete else []
        for number in chosen:
            (parent, pair_id), pair = complete[number]
            scene = next((part for part in reversed(PurePosixPath(parent).parts) if part in scene_dict), None)
            if scene is None and validation:
                scale = re.search(r'_scale_([\d.]+)_', parent)
                if scale is None:
                    raise ValueError('Validation folder is missing its generation scale.')
                # These settings are explicit in data_preparation_uco3d.py.
                config = {'scale_physical2world': float(scale[1]), 'thickness': 6,
                          'vertical': 'y', 'ground_coefficient': -.5, 'orientation': 'xoy'}
                scene = PurePosixPath(parent).parent.name
            elif scene is not None:
                config = scene_dict[scene]
            else:
                raise ValueError(f'No verified scene geometry for {parent}; refusing to guess.')
            images, views = [], []
            for eye in range(2):
                name, match = pair[eye]
                image = Image.open(io.BytesIO(archive.read(name))).convert('RGB')
                image = image.resize((width, height), Image.Resampling.BILINEAR)
                images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()/255)
                position = torch.tensor([float(match[i]) for i in (2, 3, 4)])
                views.append(eye2world_pytroch(config['vertical'], position))
            holder = SimpleNamespace(H=height, W=width, N_screen=3)
            panels = EyeRealDataset.get_screen_coords_world(holder,
                thickness=config['thickness'], scale_physical2world=config['scale_physical2world'],
                physical_width=config.get('physical_width'), ground=config.get('ground'),
                ground_coefficient=config.get('ground_coefficient'), orientation=config['orientation'],
                delta=torch.tensor([config.get('delta_x', 0.), config.get('delta_y', 0.), config.get('delta_z', 0.)]))
            yield {'name': f'public_{number:04d}_{scene}_pair{pair_id}', 'target': torch.stack(images).to(device),
                   'views': torch.stack(views).to(device), 'screens': panels.to(device),
                   'metadata': {'kind': 'public evaluation archive', 'scene': scene,
                       'zip_pair': [pair[i][0] for i in range(2)], 'scene_config': config,
                       'sampling': ('evenly spaced across subset pairs' if validation else
                                    f'evenly spaced across sorted complete pairs with distance-bin lower bound >= {min_distance}cm'),
                       'geometry_source': ('folder scale and data_preparation_uco3d.py; eval/evaluate.py has a conflicting 0.28 default, not silently substituted'
                                           if validation and scene not in scene_dict else 'config/scene_dict.py'),
                       'distance_scope_reason': 'Current cached-homography adapter rejects panels crossing the eye near plane; close ranges are not included.'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['public', 'validation', 'analytic'], default='analytic')
    parser.add_argument('--init', choices=['network', 'random'], default='network')
    parser.add_argument('--cases', type=int, default=4)
    parser.add_argument('--height', type=int, default=180)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--steps', type=int, nargs='+', default=[10, 30, 100])
    parser.add_argument('--anchors', type=float, nargs='+', default=[0., .01, .1])
    parser.add_argument('--neighbour-weights', type=float, nargs='*', default=[])
    parser.add_argument('--lr', type=float, default=.02)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--archive', type=Path, default=ROOT/'datasets/eval.zip')
    parser.add_argument('--min-distance', type=int, default=70)
    parser.add_argument('--texture-scale', type=float, default=1., help='Analytic-scene card texture frequency multiplier.')
    parser.add_argument('--geometry-profile', choices=['ta', 'legacy'], default='ta', help='Analytic scenes only; upstream public scenes keep their original geometry.')
    parser.add_argument('--out', type=Path, default=ROOT/'runs/exploration')
    args = parser.parse_args()
    args.renderer_profile = 'ta' if args.source == 'analytic' and args.geometry_profile == 'ta' else 'legacy'
    if args.cases < 1 or min(args.steps) < 1 or min(args.anchors) < 0:
        parser.error('Require positive case/step counts and nonnegative anchors.')
    if args.neighbour_weights and args.source != 'analytic':
        parser.error('Neighbour diagnostics currently require analytic scenes.')
    if args.texture_scale <= 0 or (args.texture_scale != 1 and args.source != 'analytic'):
        parser.error('A positive texture-scale override requires analytic scenes.')
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    args.out.mkdir(parents=True, exist_ok=True)
    model = (make_model(args.height, args.width, args.device, ROOT/'weights/pretrained_model.pth')
             if args.init == 'network' else None)
    cases = (public_cases(args.archive, args.cases, args.height, args.width, args.device, args.min_distance,
                         validation=args.source == 'validation') if args.source != 'analytic'
             else (analytic_case(seed, args.height, args.width, args.device, args.texture_scale, args.geometry_profile) for seed in range(args.cases)))
    records = []
    def write_report():
        report = {'scope': 'Development experiments, prototype renderer. Not official camp scores.',
                  'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  'selection': 'Each run selects its best quantized fitted-eye PSNR checkpoint; neighbour targets are evaluation-only.',
                  'cases': records}
        (args.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for case in cases:
        began = time.perf_counter()
        def make_renderer(views):
            return (ExamRenderer.from_centimetres(views, args.height, args.width) if args.renderer_profile == 'ta'
                    else CachedRenderer(views, case['screens'], args.height, args.width))
        renderer = make_renderer(case['views'])
        with torch.no_grad():
            raw = ((network_initial(model, case['target'], case['views'], case['screens']) if args.renderer_profile == 'ta' else
                    model(case['target'][None], case['views'][None], case['screens'][None])[0])
                   if model is not None else .1+.2*torch.rand(3, 3, args.height, args.width, device=args.device))
        initial = raw.clamp(0, 1).detach()
        foreground_mask = renderer.geometry_mask*(case['target'].amin(1, keepdim=True) < .95)
        neighbour_renderer = (make_renderer(case['neighbour_views'])
                              if 'neighbour_views' in case else None)
        folder = args.out/case['name']
        save_images(case['target'], folder/'targets', 'eye')
        if neighbour_renderer:
            save_images(case['neighbour_target'], folder/'neighbour_targets', 'eye')
        result = {'name': case['name'], 'metadata': {**case['metadata'], 'initialization': args.init}, 'variants': {}}
        def evaluate(name, patterns, trace=None):
            metric = scores(renderer, patterns, case['target'])
            metric['pattern_rmse_from_network'] = float((patterns-initial).square().mean().sqrt())
            if (foreground_mask.sum((1, 2, 3)) > 0).all():
                with torch.no_grad():
                    error = (renderer(patterns)-case['target']).square()*foreground_mask
                    mse = error.sum((1, 2, 3))/(foreground_mask.sum((1, 2, 3))*3)
                    metric['diagnostic_nonwhite_psnr'] = float((-10*mse.clamp_min(1e-12).log10()).mean())
                    metric['diagnostic_nonwhite_valid_fraction'] = (foreground_mask.sum()/renderer.geometry_mask.sum()).item()
            if neighbour_renderer:
                neighbour = scores(neighbour_renderer, patterns, case['neighbour_target'])
                metric['neighbour_psnr_1p5mm'] = float(np.mean(neighbour['psnr_each_eye'][:8]))
                metric['neighbour_psnr_6mm'] = float(np.mean(neighbour['psnr_each_eye'][8:]))
                metric['neighbour_each_view_psnr'] = neighbour['psnr_each_eye']
            if trace:
                metric.update(trace)
            result['variants'][name] = metric
            save_images(patterns, folder/name, 'layer')
            if name == 'network' or name == f'adam_{max(args.steps)}_anchor_0':
                save_images(renderer(patterns), folder/name, 'eye')
                if neighbour_renderer:
                    save_images(neighbour_renderer(patterns), folder/name, 'neighbour-eye')
            return metric
        evaluate('network', getattr(renderer, 'quantize_patterns', png_values)(initial))
        for steps in args.steps:
            for anchor in args.anchors:
                name = f'adam_{steps}_anchor_{anchor:g}'
                patterns, trace = refine(renderer, case['target'], initial, steps=steps, lr=args.lr,
                                          anchor_weight=anchor, eval_every=min(10, steps))
                evaluate(name, patterns, trace)
        if args.neighbour_weights:
            teacher_renderer = make_renderer(case['neighbour_views'][:8])
            for weight in args.neighbour_weights:
                name = f'consistency_{max(args.steps)}_weight_{weight:g}'
                patterns, trace = refine(renderer, case['target'], initial, steps=max(args.steps), lr=args.lr,
                    consistency_renderer=teacher_renderer, consistency_weight=weight)
                trace['teacher'] = 'legal network-initialization predictions at 1.5mm neighbours; no neighbouring GT'
                evaluate(name, patterns, trace)
        synchronize(args.device)
        result['seconds_case_including_all_variants_io'] = time.perf_counter()-began
        records.append(result)
        write_report()
        best = max(result['variants'], key=lambda k: result['variants'][k]['psnr_mean_eyes'])
        print(f'{case["name"]}: baseline={result["variants"]["network"]["psnr_mean_eyes"]:.2f}, '
              f'best={result["variants"][best]["psnr_mean_eyes"]:.2f} ({best})', flush=True)
    if not records:
        raise ValueError('No complete samples found.')
    print('Wrote', args.out/'summary.json', flush=True)


if __name__ == '__main__':
    main()
