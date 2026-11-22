"""Run a one-sample, reproducible baseline/refinement comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from .core import (ROOT, CachedRenderer, EyeRealNet, load_demo, make_model,
                   png_values, read_patterns, refine, save_images, scores,
                   synchronize, synthetic_geometry)
from .exam_adapter import network_initial
from .exam_geometry import ExamRenderer


def upstream_render(patterns, views, screens):
    from types import SimpleNamespace
    height, width = patterns.shape[-2:]
    corners = torch.tensor([[0, 0], [width, 0], [0, height], [width, height]]).float()
    holder = SimpleNamespace(view_tranform=EyeRealNet.view_tranform,
                             coord_pixel_init=corners[None].repeat(3, 1, 1))
    prediction, mask = EyeRealNet.aggregation(holder, patterns[None], views[None], screens[None], math.radians(40))
    return prediction[0], mask[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['demo', 'synthetic'], default='demo')
    parser.add_argument('--init', choices=['network', 'random', 'zero'], default='network')
    parser.add_argument('--height', type=int, default=180)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--phase-steps', type=int, default=15)
    parser.add_argument('--lr', type=float, default=.02)
    parser.add_argument('--anchor-weight', type=float, default=0.)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--geometry-profile', choices=['ta', 'legacy'], default='ta', help='Synthetic case only; public EyeReal demo keeps its original scene.')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'weights/pretrained_model.pth')
    parser.add_argument('--out', type=Path, default=ROOT/'runs/demo_180p')
    args = parser.parse_args()
    if min(args.height, args.width) < 32 or not 0 <= args.phase_steps <= args.steps:
        parser.error('Resolution must be at least 32 and phase-steps must be within [0,steps].')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable in this interpreter.')
    args.out.mkdir(parents=True, exist_ok=True)
    synchronize(args.device)
    total_started = time.perf_counter()
    if args.case == 'demo':
        target, views, screens, case_info = load_demo(args.height, args.width, args.device)
    else:
        views, screens = synthetic_geometry(args.height, args.width, args.device, oblique=True, profile=args.geometry_profile)
        target = None
        case_info = {'source': 'known layer patterns rendered by the cached prototype; numerical sanity case, not scene/depth evidence',
                     'physical_width_cm': float(screens[0, 1, 1]-screens[0, 0, 1]), 'adjacent_gap_cm': 6,
                     'geometry_profile': args.geometry_profile}
    synchronize(args.device)
    build_started = time.perf_counter()
    use_ta = args.case == 'synthetic' and args.geometry_profile == 'ta'
    renderer = ExamRenderer.from_centimetres(views, args.height, args.width) if use_ta else CachedRenderer(views, screens, args.height, args.width)
    synchronize(args.device)
    geometry_seconds = time.perf_counter()-build_started
    if target is None:
        y, x = torch.meshgrid(torch.linspace(0, 1, args.height, device=args.device),
                              torch.linspace(0, 1, args.width, device=args.device), indexing='ij')
        truth = torch.stack([torch.stack([.19+.12*torch.sin((i+1)*6*x+c)*torch.cos(5*y+c)
                                          for c in range(3)]) for i in range(3)])
        target = renderer(truth).detach()
        save_images(truth, args.out/'known_patterns', 'layer')
    save_images(target, args.out/'targets', 'eye')
    save_images(renderer.geometry_mask.expand(-1, 3, -1, -1), args.out/'targets', 'valid-mask')
    model_seconds = inference_seconds = 0.
    if args.init == 'network':
        if not args.checkpoint.is_file():
            raise FileNotFoundError('Run fetch_weights.py first, or use --init random.')
        started = time.perf_counter()
        model = make_model(args.height, args.width, args.device, args.checkpoint)
        synchronize(args.device)
        model_seconds = time.perf_counter()-started
        started = time.perf_counter()
        with torch.no_grad():
            raw = network_initial(model, target, views, screens) if use_ta else model(target[None], views[None], screens[None])[0]
        synchronize(args.device)
        inference_seconds = time.perf_counter()-started
        del model
    elif args.init == 'random':
        raw = .1+.2*torch.rand(3, 3, args.height, args.width, device=args.device)
    else:
        raw = torch.zeros(3, 3, args.height, args.width, device=args.device)
    initial = raw.detach().clamp(0, 1)
    with torch.no_grad():
        reference, reference_mask = upstream_render(initial, views, screens)
        parity = {'prediction_max_abs': float((reference-renderer(initial)).abs().max()),
                  'mask_max_abs': float((reference_mask-renderer.soft_mask).abs().max())}
        # Warm both paths before reporting renderer-only steady-state timings.
        for _ in range(2):
            upstream_render(initial, views, screens)
            renderer(initial)
        timings = {}
        for label, operation in [('upstream', lambda: upstream_render(initial, views, screens)),
                                 ('cached', lambda: renderer(initial))]:
            synchronize(args.device)
            started = time.perf_counter()
            for _ in range(5):
                operation()
            synchronize(args.device)
            timings[label+'_render_ms'] = (time.perf_counter()-started)*1000/5
    variants = {}

    def export_variant(name, patterns, optimization=None):
        started = time.perf_counter()
        folder = args.out/name
        save_images(patterns, folder, 'layer')
        reloaded = read_patterns(folder, torch.float64 if use_ta else torch.float32).to(args.device)
        metrics = scores(renderer, reloaded, target)
        prediction = renderer(reloaded).detach()
        save_images(prediction, folder, 'eye')
        save_images((prediction-target).abs()*renderer.geometry_mask*4, folder, 'error-x4')
        synchronize(args.device)
        record = {**metrics, 'seconds_export_reload_score_visuals': time.perf_counter()-started}
        if optimization is not None:
            record.update(optimization)
        variants[name] = record
        print(f'{name}: PNG mean-eye PSNR={metrics["psnr_mean_eyes"]:.4f} dB', flush=True)

    export_variant('initial_png', png_values(initial))
    for name, phase_count in [('adam', 0), ('phase_then_adam', args.phase_steps)]:
        result, record = refine(renderer, target, initial, steps=args.steps, lr=args.lr,
            phase_steps=phase_count, anchor_weight=args.anchor_weight)
        export_variant(name, result, record)
    synchronize(args.device)
    report = {'scope': 'One public demo or synthetic case. Prototype geometric-mask metrics; not official camp scores.',
        'upstream_commit': '833e24f10b91fabb499d3b3cc2ea137fbccd967d',
        'configuration': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'runtime': {'torch': torch.__version__, 'device': torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
            'seconds_cached_geometry_build': geometry_seconds, 'seconds_model_load': model_seconds,
            'seconds_first_network_inference': inference_seconds, **timings,
            'seconds_total_experiment': time.perf_counter()-total_started},
        'case': case_info, 'raw_pattern': {'min': float(raw.min()), 'max': float(raw.max()),
            'negative_fraction': float((raw < 0).float().mean()),
            'raw_metrics': scores(renderer, raw, target), 'clipped_float_metrics': scores(renderer, initial, target)},
        'renderer_parity': parity,
        'valid_geometry_fraction_each_eye': renderer.geometry_mask.mean((1, 2, 3)).cpu().tolist(),
        'variants': variants,
        'limitations': ['No official data or evaluator available.',
            'Raw and clipped metrics use this prototype renderer; parity was separately measured.',
            'Benchmark uses five warmed render-only calls; cache construction and image I/O are separate.',
            'Best checkpoint selection uses quantized candidates; total timing includes scoring.',
            'No depth-layering or off-axis bonus score is claimed.']}
    (args.out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('Wrote', args.out/'summary.json', flush=True)


if __name__ == '__main__':
    main()
