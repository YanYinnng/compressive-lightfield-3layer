"""Evaluate saved analytic-scene PNGs; never optimize or regenerate patterns."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .core import ROOT, CachedRenderer, read_patterns, save_images, scores, synchronize
from .focus import FocusGeometry, focal_stack, focal_stack_many, focus_metrics, roi_masks
from .scenes import analytic_case, camera_rays, render_scene_rays
from .exam_geometry import ExamRenderer


def summarize(records):
    summary = {}
    for diameter in sorted({row['diameter_mm'] for case in records for row in case['eyes']}):
        rows = [row for case in records for row in case['eyes'] if row['diameter_mm'] == diameter]
        methods = rows[0]['variants']
        aggregate = {}
        for method in methods:
            values = [row['variants'][method] for row in rows]
            peaks = [roi['peak_error_diopters'] for value in values for label, roi in value['rois'].items()
                     if label != 'background' and roi['peak_error_diopters'] is not None
                     and not roi['gt_peak_at_scan_boundary']]
            card_rois = [roi for value in values for label, roi in value['rois'].items() if label != 'background']
            responses = [roi for roi in card_rois if roi.get('focus_response_nrmse') is not None]
            aggregate[method] = {
                'stack_psnr_mean_dB': float(np.mean([v['stack_psnr_mean_dB'] for v in values])),
                'stack_mse_mean': float(np.mean([v['stack_mse_mean'] for v in values])),
                'card_roi_stack_psnr_mean_dB': float(np.mean([v['stack_psnr_mean_dB'] for v in card_rois])),
                'peak_error_diopters_mean': float(np.mean(peaks)) if peaks else None,
                'identifiable_card_peaks': len(peaks), 'total_card_rois': len(card_rois),
                'gt_card_focus_signal_rms_mean': float(np.mean([v['gt_focus_signal_rms'] for v in card_rois])),
                'valid_fraction_mean': float(np.mean([v['valid_fraction'] for v in values])),
                'focus_response_nrmse_mean': float(np.mean([v['focus_response_nrmse'] for v in responses])) if responses else None,
                'focus_response_gain_mean': float(np.mean([v['focus_response_gain'] for v in responses])) if responses else None,
                'focus_response_identifiable_card_rois': len(responses),
            }
        summary[str(diameter)] = aggregate
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'runs/analytic_consistency_sweep')
    parser.add_argument('--out', type=Path, default=ROOT/'runs/focus_evaluation')
    parser.add_argument('--variants', nargs='+', default=['network', 'adam_30_anchor_0', 'adam_100_anchor_0',
                        'adam_100_anchor_0.1', 'consistency_100_weight_0.1', 'consistency_100_weight_1'])
    parser.add_argument('--cases', type=int, default=4)
    parser.add_argument('--diameters', type=float, nargs='+', default=[2., 4., 6.])
    parser.add_argument('--rings', type=int, default=3)
    parser.add_argument('--focus-count', type=int, default=25)
    parser.add_argument('--focus-span-cm', type=float, default=12.)
    parser.add_argument('--height', type=int, help='Evaluation image height; fixed pattern resolution is unchanged.')
    parser.add_argument('--jitter-mm', type=float, default=0., help='Fixed-pattern pupil-centre displacement along local camera X.')
    parser.add_argument('--signal-threshold', type=float, default=1/255)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.cases < 1 or args.focus_count < 3 or args.focus_span_cm <= 0:
        parser.error('Require positive cases/span and at least three focal planes.')
    torch.set_num_threads(4)
    source = json.loads((args.input/'summary.json').read_text())
    if source['configuration']['source'] != 'analytic':
        parser.error('Independent 3D ground truth is available only for analytic scenes.')
    args.out.mkdir(parents=True, exist_ok=True)
    config = source['configuration']
    texture_scale = config.get('texture_scale', 1.)
    geometry_profile = config.get('geometry_profile', 'legacy')
    renderer_profile = config.get('renderer_profile', 'legacy')
    height = args.height or config['height']
    width = round(height*config['width']/config['height'])
    records = []
    started = time.perf_counter()
    def write_report():
        report = {'scope': 'Frozen saved 8-bit patterns; geometric-optics pupil integration, not official depth scores.',
            'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'protocol': {'units': 'cm in world geometry, mm in pupil diameter, diopters = 100/distance_cm',
                'focal_planes': 'perpendicular to each reference camera optical axis',
                'quadrature': '6*rings^2 equal-area annular samples, equal weights',
                'truth': 'independent opaque 3D ray tracer; no LCD forward model', 'texture_scale': texture_scale,
                'geometry_profile': geometry_profile, 'native_center_renderer': renderer_profile,
                'mask': 'all-pupil LCD validity intersected over all focal planes, identical for every method',
                'roi': 'reference-camera visible surface IDs, eroded by 2 evaluation pixels',
                'peak_gate': 'GT focal-stack variation RMS >= signal_threshold; GT boundary peaks excluded from aggregate',
                'response_metric': 'subtract each pixel mean over focus before comparing; NRMSE=1 for a focus-invariant output',
                'patterns_frozen': True, 'pattern_resolution': [config['height'], config['width']],
                'evaluation_resolution': [height, width], 'optimization_performed': False},
            'cases': records, 'aggregate': summarize(records) if records else {},
            'seconds_total_so_far': time.perf_counter()-started}
        (args.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for saved in source['cases'][:args.cases]:
        seed = saved['metadata']['seed']
        case = analytic_case(seed, height, width, args.device, texture_scale, geometry_profile)
        patterns = {name: read_patterns(args.input/saved['name']/name, torch.float64 if renderer_profile == 'ta' else torch.float32).to(args.device) for name in args.variants}
        # Independently verify that geometry/scene refactoring preserves the old fitted-eye scores.
        native = analytic_case(seed, config['height'], config['width'], args.device, texture_scale, geometry_profile)
        native_renderer = (ExamRenderer.from_centimetres(native['views'], config['height'], config['width'])
                           if renderer_profile == 'ta' else
                           CachedRenderer(native['views'], native['screens'], config['height'], config['width']))
        centers = {}
        for name, pattern in patterns.items():
            value = scores(native_renderer, pattern, native['target'])['psnr_mean_eyes']
            expected = saved['variants'][name]['psnr_mean_eyes']
            if abs(value-expected) > .001:
                raise RuntimeError(f'Saved-pattern score changed: {saved["name"]}/{name}: {value} vs {expected}')
            centers[name] = {'psnr_dB': value, 'difference_from_saved_dB': value-expected}
        row = {'name': saved['name'], 'seed': seed, 'native_center_scores': centers, 'eyes': []}
        for eye, original_view in enumerate(case['views']):
            view = original_view.clone()
            view[:3, 3] += args.jitter_mm/10*view[:3, 0]
            origins, directions = camera_rays(view[None], height, width)
            sharp, ids = render_scene_rays(origins, directions, seed, return_ids=True, texture_scale=texture_scale)
            base_distance = float((view[:3, 3]*view[:3, 2]).sum())
            distances = (base_distance+torch.linspace(-args.focus_span_cm, args.focus_span_cm, args.focus_count)).tolist()
            for diameter in args.diameters:
                began = time.perf_counter()
                geometry = FocusGeometry(view, height, width, diameter, args.rings)
                truth, _ = focal_stack(geometry, distances, seed=seed, texture_scale=texture_scale)
                eye_row = {'eye': eye, 'diameter_mm': diameter, 'pupil_samples': len(geometry.origins),
                           'distances_cm': distances, 'variants': {}}
                predictions, masks = focal_stack_many(geometry, distances, case['screens'], torch.stack(list(patterns.values())))
                valid = masks.all(0).float()
                rois = roi_masks(ids[0], valid)
                if valid.sum() == 0:
                    raise RuntimeError('Empty common focus mask.')
                for method, prediction in zip(patterns, predictions):
                    metric = focus_metrics(prediction, truth, sharp[0], valid, rois, distances, args.signal_threshold)
                    eye_row['variants'][method] = metric
                    if eye == 0 and diameter == args.diameters[0]:
                        indices = [0, args.focus_count//2, args.focus_count-1]
                        folder = args.out/saved['name']/method
                        save_images(prediction[indices], folder, 'focus-end-centre-end')
                if eye == 0 and diameter == args.diameters[0]:
                    save_images(truth[[0, args.focus_count//2, args.focus_count-1]], args.out/saved['name']/'truth', 'focus-end-centre-end')
                    save_images(valid[None].expand(1, 3, -1, -1), args.out/saved['name'], 'valid')
                synchronize(args.device)
                eye_row['seconds_evaluate_all_variants'] = time.perf_counter()-began
                row['eyes'].append(eye_row)
                print(f'{saved["name"]}, eye {eye}, {diameter:g}mm: '+', '.join(
                    f'{name}={v["stack_psnr_mean_dB"]:.3f}' for name, v in eye_row['variants'].items()), flush=True)
                del predictions, truth
        records.append(row)
        write_report()
    print('Wrote', args.out/'summary.json', flush=True)


if __name__ == '__main__':
    main()
