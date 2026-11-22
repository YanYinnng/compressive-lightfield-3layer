"""Validate metric shift-and-add on known depths and evaluate fixed LCD patterns."""
import argparse
import json
import math
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.refocus import refocus_views
from lightfield_lab.focus import focus_metrics
from lightfield_lab.exam_geometry import ExamRenderer
from lightfield_lab.core import refine, scores
from lightfield_lab.ray_eval import ta_grids, render_views


def known_depth_views(height=360, width=640, device='cpu'):
    """Independent ray/plane textured stripes, no LCD forward used for truth."""
    dtype = torch.float64
    offsets = torch.tensor([(x*.004, y*.004) for x in (-1, 0, 1) for y in (-1, 0, 1)], device=device, dtype=dtype)
    focal = width/(2*math.tan(math.radians(20)))
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype)+.5,
                          torch.arange(width, device=device, dtype=dtype)+.5, indexing='ij')
    nx, ny = (x-width/2)/focal, -(y-height/2)/focal
    images = torch.zeros(len(offsets), 3, height, width, device=device, dtype=dtype)
    rois = {}
    depths = {'near': .5, 'middle': .8, 'far': 1.2}
    for label, (lo, hi), depth in zip(depths, [(-.32, -.12), (-.1, .1), (.12, .32)], depths.values()):
        px = offsets[:, 0, None, None]+depth*nx
        py = offsets[:, 1, None, None]+depth*ny
        inside = (px/depth >= lo)&(px/depth <= hi)
        texture = .5+.3*torch.sin(px*450)*torch.cos(py*370)
        images = torch.where(inside[:, None], texture[:, None].expand(-1, 3, -1, -1), images)
        rois[label] = ((nx > lo+.035)&(nx < hi-.035)&(ny.abs() < .14))[None]
    return images, offsets, rois, depths


def direct_focus_truth(offsets, distances, height, width):
    """Independent analytic pupil rays through each focal plane, without image warps."""
    f = width/(2*math.tan(math.radians(20)))
    y, x = torch.meshgrid(torch.arange(height, device=offsets.device, dtype=offsets.dtype)+.5,
                          torch.arange(width, device=offsets.device, dtype=offsets.dtype)+.5, indexing='ij')
    nx, ny = (x-width/2)/f, -(y-height/2)/f
    stack = []
    for focus in distances:
        values = torch.zeros(len(offsets), 3, height, width, device=offsets.device, dtype=offsets.dtype)
        for (lo, hi), depth in zip([(-.32, -.12), (-.1, .1), (.12, .32)], [.5, .8, 1.2]):
            # Ray starts at pupil displacement a and passes through d_focus*nx.
            px = depth*nx+offsets[:, 0, None, None]*(1-depth/focus)
            py = depth*ny+offsets[:, 1, None, None]*(1-depth/focus)
            inside = (px/depth >= lo)&(px/depth <= hi)
            texture = .5+.3*torch.sin(px*450)*torch.cos(py*370)
            values = torch.where(inside[:, None], texture[:, None].expand(-1, 3, -1, -1), values)
        stack.append(values.mean(0))
    return torch.stack(stack)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    images, offsets, rois, depths = known_depth_views(device=args.device)
    distances = [.4, .5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.3, 1.4]
    truth, support = refocus_views(images, offsets, distances)
    valid = support.all(0)
    rois = {k: v*valid for k, v in rois.items()}
    sharp = images[len(images)//2]
    arguments = dict(truth=truth, sharp=sharp, valid=valid, rois=rois, distances_cm=[100*d for d in distances])
    correct = focus_metrics(truth, **arguments)
    flat = focus_metrics(truth.mean(0).expand_as(truth), **arguments)
    direct = direct_focus_truth(offsets, distances, *images.shape[-2:])
    direct_metrics = focus_metrics(direct, **{**arguments, 'truth': direct})
    resampling_rmse = {label: float((((truth-direct).square()*roi).sum()/(len(truth)*3*roi.sum())).sqrt())
                       for label, roi in rois.items()}
    for label, depth in depths.items():
        assert abs(correct['rois'][label]['gt_peak_distance_cm']/100-depth) < 1e-6, (label, correct['rois'][label])
        assert abs(direct_metrics['rois'][label]['gt_peak_distance_cm']/100-depth) < 1e-6
        assert resampling_rmse[label] < .01, (label, resampling_rmse[label])
        assert abs(flat['rois'][label]['focus_response_nrmse']-1) < 1e-6
    # Fixed LCD patterns optimized only at the central stereo pair. Cameras look
    # along world -X; local right=+Y, up=+Z. Metric offsets preserve rotation.
    height, width = images.shape[-2:]
    def poses_at(local_offsets):
        poses = torch.eye(4, device=args.device, dtype=torch.float64).repeat(len(local_offsets), 1, 1)
        poses[:, :3, :3] = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], device=args.device)
        poses[:, 0, 3] = 1.
        poses[:, 1:3, 3] = local_offsets
        return poses
    # Generate exact stereo targets with the same independent scene, at +/-3cm.
    def scene_at(local_offsets):
        f = width/(2*math.tan(math.radians(20)))
        y, x = torch.meshgrid(torch.arange(height, device=args.device, dtype=torch.float64)+.5,
                              torch.arange(width, device=args.device, dtype=torch.float64)+.5, indexing='ij')
        nx, ny = (x-width/2)/f, -(y-height/2)/f
        result = torch.zeros(len(local_offsets), 3, height, width, device=args.device, dtype=torch.float64)
        for (lo, hi), depth in zip([(-.32, -.12), (-.1, .1), (.12, .32)], depths.values()):
            px = local_offsets[:, 0, None, None]+depth*nx
            py = local_offsets[:, 1, None, None]+depth*ny
            inside = (px/depth >= lo)&(px/depth <= hi)
            value = .5+.3*torch.sin(px*450)*torch.cos(py*370)
            result = torch.where(inside[:, None], value[:, None].expand(-1, 3, -1, -1), result)
        return result
    stereo_offsets = torch.tensor([[-.03, 0.], [.03, 0.]], dtype=torch.float64, device=args.device)
    renderer = ExamRenderer(poses_at(stereo_offsets), height, width, args.device)
    target = scene_at(stereo_offsets).float()
    torch.manual_seed(7)
    initial = torch.rand(3, 3, height, width, device=args.device)*.4
    patterns, trace = refine(renderer, target, initial, steps=100, lr=.02, iteration_dtype=torch.float32)
    grids, mask = ta_grids(poses_at(offsets).cpu().numpy(), height, width, args.device, torch.float64)
    with torch.no_grad():
        output = render_views(patterns, grids)
    predicted, predicted_support = refocus_views(output, offsets, distances, mask[:, None])
    common = valid & predicted_support.all(0)
    arguments['valid'] = common
    arguments['rois'] = {k: v*common for k, v in rois.items()}
    lcd = focus_metrics(predicted, **arguments)
    # Same evaluator for the initial frozen patterns; extra views never optimize them.
    with torch.no_grad():
        initial_output = render_views(renderer.quantize_patterns(initial), grids)
    initial_stack, _ = refocus_views(initial_output, offsets, distances, mask[:, None])
    initial_metrics = focus_metrics(initial_stack, **arguments)
    result = {'distances_m': distances, 'known_depths_m': depths,
              'aperture': '3x3 parallel views at -4/0/+4 mm per axis; quadrature smoke test',
              'truth_self_check': correct, 'focus_invariant_negative_control': flat,
              'direct_ray_self_check': direct_metrics, 'shift_add_vs_direct_roi_rmse': resampling_rmse,
              'lcd_random_initial': initial_metrics, 'lcd_stereo_refined100': lcd,
              'centre_psnr_initial': scores(renderer, renderer.quantize_patterns(initial), target),
              'centre_psnr_refined': scores(renderer, patterns, target), 'trace': trace,
              'limits': 'Low-resolution analytic scene, random initialization; not Lego or pretrained-network evidence.'}
    (args.out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for name, stack in [('truth', truth), ('lcd', predicted)]:
        for i, distance in enumerate(distances):
            array = (stack[i].clamp(0, 1)*255+.5).byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(array).save(args.out/f'{name}_{distance:.1f}m.png')
    print(json.dumps({'self_check_peaks_cm': {k: v['gt_peak_distance_cm'] for k, v in correct['rois'].items()},
                      'centre_initial': result['centre_psnr_initial']['psnr_mean_eyes'],
                      'centre_refined': result['centre_psnr_refined']['psnr_mean_eyes'],
                      'lcd_focus_response_nrmse': {k: v['focus_response_nrmse'] for k, v in lcd['rois'].items()}}))


if __name__ == '__main__':
    main()
