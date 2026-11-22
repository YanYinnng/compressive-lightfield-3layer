"""Focus calibration without image-warp interpolation: directly sample focal-plane rays."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT/'python_packages').resolve()))
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCH_EXTENSIONS_DIR', str((ROOT/'cache/torch_extensions').resolve()))
os.environ.setdefault('MAX_JOBS', '4')
from lightfield_lab.core import read_patterns
from lightfield_lab.depth_eval import depth_regions
from lightfield_lab.exam_data import read_manifest
from lightfield_lab.focus import focus_metrics
from lightfield_lab.gaussian_scene import GaussianTargetRenderer
from lightfield_lab.intrinsics import standard_intrinsics
from lightfield_lab.ray_eval import ta_grids, render_views
from lightfield_lab.refocus import aperture_subset
from tools.overnight_eval import append, evaluation_subset


def save_preview(path, stacks, distances):
    indices = (0, len(distances)//2, len(distances)-1)
    w, h, label, header = 320, 180, 125, 30
    sheet = Image.new('RGB', (label+3*w, header+len(stacks)*h), '#15191e')
    draw = ImageDraw.Draw(sheet)
    for j, index in enumerate(indices):
        draw.text((label+j*w+8, 8), f'Focus {distances[index]*100:.1f} cm', fill='white')
    for i, (name, stack) in enumerate(stacks.items()):
        y = header+i*h
        draw.text((8, y+h//2), name, fill='white')
        arrays = torch.floor(stack[list(indices)].clamp(0, 1)*255+.5).byte().permute(0, 2, 3, 1).cpu().numpy()
        for j, array in enumerate(arrays):
            sheet.paste(Image.fromarray(array).resize((w, h), Image.Resampling.LANCZOS), (label+j*w, y))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--gt', type=Path, default=ROOT/'datasets/lego_gt_v1')
    parser.add_argument('--asset', type=Path, default=ROOT/'assets/lego_bulldozer.ply')
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--methods', nargs='+', default=None)
    parser.add_argument('--preview-id', action='append', default=[])
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    source = {r['id']: r for r in [json.loads(s) for s in args.records.read_text().splitlines() if s.strip()]}
    manifest, rows = read_manifest(args.gt/'eyebox.json')
    rows = [row for row in rows if row['id'] in source]
    scene = GaussianTargetRenderer(args.asset)
    size = (args.height, args.height*16//9)
    height, width = size
    base_k = standard_intrinsics(height, width)
    output = args.out/'records.jsonl'
    finished = {json.loads(s)['id'] for s in output.read_text().splitlines()} if output.exists() else set()
    for row in rows[:args.limit]:
        if row['id'] in finished:
            continue
        record = {'id': row['id'], 'split': row['split'], 'evaluation_subset': evaluation_subset(row),
                  'ray_model': 'parallel pupil cameras, cx=cx0+f*a/d, cy=cy0-f*b/d; no image warping', 'eyes': {}}
        methods = {}
        for name, files in source[row['id']]['patterns'].items():
            if args.methods and name not in args.methods:
                continue
            arrays = [np.asarray(Image.open(args.records.parent/f).convert('RGB')).copy() for f in files]
            methods[name] = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).cuda().double()/255
        for eye, eye_name in enumerate(('left', 'right')):
            sub = [v for v in row['views'] if v['kind'] == 'subaperture' and v['eye'] == eye_name]
            indices = aperture_subset([v['offset_mm'] for v in sub], 1., radius_mm=2., shape='circle')
            sub = [sub[i] for i in indices]
            offsets = np.array([v['offset_mm'] for v in sub])*.001
            poses = np.array([v['pose'] for v in sub])
            centre_pose = np.array(row['pose_l' if eye == 0 else 'pose_r'])
            with np.load(manifest.parent/row['depth_l' if eye == 0 else 'depth_r']) as data:
                depth, alpha = data['depth_m'].astype(float), data['alpha_u8']
            regions, _ = depth_regions(depth, alpha, centre_pose)
            foreground = (alpha >= 128)&np.isfinite(depth)&(depth > 0)
            if foreground.sum() < 16:
                record['eyes'][eye_name] = {'reason': 'insufficient foreground'}; continue
            lo, hi = np.quantile(depth[foreground], [.05, .95])
            distances = np.linspace(max(.05, lo-.04), hi+.04, 13).tolist()
            # Direct pinhole truth at matching raster resolution, not resized cached RGB.
            sharp_rgb, _ = scene.render(centre_pose[None], height=height, width=width)
            sharp = torch.from_numpy(sharp_rgb[0]).permute(2, 0, 1).cuda().double()/255
            truth_stack, supports = [], []
            predictions = {name: [] for name in methods}
            for distance in distances:
                k = np.tile(base_k, (len(sub), 1))
                k[:, 2] += base_k[0]*offsets[:, 0]/distance
                k[:, 3] -= base_k[1]*offsets[:, 1]/distance
                rgb, _ = scene.render(poses, height=height, width=width, intrinsics=k)
                truth_stack.append((torch.from_numpy(rgb).permute(0, 3, 1, 2).cuda().double()/255).mean(0))
                grids, mask = ta_grids(poses, height, width, 'cuda', torch.float64, intrinsics=k)
                supports.append(mask.all(0)[None])
                with torch.no_grad():
                    for name, patterns in methods.items():
                        predictions[name].append(render_views(patterns, grids).mean(0))
            truth_stack = torch.stack(truth_stack)
            valid = torch.stack(supports).all(0)
            rois = {name: F.interpolate(torch.as_tensor(mask[None, None], device='cuda', dtype=torch.float32),
                                        size=size, mode='area')[0] > .99 for name, mask in regions.items()}
            rois = {name: mask*valid for name, mask in rois.items() if (mask*valid).sum() >= 16}
            results = {name: focus_metrics(torch.stack(stack), truth_stack, sharp, valid, rois, [100*d for d in distances])
                       for name, stack in predictions.items()}
            if row['id'] in args.preview_id:
                save_preview(args.out/f"{row['id']}_{eye_name}_focus.png",
                             {'scene reference': truth_stack, **{k: torch.stack(v) for k, v in predictions.items()}},
                             distances)
            record['eyes'][eye_name] = {'distances_m': distances, 'height': height, 'pupil_samples': len(sub),
                'geometric_roi_depth_medians_m': {name: float(np.median(depth[mask])) if mask.any() else None for name, mask in regions.items()},
                'methods': results}
        append(output, record)
        print(json.dumps({'id': row['id'], 'direct_focus_completed': True}), flush=True)
    (args.out/'complete.json').write_text(json.dumps({'state': 'SUCCEEDED'}))


if __name__ == '__main__':
    main()
