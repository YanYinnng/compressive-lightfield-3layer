"""Off-axis benchmark under the TA protocol: target = the unchanged input images, scored with
a ground-truth camera that is defined independently of the submitted / solved camera.

  score    re-score submitted off-axis patterns under every GT camera hypothesis
  matrix   solve off-axis patterns under each K model, score under every hypothesis
  physical render the Lego Gaussian scene with each hypothesis camera (physically
           consistent off-axis views, stored with their K) for later experiments

Scores: TA ray model on GPU (parity with TA forward_render <1e-7, identical masks), RGB8
patterns, float eyes, the GT camera's strict mask, mean of eyes then samples.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.core import scores  # noqa: E402
from lightfield_lab.exam_data import read_rgb8  # noqa: E402
from lightfield_lab.exam_geometry import ExamRenderer, HEIGHT, WIDTH  # noqa: E402
from lightfield_lab.offaxis_models import MODELS, camera, from_matrices  # noqa: E402


def rows_of(index):
    data = json.loads(Path(index).read_text(encoding='utf-8'))
    return Path(index).resolve().parent, data['samples'] if isinstance(data, dict) else data


def images_tensor(base, row, device):
    return torch.from_numpy(np.stack([read_rgb8(base/row[k]) for k in ('left', 'right')])).permute(0, 3, 1, 2).to(device).double()/255


def layers_tensor(arrays, device):
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device).double()/255


def evaluate(k, c2w, patterns, images, device):
    renderer = ExamRenderer(c2w, HEIGHT, WIDTH, device, geometry_backend='torch', intrinsics=k)
    with torch.no_grad():
        s = scores(renderer, patterns, images)
    return {'psnr': s['psnr_mean_eyes'], 'psnr_each_eye': s['psnr_each_eye'],
            'valid': renderer.geometry_mask.mean((1, 2, 3)).tolist()}


def summarize(records, key_path):
    """records: list of dicts; key_path: function record -> {name: psnr}."""
    table = {}
    for record in records:
        for name, value in key_path(record).items():
            table.setdefault(name, []).append(value)
    return {name: {'mean': float(np.mean(v)), 'median': float(np.median(v)), 'min': float(np.min(v)),
                   'worst5_mean': float(np.mean(sorted(v)[:max(1, len(v)//20)])), 'n': len(v)} for name, v in table.items()}


def cmd_score(args):
    device = torch.device(args.device)
    base, rows = rows_of(args.index)
    out = {}
    for folder in args.submission:
        folder = Path(folder)
        manifest = json.loads((folder/'index.json').read_text(encoding='utf-8'))
        by_id = {r['id']: r for r in manifest['samples']}
        records = []
        for row in rows[:args.limit]:
            sub = by_id[row['id']]
            images = images_tensor(base, row, device)
            off = sub['offaxis']
            patterns = layers_tensor([np.array(Image.open(folder/off[f'layer{i}'])) for i in (1, 2, 3)], device)
            declared_k = from_matrices(off['K_l'], off['K_r'])
            declared_pose = np.array([off['pose_l'], off['pose_r']])
            record = {'id': row['id'], 'declared': evaluate(declared_k, declared_pose, patterns, images, device), 'under': {}}
            for model in args.models:
                k, c2w = camera(model, row['pose_l'], row['pose_r'])
                record['under'][model] = evaluate(k, c2w, patterns, images, device)
                record['under'][model]['declared_minus_gt_px'] = (declared_k-k).tolist()
            normal = layers_tensor([np.array(Image.open(folder/sub[f'layer{i}'])) for i in (1, 2, 3)], device)
            record['normal_branch_under_standard'] = evaluate(None, np.array([row['pose_l'], row['pose_r']]), normal, images, device)['psnr']
            records.append(record)
        out[str(folder)] = {'declared': summarize(records, lambda r: {'declared': r['declared']['psnr']})['declared'],
                            'normal_branch': summarize(records, lambda r: {'n': r['normal_branch_under_standard']})['n'],
                            'under_hypothesis': summarize(records, lambda r: {m: v['psnr'] for m, v in r['under'].items()}),
                            'records': records}
        print(f'== {folder}', flush=True)
        print(f'   normal branch (standard K)  mean {out[str(folder)]["normal_branch"]["mean"]:.2f} dB', flush=True)
        print(f'   off-axis, declared camera   mean {out[str(folder)]["declared"]["mean"]:.2f} dB', flush=True)
        for model, s in out[str(folder)]['under_hypothesis'].items():
            print(f'   off-axis, GT = {model:16s} mean {s["mean"]:6.2f}  median {s["median"]:6.2f}  min {s["min"]:6.2f} dB', flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding='utf-8')


def cmd_matrix(args):
    from lightfield_lab.exam_adapter import ExamPredictor
    device = torch.device(args.device)
    base, rows = rows_of(args.index)
    rows = rows[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'config.json').write_text(json.dumps(vars(args), indent=1, default=str))
    predictor = ExamPredictor(args.weights, steps=args.steps, lr=args.lr, diagnostics=False)
    records = []
    with (out/'records.jsonl').open('w', encoding='utf-8') as stream:
        for number, row in enumerate(rows):
            started = time.time()
            left, right = read_rgb8(base/row['left']), read_rgb8(base/row['right'])
            images = images_tensor(base, row, device)
            pose_l, pose_r = np.array(row['pose_l']), np.array(row['pose_r'])
            record = {'id': row['id'], 'solve': {}}
            for model in args.solve:
                k, c2w = camera(model, pose_l, pose_r)
                rotated = not np.allclose(c2w[:, :3, :3], np.stack([pose_l, pose_r])[:, :3, :3])
                layers = predictor(left, right, pose_l, pose_r, intrinsics=None if model == 'standard' else k,
                                   render_poses=c2w if rotated else None)
                patterns = layers_tensor(layers, device)
                entry = {'initial': predictor.records[-1]['network_initial_psnr_cached_dB'], 'k': k.tolist(), 'under': {}}
                for hypothesis in args.hyps:
                    hk, hc = camera(hypothesis, pose_l, pose_r)
                    entry['under'][hypothesis] = evaluate(hk, hc, patterns, images, device)['psnr']
                record['solve'][model] = entry
            record['seconds'] = time.time()-started
            stream.write(json.dumps(record)+'\n')
            stream.flush()
            records.append(record)
            print(f'[{number+1}/{len(rows)}] {row["id"]} ' + ' '.join(
                f'{m}:{record["solve"][m]["under"][m]:.1f}' for m in args.solve if m in args.hyps) + f' ({record["seconds"]:.1f}s)', flush=True)
    matrix = {m: summarize(records, lambda r, m=m: r['solve'][m]['under']) for m in args.solve}
    (out/'summary.json').write_text(json.dumps({'rows_solve_model_cols_gt_hypothesis': matrix, 'n': len(records)}, indent=1))
    print('mean PSNR (dB): rows = camera used to solve, columns = GT camera used to score', flush=True)
    print(' '*18+''.join(f'{h[:14]:>15s}' for h in args.hyps), flush=True)
    for m in args.solve:
        print(f'{m:18s}'+''.join(f'{matrix[m][h]["mean"]:15.2f}' for h in args.hyps), flush=True)


def cmd_physical(args):
    project = ROOT if (ROOT/'python_packages').is_dir() else ROOT.parents[1]  # dev tree or releases/<commit>
    sys.path.insert(0, str(project/'python_packages'))
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(project/'cache/torch_extensions'))
    os.environ.setdefault('MAX_JOBS', '4')
    from lightfield_lab.gaussian_scene import GaussianTargetRenderer
    base, rows = rows_of(args.index)
    rows = rows[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    scene = GaussianTargetRenderer(args.asset or project/'assets/lego_bulldozer.ply')
    manifests = {model: [] for model in args.hyps}
    for number, row in enumerate(rows):
        for model in args.hyps:
            k, c2w = camera(model, row['pose_l'], row['pose_r'])
            images, info = scene.render(c2w, intrinsics=k)
            folder = out/model/'images'
            folder.mkdir(parents=True, exist_ok=True)
            names = []
            for side, image in zip('LR', images):
                name = f'images/sample_{number:06d}_{side}.png'
                Image.fromarray(image).save(out/model/name, compress_level=1)
                names.append(name)
            manifests[model].append({'id': row['id'], 'pose_l': row['pose_l'], 'pose_r': row['pose_r'],
                                     'left': names[0], 'right': names[1], 'gt_pose_l': c2w[0].tolist(), 'gt_pose_r': c2w[1].tolist(),
                                     'gt_K_fx_fy_cx_cy': k.tolist(), 'foreground_fraction': info['foreground_fraction_each_eye']})
        print(f'[{number+1}/{len(rows)}] {row["id"]}', flush=True)
    for model, samples in manifests.items():
        (out/model/'test.json').write_text(json.dumps({'phase': 'offaxis_physical', 'gt_camera_model': model,
            'note': 'Lego Gaussian scene rendered with this off-axis camera (same eye centres); input poses kept in pose_l/pose_r.',
            'scene_transform': scene.metadata['scene_transform'], 'samples': samples}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('score', 'matrix', 'physical'):
        p = sub.add_parser(name)
        p.add_argument('--index', required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--limit', type=int, default=None)
        p.add_argument('--device', default='cuda')
        p.add_argument('--hyps' if name != 'score' else '--models', nargs='+', default=list(MODELS))
    sub.choices['score'].add_argument('--submission', nargs='+', required=True)
    sub.choices['matrix'].add_argument('--solve', nargs='+', default=list(MODELS))
    sub.choices['matrix'].add_argument('--steps', type=int, default=100)
    sub.choices['matrix'].add_argument('--lr', type=float, default=.08)
    sub.choices['matrix'].add_argument('--weights', default=None)
    sub.choices['physical'].add_argument('--asset', default=None, help='default: <project>/assets/lego_bulldozer.ply')
    args = parser.parse_args()
    {'score': cmd_score, 'matrix': cmd_matrix, 'physical': cmd_physical}[args.command](args)


if __name__ == '__main__':
    main()
