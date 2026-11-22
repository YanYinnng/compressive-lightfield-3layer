"""Solver comparison per camera model (scored under the same camera = TA protocol diagonal):
network initial value (EyeReal, warped to the standard camera) vs training-free initial values
(zero, linearised SART tomography), each followed by projected Adam.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.core import make_model, refine, scores, synchronize  # noqa: E402
from lightfield_lab.exam_adapter import network_initial  # noqa: E402
from lightfield_lab.exam_data import read_rgb8  # noqa: E402
from lightfield_lab.exam_geometry import ExamRenderer, HEIGHT, WIDTH, screen_corners  # noqa: E402
from lightfield_lab.intrinsics import warp_between  # noqa: E402
from lightfield_lab.offaxis_models import camera  # noqa: E402
from lightfield_lab.tomographic import sart  # noqa: E402

METHODS = ['net+adam100', 'net+adam300', 'zero+adam100', 'zero+adam300', 'sart30', 'sart100', 'sart30+adam100',
           'sart30+adam300', 'sart100+adam100']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', action='append', required=True, help='name=path/test.json (repeatable)')
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--models', nargs='+', default=['standard', 'l2_camera', 'bbox_eye'])
    parser.add_argument('--methods', nargs='+', default=METHODS)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    model = make_model(HEIGHT, WIDTH, device, ROOT/'weights/pretrained_model.pth')
    screens = screen_corners(device)
    records = []
    stream = (args.out/'records.jsonl').open('w')
    for spec in args.index:
        name, path = spec.split('=', 1)
        path = Path(path)
        rows = json.loads(path.read_text(encoding='utf-8'))['samples'][:args.limit]
        for row in rows:
            images = torch.from_numpy(np.stack([read_rgb8(path.parent/row[k]) for k in ('left', 'right')])).permute(0, 3, 1, 2).to(device).float()/255
            poses = np.array([row['pose_l'], row['pose_r']])
            views = torch.from_numpy(poses).to(device)
            for m in args.models:
                k, c2w = camera(m, *poses)
                renderer = ExamRenderer(c2w, HEIGHT, WIDTH, device, geometry_backend='torch', intrinsics=None if m == 'standard' else k)
                record = {'set': name, 'id': row['id'], 'model': m, 'valid': renderer.geometry_mask.mean().item()}
                for method in args.methods:
                    synchronize(device)
                    started = time.perf_counter()
                    init, _, tail = method.partition('+')
                    if init == 'net':
                        if m == 'standard':
                            source = images
                        else:
                            source, _ = warp_between(images, k, c2w, poses)
                        initial = network_initial(model, source, views, screens)
                    elif init == 'zero':
                        initial = torch.zeros(3, 3, HEIGHT, WIDTH, device=device)
                    else:
                        initial = sart(renderer, images, iterations=int(init[4:]))
                    steps = int(tail[4:]) if tail else 0
                    if steps:
                        patterns, _ = refine(renderer, images, initial, steps=steps, lr=.08, iteration_dtype=torch.float32)
                    else:
                        patterns = renderer.quantize_patterns(initial)
                    synchronize(device)
                    seconds = time.perf_counter()-started
                    with torch.no_grad():
                        record[method] = {'psnr': scores(renderer, patterns, images.double())['psnr_mean_eyes'], 'seconds': seconds,
                                          'initial_psnr': scores(renderer, renderer.quantize_patterns(initial), images.double())['psnr_mean_eyes']}
                stream.write(json.dumps(record)+'\n')
                stream.flush()
                records.append(record)
                print(f"{name} {row['id']} {m:10s} " + ' '.join(f"{x}:{record[x]['psnr']:.1f}" for x in args.methods), flush=True)
    summary = {}
    for name in sorted({r['set'] for r in records}):
        for m in args.models:
            group = [r for r in records if r['set'] == name and r['model'] == m]
            summary[f'{name}/{m}'] = {x: {'psnr_mean': float(np.mean([r[x]['psnr'] for r in group])),
                                          'psnr_worst5pct': float(np.mean(sorted(r[x]['psnr'] for r in group)[:max(1, len(group)//20)])),
                                          'initial_mean': float(np.mean([r[x]['initial_psnr'] for r in group])),
                                          'seconds_mean': float(np.mean([r[x]['seconds'] for r in group]))} for x in args.methods}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=1))
    print('mean PSNR dB (initial -> final) [s per sample]')
    for key, value in summary.items():
        print(f'{key:28s} ' + '  '.join(f"{x}: {v['initial_mean']:.1f}->{v['psnr_mean']:.1f} [{v['seconds_mean']:.2f}s]" for x, v in value.items()))


if __name__ == '__main__':
    main()
