"""Timing profile of the current pipeline and candidate speed-ups on the evaluation GPU.

Read-only with respect to project code and datasets; writes one JSON (plus temporary
files that it deletes). Each section is isolated: a failure is recorded, not fatal.
"""
import argparse
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, repeat=1):
    sync(); started = time.perf_counter()
    for _ in range(repeat):
        value = fn()
    sync()
    return (time.perf_counter()-started)/repeat, value


def environment():
    info = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'cpu_count': os.cpu_count(), 'cpu_affinity': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'disk_free_gib': {str(p): shutil.disk_usage(p).free/2**30 for p in [ROOT, Path(tempfile.gettempdir())]},
            'env': {k: os.environ.get(k) for k in ['TORCH_CUDA_ARCH_LIST', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES']}}
    meminfo = Path('/proc/meminfo')
    if meminfo.is_file():
        info['meminfo_kib'] = {line.split(':')[0]: int(line.split()[1]) for line in meminfo.read_text().splitlines()[:3]}
    try:
        info['nvidia_smi'] = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,clocks.max.sm',
                                             '--format=csv,noheader'], capture_output=True, text=True, timeout=30).stdout.strip()
        info['gpu_processes'] = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,name,used_memory', '--format=csv,noheader'],
                                               capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as error:
        info['nvidia_smi_error'] = repr(error)
    started = time.perf_counter()
    subprocess.run([sys.executable, '-c', 'import torch, torchvision'], check=True)
    info['fresh_process_import_torch_torchvision_s'] = time.perf_counter()-started
    return info


def disk(megabytes):
    result = {}
    weights = ROOT/'weights/pretrained_model.pth'
    if weights.is_file():
        seconds, _ = timed(lambda: weights.read_bytes())
        result['read_weights_s'] = seconds
        result['read_weights_MBps'] = weights.stat().st_size/2**20/seconds
    payload = os.urandom(megabytes*2**20)
    for label, folder in [('shared_project', ROOT/'runs'), ('local_tmp', Path(tempfile.gettempdir()))]:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/f'.profile_io_{os.getpid()}.bin'
        try:
            started = time.perf_counter()
            with path.open('wb') as stream:
                stream.write(payload); stream.flush(); os.fsync(stream.fileno())
            write = time.perf_counter()-started
            started = time.perf_counter(); path.read_bytes(); read = time.perf_counter()-started
            result[label] = {'write_MBps': megabytes/write, 'read_MBps_maybe_cached': megabytes/read}
        finally:
            path.unlink(missing_ok=True)
    return result


def png(sample_png):
    image = np.asarray(Image.open(sample_png).convert('RGB'))
    result = {'source': str(sample_png.relative_to(ROOT)), 'shape': list(image.shape)}
    for level in [0, 1, 2, 6]:
        def encode():
            buffer = io.BytesIO(); Image.fromarray(image).save(buffer, format='PNG', compress_level=level); return buffer.getvalue()
        seconds, data = timed(encode, repeat=3)
        result[f'level{level}'] = {'encode_s': seconds, 'bytes': len(data)}
    data = io.BytesIO(); Image.fromarray(image).save(data, format='PNG', compress_level=6)
    result['decode_s'] = timed(lambda: np.asarray(Image.open(io.BytesIO(data.getvalue())).convert('RGB')), repeat=3)[0]
    return result


def pipeline(rows, index, repeats):
    from lightfield_lab.exam_adapter import ExamPredictor
    from lightfield_lab.exam_data import read_rgb8
    started = time.perf_counter()
    predictor = ExamPredictor(steps=30)
    result = {'predictor_init_s': time.perf_counter()-started, 'model_load_s': predictor.seconds_model_load, 'calls': []}
    for _ in range(repeats):
        for row in rows:
            started = time.perf_counter()
            left, right = read_rgb8(index.parent/row['left']), read_rgb8(index.parent/row['right'])
            read_s = time.perf_counter()-started
            predictor(left, right, np.array(row['pose_l']), np.array(row['pose_r']))
            record = predictor.records[-1]
            result['calls'].append({'id': row['id'], 'read_inputs_s': read_s, 'network_s': record['seconds_network_and_input'],
                'geometry_cache_s': record['seconds_geometry_cache'], 'predictor_total_s': record['seconds_predictor_total'],
                'refine_s': record['refinement']['seconds_refine_and_checkpoint_scoring'],
                'final_psnr_cached': record['refinement']['history'][-1]['png_psnr']})
    return result, predictor


def solver(predictor, rows, index, step_counts, lrs, batch_sizes):
    from lightfield_lab.exam_adapter import network_initial
    from lightfield_lab.exam_data import read_rgb8
    from lightfield_lab.exam_geometry import ExamRenderer
    from lightfield_lab.core import image_loss, scores
    device = predictor.device
    row = rows[0]
    images = torch.from_numpy(np.stack([read_rgb8(index.parent/row[k]) for k in ['left', 'right']])).permute(0, 3, 1, 2).to(device).float()/255
    poses = np.stack([row['pose_l'], row['pose_r']])
    initial = network_initial(predictor.model, images, torch.from_numpy(poses).to(device), predictor.screens)
    renderer = ExamRenderer(poses, images.shape[-2], images.shape[-1], device)
    height, width = images.shape[-2:]
    def run(steps, lr, dtype, batch=1):
        grid = renderer.grid.to(dtype).repeat(batch, 1, 1, 1)
        mask, target = renderer.geometry_mask.to(dtype), images.to(dtype)
        p = initial.detach().to(dtype)[None].repeat(batch, 1, 1, 1, 1).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([p], lr=lr)
        sync(); started = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            source = p[:, :, None].expand(-1, -1, 2, -1, -1, -1).reshape(-1, 3, height, width)
            accumulated = F.grid_sample(source, grid, align_corners=False).reshape(batch, 3, 2, 3, height, width).sum(1)
            loss = sum(image_loss(torch.sin(accumulated[b]*math.pi/2).square(), target, mask) for b in range(batch))
            loss.backward(); optimizer.step()
            with torch.no_grad():
                p.clamp_(0, 1)
        sync(); seconds = time.perf_counter()-started
        psnr = scores(renderer, renderer.quantize_patterns(p[0].detach().double()), images)['psnr_mean_eyes']
        return {'steps': steps, 'lr': lr, 'dtype': str(dtype), 'batch': batch, 'seconds': seconds,
                'ms_per_step_per_sample': 1000*seconds/steps/batch, 'psnr_sample0': psnr}
    result = {'precision_and_steps': [], 'learning_rate': [], 'batching': []}
    run(3, .02, torch.float32)  # warm-up
    for dtype in [torch.float64, torch.float32]:
        for steps in step_counts:
            result['precision_and_steps'].append(run(steps, .02, dtype))
    for lr in lrs:
        result['learning_rate'].append(run(max(step_counts), lr, torch.float32))
    for batch in batch_sizes:
        try:
            result['batching'].append(run(20, .02, torch.float32, batch))
        except torch.cuda.OutOfMemoryError as error:
            result['batching'].append({'batch': batch, 'error': repr(error)}); torch.cuda.empty_cache()
    # Network forward precision (initialization only).
    views = torch.from_numpy(poses).to(device)
    result['network_fp32_s'] = timed(lambda: network_initial(predictor.model, images, views, predictor.screens), repeat=3)[0]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result['network_bf16_autocast_s'], bf16 = timed(lambda: network_initial(predictor.model, images, views, predictor.screens), repeat=3)
    result['network_bf16_vs_fp32_max_abs'] = float((bf16.float()-initial.float()).abs().max())
    result['geometry_cache_numpy_s'] = timed(lambda: ExamRenderer(poses, height, width, device))[0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index', type=Path, default=ROOT/'testing/demo_input/test.json')
    parser.add_argument('--quick', action='store_true', help='Short local smoke test')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from lightfield_lab.exam_data import read_manifest
    index, rows = read_manifest(args.index)
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'quick': args.quick, 'training_performed': False}
    def section(name, fn):
        started = time.perf_counter()
        try:
            report[name] = fn()
        except Exception:
            report[name] = {'error': traceback.format_exc()}
        report.setdefault('section_seconds', {})[name] = time.perf_counter()-started
        (args.out/'profile.json').write_text(json.dumps(report, indent=2, default=str))
        print(f'[profile] {name} done in {report["section_seconds"][name]:.1f}s', flush=True)
    section('environment', environment)
    section('disk', lambda: disk(16 if args.quick else 256))
    patterns = sorted((ROOT/'runs').glob('*/patterns/*.png'))
    section('png', lambda: png(patterns[0] if patterns else index.parent/rows[0]['left']))
    holder = {}
    def run_pipeline():
        result, holder['predictor'] = pipeline(rows, index, 1 if args.quick else 3)
        return result
    section('pipeline_current_code', run_pipeline)
    if 'predictor' in holder:
        section('solver_variants', lambda: solver(holder['predictor'], rows, index,
            [10] if args.quick else [30, 100], [.05] if args.quick else [.01, .03, .05, .08],
            [1, 2] if args.quick else [1, 2, 4, 8]))
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (args.out/'profile.json').write_text(json.dumps(report, indent=2, default=str))
    print('[profile] complete', flush=True)


if __name__ == '__main__':
    main()
