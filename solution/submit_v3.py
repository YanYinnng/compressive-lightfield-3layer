"""One-command TA v3 submission: normal + off-axis patterns, report, code, format check.

Output folder <out>/ (named group_xxx) contains only code/, patterns/, offaxis_patterns/,
report.md and index.json (the TA now takes the folder path; --zip also writes <out>.zip
unpacking to group_xxx/). Threaded PNG I/O overlaps the
GPU solve. Format check: by default a fast check (everything the TA checker tests except the
full PNG decode) gates publishing, and run_formal.sh runs the full TA checker
after the path is published; --ta-check full runs the TA checker here instead. Runs from development/solution or from a packaged group_xxx/code directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
PACKAGED = (HERE/'lightfield_lab').is_dir()
ROOT = HERE if PACKAGED else HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

TA_IO = HERE/'inference.py' if PACKAGED else HERE/'ta_v3_inference.py'
CHECKER = HERE/'check_submission.py' if PACKAGED else ROOT/'testing_v3/check_submission.py'
SCHEMA = 'lightfield_submission_v3'


def collect_manifests(index):
    """A test.json, or a folder: every test.json below it (test_input/<scene>/test.json,
    one per scene), skipping submission examples. Sorted for a fixed order."""
    index = Path(index).resolve()
    if index.is_file():
        return [index]
    found = sorted(p for p in index.rglob('test.json') if 'submission_example' not in p.parts)
    if not found:
        raise FileNotFoundError(f'no test.json under {index}')
    return found


def fast_check(ids, manifest, ta, workers=32):
    """The TA checker's tests (IDs, cameras, paths, PNG signature/IHDR) without decoding each PNG;
    a complete IEND trailer stands in for the decode. Seconds instead of ~2.5 min per 1000 samples."""
    rows = ta.read_rows(manifest)
    got = [r['id'] for r in rows]
    if len(set(ids)) != len(ids) or len(set(got)) != len(got) or set(ids) != set(got):
        raise ValueError('Missing/extra/duplicate IDs')
    root = manifest.parent.resolve()
    paths = []
    for row in rows:
        off = row['offaxis']
        if off['reference_layer'] != 'L2':
            raise ValueError('Reference must be L2')
        for side in ('l', 'r'):
            ta.intrinsic_array(off['K_'+side])
            ta.pose_array(off['pose_'+side])
        for branch, folder in ((row, 'patterns'), (off, 'offaxis_patterns')):
            for k in (1, 2, 3):
                name = branch[f'layer{k}']
                rel = Path(name)
                if '\\' in name or rel.is_absolute() or '..' in rel.parts or rel.parts[0] != folder:
                    raise ValueError(f'Invalid pattern path {name}')
                paths.append((root/rel).resolve())
    if len(set(paths)) != len(paths) or not all(p.is_relative_to(root) for p in paths):
        raise ValueError('Escaping/reused path')

    def png_ok(path):
        with path.open('rb') as f:
            head = f.read(26)
            f.seek(-12, os.SEEK_END)
            tail = f.read(12)
        return (head[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>IIBB', head[16:26]) == (1920, 1080, 8, 2)
                and tail[4:8] == b'IEND')
    with ThreadPoolExecutor(workers) as pool:
        bad = [p for p, ok in zip(paths, pool.map(png_ok, paths)) if not ok]
    if bad:
        raise ValueError(f'Expected complete RGB8 PNG 1920x1080: {bad[:3]}')
    print(f'PASS FAST FORMAT: {len(rows)} IDs, {len(paths)} PNGs (headers, sizes, paths, cameras).', flush=True)


def layering_options(args):
    if args.solver != 'fused':
        return {}
    return {'layering': args.layering, 'layering_eps': args.layering_eps}


def torch_stream(solver):
    import torch
    return torch.cuda.Stream(device=solver.device) if solver.device.type == 'cuda' else None


def load_ta_io():
    spec = importlib.util.spec_from_file_location('ta_v3_io', TA_IO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_sources():
    """(source, relative target in code/) for the reproducible solver bundle."""
    if PACKAGED:
        return [(p, p.relative_to(HERE)) for p in HERE.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts and p.name != 'metrics.json']
    files = [(HERE/'ta_v3_inference.py', Path('inference.py')), (CHECKER, Path('check_submission.py')),
             (HERE/'submit_v3.py', Path('submit_v3.py')), (HERE/'run_formal.sh', Path('run_formal.sh')),
             (HERE/'requirements_v3.txt', Path('requirements.txt')), (HERE/'LICENSE', Path('LICENSE')),
             (HERE/'README_v3.md', Path('README.md')), (HERE/'THIRD_PARTY_NOTICES.md', Path('THIRD_PARTY_NOTICES.md')),
             (HERE/'report_template.md', Path('report_template.md')),
             (ROOT/'weights/pretrained_model.pth', Path('weights/pretrained_model.pth')),
             (ROOT/'EyeReal/LICENSE', Path('EyeReal/LICENSE'))]
    files += [(p, Path('lightfield_lab')/p.name) for p in (ROOT/'lightfield_lab').glob('*.py') if not p.name.startswith('test_')]
    for directory in ('model', 'data', 'config'):
        files += [(p, Path('EyeReal')/directory/p.name) for p in (ROOT/'EyeReal'/directory).glob('*.py')]
    # Layering v5 (--layering mono): Depth Anything V2 Small source (Apache-2.0, unchanged) and weights
    third_party = ROOT/'third_party'
    files += [(p, p.relative_to(ROOT)) for p in third_party.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    mono = ROOT/'weights/depth_anything_v2_vits.pth'
    if mono.is_file():
        files.append((mono, Path('weights/depth_anything_v2_vits.pth')))
    return files


def copy_code(destination):
    for source, relative in code_sources():
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == '.pth':
            try:
                os.link(source, target)  # same filesystem: no 200 MB copy
                continue
            except OSError:
                pass
        shutil.copy2(source, target)


def write_zip(folder, archive):
    partial = archive.with_name(archive.name+'.part')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                bundle.write(path, (Path(folder.name)/path.relative_to(folder)).as_posix())
    partial.replace(archive)


def fill_report(template, values):
    text = template.read_text(encoding='utf-8')
    for key, value in values.items():
        text = text.replace('{{'+key+'}}', str(value))
    return text


def main(argv=None, predict=None):
    """predict: an already-loaded V3Predictor (pre-warmed daemon); built here otherwise."""
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True, help='test.json, or a folder with one test.json per scene')
    parser.add_argument('--out', type=Path, required=True, help='new folder named group_xxx')
    parser.add_argument('--weights')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.08)
    parser.add_argument('--offaxis-model', choices=['bbox_eye_exact', 'bbox_eye', 'l2_camera', 'bbox_square_eye', 'lens_shift', 'lens_shift_stereo', 'lens_shift_cyclopean', 'screen_fit'], default='l2_camera')
    parser.add_argument('--offaxis-initial', choices=['network', 'sart'], default='sart',
                        help='off-axis branch initial value: EyeReal network or training-free SART tomography')
    parser.add_argument('--hard-layering', type=float, default=0.,
                        help='normal branch: margin of the hard three-layer contrast constraint (0 disables)')
    parser.add_argument('--solver', choices=['torch', 'fused'], default='fused')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None)
    parser.add_argument('--time-limit', type=float, default=0.,
                        help='0 (default): no time budget, every sample runs all steps; >0: old soft budget (cuts steps)')
    parser.add_argument('--workers', type=int, default=48, help='PNG writer threads')
    parser.add_argument('--layering', choices=['none', 'mono'], default='mono',
                        help='normal branch (fused): mono = relative-depth layer ownership (Depth Anything V2 Small) + weighted SART initial value, no network')
    parser.add_argument('--layering-eps', type=float, default=.3, help='ownership floor of the weighted SART')
    parser.add_argument('--lanes', type=int, default=2,
                        help='fused solver: samples solved concurrently, one predictor and CUDA stream each (fills GPU idle time)')
    parser.add_argument('--png-level', type=int, choices=range(10), default=1)
    parser.add_argument('--limit', type=int, default=None, help='debug: first N samples only (fails the ID check)')
    parser.add_argument('--zip', action='store_true', help='also write <out>.zip (no longer required)')
    parser.add_argument('--ta-check', choices=['fast', 'full'], default='fast',
                        help='fast: quick check here, full TA checker after publishing (run_formal.sh); full: TA checker here')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.solver == 'fused' and (args.time_limit or args.hard_layering):
        parser.error('--solver fused requires --time-limit 0 and --hard-layering 0')
    out = args.out.resolve()
    group = out.name
    if not group.startswith('group_') or not group.isascii() or any(c in group for c in ' /\\:'):
        parser.error('--out must end in an ASCII folder name group_xxx.')
    if out.exists() and any(out.iterdir()):
        parser.error('Choose a new empty --out.')
    ta = load_ta_io()
    from lightfield_lab.v3_submission import make_v3_predictor
    index = args.index.resolve()
    manifests = collect_manifests(index)
    rows = []
    for manifest in manifests:  # each row keeps its own scene folder for its image paths
        rows += [dict(r, _base=manifest.parent, _scene=manifest.parent.name) for r in ta.read_rows(manifest)]
    rows = rows[:args.limit]
    ids = [r['id'] for r in rows]
    unique_ids = len(set(ids)) == len(ids)
    print(f'[index] {len(manifests)} manifest(s), {len(rows)} samples: '
          + ', '.join(f'{m.parent.name}={sum(r["_base"] == m.parent for r in rows)}' for m in manifests), flush=True)
    if not unique_ids:
        print('[warning] sample IDs repeat across scenes; each index.json entry carries its scene', flush=True)
    predicts = list(predict) if isinstance(predict, (list, tuple)) else ([predict] if predict else [])
    lanes = max(1, args.lanes) if args.solver == 'fused' else 1
    while len(predicts) < lanes:  # each lane needs its own predictor (per-call state); same weights, same results
        predicts.append(make_v3_predictor(args.weights, steps=args.steps, lr=args.lr, offaxis_model=args.offaxis_model,
                                          device=args.device, offaxis_initial=args.offaxis_initial, hard_layering=args.hard_layering,
                                          solver=args.solver, **layering_options(args)))
    predicts = predicts[:lanes]
    predict = predicts[0]
    solver = predict.solver
    if args.solver != solver.solver_backend:
        parser.error('--solver does not match the preloaded predictor')
    if (getattr(solver, 'mono', None) is not None) != (args.solver == 'fused' and args.layering == 'mono'):
        parser.error('--layering does not match the preloaded predictor')
    if args.time_limit > 0:  # by default no allocation: it starved early samples in the 11-08 test
        solver.configure_run(started+args.time_limit, 2*len(rows), export_seconds=.2, final_seconds=30.)
    for name in ('patterns', 'offaxis_patterns'):
        (out/name).mkdir(parents=True)

    layering = getattr(predict, 'hard_layering', 0)
    if layering:
        from lightfield_lab.hard_layering import ensure_cv2, estimate_depth
        ensure_cv2()

    def read(row):
        left, right = ta.read_rgb(row['_base']/row['left']), ta.read_rgb(row['_base']/row['right'])
        depths = None
        if layering:  # stereo depth on CPU threads, overlapping the GPU solve
            try:
                depths = estimate_depth(left, right, np.array([row['pose_l'], row['pose_r']]))
            except Exception as error:
                print(f'[warning] stereo depth failed for {row["id"]}: {error!r}', flush=True)
        return left, right, depths

    from lightfield_lab.png_fast import BACKEND, write_png
    print(f'[png] lossless Up-filtered PNG, deflate {BACKEND} level {args.png_level}', flush=True)

    def write(path, array):
        partial = path.with_name(path.name+'.part')
        write_png(partial, array, args.png_level)
        partial.replace(path)

    import collections, queue
    from lightfield_lab.fused import stream_context
    free = queue.Queue()  # (predictor, stream) pairs; a lane solves one whole sample on its own stream
    for p in predicts:
        free.put((p, torch_stream(solver) if lanes > 1 else None))
    print(f'[lanes] {lanes} sample(s) in flight', flush=True)
    if lanes > 1 and rows:
        # One sequential solve per lane before going concurrent: lazy CUDA/library initialisation (e.g. torch.inverse's
        # linalg loader) is not thread-safe ("lazy wrapper should be called at most once" on the H100).
        warm = read(rows[0])
        for p in predicts:
            p(warm[0], warm[1], ta.pose_array(rows[0]['pose_l']), ta.pose_array(rows[0]['pose_r']), depths=warm[2])

    def solve(left, right, pose_l, pose_r, depths):
        started_lane = time.perf_counter()
        p, stream = free.get()
        try:
            with stream_context(stream):
                result = p(left, right, pose_l, pose_r, depths=depths)
            return result, p.last_scores, time.perf_counter()-started_lane
        finally:
            free.put((p, stream))

    samples, metrics = [], []
    prefetch = 16
    with (ThreadPoolExecutor(max_workers=min(prefetch, args.workers)) as readers,
          ThreadPoolExecutor(max_workers=args.workers) as writers,
          ThreadPoolExecutor(max_workers=lanes) as solvers):
        reads = {i: readers.submit(read, row) for i, row in enumerate(rows[:prefetch])}
        pending = []
        flight = collections.deque()

        def submit_next(number):
            left, right, depths = reads.pop(number).result()
            if number+prefetch < len(rows):
                reads[number+prefetch] = readers.submit(read, rows[number+prefetch])
            row = rows[number]
            flight.append(solvers.submit(solve, left, right, ta.pose_array(row['pose_l']), ta.pose_array(row['pose_r']), depths))

        for number in range(min(lanes, len(rows))):
            submit_next(number)
        for number, row in enumerate(rows):
            result, s, seconds = flight.popleft().result()  # results are consumed in sample order
            if number+lanes < len(rows):
                submit_next(number+lanes)
            record = {'id': row['id'], 'scene': row['_scene']}
            off = {'reference_layer': 'L2'}
            for folder, branch, layers in (('patterns', record, result['patterns']),
                                           ('offaxis_patterns', off, result['offaxis']['patterns'])):
                for k, array in enumerate(layers, 1):
                    relative = f'{folder}/sample_{number:06d}_layer{k}.png'
                    pending.append(writers.submit(write, out/relative, array))
                    branch[f'layer{k}'] = relative
            for key in ('K_l', 'K_r'):
                off[key] = ta.intrinsic_array(result['offaxis'][key]).tolist()
            for key in ('pose_l', 'pose_r'):
                off[key] = ta.pose_array(result['offaxis'][key]).tolist()
            record['offaxis'] = off
            samples.append(record)
            metrics.append({'id': row['id'], 'normal_psnr_dB': s['normal']['psnr_mean_eyes'],
                            'offaxis_psnr_dB': s['offaxis']['psnr_mean_eyes'],
                            'normal_valid_pixels': s['normal_valid_pixels'], 'offaxis_valid_pixels': s['offaxis_valid_pixels'],
                            'offaxis_fx_fy_cx_cy': s['offaxis_fx_fy_cx_cy'],
                            'normal_initial_psnr_dB': s['solver_normal']['network_initial_psnr_cached_dB'],
                            'offaxis_initial_psnr_dB': s['solver_offaxis']['network_initial_psnr_cached_dB'],
                            'steps_run': [s['solver_normal']['refinement']['steps_run'], s['solver_offaxis']['refinement']['steps_run']],
                            'hard_layering': {k: v for k, v in (s['solver_normal'].get('hard_layering') or {}).items() if k != 'history'},
                            'seconds': seconds})
            print(f'[sample {number+1}/{len(rows)}] {row["id"]}: normal {metrics[-1]["normal_psnr_dB"]:.3f} dB, '
                  f'offaxis {metrics[-1]["offaxis_psnr_dB"]:.3f} dB, {metrics[-1]["seconds"]:.2f}s', flush=True)
            done = [f for f in pending if f.done()]
            for future in done:
                future.result()
            pending = [f for f in pending if not f.done()]
            if len(pending) > 2*args.workers:
                for future in pending[:args.workers]:
                    future.result()
                pending = pending[args.workers:]
        for future in pending:
            future.result()
        solved = time.perf_counter()
        (out/'index.json').write_text(json.dumps({'schema_version': SCHEMA, 'group': group, 'format_demo': False,
                                                  'samples': samples}, ensure_ascii=False, indent=2), encoding='utf-8')
        copy_code(out/'code')
    normal = float(np.mean([m['normal_psnr_dB'] for m in metrics]))
    offaxis = float(np.mean([m['offaxis_psnr_dB'] for m in metrics]))
    shifts = np.array([[k[2]-960., k[3]-540.] for m in metrics for k in m['offaxis_fx_fy_cx_cy']])
    command = (f'python code/submit_v3.py --index /path/to/test.json --out group_new --steps {args.steps} '
               f'--lr {args.lr} --offaxis-model {args.offaxis_model} --solver {args.solver} --png-level {args.png_level} --workers {args.workers}')
    gpu = None
    if solver.device.type == 'cuda':
        import torch
        gpu = torch.cuda.get_device_name(solver.device)
    values = {'GROUP': group, 'N': len(rows), 'NORMAL_PSNR': f'{normal:.3f}', 'OFFAXIS_PSNR': f'{offaxis:.3f}',
              'STEPS': args.steps, 'LR': args.lr, 'GPU': gpu or 'CPU',
              'SOLVE_SECONDS': f'{solved-started:.1f}', 'COMMAND': command, 'OFFAXIS_MODEL': args.offaxis_model, 'SOLVER_BACKEND': args.solver,
              'SHIFT_MEAN_ABS_X': f'{np.abs(shifts[:, 0]).mean():.1f}', 'SHIFT_MAX_ABS_X': f'{np.abs(shifts[:, 0]).max():.1f}',
              'SHIFT_MEAN_ABS_Y': f'{np.abs(shifts[:, 1]).mean():.1f}', 'SHIFT_MAX_ABS_Y': f'{np.abs(shifts[:, 1]).max():.1f}'}
    (out/'report.md').write_text(fill_report(out/'code/report_template.md', values), encoding='utf-8')
    (out/'code/metrics.json').write_text(json.dumps({
        'scope': ('Self-scores with the fused CUDA solver; RGB8 patterns, float eyes, TA masks.'
                  if args.solver == 'fused' else
                  'Self-scores with the TA optics (GPU renderer, parity <1e-7 to TA forward_render); RGB8 patterns, float eyes, TA masks.'),
        'solver': args.solver,
        'num_samples': len(rows), 'mean_normal_psnr_dB': normal, 'mean_offaxis_psnr_dB': offaxis,
        'seconds_main_to_solved_and_written': solved-started, 'configuration': vars(args) | {'index': str(index), 'out': str(out),
        'weights': str(solver.weights_path)}, 'gpu': gpu, 'samples': metrics}, indent=2, default=str), encoding='utf-8')
    if unique_ids:
        # The TA checker takes one manifest: give it all IDs at once (kept outside the submission folder);
        # its presence tells run_formal.sh to run the full TA checker after publishing.
        merged = out.parent/f'{out.name}_all_test.json'
        merged.write_text(json.dumps({'samples': [{'id': r['id']} for r in rows]}), encoding='utf-8')
        if args.ta_check == 'full':
            print(f'[check] TA format check: verifying {6*len(rows)} PNGs from disk (about 2.5 min per 1000 samples on the shared disk)', flush=True)
            subprocess.run([sys.executable, '-B', str(CHECKER), '--index', str(merged), '--submission', str(out/'index.json')],
                           check=args.limit is None)
        elif args.limit is None:
            print(f'[check] fast format check of {6*len(rows)} PNGs; the full TA checker runs after publishing', flush=True)
            fast_check(ids, out/'index.json', ta)
    else:
        print('[warning] TA checker skipped (it rejects repeated IDs); PNG and path checks below', flush=True)
        for record in samples:
            for branch in (record, record['offaxis']):
                for k in (1, 2, 3):
                    assert (out/branch[f'layer{k}']).is_file(), branch[f'layer{k}']
    extra = sorted(p.name for p in out.iterdir() if p.name not in ('code', 'patterns', 'offaxis_patterns', 'report.md', 'index.json'))
    if extra:
        raise RuntimeError(f'Unexpected top-level entries: {extra}')
    checked = time.perf_counter()
    print(f'[timing] solved+written {solved-started:.1f}s, checked {checked-started:.1f}s; '
          f'normal {normal:.3f} dB, offaxis {offaxis:.3f} dB', flush=True)
    if args.zip:
        archive = out.with_suffix('.zip')
        write_zip(out, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        print(f'[zip] {archive} {archive.stat().st_size} bytes sha256 {digest} at {time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
