"""Which part of the submission loop slows the fused predictor down? Same predictor, same samples, add I/O step by step.

python tools/speed_loop_modes.py --index <test.json> [--n 40]       (run from the repo root, H100)
modes: predict only (images preloaded) | + reader threads | + writer threads | both (= submit_v3 loop) | ...
"""
import argparse, io, json, sys, time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '.')
from PIL import Image
from lightfield_lab.v3_submission import make_v3_predictor

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=Path, required=True)
parser.add_argument('--n', type=int, default=40)
args = parser.parse_args()
rows = json.load(open(args.index))['samples'][:args.n]
base = args.index.parent
predict = make_v3_predictor(None, steps=100, lr=.08, solver='fused')


def read(row):
    return [np.asarray(Image.open(base/row[k]).convert('RGB')).copy() for k in ('left', 'right')]


def encode(array):
    Image.fromarray(array).save(io.BytesIO(), format='PNG', compress_level=1)


def encode_bytes(array):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format='PNG', compress_level=1)
    return len(buf.getvalue())


preloaded = [read(r) for r in rows]
poses = [(np.asarray(r['pose_l'], float), np.asarray(r['pose_r'], float)) for r in rows]
for _ in range(2):
    predict(*preloaded[0], *poses[0])


def run(name, readers=0, writers=0, procs=False):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    per = []
    rp = ThreadPoolExecutor(readers) if readers else None
    wp = (ProcessPoolExecutor if procs else ThreadPoolExecutor)(writers) if writers else None
    futures = {i: rp.submit(read, rows[i]) for i in range(min(16, len(rows)))} if rp else None
    pending = []
    for i in range(len(rows)):
        tick = time.perf_counter()
        if rp:
            left, right = futures.pop(i).result()
            if i+16 < len(rows):
                futures[i+16] = rp.submit(read, rows[i+16])
        else:
            left, right = preloaded[i]
        result = predict(left, right, *poses[i])
        if wp:
            for a in list(result['patterns'])+list(result['offaxis']['patterns']):
                pending.append(wp.submit(encode_bytes if procs else encode, a))
        per.append(time.perf_counter()-tick)
    for f in pending:
        f.result()
    total = time.perf_counter()-t0
    for p in (rp, wp):
        if p:
            p.shutdown()
    print(f'{name:38s} per-sample median {np.median(per)*1000:6.1f} ms, total {total/len(rows)*1000:6.1f} ms/sample', flush=True)


run('predict only (preloaded)')
run('+ 16 reader threads', readers=16)
run('+ 48 writer threads', writers=48)
run('+ 16 readers + 48 writers (submit loop)', readers=16, writers=48)
run('+ 16 readers + 16 writer processes', readers=16, writers=16, procs=True)
run('+ 4 readers + 8 writer threads', readers=4, writers=8)
