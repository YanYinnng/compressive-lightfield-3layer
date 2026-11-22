"""How fast can this machine write/read 1080p RGB PNGs in parallel? Threads vs processes, Pillow vs OpenCV.

python tools/speed_png_parallel.py <one layer png to use as content>
"""
import io, os, sys, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
from PIL import Image

src = sys.argv[1]
arr = np.asarray(Image.open(src).convert('RGB'))
data = Image.open(src).convert('RGB')
try:
    import cv2
except ImportError:
    cv2 = None
print('cpu_count', os.cpu_count(), 'sched_getaffinity', len(os.sched_getaffinity(0)))
for f in ('/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu/cpu.cfs_quota_us', '/sys/fs/cgroup/cpu/cpu.cfs_period_us'):
    if os.path.exists(f):
        print(f, open(f).read().strip())


def pil_encode(_):
    Image.fromarray(arr).save(io.BytesIO(), format='PNG', compress_level=1)


def cv2_encode(_):
    cv2.imencode('.png', arr[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])


def pil_decode(_):
    np.asarray(Image.open(src).convert('RGB'))


def cv2_decode(_):
    cv2.imread(src, cv2.IMREAD_COLOR)


N = 96
jobs = [('PIL encode', pil_encode), ('PIL decode', pil_decode)]
if cv2 is not None:
    jobs += [('cv2 encode', cv2_encode), ('cv2 decode', cv2_decode)]
    cv2.setNumThreads(1)
for name, fn in jobs:
    t = time.perf_counter(); [fn(0) for _ in range(8)]; single = (time.perf_counter()-t)/8
    line = f'{name:11s} single {single*1000:6.1f} ms |'
    for kind, Pool in (('thr', ThreadPoolExecutor), ('proc', ProcessPoolExecutor)):
        for workers in (8, 16, 32, 48):
            with Pool(workers) as pool:
                list(pool.map(fn, range(workers)))  # warm
                t = time.perf_counter(); list(pool.map(fn, range(N))); dt = time.perf_counter()-t
            line += f' {kind}{workers} {N/dt:6.1f}/s'
    print(line, flush=True)
