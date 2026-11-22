"""PNG writers for RGB8 1920x1080 layer patterns: CPU time, size, and lossless round-trip.

python tools/speed_png_encoders.py <layer png>
"""
import io, os, struct, sys, time, zlib
import numpy as np
from PIL import Image


def png_bytes(array, level=1, filt=0, compress=zlib.compress):
    """Minimal valid PNG (IHDR RGB8, one IDAT, IEND). filt 0 = no filter; 2 = 'Up' filter (row difference)."""
    h, w, _ = array.shape
    rows = np.ascontiguousarray(array, dtype=np.uint8).reshape(h, w*3)
    if filt == 2:
        data = rows.copy()
        data[1:] -= rows[:-1]  # uint8 wraparound, as the PNG Up filter defines
    else:
        data = rows
    raw = np.empty((h, w*3+1), np.uint8)
    raw[:, 0] = filt
    raw[:, 1:] = data
    def chunk(kind, body):
        return struct.pack('>I', len(body))+kind+body+struct.pack('>I', zlib.crc32(kind+body) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', compress(raw.tobytes(), level)) + chunk(b'IEND', b''))


arr = np.asarray(Image.open(sys.argv[1]).convert('RGB')).copy()
cases = {'Pillow level 1': lambda a: (lambda b: (Image.fromarray(a).save(b, format='PNG', compress_level=1), b.getvalue())[1])(io.BytesIO()),
         'Pillow level 0': lambda a: (lambda b: (Image.fromarray(a).save(b, format='PNG', compress_level=0), b.getvalue())[1])(io.BytesIO())}
for level in (0, 1):
    for filt in (0, 2):
        cases[f'raw zlib level {level} filter {filt}'] = lambda a, l=level, f=filt: png_bytes(a, l, f)
sys.path.insert(0, os.environ.get('LF_PYDEPS', 'pydeps'))
try:
    from isal import isal_zlib
    for level in (0, 1, 2):
        cases[f'isal level {level} filter 2'] = lambda a, l=level: png_bytes(a, l, 2, isal_zlib.compress)
        cases[f'isal level {level} filter 0'] = lambda a, l=level: png_bytes(a, l, 0, isal_zlib.compress)
except ImportError as error:
    print('isal unavailable', error)
for name, fn in cases.items():
    fn(arr)
    t = time.perf_counter()
    for _ in range(5):
        data = fn(arr)
    dt = (time.perf_counter()-t)/5
    back = np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
    print(f'{name:28s} {dt*1000:7.1f} ms  {len(data)/1e6:6.2f} MB  lossless {bool((back == arr).all())}', flush=True)
