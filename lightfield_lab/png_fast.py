"""Fast lossless PNG writer for RGB8 layer patterns (the H100 host has a 20-CPU quota; Pillow's encoder at level 1
costs ~136 ms per 1080p layer, which starved the GPU-driving thread).

PNG: IHDR (8-bit RGB, no interlace), one IDAT with every row 'Up'-filtered, IEND. Deflate by ISA-L (python-isal,
installed or placed in the folder named by LF_PYDEPS; ~15 ms/layer, ~1.6 MB) or, without it, zlib (~63 ms/layer).
Lossless round-trip through Pillow verified (tools/speed_png_encoders.py).
"""
import os
from pathlib import Path
import struct
import sys
import zlib

import numpy as np

PYDEPS = Path(os.environ.get('LF_PYDEPS', 'pydeps'))
try:
    if PYDEPS.is_dir() and str(PYDEPS) not in sys.path:
        sys.path.append(str(PYDEPS))
    from isal import isal_zlib as _deflate
    BACKEND = 'isal'
except ImportError:
    _deflate = zlib
    BACKEND = 'zlib'


def _chunk(kind, body):
    return struct.pack('>I', len(body))+kind+body+struct.pack('>I', zlib.crc32(kind+body) & 0xffffffff)


def png_bytes(array, level=1):
    """array: HxWx3 uint8 -> PNG file bytes."""
    array = np.ascontiguousarray(array, dtype=np.uint8)
    h, w, c = array.shape
    if c != 3:
        raise ValueError('RGB arrays only')
    rows = array.reshape(h, w*3)
    raw = np.empty((h, w*3+1), np.uint8)
    raw[:, 0] = 2                                    # filter type Up for every row
    raw[0, 1:] = rows[0]                             # the row above the first row is zero
    np.subtract(rows[1:], rows[:-1], out=raw[1:, 1:])  # uint8 wraparound = PNG Up filter
    return (b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + _chunk(b'IDAT', _deflate.compress(raw.tobytes(), level)) + _chunk(b'IEND', b''))


def write_png(path, array, level=1):
    with open(path, 'wb') as stream:
        stream.write(png_bytes(array, level))
