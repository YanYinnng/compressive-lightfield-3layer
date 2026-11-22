"""Download the public EyeReal checkpoint and verify its published LFS SHA256."""
import hashlib
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'weights' / 'pretrained_model.pth'
SHA256 = 'd6b08d0a6ceca31fb0b45e351b10ddf0e7c5265f14ea374858427bd45f480ff3'
URL = 'https://hf-mirror.com/datasets/WeijieMa/EyeReal/resolve/main/pretrained_model.pth'

if __name__ == '__main__':
    DEST.parent.mkdir(parents=True, exist_ok=True)
    partial = DEST.with_suffix('.pth.part')
    if DEST.exists():
        digest = hashlib.file_digest(DEST.open('rb'), 'sha256').hexdigest()
        if digest != SHA256:
            raise RuntimeError('Existing checkpoint has an unexpected checksum; left untouched.')
        print('Already verified:', DEST, flush=True)
    else:
        started = time.perf_counter()
        digest = hashlib.sha256()
        with urllib.request.urlopen(URL, timeout=45) as response, partial.open('wb') as stream:
            size = 0
            last_report = 0
            while block := response.read(1024 * 1024):
                stream.write(block)
                digest.update(block)
                size += len(block)
                if size - last_report >= 25 * 1024 * 1024:
                    print(f'Downloaded {size / 1e6:.1f} MB', flush=True)
                    last_report = size
        if digest.hexdigest() != SHA256:
            raise RuntimeError('Checkpoint checksum mismatch; partial file was not activated.')
        partial.replace(DEST)
        print(f'Verified {size:,} bytes in {time.perf_counter() - started:.1f}s: {DEST}', flush=True)
