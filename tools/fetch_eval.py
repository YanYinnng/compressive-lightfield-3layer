"""Fetch and checksum the small public EyeReal evaluation archive."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'datasets/eval.zip'
path.parent.mkdir(parents=True, exist_ok=True)
expected = 'dca46a9704ec1e6caa53d82229ecbb47fd5456cd8ad2ebbd4deb07cb1e697f33'
if not path.exists():
    partial = path.with_suffix('.zip.part')
    url = 'https://hf-mirror.com/datasets/WeijieMa/EyeReal/resolve/main/eval.zip'
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=45) as response, partial.open('wb') as output:
        total, reported = 0, 0
        while block := response.read(1024*1024):
            output.write(block)
            digest.update(block)
            total += len(block)
            if total-reported > 25*1024*1024:
                print(f'Downloaded {total/1e6:.1f} MB', flush=True)
                reported = total
    if digest.hexdigest() != expected:
        raise RuntimeError('Checksum mismatch; partial archive was not activated.')
    partial.replace(path)
else:
    with path.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != expected:
            raise RuntimeError('Unexpected existing archive; left untouched.')
with zipfile.ZipFile(path) as archive:
    names = archive.namelist()
    from collections import Counter
    print('Verified:', path)
    print('Total members:', len(names))
    print('Top folders:', Counter('/'.join(n.split('/')[:2]) for n in names).most_common(20))
    print('Example names:', [n for n in names if n.lower().endswith(('.jpg', '.png'))][:8])
