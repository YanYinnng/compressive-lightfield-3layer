"""Read a bounded public validation subset using cached HTTP byte ranges.

No 1.26GB full download. zipfile verifies the CRC of each selected member.
The full archive's SHA256 is recorded, not claimed verified by partial reading.
"""
import argparse
import io
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://hf-mirror.com/datasets/WeijieMa/EyeReal/resolve/main/uco3d_processed_val_sample.zip'
SIZE = 1259756422
LFS_SHA256 = '7a85b3600fe7473c679e9684ed351424f379e1a7e6be7bead1189fcfa6ce56ca'
DISRUPT_SIZE = 1257271625
DISRUPT_SHA256 = 'dc7161027a7620318e946ca0652279208b3d574e95385db424fb0ef277441654'


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, cache):
        self.url, self.size, self.cache = url, size, cache
        self.position, self.downloaded = 0, 0
        self.block_size = 512*1024
        cache.mkdir(parents=True, exist_ok=True)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if not 0 <= self.position <= self.size:
            raise ValueError('Seek outside archive.')
        return self.position

    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        pieces = []
        while size:
            block = self.position//self.block_size
            start = block*self.block_size
            end = min(start+self.block_size, self.size)-1
            path = self.cache/f'{block:06d}.bin'
            if path.exists():
                data = path.read_bytes()
            else:
                request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=40) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                        raise RuntimeError('Server did not return the requested byte range.')
                    data = response.read(end-start+2)
                if len(data) != end-start+1:
                    raise RuntimeError('Incomplete range response.')
                path.write_bytes(data)
                self.downloaded += len(data)
            local = self.position-start
            take = min(size, len(data)-local)
            if take <= 0:
                raise RuntimeError('Invalid cached block.')
            pieces.append(data[local:local+take])
            self.position += take
            size -= take
        return b''.join(pieces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', type=int, default=6)
    parser.add_argument('--pairs-per-object', type=int, default=2)
    parser.add_argument('--list-only', action='store_true')
    parser.add_argument('--disrupt', action='store_true')
    parser.add_argument('--uco3d-only', action='store_true', help='exclude named synthetic/real scenes without uCO3D sequence IDs')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if min(args.objects, args.pairs_per_object) < 1:
        parser.error('Sample counts must be positive.')
    if args.disrupt:
        url, size, digest = URL.replace('processed_val_sample', 'processed_disrupt_val'), DISRUPT_SIZE, DISRUPT_SHA256
        cache, basename = 'range_cache_disrupt_val', 'disrupt_validation_subset.zip'
    else:
        url, size, digest = URL, SIZE, LFS_SHA256
        cache, basename = 'range_cache_val', 'validation_subset.zip'
    remote = RangeFile(url, size, ROOT/'datasets'/cache)
    pattern = re.compile(r'pair(\d+)_x.*_([01])\.(?:jpg|png)$', re.I)
    with zipfile.ZipFile(remote) as source:
        folders = {}
        for name in source.namelist():
            path = PurePosixPath(name)
            match = pattern.fullmatch(path.name)
            if args.uco3d_only and not re.fullmatch(r'\d+-\d+-\d+', path.parent.parent.name):
                continue
            if match and '__MACOSX' not in name:
                folders.setdefault(str(path.parent), {}).setdefault(int(match[1]), {})[int(match[2])] = name
        keys = sorted(folders)
        print('Scene/object folders:', len(keys), 'Examples:', keys[:3], flush=True)
        if args.list_only:
            print('Downloaded bytes:', remote.downloaded)
            return
        if not keys:
            raise RuntimeError('Could not find paired images.')
        count = min(args.objects, len(keys))
        indices = [round(i*(len(keys)-1)/max(count-1, 1)) for i in range(count)]
        selected = []
        destination = args.out or ROOT/'datasets'/basename
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_STORED) as target:
            for index in indices:
                folder = keys[index]
                pairs = [pair for pair in sorted(folders[folder]) if len(folders[folder][pair]) == 2]
                take = min(args.pairs_per_object, len(pairs))
                pair_indices = [round(i*(len(pairs)-1)/max(take-1, 1)) for i in range(take)]
                for pair_index in pair_indices:
                    for name in folders[folder][pairs[pair_index]].values():
                        target.writestr(name, source.read(name))
                        selected.append(name)
                print('Fetched folder', index, flush=True)
    manifest = {'source_url': url, 'full_archive_bytes': size, 'published_full_archive_sha256': digest,
                'full_archive_sha256_verified': False, 'selected_member_crc_verified': True,
                'sampling': 'evenly spaced scene/object folders, evenly spaced pair indices; no score-based selection',
                'downloaded_this_run_bytes': remote.downloaded, 'members': selected}
    destination.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Wrote', destination, 'Downloaded bytes:', remote.downloaded, flush=True)


if __name__ == '__main__':
    main()
