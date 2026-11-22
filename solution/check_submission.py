"""Validate candidate output using only the published input manifest."""
import argparse
import json
from pathlib import Path
import struct
from PIL import Image


def records(p):
    if p.suffix.lower()=='.jsonl':
        return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
    data=json.loads(p.read_text(encoding='utf-8'))
    return data['samples'] if isinstance(data,dict) else data


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--index',required=True)
    ap.add_argument('--submission',required=True,help='Path to patterns.json')
    args=ap.parse_args()
    src=records(Path(args.index))
    manifest=Path(args.submission).resolve()
    data=json.loads(manifest.read_text(encoding='utf-8'))
    assert isinstance(data,dict) and isinstance(data.get('group'),str) and data['group'].strip(), 'Missing group'
    out=records(manifest)
    ids=[r['id'] for r in src]
    got=[r['id'] for r in out]
    assert len(ids)==len(set(ids)), 'Duplicate input IDs'
    assert len(got)==len(set(got)) and set(got)==set(ids), 'Missing/extra/duplicate output IDs'
    used=set()
    for row in out:
        for li in (1,2,3):
            rel=Path(row[f'layer{li}'])
            assert not rel.is_absolute() and '..' not in rel.parts
            p=(manifest.parent/rel).resolve()
            assert p.is_relative_to(manifest.parent) and p not in used
            used.add(p)
            with p.open('rb') as f:
                h=f.read(26)
            assert h[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>IIBB',h[16:26])==(1920,1080,8,2)
            with Image.open(p) as im:
                assert im.size==(1920,1080) and im.mode=='RGB'
                im.verify()
    print(f'PASS: {len(out)} IDs, {len(used)} unique 1920x1080 RGB8 PNGs.')


if __name__=='__main__':
    main()
