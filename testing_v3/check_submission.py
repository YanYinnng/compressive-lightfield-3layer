"""Format checks only; no GT camera answers or numerical scoring."""
import argparse
from pathlib import Path
import struct
from PIL import Image
from inference import read_rows,pose_array,intrinsic_array

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',required=True);ap.add_argument('--submission',required=True)
    args=ap.parse_args();src=read_rows(Path(args.index));manifest=Path(args.submission).resolve();rows=read_rows(manifest)
    ids=[r['id'] for r in src];got=[r['id'] for r in rows]
    if len(set(ids))!=len(ids) or len(set(got))!=len(got) or set(ids)!=set(got):raise ValueError('Missing/extra/duplicate IDs')
    used=set()
    for row in rows:
        off=row['offaxis']
        if off['reference_layer']!='L2':raise ValueError('Reference must be L2')
        for side in ('l','r'):intrinsic_array(off['K_'+side]);pose_array(off['pose_'+side])
        for branch,folder in ((row,'patterns'),(off,'offaxis_patterns')):
            for k in (1,2,3):
                name=branch[f'layer{k}'];rel=Path(name)
                if '\\' in name or rel.is_absolute() or '..' in rel.parts or rel.parts[0]!=folder:raise ValueError('Invalid pattern path')
                path=(manifest.parent/rel).resolve()
                if not path.is_relative_to(manifest.parent) or path in used:raise ValueError('Escaping/reused path')
                used.add(path)
                with path.open('rb') as f:head=f.read(26)
                if head[:8]!=b'\x89PNG\r\n\x1a\n' or struct.unpack('>IIBB',head[16:26])!=(1920,1080,8,2):raise ValueError('Expected RGB8 PNG 1920x1080')
                with Image.open(path) as image:image.verify()
    print(f'PASS FORMAT: {len(rows)} IDs, {len(used)} PNGs. Camera values are not compared with GT.')

if __name__=='__main__':main()
