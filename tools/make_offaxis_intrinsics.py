"""Write a declared screen-framing K candidate without changing input images/poses."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.exam_data import read_manifest
from lightfield_lab.exam_geometry import HEIGHT, WIDTH
from lightfield_lab.intrinsics import screen_adaptive_intrinsics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--model', choices=['lens_shift', 'screen_fit'], default='lens_shift')
args = parser.parse_args()
if args.out.exists():
    parser.error('Choose a new output.')
_, rows = read_manifest(args.index)
samples = [{'id': r['id'], 'intrinsics': screen_adaptive_intrinsics([r['pose_l'], r['pose_r']], HEIGHT, WIDTH, args.model).tolist()}
           for r in rows]
args.out.write_text(json.dumps({'design_model': args.model, 'samples': samples,
    'scope': 'screen-framing hypothesis, fixed extrinsics; official format pending'}, indent=2))
