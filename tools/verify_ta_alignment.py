"""Native-resolution audit against the unchanged TA forward reference."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightfield_lab.exam_geometry import ExamRenderer
from lightfield_lab.ta_reference import load_reference

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--geometry-backend', choices=['numpy', 'torch', 'both'], default='both')
args = parser.parse_args()
backends = ['numpy', 'torch'] if args.geometry_backend == 'both' else [args.geometry_backend]
reference = load_reference()
generator = np.random.default_rng(20261106)
rows = json.loads((ROOT/'testing/demo_input/test.json').read_text())['samples']
torch.set_num_threads(4)
records = []
for row in rows:
    patterns = generator.integers(0, 256, (3, 1080, 1920, 3), dtype=np.uint8)
    poses = np.array([row['pose_l'], row['pose_r']], dtype=np.float64)
    truth, mask = reference.forward_render(patterns, *poses)
    for backend in backends:
        renderer = ExamRenderer(poses, 1080, 1920, device='cuda' if torch.cuda.is_available() else 'cpu', geometry_backend=backend)
        source = torch.from_numpy(patterns).permute(0, 3, 1, 2).to(renderer.device)
        with torch.no_grad():
            actual = renderer(source).permute(0, 2, 3, 1).cpu().numpy()
        delta = np.abs(actual-truth)
        value = {'id': row['id'], 'geometry_backend': backend, 'device': str(renderer.device), 'max_abs_error': float(delta.max()),
                 'mean_abs_error': float(delta.mean()),
                 'mask_different_pixels': int(np.count_nonzero(mask != renderer.geometry_mask[:, 0].cpu().numpy().astype(bool)))}
        assert value['max_abs_error'] <= 1e-7, value
        assert value['mask_different_pixels'] == 0, value
        records.append(value)
        print(value, flush=True)
        del renderer, source

def function_source(path, name):
    text = path.read_text(encoding='utf-8')
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(text, node)

for name in ['forward_render', '_bilinear_rgb']:
    assert function_source(ROOT/'testing/inference.py', name) == function_source(ROOT/'solution/inference.py', name), name
assert (ROOT/'testing/check_submission.py').read_bytes() == (ROOT/'solution/check_submission.py').read_bytes()
out = ROOT/'runs/ta_alignment'
out.mkdir(parents=True, exist_ok=True)
result = {'native_resolution': [1080, 1920], 'records': records, 'forward_functions_unchanged': True,
          'reference_sha256': hashlib.sha256((ROOT/'testing/inference.py').read_bytes()).hexdigest()}
(out/'renderer_parity.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print('TA reference functions and checker preserved; native-resolution ray alignment passed.')
