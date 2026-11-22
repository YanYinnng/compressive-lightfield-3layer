"""Reproduce upstream first-subdataset indexing with distinct temporary pairs."""
import difflib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightfield_lab.core import EyeRealDataset, ROOT, UPSTREAM


def to_tensor(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()/255


with tempfile.TemporaryDirectory() as temporary:
    scene = Path(temporary)/'lego_bulldozer'
    scene.mkdir()
    expected = [32, 96, 160]
    for pair, value in enumerate(expected):
        for eye in range(2):
            Image.new('RGB', (64, 36), (value, value, value)).save(scene/f'pair{pair}_x6_y{eye*.5}_z2_{eye}.png')
    dataset = EyeRealDataset(temporary, transform=to_tensor, pattern_size=(36, 64),
                            data_mode='scene', choose_subdataset_names=['lego_bulldozer'])
    observed = [round(float(dataset[index][0][0, :, 0, 0].mean())*255) for index in range(3)]
result = {'requested_indices': [0, 1, 2], 'expected_first_eye_gray': expected,
          'observed_upstream_first_eye_gray': observed,
          'issue_reproduced': observed != expected,
          'scope': 'Temporary fixture, original repository unmodified; not a training performance claim.'}
out = ROOT/'runs/dataset_audit.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding='utf-8')

path = UPSTREAM/'data/dataset.py'
original = path.read_text(encoding='utf-8')
old = 'data_ind = ind_ - self.subdataset_image_num[subdataset_ind-1] if subdataset_ind > 0 else 0'
if old in original:
    corrected = original.replace(old, old[:-1]+'ind_', 1)
    patch = ''.join(difflib.unified_diff(original.splitlines(True), corrected.splitlines(True),
                                       fromfile='a/data/dataset.py', tofile='b/data/dataset.py'))
    folder = ROOT/'patches'
    folder.mkdir(exist_ok=True)
    (folder/'0001-first-subdataset-index.patch').write_text(patch, encoding='utf-8', newline='\n')
print(json.dumps(result, indent=2))
