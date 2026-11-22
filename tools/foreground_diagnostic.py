"""Object-region diagnostic only. Does not change optimization/scoring masks."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
load = lambda path: np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)/255
targets = [load(args.run/'targets'/f'eye-{i+1}.png') for i in range(2)]
masks = [(load(args.run/'targets'/f'valid-mask-{i+1}.png')[..., 0] > .5)&(target.min(-1) < .95)
         for i, target in enumerate(targets)]
result = {'scope': 'Diagnostic nonwhite foreground (min RGB < .95), restricted to saved geometric validity. Saved eye PNGs introduce extra 8-bit image quantization. Not official scoring.',
          'foreground_fraction_each_eye': [float(mask.mean()) for mask in masks], 'variants': {}}
for name in ['initial_png', 'adam', 'phase_then_adam']:
    values = []
    for eye, (target, mask) in enumerate(zip(targets, masks)):
        if not mask.any():
            raise ValueError('No nonwhite foreground in one eye.')
        prediction = load(args.run/name/f'eye-{eye+1}.png')
        mse = np.square(prediction-target)[mask].mean()
        values.append(float(-10*np.log10(max(mse, 1e-12))))
    result['variants'][name] = {'mean_eye_psnr': float(np.mean(values)), 'each_eye_psnr': values}
(args.run/'foreground_diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
