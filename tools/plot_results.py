"""Export scientific comparison figures from already completed experiments."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
report = json.loads((args.run/'summary.json').read_text(encoding='utf-8'))
variants = ['initial_png', 'adam', 'phase_then_adam']
fig, axes = plt.subplots(4, 6, figsize=(16, 7.8))
columns = [('eye-1.png', 'Eye 0'), ('eye-2.png', 'Eye 1'), ('error-x4-1.png', 'Eye 0 valid error x4'),
           ('layer-1.png', 'LCD 1'), ('layer-2.png', 'LCD 2'), ('layer-3.png', 'LCD 3')]
for row, name in enumerate(['targets']+variants):
    for col, (filename, title) in enumerate(columns):
        ax = axes[row, col]
        ax.set_xticks([])
        ax.set_yticks([])
        path = args.run/name/filename
        if path.exists():
            ax.imshow(np.asarray(Image.open(path)))
        else:
            ax.set_facecolor('#eeeeee')
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row == 0:
            ax.set_title(title, fontsize=10)
        if col == 0:
            label = 'Target eye images' if row == 0 else name+'\n'+f'{report["variants"][name]["psnr_mean_eyes"]:.2f} dB'
            ax.set_ylabel(label, fontsize=9)
fig.suptitle('EyeReal / per-sample refinement | '+str(report['configuration']['width'])+' x '+str(report['configuration']['height']), fontsize=14)
fig.text(.5, .015, 'Prototype geometry-mask metrics. One public/synthetic sample; not official camp scores.', ha='center', fontsize=9)
fig.tight_layout(rect=(0, .04, 1, .95))
fig.savefig(args.run/'comparison.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4))
for name, color in [('adam', '#0072B2'), ('phase_then_adam', '#D55E00')]:
    history = report['variants'][name]['history']
    ax.plot([r['step'] for r in history], [r['png_psnr'] for r in history], marker='o', color=color, label=name)
ax.axhline(report['variants']['initial_png']['psnr_mean_eyes'], color='#555555', ls='--', label='initial PNG')
ax.set(xlabel='Optimization step', ylabel='PNG mean-eye PSNR (dB)', title='Checkpoint scores after 8-bit quantization')
ax.grid(alpha=.2)
ax.legend()
fig.tight_layout()
fig.savefig(args.run/'convergence.png', dpi=160)
plt.close(fig)
print(args.run/'comparison.png')
print(args.run/'convergence.png')
