"""Visualize sample variability and the centre/neighbour quality trade-off."""
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
objects = json.loads((ROOT/'runs/public_object_sweep/summary.json').read_text())
analytic = json.loads((ROOT/'runs/analytic_consistency_sweep/summary.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
colors = ['#777777', '#0072B2', '#D55E00']
source_rows = []
for variant, label, color, marker in zip(['network', 'adam_30_anchor_0', 'adam_100_anchor_0'],
                                ['Network PNG', '30-step refinement', '100-step refinement'], colors, ['s', 'o', '^']):
    y = [case['variants'][variant]['psnr_mean_eyes'] for case in objects['cases']]
    axes[0].plot(np.arange(len(y)), y, marker=marker, linestyle='none', label=label, color=color)
    source_rows.extend({'panel': 'public', 'case': case['name'], 'variant': variant,
                        'fitted_psnr_dB': value, 'neighbour_6mm_psnr_dB': ''}
                       for case, value in zip(objects['cases'], y))
axes[0].set_xticks(np.arange(8), ['A/0', 'A/19', 'B/0', 'B/19', 'C/0', 'C/19', 'Room/0', 'Room/1499'], rotation=25)
axes[0].set(xlabel='Public samples (3 objects + 1 room scene)', ylabel='PNG mean-eye PSNR (dB)',
            title='Refinement gains across 8 public validation samples')
axes[0].grid(alpha=.2)
axes[0].legend(fontsize=8)

variants = [('network', 'Network', '#777777', 's'),
            ('adam_30_anchor_0', '30 steps', '#0072B2', 'o'),
            ('adam_100_anchor_0', '100 steps', '#0072B2', '^'),
            ('adam_100_anchor_0.1', '100 steps + pattern anchor 0.1', '#009E73', 'D'),
            ('consistency_100_weight_0.1', '100 steps + view consistency 0.1', '#D55E00', 'v'),
            ('consistency_100_weight_1', '100 steps + view consistency 1', '#CC79A7', 'P')]
for key, label, color, marker in variants:
    x = np.mean([case['variants'][key]['psnr_mean_eyes'] for case in analytic['cases']])
    y = np.mean([case['variants'][key]['neighbour_psnr_6mm'] for case in analytic['cases']])
    axes[1].scatter(x, y, color=color, marker=marker, s=75, label=label)
    source_rows.extend({'panel': 'analytic', 'case': case['name'], 'variant': key,
                       'fitted_psnr_dB': case['variants'][key]['psnr_mean_eyes'],
                       'neighbour_6mm_psnr_dB': case['variants'][key]['neighbour_psnr_6mm']}
                      for case in analytic['cases'])
axes[1].set(xlabel='Fitted centre-eye PSNR (dB)', ylabel='Held-out 6 mm neighbour PSNR (dB)',
            title='Quality trade-off: mean of 4 known-depth toy scenes')
axes[1].grid(alpha=.2)
axes[1].legend(fontsize=8, loc='lower right')
fig.suptitle('Development experiments using a prototype renderer', fontsize=13)
fig.tight_layout()
fig.savefig(ROOT/'runs/exploration_comparison.png', dpi=160)
with (ROOT/'runs/exploration_plot_data.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
    writer.writeheader()
    writer.writerows(source_rows)
(ROOT/'runs/exploration_plot_provenance.json').write_text(json.dumps({
    'sources': ['runs/public_object_sweep/summary.json', 'runs/analytic_consistency_sweep/summary.json'],
    'script': 'tools/plot_exploration.py', 'output': 'runs/exploration_comparison.png', 'dpi': 160,
    'pixel_dimensions': [2080, 720], 'public_panel': 'Each of 8 selected samples, no pooling.',
    'analytic_panel': 'Arithmetic mean over 4 fixed toy scenes, each scene averaged over eyes/views.',
    'uncertainty': 'Descriptive means only; no population confidence intervals. Individual values in CSV.',
    'matplotlib_version': matplotlib.__version__,
}, indent=2), encoding='utf-8')
print(ROOT/'runs/exploration_comparison.png')
