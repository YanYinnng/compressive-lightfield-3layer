"""Export descriptive focus comparisons with per-scene data and provenance."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'runs/focus_textured_evaluation/summary.json'
report = json.loads(source.read_text())
methods = ['network', 'adam_30_anchor_0', 'adam_100_anchor_0', 'adam_100_anchor_0.1',
           'consistency_100_weight_0.1', 'consistency_100_weight_1']
labels = ['Network', 'Adam 30', 'Adam 100', 'Anchor 0.1', 'Teacher 0.1', 'Teacher 1']
colors = ['#666666', '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#8C6500']
markers = ['s', 'o', '^', 'D', 'v', 'P']
rows = []
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), layout='constrained')
for index, method in enumerate(methods):
    center = [case['native_center_scores'][method]['psnr_dB'] for case in report['cases']]
    axes[0].scatter(np.full(len(center), index), center, color=colors[index], marker=markers[index], alpha=.35, s=25)
    axes[0].scatter(index, np.mean(center), color=colors[index], marker=markers[index], s=80)
    for diameter, offset, fill in [('4.0', -.12, 'none'), ('6.0', .12, colors[index])]:
        value = report['aggregate'][diameter][method]
        axes[1].scatter(index+offset, value['stack_psnr_mean_dB'], marker=markers[index], s=80,
                        facecolors=fill, edgecolors=colors[index])
        axes[2].scatter(index+offset, value['focus_response_nrmse_mean'], marker=markers[index], s=80,
                        facecolors=fill, edgecolors=colors[index])
    for case in report['cases']:
        for eye in case['eyes']:
            metric = eye['variants'][method]
            observable = [v['focus_response_nrmse'] for k, v in metric['rois'].items()
                          if k != 'background' and v.get('focus_response_nrmse') is not None]
            rows.append({'case': case['name'], 'eye': eye['eye'], 'pupil_diameter_mm': eye['diameter_mm'],
                'method': method, 'center_psnr_dB': case['native_center_scores'][method]['psnr_dB'],
                'stack_psnr_dB': metric['stack_psnr_mean_dB'],
                'response_nrmse_mean_over_identifiable_rois': np.mean(observable) if observable else '',
                'identifiable_rois': len(observable)})
axes[0].set(title='Fitted binocular images', ylabel='Mean-eye PSNR (dB)')
axes[1].set(title='Full focal-stack fidelity', ylabel='Mean PSNR (dB)')
axes[2].set(title='Focus-dependent response only', ylabel='Response NRMSE (lower is better)')
axes[2].axhline(1, color='#555555', linestyle='--', linewidth=1, label='Focus-invariant response = 1')
axes[2].legend(fontsize=8, loc='best')
for ax in axes:
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha='right')
    ax.grid(axis='y', alpha=.2)
fig.suptitle('4 known-depth textured scenes, 960 x 540; saved 8-bit patterns fixed throughout', fontsize=13)
fig.supxlabel('Open markers: 4 mm pupil; filled: 6 mm. Left panel: per-scene points and mean.', fontsize=10)
fig.savefig(ROOT/'runs/focus_comparison.png', dpi=160)
plt.close(fig)

# Preselect nearest of the four camera configurations, left eye, 6 mm.
# Curves are shifted by their own maxima to expose focus shape, not absolute fidelity.
case = report['cases'][3]
eye = next(e for e in case['eyes'] if e['eye'] == 0 and e['diameter_mm'] == 6)
depth_axis = np.array(eye['distances_cm'])
depth_axis -= depth_axis[len(depth_axis)//2]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout='constrained', sharey=True)
for axis, label in zip(axes, ['near', 'middle', 'far']):
    truth = np.array(eye['variants']['network']['rois'][label]['gt_curve_psnr_dB'])
    axis.plot(depth_axis, truth-truth.max(), color='black', linewidth=2, label='3D ground truth')
    for index in [0, 2, 3, 4, 5]:
        curve = np.array(eye['variants'][methods[index]]['rois'][label]['curve_psnr_dB'])
        axis.plot(depth_axis, curve-curve.max(), color=colors[index], marker=markers[index], markevery=4,
                  markersize=4, label=labels[index])
    axis.set(title=label.capitalize()+' card', xlabel='Focus distance offset from screen centre (cm)')
    axis.grid(alpha=.2)
axes[0].set_ylabel('ROI PSNR minus its own peak (dB)')
axes[-1].legend(fontsize=8)
fig.suptitle('Focus selectivity: scene 3, left eye, 6 mm pupil; each curve normalized separately', fontsize=12)
fig.savefig(ROOT/'runs/focus_curves.png', dpi=160)
plt.close(fig)
with (ROOT/'runs/focus_plot_data.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(ROOT/'runs/focus_plot_provenance.json').write_text(json.dumps({
    'source': str(source.relative_to(ROOT)), 'script': 'tools/plot_focus_results.py', 'dpi': 160,
    'aggregation': 'Descriptive means only, 4 fixed scenes; not population confidence intervals.',
    'response': 'Equal ROI means over GT-observable cards; 4 and 6 mm can contain different ROI sets.',
    'curve_selection': 'Scene 3 (nearest camera configuration), left eye, 6 mm; no best-method selection.',
    'curve_normalization': 'Subtract each curve own maximum; absolute fidelity is in overview and JSON.',
    'matplotlib_version': matplotlib.__version__,
}, indent=2), encoding='utf-8')
print(ROOT/'runs/focus_comparison.png')
print(ROOT/'runs/focus_curves.png')
