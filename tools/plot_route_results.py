"""Research figures from immutable per-sample sweep records; no smoothing or hidden filtering."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sweep', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
rows = [json.loads(s) for s in args.sweep.read_text().splitlines() if s.strip()]
table = [{'id': r['id'], 'subset': r['evaluation_subset'], 'method': name, **v}
         for r in rows for name, v in r['methods'].items()]
with (args.out/'source_data.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'subset', 'method', 'psnr_dB', 'seconds', 'selected_step'], extrasaction='ignore')
    writer.writeheader(); writer.writerows(table)
with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'}):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained', sharex=True, sharey=True)
    for ax, subset in zip(axes.flat, ('train_tune', 'train_verify', 'val', 'test')):
        selected = [r for r in rows if r['evaluation_subset'] == subset]
        methods = ['baseline30', 'fixed08_100', 'fixed08_150', 'fixed08_200']
        for statistic, function, color, marker, linestyle in [('Mean', np.mean, '#0072B2', 'o', '-'),
                ('10th percentile', lambda v: np.percentile(v, 10), '#009E73', 's', '--'),
                ('Minimum', np.min, '#D55E00', '^', ':')]:
            values = [function([r['methods'][m]['psnr_dB'] for r in selected]) for m in methods]
            ax.plot([30, 100, 150, 200], values, color=color, marker=marker, linestyle=linestyle, label=statistic)
        ax.set_title(f'{subset} (n={len(selected)} pairs)')
        ax.set(xlabel='Iterations', ylabel='Centre PSNR (dB)', xticks=[30, 100, 150, 200])
        ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Fixed lr=0.08 at 100/150/200 steps; baseline: 30 steps, lr=0.02\nLines connect measured operating points; percentiles show sample spread, not uncertainty')
    for suffix in ('png', 'svg'):
        fig.savefig(args.out/f'operating_points.{suffix}', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    tune = [r for r in rows if r['evaluation_subset'] == 'train_tune']
    for prefix, label, color, marker in [('fixed08_', 'Fixed 0.08', '#0072B2', 'o'), ('decay08_02_', '0.08 then 0.02', '#D55E00', '^')]:
        ax.plot([100, 150, 200], [np.mean([r['methods'][prefix+str(n)]['psnr_dB'] for r in tune]) for n in (100, 150, 200)],
                color=color, marker=marker, label=label)
    ax.set(xlabel='Iteration budget', ylabel='Mean centre PSNR (dB)', xticks=[100, 150, 200],
           title=f'Schedule selection on train_tune only (n={len(tune)})')
    ax.legend(frameon=False); ax.grid(alpha=.2)
    for suffix in ('png', 'svg'):
        fig.savefig(args.out/f'schedule_comparison.{suffix}', dpi=180)
    plt.close(fig)
(args.out/'provenance.json').write_text(json.dumps({'source': str(args.sweep.resolve()),
    'source_sha256': hashlib.sha256(args.sweep.read_bytes()).hexdigest(), 'unit': 'stereo scene/view pair',
    'selection': 'none excluded; subsets retained separately; fixed candidates selected on train_tune',
    'transformations': ['arithmetic mean in dB', '10th percentile across pairs', 'minimum across pairs'],
    'uncertainty': 'none estimated; percentiles represent observed distribution',
    'destination': 'project research report, not a journal submission', 'matplotlib': matplotlib.__version__}, indent=2))
