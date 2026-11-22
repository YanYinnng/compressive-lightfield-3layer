"""T1-3: predeclared schedule candidates, selection on tune only, frozen confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.core import refine, scores
from lightfield_lab.exam_geometry import ExamRenderer, HEIGHT, WIDTH
from lightfield_lab.exam_data import read_manifest, read_rgb8
from tools.overnight_eval import Context, evaluation_subset, append, done_ids, to_images

CONFIGS = {
    'baseline30': {'steps': 30, 'lr': .02},
    'fixed05_100': {'steps': 100, 'lr': .05},
    **{f'fixed08_{n}': {'steps': n, 'lr': .08} for n in (100, 150, 200)},
    **{f'decay08_02_{n}': {'steps': n, 'lr': .08, 'lr_schedule': [[at, .02]]}
       for n, at in ((100, 61), (150, 101), (200, 151))},
}


def summarize(rows):
    methods = sorted({m for r in rows for m in r['methods']})
    return {m: {'samples': sum(m in r['methods'] for r in rows),
                'mean_dB': float(np.mean([r['methods'][m]['psnr_dB'] for r in rows if m in r['methods']])),
                'min_dB': float(np.min([r['methods'][m]['psnr_dB'] for r in rows if m in r['methods']])),
                'p10_dB': float(np.percentile([r['methods'][m]['psnr_dB'] for r in rows if m in r['methods']], 10)),
                'mean_s': float(np.mean([r['methods'][m]['seconds'] for r in rows if m in r['methods']]))}
            for m in methods}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lego', type=Path, default=ROOT/'datasets/lego_v1')
    parser.add_argument('--weights', type=Path, default=ROOT/'weights/pretrained_model.pth')
    parser.add_argument('--limit', type=int, default=None, help='Smoke check only: at most N per subset')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = {'candidates': CONFIGS, 'limit': args.limit, 'source': str(args.lego.resolve()),
              'weights_sha256': hashlib.file_digest(args.weights.open('rb'), 'sha256').hexdigest(),
              'selection': 'Highest train_tune mean for each budget, tie by minimum; freeze before verification.'}
    path = args.out/'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise SystemExit('Resume configuration differs; choose a new output.')
    path.write_text(json.dumps(config, indent=2))
    manifest, rows = read_manifest(args.lego/'all.json')
    context = Context(args)
    records_path = args.out/'records.jsonl'
    records = [json.loads(s) for s in records_path.read_text().splitlines() if s.strip()] if records_path.exists() else []
    finished = {r['id'] for r in records}
    choice_path = args.out/'choices.json'
    for subset in ('train_tune', 'train_verify', 'val', 'test'):
        selected_rows = [r for r in rows if evaluation_subset(r) == subset][:args.limit]
        if subset == 'train_tune':
            names = list(CONFIGS)
        else:
            frozen = json.loads(choice_path.read_text())
            names = list(dict.fromkeys(['baseline30', *frozen['selected'].values()]))
        for row in selected_rows:
            if row['id'] in finished:
                continue
            images = to_images([read_rgb8(manifest.parent/row[k]) for k in ('left', 'right')], context.device)
            poses = np.array([row['pose_l'], row['pose_r']])
            initial = context.initial(images, poses)
            renderer = ExamRenderer(poses, HEIGHT, WIDTH, context.device, geometry_backend='torch')
            methods = {}
            # One fixed-lr trajectory supplies its 100/150/200 checkpoints on tune.
            if subset == 'train_tune':
                captured = {}
                started = time.perf_counter()
                def capture(step, patterns, score):
                    if step in (100, 150, 200):
                        captured[f'fixed08_{step}'] = {'psnr_dB': score, 'seconds': time.perf_counter()-started}
                refine(renderer, images, initial, steps=200, lr=.08, iteration_dtype=torch.float32,
                       checkpoint_callback=capture)
                methods.update(captured)
            for name in names:
                if name in methods:
                    continue
                started = time.perf_counter()
                patterns, trace = refine(renderer, images, initial, iteration_dtype=torch.float32, **CONFIGS[name])
                methods[name] = {'psnr_dB': scores(renderer, patterns, images)['psnr_mean_eyes'],
                                 'seconds': time.perf_counter()-started, 'selected_step': trace['selected_step']}
            record = {'id': row['id'], 'split': row['split'], 'evaluation_subset': subset, 'methods': methods}
            append(records_path, record); records.append(record); finished.add(row['id'])
            print(json.dumps({'id': row['id'], 'subset': subset, 'completed': len(records)}), flush=True)
        subset_records = [r for r in records if r['evaluation_subset'] == subset]
        summary = summarize(subset_records)
        (args.out/f'{subset}_summary.json').write_text(json.dumps(summary, indent=2))
        if subset == 'train_tune':
            choices = {str(n): max([m for m in CONFIGS if CONFIGS[m]['steps'] == n],
                                  key=lambda m: (summary[m]['mean_dB'], summary[m]['min_dB'])) for n in (100, 150, 200)}
            frozen = {'selected': choices, 'tune_samples': len(subset_records), 'selection_summary': summary,
                      'candidate_configs': CONFIGS}
            if choice_path.exists() and json.loads(choice_path.read_text()) != frozen:
                raise SystemExit('Frozen choices differ on resume.')
            choice_path.write_text(json.dumps(frozen, indent=2))
    (args.out/'complete.json').write_text(json.dumps({'state': 'SUCCEEDED', 'samples': len(records)}))


if __name__ == '__main__':
    main()
