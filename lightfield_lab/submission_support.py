"""Bundle the actual solver and reference with reproducible metadata."""
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torchvision

from .core import ROOT, UPSTREAM


def finish_delivery(out, predictor, metrics, args, started):
    out = Path(out)
    code = out/'code'
    destination = code/'lightfield_lab'
    destination.mkdir(exist_ok=True)
    for path in (ROOT/'lightfield_lab').glob('*.py'):
        if not path.name.startswith('test_'):
            shutil.copy2(path, destination/path.name)
    for directory in ['model', 'data', 'config']:
        target = code/'EyeReal'/directory
        target.mkdir(parents=True, exist_ok=True)
        for path in (UPSTREAM/directory).glob('*.py'):
            shutil.copy2(path, target/path.name)
    shutil.copy2(UPSTREAM/'LICENSE', code/'EyeReal/LICENSE')
    # The reference remains byte-for-byte unchanged in a separate file.
    reference = ROOT/'ta_reference.py' if (ROOT/'ta_reference.py').is_file() else ROOT/'testing/inference.py'
    shutil.copy2(reference, code/'ta_reference.py')
    scripts = ROOT/'solution' if (ROOT/'solution').is_dir() else ROOT
    for name in ['inference.sh', 'create_env.sh']:
        if (scripts/name).is_file():
            shutil.copy2(scripts/name, code/name)
    notice = ROOT/'solution/THIRD_PARTY_NOTICES.md'
    if not notice.is_file():
        notice = ROOT/'THIRD_PARTY_NOTICES.md'
    text = notice.read_text(encoding='utf-8') if notice.is_file() else '# Sources\n'
    text += ('\nEyeReal model/data/config source: https://github.com/WeijieMax/EyeReal, '
             'commit 833e24f10b91fabb499d3b3cc2ea137fbccd967d, AGPL-3.0; license in EyeReal/LICENSE.\n'
             'The supplied TA optical reference is retained unchanged as ta_reference.py.\n')
    (code/'THIRD_PARTY_NOTICES.md').write_text(text, encoding='utf-8')
    scorer = ('TA NumPy forward_render (unchanged reference)' if args.reference_score else
              'cached GPU ExamRenderer on the exported RGB8 patterns (TA NumPy parity: max abs error <1e-7, identical masks at 1080p; --reference-score uses the NumPy forward)')
    config = {'steps': predictor.steps, 'lr': predictor.lr, 'phase_steps': predictor.phase_steps,
              'anchor_weight': predictor.anchor_weight, 'time_limit_seconds': args.time_limit,
              'iteration_dtype': predictor.iteration_dtype,
              'geometry_backend': predictor.geometry_backend, 'quantized_tail': predictor.quantized_tail,
              'device': str(predictor.device), 'geometry': 'TA metres, exact .596736m width, +.06/0/-.06m',
              'network_input_order': 'EyeReal checkpoint: right eye first, back-to-front screens; outputs converted to TA L1/L2/L3',
              'self_score_renderer': scorer,
              'checkpoint': f'weights/{predictor.weights_path.name}'}
    (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    entrypoint = getattr(args, 'entrypoint', 'inference.py')
    command = (f'python code/{entrypoint} --index /path/to/test.json --out reproduced --group {args.group} '
               f'--weights weights/{predictor.weights_path.name} --steps {predictor.steps} '
               f'--lr {predictor.lr} --phase-steps {predictor.phase_steps} --anchor-weight {predictor.anchor_weight} '
               f'--time-limit {args.time_limit} --device {predictor.device} '
               f'--iteration-dtype {config["iteration_dtype"]} --geometry-backend {predictor.geometry_backend}'
               + (' --reference-score' if args.reference_score else '')
               + (' --quantized-tail' if predictor.quantized_tail else ''))
    if entrypoint != 'inference.py':
        shutil.copy2(scripts/entrypoint, code/entrypoint)
        command += ' --workers '+str(args.workers)
        launcher = code/'inference.sh'
        if launcher.exists():
            launcher.write_text(launcher.read_text(encoding='utf-8').replace('$ROOT/inference.py', '$ROOT/'+entrypoint), encoding='utf-8', newline='\n')
    (code/'README.md').write_text('# Reproduce\n\nRun from the group directory; replace the input manifest path.\n\n'
        f'```bash\npip install -r code/requirements.txt\n{command}\n```\n\n'
        'Install a CUDA-enabled PyTorch build compatible with the host driver for GPU inference. '
        'The input is external; model code and weights are included in this bundle.\n', encoding='utf-8')
    elapsed = time.perf_counter()-started
    result = {'scope': 'TA float optical reference on supplied inputs; not hidden depth scores.',
        'num_samples': len(metrics), 'mean_psnr_dB': float(np.mean([m['psnr_mean_eyes_dB'] for m in metrics])),
        'seconds_model_load': predictor.seconds_model_load,
        'seconds_entry_main_through_packaging': elapsed, 'time_limit_seconds': args.time_limit,
        'within_time_limit': elapsed <= args.time_limit,
        'timing_scope': 'Includes Torch/model load inside predictor, input/output, refinement, float scoring, format check and packaging; excludes Python process startup and imports before main.',
        'psnr_rule': 'Float eye images versus RGB8 target/255, TA hard mask, equal mean of eyes then samples; MSE floor 1e-12.',
        'self_score_renderer': scorer,
        'configuration': config, 'runtime': {'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'numpy': np.__version__,
            'device': str(predictor.device), 'gpu': torch.cuda.get_device_name(predictor.device) if predictor.device.type == 'cuda' else None},
        'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(), 'samples': metrics}
    (out/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    rows = '\n'.join(f'| {row["id"]} | {row["solver"]["network_initial_psnr_cached_dB"]:.3f} | '
                      f'{row["psnr_mean_eyes_dB"]:.3f} | {row["solver"]["refinement"]["steps_run"]} |' for row in metrics)
    report = f'''# EyeReal + TA geometry submission

EyeReal pretrained initialization followed by projected Adam on three RGB phase patterns.
Input c2w matrices are used directly in metres. TA L1/L2/L3 are front to back;
the checkpoint expects the right eye first and back-to-front screens, so network
inputs are reordered (each image keeps its pose) and outputs converted back.
Double-precision ray grids and the TA hard mask are cached; candidates are checked after RGB8 quantization.
Self-score renderer: {scorer}. Scores use float eyes, not PNG eye previews.

| Sample | Initial cached PSNR dB | Final PSNR dB | Refinement steps |
|---|---:|---:|---:|
{rows}

Mean PSNR: {result['mean_psnr_dB']:.3f} dB. Samples: {len(metrics)}, batch size 1.
Measured main-entry through packaging: {elapsed:.3f} s, hardware {result['runtime']['gpu'] or 'CPU'}.
The detailed timing scope, parameters and per-eye masks/errors are in metrics.json and config.json.
The global time budget is a soft allocation with an export reserve, not an OS-enforced deadline;
the measured within_time_limit flag is authoritative for this run.

This run makes no hidden-depth or off-axis bonus claim. Geometry supports supplied poses;
the pretrained inverse network has not been retrained for every possible viewing condition.
No training is performed during inference. EyeReal source and its AGPL-3.0 license are included.

```bash
{command}
```
'''
    (out/'REPORT.md').write_text(report, encoding='utf-8')
    demo = out/'demo/index.html'
    if demo.is_file():
        html = demo.read_text(encoding='utf-8')
        html = html.replace('提交格式示范', 'EyeReal图样与重建结果')
        html = html.replace('pattern为占位图，眼图为其光学前向结果，不代表实际预测方法的效果。',
                            '图样由EyeReal初值及逐例精修产生；眼图为浮点前向结果的量化预览（渲染器见metrics.json）。分数以metrics.json的浮点计分为准。')
        demo.write_text(html, encoding='utf-8')
    (out/'delivery.json').write_text(json.dumps({'format_demo': False, 'num_samples': len(metrics),
        'format_checked': True, 'solver_code_and_weights_bundled': True, 'within_time_limit': result['within_time_limit']}, indent=2), encoding='utf-8')
    print(f'[metrics] {result["mean_psnr_dB"]:.3f} dB, {elapsed:.3f}s including export and packaging', flush=True)
