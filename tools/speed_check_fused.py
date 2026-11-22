"""Fused solver vs current torch solver on one machine: parity (mask, phase, gradient, PSNR) and timing.

python tools/speed_check_fused.py --root <dir with test.json> [--n 4]   (run from the repo root)
Per sample and branch (normal FOV40 / Kooima): SART-30 init, then Adam-100 by both solvers from the same init.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '.')
from PIL import Image
from lightfield_lab.exam_geometry import ExamRenderer, HEIGHT, WIDTH, validate_pose
from lightfield_lab.core import image_loss, refine, scores
from lightfield_lab.tomographic import sart
from lightfield_lab.v3_submission import offaxis_camera
from lightfield_lab import fused

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--n', type=int, default=4)
parser.add_argument('--steps', type=int, default=100)
args = parser.parse_args()
dev = torch.device('cuda')
rows = json.load(open(args.root/'test.json'))['samples']
rows = rows[::max(1, len(rows)//args.n)][:args.n]
t = time.perf_counter(); fused.backend(); print(f'[backend] {fused.backend()[0]} ready in {time.perf_counter()-t:.1f}s', flush=True)


def timed(fn):
    torch.cuda.synchronize(); t = time.perf_counter(); out = fn(); torch.cuda.synchronize()
    return out, time.perf_counter()-t


summary = []
for row in rows:
    imgs = [np.asarray(Image.open(args.root/row[s]).convert('RGB')) for s in ('left', 'right')]
    pl, pr = validate_pose(row['pose_l']), validate_pose(row['pose_r'])
    images = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(dev).float()/255
    k, cams = offaxis_camera(pl, pr, 'l2_camera')
    eye = (pl[:3, 3]+pr[:3, 3])/2
    rec = {'id': row['id'], 'eye_m': np.round(eye, 3).tolist()}
    for branch, poses, K in (('normal', np.stack([pl, pr]), None), ('kooima', cams, k)):
        r = ExamRenderer(poses, HEIGHT, WIDTH, dev, geometry_backend='torch', intrinsics=K)
        it = r.iteration_copy(torch.float32)
        init_ref, t_sart_ref = timed(lambda: sart(r, images, iterations=30).float().contiguous())
        fs, t_setup = timed(lambda: fused.FusedSolver(poses, images, K))
        init_f, t_sart_f = timed(lambda: fused.sart(fs, images, 30))
        mask_ok = bool((fs.count == r.geometry_mask[:, 0].sum((1, 2)).cpu().numpy()).all())
        ph_ref = it.accumulate(init_ref)
        msk = r.geometry_mask.bool().expand_as(ph_ref)
        phase_err = (fs.phase(init_ref)-ph_ref)[msk].abs().max().item()
        p = init_ref.clone().requires_grad_(True)
        image_loss(torch.sin(it.accumulate(p)*math.pi/2).square(), images, it.geometry_mask).backward()
        grad_err = ((fs.gradient(init_ref)-p.grad).norm()/p.grad.norm()).item()
        sart_diff = scores(r, r.quantize_patterns(init_f), images)['psnr_mean_eyes']-scores(r, r.quantize_patterns(init_ref), images)['psnr_mean_eyes']
        (best_ref, trace), t_ref = timed(lambda: refine(r, images, init_ref, steps=args.steps, lr=.08, iteration_dtype=torch.float32))
        (best_f, score_f, _), t_f = timed(lambda: fs.adam(init_ref, steps=args.steps, lr=.08))
        ref = scores(r, best_ref, images.double())['psnr_mean_eyes']
        ta = scores(r, r.quantize_patterns(best_f), images.double())['psnr_mean_eyes']
        g = init_ref.clone(); m = torch.zeros_like(g); v = torch.zeros_like(g)

        def steps():
            for s in range(1, 51):
                fs.k1(g, 0); fs._k2(g, m, v, None, .08, 0, s)
        _, t_steps = timed(steps)
        rec[branch] = dict(mask_ok=mask_ok, phase_err=phase_err, grad_rel_err=grad_err, sart_psnr_diff=sart_diff,
                           psnr_torch=ref, psnr_fused=ta, psnr_diff=ta-ref, radii=np.round(fs.radii_used, 2).reshape(-1).tolist(),
                           s_torch_sart=t_sart_ref, s_torch_adam=t_ref, s_fused_setup=t_setup, s_fused_sart=t_sart_f,
                           s_fused_adam=t_f, ms_fused_step=t_steps/50*1000)
        b = rec[branch]
        print(f'{row["id"]} {branch:6s} eye {rec["eye_m"]}: mask {"ok" if mask_ok else "MISMATCH"}, phase {phase_err:.1e}, '
              f'grad {grad_err:.1e}, PSNR torch {ref:.3f} fused {ta:.3f} (diff {ta-ref:+.4f}), SART diff {sart_diff:+.4f} | '
              f'torch sart {t_sart_ref:.3f}s adam {t_ref:.3f}s | fused setup {t_setup:.3f}s sart {t_sart_f:.3f}s adam {t_f:.3f}s '
              f'({b["ms_fused_step"]:.2f} ms/step)', flush=True)
    summary.append(rec)
out = Path('runs/speed_20261108'); out.mkdir(parents=True, exist_ok=True)
name = out/f'check_{args.root.parent.parent.name}_{args.root.name}_{"h100" if "H100" in torch.cuda.get_device_name() else "local"}.json'
name.write_text(json.dumps(summary, indent=1))
for branch in ('normal', 'kooima'):
    b = [r[branch] for r in summary[1:] or summary]  # first sample includes warm-up
    print(f'[{branch}] torch sart+adam {np.mean([x["s_torch_sart"]+x["s_torch_adam"] for x in b]):.3f}s, '
          f'fused setup+sart+adam {np.mean([x["s_fused_setup"]+x["s_fused_sart"]+x["s_fused_adam"] for x in b]):.3f}s, '
          f'{np.mean([x["ms_fused_step"] for x in b]):.3f} ms/step; worst |PSNR diff| {max(abs(x["psnr_diff"]) for x in summary for x in [x[branch]]):.4f} dB; '
          f'worst grad err {max(x[branch]["grad_rel_err"] for x in summary):.1e}; all masks ok {all(x[branch]["mask_ok"] for x in summary)}')
print('[saved]', name)
