"""EyeReal-based TA-compatible submission. Optical forward is unchanged."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image

HEIGHT, WIDTH = 1080, 1920


def create_predictor(weights: str | None, **options):
    """Initialize your method and return predict(left_rgb,right_rgb,pose_l,pose_r).

    Inputs: uint8 HWC RGB images and 4x4 c2w arrays in meters.
    Outputs: three uint8 HWC RGB images, ordered L1 (nearest), L2, L3.
    The published EyeReal checkpoint initializes patterns; the local solver
    refines them using the TA geometry. Optical forward is preserved below.
    """
    root = Path(__file__).resolve().parent
    if not (root/'lightfield_lab').is_dir():
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lightfield_lab.exam_adapter import ExamPredictor
    return ExamPredictor(weights, **options)


def read_rows(index):
    text = index.read_text(encoding='utf-8')
    data = [json.loads(s) for s in text.splitlines() if s.strip()] if index.suffix.lower()=='.jsonl' else json.loads(text)
    rows = data['samples'] if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty samples list')
    for row in rows:
        if not isinstance(row,dict) or any(k not in row for k in ('id','left','right','pose_l','pose_r')):
            raise ValueError('Missing input fields')
        if not isinstance(row['id'],str) or not row['id']:
            raise ValueError('id must be a nonempty string')
    if len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate IDs')
    return rows


def read_rgb(path):
    with Image.open(path) as image:
        if image.mode!='RGB' or image.size!=(WIDTH,HEIGHT):
            raise ValueError(f'Expected {WIDTH}x{HEIGHT} RGB: {path}')
        return np.asarray(image).copy()


def read_pose(value):
    pose=np.asarray(value,dtype=np.float64)
    if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]):
        raise ValueError('Expected finite 4x4 c2w, last row [0,0,0,1]')
    return pose


def _bilinear_rgb(image, u, v):
    """Screen-edge pixel coordinates -> RGB; pixel centers are (i+.5,j+.5).

    This is bilinear sampling with zero padding, equivalent to
    grid_sample(..., align_corners=False). No color/Gamma conversion is applied.
    """
    height, width = image.shape[:2]
    x, y = u - .5, v - .5
    support = np.isfinite(x) & np.isfinite(y) & (x > -1) & (x < width) & (y > -1) & (y < height)
    x, y = np.where(support, x, 0), np.where(support, y, 0)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    ax, ay = x - x0, y - y0
    result = np.zeros(x.shape + (3,), dtype=np.float64)
    for ox, oy, weight in ((0, 0, (1-ax)*(1-ay)), (1, 0, ax*(1-ay)),
                           (0, 1, (1-ax)*ay), (1, 1, ax*ay)):
        ix, iy = x0 + ox, y0 + oy
        valid = support & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        values = image[np.clip(iy, 0, height-1), np.clip(ix, 0, width-1)]
        result += values * (weight * valid)[..., None]
    return result


def forward_render(patterns_rgb, pose_l, pose_r, tile_rows=64):
    """Three existing RGB patterns -> left/right floating-point eye images.

    patterns_rgb: list/array (3,H,W,3), L1 nearest and L3 farthest;
      uint8 values are divided by 255, floating values must already be in [0,1].
    pose_l/pose_r: 4x4 c2w in meters, columns right/up/back; view direction -z.
    Returns eyes (2,H,W,3) float32 in [0,1], masks (2,H,W) bool. The mask
      requires positive ray intersections inside all three physical screens.

    Fixed exam geometry: horizontal FOV40, screen width .596736m,
      L1/L2/L3 x=(+.06,0,-.06)m; screen height follows the image aspect ratio.
    For each RGB channel: I=sin((pi/2)*sum_i bilinear(g_i, hit_i))**2.
    No sigmoid, per-image normalization, Gamma correction or network is used.
    """
    layers = np.asarray(patterns_rgb)
    if layers.ndim != 4 or layers.shape[0] != 3 or layers.shape[-1] != 3:
        raise ValueError('patterns_rgb must have shape (3,H,W,3), ordered L1/L2/L3')
    if layers.dtype == np.uint8:
        layers = layers.astype(np.float64) / 255.0
    elif np.issubdtype(layers.dtype, np.floating):
        if not np.isfinite(layers).all() or np.any(layers < 0) or np.any(layers > 1):
            raise ValueError('Floating patterns must be finite and in [0,1]')
        layers = layers.astype(np.float64, copy=False)
    else:
        raise ValueError('Patterns must be uint8 RGB or floating RGB in [0,1]')
    _, height, width, _ = layers.shape
    if min(height, width, tile_rows) < 1:
        raise ValueError('Image dimensions and tile_rows must be positive')
    screen_width = .596736
    screen_height = screen_width * height / width
    focal = width / (2 * np.tan(np.deg2rad(40.0) / 2))
    local_x = (np.arange(width, dtype=np.float64) + .5 - width/2) / focal
    eyes = np.empty((2, height, width, 3), dtype=np.float32)
    masks = np.empty((2, height, width), dtype=bool)
    for eye, pose in enumerate((read_pose(pose_l), read_pose(pose_r))):
        rotation, origin = pose[:3, :3], pose[:3, 3]
        for top in range(0, height, tile_rows):
            bottom = min(top + tile_rows, height)
            local_y = -(np.arange(top, bottom, dtype=np.float64) + .5 - height/2) / focal
            directions = [rotation[j, 0]*local_x[None, :] + rotation[j, 1]*local_y[:, None] - rotation[j, 2]
                          for j in range(3)]
            dx, dy, dz = directions
            parallel = np.abs(dx) < 1e-9
            phase_sum = np.zeros((bottom-top, width, 3), dtype=np.float64)
            valid_all = np.ones((bottom-top, width), dtype=bool)
            for layer, layer_x in enumerate((.06, 0., -.06)):
                distance = np.divide(layer_x-origin[0], dx, out=np.zeros_like(dx), where=~parallel)
                hit_y, hit_z = origin[1] + distance*dy, origin[2] + distance*dz
                u = (hit_y + screen_width/2) / screen_width * width
                v = (screen_height/2 - hit_z) / screen_height * height
                valid_all &= (~parallel) & (distance > 0) & (u >= 0) & (u <= width) & (v >= 0) & (v <= height)
                phase_sum += _bilinear_rgb(layers[layer], u, v)
            eyes[eye, top:bottom] = np.sin((np.pi/2) * phase_sum)**2
            masks[eye, top:bottom] = valid_all
    return eyes, masks


def save_eye_images(out, number, layers, pose_l, pose_r, return_arrays=False):
    """Save previews; the function forward_render retains floating precision."""
    eyes, masks = forward_render(layers, pose_l, pose_r)
    record = save_eye_arrays(out, number, eyes, masks)
    return (record, eyes, masks) if return_arrays else record


def save_eye_arrays(out, number, eyes, masks):
    """Quantized previews of already-rendered float eyes and boolean masks."""
    (out/'eye_images').mkdir(exist_ok=True)
    (out/'masks').mkdir(exist_ok=True)
    record = {}
    for eye, name in enumerate(('left', 'right')):
        relative = f'eye_images/sample_{number:06d}_{name}.png'
        image = np.floor(np.clip(eyes[eye], 0, 1)*255 + .5).astype(np.uint8)
        Image.fromarray(image).save(out/relative)
        mask_path = f'masks/sample_{number:06d}_{name}.png'
        Image.fromarray(masks[eye].astype(np.uint8)*255).save(out/mask_path)
        record[name], record[f'mask_{name}'] = relative, mask_path
    return record


def render_submission(index, rows, submission, out):
    """Read already-saved RGB pattern PNGs, without invoking create_predictor."""
    source = Path(__file__).resolve().parent
    subprocess.run([sys.executable, '-B', str(source/'check_submission.py'), '--index', str(index),
                    '--submission', str(submission)], check=True)
    manifest = json.loads(submission.read_text(encoding='utf-8'))
    patterns_by_id = {row['id']: row for row in manifest['samples']}
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new empty --out directory')
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for number, row in enumerate(rows):
        pattern = patterns_by_id[row['id']]
        layers = [read_rgb(submission.parent/pattern[f'layer{k}']) for k in (1,2,3)]
        record = save_eye_images(out, number, layers, read_pose(row['pose_l']), read_pose(row['pose_r']))
        records.append(dict(id=row['id'], **record))
    (out/'eye_images.json').write_text(json.dumps({'samples':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[forward] {len(records)} RGB pattern sets -> {out}')


def main():
    started = time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',required=True)
    parser.add_argument('--out',required=True,help='New empty submission directory')
    parser.add_argument('--group',default='group_001')
    parser.add_argument('--weights',help='Optional weights for your own implementation')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--phase-steps', type=int, default=0)
    parser.add_argument('--lr', type=float, default=.02)
    parser.add_argument('--quantized-tail', action='store_true', help='Research candidate: freeze RGB8 L1/L2 in the last 30 steps')
    parser.add_argument('--anchor-weight', type=float, default=0.)
    parser.add_argument('--iteration-dtype', choices=['legacy', 'float32', 'float64'], default='float32',
                        help='legacy: original baseline (float32 parameters, float64 sampling); '
                             'float32/float64: parameters, sampling and loss all in that precision')
    parser.add_argument('--geometry-backend', choices=['numpy', 'torch'], default='torch')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None)
    parser.add_argument('--time-limit', type=float, default=600., help='Whole-run soft budget, including model load and export')
    parser.add_argument('--reference-score', action='store_true',
                        help='Self-score with the TA NumPy forward (slow, ~5-12 s/sample) instead of the parity-checked GPU renderer')
    parser.add_argument('--format-demo',action='store_true',help='Placeholder images only; no prediction algorithm')
    parser.add_argument('--submission',help='Existing patterns.json: render RGB patterns to eye images only')
    args=parser.parse_args()
    if not args.group or args.group in ('.','..') or any(c in args.group for c in '/\\:'):
        parser.error('group must be a single directory name')
    if args.format_demo and args.weights:
        parser.error('Format demo does not load weights')
    index=Path(args.index).resolve()
    rows=read_rows(index)
    if args.submission:
        if args.format_demo or args.weights:
            parser.error('--submission cannot be combined with prediction options')
        render_submission(index, rows, Path(args.submission).resolve(), Path(args.out).resolve())
        return
    out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Choose a new empty --out directory')
    predictor=None if args.format_demo else create_predictor(args.weights, steps=args.steps, lr=args.lr,
        phase_steps=args.phase_steps, anchor_weight=args.anchor_weight, device=args.device,
        iteration_dtype=args.iteration_dtype, geometry_backend=args.geometry_backend, quantized_tail=args.quantized_tail)
    if predictor is not None:
        from lightfield_lab.exam_data import read_manifest, read_rgb8
        from lightfield_lab.exam_geometry import masked_psnr
        index, rows = read_manifest(index)
        predictor.configure_run(started+args.time_limit, len(rows), export_seconds=15. if args.reference_score else 3.)
        args.weights = str(predictor.weights_path)
    (out/'patterns').mkdir(parents=True)
    outputs=[]
    eye_records=[]
    metrics=[]
    for number,row in enumerate(rows):
        sample_started = time.perf_counter()
        reader = read_rgb8 if predictor is not None else read_rgb
        left,right=reader(index.parent/row['left']),reader(index.parent/row['right'])
        pose_l,pose_r=read_pose(row['pose_l']),read_pose(row['pose_r'])
        layers=([np.full((HEIGHT,WIDTH,3),128,dtype=np.uint8) for _ in range(3)]
                if args.format_demo else predictor(left,right,pose_l,pose_r))
        if len(layers)!=3:
            raise ValueError('Return exactly three layers')
        result={'id':row['id']}
        for layer,value in enumerate(layers,1):
            image=np.asarray(value)
            if image.shape!=(HEIGHT,WIDTH,3) or image.dtype!=np.uint8:
                raise ValueError('Each layer must be 1920x1080 HWC uint8 RGB')
            relative=f'patterns/sample_{number:06d}_layer{layer}.png'
            Image.fromarray(image).save(out/relative)
            result[f'layer{layer}']=relative
        outputs.append(result)
        if predictor is not None and not args.reference_score:
            eyes, masks = predictor.last_eyes, predictor.last_masks
            eye_record = save_eye_arrays(out, number, eyes, masks)
        else:
            eye_record, eyes, masks = save_eye_images(out,number,layers,pose_l,pose_r,return_arrays=True)
        eye_records.append(dict(id=row['id'], **eye_record))
        if predictor is not None:
            metric = masked_psnr(eyes, np.stack([left,right]).astype(np.float64)/255, masks)
            elapsed = time.perf_counter()-sample_started
            predictor.note_export(elapsed-predictor.records[-1]['seconds_predictor_total'])
            metrics.append(dict(id=row['id'], **metric, solver=predictor.records[-1],
                                seconds_sample_io_predict_render_score=elapsed))
            print(f'[sample] {row["id"]}: {metric["psnr_mean_eyes_dB"]:.3f} dB', flush=True)
    (out/'patterns.json').write_text(json.dumps({'group':args.group,'samples':outputs},ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'eye_images.json').write_text(json.dumps({'samples':eye_records},ensure_ascii=False,indent=2),encoding='utf-8')
    code=out/'code'
    code.mkdir()
    source=Path(__file__).resolve().parent
    for name in ('inference.py','check_submission.py','requirements.txt','LICENSE'):
        if (source/name).is_file(): shutil.copy2(source/name,code/name)
    option=' --format-demo' if args.format_demo else ''
    if args.weights:
        weight=out/'weights'/Path(args.weights).name
        weight.parent.mkdir()
        shutil.copy2(args.weights,weight)
        option+=f' --weights weights/{weight.name}'
    command=f'python code/inference.py --index /path/to/test.json --out reproduced --group {args.group}{option}'
    (code/'README.md').write_text('# 复现命令\n\n从本提交目录运行，替换输入路径。\n\n```bash\n'+command+'\n```\n',encoding='utf-8')
    description=('本目录仅演示提交格式，pattern为固定占位图；眼图由这些pattern经过题面光学前向计算，未运行图样预测模型，不作为正式答案。' if args.format_demo else '请填写与最终实现一致的方法说明。')
    (out/'REPORT.md').write_text('# 方法说明\n\n'+description+f'\n\n包含{len(outputs)}对样本、{3*len(outputs)}张图样。\n\n正式报告须说明自己的方法、数据、参数、依赖、必要权重、运行命令及实际验证结果。\n\n```bash\n'+command+'\n```\n',encoding='utf-8')
    demo=out/'demo'
    demo.mkdir()
    sections=[]
    for row, eye_row in zip(outputs[:12],eye_records[:12]):
        images=''.join(f'<div>L{k}<img src="../{html.escape(row[f"layer{k}"],quote=True)}"></div>' for k in (1,2,3))
        eye_previews=''.join(f'<div>{name}<img src="../{eye_row[name]}"></div>' for name in ('left','right'))
        sections.append('<h2>'+html.escape(row['id'])+'</h2><section>'+images+'</section><section>'+eye_previews+'</section>')
    (demo/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>提交格式示范</title><style>body{font:16px system-ui;margin:24px}section{display:flex;gap:12px}section div{width:32%}img{width:100%}</style><h1>提交格式示范</h1><p>L1最近，L3最远。pattern为占位图，眼图为其光学前向结果，不代表实际预测方法的效果。</p>'+''.join(sections)+'</html>',encoding='utf-8')
    subprocess.run([sys.executable,'-B',str(source/'check_submission.py'),'--index',str(index),'--submission',str(out/'patterns.json')],check=True)
    (out/'delivery.json').write_text(json.dumps({'format_demo':args.format_demo,'num_samples':len(outputs),'format_checked':True},indent=2),encoding='utf-8')
    if predictor is not None:
        from lightfield_lab.submission_support import finish_delivery
        finish_delivery(out, predictor, metrics, args, started)
    print(f'[inference] {len(outputs)} pairs -> {out}')


if __name__=='__main__':
    main()
