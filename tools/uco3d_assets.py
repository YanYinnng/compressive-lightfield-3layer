"""uCO3D Gaussian splats -> Graphdeco PLY + world placement, for multi-object benchmarks.

Source: facebook/uco3d on Hugging Face, examples.zip (the 52-video preview subset, CC BY 4.0;
EyeReal's uCO3D data were rendered from the same per-video splats). Only the
<category>/<sequence>/gaussian_splats/ folders and metadata.sqlite are needed.

  convert  decode each compressed splat folder (the decoding of uco3d.dataset_utils.gauss3d_utils:
           16-bit PNG means with inverse log transform, 8-bit PNG scales/quats/opacities/sh0,
           k-means shN, npz fg_mask), keep the foreground Gaussians and write <out>/<sequence>.ply
           in native coordinates (gsplat parameters are the Graphdeco ones: log scales, opacity
           logits, wxyz quaternions, SH coefficients). <sequence>.json holds the placement that
           GaussianTargetRenderer applies: world = scale_m * rotation * (native - center_native),
           composed of uCO3D's alignment (x_al = s*(R^T x + T)), the aligned up axis -> world z
           (TA world: x toward viewer, y screen right, z up, origin O2), and a uniform scale that
           makes the robust foreground extent (1-99 percentile, opacity > 0.5) --size metres along
           its longest side (Lego: 0.27 m), centred on O2. SH stay native: cameras are moved into
           the scene instead.
  preview  front views of every converted asset at 1 m (H100, gsplat) as one contact sheet.
"""
import argparse
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AXES = {'+x': (0, 1.), '-x': (0, -1.), '+y': (1, 1.), '-y': (1, -1.), '+z': (2, 1.), '-z': (2, -1.)}


def _unit(name):
    axis, sign = AXES[name]
    v = np.zeros(3)
    v[axis] = sign
    return v


def _png(path):
    return np.asarray(Image.open(path)).astype(np.float64)


def decode(folder):
    """Compressed uCO3D splats -> dict of numpy arrays (gsplat parameter domains)."""
    meta = json.loads((folder/'meta.json').read_text())
    n = meta['means']['shape'][0]
    meta['sh0']['shape'] = [n, 3]
    out = {}
    for name, m in meta.items():
        if not np.all(m['shape']):
            out[name] = np.zeros(m['shape'], dtype=np.float32)
            continue
        mins, maxs = np.asarray(m.get('mins', 0.)), np.asarray(m.get('maxs', 1.))  # fg_mask has none
        if name == 'means':
            grid = ((_png(folder/'means_u.png').astype(np.uint16) << 8) + _png(folder/'means_l.png'))/(2**16-1)
            value = grid*(maxs-mins)+mins
            value = np.sign(value)*np.expm1(np.abs(value))  # inverse log transform
        elif name == 'shN':
            data = np.load(folder/'shN.npz')
            value = (data['centroids']/(2**m['quantization']-1)*(maxs-mins)+mins)[data['labels'].astype(np.int64)]
        elif name == 'fg_mask':
            value = np.load(folder/'fg_mask.npz')['arr']
        else:
            value = _png(folder/f'{name}.png')/(2**8-1)*(maxs-mins)+mins
        out[name] = np.asarray(value, dtype=np.float32).reshape(m['shape'])
    return out


def write_ply(path, splats, keep):
    sh0 = splats['sh0'][keep].reshape(-1, 3)
    shn = splats['shN'][keep]
    shn = shn.reshape(len(shn), -1, 3).transpose(0, 2, 1).reshape(len(shn), -1)  # channel-major f_rest
    columns = ([splats['means'][keep], np.zeros((int(keep.sum()), 3), np.float32), sh0, shn,
                splats['opacities'][keep].reshape(-1, 1), splats['scales'][keep], splats['quats'][keep]])
    names = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(shn.shape[1])]
             + ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
    data = np.concatenate(columns, axis=1).astype('<f4')
    assert data.shape[1] == len(names)
    header = ('ply\nformat binary_little_endian 1.0\n' + f'element vertex {len(data)}\n'
              + ''.join(f'property float {n}\n' for n in names) + 'end_header\n')
    with open(path, 'wb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(np.ascontiguousarray(data).tobytes())


def alignments(sqlite_path):
    rows = sqlite3.connect(sqlite_path).execute(
        'select sequence_name, category, super_category, _gaussian_splats_dir, _alignment_R, _alignment_T, _alignment_scale,'
        ' _reconstruction_quality_gaussian_splats_psnr, _short_caption_text from sequence_annots')
    out = {}
    for name, category, super_category, folder, r, t, s, psnr, caption in rows:
        out[name] = {'category': category, 'super_category': super_category, 'dir': folder,
                     'R': np.frombuffer(r, '<f4').reshape(3, 3).astype(np.float64), 'T': np.frombuffer(t, '<f4').astype(np.float64),
                     's': float(s), 'splat_psnr_dB': psnr, 'caption': caption}
    return out


def cmd_convert(args):
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    up, front = _unit(args.up), _unit(args.front)
    assert abs(up@front) < 1e-9
    world_from_aligned = np.stack([front, np.cross(up, front), up])  # rows: world x (toward viewer), y, z (up)
    index = []
    for name, a in sorted(alignments(src/'metadata.sqlite').items()):
        folder = src/a['dir']
        if not (folder/'meta.json').is_file():
            continue
        splats = decode(folder)
        fg = splats['fg_mask'].reshape(-1) > .5 if 'fg_mask' in splats else np.ones(len(splats['means']), bool)
        R, T, s = a['R'], a['T'], a['s']
        u, _, vt = np.linalg.svd(R)  # stored as float32: snap to the nearest exact rotation
        R = u@vt
        if not np.isclose(np.linalg.det(R), 1, atol=1e-4):
            print(f'[skip] {name}: alignment is not a rotation', flush=True)
            continue
        aligned = s*(splats['means']@R + T)  # uCO3D convention, row vectors
        solid = fg & (1/(1+np.exp(-splats['opacities'].reshape(-1))) > .5)
        world_dirs = aligned[solid]@world_from_aligned.T
        lo, hi = np.percentile(world_dirs, 1, 0), np.percentile(world_dirs, 99, 0)
        k = args.size/float((hi-lo).max())
        centre_aligned = world_from_aligned.T@((lo+hi)/2)
        # world = k*W*(x_al - c) = (k*s) * (W R^T) * (x - x0), x0 = R (c/s - T)
        placement = {'scale_m': k*s, 'rotation_scene_to_world': (world_from_aligned@R.T).tolist(),
                     'center_native': (R@(centre_aligned/s - T)).tolist()}
        write_ply(out/f'{name}.ply', splats, fg)
        info = {'sequence': name, 'category': a['category'], 'super_category': a['super_category'],
                'caption': a['caption'], 'splat_psnr_dB': a['splat_psnr_dB'], 'gaussians': int(fg.sum()),
                'world_extent_m': ((hi-lo)*k).tolist(), 'placement': placement,
                'source': 'facebook/uco3d examples.zip (CC BY 4.0)', 'up_axis_aligned': args.up, 'front_axis_aligned': args.front}
        (out/f'{name}.json').write_text(json.dumps(info, indent=1))
        index.append(info)
        print(f'[convert] {name} {a["category"]}: {fg.sum()} fg Gaussians, extent {np.round((hi-lo)*k, 3).tolist()} m', flush=True)
    (out/'assets.json').write_text(json.dumps(index, indent=1))
    print(f'[convert] {len(index)} assets -> {out}')


def cmd_preview(args):
    from lightfield_lab.gaussian_scene import GaussianTargetRenderer, stereo_poses
    folder = Path(args.assets)
    tiles = []
    for info in json.loads((folder/'assets.json').read_text()):
        scene = GaussianTargetRenderer(folder/f'{info["sequence"]}.ply', placement=info['placement'])
        images, _ = scene.render(stereo_poses(1., args.yaw, 0.)[:1])
        tile = Image.fromarray(images[0]).resize((320, 180))
        tiles.append((tile, f'{info["category"]}'))
        del scene
    cols = 8
    sheet = Image.new('RGB', (320*cols, 200*math.ceil(len(tiles)/cols)), 'white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    for i, (tile, label) in enumerate(tiles):
        x, y = 320*(i % cols), 200*(i//cols)
        sheet.paste(tile, (x, y))
        draw.text((x+4, y+182), label[:40], fill='black')
    sheet.save(args.out, quality=80)
    print(f'[preview] {len(tiles)} assets -> {args.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('convert')
    c.add_argument('--src', required=True, help='folder with metadata.sqlite and <category>/<seq>/gaussian_splats/')
    c.add_argument('--out', required=True)
    c.add_argument('--size', type=float, default=.27, help='longest robust foreground extent in metres')
    c.add_argument('--up', choices=list(AXES), default='+y', help='up axis of the uCO3D aligned frame')
    c.add_argument('--front', choices=list(AXES), default='+z', help='aligned axis turned toward the viewer')
    p = sub.add_parser('preview')
    p.add_argument('--assets', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--yaw', type=float, default=0.)
    args = parser.parse_args()
    {'convert': cmd_convert, 'preview': cmd_preview}[args.command](args)


if __name__ == '__main__':
    main()
