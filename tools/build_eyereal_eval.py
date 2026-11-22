"""Convert EyeReal's uCO3D RGB validation pairs to TA v3 research inputs."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from testing_v3.inference import DEFAULT_GEOMETRY, HEIGHT, WIDTH, fixed_intrinsics, pose_array


NUMBER = r'-?\d+(?:\.\d+)?'
PAIR = re.compile(rf'pair(\d+)_x({NUMBER})_y({NUMBER})_z({NUMBER})_([01])\.(?:jpg|png)', re.I)
SEQUENCE = re.compile(r'\d+-\d+-\d+')
CAMERA = re.compile(r'_scale_(\d+(?:\.\d+)?)_R_.*?_FOV_(\d+(?:\.\d+)?)_theta_')
# EyeReal +Z faces the observer, +X is screen-right, +Y is up.
WORLD_TO_TA = np.eye(3)[[2, 0, 1]]
TA_K = fixed_intrinsics()


def source_pose(position, scale):
    """Reproduce EyeReal's vertical=y look-at, then rotate into the TA basis."""
    back = position/np.linalg.norm(position)
    right = np.cross([1e-6, 1., 1e-5], back)
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(back, right), back))
    result = np.eye(4)
    result[:3, :3] = WORLD_TO_TA@rotation
    result[:3, 3] = WORLD_TO_TA@position/(100*scale)
    return pose_array(result)


def source_intrinsics(size, fov):
    """Use the pinned EyeReal data/render.py's fovy and fovx expressions."""
    width, height = size
    vertical = np.deg2rad(fov)
    horizontal = vertical*width/height
    return np.array([[width/(2*np.tan(horizontal/2)), 0., width/2],
                     [0., height/(2*np.tan(vertical/2)), height/2],
                     [0., 0., 1.]])


def to_ta_image(image, source_k):
    # Pillow affine maps output pixel centres in edge coordinates to input.
    affine = source_k@np.linalg.inv(TA_K)
    edges = affine@np.array([[0., WIDTH], [0., HEIGHT], [1., 1.]])
    if np.any(edges[:2, 0] < -1e-8) or np.any(edges[:2, 1] > np.array(image.size)+1e-8):
        raise ValueError('The TA camera sees rays beyond the source image.')
    return image.convert('RGB').transform(
        (WIDTH, HEIGHT), Image.Transform.AFFINE, tuple(affine[:2].flat), Image.Resampling.BICUBIC)


def read_pairs(source):
    pairs, excluded = defaultdict(dict), Counter()
    for name in source.namelist():
        path = PurePosixPath(name)
        match = PAIR.fullmatch(path.name)
        if match is None or '__MACOSX' in path.parts:
            continue
        sequence = path.parent.parent.name
        if not SEQUENCE.fullmatch(sequence):
            excluded[sequence] += 1
            continue
        if CAMERA.search(path.parent.name) is None:
            raise ValueError(f'No camera metadata in {name}')
        key = str(path.parent), int(match[1])
        eye = int(match[5])
        if eye in pairs[key]:
            raise ValueError(f'Duplicate eye for {key}')
        pairs[key][eye] = name, np.array([float(match[i]) for i in (2, 3, 4)])
    if any(len(eyes) != 2 for eyes in pairs.values()):
        raise ValueError('The archive contains incomplete uCO3D pairs.')
    return pairs, excluded


def split_for(sequence, seed):
    digest = sha256(f'{seed}:{sequence}'.encode()).digest()
    return 'verify' if digest[0] < 64 else 'tune'


def build_archive(archive, variant, output, rows, seed, pairs_per_sequence):
    with ZipFile(archive) as source:
        pairs, excluded = read_pairs(source)
        sequences = defaultdict(list)
        for key in sorted(pairs):
            folder, _ = key
            sequences[PurePosixPath(folder).parent.name].append(key)
        for sequence, keys in sorted(sequences.items()):
            split = split_for(sequence, seed)
            rng = np.random.default_rng(int.from_bytes(sha256(f'{seed}:{variant}:{sequence}'.encode()).digest()[:8], 'big'))
            indices = sorted(rng.choice(len(keys), min(pairs_per_sequence, len(keys)), replace=False))
            for index in indices:
                key = keys[index]
                folder, pair = key
                scale, fov = map(float, CAMERA.search(folder).groups())
                poses = [source_pose(pairs[key][eye][1], scale) for eye in (1, 0)]
                baseline = poses[1][:3, 3]-poses[0][:3, 3]
                ipd = float(np.linalg.norm(baseline))
                if abs(ipd-.06) > .0004 or min(pose[:3, 0]@baseline for pose in poses) <= 0:
                    raise ValueError(f'Unexpected stereo baseline or eye order in {key}: {ipd}')
                sample_id = f'{variant}_{sequence}_{pair:04d}'
                row = {'id': sample_id, 'sequence': sequence, 'variant': variant, 'ipd_m': ipd}
                for side, eye, pose in zip(('l', 'r'), (1, 0), poses):
                    name = pairs[key][eye][0]
                    relative = f'images/{sample_id}_{side}.png'
                    with Image.open(io.BytesIO(source.read(name))) as image:
                        k = source_intrinsics(image.size, fov)
                        to_ta_image(image, k).save(output/split/relative, compress_level=3)
                        row[f'source_K_{side}'] = k.tolist()
                    row['left' if side == 'l' else 'right'] = relative
                    row[f'pose_{side}'] = pose.tolist()
                    row[f'source_{side}'] = name
                rows[split].append(row)
    return {'archive': str(archive), 'uCO3D_sequences': len(sequences), 'uCO3D_pairs': len(pairs),
            'skipped_named_scene_images': dict(sorted(excluded.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--val', type=Path, required=True)
    parser.add_argument('--disrupt', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pairs-per-sequence', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    if args.pairs_per_sequence < 1:
        parser.error('--pairs-per-sequence must be positive.')
    args.out.mkdir(parents=True, exist_ok=False)
    rows = {'tune': [], 'verify': []}
    for split in rows:
        (args.out/split/'images').mkdir(parents=True)
    sources = [build_archive(args.val, 'val', args.out, rows, args.seed, args.pairs_per_sequence)]
    if args.disrupt is not None:
        sources.append(build_archive(args.disrupt, 'disrupt', args.out, rows, args.seed, args.pairs_per_sequence))
    for split, samples in rows.items():
        ids = {row['id'] for row in samples}
        if len(ids) != len(samples) or not samples:
            raise ValueError(f'{split}: expected a nonempty split with unique IDs.')
        manifest = {'phase': 'eyereal_research', 'counts_toward_grade': False, 'samples': samples,
                    'input_camera': {'horizontal_fov_deg': 40, 'K': TA_K.tolist(), 'image_size': [WIDTH, HEIGHT]},
                    'geometry': DEFAULT_GEOMETRY, 'offaxis_task': {'reference_layer': 'L2', 'intrinsics': 'dynamic',
                    'target_policy': 'reuse the same left/right RGB images without rerendering'}}
        (args.out/split/'test.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    summary = {'sources': sources, 'seed': args.seed, 'pairs_per_sequence': args.pairs_per_sequence,
               'splits': {name: {'pairs': len(samples), 'sequences': sorted({r['sequence'] for r in samples}),
                                 'variants': dict(Counter(r['variant'] for r in samples))}
                          for name, samples in rows.items()}}
    (args.out/'conversion.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
