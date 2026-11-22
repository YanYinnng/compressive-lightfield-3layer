"""Read TA or self-collected manifests without filename-derived cameras."""
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .exam_geometry import (HEIGHT, WIDTH, IPD_M, NETWORK_EYE_ORDER, NETWORK_LAYER_ORDER,
                            screen_corners, validate_pose)


def read_manifest(path):
    path = Path(path).resolve()
    text = path.read_text(encoding='utf-8')
    data = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix.lower() == '.jsonl' else json.loads(text)
    rows = data['samples'] if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected nonempty samples.')
    ids = []
    for row in rows:
        if not isinstance(row.get('id'), str) or not row['id']:
            raise ValueError('Sample ID must be a nonempty string.')
        for key in ['left', 'right', 'pose_l', 'pose_r']:
            if key not in row:
                raise ValueError(f'Missing {key} in {row["id"]}.')
        if any(Path(row[key]).is_absolute() for key in ['left', 'right']):
            raise ValueError('Input image paths must be relative to the manifest.')
        left, right = validate_pose(row['pose_l']), validate_pose(row['pose_r'])
        if not np.isclose(np.linalg.norm(left[:3, 3]-right[:3, 3]), IPD_M, atol=1e-5):
            raise ValueError(f'{row["id"]}: expected 0.06m eye baseline; check scene units.')
        ids.append(row['id'])
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample IDs.')
    return path, rows


def read_rgb8(path):
    path = Path(path)
    with path.open('rb') as stream:
        header = stream.read(26)
    if header[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('>IIBB', header[16:26]) != (WIDTH, HEIGHT, 8, 2):
        raise ValueError(f'Expected {WIDTH}x{HEIGHT} RGB8 PNG: {path}')
    with Image.open(path) as image:
        if image.mode != 'RGB':
            raise ValueError(f'Expected RGB: {path}')
        return np.array(image)


class ExamDataset(Dataset):
    """EyeReal-native training tuples: images R/L, c2w metres, back-first screens.

    Already in model input order: call the model directly, not network_initial.
    network_initial instead accepts TA order and converts it. Reverse model outputs before TA
    rendering/export. Each image keeps its own pose; no filename parsing.
    """
    def __init__(self, manifest):
        self.path, self.rows = read_manifest(manifest)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        eyes = [('left', 'pose_l'), ('right', 'pose_r')]
        eyes = [eyes[i] for i in NETWORK_EYE_ORDER]
        images = np.stack([read_rgb8(self.path.parent/row[image]) for image, _ in eyes])
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()/255
        views = torch.tensor([row[pose] for _, pose in eyes], dtype=torch.float64)
        return images, views, screen_corners()[list(NETWORK_LAYER_ORDER)]
