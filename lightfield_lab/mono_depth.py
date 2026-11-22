"""Monocular relative depth (Depth Anything V2 Small, Apache-2.0) for layer ownership.

Weights: weights/depth_anything_v2_vits.pth (HF depth-anything/Depth-Anything-V2-Small, 99,218,434 bytes).
Code: third_party/depth_anything_v2_src (upstream DepthAnything/Depth-Anything-V2, Apache-2.0), used unchanged;
preprocessing is done here on the GPU (resize so the short side is 518 and both sides are multiples of 14,
ImageNet normalisation), matching the upstream infer_image up to the resize kernel.
Output is affine-invariant inverse depth ("disparity"): larger = nearer. Only the ordering is used.
"""
from pathlib import Path
import sys

import numpy as np

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class MonoDepth:
    def __init__(self, weights=None, device='cuda', input_size=518, half=True):
        source = str(ROOT/'third_party/depth_anything_v2_src')
        if source not in sys.path:
            sys.path.insert(0, source)
        try:
            import cv2  # noqa: F401  (upstream imports it at module level; only its own preprocessing uses it)
        except ImportError:  # the H100 image has no OpenCV: a stub is enough, our GPU preprocessing replaces that path
            import types
            stub = types.ModuleType('cv2')
            stub.__getattr__ = lambda name: 0
            sys.modules['cv2'] = stub
        from depth_anything_v2.dpt import DepthAnythingV2
        weights = Path(weights or ROOT/'weights/depth_anything_v2_vits.pth')
        model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.half = half and torch.device(device).type == 'cuda'
        self.model = model.to(device).eval()
        if self.half:
            self.model = self.model.half()
        self.size = input_size
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, images):
        """images [N,3,H,W] RGB in [0,1] -> disparity [N,H,W] float32 (larger = nearer, per-image affine scale)."""
        _, _, h, w = images.shape
        scale = self.size/min(h, w)
        th, tw = (round(h*scale/14)*14, round(w*scale/14)*14)
        x = F.interpolate(images.float(), (th, tw), mode='bicubic', align_corners=False).clamp(0, 1)
        x = (x-self.mean)/self.std
        disp = self.model(x.half() if self.half else x).float()
        return F.interpolate(disp[:, None], (h, w), mode='bilinear', align_corners=False)[:, 0]


def otsu_foreground(disp, bins=256):
    """Per-image foreground = disparity above the Otsu threshold (near objects vs far background)."""
    out = []
    for d in disp:
        lo, hi = d.min(), d.max()
        hist = torch.histc(d, bins, float(lo), float(hi))
        p = hist/hist.sum()
        omega = p.cumsum(0)
        mu = (p*torch.arange(bins, device=d.device)).cumsum(0)
        between = (mu[-1]*omega-mu).square()/(omega*(1-omega)).clamp_min(1e-12)
        t = lo+(hi-lo)*(between.argmax().float()+.5)/bins
        out.append(d > t)
    return torch.stack(out)


def normalised_from_disparity(disp, fg, low=2., high=98.):
    """Per eye: d in [0,1], 0 = nearest foreground (largest disparity); percentiles over that eye's foreground."""
    out = torch.zeros_like(disp)
    for e in range(disp.shape[0]):
        v = disp[e][fg[e]]
        if v.numel() < 100:
            v = disp[e].flatten()
        v = v[::max(1, v.numel()//200000)]
        hi, lo = torch.quantile(v, torch.tensor([high/100, low/100], device=v.device))
        out[e] = ((hi-disp[e])/(hi-lo).clamp_min(1e-6)).clamp(0, 1)
    return out


def border_background(images, tolerance=10/255, min_fraction=.03, step=4):
    """Per eye: flat background = pixels within `tolerance` (max over RGB) of the median border colour AND
    4-connected to the image border (evaluated at 1/step resolution). Returns foreground [N,H,W] bool;
    all-foreground when the background is under min_fraction of the frame (e.g. real scenes)."""
    from scipy import ndimage
    out = []
    for image in images:
        small = image[:, ::step, ::step]
        border = torch.cat([small[:, 0], small[:, -1], small[:, :, 0], small[:, :, -1]], 1)
        colour = border.median(1).values[:, None, None]
        candidate = ((small-colour).abs().amax(0) < tolerance).cpu().numpy()
        labels, _ = ndimage.label(candidate)
        touching = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
        background = np.isin(labels, list(touching))
        if background.mean() < min_fraction:
            background[:] = False
        bg = torch.from_numpy(background).to(image.device)
        bg = bg.repeat_interleave(step, 0).repeat_interleave(step, 1)[:image.shape[1], :image.shape[2]]
        out.append(~bg)
    return torch.stack(out)


CENTRES = (0., .5, 1.)


def eye_weights(d, fg):
    """[2,3,H,W] triangle ownership per eye pixel from normalised depth d (0 = nearest); background -> L3."""
    w = torch.stack([(1-(d-c).abs()/.5).clamp_min(0) for c in CENTRES], 1)
    back = w.new_tensor([0., 0., 1.])[None, :, None, None]
    return torch.where(fg[:, None], w, back)


def layer_weights(fs, w_eye):
    """Ownership of each layer pixel: mean over the eyes that see it of the ownership of what they see there."""
    fields = torch.cat([w_eye, torch.ones_like(w_eye[:, :1])], 1)
    pulled = fs.pull_to_layers(fields)                    # [3 layers, 2 eyes, 4, H, W]
    out = []
    for k in range(3):
        support = pulled[k, :, 3]
        value = pulled[k, :, k].sum(0)/support.sum(0).clamp_min(1e-6)
        out.append(torch.where(support.sum(0) > .5, value, torch.ones_like(value)))
    return torch.stack(out).clamp(0, 1)


def ownership(mono, fs, images):
    """Layer ownership [3,H,W] for the normal branch: relative depth (front/middle/back -> L1/L2/L3, TA rule)."""
    fg = border_background(images)
    return layer_weights(fs, eye_weights(normalised_from_disparity(mono(images), fg), fg))
