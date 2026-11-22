"""Training-free layer solver: linearised polarization-field tomography (SART).

The TA optics are I = sin^2(pi/2 * a) with a = g1(q1) + g2(q2) + g3(q3) summed along each eye
ray (Lanman et al., "Polarization Fields", SIGGRAPH Asia 2011). Per ray the target phase
a* = 2/pi * asin(sqrt(I)) (principal branch) makes the problem linear in the patterns,
A g = a*, with A the cached bilinear ray-layer sampling of ExamRenderer. SART:
    g <- clip(g + relax * A^T(mask * (a* - A g) / A1) / (A^T mask), 0, 1).
It works for any camera (K, pose) the renderer was built with; no network is involved.
"""
import math

import torch


def _adjoint(renderer, residual, like):
    """A^T residual (bilinear splat back onto the three layers) via autograd of the linear A."""
    probe = torch.zeros_like(like, requires_grad=True)
    with torch.enable_grad():
        value = (renderer.accumulate(probe)*residual).sum()
        return torch.autograd.grad(value, probe)[0]


def sart(renderer, target, iterations=30, relax=1., initial=None, dtype=torch.float32, branch_from=None):
    """renderer: ExamRenderer (any K / pose); target [2,3,H,W] in [0,1]. Returns [3,3,H,W] in [0,1].

    branch_from: from this iteration on, each ray's target phase is re-chosen among the periodic
    branches b, 2-b, 2+b (sin^2 has period 2 in normalised phase) as the one nearest the current sum.
    """
    work = renderer.iteration_copy(dtype)
    mask = work.geometry_mask.to(dtype)  # [2,1,H,W]
    phase = (2/math.pi)*torch.asin(target.to(dtype).clamp(0, 1).sqrt())*mask
    g = torch.zeros(3, 3, renderer.height, renderer.width, dtype=dtype, device=phase.device) if initial is None \
        else initial.to(dtype).clone()
    with torch.no_grad():
        rows = work.accumulate(torch.ones_like(g)).clamp_min(1e-6)  # sum of bilinear weights per ray (3 inside)
        cols = _adjoint(work, mask.expand_as(phase), g).clamp_min(1e-6)  # rays hitting each layer pixel
        branches = torch.stack([phase, 2-phase, 2+phase])
        for iteration in range(iterations):
            current = work.accumulate(g)
            goal = phase
            if branch_from is not None and iteration >= branch_from:
                goal = branches.gather(0, (current[None]-branches).abs().argmin(0, keepdim=True))[0]*mask
            residual = mask*(goal-current)/rows
            g = (g+relax*_adjoint(work, residual, g)/cols).clamp_(0, 1)
    return g
