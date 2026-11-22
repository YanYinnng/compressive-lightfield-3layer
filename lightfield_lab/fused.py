"""Prototype fused solver step for I = sin^2(pi/2 * sum_k g_k(F_ek(p))), TA geometry.

Two kernels per Adam step, no grid tensors, no atomics:
  K1 (one thread per eye pixel): homography -> 3 layers x bilinear -> phase -> sin^2 -> residual -> dL/da (G).
  K2 (one thread per layer pixel): exact transpose of bilinear sampling by gathering G from the eye
      pixels whose footprint covers it (inverse homography + small window), then Adam + clamp in place.
Eval mode of K1 scores RGB8-quantised layers (per-eye masked SSE) without writing anything else.
Same CUDA source on two launch backends: CuPy/NVRTC (local Windows) or a torch extension (H100); see backend().
"""
import math
import os
from pathlib import Path

import numpy as np
import torch

H, W = 1080, 1920
WS = .596736
HS = WS*H/W
LAYER_X = (.06, 0., -.06)

SRC = r'''
#define H 1080
#define W 1920
#define HW (H*W)
extern "C" {

__device__ __forceinline__ void hmap(const float* f, float j, float i, float& xs, float& ys, float& den) {
    j -= W/2; i -= H/2;
    den = f[6]*j + f[7]*i + f[8];
    xs = (f[0]*j + f[1]*i + f[2]) / den;
    ys = (f[3]*j + f[4]*i + f[5]) / den;
}

// TA validity mask, the TA ray expression in float64 without FMA contraction (ExamRenderer numpy branch):
// all three layer hits must have t > 0 and land on the panel [0,W] x [0,H] (inclusive).
__global__ void kmask(const double* cam, unsigned char* out) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= 2*HW) return;
    int e = idx / HW, p = idx - e*HW, i = p / W, j = p - i*W;
    const double* c = cam + e*16;  // R (row-major 9), o (3), fx, fy, cx, cy
    const double WS = 0.596736, HS = 0.596736*H/W;
    double lx = __ddiv_rn(__dadd_rn(j + .5, -c[14]), c[12]);
    double ly = -__ddiv_rn(__dadd_rn(i + .5, -c[15]), c[13]);
    double d[3];
    for (int m = 0; m < 3; ++m) d[m] = __dadd_rn(__dadd_rn(__dmul_rn(c[3*m], lx), __dmul_rn(c[3*m + 1], ly)), -c[3*m + 2]);
    bool valid = fabs(d[0]) >= 1e-9;
    const double X[3] = {.06, 0., -.06};
    for (int k = 0; k < 3 && valid; ++k) {
        double t = __ddiv_rn(__dadd_rn(X[k], -c[9]), d[0]);
        double y = __dadd_rn(c[10], __dmul_rn(t, d[1])), z = __dadd_rn(c[11], __dmul_rn(t, d[2]));
        double u = __dmul_rn(__ddiv_rn(__dadd_rn(y, WS/2), WS), (double)W);
        double v = __dmul_rn(__ddiv_rn(__dadd_rn(HS/2, -z), HS), (double)H);
        valid = t > 0 && u >= 0 && u <= W && v >= 0 && v <= H;
    }
    out[idx] = valid;
}

__device__ __forceinline__ float tap(const float* L, int x, int y) {
    return (x >= 0 && x < W && y >= 0 && y < H) ? L[y*W + x] : 0.f;
}

// bilinear tap weight times the layer-ownership weight (1 without weights); zero outside the panel
__device__ __forceinline__ float owned(const float* Wk, float w, int x, int y) {
    if (!(x >= 0 && x < W && y >= 0 && y < H)) return 0.f;
    return Wk ? w*Wk[y*W + x] : w;
}

// mode 0: training, writes G (dL/da per eye pixel, channel).  mode 1: eval, quantised layers, SSE + count.
// mode 2: debug, writes the phase sum into G.
__global__ void k1(const float* g, const float* target, const float* F, const float* sgn, float* G,
                   double* sse, double* count, const float* inv_n, const unsigned char* vmask, const float* Wl, int mode) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    double my_sse = 0., my_cnt = 0.;
    if (idx < 2*HW) {
        int e = idx / HW, p = idx - e*HW, i = p / W, j = p - i*W;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, rows = 0.f;
        bool valid = true;
        for (int k = 0; k < 3; ++k) {
            const float* f = F + (e*3 + k)*9;
            float xs, ys, den;
            hmap(f, j, i, xs, ys, den);
            valid = valid && fabs(den) >= 1e-9 && sgn[e*3 + k]*den > 0 && xs >= -.5 && xs <= W - .5 && ys >= -.5 && ys <= H - .5;
            if (!(xs > -1.5 && xs < W + .5 && ys > -1.5 && ys < H + .5)) continue;  // fully outside: zero padding
            float fx = xs, fy = ys;
            int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
            float ax = fx - x0, ay = fy - y0;
            float w00 = (1-ax)*(1-ay), w10 = ax*(1-ay), w01 = (1-ax)*ay, w11 = ax*ay;
            if (mode == 3 || mode == 6) {  // SART row sum, weighted by layer ownership when given (weighted min-norm SART)
                const float* Wk = Wl ? Wl + (size_t)k*HW : nullptr;
                rows += owned(Wk, w00, x0, y0) + owned(Wk, w10, x0+1, y0) + owned(Wk, w01, x0, y0+1) + owned(Wk, w11, x0+1, y0+1);
            }
            const float* L = g + (size_t)k*3*HW;
            for (int c = 0; c < 3; ++c) {
                const float* Lc = L + (size_t)c*HW;
                float v00 = tap(Lc, x0, y0), v10 = tap(Lc, x0+1, y0), v01 = tap(Lc, x0, y0+1), v11 = tap(Lc, x0+1, y0+1);
                if (mode == 1) {  // RGB8 quantisation of the layer values
                    v00 = floorf(v00*255.f + .5f)/255.f; v10 = floorf(v10*255.f + .5f)/255.f;
                    v01 = floorf(v01*255.f + .5f)/255.f; v11 = floorf(v11*255.f + .5f)/255.f;
                }
                float s = w00*v00 + w10*v10 + w01*v01 + w11*v11;
                if (c == 0) a0 += s; else if (c == 1) a1 += s; else a2 += s;
            }
        }
        float a[3] = {a0, a1, a2};
        if (vmask) valid = vmask[idx];  // exact TA mask when available
        if (mode == 3 || mode == 6) {  // 6: periodic branch nearest the current sum (b, 2-b, 2+b; sin^2 has period 2)
            for (int c = 0; c < 3; ++c) {
                size_t o = (size_t)(e*3 + c)*HW + p;
                float goal = target[o];
                if (mode == 6) {
                    float b2 = 2.f - goal, b3 = 2.f + goal;
                    if (fabsf(b2 - a[c]) < fabsf(goal - a[c])) goal = b2;
                    if (fabsf(b3 - a[c]) < fabsf(goal - a[c])) goal = b3;
                }
                G[o] = valid && rows > 1e-8f ? (goal - a[c])/rows : 0.f;
            }
            return;
        }
        if (mode == 4) {  // mask (for SART column sums)
            for (int c = 0; c < 3; ++c) G[(size_t)(e*3 + c)*HW + p] = valid ? 1.f : 0.f;
            return;
        }
        for (int c = 0; c < 3; ++c) {
            size_t o = (size_t)(e*3 + c)*HW + p;
            if (mode == 2) { G[o] = a[c]; continue; }
            float sn = sinf(1.57079632679f*a[c]);
            float r = sn*sn - target[o];
            if (mode == 0) G[o] = valid ? r*1.57079632679f*sinf(3.14159265359f*a[c])*inv_n[e] : 0.f;
            else if (valid) my_sse += (double)r*r;
        }
        my_cnt = valid ? 1. : 0.;
    }
    if (mode == 1) {  // warp reduction then one atomic per warp (both eyes never share a warp: HW % 32 == 0)
        for (int off = 16; off > 0; off >>= 1) {
            my_sse += __shfl_down_sync(0xffffffff, my_sse, off);
            my_cnt += __shfl_down_sync(0xffffffff, my_cnt, off);
        }
        if ((threadIdx.x & 31) == 0 && idx < 2*HW) {
            int e = idx / HW;
            atomicAdd(sse + e, my_sse);
            atomicAdd(count + e, my_cnt);
        }
    }
}

// One thread per layer pixel (k, y, x): grad = sum_e sum_{p near Finv(q)} w(F(p), q) * G[e, :, p]; then Adam + clamp.
// mode 0: Adam.  mode 1: write the gradient only (debug).  mode 2: SART-style update g += relax*grad/colsum (not used here).
// Wl: layer ownership weights [3,H,W] for SART (mode 2), Wa: per-pixel Adam step scale [3,H,W] (mode 0); null = 1.
__global__ void k2(float* g, float* m, float* v, const float* G, const float* F, const double* Finv, const float* rad,
                   float* grad_out, float lr, float b1, float b2, float bc1, float bc2s, float eps, const float* Wl, const float* Wa, int mode) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= 3*HW) return;
    int k = idx / HW, q = idx - k*HW, y = q / W, x = q - y*W;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f;
    for (int e = 0; e < 2; ++e) {
        const float* f = F + (e*3 + k)*9;
        const double* fi = Finv + (e*3 + k)*9;
        double dj = fi[6]*x + fi[7]*y + fi[8];
        double jc = (fi[0]*x + fi[1]*y + fi[2]) / dj, ic = (fi[3]*x + fi[4]*y + fi[5]) / dj;
        float rj = rad[(e*3 + k)*2], ri = rad[(e*3 + k)*2 + 1];
        int j0 = max(0, (int)ceilf(jc - rj)), j1 = min(W - 1, (int)floorf(jc + rj));
        int i0 = max(0, (int)ceilf(ic - ri)), i1 = min(H - 1, (int)floorf(ic + ri));
        for (int i = i0; i <= i1; ++i)
            for (int j = j0; j <= j1; ++j) {
                float xs, ys, den;
                hmap(f, j, i, xs, ys, den);
                float wx = 1.f - fabsf(xs - x), wy = 1.f - fabsf(ys - y);
                if (wx <= 0.f || wy <= 0.f) continue;
                float w = wx*wy;
                size_t o = (size_t)(e*3)*HW + (size_t)i*W + j;
                acc0 += w*G[o]; acc1 += w*G[o + HW]; acc2 += w*G[o + 2*HW];
            }
    }
    float acc[3] = {acc0, acc1, acc2};
    for (int c = 0; c < 3; ++c) {
        size_t o = (size_t)(k*3 + c)*HW + q;
        float gr = acc[c];
        if (mode == 1) { grad_out[o] = gr; continue; }
        if (mode == 2) { g[o] = fminf(fmaxf(g[o] + lr*(Wl ? Wl[(size_t)k*HW + q] : 1.f)*gr/fmaxf(m[o], 1e-6f), 0.f), 1.f); continue; }
        float mm = b1*m[o] + (1.f - b1)*gr;
        float vv = b2*v[o] + (1.f - b2)*gr*gr;
        m[o] = mm; v[o] = vv;
        float p = g[o] - (Wa ? Wa[(size_t)k*HW + q] : 1.f)*(lr/bc1)*mm/(sqrtf(vv)/bc2s + eps);
        g[o] = fminf(fmaxf(p, 0.f), 1.f);
    }
}
}
'''
def homographies(poses, intrinsics=None):
    """F[e,k]: (j, i, 1) eye pixel index -> layer-k sample position (xs, ys) = (u-.5, v-.5) (projective), den = d_x."""
    F, sgn = np.zeros((2, 3, 9)), np.zeros((2, 3))
    focal = W/(2*math.tan(math.radians(20)))
    for e, pose in enumerate(poses):
        R, o = np.asarray(pose)[:3, :3], np.asarray(pose)[:3, 3]
        fx, fy, cx, cy = (focal, focal, W/2, H/2) if intrinsics is None else intrinsics[e]
        D = np.array([[R[m, 0]/fx, -R[m, 1]/fy, R[m, 0]*(.5-cx)/fx - R[m, 1]*(.5-cy)/fy - R[m, 2]] for m in range(3)])
        for k, X in enumerate(LAYER_X):
            d = X - o[0]
            row0 = ((W/WS)*(o[1]+WS/2) - .5)*D[0] + (W/WS)*d*D[1]
            row1 = ((H/HS)*(HS/2-o[2]) - .5)*D[0] - (H/HS)*d*D[2]
            M = np.stack([row0, row1, D[0]])
            M[:, 2] += M[:, 0]*W/2 + M[:, 1]*H/2  # centred indices j-W/2, i-H/2
            F[e, k] = M.ravel()
            sgn[e, k] = np.sign(d)
    return F, sgn


def radii(F):
    """Half-window (in eye pixels) containing every eye pixel whose bilinear footprint touches a layer pixel."""
    rad = np.zeros((2, 3, 2), dtype=np.float32)
    Finv = np.zeros_like(F)
    xs, ys = np.meshgrid(np.linspace(-.5, W-.5, 33), np.linspace(-.5, H-.5, 19))
    for e in range(2):
        for k in range(3):
            C = F[e, k].reshape(3, 3).copy()
            C[:, 2] -= C[:, 0]*W/2 + C[:, 1]*H/2  # back to raw indices
            M = np.linalg.inv(C)
            Finv[e, k] = M.ravel()
            def inv(x, y):
                d = M[2, 0]*x + M[2, 1]*y + M[2, 2]
                return (M[0, 0]*x + M[0, 1]*y + M[0, 2])/d, (M[1, 0]*x + M[1, 1]*y + M[1, 2])/d
            h = 1e-3
            j0, i0 = inv(xs, ys)
            jx, ix = inv(xs+h, ys)
            jy, iy = inv(xs, ys+h)
            a, b, c, dd = (jx-j0)/h, (jy-j0)/h, (ix-i0)/h, (iy-i0)/h   # d(j,i)/d(xs,ys)
            rj = np.nanmax(np.abs(a)+np.abs(b))
            ri = np.nanmax(np.abs(c)+np.abs(dd))
            rad[e, k] = [rj*1.02+.01, ri*1.02+.01]
    return Finv, rad


_WRAP = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
''' + SRC + r'''
static float* fp(const torch::Tensor& t) { return t.numel() ? t.data_ptr<float>() : nullptr; }
void k1_launch(torch::Tensor g, torch::Tensor target, torch::Tensor F, torch::Tensor sgn, torch::Tensor G,
               torch::Tensor sse, torch::Tensor count, torch::Tensor inv_n, torch::Tensor vmask, torch::Tensor Wl, int64_t mode) {
    k1<<<(2*HW + 255)/256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(fp(g), fp(target), fp(F), fp(sgn), fp(G),
        sse.data_ptr<double>(), count.data_ptr<double>(), fp(inv_n), vmask.numel() ? vmask.data_ptr<unsigned char>() : nullptr, fp(Wl), (int)mode);
}
void kmask_launch(torch::Tensor cam, torch::Tensor out) {
    kmask<<<(2*HW + 255)/256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(cam.data_ptr<double>(), out.data_ptr<unsigned char>());
}
void k2_launch(torch::Tensor g, torch::Tensor m, torch::Tensor v, torch::Tensor G, torch::Tensor F, torch::Tensor Finv,
               torch::Tensor rad, torch::Tensor out, double lr, double b1, double b2, double bc1, double bc2s, double eps,
               torch::Tensor Wl, torch::Tensor Wa, int64_t mode) {
    k2<<<(3*HW + 255)/256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(fp(g), fp(m), fp(v), fp(G), fp(F),
        Finv.data_ptr<double>(), fp(rad), fp(out), (float)lr, (float)b1, (float)b2, (float)bc1, (float)bc2s, (float)eps, fp(Wl), fp(Wa), (int)mode);
}
'''
_DECL = '''
void k1_launch(torch::Tensor g, torch::Tensor target, torch::Tensor F, torch::Tensor sgn, torch::Tensor G,
               torch::Tensor sse, torch::Tensor count, torch::Tensor inv_n, torch::Tensor vmask, torch::Tensor Wl, int64_t mode);
void kmask_launch(torch::Tensor cam, torch::Tensor out);
void k2_launch(torch::Tensor g, torch::Tensor m, torch::Tensor v, torch::Tensor G, torch::Tensor F, torch::Tensor Finv,
               torch::Tensor rad, torch::Tensor out, double lr, double b1, double b2, double bc1, double bc2s, double eps,
               torch::Tensor Wl, torch::Tensor Wa, int64_t mode);
'''
_EMPTY = None
_BACKEND = None


def backend():
    """'cupy' (NVRTC; no host compiler, used on Windows) or 'torch' (torch C++/CUDA extension; needs nvcc,
    as on the H100 image). LF_FUSED_BACKEND forces one; the extension build is cached in LF_FUSED_BUILD
    (default <repo>/.fused_build), so only the first process compiles (about a minute)."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    choice = os.environ.get('LF_FUSED_BACKEND')
    if choice in (None, 'cupy'):
        try:
            import cupy as cp
            module = cp.RawModule(code=SRC, options=('--use_fast_math',))
            k1, k2, km = module.get_function('k1'), module.get_function('k2'), module.get_function('kmask')

            def ptr(a):
                # raw device pointer: no CuPy array per launch (from_dlpack cost ~20 us x 10 args x 2 kernels per step);
                # the caller keeps the tensors alive. Empty tensor -> null pointer.
                return np.uint64(a.data_ptr() if a.numel() else 0)

            def run1(*args):
                k1(((2*H*W + 255)//256,), (256,), tuple(ptr(a) for a in args[:-1]) + (np.int32(args[-1]),))

            def run2(*args):  # 8 tensors, 6 floats, 2 weight tensors, mode
                k2(((3*H*W + 255)//256,), (256,), tuple(ptr(a) for a in args[:8])
                   + tuple(np.float32(s) for s in args[8:14]) + (ptr(args[14]), ptr(args[15]), np.int32(args[16])))
            def run_mask(cam, out):
                km(((2*H*W + 255)//256,), (256,), (ptr(cam), ptr(out)))
            _BACKEND = ('cupy', run1, run2, run_mask)
            return _BACKEND
        except ImportError:
            if choice == 'cupy':
                raise
    from torch.utils.cpp_extension import load_inline
    build = Path(os.environ.get('LF_FUSED_BUILD', Path(__file__).resolve().parents[1]/'.fused_build'))
    build.mkdir(parents=True, exist_ok=True)
    ext = load_inline('lf_fused', cpp_sources=_DECL, cuda_sources=_WRAP, functions=['k1_launch', 'k2_launch', 'kmask_launch'],
                      extra_cuda_cflags=['-O3', '--use_fast_math'], build_directory=str(build), verbose=False)
    _BACKEND = ('torch', ext.k1_launch, ext.k2_launch, ext.kmask_launch)
    return _BACKEND


class FusedSolver:
    """All tensors are torch CUDA tensors; the kernels run on the current torch stream (torch backend)."""

    def __init__(self, poses, target, intrinsics=None):
        """target: [2,3,H,W] images in [0,1]."""
        self.name, self.run1, self.run2, run_mask = backend()
        dev = target.device
        F, sgn = homographies(poses, intrinsics)
        Finv, rad = radii(F)
        f32 = dict(device=dev, dtype=torch.float32)
        self.F, self.Finv = torch.tensor(F, **f32), torch.tensor(Finv, device=dev, dtype=torch.float64)
        self.sgn, self.rad = torch.tensor(sgn, **f32), torch.tensor(rad, **f32)
        self.radii_used = rad
        self.T = target.float().contiguous()
        self.G = torch.zeros((2, 3, H, W), **f32)
        self.sse = torch.zeros(2, device=dev, dtype=torch.float64)
        self.cnt = torch.zeros(2, device=dev, dtype=torch.float64)
        self.inv_n = torch.ones(2, **f32)
        self.none = torch.empty(0, **f32)
        self.w_sart = self.w_adam = self.none  # layer ownership weights [3,H,W]; empty = unweighted
        focal = W/(2*math.tan(math.radians(20)))
        cam = [np.concatenate([np.asarray(pose, float)[:3, :3].ravel(), np.asarray(pose, float)[:3, 3],
                               (focal, focal, W/2, H/2) if intrinsics is None else np.asarray(intrinsics[e], float)])
               for e, pose in enumerate(poses)]
        self.vmask = torch.empty((2, H, W), device=dev, dtype=torch.uint8)
        run_mask(torch.tensor(np.stack(cam), device=dev, dtype=torch.float64), self.vmask)
        self.eval_raw(torch.zeros((3, 3, H, W), **f32))
        self.count = self.cnt.cpu().numpy()   # valid pixels per eye (TA mask)
        self.inv_n = (1./(3.*self.cnt)).float()  # d mean_e[sum r^2/(3 n_e)] / dI = r/(3 n_e)

    def k1(self, g, mode):
        self.run1(g, self.T, self.F, self.sgn, self.G, self.sse, self.cnt, self.inv_n, self.vmask, self.w_sart, mode)

    def _k2(self, g, m, v, out, lr, mode, t=1, b1=.9, b2=.999, eps=1e-8):
        self.run2(g, m, v, self.G, self.F, self.Finv, self.rad, self.none if out is None else out,
                  lr, b1, b2, 1-b1**t, math.sqrt(1-b2**t), eps, self.w_sart, self.w_adam, mode)

    def set_layer_weights(self, sart=None, adam=None):
        """Ownership weights [3,H,W] in (0,1] (front-to-back layers). sart: weighted SART, whose fixed point from zero
        is the minimum sum_k g_k^2/w_k solution of the two-eye equations; adam: per-pixel Adam step scale."""
        self.w_sart = self.none if sart is None else sart.float().contiguous()
        self.w_adam = self.none if adam is None else adam.float().contiguous()

    def layer_grids(self):
        """[3,2,H,W,2] grid_sample coordinates: layer pixel (k, y, x) -> the eye-e pixel whose ray hits its centre."""
        if getattr(self, '_grids', None) is None:
            dev = self.F.device
            y, x = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float64), torch.arange(W, device=dev, dtype=torch.float64), indexing='ij')
            out = torch.empty((3, 2, H, W, 2), device=dev, dtype=torch.float32)
            for e in range(2):
                for k in range(3):
                    M = self.Finv[e, k].reshape(3, 3)
                    d = M[2, 0]*x + M[2, 1]*y + M[2, 2]
                    j = (M[0, 0]*x + M[0, 1]*y + M[0, 2])/d
                    i = (M[1, 0]*x + M[1, 1]*y + M[1, 2])/d
                    bad = ~(d.abs() > 1e-12)
                    out[k, e, ..., 0] = torch.where(bad, 3., 2*(j+.5)/W-1).float()
                    out[k, e, ..., 1] = torch.where(bad, 3., 2*(i+.5)/H-1).float()
            self._grids = out
        return self._grids

    def pull_to_layers(self, eye_fields):
        """eye_fields [2,C,H,W] (per eye pixel) -> [3,2,C,H,W]: value seen through each layer pixel by each eye."""
        grids = self.layer_grids()
        out = [torch.nn.functional.grid_sample(eye_fields.float(), grids[k], align_corners=False) for k in range(3)]
        return torch.stack(out)

    def eval_raw(self, g):
        self.sse.zero_(); self.cnt.zero_()
        self.k1(g, 1)

    def psnr_tensor(self, g):
        """Quantised-layer PSNR (mean over eyes, float32 eyes) as a 0-d GPU tensor: no host sync."""
        self.eval_raw(g)
        mse = self.sse/(self.cnt*3)
        return (-10*torch.log10(mse.clamp_min(1e-12))).mean()

    def psnr(self, g):
        return float(self.psnr_tensor(g))

    def phase(self, g):
        self.k1(g, 2)
        return self.G.clone()

    def gradient(self, g):
        self.k1(g, 0)
        out = torch.zeros_like(g)
        self._k2(g, g, g, out, 0., 1)
        return out

    def adam(self, initial, steps=100, lr=.08, eval_every=10):
        """Projected Adam, same formula as torch.optim.Adam + clamp; best RGB8 checkpoint by PSNR, selected on
        the GPU (one host sync at the end). Returns (float layers, best PSNR, PSNR history)."""
        g = initial.detach().float().clamp(0, 1).contiguous().clone()
        m, v = torch.zeros_like(g), torch.zeros_like(g)
        best, best_score = g.clone(), self.psnr_tensor(g)
        history = [best_score]
        for t in range(1, steps+1):
            self.k1(g, 0)
            self._k2(g, m, v, None, lr, 0, t)
            if t % eval_every == 0 or t == steps:
                s = self.psnr_tensor(g)
                history.append(s)
                better = s > best_score
                best = torch.where(better, g, best)
                best_score = torch.where(better, s, best_score)
        history = torch.stack(history).cpu().tolist()
        return best, float(best_score), history


def sart(solver, images, iterations=30, relax=1., branch_from=None, initial=None):
    """Fused version of lightfield_lab.tomographic.sart. branch_from: from this iteration on, each ray's target phase
    is the periodic branch nearest the current sum (principal branch before)."""
    phase = ((2/math.pi)*torch.asin(images.float().clamp(0, 1).sqrt())).contiguous()
    g = torch.zeros((3, 3, H, W), device=phase.device, dtype=torch.float32) if initial is None else initial.float().contiguous().clone()
    cols = torch.zeros_like(g)
    solver.k1(g, 4)
    solver._k2(cols, cols, cols, cols, 0., 1)
    target, solver.T = solver.T, phase
    for it in range(iterations):
        solver.k1(g, 6 if branch_from is not None and it >= branch_from else 3)
        solver._k2(g, cols, cols, None, relax, 2)
    solver.T = target
    return g


class stream_context:
    """Run torch ops and the fused kernels of this thread on one CUDA stream (torch.cuda.Stream or None).
    The CuPy backend launches on CuPy's thread-local current stream, so it is pointed at the same stream."""

    def __init__(self, stream):
        self.stream = stream
        self.contexts = []

    def __enter__(self):
        if self.stream is None:
            return self
        self.contexts = [torch.cuda.stream(self.stream)]
        if backend()[0] == 'cupy':
            import cupy as cp
            self.contexts.append(cp.cuda.ExternalStream(self.stream.cuda_stream))
        for c in self.contexts:
            c.__enter__()
        return self

    def __exit__(self, *exc):
        for c in reversed(self.contexts):
            c.__exit__(*exc)
        self.contexts = []
        return False
