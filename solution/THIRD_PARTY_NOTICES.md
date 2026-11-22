# Third-party sources

The interface example is distributed with LICENSE.
Python dependencies retain their respective upstream licenses.
The collection directory includes the supplied public Lego scene asset. No Gaussian rasterizer, network architecture, training implementation or optical solver is bundled.

## Depth Anything V2 (Small)

- Source: https://github.com/DepthAnything/Depth-Anything-V2 (code in `third_party/depth_anything_v2_src`, unchanged, with its LICENSE).
- Weights: `weights/depth_anything_v2_vits.pth` from https://huggingface.co/depth-anything/Depth-Anything-V2-Small (99,218,434 bytes).
- License: Apache-2.0 (Small model; the TA confirmed its use). Used only for relative depth (layer ownership of the normal branch, `--layering mono`); no training.
