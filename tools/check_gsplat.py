"""GPU integration check: packed white compositing vs dense white backgrounds."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python_packages'))
os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(ROOT/'cache/torch_extensions'))
os.environ.setdefault('MAX_JOBS','4')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0')
import torch
from gsplat import rasterization

torch.set_num_threads(4)
with torch.no_grad():
    means=torch.tensor([[0.,0.,2.],[.2,0.,2.5]],device='cuda')
    quats=torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]],device='cuda')
    scales=torch.full((2,3),.15,device='cuda')
    opacities=torch.tensor([.8,.7],device='cuda')
    colours=torch.tensor([[1.,.2,.1],[.1,.3,1.]],device='cuda')
    views=torch.eye(4,device='cuda').repeat(2,1,1);views[1,0,3]=.06
    ks=torch.tensor([[80.,0.,32.],[0.,80.,24.],[0.,0.,1.]],device='cuda').repeat(2,1,1)
    packed,alpha,_=rasterization(means,quats,scales,opacities,colours,views,ks,64,48,packed=True)
    dense,dense_alpha,_=rasterization(means,quats,scales,opacities,colours,views,ks,64,48,
        packed=False,backgrounds=torch.ones(2,3,device='cuda'))
    error=(packed+(1-alpha)-dense).abs().max().item()
    alpha_error=(alpha-dense_alpha).abs().max().item()
    assert packed.shape==(2,48,64,3)
    assert error<2e-6 and alpha_error<2e-6,(error,alpha_error)
result={'passed':True,'packed_vs_dense_white_max_error':error,'alpha_max_error':alpha_error,'cameras':2}
(ROOT/'logs/gsplat_integration.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
