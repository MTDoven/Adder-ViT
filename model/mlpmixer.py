"""
MLP-Mixer: An all-MLP Architecture for Vision
Paper: https://arxiv.org/abs/2105.01601
Codes: https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
"""

from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce


pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0.):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        OperateBase(dim, inner_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        OperateBase(inner_dim, dim),
        nn.Dropout(dropout)
    )

class MLPMixer(nn.Module):
    def __init__(self, *, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
        super().__init__()
        image_h, image_w = pair(image_size)
        num_patches = (image_h // patch_size) * (image_w // patch_size)
        self.in_func = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
                                nn.Linear((patch_size ** 2) * channels, dim))
        self.out_func = nn.Sequential(nn.LayerNorm(dim),
                                 Reduce('b n c -> b c', 'mean'),
                                 nn.Linear(dim, num_classes))
        self.middle_list_first = nn.ModuleList([
            PreNormResidual(num_patches, FeedForward(num_patches, expansion_factor, dropout)) for _ in range(depth)])
        self.middle_list_last = nn.ModuleList([
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout)) for _ in range(depth)])
        self.depth = depth
    def forward(self, x):
        x = self.in_func(x)
        for i in range(self.depth):
            x = x.transpose(-2,-1)
            x = self.middle_list_first[i](x)
            x = x.transpose(-2,-1)
            x = self.middle_list_last[i](x)
        x = self.out_func(x)
        return x





ConvBase = None  
OperateBase = None
def get_model(operate_base, conv_base=None **kwargs):
    global OperateBase
    OperateBase = operate_base
    return MLPMixer(**kwargs)