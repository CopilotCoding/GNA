"""
primitives/norm.py — Normalization primitives.
Each wraps a (B, T, D) tensor and returns (B, T, D).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    name = "layer_norm"
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x)


class RMSNorm(nn.Module):
    name = "rms_norm"
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = 1e-6
    def forward(self, x):
        return x * x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt() * self.weight


class ScaleNorm(nn.Module):
    name = "scale_norm"
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(dim ** 0.5))
        self.eps   = 1e-5
    def forward(self, x):
        return self.scale * F.normalize(x, dim=-1, eps=self.eps)


class NoNorm(nn.Module):
    name = "no_norm"
    def __init__(self, dim: int, **kwargs):
        super().__init__()
    def forward(self, x): return x


NORM = {
    "layer_norm":  LayerNorm,
    "rms_norm":    RMSNorm,
    "scale_norm":  ScaleNorm,
    "no_norm":     NoNorm,
}
