"""
primitives/channel.py — Channel transformation primitives.
All constructors accept **kwargs to swallow irrelevant cfg keys.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    name = "mlp"
    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.0, **kwargs):
        super().__init__()
        inner    = dim * expand
        self.fc1 = nn.Linear(dim, inner, bias=False)
        self.fc2 = nn.Linear(inner, dim, bias=False)
        self.drop = dropout
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        if self.training and self.drop > 0:
            h = F.dropout(h, p=self.drop)
        return self.fc2(h)


class SwiGLU(nn.Module):
    name = "swiglu"
    def __init__(self, dim: int, expand: int = 4, **kwargs):
        super().__init__()
        inner     = dim * expand
        self.gate = nn.Linear(dim, inner, bias=False)
        self.val  = nn.Linear(dim, inner, bias=False)
        self.out  = nn.Linear(inner, dim, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.val.weight,  std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x):
        return self.out(F.silu(self.gate(x)) * self.val(x))


class GatedChannel(nn.Module):
    name = "gated_channel"
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x):
        return x + torch.sigmoid(self.gate) * F.silu(self.proj(x))


class SquaredReLUMLP(nn.Module):
    name = "sqrelu_mlp"
    def __init__(self, dim: int, expand: int = 4, **kwargs):
        super().__init__()
        inner    = dim * expand
        self.fc1 = nn.Linear(dim, inner, bias=False)
        self.fc2 = nn.Linear(inner, dim, bias=False)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)).pow(2))


class IdentityChannel(nn.Module):
    name = "identity_channel"
    def __init__(self, dim: int, **kwargs):
        super().__init__()
    def forward(self, x): return x


CHANNEL = {
    "mlp":              MLP,
    "swiglu":           SwiGLU,
    "gated_channel":    GatedChannel,
    "sqrelu_mlp":       SquaredReLUMLP,
    "identity_channel": IdentityChannel,
}
