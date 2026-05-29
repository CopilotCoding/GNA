"""
primitives/mixing.py — Token mixing primitives.
Every way information can flow between sequence positions.
Each is a drop-in: (B, T, D) -> (B, T, D)
All constructors accept **kwargs to swallow irrelevant cfg keys.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    name = "full_attention"
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.0, **kwargs):
        super().__init__()
        self.dim, self.n_heads = dim, n_heads
        self.d_head    = dim // n_heads
        self.dropout_p = dropout
        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.out  = nn.Linear(dim, dim,     bias=False)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(self.dim, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        return self.out(o.transpose(1, 2).contiguous().view(B, T, D))


class LinearAttention(nn.Module):
    name = "linear_attention"
    def __init__(self, dim: int, n_heads: int = 8, **kwargs):
        super().__init__()
        self.dim, self.n_heads = dim, n_heads
        self.d_head = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim,     bias=False)
        nn.init.normal_(self.qkv.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(self.dim, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        kv  = torch.cumsum(k.unsqueeze(-1) * v.unsqueeze(-2), dim=2)
        z   = 1.0 / ((k * q).sum(-1, keepdim=True).cumsum(2) + 1e-6)
        o   = (q.unsqueeze(-2) @ kv).squeeze(-2) * z
        return self.out(o.transpose(1, 2).contiguous().view(B, T, D))


class SSM(nn.Module):
    name = "ssm"
    def __init__(self, dim: int, state_dim: int = 64, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.in_proj   = nn.Linear(dim, state_dim * 2, bias=False)
        self.out_proj  = nn.Linear(state_dim, dim,     bias=False)
        self.log_A     = nn.Parameter(torch.zeros(state_dim))
        self.B         = nn.Parameter(torch.randn(state_dim) * 0.02)
        nn.init.normal_(self.in_proj.weight,  std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        h  = self.in_proj(x)
        u  = h[..., :self.state_dim]
        g  = torch.sigmoid(h[..., self.state_dim:])
        log_A_cum = self.log_A.unsqueeze(0).unsqueeze(0).expand(B, T, -1).cumsum(1)
        A_pre     = torch.exp(torch.roll(log_A_cum, 1, 1))
        A_pre[:, 0, :] = 1.0
        b_cum  = torch.cumsum(self.B * u / A_pre.clamp(min=1e-7), dim=1)
        states = A_pre * b_cum * g
        return self.out_proj(states)


class LocalConv(nn.Module):
    name = "local_conv"
    def __init__(self, dim: int, kernel_size: int = 7, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size,
                              padding=kernel_size - 1, groups=dim, bias=False)
        nn.init.normal_(self.conv.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        return self.conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)


class FieldMix(nn.Module):
    name = "field_mix"
    def __init__(self, dim: int, rank: int = 64, **kwargs):
        super().__init__()
        self.proj = nn.Linear(dim, dim,  bias=False)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up   = nn.Linear(rank, dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        nn.init.orthogonal_(self.down.weight)
        nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, x):
        r = self.proj(x)
        return r + self.up(F.silu(self.down(r)))


class Hadamard(nn.Module):
    name = "hadamard"
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = torch.fft.rfft(x, dim=1, norm="ortho")
        return torch.fft.irfft(out, n=x.shape[1], dim=1, norm="ortho") * self.scale


class Identity(nn.Module):
    name = "identity"
    def __init__(self, dim: int, **kwargs):
        super().__init__()
    def forward(self, x): return x


MIXING = {
    "full_attention":   FullAttention,
    "linear_attention": LinearAttention,
    "ssm":              SSM,
    "local_conv":       LocalConv,
    "field_mix":        FieldMix,
    "hadamard":         Hadamard,
    "identity":         Identity,
}
