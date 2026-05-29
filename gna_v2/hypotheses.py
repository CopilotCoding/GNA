"""
hypotheses.py — Test four architectural hypotheses against baseline.
Run from gna_v2 directory:
    python hypotheses.py

Tests:
  1. PARALLEL  — mix and channel run simultaneously, outputs summed
  2. WIDE      — 2 blocks at dim=512 vs 4 blocks at dim=256, same params
  3. GLOBAL    — learned summary vector, O(T) cross-sequence mixing
  4. SHIFT     — token shift: free positional context, zero parameters

Baseline: field_mix | identity_channel | no_norm (fastest good result from search)
Ref:      full_attention | mlp | scale_norm (best quality from search)
"""

import math
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import torch
import torch.nn as nn
import torch.nn.functional as F

from gna_v2.data.ptb    import load
from gna_v2.bench.measure import throughput, vram_mb, train_to_convergence


# ─────────────────────────────────────────────────────────────────────────────
# SHARED PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def _init(model):
    """Global std=0.02 init — the key that made full_attention learn."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    return model


class FieldMix(nn.Module):
    """GNA field mix — fastest mixer from search (1.8M tok/s)."""
    def __init__(self, dim, rank=64):
        super().__init__()
        self.proj = nn.Linear(dim, dim,  bias=False)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up   = nn.Linear(rank, dim, bias=False)

    def forward(self, x):
        r = self.proj(x)
        return r + self.up(F.silu(self.down(r)))


class MLP(nn.Module):
    def __init__(self, dim, expand=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expand, bias=False)
        self.fc2 = nn.Linear(dim * expand, dim,  bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class CausalAttn(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.0):
        super().__init__()
        self.dim, self.n_heads = dim, n_heads
        self.d_head = dim // n_heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim,     bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(self.dim, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        return self.out(o.transpose(1, 2).contiguous().view(B, T, D))


# ─────────────────────────────────────────────────────────────────────────────
# HYPOTHESIS 1: PARALLEL BLOCK
# mix and channel run simultaneously — GPU executes both matmul streams
# concurrently. Outputs summed. Same wall time as one path if GPU has headroom.
# ─────────────────────────────────────────────────────────────────────────────

class ParallelBlock(nn.Module):
    """Mix and channel in parallel, not serial."""
    def __init__(self, dim, rank=64, expand=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mix  = FieldMix(dim, rank)
        self.chan = MLP(dim, expand)

    def forward(self, x):
        n = self.norm(x)
        # Both paths see the same normed input simultaneously
        return x + self.mix(n) + self.chan(n)


class ParallelModel(nn.Module):
    name = "PARALLEL"
    def __init__(self, vocab_size, dim=256, n_blocks=4, rank=64, expand=4,
                 dropout=0.1, seq_len=128):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, dim)
        self.pos     = nn.Embedding(seq_len, dim)
        self.drop_p  = dropout
        self.blocks  = nn.Sequential(*[ParallelBlock(dim, rank, expand)
                                       for _ in range(n_blocks)])
        self.norm    = nn.LayerNorm(dim)
        self.decoder = nn.Linear(dim, vocab_size, bias=False)
        self.decoder.weight = self.embed.weight
        _init(self)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        if self.training: h = F.dropout(h, p=self.drop_p)
        h = self.blocks(h)
        return self.decoder(self.norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# HYPOTHESIS 2: WIDE vs DEEP
# 2 blocks at dim=512 vs 4 blocks at dim=256.
# Same param budget. Half the serial depth. GPU prefers wide matmuls.
# ─────────────────────────────────────────────────────────────────────────────

class WideBlock(nn.Module):
    def __init__(self, dim, rank=128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mix  = FieldMix(dim, rank)
        self.norm2 = nn.LayerNorm(dim)
        self.chan  = MLP(dim, expand=4)

    def forward(self, x):
        x = x + self.mix(self.norm(x))
        x = x + self.chan(self.norm2(x))
        return x


class WideModel(nn.Module):
    name = "WIDE"
    def __init__(self, vocab_size, dim=512, n_blocks=2, rank=128,
                 dropout=0.1, seq_len=128):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, dim)
        self.pos     = nn.Embedding(seq_len, dim)
        self.drop_p  = dropout
        self.blocks  = nn.Sequential(*[WideBlock(dim, rank)
                                       for _ in range(n_blocks)])
        self.norm    = nn.LayerNorm(dim)
        self.decoder = nn.Linear(dim, vocab_size, bias=False)
        self.decoder.weight = self.embed.weight
        _init(self)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        if self.training: h = F.dropout(h, p=self.drop_p)
        h = self.blocks(h)
        return self.decoder(self.norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# HYPOTHESIS 3: GLOBAL SUMMARY TOKEN
# One extra learned token that every position writes to and reads from.
# O(T) cross-sequence mixing. No attention matrix. Global context for free.
# ─────────────────────────────────────────────────────────────────────────────

class GlobalBlock(nn.Module):
    """
    Each position writes a summary to a global vector via mean pooling.
    Global vector is then broadcast back and added to each position.
    O(T) — two reductions, no O(T²) matrix.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm      = nn.LayerNorm(dim)
        self.write     = nn.Linear(dim, dim, bias=False)  # position → global
        self.read      = nn.Linear(dim, dim, bias=False)  # global → position
        self.gate      = nn.Parameter(torch.zeros(dim))
        self.norm2     = nn.LayerNorm(dim)
        self.chan      = MLP(dim, expand=4)

    def forward(self, x):
        n = self.norm(x)
        # Write: each position contributes to global summary
        summary = self.write(n).mean(dim=1, keepdim=True)    # [B, 1, D]
        # Read: global summary broadcast back to all positions
        x = x + torch.sigmoid(self.gate) * self.read(summary)
        x = x + self.chan(self.norm2(x))
        return x


class GlobalModel(nn.Module):
    name = "GLOBAL"
    def __init__(self, vocab_size, dim=256, n_blocks=4, dropout=0.1, seq_len=128):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, dim)
        self.pos     = nn.Embedding(seq_len, dim)
        self.drop_p  = dropout
        self.blocks  = nn.Sequential(*[GlobalBlock(dim) for _ in range(n_blocks)])
        self.norm    = nn.LayerNorm(dim)
        self.decoder = nn.Linear(dim, vocab_size, bias=False)
        self.decoder.weight = self.embed.weight
        _init(self)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        if self.training: h = F.dropout(h, p=self.drop_p)
        h = self.blocks(h)
        return self.decoder(self.norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# HYPOTHESIS 4: TOKEN SHIFT
# Shift sequence by 1, add to current. Free positional context.
# Zero learned parameters. One memory op. Pure GPU-native.
# ─────────────────────────────────────────────────────────────────────────────

class ShiftBlock(nn.Module):
    """
    Token shift: x[t] += x[t-1] (causal — only past context).
    Zero parameters. Pure memory op. Then channel transform.
    """
    def __init__(self, dim, rank=64, expand=4):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.mix   = FieldMix(dim, rank)   # still mix after shift
        self.norm2 = nn.LayerNorm(dim)
        self.chan  = MLP(dim, expand)

    def forward(self, x):
        # Causal token shift: prepend zero, drop last
        shifted = torch.roll(x, 1, dims=1)
        shifted[:, 0, :] = 0.0            # zero out position 0 (no past)
        x = x + shifted                   # free context injection
        x = x + self.mix(self.norm(x))
        x = x + self.chan(self.norm2(x))
        return x


class ShiftModel(nn.Module):
    name = "SHIFT"
    def __init__(self, vocab_size, dim=256, n_blocks=4, rank=64, expand=4,
                 dropout=0.1, seq_len=128):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, dim)
        self.pos     = nn.Embedding(seq_len, dim)
        self.drop_p  = dropout
        self.blocks  = nn.Sequential(*[ShiftBlock(dim, rank, expand)
                                       for _ in range(n_blocks)])
        self.norm    = nn.LayerNorm(dim)
        self.decoder = nn.Linear(dim, vocab_size, bias=False)
        self.decoder.weight = self.embed.weight
        _init(self)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        if self.training: h = F.dropout(h, p=self.drop_p)
        h = self.blocks(h)
        return self.decoder(self.norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# BASELINES
# ─────────────────────────────────────────────────────────────────────────────

class BaselineFieldMix(nn.Module):
    """field_mix | identity | no_norm — fastest from search (1.8M tok/s, 326 ppl)"""
    name = "BASELINE_FIELD"
    def __init__(self, vocab_size, dim=256, n_blocks=4, rank=64,
                 dropout=0.1, seq_len=128):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, dim)
        self.pos     = nn.Embedding(seq_len, dim)
        self.drop_p  = dropout
        self.blocks  = nn.Sequential(*[FieldMix(dim, rank) for _ in range(n_blocks)])
        self.norm    = nn.LayerNorm(dim)
        self.decoder = nn.Linear(dim, vocab_size, bias=False)
        self.decoder.weight = self.embed.weight
        _init(self)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        if self.training: h = F.dropout(h, p=self.drop_p)
        h = self.blocks(h)
        return self.decoder(self.norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineAttention(nn.Module):
    """full_attention | mlp | scale_norm — best quality from search (261 ppl)"""
    name = "BASELINE_ATTN"

    class _Block(nn.Module):
        def __init__(self, dim, n_heads=8, expand=4, dropout=0.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn  = CausalAttn(dim, n_heads, dropout)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp   = MLP(dim, expand)
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    def __init__(self, vocab_size, dim=256, n_blocks=4, n_heads=8,
                 dropout=0.1, seq_len=128):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, dim)
        self.pos     = nn.Embedding(seq_len, dim)
        self.drop_p  = dropout
        self.blocks  = nn.Sequential(*[
            self._Block(dim, n_heads, dropout=dropout) for _ in range(n_blocks)])
        self.norm    = nn.LayerNorm(dim)
        self.decoder = nn.Linear(dim, vocab_size, bias=False)
        self.decoder.weight = self.embed.weight
        _init(self)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        if self.training: h = F.dropout(h, p=self.drop_p)
        h = self.blocks(h)
        return self.decoder(self.norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nHypothesis Test — {torch.cuda.get_device_name(device)}")
    print(f"{'='*65}")

    train_ds, valid_ds, vocab_sz = load(seq_len=128, device=device)

    models = [
        BaselineFieldMix(vocab_sz),
        BaselineAttention(vocab_sz),
        ParallelModel(vocab_sz),
        WideModel(vocab_sz),
        GlobalModel(vocab_sz),
        ShiftModel(vocab_sz),
    ]

    results = []

    for model in models:
        model = model.to(device)
        tps   = throughput(model, 128, 128, device)
        vmb   = vram_mb(model,   128, 128, device)
        params = model.count_parameters()

        print(f"\n{'─'*65}")
        print(f"  {model.name:<20} {params:>10,} params  {tps:>12,.0f} tok/s  {vmb:.0f}MB")

        result = train_to_convergence(
            model, train_ds, valid_ds, batch_size=128, device=device,
            max_epochs=20, patience=5, lr=3e-4,
        )

        results.append(dict(
            name    = model.name,
            params  = params,
            tps     = tps,
            ppl     = result["best_val_ppl"],
            epochs  = result["n_epochs"],
            time_m  = result["total_s"] / 60,
        ))

    # Report
    print(f"\n{'='*65}")
    print(f"  HYPOTHESIS RESULTS")
    print(f"{'='*65}")
    print(f"  {'Model':<20} {'Params':>10} {'PPL':>7} {'Tok/s':>12} {'Epochs':>7} {'Time':>6}")
    print(f"  {'─'*61}")
    for r in sorted(results, key=lambda x: x["ppl"]):
        print(f"  {r['name']:<20} {r['params']:>10,} {r['ppl']:>7.1f} "
              f"{r['tps']:>12,.0f} {r['epochs']:>7} {r['time_m']:>5.1f}m")

    print(f"\n  Pareto frontier (quality vs speed):")
    print(f"  {'─'*61}")
    for r in sorted(results, key=lambda x: x["tps"], reverse=True):
        score = r["tps"] / max(r["ppl"], 1)
        print(f"  {r['name']:<20} ppl={r['ppl']:6.1f}  tok/s={r['tps']:>12,.0f}  "
              f"score={score:>8.0f}")


if __name__ == "__main__":
    run()
