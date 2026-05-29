"""
blocks/model.py — Language model wrapper.
Global std=0.02 init applied after all blocks built — same pattern as working Transformer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from gna_v2.blocks.block import Block


class LM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dim        = cfg["dim"]
        vocab_size = cfg["vocab_size"]
        n_blocks   = cfg.get("n_blocks", 6)
        dropout    = cfg.get("dropout",  0.1)
        seq_len    = cfg.get("seq_len",  128)

        self.embed   = nn.Embedding(vocab_size, dim)
        self.pos     = nn.Embedding(seq_len, dim)
        self.drop_p  = dropout
        self.blocks  = nn.Sequential(*[Block(cfg) for _ in range(n_blocks)])
        self.norm    = nn.LayerNorm(dim)
        self.decoder = nn.Linear(dim, vocab_size, bias=False)
        self.decoder.weight = self.embed.weight

        # Global init — hits every Linear and Embedding uniformly.
        # This is the key: consistent std=0.02 everywhere, no dead gradients.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        h    = self.embed(x) + self.pos(pos)
        if self.training and self.drop_p > 0:
            h = F.dropout(h, p=self.drop_p)
        h = self.blocks(h)
        h = self.norm(h)
        return self.decoder(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
