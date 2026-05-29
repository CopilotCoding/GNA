"""
blocks/block.py — Composable block.
Assembles norm + mixing + norm + channel into one layer.
"""
import torch.nn as nn
from gna_v2.primitives.mixing  import MIXING
from gna_v2.primitives.channel import CHANNEL
from gna_v2.primitives.norm    import NORM


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dim  = cfg["dim"]

        self.prenorm  = cfg.get("prenorm",  True)
        self.residual = cfg.get("residual", True)

        NormCls = NORM   [cfg.get("norm",    "layer_norm")]
        MixCls  = MIXING [cfg.get("mixing",  "full_attention")]
        ChanCls = CHANNEL[cfg.get("channel", "mlp")]

        # Norms only need dim
        self.norm1 = NormCls(dim)
        self.norm2 = NormCls(dim)
        # Mixing and channel get full cfg for their extra params
        self.mix   = MixCls(dim,  **{k: v for k, v in cfg.items() if k != "dim"})
        self.chan  = ChanCls(dim, **{k: v for k, v in cfg.items() if k != "dim"})

    def forward(self, x):
        if self.prenorm:
            x = x + self.mix(self.norm1(x))  if self.residual else self.mix(self.norm1(x))
            x = x + self.chan(self.norm2(x)) if self.residual else self.chan(self.norm2(x))
        else:
            x = self.norm1(x + self.mix(x)  if self.residual else self.mix(x))
            x = self.norm2(x + self.chan(x) if self.residual else self.chan(x))
        return x
