from typing import Optional

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            layers: int = 1,
            num_groups: int = 8,
            cond_dim: Optional[int] = None,
    ):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim

        # If channel dimensions differ, adapt with 1x1 convolution
        self.channel_adaptation = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

        groups = min(num_groups, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1  # Ensure groups divide out_channels

        # Build per-layer GN -> (AdaGN) -> SiLU -> Conv(3x3)
        self.gns = nn.ModuleList([nn.GroupNorm(groups, out_channels, affine=False) for _ in range(layers)])
        self.convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, 1, 1) for _ in range(layers)])
        self.act = nn.SiLU()

        # For AdaGN: one small linear per layer producing [gamma, beta]
        if cond_dim is not None:
            self.films = nn.ModuleList([nn.Linear(cond_dim, 2 * out_channels) for _ in range(layers)])
        else:
            self.films = None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self.channel_adaptation(x)
        out = residual
        for i, (gn, conv) in enumerate(zip(self.gns, self.convs)):
            out = gn(out)

            # Apply AdaGN if conditioning is provided
            if self.films is not None:
                if cond is None:
                    raise ValueError("ResidualBlock expects a conditioning vector 'cond' when cond_dim is set.")
                gb = self.films[i]
                gamma, beta = gb.chunk(2, dim=-1)
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)
                out = (1 + gamma) * out + beta

            out = self.act(out)
            out = conv(out)
        return out + residual
