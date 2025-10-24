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
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1x1 to match channels for the residual path if needed
        self.channel_adaptation = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

        groups = min(num_groups, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1

        # Per-layer: GN -> (AdaGN) -> SiLU -> Conv(3x3)
        self.gns = nn.ModuleList([nn.GroupNorm(groups, out_channels, affine=False) for _ in range(layers)])
        self.convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, 1, 1) for _ in range(layers)])
        self.act = nn.SiLU()

        # Always a ModuleList (maybe empty) -> avoids Optional typing headaches
        self.films = nn.ModuleList(
            [nn.Linear(cond_dim, 2 * out_channels) for _ in range(layers)]
        ) if cond_dim is not None else nn.ModuleList()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self.channel_adaptation(x)
        out = residual

        use_film = len(self.films) > 0
        if use_film and cond is None:
            raise ValueError("ResidualBlock received no 'cond' but was built with conditioning layers.")

        for i, (gn, conv) in enumerate(zip(self.gns, self.convs)):
            out = gn(out)

            if use_film:
                film_out: torch.Tensor = self.films[i](cond)  # (B, 2*C_out)
                gamma, beta = film_out.chunk(2, dim=-1)  # split features
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C_out, 1, 1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)  # (B, C_out, 1, 1)
                out = (1 + gamma) * out + beta

            out = self.act(out)
            out = conv(out)

        return out + residual
