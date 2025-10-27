from typing import Optional

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_groups: int = 8,
            cond_time_dim: Optional[int] = None,
            cond_action_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.channel_adaptation = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

        groups = min(num_groups, out_channels)
        while out_channels % groups and groups > 1:
            groups -= 1

        uses_film = (cond_time_dim is not None) or (cond_action_dim is not None)
        self.gn = nn.GroupNorm(groups, out_channels, affine=not uses_film)
        self.act = nn.SiLU(inplace=True)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        # Build only what you need
        self.film_time = None
        if cond_time_dim is not None:
            self.film_time = nn.Linear(cond_time_dim, 2 * out_channels)
            nn.init.zeros_(self.film_time.weight)
            nn.init.zeros_(self.film_time.bias)

        self.film_act = None
        if cond_action_dim is not None:
            self.film_act = nn.Linear(cond_action_dim, 2 * out_channels)
            nn.init.zeros_(self.film_act.weight)
            nn.init.zeros_(self.film_act.bias)

    def forward(self, x, cond_time: Optional[torch.Tensor] = None, cond_action: Optional[torch.Tensor] = None):
        residual = self.channel_adaptation(x)
        y = self.gn(residual)

        gamma: Optional[torch.Tensor] = None
        beta: Optional[torch.Tensor] = None
        if (self.film_time is not None) and (cond_time is not None):
            gt, bt = self.film_time(cond_time).chunk(2, -1)
            gamma = gt if gamma is None else gamma + gt
            beta = bt if beta is None else beta + bt
        if (self.film_act is not None) and (cond_action is not None):
            ga, ba = self.film_act(cond_action).chunk(2, -1)
            gamma = ga if gamma is None else gamma + ga
            beta = ba if beta is None else beta + ba
        if gamma is not None:
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            y = (1 + gamma) * y + beta

        y = self.act(y)
        y = self.conv(y)
        return y + residual
