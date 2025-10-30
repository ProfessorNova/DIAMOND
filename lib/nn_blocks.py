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
            zero_init_last: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.channel_adaptation = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

        groups = min(num_groups, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1

        uses_film = (cond_time_dim is not None) or (cond_action_dim is not None)

        self.gn1 = nn.GroupNorm(groups, out_channels, affine=not uses_film)
        self.gn2 = nn.GroupNorm(groups, out_channels, affine=not uses_film)

        self.act = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        if zero_init_last:
            nn.init.zeros_(self.conv2.weight)

        def make_film(in_dim: int):
            lin = nn.Linear(in_dim, 2 * out_channels)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            return lin

        self.film1_time = make_film(cond_time_dim) if cond_time_dim is not None else None
        self.film1_act = make_film(cond_action_dim) if cond_action_dim is not None else None
        self.film2_time = make_film(cond_time_dim) if cond_time_dim is not None else None
        self.film2_act = make_film(cond_action_dim) if cond_action_dim is not None else None

    @staticmethod
    def _apply_film(y: torch.Tensor,
                    film_time: Optional[nn.Linear],
                    film_act: Optional[nn.Linear],
                    cond_time: Optional[torch.Tensor],
                    cond_action: Optional[torch.Tensor]) -> torch.Tensor:
        gamma = beta = None
        if (film_time is not None) and (cond_time is not None):
            gt, bt = film_time(cond_time).chunk(2, dim=-1)
            gamma = gt if gamma is None else gamma + gt
            beta = bt if beta is None else beta + bt
        if (film_act is not None) and (cond_action is not None):
            ga, ba = film_act(cond_action).chunk(2, dim=-1)
            gamma = ga if gamma is None else gamma + ga
            beta = ba if beta is None else beta + ba
        if gamma is not None:
            y = (1 + gamma.unsqueeze(-1).unsqueeze(-1)) * y + beta.unsqueeze(-1).unsqueeze(-1)
        return y

    def forward(
            self,
            x: torch.Tensor,
            cond_time: Optional[torch.Tensor] = None,
            cond_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = self.channel_adaptation(x)
        y = residual

        # GN1 -> (AdaGN) -> SiLU -> Conv1
        y = self.gn1(y)
        y = self._apply_film(y, self.film1_time, self.film1_act, cond_time, cond_action)
        y = self.act(y)
        y = self.conv1(y)

        # GN2 -> (AdaGN) -> SiLU -> Conv2
        y = self.gn2(y)
        y = self._apply_film(y, self.film2_time, self.film2_act, cond_time, cond_action)
        y = self.act(y)
        y = self.conv2(y)

        return y + residual
