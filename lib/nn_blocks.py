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
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

        groups = min(num_groups, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1

        # Per-layer: GN -> (AdaGN) -> SiLU -> Conv(3x3)
        self.gns = nn.ModuleList([nn.GroupNorm(groups, out_channels, affine=False) for _ in range(layers)])
        self.convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) for _ in range(layers)
        ])
        self.act = nn.SiLU(inplace=True)

        # Two *separate* FiLM projections per layer: one for time, one for action.
        # Each maps conditioning_dimensions -> 2*out_channels  (gamma, beta)
        if cond_dim is not None:
            self.films_time = nn.ModuleList([nn.Linear(cond_dim, 2 * out_channels) for _ in range(layers)])
            self.films_act = nn.ModuleList([nn.Linear(cond_dim, 2 * out_channels) for _ in range(layers)])
        else:
            self.films_time = nn.ModuleList()
            self.films_act = nn.ModuleList()

    def forward(
            self,
            x: torch.Tensor,
            cond_time: Optional[torch.Tensor] = None,  # (B, conditioning_dimensions) or None
            cond_action: Optional[torch.Tensor] = None,  # (B, conditioning_dimensions) or None
    ) -> torch.Tensor:
        residual = self.channel_adaptation(x)
        out = residual

        use_time = len(self.films_time) > 0 and cond_time is not None
        use_act = len(self.films_act) > 0 and cond_action is not None
        use_film = use_time or use_act

        for i, (gn, conv) in enumerate(zip(self.gns, self.convs)):
            out = gn(out)

            if use_film:
                gamma_terms = []
                beta_terms = []

                if use_time:
                    t = self.films_time[i](cond_time)  # (B, 2*C_out)
                    gt, bt = t.chunk(2, dim=-1)
                    gamma_terms.append(gt)
                    beta_terms.append(bt)

                if use_act:
                    a = self.films_act[i](cond_action)  # (B, 2*C_out)
                    ga, ba = a.chunk(2, dim=-1)
                    gamma_terms.append(ga)
                    beta_terms.append(ba)

                if gamma_terms:  # at least one branch present
                    gamma = torch.stack(gamma_terms, dim=0).sum(dim=0).unsqueeze(-1).unsqueeze(-1)
                    beta = torch.stack(beta_terms, dim=0).sum(dim=0).unsqueeze(-1).unsqueeze(-1)
                    out = (1 + gamma) * out + beta

            out = self.act(out)
            out = conv(out)

        return out + residual
