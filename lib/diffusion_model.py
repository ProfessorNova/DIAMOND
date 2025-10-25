import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.nn_blocks import ResidualBlock


def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 2 == 0, "sinusoidal emb dim must be even"
    half = dim // 2
    frequencies = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), steps=half, device=x.device, dtype=x.dtype)
    )
    args = x[..., None] * frequencies[None, ...]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(self.up(x))


class DiffusionModel(nn.Module):
    """
    The diffusion model is a standard U-Net 2D, conditioned on the
    last frames and actions, as well as the diffusion time. We use frame stacking for observation
    conditioning, and adaptive group normalization for action and diffusion time
    conditioning.
    """

    def __init__(
            self,
            obs_shape: Tuple[int, ...],
            num_actions: int,
            num_conditioning_obs_and_acts: int,
            residual_blocks_layers: List[int],
            residual_blocks_channels: List[int],
            residual_blocks_conditioning_dimensions: int = 256,
            num_denoising_steps: int = 3,
            sigma_data: float = 0.5,
            p_mean: float = -0.4,
            p_std: float = 1.2,
    ):
        super().__init__()
        assert len(residual_blocks_layers) == len(residual_blocks_channels)

        self.obs_channels, self.H, self.W = obs_shape
        self.L = num_conditioning_obs_and_acts
        self.in_channels = self.obs_channels * (self.L + 1)  # noisy next frame + L past clean frames

        self.cond_dim = residual_blocks_conditioning_dimensions
        self.num_denoising_steps = int(num_denoising_steps)
        self.sigma_data = float(sigma_data)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)

        # Precompute noise schedule and coefficients
        z = torch.linspace(1.0, -1.0, steps=num_denoising_steps)

        log_sigma = self.p_mean + self.p_std * z  # (N,)
        sigma_i = torch.exp(log_sigma)  # (N,)
        sigma_next = torch.cat([sigma_i[1:], torch.zeros(1, dtype=sigma_i.dtype)], dim=0)  # (N,)

        denomination = torch.sqrt(sigma_i * sigma_i + self.sigma_data * self.sigma_data)

        c_in_all = 1.0 / denomination
        c_out_all = (sigma_i * self.sigma_data) / denomination
        c_skip_all = (self.sigma_data * self.sigma_data) / (self.sigma_data * self.sigma_data + sigma_i * sigma_i)
        c_noise_all = 0.25 * sigma_i.log()
        inv_sigma = 1.0 / sigma_i
        delta_sigma = sigma_next - sigma_i

        # Register as buffers so they follow .to(device/dtype)
        self.register_buffer("c_in_all", c_in_all, persistent=False)  # (N,)
        self.register_buffer("c_out_all", c_out_all, persistent=False)  # (N,)
        self.register_buffer("c_skip_all", c_skip_all, persistent=False)  # (N,)
        self.register_buffer("c_noise_all", c_noise_all, persistent=False)  # (N,)
        self.register_buffer("inv_sigma_all", inv_sigma, persistent=False)  # (N,)
        self.register_buffer("delta_all", delta_sigma, persistent=False)  # (N,)

        # Embeddings
        self.action_embedding = nn.Embedding(num_actions, self.cond_dim)
        self.action_pe_proj = nn.Linear(self.L * self.cond_dim, self.cond_dim)

        # Stem
        self.conv_in = nn.Conv2d(self.in_channels, residual_blocks_channels[0], kernel_size=3, padding=1, bias=False)

        # Down path
        downs = []
        in_ch = residual_blocks_channels[0]
        for idx, (out_ch, n_layers) in enumerate(zip(residual_blocks_channels, residual_blocks_layers)):
            downs.append(
                ResidualBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    layers=n_layers,
                    cond_dim=self.cond_dim
                )
            )
            downs.append(
                ResidualBlock(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    layers=n_layers,
                    cond_dim=self.cond_dim
                )
            )
            # Downsample except for last block
            if idx < len(residual_blocks_channels) - 1:
                next_ch = residual_blocks_channels[idx + 1]
                downs.append(Downsample(out_ch, next_ch))
                in_ch = next_ch
        self.down = nn.ModuleList(downs)

        # Bottleneck
        self.mid = nn.ModuleList([
            ResidualBlock(
                in_channels=in_ch,
                out_channels=in_ch,
                cond_dim=self.cond_dim
            ),
            ResidualBlock(
                in_channels=in_ch,
                out_channels=in_ch,
                cond_dim=self.cond_dim
            ),
        ])

        # Up path
        ups = []
        num_scales = len(residual_blocks_channels)
        in_ch_up = in_ch
        for level in reversed(range(num_scales)):
            out_ch = residual_blocks_channels[level]
            n_layers = residual_blocks_layers[level]

            if level == num_scales - 1:
                in_cur = in_ch_up
            else:
                in_cur = in_ch_up + out_ch

            # First block at this scale
            ups.append(
                ResidualBlock(
                    in_channels=in_cur,
                    out_channels=out_ch,
                    layers=n_layers,
                    cond_dim=self.cond_dim
                )
            )
            ups.append(
                ResidualBlock(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    layers=n_layers,
                    cond_dim=self.cond_dim
                )
            )
            in_cur = out_ch

            if level > 0:
                ups.append(Upsample(in_cur, residual_blocks_channels[level - 1]))
                in_ch_up = residual_blocks_channels[level - 1]
            else:
                in_ch_up = in_cur
        self.up = nn.ModuleList(ups)

        # Head
        self.conv_out = nn.Conv2d(in_ch_up, self.obs_channels, kernel_size=1, bias=False)

    def _build_conditions(self, c_noise, act_cond):
        B = c_noise.shape[0]
        n_emb = sinusoidal_embedding(c_noise.reshape(B), dim=self.cond_dim)  # (B, D)

        acts = self.action_embedding(act_cond)  # (B, L, D)
        pos = sinusoidal_embedding(torch.arange(self.L, device=acts.device), dim=self.cond_dim)  # (L, D)

        acts_pe = acts + pos.unsqueeze(0)  # (B, L, D)
        a_emb = self.action_pe_proj(acts_pe.reshape(B, self.L * self.cond_dim))  # (B, D)
        return n_emb, a_emb

    def forward(self, x_in, c_noise, obs_cond, act_cond):
        B, C, H, W = x_in.shape
        assert C == self.obs_channels and H == self.H and W == self.W
        assert obs_cond.shape == (B, self.L, self.obs_channels, H, W)
        assert act_cond.shape == (B, self.L)

        x_hist = obs_cond.reshape(B, self.L * self.obs_channels, H, W)
        x = torch.cat([x_in, x_hist], dim=1)

        cond_time, cond_action = self._build_conditions(c_noise, act_cond)

        # Down
        x = self.conv_in(x)
        skips: List[torch.Tensor] = []
        for layer in self.down:
            if isinstance(layer, ResidualBlock):
                x = layer(x, cond_time, cond_action)
            else:
                skips.append(x)
                x = layer(x)

        # Mid
        for layer in self.mid:
            x = layer(x, cond_time, cond_action)

        # Up
        skip_idx = len(skips) - 1
        need_skip = False
        for layer in self.up:
            if isinstance(layer, Upsample):
                x = layer(x)
                need_skip = True
            else:
                if need_skip:
                    target_hw = skips[skip_idx].shape[-2:]
                    if x.shape[-2:] != target_hw:
                        x = F.interpolate(x, size=target_hw, mode="nearest")
                    x = torch.cat([x, skips[skip_idx]], dim=1)
                    skip_idx -= 1
                    need_skip = False
                x = layer(x, cond_time, cond_action)

        return self.conv_out(x)

    @torch.no_grad()
    def sample_next_observation(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = obs_hist.shape
        device = obs_hist.device
        dtype = obs_hist.dtype
        assert L == self.L
        assert C == self.obs_channels
        assert H == self.H
        assert W == self.W

        # init noise
        x = torch.randn(B, C, H, W, device=device, dtype=dtype) * math.exp(self.p_mean + self.p_std * 1.0)

        for i in range(self.num_denoising_steps):
            c_in = self.c_in_all[i]
            c_out = self.c_out_all[i]
            c_skip = self.c_skip_all[i]
            inv_sig = self.inv_sigma_all[i]
            delta = self.delta_all[i]
            c_noise = self.c_noise_all[i].expand(B, 1, 1, 1)

            F_pred = self.forward(c_in * x, c_noise, obs_hist, act_hist)
            denoised = c_skip * x + c_out * F_pred
            d = (x - denoised) * inv_sig
            x = x + delta * d

        return x.clamp(0.0, 1.0)
