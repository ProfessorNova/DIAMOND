import math
from typing import List, Tuple

import torch
import torch.nn as nn

from lib.nn_blocks import ResidualBlock


def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 2 == 0, "sinusoidal emb dim must be even"
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), steps=half, device=x.device, dtype=x.dtype)
    )
    args = x[..., None] * freqs[None, ...]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


def karras_sigmas(n: int, sigma_min: float, sigma_max: float, rho: float, device) -> torch.Tensor:
    i = torch.linspace(0, 1, steps=n, device=device)
    sigmas = (sigma_max ** (1 / rho) + i * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])], dim=0)
    return sigmas


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
            sigma_min: float = 0.01,
            sigma_max: float = 1.0,
            rho: float = 7.0,
    ):
        super().__init__()
        assert len(residual_blocks_layers) == len(residual_blocks_channels)

        self.obs_channels, self.H, self.W = obs_shape
        self.L = num_conditioning_obs_and_acts
        self.in_channels = self.obs_channels * (self.L + 1)  # noisy next frame + L past clean frames

        self.cond_dim = residual_blocks_conditioning_dimensions
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.num_denoising_steps = int(num_denoising_steps)

        # Embeddings
        self.action_embedding = nn.Embedding(num_actions, self.cond_dim)

        # Stem
        self.conv_in = nn.Conv2d(self.in_channels, residual_blocks_channels[0], kernel_size=3, padding=1)

        # Down path
        downs = []
        in_ch = residual_blocks_channels[0]
        self.down_skips_channels: List[int] = []

        for idx, (out_ch, n_layers) in enumerate(zip(residual_blocks_channels, residual_blocks_layers)):
            for _ in range(n_layers):
                downs.append(ResidualBlock(in_channels=in_ch, out_channels=out_ch, layers=1, cond_dim=self.cond_dim))
                in_ch = out_ch
            self.down_skips_channels.append(in_ch)  # skip after this scale's residual stack
            if idx < len(residual_blocks_channels) - 1:
                next_ch = residual_blocks_channels[idx + 1]
                downs.append(Downsample(in_ch, next_ch))
                in_ch = next_ch

        self.down = nn.ModuleList(downs)

        # Bottleneck
        self.mid = nn.ModuleList([
            ResidualBlock(in_channels=in_ch, out_channels=in_ch, layers=1, cond_dim=self.cond_dim),
            ResidualBlock(in_channels=in_ch, out_channels=in_ch, layers=1, cond_dim=self.cond_dim),
        ])

        # Up path
        ups = []
        num_scales = len(residual_blocks_channels)
        in_ch_up = in_ch  # from bottleneck

        for level in reversed(range(num_scales)):
            out_ch = residual_blocks_channels[level]
            n_layers = residual_blocks_layers[level]

            if level == num_scales - 1:
                # Deepest scale: no skip yet
                in_cur = in_ch_up
            else:
                # After upsample we will concat the skip (out_ch at this scale)
                in_cur = in_ch_up + out_ch

            # First block at this scale
            ups.append(ResidualBlock(in_channels=in_cur, out_channels=out_ch, layers=1, cond_dim=self.cond_dim))
            in_cur = out_ch

            # Remaining blocks at this scale (if any) take out_ch -> out_ch
            for _ in range(n_layers - 1):
                ups.append(ResidualBlock(in_channels=in_cur, out_channels=out_ch, layers=1, cond_dim=self.cond_dim))
                in_cur = out_ch

            if level > 0:
                # Prepare for the next finer scale
                ups.append(Upsample(in_cur, residual_blocks_channels[level - 1]))
                in_ch_up = residual_blocks_channels[level - 1]
            else:
                in_ch_up = in_cur

        self.up = nn.ModuleList(ups)

        # Head
        self.conv_out = nn.Conv2d(in_ch_up, self.obs_channels, kernel_size=1)

    def _build_condition(self, c_noise: torch.Tensor, act_cond: torch.Tensor) -> torch.Tensor:
        B = c_noise.shape[0]
        n_emb = sinusoidal_embedding(c_noise.view(B), dim=self.cond_dim)  # (B, cond_dim)
        a_emb = self.action_embedding(act_cond).mean(dim=1)  # (B, L, cond_dim) -> (B, cond_dim)
        cond = n_emb + a_emb  # (B, cond_dim)
        return cond

    def forward(self, x_in, c_noise, obs_cond, act_cond):
        B, C, H, W = x_in.shape
        assert C == self.obs_channels and H == self.H and W == self.W
        assert obs_cond.shape[1] == self.L and obs_cond.shape[2] == self.obs_channels

        x_hist = obs_cond.reshape(B, self.L * self.obs_channels, H, W)
        x = torch.cat([x_in, x_hist], dim=1)

        cond = self._build_condition(c_noise, act_cond)  # (B, cond_dim)

        # Down
        x = self.conv_in(x)
        skips: List[torch.Tensor] = []
        for layer in self.down:
            if isinstance(layer, ResidualBlock):
                x = layer(x, cond)
            else:
                skips.append(x)  # save before each downsample
                x = layer(x)

        # Mid
        for layer in self.mid:
            x = layer(x, cond)

        # Up — concat skip ONCE right after each upsample
        skip_idx = len(skips) - 1
        need_skip = False
        for layer in self.up:
            if isinstance(layer, Upsample):
                x = layer(x)
                need_skip = True
            else:
                if need_skip:
                    # concat the corresponding skip exactly once at this scale
                    x = torch.cat([x, skips[skip_idx]], dim=1)
                    skip_idx -= 1
                    need_skip = False
                x = layer(x, cond)

        return self.conv_out(x)

    @torch.no_grad()
    def sample_next_observation(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = obs_hist.shape
        device = obs_hist.device
        dtype = obs_hist.dtype
        assert L == self.L and C == self.obs_channels and H == self.H and W == self.W

        x = torch.randn(B, C, H, W, device=device, dtype=dtype) * self.sigma_max
        sigmas = karras_sigmas(self.num_denoising_steps, self.sigma_min, self.sigma_max, self.rho, device)

        for i in range(self.num_denoising_steps):
            sigma_i = sigmas[i]
            sigma_next = sigmas[i + 1]

            denom = torch.sqrt(sigma_i ** 2 + self.sigma_data ** 2)
            c_in = 1.0 / denom
            c_out = (sigma_i * self.sigma_data) / denom
            c_skip = (self.sigma_data ** 2) / (self.sigma_data ** 2 + sigma_i ** 2)
            c_noise = (0.25 * sigma_i.log()).expand(B, 1, 1, 1)

            F_pred = self.forward(c_in * x, c_noise, obs_hist, act_hist)
            denoised = c_skip * x + c_out * F_pred
            d = (x - denoised) / sigma_i
            x = x + (sigma_next - sigma_i) * d

        return x.clamp(0.0, 1.0)
