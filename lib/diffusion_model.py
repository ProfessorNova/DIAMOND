from typing import List, Tuple

import torch.nn as nn


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
            residual_blocks_layers: List[int, ...],
            residual_blocks_channels: List[int, ...],
            residual_blocks_conditioning_dimensions: int,
    ):
        super(DiffusionModel, self).__init__()
        assert len(residual_blocks_layers) == len(residual_blocks_channels)
