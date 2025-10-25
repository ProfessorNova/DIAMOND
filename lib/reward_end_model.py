import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn

from lib.nn_blocks import ResidualBlock


class RewardEndModel(nn.Module):
    """
    The reward/termination model layers are shared except for the final prediction heads.
    The model takes as input a sequence of frames and actions, and forwards it through convolutional residual blocks
    followed by an LSTM cell. Before starting the imagination procedure,
    we burn-in the conditioning frames and actions to initialize the hidden and cell states of the LSTM.
    """

    def __init__(
            self,
            obs_shape: Tuple[int, ...],
            num_actions: int,
            residual_blocks_layers: List[int],
            residual_blocks_channels: List[int],
            residual_blocks_conditioning_dimensions: int,
            lstm_dimensions: int,
    ):
        super().__init__()
        assert len(residual_blocks_layers) == len(residual_blocks_channels)

        # Conditioning for actions
        self.action_embedding = nn.Embedding(num_actions, residual_blocks_conditioning_dimensions)

        # Convolutional residual blocks
        layers_list = []
        in_channels = obs_shape[0]
        for out_channels, num_layers in zip(residual_blocks_channels, residual_blocks_layers):
            layers_list.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    layers=num_layers,
                    cond_dim=residual_blocks_conditioning_dimensions,
                )
            )
            layers_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.convolutional_residual_blocks = nn.ModuleList(layers_list)

        # Compute LSTM input size AFTER pooling (== last out_channels)
        conv_output_size = self._get_conv_output_size(obs_shape)

        # LSTM over time
        self.lstm = nn.LSTM(conv_output_size, lstm_dimensions, batch_first=True)

        # Heads
        self.reward_head = nn.Linear(lstm_dimensions, 3)
        self.termination_head = nn.Linear(lstm_dimensions, 2)

        # Initialize heads
        # reward_head: classes [-1, 0, +1]
        p_zero = 0.90
        p_neg = p_pos = (1.0 - p_zero) / 2.0
        nn.init.normal_(self.reward_head.weight, std=0.01)
        self.reward_head.bias.data = torch.log(torch.tensor([p_neg, p_zero, p_pos], dtype=torch.float))
        # termination_head: [not-done, done]
        p_done = 0.05
        nn.init.normal_(self.termination_head.weight, std=0.01)
        logit_done = math.log(p_done / (1.0 - p_done))
        self.termination_head.bias.data = torch.tensor([0.0, logit_done], dtype=torch.float)

    def _get_conv_output_size(self, obs_shape):
        with torch.no_grad():
            x = torch.zeros(1, *obs_shape)
            cond = torch.zeros(1, self.action_embedding.embedding_dim, device=x.device, dtype=x.dtype)
            for m in self.convolutional_residual_blocks:
                if isinstance(m, ResidualBlock):
                    x = m(x, cond_action=cond)
                else:
                    x = m(x)
            return x.reshape(1, -1).size(1)

    def forward(
            self,
            obs: torch.Tensor,  # (B, L, C, H, W)
            actions: torch.Tensor,  # (B, L)
            state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        B, L = obs.shape[0], obs.shape[1]

        # Action conditioning for AdaGN
        cond = self.action_embedding(actions).reshape(B * L, -1)  # (B*L, conditioning_dimensions)

        # Conv trunk (flatten time into batch), then pool away H,W
        x = obs.reshape(B * L, *obs.shape[2:])  # (B*L, C, H, W)
        for m in self.convolutional_residual_blocks:
            if isinstance(m, ResidualBlock):
                x = m(x, cond_action=cond)
            else:
                x = m(x)
        x = x.reshape(B, L, -1)  # (B, L, conv_output_size)

        # LSTM over time
        out, (h, c) = self.lstm(x, None if state in (None, (None, None)) else state)

        # Heads
        reward_logits = self.reward_head(out)  # (B, L, 3)
        termination_logits = self.termination_head(out)  # (B, L, 2)
        return reward_logits, termination_logits, (h, c)
