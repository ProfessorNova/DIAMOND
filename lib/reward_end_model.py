from typing import Tuple, List

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
            residual_blocks_layers: List[int, ...],
            residual_blocks_channels: List[int, ...],
            residual_blocks_conditioning_dimensions: int,
            lstm_dimensions: int,
    ):
        super(RewardEndModel, self).__init__()
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
            in_channels = out_channels
        self.convolutional_residual_blocks = nn.Sequential(*layers_list)

        # LSTM
        self.lstm = nn.LSTM(in_channels, lstm_dimensions, batch_first=True)

        # Heads
        self.reward_head = nn.Linear(lstm_dimensions, 3)
        self.termination_head = nn.Linear(lstm_dimensions, 2)

        # Initialize heads
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)
        nn.init.zeros_(self.termination_head.weight)
        nn.init.zeros_(self.termination_head.bias)
