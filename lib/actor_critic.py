from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from lib.nn_blocks import ResidualBlock


class ActorCritic(nn.Module):
    """
    The weights of the policy and value network are shared except for the last layer. In the
    following, we refer to it as the "actor-critic" network, even though the value network is technically a
    state-value network, not a critic. This network takes as input a frame, and forwards it through convolutional
    trunk followed by an LSTM cell. The convolutional trunk consists of four residual blocks and 2x2
    max-pooling with stride 2. The main path of the residual blocks consists of a group normalization
    layer, a SiLU activation, and a 3x3 convolution with stride 1 and padding 1.
    Before starting the imagination procedure, we burn-in the conditioning frames to
    initialize the hidden and cell states of the LSTM
    """

    def __init__(
            self,
            obs_shape: Tuple[int, ...],
            num_actions: int,
            residual_blocks_layers: List[int, ...],
            residual_blocks_channels: List[int, ...],
            lstm_dimensions: int,
    ):
        super(ActorCritic, self).__init__()
        assert len(residual_blocks_layers) == len(residual_blocks_channels)

        # Convolutional trunk
        layers_list = []
        in_channels = obs_shape[0]
        for _, (block_layers, block_channels) in enumerate(zip(residual_blocks_layers, residual_blocks_channels)):
            layers_list.append(ResidualBlock(in_channels, block_channels, block_layers))
            layers_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = block_channels
        self.conv_trunk = nn.Sequential(*layers_list)

        # LSTM
        self.lstm = nn.LSTM(in_channels, lstm_dimensions, batch_first=True)

        # Heads
        self.policy_head = nn.Linear(lstm_dimensions, num_actions)
        self.value_head = nn.Linear(lstm_dimensions, 1)

        # Initialize heads
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def _preprocess(self, obs: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        device = next(self.parameters()).device

        x = torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs

        # Convert to float in [0,1] if needed
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        else:
            x = x.float()

        return x.to(device)

    def forward(
            self,
            obs: Union[torch.Tensor, np.ndarray],
            state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self._preprocess(obs)
        B, L = x.shape[0], x.shape[1]  # Batch, Length

        # Flatten time into batch for conv trunk
        x = x.reshape(B * L, *x.shape[2:])  # (B*L, C, H, W)
        feats = self.conv_trunk(x)  # (B*L, F)
        feats = feats.view(B, L, -1)  # (B, L, F)

        # LSTM over time
        if state is None or state == (None, None):
            out, (h, c) = self.lstm(feats)  # out: (B, L, H)
        else:
            out, (h, c) = self.lstm(feats, state)

        # Heads
        policy_logits = self.policy_head(out)  # (B, L, A)
        values = self.value_head(out)  # (B, L, 1)
        return policy_logits, values, (h, c)

    @torch.no_grad()
    def sample_action(
            self,
            obs: Union[torch.Tensor, np.ndarray],
            state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        policy_logits, _, new_state = self.forward(obs, state)  # policy_logits: (B,L,A) or (1,1,A)
        logits = policy_logits.reshape(-1, policy_logits.shape[-1])[-1]  # take last time step of last batch
        dist = torch.distributions.Categorical(logits=logits)
        action = int(dist.sample().item())
        return action, new_state
