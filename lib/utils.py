import gymnasium as gym
import numpy as np
import torch

from lib.config import Config


class ImageToPyTorch(gym.ObservationWrapper):
    """HWC -> CHW for PyTorch, keeps dtype uint8."""

    def __init__(self, env):
        super().__init__(env)
        h, w, c = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(c, h, w), dtype=np.uint8
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


def make_env(cfg: Config) -> gym.Env:
    env = gym.make(cfg.env_id)
    env = gym.wrappers.ResizeObservation(env, cfg.image_size)
    env = ImageToPyTorch(env)
    return env


def lambda_returns(
        rewards: torch.Tensor,
        done: torch.Tensor,
        values_with_bootstrap: torch.Tensor,
        gamma: float,
        lam: float
) -> torch.Tensor:
    """
    Calculate lambda-returns.

    :param rewards: (B, H)
    :param done: (B, H)
    :param values_with_bootstrap: (B, H+1)
    :param gamma: discount factor
    :param lam: lambda parameter
    :return: (B, H) lambda-returns
    """
    B, H = rewards.shape
    lam_returns = torch.zeros_like(rewards)
    next_ret = values_with_bootstrap[:, -1]  # (B,)
    for t in reversed(range(H)):
        ret = rewards[:, t] + gamma * (1.0 - done[:, t]) * (
                (1.0 - lam) * values_with_bootstrap[:, t + 1] + lam * next_ret
        )
        lam_returns[:, t] = ret
        next_ret = ret
    return lam_returns
