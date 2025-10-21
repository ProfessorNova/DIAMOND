import gymnasium as gym
import torch

from lib.config import Config


def make_env(cfg: Config) -> gym.Env:
    raise NotImplementedError


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
