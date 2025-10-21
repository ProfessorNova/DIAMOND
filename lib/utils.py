import gymnasium as gym

from lib.config import Config


def make_env(cfg: Config) -> gym.Env:
    raise NotImplementedError
