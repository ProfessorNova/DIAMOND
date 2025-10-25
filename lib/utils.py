from typing import Tuple, List

import gymnasium as gym
import numpy as np
import torch
from tensorboardX import SummaryWriter


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


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        assert skip >= 1
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        obs = None
        done = False
        truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info


def make_env(env_id, image_size: Tuple[int, int] = (64, 64), frame_skip: int = 4) -> gym.Env:
    env = gym.make(env_id)
    env = FrameSkip(env, skip=frame_skip)
    env = gym.wrappers.ResizeObservation(env, image_size)
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


def _obs_to_frame_uint8(obs: np.ndarray) -> np.ndarray:
    """
    Convert an observation to a HxWxC uint8 frame.
    Accepts CHW (C,H,W) or HxW[xC]. Values can already be uint8 or 0..1/0..255.
    """
    if obs is None:
        raise ValueError("obs is None; cannot convert to frame")

    frame: np.ndarray
    if obs.ndim == 3:
        # CHW or HWC
        if obs.shape[0] in (1, 3):  # CHW -> HWC
            frame = np.moveaxis(obs, 0, -1)
        else:  # already HWC
            frame = obs
    elif obs.ndim == 2:
        frame = obs[..., None]  # HxW -> HxWx1
    else:
        raise ValueError(f"Unsupported obs shape {obs.shape}")

    # normalize to uint8
    if frame.dtype != np.uint8:
        # assume 0..1 or 0..255
        if frame.max() <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    return frame


def _frames_to_tb_video(frames_uint8: np.ndarray) -> torch.Tensor:
    """
    frames_uint8: (T, H, W, C) uint8
    returns torch.FloatTensor of shape (1, T, C, H, W) in [0,1]
    suitable for SummaryWriter.add_video
    """
    assert frames_uint8.ndim == 4, "frames must be (T,H,W,C)"
    video = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2)  # (T,C,H,W)
    video = video.unsqueeze(0).float() / 255.0  # (1,T,C,H,W)
    return video


@torch.no_grad()
def log_env_rollout_video(
        writer: SummaryWriter,
        env,
        actor_critic_network,
        device: torch.device,
        global_step: int,
        tag: str = "eval/rollout",
        max_steps: int = 1000,
        greedy: bool = True,
        fps: int = 20,
) -> Tuple[float, int]:
    """
    Runs one evaluation episode with the current policy, logs video to TensorBoard.
    Returns (episode_return, episode_length).
    """
    obs, _ = env.reset()
    h, c = None, None
    frames: List[np.ndarray] = []
    ep_ret = 0.0
    ep_len = 0

    for _ in range(max_steps):
        # record current frame from obs
        frames.append(_obs_to_frame_uint8(obs))

        # policy forward
        obs_t = torch.tensor(obs, device=device).unsqueeze(0).unsqueeze(0).float() / 255.0  # (1,1,C,H,W)
        logits, _, (h, c) = actor_critic_network(obs_t, (h, c))
        logits = logits[:, -1]  # (1, A)
        if greedy:
            act = int(torch.argmax(logits, dim=-1).item())
        else:
            dist = torch.distributions.Categorical(logits=logits)
            act = int(dist.sample().item())

        # env step
        obs, reward, terminated, truncated, _ = env.step(act)
        ep_ret += float(reward)
        ep_len += 1
        if terminated or truncated:
            break

    # push last obs frame
    frames.append(_obs_to_frame_uint8(obs))
    frames_np = np.stack(frames, axis=0)  # (T,H,W,C)

    video = _frames_to_tb_video(frames_np)
    writer.add_video(tag, video, global_step=global_step, fps=fps)
    writer.add_scalar("eval/episode_return", ep_ret, global_step)
    writer.add_scalar("eval/episode_length", ep_len, global_step)
    return ep_ret, ep_len


@torch.no_grad()
def log_imagined_trajectories_video(
        cfg,
        writer: SummaryWriter,
        diffusion_model,
        reward_end_model,
        actor_critic_network,
        replay_buffer,
        global_step: int,
        tag: str = "imagine/rollout",
        horizon: int = 200,
        fps: int = 20,
):
    """
    Builds an imagined rollout (H steps) starting from a random burn-in sequence in the buffer.
    Logs the imagined frames (first item of batch) to TensorBoard as video.
    """
    B = 1
    L = cfg.actor_critic_model_burn_in_length
    L_dm = cfg.diffusion_model_number_of_conditioning_observations_and_actions
    H = horizon

    # Sample (burn-in + 1) sequence
    batch = replay_buffer.sample(B, L + 1, avoid_term_trunc=True)
    obs_burn = (batch["observations"].float() / 255.0).to(cfg.device)  # (1,L+1,C,H,W)
    acts_burn = batch["actions"].to(cfg.device)  # (1,L+1)

    # Init states
    _, _, (h_r, c_r) = reward_end_model(obs_burn[:, :L], acts_burn[:, :L])
    _, _, (h_ac, c_ac) = actor_critic_network(obs_burn[:, :L])

    # Histories
    obs_hist = obs_burn[:, -L_dm:].clone()
    act_hist = acts_burn[:, -L_dm:].clone()
    x_i = obs_burn[:, L].clone()  # (1,C,H,W)

    frames: List[np.ndarray] = [_obs_to_frame_uint8((x_i[0].cpu().numpy() * 255.0).astype(np.uint8))]
    # push initial frame (denormalize)

    for _ in range(H):
        # sample action from current policy
        policy_logits, _, (h_ac, c_ac) = actor_critic_network(x_i.unsqueeze(1), (h_ac, c_ac))
        policy_logits = policy_logits[:, -1]
        dist = torch.distributions.Categorical(logits=policy_logits)
        act_i = dist.sample()  # (1,)

        # predict reward/done
        r_logits, d_logits, (h_r, c_r) = reward_end_model(x_i.unsqueeze(1), act_i.unsqueeze(1), (h_r, c_r))

        # roll diffusion model one step
        obs_hist = torch.cat([obs_hist[:, 1:], x_i.unsqueeze(1)], dim=1)
        act_hist = torch.cat([act_hist[:, 1:], act_i.unsqueeze(1)], dim=1)
        x_i = diffusion_model.sample_next_observation(obs_hist, act_hist)  # (1,C,H,W)

        frames.append(_obs_to_frame_uint8((x_i[0].cpu().numpy() * 255.0).astype(np.uint8)))

    frames_np = np.stack(frames, axis=0)
    video = _frames_to_tb_video(frames_np)
    writer.add_video(tag, video, global_step=global_step, fps=fps)
