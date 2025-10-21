from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from lib.config import Config
from lib.replay_buffer import ReplayBuffer


def training_loop(
        cfg: Config,
        env: gym.Env,
        policy: Any,
        replay_buffer: ReplayBuffer,
        diffusion_model: Any,
) -> None:
    for epoch in range(cfg.number_of_epochs):
        collect_experience(cfg, env, policy, replay_buffer)

        for step_diffusion_model in range(cfg.training_steps_per_epoch):
            update_diffusion_model(cfg, replay_buffer, diffusion_model)

        for step_reward_end_model in range(cfg.training_steps_per_epoch):
            update_reward_end_model(cfg)

        for step_actor_critic in range(cfg.training_steps_per_epoch):
            update_actor_critic(cfg)


# TODO: Integrate policy into the function parameters
def collect_experience(
        cfg: Config,
        env: gym.Env,
        policy: Any,
        replay_buffer: ReplayBuffer
) -> None:
    current_obs, _ = env.reset()
    for t in range(cfg.environment_steps_per_epoch):
        # Sample action from the current policy based on current_obs with epsilon-greedy exploration
        if np.random.rand() < cfg.epsilon_greedy_for_collection:
            act = env.action_space.sample()
        else:
            act = policy.sample_action(current_obs)

        # Step the environment
        next_obs, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated

        # Store the transition in the replay buffer
        replay_buffer.store(current_obs, act, reward, terminated, truncated)

        # Prepare for the next step
        if done:
            current_obs, _ = env.reset()
        else:
            current_obs = next_obs


# TODO: Integrate diffusion_model into the function parameters
def update_diffusion_model(
        cfg: Config,
        replay_buffer: ReplayBuffer,
        diffusion_model: Any,
        diffusion_model_optimizer: torch.optim.Optimizer
) -> None:
    B = cfg.batch_size
    L = cfg.diffusion_model_number_of_conditioning_observations_and_actions

    # Sample sequences from the replay buffer
    batch = replay_buffer.sample(B, L + 1, avoid_term_trunc=True)  # +1 for the target observation
    obs = batch['observations'].float() / 255.0  # (B, L+1, C, H, W)
    act = batch['actions']  # (B, L+1)

    obs_cond = obs[:, :L]  # (B, L, C, H, W)
    act_cond = act[:, :L]  # (B, L)
    clean_obs_target = obs[:, L]  # (B, C, H, W)

    # Log-normal sigma distribution from EDM
    log_sigma = torch.randn(B, device=cfg.device) * cfg.P_std + cfg.P_mean  # (B,)
    sigma = log_sigma.exp().view(-1, 1, 1, 1)  # (B,1,1,1)

    # Default identity schedule from EDM
    c_in = 1.0 / torch.sqrt(sigma ** 2 + cfg.sigma_data ** 2)  # (B,1,1,1)
    c_out = (sigma * cfg.sigma_data) / torch.sqrt(sigma ** 2 + cfg.sigma_data ** 2)
    c_skip = (cfg.sigma_data ** 2) / (cfg.sigma_data ** 2 + sigma ** 2)
    c_noise = 0.25 * torch.log(torch.clamp(sigma, min=1e-8))  # (B,1,1,1)

    # Add independent Gaussian noise
    eps = torch.randn_like(clean_obs_target)  # (B, C, H, W)
    noised_obs_target = clean_obs_target + sigma * eps  # (B, C, H, W)

    # Compute the prediction using the diffusion model
    pred_residual = diffusion_model(
        c_in * noised_obs_target,  # (B, C, H, W)
        c_noise,  # (B, 1, 1, 1)
        obs_cond,  # (B, L, C, H, W)
        act_cond  # (B, L)
    )
    pred_obs_target = c_skip * noised_obs_target + c_out * pred_residual  # (B, C, H, W)

    # Compute reconstruction loss
    loss = F.mse_loss(pred_obs_target, clean_obs_target)

    # Update the diffusion model
    diffusion_model_optimizer.zero_grad()
    loss.backward()
    diffusion_model_optimizer.step()


def update_reward_end_model(cfg: Config) -> None:
    raise NotImplementedError


def update_actor_critic(cfg: Config) -> None:
    raise NotImplementedError
