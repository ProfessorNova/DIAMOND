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
        replay_buffer: ReplayBuffer,
        diffusion_model: Any,
        reward_end_model: Any,
        policy_network: Any,
        value_network: Any,
        diffusion_model_optimizer: torch.optim.Optimizer,
        reward_end_model_optimizer: torch.optim.Optimizer,
        policy_network_optimizer: torch.optim.Optimizer,
        value_network_optimizer: torch.optim.Optimizer
) -> None:
    for epoch in range(cfg.number_of_epochs):
        collect_experience(cfg, env, policy_network, replay_buffer)

        for step_diffusion_model in range(cfg.training_steps_per_epoch):
            update_diffusion_model(cfg, replay_buffer, diffusion_model, diffusion_model_optimizer)

        for step_reward_end_model in range(cfg.training_steps_per_epoch):
            update_reward_end_model(cfg, replay_buffer, reward_end_model, reward_end_model_optimizer)

        for step_actor_critic in range(cfg.training_steps_per_epoch):
            update_actor_critic(cfg, replay_buffer, reward_end_model, policy_network, value_network)


# TODO: Integrate policy_network into the function parameters
@torch.no_grad()
def collect_experience(
        cfg: Config,
        env: gym.Env,
        policy_network: Any,
        replay_buffer: ReplayBuffer
) -> None:
    current_obs, _ = env.reset()
    for t in range(cfg.environment_steps_per_epoch):
        # Sample action from the current policy_network based on current_obs with epsilon-greedy exploration
        if np.random.rand() < cfg.epsilon_greedy_for_collection:
            act = env.action_space.sample()
        else:
            act = policy_network.sample_action(current_obs)

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
    x_0 = obs[:, L]  # (B, C, H, W)

    # Log-normal sigma distribution from EDM
    log_sigma = torch.randn(B, device=cfg.device) * cfg.P_std + cfg.P_mean  # (B,)
    sigma = log_sigma.exp()  # (B,)

    # Default identity schedule from EDM
    c_in = 1.0 / torch.sqrt(sigma ** 2 + cfg.sigma_data ** 2)
    c_out = (sigma * cfg.sigma_data) / torch.sqrt(sigma ** 2 + cfg.sigma_data ** 2)
    c_noise = 0.25 * torch.log(sigma)
    c_skip = (cfg.sigma_data ** 2) / (cfg.sigma_data ** 2 + sigma ** 2)

    # Add independent Gaussian noise
    eps = torch.randn_like(x_0)  # (B, C, H, W)
    x_tau = x_0 + sigma * eps  # (B, C, H, W)

    # Compute the prediction using the diffusion model
    prediction = diffusion_model(
        c_in * x_tau,  # (B, C, H, W)
        c_noise,  # (B,)
        obs_cond,  # (B, L, C, H, W)
        act_cond  # (B, L)
    )
    target = (x_0 - c_skip * x_tau) / c_out  # (B, C, H, W)

    # Compute reconstruction loss
    loss = F.mse_loss(prediction, target)

    # Update the diffusion model
    diffusion_model_optimizer.zero_grad()
    loss.backward()
    diffusion_model_optimizer.step()


# TODO: Integrate reward_end_model into the function parameters
def update_reward_end_model(
        cfg: Config,
        replay_buffer: ReplayBuffer,
        reward_end_model: Any,
        reward_end_model_optimizer: torch.optim.Optimizer
) -> None:
    B = cfg.batch_size
    L = cfg.reward_termination_model_burn_in_length
    H = cfg.imagination_horizon

    # Sample sequences from the replay buffer (burn-in + imagination horizon)
    batch = replay_buffer.sample(B, L + H)
    obs = batch['observations'].float() / 255.0  # (B, L+H, C, H, W)
    acts = batch['actions']  # (B, L+H)
    rewards = batch['rewards']  # (B, L+H)
    done = (batch['terminated'] | batch['truncated']).float()  # (B, L+H)

    # LSTM hidden and cell states default to zeros if not provided
    r_logits, d_logits, _ = reward_end_model(obs, acts)  # (B, L+H, 3), (B, L+H, 2)

    # Cross-entropy loss
    # Targets: sign of rewards for reward prediction {0, 1, 2},
    # done flags for termination prediction {0, 1}
    r_targets = torch.sign(rewards).clamp(-1, 1).long() + 1  # Map {-1,0,1} to {0,1,2}
    d_targets = done.long()  # {0,1}
    reward_loss = F.cross_entropy(
        r_logits[:, L:, :].reshape(-1, 3), r_targets[:, L:].reshape(-1)
    )
    end_loss = F.cross_entropy(
        d_logits[:, L:, :].reshape(-1, 2), d_targets[:, L:].reshape(-1)
    )
    loss = reward_loss + end_loss

    # Update the reward end model
    reward_end_model_optimizer.zero_grad()
    loss.backward()
    reward_end_model_optimizer.step()


def update_actor_critic(
        cfg: Config,
        replay_buffer: ReplayBuffer,
        reward_end_model: Any,
        policy_network: Any,
        value_network: Any,
) -> None:
    B = cfg.batch_size
    L = cfg.actor_critic_model_burn_in_length
    H = cfg.imagination_horizon

    # Sample initial buffer
    batch = replay_buffer.sample(B, L + 1, avoid_term_trunc=True)
    obs_burn = batch['observations'].float() / 255.0  # (B, L+1, C, H, W)
    acts_burn = batch['actions']

    # Burn-in buffer with reward end model, policy_network, and value_network to initialize LSTM states
    with torch.no_grad():
        _, _, (h_r, c_r) = reward_end_model(obs_burn, acts_burn)
        _, (h_p, c_p) = policy_network(obs_burn, acts_burn)
        _, (h_v, c_v) = value_network(obs_burn, acts_burn)

    raise NotImplementedError
