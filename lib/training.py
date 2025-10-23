import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from lib.actor import Actor
from lib.config import Config
from lib.critic import Critic
from lib.diffusion_model import DiffusionModel
from lib.replay_buffer import ReplayBuffer
from lib.reward_end_model import RewardEndModel
from lib.utils import lambda_returns


def training_loop(
        cfg: Config,
        env: gym.Env,
        replay_buffer: ReplayBuffer,
        diffusion_model: DiffusionModel,
        reward_end_model: RewardEndModel,
        policy_network: Actor,
        value_network: Critic,
        diffusion_model_optimizer: torch.optim.Optimizer,
        reward_end_model_optimizer: torch.optim.Optimizer,
        policy_network_optimizer: torch.optim.Optimizer,
        value_network_optimizer: torch.optim.Optimizer
) -> None:
    for epoch in range(cfg.number_of_epochs):
        collect_experience(cfg, env, policy_network, replay_buffer)

        for step_diffusion_model in range(cfg.training_steps_per_epoch):
            print(update_diffusion_model(cfg, replay_buffer, diffusion_model, diffusion_model_optimizer))

        for step_reward_end_model in range(cfg.training_steps_per_epoch):
            print(update_reward_end_model(cfg, replay_buffer, reward_end_model, reward_end_model_optimizer))

        for step_actor_critic in range(cfg.training_steps_per_epoch):
            print(update_actor_critic(cfg, replay_buffer, diffusion_model, reward_end_model, policy_network,
                                      value_network, policy_network_optimizer, value_network_optimizer))


@torch.no_grad()
def collect_experience(
        cfg: Config,
        env: gym.Env,
        policy_network: Actor,
        replay_buffer: ReplayBuffer
) -> None:
    current_obs, _ = env.reset()
    h_t, c_t = None, None
    for t in range(cfg.environment_steps_per_epoch):
        # Sample action from the current policy_network based on current_obs with epsilon-greedy exploration
        current_obs_tensor = torch.tensor(current_obs, device=cfg.device).float() / 255.0
        act, (h_t, c_t) = policy_network.sample_action(current_obs_tensor, (h_t, c_t))
        if np.random.rand() < cfg.epsilon_greedy_for_collection:
            act = env.action_space.sample()

        # Step the environment
        next_obs, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated

        # Store the transition in the replay buffer
        replay_buffer.store(current_obs, act, reward, terminated, truncated)

        # Prepare for the next step
        if done:
            current_obs, _ = env.reset()
            h_t, c_t = None, None
        else:
            current_obs = next_obs


def update_diffusion_model(
        cfg: Config,
        replay_buffer: ReplayBuffer,
        diffusion_model: DiffusionModel,
        diffusion_model_optimizer: torch.optim.Optimizer
) -> float:
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
    sigma = log_sigma.exp().view(B, 1, 1, 1)  # (B, 1, 1, 1)

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
        c_noise,  # (B, 1, 1, 1)
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

    return loss.item()


def update_reward_end_model(
        cfg: Config,
        replay_buffer: ReplayBuffer,
        reward_end_model: RewardEndModel,
        reward_end_model_optimizer: torch.optim.Optimizer
) -> float:
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

    return loss.item()


def update_actor_critic(
        cfg: Config,
        replay_buffer: ReplayBuffer,
        diffusion_model: DiffusionModel,
        reward_end_model: RewardEndModel,
        policy_network: Actor,
        value_network: Critic,
        policy_network_optimizer: torch.optim.Optimizer,
        value_network_optimizer: torch.optim.Optimizer
) -> float:
    B = cfg.batch_size
    L = cfg.actor_critic_model_burn_in_length
    L_dm = cfg.diffusion_model_number_of_conditioning_observations_and_actions
    H = cfg.imagination_horizon

    # Sample initial buffer (x_{t-L+1}, a_{t-L+1}, ..., x_t, a_t)
    batch = replay_buffer.sample(B, L + 1, avoid_term_trunc=True)
    obs_burn = batch['observations'].float() / 255.0  # (B, L+1, C, H, W)
    acts_burn = batch['actions']

    # Burn-in buffer with reward end model, policy_network, and value_network to initialize LSTM states
    h_r, c_r, h_p, c_p, h_v, c_v = None, None, None, None, None, None
    with torch.no_grad():
        _, _, (h_r, c_r) = reward_end_model(obs_burn[:, :L], acts_burn[:, :L])
        _, (h_p, c_p) = policy_network(obs_burn[:, :L], acts_burn[:, :L])
        _, (h_v, c_v) = value_network(obs_burn[:, :L], acts_burn[:, :L])

    # Rolling variables for imagination
    obs_hist = obs_burn[:, -L_dm:].clone()  # (B, L_dm, C, H, W)
    act_hist = acts_burn[:, -L_dm:].clone()  # (B, L_dm)
    x_i = obs_burn[:, L].clone()  # (B, C, H, W)

    # Buffers to store imagined trajectories
    obs_buf = [x_i]  # Will be (H+1) observations
    log_prob_buf = []  # Will be H log probabilities
    ent_buf = []  # Will be H entropies
    reward_buf = []  # Will be H rewards
    done_buf = []  # Will be H done flags

    for i in range(H):
        # Sample action from the policy_network
        policy_logits, (h_p, c_p) = policy_network(x_i.unsqueeze(1), (h_p, c_p)).squeeze(1)  # (B, A)
        dist = torch.distributions.Categorical(logits=policy_logits)
        act_i = dist.sample()  # (B,)
        log_prob_i = dist.log_prob(act_i)  # (B,)
        ent_i = dist.entropy()  # (B,)
        log_prob_buf.append(log_prob_i)
        ent_buf.append(ent_i)

        with torch.no_grad():
            # Sample reward r_i and termination d_i from reward_end_model
            reward_logits, end_logits, (h_r, c_r) = reward_end_model(
                x_i.unsqueeze(1), act_i.unsqueeze(1), (h_r, c_r)
            )
            reward_logits = reward_logits.squeeze(1)  # (B, 3)
            reward_vals = torch.tensor([-1.0, 0.0, 1.0], device=x_i.device)
            reward_i = (F.softmax(reward_logits, dim=-1) * reward_vals).sum(-1)  # (B,)
            end_logits = end_logits.squeeze(1)  # (B, 2)
            done_i = F.softmax(end_logits, dim=-1)[..., 1]  # (B,)
            reward_buf.append(reward_i)
            done_buf.append(done_i)

        # Update observation and action history to condition diffusion model with latest (x_i, a_i)
        obs_hist = torch.cat([obs_hist[:, 1:], x_i.unsqueeze(1)], dim=1)  # (B, L_dm, C, H, W)
        act_hist = torch.cat([act_hist[:, 1:], act_i.unsqueeze(1)], dim=1)  # (B, L_dm)

        with torch.no_grad():
            # Sample next observation x_{i+1} by reverse diffusion process with diffusion_model
            x_ip1 = diffusion_model.sample_reverse_diffusion(obs_hist, act_hist)  # (B, C, H, W)

        # Prepare for next step
        obs_buf.append(x_ip1)
        x_i = x_ip1

    # Stack buffers
    obs_buf = torch.stack(obs_buf, dim=1)  # (B, H+1, C, H, W)
    log_prob_buf = torch.stack(log_prob_buf, dim=1)  # (B, H)
    ent_buf = torch.stack(ent_buf, dim=1)  # (B, H
    reward_buf = torch.stack(reward_buf, dim=1)  # (B, H)
    done_buf = torch.stack(done_buf, dim=1)  # (B, H

    # Compute value_network for all imagined observations
    values, _ = value_network(obs_buf, (h_v, c_v))  # (B, H+1, 1)
    values = values.squeeze(-1)  # (B, H+1)

    # Compute RL losses for policy_network and value_network
    returns = lambda_returns(reward_buf, done_buf, values, cfg.discount_factor, cfg.lambda_returns_coefficient)
    value_loss = F.mse_loss(values[:, :-1], returns.detach())
    advantage = (returns - values[:, :-1]).detach()
    policy_loss = -(log_prob_buf * advantage + cfg.entropy_weight * ent_buf).mean()

    # Update policy_network and value_network
    policy_network_optimizer.zero_grad()
    policy_loss.backward()
    policy_network_optimizer.step()
    value_network_optimizer.zero_grad()
    value_loss.backward()
    value_network_optimizer.step()

    return policy_loss.item() + value_loss.item()
