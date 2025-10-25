import datetime

import ale_py
import torch
from gymnasium import register_envs
from tensorboardX import SummaryWriter

from lib.actor_critic import ActorCritic
from lib.config import Config
from lib.diffusion_model import DiffusionModel
from lib.replay_buffer import ReplayBuffer
from lib.reward_end_model import RewardEndModel
from lib.training import training_loop
from lib.utils import make_env


def main():
    cfg = Config()

    register_envs(ale_py)
    env = make_env(cfg.env_id, cfg.image_size, cfg.frame_skip)
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    replay_buffer = ReplayBuffer(
        capacity=cfg.number_of_epochs * cfg.environment_steps_per_epoch,
        obs_shape=obs_shape,
        device=cfg.device,
    )

    diffusion_model = DiffusionModel(
        obs_shape=obs_shape,
        num_actions=num_actions,
        num_conditioning_obs_and_acts=cfg.diffusion_model_number_of_conditioning_observations_and_actions,
        residual_blocks_layers=cfg.diffusion_model_residual_blocks_layers,
        residual_blocks_channels=cfg.diffusion_model_residual_blocks_channels,
        residual_blocks_conditioning_dimensions=cfg.diffusion_model_residual_blocks_conditioning_dimensions,
        num_denoising_steps=cfg.number_of_steps,
        sigma_data=cfg.sigma_data,
    )
    diffusion_model.to(cfg.device)
    diffusion_model = torch.compile(diffusion_model)
    diffusion_model.sample_next_observation = torch.compile(diffusion_model.sample_next_observation)

    reward_end_model = RewardEndModel(
        obs_shape=obs_shape,
        num_actions=num_actions,
        residual_blocks_layers=cfg.reward_termination_model_residual_blocks_layers,
        residual_blocks_channels=cfg.reward_termination_model_residual_blocks_channels,
        residual_blocks_conditioning_dimensions=cfg.reward_termination_model_residual_blocks_conditioning_dimensions,
        lstm_dimensions=cfg.reward_termination_model_lstm_dimension,
    )
    reward_end_model.to(cfg.device)
    reward_end_model = torch.compile(reward_end_model)

    actor_critic_network = ActorCritic(
        obs_shape=obs_shape,
        num_actions=num_actions,
        residual_blocks_layers=cfg.actor_critic_model_residual_blocks_layers,
        residual_blocks_channels=cfg.actor_critic_model_residual_blocks_channels,
        lstm_dimensions=cfg.actor_critic_model_lstm_dimension,
    )
    actor_critic_network.to(cfg.device)
    actor_critic_network = torch.compile(actor_critic_network)

    # Print model parameter counts nicely formatted
    diffusion_params = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    reward_end_params = sum(p.numel() for p in reward_end_model.parameters() if p.requires_grad)
    actor_critic_params = sum(p.numel() for p in actor_critic_network.parameters() if p.requires_grad)

    total_params = diffusion_params + reward_end_params + actor_critic_params
    print(f"{'Model':<20}{'Parameters':>15}{'Share':>10}")
    print("-" * 45)
    print(f"{'Diffusion':<20}{diffusion_params:15,d}{diffusion_params / total_params:10.2%}")
    print(f"{'Reward/End':<20}{reward_end_params:15,d}{reward_end_params / total_params:10.2%}")
    print(f"{'Actor-Critic':<20}{actor_critic_params:15,d}{actor_critic_params / total_params:10.2%}")
    print("-" * 45)
    print(f"{'Total':<20}{total_params:15,d}")

    diffusion_model_optimizer = torch.optim.AdamW(
        diffusion_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.diffusion_model_weight_decay,
        eps=cfg.epsilon,
    )

    reward_end_model_optimizer = torch.optim.AdamW(
        reward_end_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.reward_termination_model_weight_decay,
        eps=cfg.epsilon,
    )

    actor_critic_optimizer = torch.optim.AdamW(
        actor_critic_network.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.actor_critic_model_weight_decay,
        eps=cfg.epsilon,
    )

    if cfg.create_artifacts:
        run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter(log_dir=f"runs/{run_name}")
    else:
        writer = None

    training_loop(
        cfg=cfg,
        env=env,
        replay_buffer=replay_buffer,
        diffusion_model=diffusion_model,
        reward_end_model=reward_end_model,
        actor_critic_network=actor_critic_network,
        diffusion_model_optimizer=diffusion_model_optimizer,
        reward_end_model_optimizer=reward_end_model_optimizer,
        actor_critic_network_optimizer=actor_critic_optimizer,
        writer=writer,
    )


if __name__ == "__main__":
    main()
