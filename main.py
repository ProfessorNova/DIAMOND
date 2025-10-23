import ale_py
import torch
from gymnasium import register_envs

from lib.actor import Actor
from lib.config import Config
from lib.critic import Critic
from lib.diffusion_model import DiffusionModel
from lib.replay_buffer import ReplayBuffer
from lib.reward_end_model import RewardEndModel
from lib.training import training_loop
from lib.utils import make_env


def main():
    cfg = Config()

    register_envs(ale_py)
    env = make_env(cfg)
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    replay_buffer = ReplayBuffer(
        capacity=cfg.number_of_epochs * cfg.environment_steps_per_epoch,
        obs_shape=obs_shape,
        device=cfg.device,
    )

    diffusion_model = DiffusionModel(
        obs_channels=obs_shape[0],
        num_actions=num_actions,
        residual_blocks_layers=cfg.diffusion_model_residual_blocks_layers,
        residual_blocks_channels=cfg.diffusion_model_residual_blocks_channels,
        cond_dim=cfg.diffusion_model_residual_blocks_conditioning_dimensions,
    )
    diffusion_model.to(cfg.device)

    reward_end_model = RewardEndModel(
        obs_channels=obs_shape[0],
        num_actions=num_actions,
        residual_blocks_layers=cfg.reward_termination_model_residual_blocks_layers,
        residual_blocks_channels=cfg.reward_termination_model_residual_blocks_channels,
        cond_dim=cfg.reward_termination_model_residual_blocks_conditioning_dimensions,
        lstm_dim=cfg.reward_termination_model_lstm_dimension,
    )
    reward_end_model.to(cfg.device)

    actor = Actor(
        obs_channels=obs_shape[0],
        num_actions=num_actions,
        residual_blocks_layers=cfg.actor_critic_model_residual_blocks_layers,
        residual_blocks_channels=cfg.actor_critic_model_residual_blocks_channels,
        lstm_dim=cfg.actor_critic_model_lstm_dimension,
    )
    actor.to(cfg.device)

    critic = Critic(
        obs_channels=obs_shape[0],
        residual_blocks_layers=cfg.actor_critic_model_residual_blocks_layers,
        residual_blocks_channels=cfg.actor_critic_model_residual_blocks_channels,
        lstm_dim=cfg.actor_critic_model_lstm_dimension,
    )
    critic.to(cfg.device)

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

    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.actor_critic_model_weight_decay,
        eps=cfg.epsilon,
    )

    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.actor_critic_model_weight_decay,
        eps=cfg.epsilon,
    )

    training_loop(
        cfg=cfg,
        env=env,
        replay_buffer=replay_buffer,
        diffusion_model=diffusion_model,
        reward_end_model=reward_end_model,
        policy_network=actor,
        value_network=critic,
        diffusion_model_optimizer=diffusion_model_optimizer,
        reward_end_model_optimizer=reward_end_model_optimizer,
        policy_network_optimizer=actor_optimizer,
        value_network_optimizer=critic_optimizer,
    )


if __name__ == "__main__":
    main()
