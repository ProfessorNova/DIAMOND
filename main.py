import ale_py
import torch
from gymnasium import register_envs

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
    env = make_env(cfg)
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    replay_buffer = ReplayBuffer(
        capacity=cfg.number_of_epochs * cfg.environment_steps_per_epoch,
        obs_shape=obs_shape,
        device=cfg.device,
    )

    diffusion_model = DiffusionModel(
    )
    diffusion_model.to(cfg.device)

    reward_end_model = RewardEndModel(
    )
    reward_end_model.to(cfg.device)

    actor_critic_network = ActorCritic(
        obs_shape=obs_shape,
        num_actions=num_actions,
        residual_blocks_layers=cfg.actor_critic_model_residual_blocks_layers,
        residual_blocks_channels=cfg.actor_critic_model_residual_blocks_channels,
        lstm_dimensions=cfg.actor_critic_model_lstm_dimension,
    )
    actor_critic_network.to(cfg.device)

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
    )


if __name__ == "__main__":
    main()
