from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Config:
    # --- General Settings ---
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_id: str = "ALE/BeamRider-v5"
    image_size: Tuple[int, int] = (64, 64)
    create_artifacts: bool = True
    log_interval: int = 10  # in training steps
    video_log_interval: int = 5  # in epochs

    # --- Hyperparameters ---
    # Training loop
    number_of_epochs: int = 1000
    training_steps_per_epoch: int = 4000
    batch_size: int = 32
    environment_steps_per_epoch: int = 1000
    epsilon_greedy_for_collection: float = 0.01

    # RL hyperparameters
    imagination_horizon: int = 15
    discount_factor: float = 0.985
    entropy_weight: float = 0.001
    lambda_returns_coefficient: float = 0.95

    # Sequence construction during training
    diffusion_model_number_of_conditioning_observations_and_actions: int = 4
    reward_termination_model_burn_in_length: int = diffusion_model_number_of_conditioning_observations_and_actions
    actor_critic_model_burn_in_length: int = diffusion_model_number_of_conditioning_observations_and_actions

    # Optimization
    learning_rate: float = 1e-4
    epsilon: float = 1e-8
    diffusion_model_weight_decay: float = 1e-2
    reward_termination_model_weight_decay: float = 1e-2
    actor_critic_model_weight_decay: float = 0.0

    # Diffusion Sampling
    number_of_steps: int = 3
    P_mean: float = -0.4
    P_std: float = 1.2
    sigma_data: float = 0.5

    # --- Architecture details ---
    # Diffusion Model
    diffusion_model_residual_blocks_layers = [2, 2, 2, 2]
    diffusion_model_residual_blocks_channels = [64, 64, 64, 64]
    diffusion_model_residual_blocks_conditioning_dimensions = 256

    # Reward/Termination Model
    reward_termination_model_residual_blocks_layers = [2, 2, 2, 2]
    reward_termination_model_residual_blocks_channels = [32, 32, 32, 32]
    reward_termination_model_residual_blocks_conditioning_dimensions = 128
    reward_termination_model_lstm_dimension = 512

    # Actor-Critic Model
    actor_critic_model_residual_blocks_layers = [1, 1, 1, 1]
    actor_critic_model_residual_blocks_channels = [32, 32, 64, 64]
    actor_critic_model_lstm_dimension = 512
