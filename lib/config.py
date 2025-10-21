from dataclasses import dataclass

import torch


@dataclass
class Config:
    # --- General Settings ---
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Hyperparameters ---
    # Training loop
    number_of_epochs: int = 1000
    training_steps_per_epoch: int = 400
    batch_size: int = 32
    environment_steps_per_epoch: int = 100
    epsilon_greedy_for_collection: float = 0.01

    # RL hyperparameters
    imagination_horizon: int = 15
    discount_factor: float = 0.985
    entropy_weight: float = 0.001
    lambda_returns_coefficient: float = 0.95

    # Sequence construction during training
    diffusion_model_number_of_conditioning_observations_and_actions: int = 4
    reward_termination_model_burn_in_length: int = diffusion_model_number_of_conditioning_observations_and_actions
    reward_termination_model_training_sequence_length: int = reward_termination_model_burn_in_length + imagination_horizon
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
