from dataclasses import dataclass


@dataclass
class TrainConfig:
    seed: int = 42
    episodes: int = 160
    steps_per_episode: int = 240
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    update_epochs: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    hidden_size: int = 64
    max_grad_norm: float = 0.5
    curriculum: str = "linear"  # linear | max | off


@dataclass
class EnvConfig:
    min_green_steps: int = 8
    max_green_steps: int = 45
    queue_cap: int = 60
    emergency_prob: float = 0.02
    preemption_cap_steps: int = 12
    preemption_recovery_steps: int = 8
    base_arrival_rate: float = 0.9
    base_service_rate: float = 1.2
    queue_penalty_coef: float = 1.0
    switch_penalty: float = 0.15
    wait_penalty_coef: float = 0.10
    emergency_delay_penalty: float = 0.35
    emergency_served_bonus: float = 2.0
    reward_clip: float = 5.0
