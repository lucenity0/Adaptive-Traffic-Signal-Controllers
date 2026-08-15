from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from .config import EnvConfig, TrainConfig
from .env import MixedTrafficSignalEnv
from .ppo import PPOAgent, Trajectory


@dataclass
class EvalSummary:
    method: str
    seed: int
    mean_reward: float
    mean_queue: float
    mean_max_wait: float
    emergency_served_rate: float


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _rollout_with_policy(env: MixedTrafficSignalEnv, steps: int, policy_fn) -> Dict[str, float]:
    state = env.reset()

    total_reward = 0.0
    queue_sum = 0.0
    max_wait_sum = 0.0
    emergency_event_count = 0.0
    emergency_served_count = 0.0

    for _ in range(steps):
        action = policy_fn(state, env)
        state, reward, _, info = env.step(action)

        total_reward += reward
        queue_sum += info["queue_sum"]
        max_wait_sum += info["max_wait"]
        emergency_event_count += float(info.get("emergency_event_started", 0.0))
        emergency_served_count += info["emergency_served"]

    served_rate = emergency_served_count / max(1.0, emergency_event_count)

    return {
        "mean_reward": total_reward / float(steps),
        "mean_queue": queue_sum / float(steps),
        "mean_max_wait": max_wait_sum / float(steps),
        "emergency_served_rate": served_rate,
    }


def run_fixed_time(seed: int, env_cfg: EnvConfig, episodes: int, steps_per_episode: int, hold_steps: int = 20) -> EvalSummary:
    _seed_all(seed)
    env = MixedTrafficSignalEnv(env_cfg, seed=seed)
    env.set_curriculum(av_ratio=0.5, difficulty=0.6)

    def policy(_, e: MixedTrafficSignalEnv):
        return 1 if e.phase_age >= hold_steps else 0

    aggregate = {"mean_reward": 0.0, "mean_queue": 0.0, "mean_max_wait": 0.0, "emergency_served_rate": 0.0}
    for _ in range(episodes):
        stats = _rollout_with_policy(env, steps_per_episode, policy)
        for k in aggregate:
            aggregate[k] += stats[k]

    for k in aggregate:
        aggregate[k] /= float(episodes)

    return EvalSummary(
        method="fixed_time",
        seed=seed,
        mean_reward=aggregate["mean_reward"],
        mean_queue=aggregate["mean_queue"],
        mean_max_wait=aggregate["mean_max_wait"],
        emergency_served_rate=aggregate["emergency_served_rate"],
    )


def _discretize_state(state: np.ndarray) -> Tuple[int, int, int, int]:
    eb = float(np.sum(state[:3]))
    sb = float(np.sum(state[3:6]))

    eb_bin = min(4, int(math.floor(eb * 5)))
    sb_bin = min(4, int(math.floor(sb * 5)))
    phase = int(state[6] > 0.5)
    phase_age_bin = min(4, int(math.floor(state[7] * 5)))
    return eb_bin, sb_bin, phase, phase_age_bin


def run_legacy_ql(
    seed: int,
    env_cfg: EnvConfig,
    train_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    alpha: float = 0.15,
    gamma: float = 0.95,
) -> EvalSummary:
    _seed_all(seed)
    env = MixedTrafficSignalEnv(env_cfg, seed=seed)
    env.set_curriculum(av_ratio=0.5, difficulty=0.6)

    q_table: Dict[Tuple[int, int, int, int], np.ndarray] = {}

    eps_start = 0.4
    eps_end = 0.05

    def get_q(s_key):
        if s_key not in q_table:
            q_table[s_key] = np.zeros(2, dtype=np.float32)
        return q_table[s_key]

    for ep in range(train_episodes):
        state = env.reset()
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - ep / max(1, train_episodes - 1))

        for _ in range(steps_per_episode):
            s_key = _discretize_state(state)
            q_vals = get_q(s_key)

            if random.random() < eps:
                action = random.randint(0, 1)
            else:
                action = int(np.argmax(q_vals))

            next_state, reward, _, _ = env.step(action)
            ns_key = _discretize_state(next_state)

            td_target = reward + gamma * float(np.max(get_q(ns_key)))
            q_vals[action] = q_vals[action] + alpha * (td_target - q_vals[action])
            state = next_state

    def greedy_policy(state: np.ndarray, _env: MixedTrafficSignalEnv) -> int:
        s_key = _discretize_state(state)
        return int(np.argmax(get_q(s_key)))

    aggregate = {"mean_reward": 0.0, "mean_queue": 0.0, "mean_max_wait": 0.0, "emergency_served_rate": 0.0}
    for _ in range(eval_episodes):
        stats = _rollout_with_policy(env, steps_per_episode, greedy_policy)
        for k in aggregate:
            aggregate[k] += stats[k]

    for k in aggregate:
        aggregate[k] /= float(eval_episodes)

    return EvalSummary(
        method="legacy_ql",
        seed=seed,
        mean_reward=aggregate["mean_reward"],
        mean_queue=aggregate["mean_queue"],
        mean_max_wait=aggregate["mean_max_wait"],
        emergency_served_rate=aggregate["emergency_served_rate"],
    )


def run_ppo(
    seed: int,
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
    eval_episodes: int,
    device: str = "cpu",
) -> EvalSummary:
    _seed_all(seed)
    env = MixedTrafficSignalEnv(env_cfg, seed=seed)
    agent = PPOAgent(env.state_dim, env.action_dim, train_cfg, device=device)

    for ep in range(train_cfg.episodes):
        state = env.reset()
        env.set_curriculum(av_ratio=min(1.0, ep / max(1, train_cfg.episodes - 1)), difficulty=0.2 + 0.8 * min(1.0, ep / max(1, train_cfg.episodes - 1)))

        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []

        for _ in range(train_cfg.steps_per_episode):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, _, _ = env.step(action)
            states.append(state.copy())
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            state = next_state

        with torch.no_grad():
            bootstrap_value = float(
                agent.critic(torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0))
                .squeeze(-1)
                .item()
            )

        traj = Trajectory(
            states=torch.tensor(np.array(states), dtype=torch.float32),
            actions=torch.tensor(np.array(actions), dtype=torch.long),
            log_probs=torch.tensor(np.array(log_probs), dtype=torch.float32),
            rewards=torch.tensor(np.array(rewards), dtype=torch.float32),
            values=torch.tensor(np.array(values), dtype=torch.float32),
            last_value=torch.tensor(bootstrap_value, dtype=torch.float32),
        )
        agent.update(traj)

    env.set_curriculum(av_ratio=0.5, difficulty=0.6)

    def greedy_policy(state: np.ndarray, _env: MixedTrafficSignalEnv) -> int:
        return agent.select_action_greedy(state)

    aggregate = {"mean_reward": 0.0, "mean_queue": 0.0, "mean_max_wait": 0.0, "emergency_served_rate": 0.0}
    for _ in range(eval_episodes):
        stats = _rollout_with_policy(env, train_cfg.steps_per_episode, greedy_policy)
        for k in aggregate:
            aggregate[k] += stats[k]

    for k in aggregate:
        aggregate[k] /= float(eval_episodes)

    return EvalSummary(
        method="ppo",
        seed=seed,
        mean_reward=aggregate["mean_reward"],
        mean_queue=aggregate["mean_queue"],
        mean_max_wait=aggregate["mean_max_wait"],
        emergency_served_rate=aggregate["emergency_served_rate"],
    )
