from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict
from typing import List

import numpy as np
import torch

from .config import EnvConfig, TrainConfig
from .env import MixedTrafficSignalEnv, curriculum_schedule
from .explain import summarize_attributions
from .ppo import PPOAgent, Trajectory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


FEATURE_NAMES = [
    "q_eb_0",
    "q_eb_1",
    "q_eb_2",
    "q_sb_0",
    "q_sb_1",
    "q_sb_2",
    "phase",
    "phase_age",
    "av_ratio",
    "emergency_flag",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO traffic controller with curriculum + explainability")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--steps-per-episode", type=int, default=240)
    p.add_argument("--curriculum", type=str, default="linear", choices=["linear", "max", "off"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="outputs")
    return p


def main() -> None:
    args = build_parser().parse_args()

    train_cfg = TrainConfig(
        seed=args.seed,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        curriculum=args.curriculum,
    )
    env_cfg = EnvConfig()

    set_seed(train_cfg.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    env = MixedTrafficSignalEnv(env_cfg, seed=train_cfg.seed)
    agent = PPOAgent(env.state_dim, env.action_dim, train_cfg, device=args.device)

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    summary_path = os.path.join(args.out_dir, "training_summary.json")
    attr_path = os.path.join(args.out_dir, "attributions_latest.json")

    with open(metrics_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "episode",
                "av_ratio",
                "difficulty",
                "episode_reward",
                "mean_queue",
                "max_wait",
                "emergency_events",
                "emergency_served",
                "actor_loss",
                "critic_loss",
                "entropy",
            ],
        )
        writer.writeheader()

        sampled_states: List[np.ndarray] = []
        sampled_actions: List[int] = []

        last_update_stats = {}

        for ep in range(train_cfg.episodes):
            av_ratio, difficulty = curriculum_schedule(ep, train_cfg.episodes, mode=train_cfg.curriculum)
            env.set_curriculum(av_ratio=av_ratio, difficulty=difficulty)

            state = env.reset()

            states = []
            actions = []
            log_probs = []
            rewards = []
            values = []

            ep_reward = 0.0
            queue_sum_acc = 0.0
            max_wait_seen = 0.0
            emergency_events = 0
            emergency_served = 0

            for _ in range(train_cfg.steps_per_episode):
                action, log_prob, value = agent.select_action(state)
                next_state, reward, _, info = env.step(action)

                states.append(state.copy())
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)

                ep_reward += reward
                queue_sum_acc += info["queue_sum"]
                max_wait_seen = max(max_wait_seen, info["max_wait"])
                emergency_events += int(info["emergency_active"])
                emergency_served += int(info["emergency_served"])

                if len(sampled_states) < 256 and (ep % 4 == 0):
                    sampled_states.append(state.copy())
                    sampled_actions.append(action)

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

            last_update_stats = agent.update(traj)

            writer.writerow(
                {
                    "episode": ep,
                    "av_ratio": round(av_ratio, 4),
                    "difficulty": round(difficulty, 4),
                    "episode_reward": round(ep_reward, 4),
                    "mean_queue": round(queue_sum_acc / float(train_cfg.steps_per_episode), 4),
                    "max_wait": round(max_wait_seen, 4),
                    "emergency_events": emergency_events,
                    "emergency_served": emergency_served,
                    "actor_loss": round(last_update_stats["actor_loss"], 6),
                    "critic_loss": round(last_update_stats["critic_loss"], 6),
                    "entropy": round(last_update_stats["entropy"], 6),
                }
            )

            if (ep + 1) % 20 == 0:
                print(
                    f"Episode {ep + 1}/{train_cfg.episodes} | reward={ep_reward:.2f} | "
                    f"mean_queue={queue_sum_acc / train_cfg.steps_per_episode:.2f} | "
                    f"entropy={last_update_stats['entropy']:.3f}"
                )

    attributions = summarize_attributions(
        actor=agent.actor,
        state_samples=sampled_states,
        actions=sampled_actions,
        feature_names=FEATURE_NAMES,
        device=args.device,
    )

    with open(attr_path, "w", encoding="utf-8") as fp:
        json.dump(attributions, fp, indent=2)

    summary = {
        "train_config": asdict(train_cfg),
        "env_config": asdict(env_cfg),
        "last_update_stats": last_update_stats,
        "outputs": {
            "metrics_csv": metrics_path,
            "attributions_json": attr_path,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("Training complete.")
    print(f"Metrics: {metrics_path}")
    print(f"Attributions: {attr_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
