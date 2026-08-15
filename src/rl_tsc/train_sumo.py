from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict

import numpy as np
import torch

from .config import TrainConfig
from .ppo import PPOAgent, Trajectory
from .sumo_adapter import SumoAdapterConfig, SumoTraciAdapterEnv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO controller directly on a SUMO scenario via TraCI")
    p.add_argument("--sumocfg", type=str, required=True, help="Path to SUMO .sumocfg file")
    p.add_argument("--tls-id", type=str, default="", help="Traffic light ID to control (default: first TLS)")
    p.add_argument("--sumo-binary", type=str, default="sumo", help="sumo or sumo-gui")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--steps-per-episode", type=int, default=600)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="outputs")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.exists(args.sumocfg):
        raise FileNotFoundError(f"SUMO config not found: {args.sumocfg}")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_cfg = TrainConfig(
        seed=args.seed,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        curriculum="off",
    )

    sumo_cfg = SumoAdapterConfig(
        sumo_cfg_path=args.sumocfg,
        tls_id=args.tls_id or None,
        sumo_binary=args.sumo_binary,
    )

    env = SumoTraciAdapterEnv(sumo_cfg)
    if not env.available:
        raise RuntimeError(
            "TraCI module is not available. Ensure SUMO tools are on PYTHONPATH, e.g. "
            "PYTHONPATH=src:$HOME/sumo/share/sumo/tools"
        )

    agent = PPOAgent(env.state_dim, env.action_dim, train_cfg, device=args.device)

    metrics_path = os.path.join(args.out_dir, "metrics_sumo.csv")
    summary_path = os.path.join(args.out_dir, "training_summary_sumo.json")

    with open(metrics_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "episode",
                "episode_reward",
                "mean_queue",
                "max_wait",
                "emergency_active_steps",
                "actor_loss",
                "critic_loss",
                "entropy",
            ],
        )
        writer.writeheader()

        last_update_stats = {}

        try:
            for ep in range(train_cfg.episodes):
                state = env.reset()

                states = []
                actions = []
                log_probs = []
                rewards = []
                values = []

                ep_reward = 0.0
                queue_sum_acc = 0.0
                max_wait_seen = 0.0
                emergency_steps = 0

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
                    emergency_steps += int(info["emergency_active"])

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
                        "episode_reward": round(ep_reward, 4),
                        "mean_queue": round(queue_sum_acc / float(train_cfg.steps_per_episode), 4),
                        "max_wait": round(max_wait_seen, 4),
                        "emergency_active_steps": emergency_steps,
                        "actor_loss": round(last_update_stats["actor_loss"], 6),
                        "critic_loss": round(last_update_stats["critic_loss"], 6),
                        "entropy": round(last_update_stats["entropy"], 6),
                    }
                )

                if (ep + 1) % 5 == 0:
                    print(
                        f"Episode {ep + 1}/{train_cfg.episodes} | reward={ep_reward:.2f} | "
                        f"mean_queue={queue_sum_acc / train_cfg.steps_per_episode:.2f} | "
                        f"entropy={last_update_stats['entropy']:.3f}"
                    )
        finally:
            env.close()

    summary = {
        "train_config": asdict(train_cfg),
        "sumo_config": asdict(sumo_cfg),
        "last_update_stats": last_update_stats,
        "outputs": {
            "metrics_csv": metrics_path,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("SUMO training complete.")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
