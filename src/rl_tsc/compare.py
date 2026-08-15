from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict
from statistics import mean, stdev
from typing import Dict, List

from .baselines import EvalSummary, run_fixed_time, run_legacy_ql, run_ppo
from .config import EnvConfig, TrainConfig


def ci95(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare Fixed-time vs Legacy QL vs PPO and export professor-ready table")
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--train-episodes", type=int, default=160)
    p.add_argument("--eval-episodes", type=int, default=12)
    p.add_argument("--steps-per-episode", type=int, default=180)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="outputs")
    return p


def summarize(rows: List[EvalSummary]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        grouped[r.method]["mean_reward"].append(r.mean_reward)
        grouped[r.method]["mean_queue"].append(r.mean_queue)
        grouped[r.method]["mean_max_wait"].append(r.mean_max_wait)
        grouped[r.method]["emergency_served_rate"].append(r.emergency_served_rate)

    summary: Dict[str, Dict[str, float]] = {}
    for method, metrics in grouped.items():
        summary[method] = {
            "reward_mean": mean(metrics["mean_reward"]),
            "reward_ci95": ci95(metrics["mean_reward"]),
            "queue_mean": mean(metrics["mean_queue"]),
            "queue_ci95": ci95(metrics["mean_queue"]),
            "max_wait_mean": mean(metrics["mean_max_wait"]),
            "max_wait_ci95": ci95(metrics["mean_max_wait"]),
            "emergency_served_mean": mean(metrics["emergency_served_rate"]),
            "emergency_served_ci95": ci95(metrics["emergency_served_rate"]),
            "n_seeds": len(metrics["mean_reward"]),
        }

    return summary


def render_professor_table(summary: Dict[str, Dict[str, float]]) -> str:
    methods = ["fixed_time", "legacy_ql", "ppo"]

    lines = []
    lines.append("# Professor-Ready Results Table (Auto-Filled)")
    lines.append("")
    lines.append("| Method | Mean Reward (95% CI) | Mean Queue (95% CI) | Max Wait (95% CI) | Emergency Served Rate (95% CI) |")
    lines.append("|---|---:|---:|---:|---:|")

    for m in methods:
        s = summary.get(m)
        if s is None:
            lines.append(f"| {m} | n/a | n/a | n/a | n/a |")
            continue

        lines.append(
            "| {m} | {r:.3f} +/- {rci:.3f} | {q:.3f} +/- {qci:.3f} | {w:.3f} +/- {wci:.3f} | {e:.3f} +/- {eci:.3f} |".format(
                m=m,
                r=s["reward_mean"],
                rci=s["reward_ci95"],
                q=s["queue_mean"],
                qci=s["queue_ci95"],
                w=s["max_wait_mean"],
                wci=s["max_wait_ci95"],
                e=s["emergency_served_mean"],
                eci=s["emergency_served_ci95"],
            )
        )

    if "fixed_time" in summary and "ppo" in summary:
        base = summary["fixed_time"]
        ppo = summary["ppo"]

        queue_gain = (base["queue_mean"] - ppo["queue_mean"]) / max(1e-8, base["queue_mean"]) * 100.0
        wait_gain = (base["max_wait_mean"] - ppo["max_wait_mean"]) / max(1e-8, base["max_wait_mean"]) * 100.0

        lines.append("")
        lines.append("## Headline Delta vs Fixed-Time")
        lines.append(f"- PPO queue reduction: {queue_gain:.2f}%")
        lines.append(f"- PPO max-wait reduction: {wait_gain:.2f}%")

    lines.append("")
    lines.append("## Notes")
    lines.append("- Lower queue and max-wait are better.")
    lines.append("- Higher emergency served rate is better.")
    lines.append("- Use this table directly in meeting slides and replace with SUMO-linked runs next.")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    env_cfg = EnvConfig()
    train_cfg = TrainConfig(
        episodes=args.train_episodes,
        steps_per_episode=args.steps_per_episode,
        curriculum="linear",
    )

    rows: List[EvalSummary] = []

    for seed in seeds:
        rows.append(run_fixed_time(seed, env_cfg, episodes=args.eval_episodes, steps_per_episode=args.steps_per_episode))
        rows.append(
            run_legacy_ql(
                seed,
                env_cfg,
                train_episodes=max(30, args.train_episodes // 2),
                eval_episodes=args.eval_episodes,
                steps_per_episode=args.steps_per_episode,
            )
        )
        rows.append(run_ppo(seed, env_cfg, train_cfg, eval_episodes=args.eval_episodes, device=args.device))

    raw_csv = os.path.join(args.out_dir, "comparison_raw.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary = summarize(rows)
    summary_json = os.path.join(args.out_dir, "comparison_summary.json")
    with open(summary_json, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    table_md = os.path.join(args.out_dir, "professor_results_table.md")
    with open(table_md, "w", encoding="utf-8") as fp:
        fp.write(render_professor_table(summary))

    print("Comparison complete.")
    print(f"Raw rows: {raw_csv}")
    print(f"Summary: {summary_json}")
    print(f"Professor table: {table_md}")


if __name__ == "__main__":
    main()
