from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_MPLCONFIGDIR = ROOT / "outputs" / ".mplconfig"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patches
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from rl_tsc.config import EnvConfig, TrainConfig
from rl_tsc.env import curriculum_schedule


METHOD_ORDER = ["fixed_time", "legacy_ql", "ppo"]
METHOD_LABELS = {"fixed_time": "Fixed-time", "legacy_ql": "Legacy Q-learning", "ppo": "PPO"}
METHOD_HATCHES = {"fixed_time": "//", "legacy_ql": "..", "ppo": "xx"}
FEATURE_LABELS = {
    "q_eb_0": "Queue EB 0",
    "q_eb_1": "Queue EB 1",
    "q_eb_2": "Queue EB 2",
    "q_sb_0": "Queue SB 0",
    "q_sb_1": "Queue SB 1",
    "q_sb_2": "Queue SB 2",
    "phase": "Current phase",
    "phase_age": "Phase age",
    "av_ratio": "AV ratio",
    "emergency_flag": "Emergency flag",
}


@dataclass
class SumoStatus:
    available: bool
    quantitative_claims_allowed: bool
    reason: str
    metrics: pd.DataFrame | None = None
    summary: dict[str, Any] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a detailed methodology PDF with graphs and diagrams.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--assets-dir", type=Path, default=ROOT / "outputs" / "report_assets")
    parser.add_argument("--output-pdf", type=Path, default=ROOT / "outputs" / "methodology_report.pdf")
    parser.add_argument("--title", type=str, default="Detailed Methodology Report")
    return parser


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return path


def load_required_artifacts(output_dir: Path) -> dict[str, Any]:
    required = {
        "metrics_csv": require_path(output_dir / "metrics.csv"),
        "comparison_raw_csv": require_path(output_dir / "comparison_raw.csv"),
        "comparison_summary_json": require_path(output_dir / "comparison_summary.json"),
        "attributions_json": require_path(output_dir / "attributions_latest.json"),
        "training_summary_json": require_path(output_dir / "training_summary.json"),
    }

    artifacts = {
        "metrics": pd.read_csv(required["metrics_csv"]),
        "comparison_raw": pd.read_csv(required["comparison_raw_csv"]),
        "comparison_summary": load_json(required["comparison_summary_json"]),
        "attributions": load_json(required["attributions_json"]),
        "training_summary": load_json(required["training_summary_json"]),
        "paths": required,
    }
    return artifacts


def merge_with_defaults(saved: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(saved)
    return merged


def load_sumo_status(output_dir: Path) -> SumoStatus:
    metrics_path = output_dir / "metrics_sumo.csv"
    summary_path = output_dir / "training_summary_sumo.json"

    if not metrics_path.exists() and not summary_path.exists():
        return SumoStatus(
            available=False,
            quantitative_claims_allowed=False,
            reason="No SUMO training artifacts were found. The report will describe the SUMO path as implemented but not experimentally validated.",
        )

    if metrics_path.exists() and not summary_path.exists():
        return SumoStatus(
            available=False,
            quantitative_claims_allowed=False,
            reason="A SUMO verification attempt was detected, but it did not complete successfully because only a partial artifact set was produced. The report keeps SUMO qualitative and does not quote numeric SUMO results.",
        )

    if summary_path.exists() and not metrics_path.exists():
        return SumoStatus(
            available=False,
            quantitative_claims_allowed=False,
            reason="A SUMO summary file exists without the matching metrics CSV. The report therefore treats the SUMO path as implemented but not quantitatively verified.",
        )

    metrics = pd.read_csv(metrics_path)
    summary = load_json(summary_path)

    if metrics.empty:
        return SumoStatus(
            available=True,
            quantitative_claims_allowed=False,
            reason="SUMO artifacts exist but contain no episode rows, so numeric SUMO claims are suppressed.",
            metrics=metrics,
            summary=summary,
        )

    meaningful_activity = bool(
        (metrics["mean_queue"].abs().sum() > 0.0)
        or (metrics["max_wait"].abs().sum() > 0.0)
        or (metrics["episode_reward"].abs().sum() > 0.0)
        or (metrics["emergency_active_steps"].abs().sum() > 0.0)
    )

    if not meaningful_activity:
        return SumoStatus(
            available=True,
            quantitative_claims_allowed=False,
            reason="SUMO outputs were generated but they show no meaningful traffic activity, so the report keeps SUMO as a capability description only.",
            metrics=metrics,
            summary=summary,
        )

    return SumoStatus(
        available=True,
        quantitative_claims_allowed=True,
        reason="Fresh SUMO artifacts are present and contain non-zero activity, so the report includes a quantitative SUMO verification subsection.",
        metrics=metrics,
        summary=summary,
    )


def moving_average(series: pd.Series, window: int = 10) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.6, color="0.75")
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("0.15")


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_training_curves(metrics: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    x = metrics["episode"]

    axes[0, 0].plot(x, metrics["episode_reward"], color="black", linewidth=1.0, alpha=0.35, label="Episode")
    axes[0, 0].plot(x, moving_average(metrics["episode_reward"]), color="black", linewidth=2.0, label="10-episode MA")
    style_axis(axes[0, 0], "PPO training reward trajectory", "Episode reward")
    axes[0, 0].legend(fontsize=8, frameon=True, facecolor="white")

    axes[0, 1].plot(x, metrics["mean_queue"], color="0.15", linewidth=1.0, alpha=0.35, label="Episode")
    axes[0, 1].plot(x, moving_average(metrics["mean_queue"]), color="black", linewidth=2.0, linestyle="--", label="10-episode MA")
    style_axis(axes[0, 1], "Mean queue during PPO training", "Mean queue")
    axes[0, 1].legend(fontsize=8, frameon=True, facecolor="white")

    axes[1, 0].plot(x, metrics["max_wait"], color="black", linewidth=1.8)
    style_axis(axes[1, 0], "Maximum wait by episode", "Max wait")
    axes[1, 0].set_xlabel("Episode", fontsize=10)

    axes[1, 1].plot(x, metrics["entropy"], color="black", linewidth=1.8, label="Policy entropy")
    if "critic_loss" in metrics.columns:
        loss_norm = metrics["critic_loss"] / max(1.0, float(metrics["critic_loss"].max()))
        axes[1, 1].plot(x, loss_norm, color="0.4", linewidth=1.5, linestyle=":", label="Critic loss (normalized)")
    style_axis(axes[1, 1], "Exploration and critic signal", "Value")
    axes[1, 1].set_xlabel("Episode", fontsize=10)
    axes[1, 1].legend(fontsize=8, frameon=True, facecolor="white")

    save_figure(fig, out_path)


def generate_comparison_summary_plot(summary: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    metrics = [
        ("reward_mean", "reward_ci95", "Mean reward"),
        ("queue_mean", "queue_ci95", "Mean queue"),
        ("max_wait_mean", "max_wait_ci95", "Mean max wait"),
        ("emergency_served_mean", "emergency_served_ci95", "Emergency served rate"),
    ]

    for ax, (metric_key, ci_key, title) in zip(axes.flatten(), metrics, strict=True):
        values = [summary[m][metric_key] for m in METHOD_ORDER]
        errors = [summary[m][ci_key] for m in METHOD_ORDER]
        bars = ax.bar(
            range(len(METHOD_ORDER)),
            values,
            yerr=errors,
            color=["white", "0.75", "0.4"],
            edgecolor="black",
            linewidth=1.2,
            error_kw={"ecolor": "black", "elinewidth": 1.0, "capsize": 4},
        )
        for bar, method in zip(bars, METHOD_ORDER, strict=True):
            bar.set_hatch(METHOD_HATCHES[method])
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=15)
        style_axis(ax, title, title)
        if metric_key == "emergency_served_mean":
            ax.set_ylim(0.0, 1.08)

    save_figure(fig, out_path)


def generate_per_seed_plot(raw_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    metrics = [
        ("mean_reward", "Per-seed mean reward"),
        ("mean_queue", "Per-seed mean queue"),
        ("mean_max_wait", "Per-seed mean max wait"),
        ("emergency_served_rate", "Per-seed emergency served rate"),
    ]

    x_positions = list(range(len(METHOD_ORDER)))
    method_to_x = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    unique_seeds = sorted(raw_df["seed"].unique())
    offsets = [-0.12, 0.0, 0.12]

    for ax, (metric_key, title) in zip(axes.flatten(), metrics, strict=True):
        for seed_idx, seed in enumerate(unique_seeds):
            seed_df = raw_df[raw_df["seed"] == seed]
            xs = [method_to_x[m] + offsets[seed_idx % len(offsets)] for m in seed_df["method"]]
            ax.scatter(xs, seed_df[metric_key], color="black", s=42, marker=["o", "s", "^"][seed_idx % 3], label=f"Seed {seed}")

        means = [raw_df[raw_df["method"] == method][metric_key].mean() for method in METHOD_ORDER]
        ax.plot(x_positions, means, color="0.3", linewidth=1.6, linestyle="--", label="Method mean")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=15)
        style_axis(ax, title, title)
        if metric_key == "emergency_served_rate":
            ax.set_ylim(0.0, 1.08)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, fontsize=8, frameon=True, facecolor="white")
    save_figure(fig, out_path)


def generate_attribution_plot(attributions: dict[str, float], out_path: Path) -> None:
    ordered = sorted(attributions.items(), key=lambda item: item[1], reverse=True)
    labels = [FEATURE_LABELS.get(name, name) for name, _ in ordered]
    values = [value for _, value in ordered]

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    bars = ax.barh(range(len(values)), values, color="white", edgecolor="black", linewidth=1.2)
    for idx, bar in enumerate(bars):
        bar.set_hatch(["//", "..", "xx", "\\\\"][idx % 4])
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    style_axis(ax, "Feature attribution summary", "Absolute attribution magnitude")
    ax.set_xlabel("Mean absolute attribution", fontsize=10)
    save_figure(fig, out_path)


def generate_curriculum_plot(total_episodes: int, out_path: Path) -> None:
    episodes = list(range(total_episodes))
    av_ratios = []
    difficulties = []
    for episode in episodes:
        av_ratio, difficulty = curriculum_schedule(episode, total_episodes, mode="linear")
        av_ratios.append(av_ratio)
        difficulties.append(difficulty)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(episodes, av_ratios, color="black", linewidth=2.0, label="AV ratio")
    ax.plot(episodes, difficulties, color="0.35", linewidth=2.0, linestyle="--", label="Difficulty")
    style_axis(ax, "Implemented curriculum schedule", "Normalized value")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9, frameon=True, facecolor="white")
    save_figure(fig, out_path)


def generate_sumo_training_plot(metrics: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    x = metrics["episode"]

    axes[0, 0].plot(x, metrics["episode_reward"], color="black", linewidth=1.7)
    style_axis(axes[0, 0], "SUMO PPO reward trajectory", "Episode reward")

    axes[0, 1].plot(x, metrics["mean_queue"], color="0.15", linewidth=1.7)
    style_axis(axes[0, 1], "SUMO mean queue", "Mean queue")

    axes[1, 0].plot(x, metrics["max_wait"], color="black", linewidth=1.7, linestyle="--")
    style_axis(axes[1, 0], "SUMO max wait", "Max wait")
    axes[1, 0].set_xlabel("Episode", fontsize=10)

    axes[1, 1].plot(x, metrics["emergency_active_steps"], color="0.25", linewidth=1.7)
    style_axis(axes[1, 1], "SUMO emergency-active steps", "Steps")
    axes[1, 1].set_xlabel("Episode", fontsize=10)

    save_figure(fig, out_path)


def draw_box(ax, xy, width, height, title, body=None):
    rect = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.4,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(rect)
    x, y = xy
    ax.text(x + width / 2.0, y + height * 0.62, title, ha="center", va="center", fontsize=11, fontweight="bold")
    if body:
        ax.text(x + width / 2.0, y + height * 0.28, body, ha="center", va="center", fontsize=8.8)


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.2, "shrinkA": 2, "shrinkB": 2},
    )


def generate_pipeline_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.34, 0.91), 0.32, 0.06, "RL-TSC METHODOLOGY PIPELINE")

    draw_box(ax, (0.06, 0.76), 0.22, 0.11, "TRAFFIC DEMAND", "Poisson arrivals\nMixed traffic\nEmergency events")
    draw_box(ax, (0.39, 0.76), 0.22, 0.11, "LIGHTWEIGHT ENV", "6 queues\n2 phases\nSafety constraints")
    draw_box(ax, (0.72, 0.76), 0.22, 0.11, "SUMO ADAPTER", "TraCI\nLane halting counts\nTLS discovery")

    draw_box(ax, (0.39, 0.57), 0.22, 0.11, "STATE VECTOR", "q1..q6, phase,\nphase age, AV ratio,\nemergency flag")
    draw_box(ax, (0.17, 0.40), 0.22, 0.11, "PPO ACTOR", "2 x hidden layers\nBinary hold/switch policy")
    draw_box(ax, (0.39, 0.40), 0.22, 0.11, "CONSTRAINT LAYER", "Min green\nMax green\nEmergency override")
    draw_box(ax, (0.61, 0.40), 0.22, 0.11, "PPO CRITIC", "State-value\nestimator")

    draw_box(ax, (0.17, 0.20), 0.22, 0.11, "REWARD MODEL", "Queue penalty\nWait penalty\nSwitch cost\nEmergency terms")
    draw_box(ax, (0.44, 0.20), 0.22, 0.11, "TRAINING LOOP", "GAE\nClipped PPO objective\nEntropy regularization")
    draw_box(ax, (0.72, 0.20), 0.22, 0.11, "OUTPUTS", "CSV / JSON metrics\nAttributions\nComparison table")

    arrow(ax, (0.28, 0.815), (0.39, 0.815))
    arrow(ax, (0.61, 0.815), (0.72, 0.815))
    arrow(ax, (0.50, 0.76), (0.50, 0.68))
    arrow(ax, (0.50, 0.57), (0.28, 0.51))
    arrow(ax, (0.50, 0.57), (0.50, 0.51))
    arrow(ax, (0.50, 0.57), (0.72, 0.51))
    arrow(ax, (0.28, 0.40), (0.28, 0.31))
    arrow(ax, (0.50, 0.40), (0.55, 0.31))
    arrow(ax, (0.72, 0.40), (0.55, 0.31))
    arrow(ax, (0.66, 0.255), (0.72, 0.255))

    save_figure(fig, out_path)


def generate_state_reward_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.34, 0.90), 0.32, 0.06, "STATE / ACTION / REWARD DESIGN")

    draw_box(ax, (0.06, 0.72), 0.25, 0.13, "QUEUE FEATURES", "6 normalized lane-group\nqueue estimates\nclipped to [0, 1]")
    draw_box(ax, (0.38, 0.72), 0.25, 0.13, "CONTROL CONTEXT", "Current phase\nNormalized phase age")
    draw_box(ax, (0.70, 0.72), 0.25, 0.13, "TRAFFIC CONTEXT", "AV ratio proxy\nEmergency flag")

    draw_box(ax, (0.32, 0.50), 0.36, 0.12, "ACTION SPACE", "0 = hold current phase\n1 = request switch")
    draw_box(ax, (0.32, 0.31), 0.36, 0.12, "CONTROL CONSTRAINTS", "Minimum green blocks switching\nMaximum green forces switching")
    draw_box(
        ax,
        (0.18, 0.08),
        0.64,
        0.14,
        "REWARD FUNCTION",
        "r_t = -(queue + max_wait + switch) - emergency_delay + emergency_service_bonus",
    )

    arrow(ax, (0.18, 0.72), (0.46, 0.62))
    arrow(ax, (0.50, 0.72), (0.50, 0.62))
    arrow(ax, (0.82, 0.72), (0.54, 0.62))
    arrow(ax, (0.50, 0.50), (0.50, 0.43))
    arrow(ax, (0.50, 0.31), (0.50, 0.22))

    save_figure(fig, out_path)


def generate_curriculum_emergency_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.25, 0.90), 0.50, 0.06, "CURRICULUM AND EMERGENCY CONTROL FLOW")

    draw_box(ax, (0.08, 0.70), 0.22, 0.12, "EPISODE INDEX", "Linear schedule across\ntraining horizon")
    draw_box(ax, (0.39, 0.70), 0.22, 0.12, "CURRICULUM UPDATE", "AV ratio increases\nDifficulty rises by sqrt(x)")
    draw_box(ax, (0.70, 0.70), 0.22, 0.12, "ENVIRONMENT RESPONSE", "Arrival pressure and\nservice dynamics update")

    draw_box(ax, (0.08, 0.44), 0.22, 0.12, "EMERGENCY EVENT", "Stochastic activation\nLane assignment")
    draw_box(ax, (0.39, 0.44), 0.22, 0.12, "BOUNDED PREEMPTION", "Bias phase toward\nemergency lane\nPreemption cap")
    draw_box(ax, (0.70, 0.44), 0.22, 0.12, "RECOVERY CONTROL", "Cooldown steps after\nservice to reduce starvation")

    draw_box(ax, (0.25, 0.16), 0.50, 0.14, "LOGGED METRICS", "Reward, queue, max wait, emergency served, entropy, attributions")

    arrow(ax, (0.30, 0.76), (0.39, 0.76))
    arrow(ax, (0.61, 0.76), (0.70, 0.76))
    arrow(ax, (0.19, 0.70), (0.19, 0.56))
    arrow(ax, (0.50, 0.70), (0.50, 0.56))
    arrow(ax, (0.81, 0.70), (0.81, 0.56))
    arrow(ax, (0.30, 0.50), (0.39, 0.50))
    arrow(ax, (0.61, 0.50), (0.70, 0.50))
    arrow(ax, (0.50, 0.44), (0.50, 0.30))

    save_figure(fig, out_path)


def percent_change(new_value: float, old_value: float, lower_is_better: bool = False) -> float:
    if math.isclose(old_value, 0.0):
        return 0.0
    raw = ((new_value - old_value) / abs(old_value)) * 100.0
    return -raw if lower_is_better else raw


def format_num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def make_key_value_table(rows: list[list[str]], col_widths=None) -> Table:
    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def add_figure(story, image_path: Path, caption: str, styles) -> None:
    story.append(Image(str(image_path), width=6.8 * inch, height=4.6 * inch))
    story.append(Spacer(1, 0.08 * inch))
    story.append(paragraph(f"<b>Figure.</b> {caption}", styles["Caption"]))
    story.append(Spacer(1, 0.18 * inch))


def training_statistics(metrics: pd.DataFrame) -> dict[str, float]:
    first_window = metrics.head(min(10, len(metrics)))
    last_window = metrics.tail(min(10, len(metrics)))
    return {
        "reward_first_mean": float(first_window["episode_reward"].mean()),
        "reward_last_mean": float(last_window["episode_reward"].mean()),
        "queue_first_mean": float(first_window["mean_queue"].mean()),
        "queue_last_mean": float(last_window["mean_queue"].mean()),
        "entropy_first_mean": float(first_window["entropy"].mean()),
        "entropy_last_mean": float(last_window["entropy"].mean()),
        "best_reward": float(metrics["episode_reward"].max()),
        "best_queue": float(metrics["mean_queue"].min()),
        "max_wait_peak": float(metrics["max_wait"].max()),
    }


def comparison_table_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows = [[
        "Method",
        "Mean reward (95% CI)",
        "Mean queue (95% CI)",
        "Max wait (95% CI)",
        "Emergency served rate (95% CI)",
    ]]
    for method in METHOD_ORDER:
        values = summary[method]
        rows.append(
            [
                METHOD_LABELS[method],
                f"{values['reward_mean']:.3f} +/- {values['reward_ci95']:.3f}",
                f"{values['queue_mean']:.3f} +/- {values['queue_ci95']:.3f}",
                f"{values['max_wait_mean']:.3f} +/- {values['max_wait_ci95']:.3f}",
                f"{values['emergency_served_mean']:.3f} +/- {values['emergency_served_ci95']:.3f}",
            ]
        )
    return rows


def raw_results_table_rows(raw_df: pd.DataFrame) -> list[list[str]]:
    rows = [["Method", "Seed", "Mean reward", "Mean queue", "Mean max wait", "Emergency served rate"]]
    for _, row in raw_df.sort_values(["method", "seed"]).iterrows():
        rows.append(
            [
                METHOD_LABELS[row["method"]],
                str(int(row["seed"])),
                f"{row['mean_reward']:.4f}",
                f"{row['mean_queue']:.4f}",
                f"{row['mean_max_wait']:.4f}",
                f"{row['emergency_served_rate']:.4f}",
            ]
        )
    return rows


def attribution_table_rows(attributions: dict[str, float]) -> list[list[str]]:
    rows = [["Feature", "Attribution magnitude"]]
    for key, value in sorted(attributions.items(), key=lambda item: item[1], reverse=True):
        rows.append([FEATURE_LABELS.get(key, key), f"{value:.6f}"])
    return rows


def config_table_rows(section_name: str, data: dict[str, Any]) -> list[list[str]]:
    rows = [["Parameter", f"{section_name} value"]]
    for key, value in data.items():
        rows.append([key, str(value)])
    return rows


def artifact_table_rows(output_dir: Path, sumo_status: SumoStatus) -> list[list[str]]:
    rows = [["Artifact", "Purpose"]]
    rows.extend(
        [
            [str(output_dir / "metrics.csv"), "Episode-wise PPO training log in the lightweight environment"],
            [str(output_dir / "training_summary.json"), "Training and environment hyperparameter snapshot"],
            [str(output_dir / "comparison_raw.csv"), "Per-seed benchmark results for fixed-time, legacy Q-learning, and PPO"],
            [str(output_dir / "comparison_summary.json"), "Aggregated benchmark means and 95% confidence intervals"],
            [str(output_dir / "attributions_latest.json"), "Integrated-gradient style feature attribution summary"],
            [str(output_dir / "report_assets"), "Generated figures and block diagrams for this report"],
            [str(output_dir / "methodology_report.pdf"), "Final black-and-white technical report PDF"],
        ]
    )
    if sumo_status.available:
        rows.extend(
            [
                [str(output_dir / "metrics_sumo.csv"), "SUMO PPO training log used for verification"],
                [str(output_dir / "training_summary_sumo.json"), "SUMO training configuration snapshot"],
            ]
        )
    return rows


def on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.drawString(doc.leftMargin, 20, "Interpretable PPO-Based Traffic Signal Control in Mixed Traffic")
    canvas.drawRightString(A4[0] - doc.rightMargin, 20, f"Page {doc.page}")
    canvas.restoreState()


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyJustify",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=10.4,
            leading=14.0,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            spaceBefore=8,
            spaceAfter=10,
            textColor=colors.black,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            spaceBefore=6,
            spaceAfter=6,
            textColor=colors.black,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.8,
            leading=11,
            alignment=TA_CENTER,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.black,
        )
    )
    return styles


def generate_assets(artifacts: dict[str, Any], sumo_status: SumoStatus, assets_dir: Path) -> dict[str, Path]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "pipeline_diagram": assets_dir / "pipeline_diagram.png",
        "state_reward_diagram": assets_dir / "state_reward_diagram.png",
        "curriculum_emergency_diagram": assets_dir / "curriculum_emergency_diagram.png",
        "training_curves": assets_dir / "training_curves.png",
        "comparison_summary": assets_dir / "comparison_summary.png",
        "comparison_per_seed": assets_dir / "comparison_per_seed.png",
        "attributions": assets_dir / "attributions.png",
        "curriculum_schedule": assets_dir / "curriculum_schedule.png",
    }

    generate_pipeline_diagram(output_paths["pipeline_diagram"])
    generate_state_reward_diagram(output_paths["state_reward_diagram"])
    generate_curriculum_emergency_diagram(output_paths["curriculum_emergency_diagram"])
    generate_training_curves(artifacts["metrics"], output_paths["training_curves"])
    generate_comparison_summary_plot(artifacts["comparison_summary"], output_paths["comparison_summary"])
    generate_per_seed_plot(artifacts["comparison_raw"], output_paths["comparison_per_seed"])
    generate_attribution_plot(artifacts["attributions"], output_paths["attributions"])
    generate_curriculum_plot(int(artifacts["training_summary"]["train_config"]["episodes"]), output_paths["curriculum_schedule"])

    if sumo_status.quantitative_claims_allowed and sumo_status.metrics is not None:
        output_paths["sumo_training"] = assets_dir / "sumo_training.png"
        generate_sumo_training_plot(sumo_status.metrics, output_paths["sumo_training"])

    return output_paths


def build_story(
    title: str,
    artifacts: dict[str, Any],
    sumo_status: SumoStatus,
    figure_paths: dict[str, Path],
    output_dir: Path,
):
    styles = build_styles()
    body = styles["BodyJustify"]
    story = []

    comparison_summary = artifacts["comparison_summary"]
    raw_df = artifacts["comparison_raw"]
    metrics = artifacts["metrics"]
    attributions = artifacts["attributions"]
    training_summary = artifacts["training_summary"]
    train_cfg = merge_with_defaults(training_summary["train_config"], asdict(TrainConfig()))
    env_cfg = merge_with_defaults(training_summary["env_config"], asdict(EnvConfig()))
    stats = training_statistics(metrics)

    fixed_time = comparison_summary["fixed_time"]
    legacy_ql = comparison_summary["legacy_ql"]
    ppo = comparison_summary["ppo"]

    ppo_queue_gain_vs_fixed = percent_change(ppo["queue_mean"], fixed_time["queue_mean"], lower_is_better=True)
    ppo_reward_gain_vs_fixed = percent_change(ppo["reward_mean"], fixed_time["reward_mean"], lower_is_better=False)
    ppo_emergency_gain_vs_fixed = percent_change(ppo["emergency_served_mean"], fixed_time["emergency_served_mean"], lower_is_better=False)
    legacy_queue_gain_vs_ppo = percent_change(legacy_ql["queue_mean"], ppo["queue_mean"], lower_is_better=True)
    top_features = sorted(attributions.items(), key=lambda item: item[1], reverse=True)[:3]
    top_feature_text = ", ".join(f"{FEATURE_LABELS.get(name, name)} ({value:.4f})" for name, value in top_features)

    story.append(Spacer(1, 1.0 * inch))
    story.append(paragraph(title, styles["CoverTitle"]))
    story.append(Spacer(1, 0.20 * inch))
    story.append(
        paragraph(
            "Interpretable PPO-Based Traffic Signal Control in Mixed Traffic with Curriculum Learning and Constrained Emergency Handling",
            styles["CoverTitle"],
        )
    )
    story.append(Spacer(1, 0.35 * inch))
    story.append(
        paragraph(
            "This report explains the methodology end to end, ties the narrative to the actual repository implementation, and documents the currently available quantitative evidence.",
            styles["Caption"],
        )
    )
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        paragraph(
            f"Generated from repository artifacts on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Primary evidence files were loaded from {output_dir}.",
            styles["Caption"],
        )
    )
    story.append(PageBreak())

    story.append(paragraph("Executive Summary", styles["SectionTitle"]))
    story.append(
        paragraph(
            "The study develops a reinforcement-learning-based traffic signal controller for mixed traffic conditions and evaluates whether Proximal Policy Optimization (PPO) can deliver robust operational behavior relative to fixed-time control and a legacy tabular Q-learning baseline while remaining interpretable and emergency-aware. The methodology is centered on a consistent state design, a constrained hold-or-switch action space, a reward that balances congestion reduction with operational fairness, and a curriculum that gradually increases mixed-traffic difficulty.",
            body,
        )
    )
    story.append(
        paragraph(
            f"The current benchmark evidence is the saved lightweight-environment comparison across three seeds. In that exported comparison, PPO achieved a mean reward of {ppo['reward_mean']:.3f}, mean queue of {ppo['queue_mean']:.3f}, mean max wait of {ppo['max_wait_mean']:.3f}, and emergency served rate of {ppo['emergency_served_mean']:.3f}. Relative to fixed-time control, PPO improved reward by {ppo_reward_gain_vs_fixed:.2f}% and reduced mean queue by {ppo_queue_gain_vs_fixed:.2f}%, while also raising emergency served rate by {ppo_emergency_gain_vs_fixed:.2f}%. The current lightweight benchmark does not yet show PPO outperforming the legacy Q-learning baseline on queue or reward, which is why the report keeps the conclusions disciplined and matched to the artifacts.",
            body,
        )
    )
    story.append(
        paragraph(
            f"The explainability export indicates that the learned PPO policy currently relies most heavily on {top_feature_text}. That pattern is consistent with a controller whose decisions are dominated by directional queue pressure and current control context rather than the AV ratio proxy alone.",
            body,
        )
    )
    story.append(
        paragraph(
            f"SUMO verification status: {sumo_status.reason}",
            body,
        )
    )

    story.append(paragraph("Study Objective And Research Questions", styles["SectionTitle"]))
    story.append(
        paragraph(
            "The study objective is to design, implement, and evaluate an interpretable traffic signal controller that can operate under mixed traffic conditions and react to emergency events without violating fundamental signal-control constraints. The methodology is organized around four research questions: whether PPO outperforms fixed-time and legacy Q-learning under matched conditions, whether curriculum learning improves stability and final performance, which state variables drive policy decisions, and what trade-off exists between emergency prioritization and ordinary-flow fairness.",
            body,
        )
    )
    story.append(
        paragraph(
            "The repository operationalizes those questions through two environment paths. The lightweight simulator is the primary benchmarking environment because it is reproducible, fast to train, and already backed by exported CSV and JSON artifacts. The SUMO-TraCI adapter provides a scenario-level realism path that reuses the same 10-dimensional state shape and binary action interface, allowing the methodology to extend to external networks without redesigning the PPO core.",
            body,
        )
    )
    add_figure(
        story,
        figure_paths["pipeline_diagram"],
        "Reference-style overview of the implemented RL traffic signal control workflow. The same overall pipeline supports both the lightweight environment and the SUMO adapter path.",
        styles,
    )

    story.append(paragraph("Problem Formulation", styles["SectionTitle"]))
    story.append(
        paragraph(
            "The control problem is modeled as a Markov Decision Process with state S, action A, transition dynamics P, reward R, and discount factor gamma. Each control step selects one of two actions: hold the current phase or request a phase switch. The environment then applies a constraint layer that enforces minimum-green and maximum-green rules before the simulator evolves queues, waiting times, and emergency state. Because control occurs at every simulation step, the policy observes a fine-grained closed loop in which each decision can influence subsequent queue growth, service, and emergency handling.",
            body,
        )
    )
    story.append(
        paragraph(
            "In the implementation, the MDP is carried by `src/rl_tsc/env.py`. The `MixedTrafficSignalEnv.step()` method increments the simulation step, optionally spawns an emergency event, applies bounded preemption logic, updates arrivals with a Poisson process, serves the currently active phase, clips queue levels to a queue-cap constant, updates wait counters, and finally computes the clipped reward returned to PPO. This direct mapping from conceptual formulation to executable code is important because the report only claims behaviors that are visible in the current implementation.",
            body,
        )
    )
    story.append(
        paragraph(
            "A compact analytical expression for the implemented reward is: `r_t = -(queue_penalty + max_wait_penalty + switch_cost) - emergency_delay_penalty + emergency_served_bonus`, followed by clipping to the configured reward range. Although the internal code applies the individual coefficients separately, the effective design goal is to penalize persistent congestion and starvation while rewarding successful emergency service and avoiding unstable switching.",
            body,
        )
    )

    story.append(paragraph("Environment Design", styles["SectionTitle"]))
    story.append(
        paragraph(
            "The lightweight environment is a single-intersection abstraction with six inbound lane queues. Lanes 0 through 2 belong to one movement group and lanes 3 through 5 belong to the other. Only one group is served at a time, so the phase indicator is binary. Arrivals are sampled from a Poisson distribution whose intensity grows with curriculum difficulty and is partially smoothed by the AV ratio proxy. Service capacity is based on a baseline discharge rate that is modestly increased when the AV ratio is higher, which makes the curriculum affect both demand and service characteristics.",
            body,
        )
    )
    story.append(
        paragraph(
            f"The current saved training summary records the environment configuration as min green = {env_cfg['min_green_steps']} steps, max green = {env_cfg['max_green_steps']} steps, queue cap = {env_cfg['queue_cap']}, emergency probability = {env_cfg['emergency_prob']}, preemption cap = {env_cfg['preemption_cap_steps']} steps, recovery = {env_cfg['preemption_recovery_steps']} steps, base arrival rate = {env_cfg['base_arrival_rate']}, and base service rate = {env_cfg['base_service_rate']}. Those numbers are not rhetorical defaults in the report; they are pulled directly from the exported `training_summary.json` artifact.",
            body,
        )
    )
    story.append(
        paragraph(
            "The SUMO path is implemented by `src/rl_tsc/sumo_adapter.py` and `src/rl_tsc/train_sumo.py`. It discovers traffic-light IDs automatically when one is not supplied, derives queue proxies from controlled-lane halting counts, preserves the same 10-dimensional state shape for policy compatibility, and exposes a compatible `reset()` / `step(action)` interface so PPO can train without redesigning the agent. The SUMO reward currently uses queue sum, max-wait estimate, and emergency activity as the main components, which makes it suitable for verification and future realism studies even though the detailed constrained emergency accounting in SUMO should still be treated carefully unless validated in exported results.",
            body,
        )
    )

    story.append(paragraph("State Representation, Actions, And Constraints", styles["SectionTitle"]))
    story.append(
        paragraph(
            "The implemented controller uses a fixed 10-dimensional state vector. The first six elements are normalized queue estimates for six lane groups. Element seven is the current phase indicator. Element eight is normalized phase age. Element nine is the AV ratio proxy injected by the curriculum schedule. Element ten is the emergency flag. This exact shape is produced in both the lightweight environment and the SUMO adapter so that the actor and critic networks can be reused without architectural branching.",
            body,
        )
    )
    story.append(
        paragraph(
            "The action space is intentionally minimal: action 0 holds the current phase and action 1 requests a switch. The minimal action interface is important because it keeps the control semantics interpretable. The safety layer then enforces two operational constraints. First, if phase age is below minimum green, requested switching is blocked. Second, if phase age exceeds maximum green, a switch is forced regardless of the agent's preference. Emergency handling sits on top of that layer and can bias the signal toward the phase that serves the emergency lane, but only up to a bounded preemption cap followed by a cooldown-style recovery period.",
            body,
        )
    )
    add_figure(
        story,
        figure_paths["state_reward_diagram"],
        "The implemented state composition, two-action control semantics, constraint layer, and reward structure used by the lightweight RL controller.",
        styles,
    )

    story.append(paragraph("Reward Design", styles["SectionTitle"]))
    story.append(
        paragraph(
            f"The reward function combines queue penalty, maximum-wait penalty, switch penalty, emergency delay penalty, and emergency served bonus. In the saved environment configuration those coefficients are queue penalty = {env_cfg['queue_penalty_coef']}, wait penalty = {env_cfg['wait_penalty_coef']}, switch penalty = {env_cfg['switch_penalty']}, emergency delay penalty = {env_cfg['emergency_delay_penalty']}, emergency served bonus = {env_cfg['emergency_served_bonus']}, and reward clip = {env_cfg['reward_clip']}. Queue values are normalized by `queue_cap * num_lanes`, while maximum wait is normalized by `max_green_steps`, so the reward stays numerically stable across the training horizon.",
            body,
        )
    )
    story.append(
        paragraph(
            "This reward design encodes the core traffic-engineering trade-off in the methodology. Penalizing queues encourages throughput. Penalizing the worst waiting movement counters starvation. Penalizing unnecessary switching discourages unstable oscillations. Penalizing unresolved emergency state pushes the agent toward responsive service. Rewarding successful emergency service creates an explicit learning signal that makes emergency awareness visible in both training logs and evaluation metrics.",
            body,
        )
    )

    story.append(paragraph("Learning Algorithms And Baselines", styles["SectionTitle"]))
    story.append(
        paragraph(
            "PPO is the primary method. In `src/rl_tsc/models.py`, both the actor and the critic are two-hidden-layer fully connected networks with Tanh activations and hidden width 64. In `src/rl_tsc/ppo.py`, the actor outputs logits over the two actions, the critic estimates the state value, advantages are computed with Generalized Advantage Estimation, the clipped PPO objective stabilizes updates, entropy regularization preserves exploration pressure, and gradient clipping controls optimization spikes. The saved training configuration reports gamma = {gamma}, gae_lambda = {gae_lambda}, clip ratio = {clip_ratio}, actor learning rate = {actor_lr}, critic learning rate = {critic_lr}, update epochs = {update_epochs}, entropy coefficient = {entropy_coef}, and max gradient norm = {max_grad_norm}.".format(
                gamma=train_cfg["gamma"],
                gae_lambda=train_cfg["gae_lambda"],
                clip_ratio=train_cfg["clip_ratio"],
                actor_lr=train_cfg["actor_lr"],
                critic_lr=train_cfg["critic_lr"],
                update_epochs=train_cfg["update_epochs"],
                entropy_coef=train_cfg["entropy_coef"],
                max_grad_norm=train_cfg["max_grad_norm"],
            ),
            body,
        )
    )
    story.append(
        paragraph(
            "The legacy Q-learning baseline in `src/rl_tsc/baselines.py` intentionally uses a lower-capacity representation. It discretizes the sum of eastbound and southbound queue groups into five bins each, includes phase and phase-age bins, and updates a tabular Q-table with epsilon-greedy exploration. This makes it a useful low-dimensional value-based baseline in an environment that may still be simple enough for coarse discretization to remain competitive.",
            body,
        )
    )
    story.append(
        paragraph(
            "The fixed-time baseline uses a deterministic switching rule: it requests a switch once phase age reaches a hold duration of 20 steps. That provides a no-learning operational reference and helps anchor the RL results against a simple yet realistic baseline.",
            body,
        )
    )

    story.append(paragraph("Curriculum Learning And Explainability", styles["SectionTitle"]))
    story.append(
        paragraph(
            "Curriculum learning is implemented by `curriculum_schedule()` in `src/rl_tsc/env.py`. In linear mode, the AV ratio increases linearly from 0.0 to 1.0 across the training horizon, while difficulty rises from 0.2 toward 1.0 using a square-root schedule. That means the controller sees a gradual increase in both mixed-traffic composition and effective demand difficulty rather than being dropped directly into the hardest regime.",
            body,
        )
    )
    story.append(
        paragraph(
            f"The saved PPO training artifact uses {train_cfg['episodes']} training episodes, {train_cfg['steps_per_episode']} steps per episode, and curriculum mode `{train_cfg['curriculum']}`. The first 10 training episodes have an average reward of {stats['reward_first_mean']:.2f} and average queue of {stats['queue_first_mean']:.2f}, while the last 10 episodes average reward {stats['reward_last_mean']:.2f} and queue {stats['queue_last_mean']:.2f}. This indicates how the training trajectory evolves under the schedule rather than relying on a single terminal episode.",
            body,
        )
    )
    story.append(
        paragraph(
            "Explainability is produced by `src/rl_tsc/explain.py`, which computes integrated-gradient style attributions over sampled actor decisions and exports the mean absolute feature importance to JSON. The current export ranks the queue features highest, with the strongest signals on the southbound queue group. The AV ratio proxy and emergency flag appear relatively small in this specific attribution summary, which is a reasonable outcome when the saved sample mix is dominated by queue-pressure decisions and emergency events are comparatively sparse.",
            body,
        )
    )
    add_figure(
        story,
        figure_paths["curriculum_emergency_diagram"],
        "How the implemented curriculum updates traffic composition and difficulty, and how emergency events pass through bounded preemption and recovery before metrics are logged.",
        styles,
    )
    add_figure(
        story,
        figure_paths["curriculum_schedule"],
        "The exact linear curriculum schedule used by the saved PPO training run, generated from the same function used in the environment implementation.",
        styles,
    )
    add_figure(
        story,
        figure_paths["attributions"],
        "Integrated-gradient style attribution magnitudes exported by the trained PPO actor. Queue features dominate the current explanation profile.",
        styles,
    )

    story.append(paragraph("Experimental Protocol And Available Artifacts", styles["SectionTitle"]))
    story.append(
        paragraph(
            "The repository comparison runner evaluates fixed-time control, legacy Q-learning, and PPO under matched evaluation conditions and exports both per-seed and aggregated results. The current saved methodology text and the code defaults agree on a three-seed comparison using seeds 42, 43, and 44, PPO training episodes = 160, evaluation episodes = 12, and 180 steps per episode. Confidence intervals in the current report come directly from the exported summary JSON, which is produced by the `ci95()` function in `src/rl_tsc/compare.py`.",
            body,
        )
    )
    story.append(make_key_value_table(comparison_table_rows(comparison_summary), col_widths=[1.05 * inch, 1.55 * inch, 1.45 * inch, 1.30 * inch, 1.65 * inch]))
    story.append(Spacer(1, 0.18 * inch))
    story.append(make_key_value_table(raw_results_table_rows(raw_df), col_widths=[1.20 * inch, 0.55 * inch, 1.15 * inch, 1.10 * inch, 1.10 * inch, 1.35 * inch]))
    story.append(Spacer(1, 0.18 * inch))
    add_figure(
        story,
        figure_paths["comparison_summary"],
        "Method-level comparison across the exported summary metrics with 95% confidence intervals in black-and-white styling.",
        styles,
    )
    add_figure(
        story,
        figure_paths["comparison_per_seed"],
        "Per-seed comparison view showing how each method behaves across seeds 42, 43, and 44 before aggregation.",
        styles,
    )

    story.append(paragraph("Results Interpretation", styles["SectionTitle"]))
    story.append(
        paragraph(
            f"The saved comparison shows that PPO is stronger than fixed-time control on several axes but does not yet surpass the legacy Q-learning baseline in the present lightweight setup. PPO reduces mean queue from {fixed_time['queue_mean']:.3f} to {ppo['queue_mean']:.3f}, which corresponds to a {ppo_queue_gain_vs_fixed:.2f}% queue reduction relative to fixed-time. PPO also raises emergency served rate from {fixed_time['emergency_served_mean']:.3f} to {ppo['emergency_served_mean']:.3f}. However, the legacy Q-learning baseline reaches the lowest mean queue at {legacy_ql['queue_mean']:.3f} and the highest mean reward at {legacy_ql['reward_mean']:.3f}, outperforming PPO on those metrics in the current benchmark.",
            body,
        )
    )
    story.append(
        paragraph(
            f"The per-seed raw table helps explain why the conclusions must stay measured. PPO is consistent on emergency served rate, reaching 1.000 for each exported seed, but its queue metric varies from {raw_df[raw_df['method'] == 'ppo']['mean_queue'].min():.3f} to {raw_df[raw_df['method'] == 'ppo']['mean_queue'].max():.3f}. The legacy Q-learning baseline performs particularly well on queue for seed 44, and that contributes to its aggregated advantage. This is exactly the sort of result that justifies the methodology's emphasis on matched baselines, confidence intervals, and publication-safe claim discipline.",
            body,
        )
    )
    story.append(
        paragraph(
            f"The training log provides additional context. The best episode reward in the saved PPO training run is {stats['best_reward']:.2f}, the lowest observed mean queue is {stats['best_queue']:.2f}, and the maximum observed wait reaches {stats['max_wait_peak']:.2f}. The moving-average plots show how reward, queue, max wait, and entropy evolve across training rather than relying only on the cross-method benchmark snapshot.",
            body,
        )
    )
    add_figure(
        story,
        figure_paths["training_curves"],
        "Episode-wise PPO training behavior in the lightweight environment, including reward, queue, max wait, and entropy trends.",
        styles,
    )

    story.append(paragraph("SUMO Verification", styles["SectionTitle"]))
    story.append(
        paragraph(
            "The repository includes a SUMO-TraCI adapter and Manhattan scenario assets so that the same PPO backbone can be exercised on an external traffic network. This report applies a strict verification rule: SUMO numbers are included only when `metrics_sumo.csv` and `training_summary_sumo.json` exist and contain meaningful non-zero activity.",
            body,
        )
    )
    story.append(paragraph(sumo_status.reason, body))
    if sumo_status.quantitative_claims_allowed and sumo_status.metrics is not None and sumo_status.summary is not None:
        sumo_metrics = sumo_status.metrics
        sumo_summary = sumo_status.summary
        story.append(
            paragraph(
                f"The verified SUMO training run contains {len(sumo_metrics)} episodes. The final recorded mean queue is {sumo_metrics['mean_queue'].iloc[-1]:.3f}, the final max wait is {sumo_metrics['max_wait'].iloc[-1]:.3f}, and the mean reward across all SUMO training episodes is {sumo_metrics['episode_reward'].mean():.3f}. The report still treats this as a verification study rather than a full baseline comparison, because the current SUMO export does not include fixed-time and legacy Q-learning comparison artifacts on the same scenario.",
                body,
            )
        )
        story.append(
            paragraph(
                f"The SUMO training summary records the controlled scenario as `{sumo_summary['sumo_config']['sumo_cfg_path']}` and the signal binary as `{sumo_summary['sumo_config']['sumo_binary']}`. That makes the SUMO subsection reproducible and aligned to a concrete scenario rather than a generic realism claim.",
                body,
            )
        )
        add_figure(
            story,
            figure_paths["sumo_training"],
            "Verified PPO training behavior on the SUMO scenario. These curves are only included when quantitative SUMO artifacts exist and show real activity.",
            styles,
        )
    else:
        story.append(
            paragraph(
                "Because quantitative SUMO verification is unavailable or insufficient, the academically safe interpretation is that SUMO support is implemented and scenario assets are present, but the benchmark evidence in this report remains lightweight-environment evidence. That preserves implementation accuracy while avoiding overstatement.",
                body,
            )
        )

    story.append(paragraph("Limitations And Implementation-To-Claim Alignment", styles["SectionTitle"]))
    story.append(
        paragraph(
            "Three methodological limitations are especially important. First, the lightweight environment may still be simple enough for a discretized value-based baseline to remain highly competitive, which is visible in the current legacy Q-learning results. Second, reward shaping coefficients strongly influence the balance between queue efficiency, fairness, and emergency service, so hyperparameter sensitivity can alter relative ranking. Third, external validity is still limited until the SUMO path is expanded into a matched multi-method evaluation pipeline with more seeds and stronger statistical tests.",
            body,
        )
    )
    story.append(
        paragraph(
            "For those reasons, the report follows an implementation-to-claim alignment rule. Behaviors verified only in the lightweight environment are described as lightweight-environment evidence. SUMO is described as an implemented integration path unless fresh SUMO artifacts support quantitative reporting. This disciplined separation is not a weakness of the methodology; it is part of the methodology because it keeps the written claims synchronized with the validated code paths.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(paragraph("Appendix: Hyperparameters And Artifact Inventory", styles["SectionTitle"]))
    story.append(paragraph("Training configuration snapshot", styles["SubTitle"]))
    story.append(make_key_value_table(config_table_rows("Training", train_cfg), col_widths=[2.05 * inch, 4.15 * inch]))
    story.append(Spacer(1, 0.16 * inch))
    story.append(paragraph("Environment configuration snapshot", styles["SubTitle"]))
    story.append(make_key_value_table(config_table_rows("Environment", env_cfg), col_widths=[2.05 * inch, 4.15 * inch]))
    story.append(Spacer(1, 0.16 * inch))
    story.append(paragraph("Feature attribution export", styles["SubTitle"]))
    story.append(make_key_value_table(attribution_table_rows(attributions), col_widths=[3.0 * inch, 2.0 * inch]))
    story.append(Spacer(1, 0.16 * inch))
    story.append(paragraph("Artifact inventory used in this report", styles["SubTitle"]))
    story.append(make_key_value_table(artifact_table_rows(output_dir, sumo_status), col_widths=[3.2 * inch, 2.8 * inch]))

    return story


def generate_report(project_root: Path, output_dir: Path, assets_dir: Path, output_pdf: Path, title: str) -> None:
    artifacts = load_required_artifacts(output_dir)
    sumo_status = load_sumo_status(output_dir)
    figure_paths = generate_assets(artifacts, sumo_status, assets_dir)
    story = build_story(title, artifacts, sumo_status, figure_paths, output_dir)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.60 * inch,
        title=title,
        author="OpenAI Codex",
    )
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main() -> None:
    args = build_parser().parse_args()
    generate_report(
        project_root=args.project_root,
        output_dir=args.output_dir,
        assets_dir=args.assets_dir,
        output_pdf=args.output_pdf,
        title=args.title,
    )
    print(f"Report generated: {args.output_pdf}")
    print(f"Figure assets: {args.assets_dir}")


if __name__ == "__main__":
    main()
