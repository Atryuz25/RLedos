"""Plotting: cost-vs-latency comparison, attack-pattern traces, learning curves.

Colour-blind-safe (Okabe-Ito) palette; a plain-text summary.json accompanies
every chart (written by the CLI), per docs/03_DATA_SCHEMAS.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rl_edos.evaluation.metrics import MetricsSummary  # noqa: E402

# Okabe & Ito (2008) colour-blind-safe qualitative palette.
OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]


def plot_comparison(metrics_by_controller: dict[str, MetricsSummary], out_path: str | Path) -> None:
    """Cost (currency) vs. p95 latency (ms) scatter, one point per controller."""
    names = list(metrics_by_controller)
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, name in enumerate(names):
        m = metrics_by_controller[name]
        ax.scatter(
            m.mean_cost_under_attack,
            m.p95_latency_ms,
            color=OKABE_ITO[i % len(OKABE_ITO)],
            s=140,
            edgecolor="black",
            label=name,
        )
    ax.set_xlabel("mean cost under attack (currency/interval)")
    ax.set_ylabel("p95 latency (ms)")
    ax.set_title("Cost vs. latency under attack (lower-left is better)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attack_trace(
    times_s: list[float], legit_rates: list[float], attack_rates: list[float], out_path: str | Path
) -> None:
    """Legit vs. attack request-rate trace over the episode."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times_s, legit_rates, color=OKABE_ITO[0], label="legit")
    ax.plot(times_s, attack_rates, color=OKABE_ITO[5], label="attack")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("requests/second")
    ax.set_title("Attack traffic pattern")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_selfplay_stability(round_metrics: list[dict], out_path: str | Path) -> None:
    """Cost / latency / detection-trip-rate trend across self-play rounds (Phase 4)."""
    rounds = [m["round"] for m in round_metrics]
    costs = [m["mean_cost_under_attack"] for m in round_metrics]
    latencies = [m["p95_latency_ms"] for m in round_metrics]
    trips = [m["detection_trip_rate"] for m in round_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(rounds, costs, color=OKABE_ITO[5], marker="o")
    axes[0].set_title("Attacker-inflicted cost")
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("mean cost / interval")

    axes[1].plot(rounds, latencies, color=OKABE_ITO[0], marker="o")
    axes[1].set_title("Defender p95 latency")
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("p95 latency (ms)")

    axes[2].plot(rounds, trips, color=OKABE_ITO[4], marker="o")
    axes[2].set_title("Attacker detection-trip rate")
    axes[2].set_xlabel("round")
    axes[2].set_ylabel("fraction of intervals")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Self-play stability across rounds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_curves(reward_history: list[float], out_path: str | Path) -> None:
    """Learning curve; supports the interface ahead of Phase 3's learner."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if reward_history:
        ax.plot(reward_history, color=OKABE_ITO[0])
        ax.set_xlabel("training step")
        ax.set_ylabel("episode reward")
        ax.set_title("Learning curve")
        ax.grid(alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No training curve yet\n(populated once a learner exists, Phase 3)",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
