"""Panel 8: Closure vs. Confidence Threshold + Scheduler Soundness
(Theorem 8.2, Theorem 10.6).

A. Two-cluster instances: confidence-threshold satisfaction vs. closure status.
B. Scheduler priority vs residue descent rate, stalled vs converging seeks.
C. Committed record over scheduler ticks (monotone climb, several live seeks).
D. 3D: scheduler priority surface over (descent rate, residual-above-floor).
"""

import os
import sys

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation"))

from style import CAT, MUTED_INK, clean_axes, clean_axes_3d, letter_label, new_panel

from scp.closure import confidence_threshold_met, equivalence_classes, is_closed
from scp.graph import two_cluster_graph

import random as _random

pyrng = _random.Random(31)
rng = np.random.default_rng(31)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panel_08_closure_scheduler.png")

THETA = 0.5


def panel_a(ax):
    n_trials = 220
    threshold_x, closed_y = [], []
    for _ in range(n_trials):
        cluster_size = pyrng.randint(2, 6)
        floor = 0.05 + pyrng.random() * 0.5
        graph, a_claims, b_claims = two_cluster_graph(pyrng, cluster_size, floor)
        target_a, target_b = a_claims[0], b_claims[0]

        confidence = 1.0 - graph.alignment(target_a, target_a) / graph.total_weight()
        classes_so_far = equivalence_classes(graph, [target_a])
        closed = is_closed([target_b], classes_so_far, graph)

        threshold_x.append(confidence)
        closed_y.append(1 if closed else 0)

    x = np.array(threshold_x) + rng.normal(0, 0.003, len(threshold_x))
    y = np.array(closed_y, dtype=float) + rng.normal(0, 0.02, len(closed_y))
    ax.scatter(x, y, s=12, c=CAT["blue"], alpha=0.55, linewidths=0)
    ax.axhline(0.0, color=MUTED_INK, linewidth=0.8, linestyle="--")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["not closed", "closed"])
    ax.set_xlabel("confidence at threshold check")
    ax.set_title("High confidence, search not closed")
    clean_axes(ax)


def panel_b(ax):
    n_points = 500
    deltas = rng.uniform(-0.5, 3.0, n_points)
    residuals = rng.uniform(0.01, 5.0, n_points)
    floor = 0.01
    priorities = np.where(deltas > 0, deltas / np.maximum(residuals - floor, floor), 0.0)
    priorities_capped = np.clip(priorities, 0, 12)

    stalled = deltas <= 0
    ax.scatter(residuals[stalled], priorities_capped[stalled], s=10, c=CAT["red"], alpha=0.5, linewidths=0, label="stalled")
    ax.scatter(residuals[~stalled], priorities_capped[~stalled], s=10, c=CAT["blue"], alpha=0.5, linewidths=0, label="descending")
    ax.set_xlabel("current residue")
    ax.set_ylabel("scheduler priority (capped)")
    ax.set_title("Stalled seeks get zero priority")
    ax.legend(loc="upper right", fontsize=6.5)
    clean_axes(ax)


def panel_c(ax):
    n_ticks = 60
    n_seeks = 4
    colors = ["blue", "aqua", "yellow", "red"]
    for i in range(n_seeks):
        rate = pyrng.uniform(0.3, 1.0)
        record = np.cumsum(rng.poisson(rate, n_ticks))
        ax.plot(np.arange(n_ticks), record, linewidth=1.7, color=CAT[colors[i]], label=f"seek {i+1}")

    ax.set_xlabel("scheduler tick")
    ax.set_ylabel("committed record M")
    ax.set_title("Monotone committed record")
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)
    clean_axes(ax)


def panel_d(ax):
    delta = np.linspace(0.001, 2.0, 45)
    residual = np.linspace(0.02, 5.0, 45)
    Delta, Residual = np.meshgrid(delta, residual)
    floor = 0.01
    Priority = Delta / np.maximum(Residual - floor, floor)
    Priority = np.clip(Priority, 0, 10)

    surf = ax.plot_surface(Delta, Residual, Priority, cmap="Blues", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("descent rate")
    ax.set_ylabel("residual")
    ax.set_zlabel("priority (capped)")
    ax.set_title("Scheduler priority surface")
    ax.view_init(elev=24, azim=-58)
    clean_axes_3d(ax)


def main():
    fig = new_panel(figsize=(16.0, 3.8))
    ax_a = fig.add_subplot(1, 4, 1)
    ax_b = fig.add_subplot(1, 4, 2)
    ax_c = fig.add_subplot(1, 4, 3)
    ax_d = fig.add_subplot(1, 4, 4, projection="3d")

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d)

    letter_label(ax_a, "A")
    letter_label(ax_b, "B")
    letter_label(ax_c, "C")
    letter_label(ax_d, "D", is_3d=True)

    fig.tight_layout(w_pad=2.4)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
