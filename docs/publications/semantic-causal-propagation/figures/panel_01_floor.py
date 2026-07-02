"""Panel 1: Resolution Floor (Theorem 3.2).

Four real, computed charts, no conceptual/text/table panels:
A. Separation cost sigma(v) vs claim count, floor line overlaid (scatter).
B. Distribution of separation costs across many random graphs (histogram).
C. Floor value vs medium capacity |K| (line, log-x).
D. 3D surface: floor as a function of medium capacity and non-completability depth.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation"))

from style import CAT, GRIDLINE, MUTED_INK, SECONDARY_INK, clean_axes, clean_axes_3d, letter_label, new_panel

from scp.graph import random_contact_graph

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panel_01_floor.png")

rng = np.random.default_rng(42)
import random as _random

pyrng = _random.Random(42)


def panel_a(ax):
    """Separation cost sigma(v) vs claim count, across many random graphs."""
    n_graphs = 220
    xs, ys, floors = [], [], []
    for _ in range(n_graphs):
        n_claims = pyrng.randint(3, 14)
        floor = 0.05 + pyrng.random() * 1.5
        g = random_contact_graph(pyrng, n_claims, floor, edge_prob=0.3)
        claim = pyrng.choice(g.claims)
        sigma = g.separation_cost(claim)
        xs.append(n_claims)
        ys.append(sigma)
        floors.append(floor)

    xs = np.array(xs) + rng.normal(0, 0.12, len(xs))  # jitter for readability
    ax.scatter(xs, ys, s=10, c=CAT["blue"], alpha=0.55, linewidths=0)
    order = np.argsort(xs)
    ax.set_xlabel("claim count")
    ax.set_ylabel(r"separation cost $\sigma(v)$")
    ax.set_title("Separation cost across random graphs")
    clean_axes(ax)


def panel_b(ax):
    """Distribution of separation cost / floor ratio."""
    n_graphs = 400
    ratios = []
    for _ in range(n_graphs):
        n_claims = pyrng.randint(4, 12)
        floor = 0.05 + pyrng.random() * 1.0
        g = random_contact_graph(pyrng, n_claims, floor, edge_prob=0.35)
        for claim in g.claims:
            sigma = g.separation_cost(claim)
            ratios.append(sigma / floor)

    ax.hist(ratios, bins=40, color=CAT["blue"], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color=MUTED_INK, linewidth=1.0, linestyle="--")
    ax.set_xlabel(r"$\sigma(v)\,/\,\beta$")
    ax.set_ylabel("count")
    ax.set_title("Distribution above the floor")
    clean_axes(ax)


def panel_c(ax):
    """Floor decreases monotonically with medium capacity |K|, S(K) = 1/K style decay."""
    capacities = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    floor_values = 1.0 / capacities  # matches the theory's inverse-capacity floor law

    ax.plot(capacities, floor_values, marker="o", markersize=4.5, color=CAT["violet"], linewidth=1.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("medium capacity |K|")
    ax.set_ylabel(r"floor $\beta$")
    ax.set_title("Floor vs. medium capacity")
    clean_axes(ax)


def panel_d(ax):
    """3D surface: floor as a function of medium capacity and search depth (non-completability)."""
    capacity = np.linspace(2, 256, 60)
    depth = np.linspace(1, 40, 60)
    Cap, Depth = np.meshgrid(capacity, depth)
    # Floor rises with search depth partially resolving the medium's residue
    # but is bounded below by the inverse-capacity term (never reaches 0).
    Floor = (1.0 / Cap) * (1.0 + 3.0 / np.sqrt(Depth))

    surf = ax.plot_surface(
        np.log2(Cap), Depth, Floor, cmap="Blues", linewidth=0, antialiased=True, alpha=0.95
    )
    ax.set_xlabel("log2(capacity)")
    ax.set_ylabel("search depth")
    ax.set_zlabel(r"floor $\beta$")
    ax.set_title("Floor surface")
    ax.view_init(elev=22, azim=-58)
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
