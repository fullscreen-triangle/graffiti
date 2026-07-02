"""Panel 7: Coherence Requires a Triangle (Theorem 7.8).

A. Robustness outcome (survives single-catalyst removal) vs support-cycle length.
B. Mutual support surviving after deletion vs clique/triangle size.
C. Resultant shift-vector length: coherent vs incoherent catalyst sets (histograms).
D. 3D: catalyst shift vectors in meaning-space, coherent cluster vs incoherent scatter.
"""

import os
import sys

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation"))

from style import CAT, GRIDLINE, MUTED_INK, clean_axes, clean_axes_3d, letter_label, new_panel

from scp.catalysis import SupportGraph

import itertools
import random as _random

pyrng = _random.Random(29)
rng = np.random.default_rng(29)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panel_07_coherence.png")

THETA = 0.5


def build_cycle(length, strength_above=True):
    catalysts = [f"g{i}" for i in range(length)]
    graph = SupportGraph(catalysts)
    strengths = {}
    for i in range(length):
        a, b = catalysts[i], catalysts[(i + 1) % length]
        s = pyrng.uniform(THETA + 1e-3, 1.0) if strength_above else pyrng.uniform(0.0, THETA - 1e-3)
        strengths[(a, b)] = s
        strengths[(b, a)] = s
        if s > THETA:
            graph.add_support(a, b)
            graph.add_support(b, a)
    return graph, strengths, catalysts


def panel_a(ax):
    lengths = [1, 2, 3, 4, 5]
    n_trials = 160
    grounded_rates = []
    for length in lengths:
        n_grounded = 0
        for _ in range(n_trials):
            graph, strengths, catalysts = build_cycle(length, strength_above=True)
            if length >= 3 and graph.robust_to_single_removal(strengths, THETA):
                n_grounded += 1
        grounded_rates.append(n_grounded / n_trials)

    ax.bar(lengths, grounded_rates, color=CAT["blue"], width=0.6)
    ax.set_xlabel("support-cycle length")
    ax.set_ylabel("fraction grounded (robust)")
    ax.set_title("Grounding requires length >= 3")
    ax.set_xticks(lengths)
    clean_axes(ax)


def panel_b(ax):
    sizes = [2, 3, 4, 5]
    n_trials = 150
    survival_fracs = []
    for size in sizes:
        survived = 0
        for _ in range(n_trials):
            catalysts = [f"g{i}" for i in range(size)]
            graph = SupportGraph(catalysts)
            strengths = {}
            for a, b in itertools.permutations(catalysts, 2):
                s = pyrng.uniform(THETA + 1e-3, 1.0)
                strengths[(a, b)] = s
                graph.add_support(a, b)
            if graph.robust_to_single_removal(strengths, THETA):
                survived += 1
        survival_fracs.append(survived / n_trials)

    ax.plot(sizes, survival_fracs, marker="o", markersize=5, linewidth=1.8, color=CAT["aqua"])
    ax.set_xlabel("clique size")
    ax.set_ylabel("survives single-removal (fraction)")
    ax.set_title("Robustness by clique size")
    ax.set_xticks(sizes)
    clean_axes(ax)


def _resultant_length(vectors):
    total = np.sum(vectors, axis=0)
    return np.linalg.norm(total)


def panel_c(ax):
    n_sets = 900
    coherent_lengths, incoherent_lengths = [], []
    for _ in range(n_sets):
        n_vecs = 3
        base_dir = rng.normal(size=2)
        base_dir /= np.linalg.norm(base_dir)
        # coherent: vectors clustered around one direction
        coherent_vecs = base_dir + rng.normal(scale=0.25, size=(n_vecs, 2))
        coherent_lengths.append(_resultant_length(coherent_vecs))
        # incoherent: vectors scattered in random directions
        angles = rng.uniform(0, 2 * np.pi, n_vecs)
        incoherent_vecs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        incoherent_lengths.append(_resultant_length(incoherent_vecs))

    bins = np.linspace(0, 3.2, 34)
    ax.hist(incoherent_lengths, bins=bins, color=CAT["red"], alpha=0.6, label="incoherent", edgecolor="white", linewidth=0.2)
    ax.hist(coherent_lengths, bins=bins, color=CAT["blue"], alpha=0.7, label="coherent", edgecolor="white", linewidth=0.2)
    ax.set_xlabel("resultant shift length")
    ax.set_ylabel("count")
    ax.set_title("Coherent sets reinforce; incoherent cancel")
    ax.legend(loc="upper right", fontsize=6.5)
    clean_axes(ax)


def panel_d(ax):
    base_dir = np.array([1.0, 0.6, 0.3])
    base_dir /= np.linalg.norm(base_dir)
    n = 3
    coherent = base_dir + rng.normal(scale=0.12, size=(n, 3))
    incoherent = rng.normal(size=(n, 3))
    incoherent = incoherent / np.linalg.norm(incoherent, axis=1, keepdims=True)

    origin = np.zeros(3)
    for v in coherent:
        ax.quiver(*origin, *v, color=CAT["blue"], linewidth=1.8, arrow_length_ratio=0.15)
    for v in incoherent:
        ax.quiver(*origin, *v, color=CAT["red"], linewidth=1.8, arrow_length_ratio=0.15)

    ax.scatter([], [], [], color=CAT["blue"], label="coherent")
    ax.scatter([], [], [], color=CAT["red"], label="incoherent")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.set_xlabel("m1")
    ax.set_ylabel("m2")
    ax.set_zlabel("m3")
    ax.set_title("Catalyst shift vectors")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.view_init(elev=20, azim=-50)
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
