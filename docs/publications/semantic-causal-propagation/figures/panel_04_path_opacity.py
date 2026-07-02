"""Panel 4: Path Opacity (Theorem 6.4).

A. Endpoint invariant (target min cut) across many interior-permutation variants (flat line).
B. Interior edit distance vs a fixed reference path (variation), overlaid against the flat invariant.
C. Distribution of Delta in the endpoint invariant across interior variants (spike at 0).
D. 3D: two propagations sharing seed/target but different interiors, plotted as trajectories over (step, claim-index).
"""

import os
import sys

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation"))

from style import CAT, MUTED_INK, clean_axes, clean_axes_3d, letter_label, new_panel

from scp.graph import random_contact_graph
from scp.propagation import endpoint_invariants, random_interior_variant

import random as _random

pyrng = _random.Random(19)
rng = np.random.default_rng(19)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panel_04_path_opacity.png")

N_CLAIMS = 14
FLOOR = 0.2
GRAPH = random_contact_graph(pyrng, N_CLAIMS, FLOOR, edge_prob=0.35)
SEED_CLAIM = GRAPH.claims[0]
TARGET_CLAIM = GRAPH.claims[-1]

N_VARIANTS = 46
VARIANTS = []
for _ in range(N_VARIANTS):
    interior_len = pyrng.randint(0, min(6, N_CLAIMS - 2))
    prop = random_interior_variant(pyrng, SEED_CLAIM, TARGET_CLAIM, GRAPH.claims, interior_len)
    VARIANTS.append(prop)

INVARIANTS = [endpoint_invariants(GRAPH, p) for p in VARIANTS]


def panel_a(ax):
    xs = np.arange(len(INVARIANTS))
    ys = [inv["target_min_cut"] for inv in INVARIANTS]
    ax.scatter(xs, ys, s=14, c=CAT["blue"], alpha=0.75, linewidths=0)
    ax.axhline(ys[0], color=MUTED_INK, linewidth=1.0, linestyle="--")
    ax.set_ylim(min(ys) - 0.4, max(ys) + 0.4)
    ax.set_xlabel("interior variant index")
    ax.set_ylabel("target min-cut value")
    ax.set_title("Endpoint invariant across interiors")
    clean_axes(ax)


def panel_b(ax):
    reference = VARIANTS[0].interior()
    edit_distances = []
    for p in VARIANTS:
        interior = p.interior()
        # simple edit distance proxy: symmetric difference of interior sets
        d = len(set(interior).symmetric_difference(set(reference)))
        edit_distances.append(d)

    xs = np.arange(len(edit_distances))
    ax.bar(xs, edit_distances, color=CAT["aqua"], width=0.7)
    ax.set_xlabel("interior variant index")
    ax.set_ylabel("edit distance from reference")
    ax.set_title("Interior varies freely")
    clean_axes(ax)


def panel_c(ax):
    ref = INVARIANTS[0]["target_min_cut"]
    deltas = [inv["target_min_cut"] - ref for inv in INVARIANTS]
    ax.hist(deltas, bins=12, color=CAT["blue"], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("delta in endpoint invariant")
    ax.set_ylabel("count")
    ax.set_title("No interior rearrangement moves it")
    clean_axes(ax)


def panel_d(ax):
    # Two distinct interior variants sharing the same seed/target, plotted as
    # trajectories in (step, claim-index, variant-id) space.
    claim_index = {c: i for i, c in enumerate(GRAPH.claims)}
    colors = [CAT["blue"], CAT["red"]]
    for vi, idx in enumerate([2, 7]):
        prop = VARIANTS[idx % len(VARIANTS)]
        steps = np.arange(len(prop.claims))
        idxs = np.array([claim_index[c] for c in prop.claims])
        zs = np.full_like(steps, vi, dtype=float)
        ax.plot(steps, idxs, zs, color=colors[vi], linewidth=2.0, marker="o", markersize=4)

    ax.set_xlabel("step")
    ax.set_ylabel("claim index")
    ax.set_zlabel("trajectory id")
    ax.set_title("Two propagations, shared endpoints")
    ax.view_init(elev=20, azim=-56)
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
