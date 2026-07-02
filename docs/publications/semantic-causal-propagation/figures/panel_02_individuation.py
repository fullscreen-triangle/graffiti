"""Panel 2: Individuation by Negation (Theorem 4.2).

A. |U| vs |co(U)| across random subsets (conservation line |U|+|coU|=|V|).
B. Double-complement error histogram: co(co(U)) == U (spike at 0).
C. Fibre size distribution of a synthetic decoder Dec/Proj pair.
D. 3D scatter: (|U|, |co(U)|, |co(co(U))|) -- all points on the identity plane.
"""

import os
import sys

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation"))

from style import CAT, MUTED_INK, clean_axes, clean_axes_3d, letter_label, new_panel

from scp.individuation import build_synthetic_decoder, complement

import random as _random

pyrng = _random.Random(7)
rng = np.random.default_rng(7)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panel_02_individuation.png")

MEDIUM_SIZE = 24
WHOLE = frozenset(f"c{i}" for i in range(MEDIUM_SIZE))


def panel_a(ax):
    sizes_u, sizes_co = [], []
    for _ in range(260):
        k = pyrng.randint(1, MEDIUM_SIZE - 1)
        u = frozenset(pyrng.sample(sorted(WHOLE), k))
        co = complement(WHOLE, u)
        sizes_u.append(len(u))
        sizes_co.append(len(co))

    sizes_u = np.array(sizes_u) + rng.normal(0, 0.12, len(sizes_u))
    sizes_co = np.array(sizes_co) + rng.normal(0, 0.12, len(sizes_co))
    ax.scatter(sizes_u, sizes_co, s=10, c=CAT["blue"], alpha=0.5, linewidths=0)
    xs = np.array([0, MEDIUM_SIZE])
    ax.plot(xs, MEDIUM_SIZE - xs, color=MUTED_INK, linewidth=1.0, linestyle="--")
    ax.set_xlabel("|U|")
    ax.set_ylabel("|co(U)|")
    ax.set_title("Conservation: |U| + |co(U)| = |V|")
    clean_axes(ax)


def panel_b(ax):
    errors = []
    for _ in range(500):
        k = pyrng.randint(1, MEDIUM_SIZE - 1)
        u = frozenset(pyrng.sample(sorted(WHOLE), k))
        co_co = complement(WHOLE, complement(WHOLE, u))
        errors.append(len(u.symmetric_difference(co_co)))

    ax.hist(errors, bins=[-.5, .5, 1.5, 2.5], color=CAT["blue"], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("symmetric difference |U xor co(co(U))|")
    ax.set_ylabel("count")
    ax.set_xticks([0, 1, 2])
    ax.set_title("Involution error (500 trials)")
    clean_axes(ax)


def panel_c(ax):
    decoder = build_synthetic_decoder(pyrng, n_claims=30, n_queries_per_claim=1)
    # vary fibre sizes by re-adding extra queries per claim non-uniformly
    mapping = dict(decoder.mapping)
    claims = sorted(set(mapping.values()))
    extra_counts = []
    for i, c in enumerate(claims):
        n_extra = pyrng.randint(0, 14)
        for q in range(n_extra):
            mapping[f"extra_{c}_{q}"] = c
        extra_counts.append(n_extra + 1)

    ax.hist(extra_counts, bins=range(0, 17), color=CAT["aqua"], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("|Proj(v)| (fibre size)")
    ax.set_ylabel("claim count")
    ax.set_title("Query fibre sizes per claim")
    clean_axes(ax)


def panel_d(ax):
    us, cos, cocos = [], [], []
    for _ in range(180):
        k = pyrng.randint(1, MEDIUM_SIZE - 1)
        u = frozenset(pyrng.sample(sorted(WHOLE), k))
        co = complement(WHOLE, u)
        co_co = complement(WHOLE, co)
        us.append(len(u))
        cos.append(len(co))
        cocos.append(len(co_co))

    us = np.array(us, dtype=float) + rng.normal(0, 0.15, len(us))
    cos = np.array(cos, dtype=float) + rng.normal(0, 0.15, len(cos))
    cocos = np.array(cocos, dtype=float) + rng.normal(0, 0.15, len(cocos))

    ax.scatter(us, cos, cocos, s=8, c=CAT["violet"], alpha=0.6, linewidths=0)
    ax.set_xlabel("|U|")
    ax.set_ylabel("|co(U)|")
    ax.set_zlabel("|co(co(U))|")
    ax.set_title("(|U|, |co(U)|, |co(co(U))|)")
    ax.view_init(elev=20, azim=-52)
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
