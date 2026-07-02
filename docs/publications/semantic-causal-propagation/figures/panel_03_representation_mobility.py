"""Panel 3: Representation Mobility (Theorem 5.2).

A. Representation mean vs target alignment, across random fibres (identity line).
B. Off-shell component fraction vs representation dimension N.
C. Committed-record delta under representation switching (spike at 0).
D. 3D scatter: three-component representations (s1,s2,s3) on the mean-recovery plane.
"""

import os
import sys

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation"))

from style import CAT, MUTED_INK, clean_axes, clean_axes_3d, letter_label, new_panel

from scp.representation import committed_record_after_switch, representation_mean, sample_representation

import random as _random

pyrng = _random.Random(11)
rng = np.random.default_rng(11)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panel_03_representation_mobility.png")


def panel_a(ax):
    targets, means = [], []
    for _ in range(300):
        alignment = pyrng.uniform(0.01, 0.99)
        dimension = pyrng.randint(2, 8)
        comps = sample_representation(pyrng, alignment, dimension)
        targets.append(alignment)
        means.append(representation_mean(comps))

    ax.scatter(targets, means, s=10, c=CAT["blue"], alpha=0.5, linewidths=0)
    xs = np.array([0, 1])
    ax.plot(xs, xs, color=MUTED_INK, linewidth=1.0, linestyle="--")
    ax.set_xlabel("target alignment a(v, x*)")
    ax.set_ylabel("representation mean")
    ax.set_title("Mean recovers target exactly")
    clean_axes(ax)


def panel_b(ax):
    dims = list(range(2, 12))
    off_shell_fracs = []
    for n in dims:
        total, off_shell = 0, 0
        for _ in range(400):
            alignment = pyrng.uniform(0.05, 0.95)
            comps = sample_representation(pyrng, alignment, n)
            for c in comps:
                total += 1
                if not (0.0 < c <= 1.0):
                    off_shell += 1
        off_shell_fracs.append(off_shell / total)

    ax.plot(dims, off_shell_fracs, marker="o", markersize=4.5, color=CAT["aqua"], linewidth=1.8)
    ax.set_xlabel("representation dimension N")
    ax.set_ylabel("off-shell component fraction")
    ax.set_title("Freedom grows with dimension")
    clean_axes(ax)


def panel_c(ax):
    deltas = []
    for _ in range(2000):
        record_before = pyrng.randint(0, 5000)
        record_after = committed_record_after_switch(record_before)
        deltas.append(record_after - record_before)

    ax.hist(deltas, bins=[-.5, .5], color=CAT["blue"], edgecolor="white", linewidth=0.3, rwidth=0.6)
    ax.set_xlim(-2, 2)
    ax.set_xticks([-1, 0, 1])
    ax.set_xlabel("committed-record delta")
    ax.set_ylabel("count")
    ax.set_title("Switching commits no new cut")
    clean_axes(ax)


def panel_d(ax):
    alignment = 0.5
    n = 260
    pts = []
    for _ in range(n):
        s1 = pyrng.uniform(-3.0, 3.0)
        s2 = pyrng.uniform(-3.0, 3.0)
        s3 = 3 * alignment - s1 - s2
        pts.append((s1, s2, s3))

    pts = np.array(pts)
    on_shell = np.all((pts > 0) & (pts <= 1), axis=1)

    ax.scatter(
        pts[on_shell, 0], pts[on_shell, 1], pts[on_shell, 2],
        s=10, c=CAT["blue"], alpha=0.75, linewidths=0, label="on-shell",
    )
    ax.scatter(
        pts[~on_shell, 0], pts[~on_shell, 1], pts[~on_shell, 2],
        s=10, c=CAT["red"], alpha=0.55, linewidths=0, label="off-shell",
    )
    ax.set_xlabel("s1")
    ax.set_ylabel("s2")
    ax.set_zlabel("s3")
    ax.set_title("Representation fibre (mean = 0.5)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=6.5)
    ax.view_init(elev=18, azim=-48)
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
