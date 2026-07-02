"""Panel 5: Multiplicative Catalytic Law (Theorem 7.3).

A. Measured composite power vs closed form 1 - prod(1-kappa_i) (identity diagonal).
B. Cumulative power under repetition of one catalyst, several kappa values.
C. Residual above-floor distance vs chain length (log-y), several kappa values.
D. 3D surface: composite power over (kappa_1, kappa_2).
"""

import os
import sys

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation"))

from style import CAT, CAT_LIST, MUTED_INK, clean_axes, clean_axes_3d, letter_label, new_panel

from scp.catalysis import composite_power, repeated_power, residual_after_chain

import random as _random

pyrng = _random.Random(23)
rng = np.random.default_rng(23)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panel_05_multiplicative_law.png")


def panel_a(ax):
    predicted, measured = [], []
    for _ in range(400):
        n = pyrng.randint(1, 8)
        powers = [pyrng.uniform(0.0, 0.99) for _ in range(n)]
        pred = composite_power(powers)
        resid = residual_after_chain(1.0, powers)
        meas = 1.0 - resid
        predicted.append(pred)
        measured.append(meas)

    ax.scatter(predicted, measured, s=10, c=CAT["blue"], alpha=0.5, linewidths=0)
    xs = np.array([0, 1])
    ax.plot(xs, xs, color=MUTED_INK, linewidth=1.0, linestyle="--")
    ax.set_xlabel("predicted composite power")
    ax.set_ylabel("measured composite power")
    ax.set_title("Multiplicative law, 400 chains")
    clean_axes(ax)


def panel_b(ax):
    kappas = [0.1, 0.3, 0.5, 0.7]
    ns = np.arange(0, 26)
    for kappa, color_key in zip(kappas, ["blue", "aqua", "yellow", "red"]):
        powers = [repeated_power(kappa, n) for n in ns]
        ax.plot(ns, powers, marker="o", markersize=3, linewidth=1.6, color=CAT[color_key], label=f"k={kappa}")

    ax.set_xlabel("number of applications n")
    ax.set_ylabel("cumulative power")
    ax.set_title("Diminishing returns")
    ax.legend(loc="lower right", ncol=1)
    clean_axes(ax)


def panel_c(ax):
    kappas = [0.1, 0.2, 0.3, 0.5, 0.7]
    ns = np.arange(0, 41)
    colors = ["blue", "aqua", "yellow", "red", "violet"]
    for kappa, color_key in zip(kappas, colors):
        residuals = [(1 - kappa) ** n for n in ns]
        ax.plot(ns, residuals, linewidth=1.7, color=CAT[color_key], label=f"k={kappa}")

    ax.set_yscale("log")
    ax.set_xlabel("chain length n")
    ax.set_ylabel("residual above-floor distance")
    ax.set_title("Geometric decay of residual")
    ax.legend(loc="upper right", fontsize=6.2, ncol=1)
    clean_axes(ax)


def panel_d(ax):
    k1 = np.linspace(0, 1, 50)
    k2 = np.linspace(0, 1, 50)
    K1, K2 = np.meshgrid(k1, k2)
    Composite = 1 - (1 - K1) * (1 - K2)

    surf = ax.plot_surface(K1, K2, Composite, cmap="Blues", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("kappa_1")
    ax.set_ylabel("kappa_2")
    ax.set_zlabel("composite power")
    ax.set_title("Two-catalyst composite surface")
    ax.view_init(elev=24, azim=-56)
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
