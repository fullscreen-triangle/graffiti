"""Panel 6: Saturation Dichotomy (Corollary 7.5).

A. Residual vs horizon (log-y) for divergent-sum vs convergent-sum power sequences.
B. Log-residual growth vs horizon (linear), showing unbounded decline for divergent sums.
C. Partial sum of powers vs horizon for the four sequence types.
D. 3D surface: log-residual over (horizon, kappa) for a constant-power family.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import CAT, clean_axes, clean_axes_3d, letter_label, new_panel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panel_06_saturation.png")

N_STEPS = 500
HORIZONS = np.arange(1, N_STEPS + 1)

SEQUENCES = {
    "constant (div.)": ([0.1] * N_STEPS, "blue"),
    "harmonic (div.)": ([1.0 / (i + 2) for i in range(N_STEPS)], "aqua"),
    "geometric (conv.)": ([2.0 ** (-(i + 1)) for i in range(N_STEPS)], "red"),
    "inverse-square (conv.)": ([1.0 / (i + 2) ** 2 for i in range(N_STEPS)], "violet"),
}


def log_residual_series(powers):
    log_res = np.cumsum([np.log1p(-min(k, 0.999999)) for k in powers])
    return log_res


def panel_a(ax):
    for name, (powers, color_key) in SEQUENCES.items():
        log_res = log_residual_series(powers)
        residual = np.exp(np.clip(log_res, -700, 0))
        ax.plot(HORIZONS, residual, linewidth=1.6, color=CAT[color_key], label=name)

    ax.set_yscale("log")
    ax.set_xlabel("horizon n")
    ax.set_ylabel("residual above-floor distance")
    ax.set_title("Residual decay by sequence type")
    ax.legend(loc="lower left", fontsize=6.0)
    clean_axes(ax)


def panel_b(ax):
    for name, (powers, color_key) in SEQUENCES.items():
        log_res = log_residual_series(powers)
        ax.plot(HORIZONS, log_res, linewidth=1.6, color=CAT[color_key], label=name)

    ax.set_xlabel("horizon n")
    ax.set_ylabel("log-residual")
    ax.set_title("Divergent sums: unbounded decline")
    ax.legend(loc="lower left", fontsize=6.0)
    clean_axes(ax)


def panel_c(ax):
    for name, (powers, color_key) in SEQUENCES.items():
        partial_sums = np.cumsum(powers)
        ax.plot(HORIZONS, partial_sums, linewidth=1.6, color=CAT[color_key], label=name)

    ax.set_xlabel("horizon n")
    ax.set_ylabel("partial sum of kappa_i")
    ax.set_title("Partial sums: divergent vs. bounded")
    ax.legend(loc="upper left", fontsize=6.0)
    clean_axes(ax)


def panel_d(ax):
    kappas = np.linspace(0.02, 0.5, 40)
    horizons = np.linspace(1, 200, 40)
    Kappa, Horizon = np.meshgrid(kappas, horizons)
    LogResidual = Horizon * np.log1p(-Kappa)

    surf = ax.plot_surface(Kappa, Horizon, LogResidual, cmap="Blues_r", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("kappa (constant)")
    ax.set_ylabel("horizon n")
    ax.set_zlabel("log-residual")
    ax.set_title("Log-residual surface")
    ax.view_init(elev=22, azim=-52)
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
