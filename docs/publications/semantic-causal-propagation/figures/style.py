"""Shared plotting style for the semantic-causal-propagation panel figures.

Palette and mark discipline follow the project's data-visualization method:
- White chart surface, minimal text, thin marks, recessive gridlines.
- Categorical hues used in FIXED order (never cycled, never reassigned).
- Sequential (magnitude) uses a single hue, light -> dark.
- Diverging (polarity) uses the blue/red pair with a neutral gray midpoint.
- No conceptual/text/table panels: every chart plots real computed data.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---- Surfaces & ink -------------------------------------------------------
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# ---- Categorical palette (fixed order; validated CVD >= 12, worst 24.2) --
CAT = {
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "magenta": "#e87ba4",
    "orange": "#eb6834",
}
CAT_ORDER = ["blue", "aqua", "yellow", "green", "violet", "red", "magenta", "orange"]
CAT_LIST = [CAT[k] for k in CAT_ORDER]

# ---- Sequential ramp (blue, light -> dark) --------------------------------
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

# ---- Diverging pair (blue <-> red), neutral midpoint ----------------------
DIV_NEG = "#2a78d6"
DIV_MID = "#f0efec"
DIV_POS = "#e34948"

# ---- Status (fixed, never repurposed as a series color) -------------------
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

FONT = "system-ui, -apple-system, Segoe UI, Arial, sans-serif"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
            "text.color": PRIMARY_INK,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": SECONDARY_INK,
            "axes.titlecolor": PRIMARY_INK,
            "xtick.color": MUTED_INK,
            "ytick.color": MUTED_INK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def clean_axes(ax) -> None:
    """Recessive spines/grid; thin baseline; no top/right spines (2D axes)."""
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.7)


def clean_axes_3d(ax) -> None:
    """Minimal chrome for a 3D axis: light panes, thin gridlines, muted ticks."""
    ax.xaxis.pane.set_facecolor(SURFACE)
    ax.yaxis.pane.set_facecolor(SURFACE)
    ax.zaxis.pane.set_facecolor(SURFACE)
    ax.xaxis.pane.set_edgecolor(GRIDLINE)
    ax.yaxis.pane.set_edgecolor(GRIDLINE)
    ax.zaxis.pane.set_edgecolor(GRIDLINE)
    ax.xaxis.pane.set_alpha(1.0)
    ax.yaxis.pane.set_alpha(1.0)
    ax.zaxis.pane.set_alpha(1.0)
    ax.xaxis._axinfo["grid"]["color"] = GRIDLINE
    ax.yaxis._axinfo["grid"]["color"] = GRIDLINE
    ax.zaxis._axinfo["grid"]["color"] = GRIDLINE
    ax.xaxis._axinfo["grid"]["linewidth"] = 0.5
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.5
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.5
    ax.tick_params(labelsize=6.5, colors=MUTED_INK, length=2)
    ax.xaxis.label.set_size(7.5)
    ax.yaxis.label.set_size(7.5)
    ax.zaxis.label.set_size(7.5)
    ax.xaxis.label.set_color(SECONDARY_INK)
    ax.yaxis.label.set_color(SECONDARY_INK)
    ax.zaxis.label.set_color(SECONDARY_INK)


def new_panel(figsize=(15.5, 3.7)):
    """A 1x4 panel row on a white surface, returning (fig, axes)."""
    apply_style()
    fig = plt.figure(figsize=figsize, dpi=200)
    return fig


def letter_label(ax, letter: str, is_3d: bool = False):
    kwargs = dict(
        fontsize=9,
        fontweight="bold",
        color=PRIMARY_INK,
        ha="left",
        va="top",
    )
    if is_3d:
        ax.text2D(-0.06, 1.06, letter, transform=ax.transAxes, **kwargs)
    else:
        ax.text(-0.14, 1.10, letter, transform=ax.transAxes, **kwargs)
