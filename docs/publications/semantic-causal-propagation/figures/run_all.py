#!/usr/bin/env python3
"""Generate all 8 panel figures for semantic-causal-propagation.tex.

Each panel is a 1x4 row of real, computed charts (never conceptual/text/
table) on a white surface, with at least one 3D chart per panel, following
the project's data-visualization method (validated categorical palette,
fixed hue order, minimal chrome).

Usage:
    python run_all.py
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PANELS = [
    "panel_01_floor",
    "panel_02_individuation",
    "panel_03_representation_mobility",
    "panel_04_path_opacity",
    "panel_05_multiplicative_law",
    "panel_06_saturation",
    "panel_07_coherence",
    "panel_08_closure_scheduler",
]


def main():
    for name in PANELS:
        module = importlib.import_module(name)
        module.main()
    print(f"\nGenerated {len(PANELS)} panels.")


if __name__ == "__main__":
    main()
