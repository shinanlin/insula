#!/usr/bin/env python3
"""Export the Fig 4 supplement Haufe brain grid (articulator vs lexicality)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.decoding.viz_insula_patterns import (
    load_assignments,
    plot_fig4_articulator_lexicality_brains,
)
from src.paths import PROJECT_ROOT, img_dir, save_svg
from src.univariate.viz_mean import BrainSurfaceContext


def main() -> None:
    for ttf in (Path(sys.prefix) / "fonts").glob("arial*.ttf"):
        font_manager.fontManager.addfont(str(ttf))
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    assignments = load_assignments(PROJECT_ROOT)
    fig, counts = plot_fig4_articulator_lexicality_brains(
        PROJECT_ROOT,
        assignments,
        ctx=BrainSurfaceContext(),
        fontsize=7,
    )
    out = save_svg(
        fig,
        img_dir("fig4") / "fig4_pattern_articulator_lexicality",
        close=True,
    )
    print("Wrote", out)
    print("\n".join(counts))


if __name__ == "__main__":
    main()
