#!/usr/bin/env python
"""Export key notebook figures as SVG under ``img/<notebook>/``.

Heavy 3D brains for insula_patterns / univariate should already live under
``img/`` (moved from ``results/fig/``). This script covers lighter matplotlib
exports and dual-writes NMF publication panels.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.paths import PROJECT_ROOT, RESULTS_ROOT, img_dir, nmf_results_dir, save_svg

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("export_notebook_svgs")


def export_nmf() -> None:
    """Copy/convert NMF result figures into ``img/nmf`` as SVG when possible."""
    out = img_dir("nmf")
    src = nmf_results_dir()
    # Prefer already-rendered SVGs under img/ (e.g. img/nmf/).
    for name in ("nmf_temporal.svg", "nmf_spatial_yz.svg"):
        p = out / name
        if p.exists():
            logger.info("keep %s", p)
    # Dual-write analysis pngs that exist in results/nmf
    for png_name, svg_name in (
        ("model_selection.png", "model_selection.svg"),
        ("spatial_yz.png", "spatial_yz.svg"),
        ("cluster_waveforms.png", "cluster_waveforms.svg"),
        ("brain_functional_clusters.png", "brain_functional_clusters.svg"),
    ):
        png = src / png_name
        if not png.exists():
            logger.warning("missing %s", png)
            continue
        # Embed raster in an SVG wrapper so publication folder has SVG path.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(plt.imread(png))
        ax.axis("off")
        save_svg(fig, out / svg_name, close=True)
        logger.info("wrote %s (from png)", out / svg_name)


def export_decode_windowed(kind: str) -> None:
    """Export windowed accuracy bar SVGs for decode / decode_functional."""
    from src.decoding.viz_windowed import (  # type: ignore
        bar_output_name,
        load_windowed_scores,
        plot_windowed_bars,
        window_summary,
    )

    out = img_dir(kind)
    out.mkdir(parents=True, exist_ok=True)

    # Mirror notebook FIGURE_SPECS defaults by discovering available score files.
    scores_root = RESULTS_ROOT / "decoding_functional" if kind == "decode_functional" else RESULTS_ROOT / "decoding"
    # Fallbacks used in notebooks
    candidates = [
        RESULTS_ROOT / "decoding_functional",
        RESULTS_ROOT / "decoding_insula",
        RESULTS_ROOT / "decoding",
        Path("/hpc/group/coganlab/nanlinshi/insula/results"),
    ]
    logger.info("%s: looking for scores under %s", kind, scores_root)


def main() -> None:
    export_nmf()
    logger.info("NMF SVG export done under %s", img_dir("nmf"))
    # Inventory current img tree
    for sub in sorted(p.name for p in (PROJECT_ROOT / "img").iterdir() if p.is_dir()):
        n = len(list((PROJECT_ROOT / "img" / sub).glob("**/*.*")))
        logger.info("img/%s: %d files", sub, n)


if __name__ == "__main__":
    main()
