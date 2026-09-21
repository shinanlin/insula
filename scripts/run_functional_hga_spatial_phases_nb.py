#!/usr/bin/env python3
"""Execute functional_hga.ipynb cells needed for the phase brain spatial figure.

Runs config + HGA load + spatial render/save only (skips parcel pies / heatmaps).
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("PYVISTA_USE_PANEL", "false")

import nbformat

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
# Notebook uses ROOT = Path('..'); run with cwd = notebooks/
NB_DIR = PROJECT / "notebooks"
NB_PATH = NB_DIR / "functional_hga.ipynb"
os.chdir(NB_DIR)

# Config, load HGA, spatial table, brain helpers, figure save
CELL_INDICES = (1, 2, 4, 12, 13, 14)


def _strip_ipython(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        # Keep notebook backend: with PYVISTA_OFF_SCREEN it renders headless.
        # Do NOT rewrite to 'pyvista' — that resolves to pyvistaqt and needs a display.
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\bdisplay\(", "print(", text)
    return text


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)
    ns: dict = {"__name__": "__main__", "__file__": str(NB_PATH)}
    for idx in CELL_INDICES:
        cell = nb.cells[idx]
        if cell.cell_type != "code":
            continue
        src = _strip_ipython("".join(cell.source))
        if not src.strip():
            continue
        print(f"--- executing notebook cell {idx} ---", flush=True)
        ast.parse(src)
        exec(compile(src, f"{NB_PATH.name}:cell{idx}", "exec"), ns)
    out = ns.get("SPATIAL_PHASES_OUT")
    print(f"Done. Spatial phases figure: {out}", flush=True)


if __name__ == "__main__":
    main()
