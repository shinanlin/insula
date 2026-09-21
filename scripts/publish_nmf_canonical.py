#!/usr/bin/env python3
"""Verify canonical NMF artifacts under ``results/nmf/``.

``scripts/plot_nmf_concat_phases.py`` already writes assignments and
``nmf_manifest.json`` to the flat ``results/nmf/`` root. This script checks that
``channel_assignments.csv`` exists and refreshes the manifest timestamp.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.nmf.waveform_analysis import CLUSTER_ORDER, TASKS
from src.paths import nmf_assignments_path, nmf_chosen_k, nmf_results_dir


WINDOWS = "stimulus=(0,1); delay=(0,1); go=(0,1); response=(0,0.5)"


def publish(*, k: int | None = None) -> Path:
    dest = nmf_assignments_path()
    if not dest.is_file():
        raise FileNotFoundError(
            f"Missing NMF assignments: {dest}. "
            "Run scripts/plot_nmf_concat_phases.py first."
        )

    out_root = nmf_results_dir()
    n_rows = sum(1 for _ in dest.open()) - 1
    publish_k = int(k) if k is not None else nmf_chosen_k(default=3)
    manifest = {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "assignments": str(dest),
        "construction": "concat_phases",
        "windows": WINDOWS,
        "k": publish_k,
        "tasks": list(TASKS),
        "condition": "Repeat",
        "n_electrodes": n_rows,
        "cluster_names": list(CLUSTER_ORDER),
    }
    (out_root / "nmf_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Canonical assignments OK ({n_rows} electrodes) → {dest}")
    print(f"Manifest → {out_root / 'nmf_manifest.json'}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="k recorded in nmf_manifest.json (default: from existing manifest or 3)",
    )
    args = parser.parse_args()
    publish(k=args.k)


if __name__ == "__main__":
    main()
