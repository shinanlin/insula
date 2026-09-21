#!/usr/bin/env python3
"""Summarize window-decode param-sweep H5 trees into a CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

TAG_RE = re.compile(
    r"^var(?P<var>[0-9p]+)C(?P<C>[0-9p]+)t(?P<tmin>-?[0-9pm]+)to(?P<tmax>-?[0-9pm]+)$"
)


def _parse_float_tag(token: str) -> float:
    s = token.replace("m", "-").replace("p", ".")
    return float(s)


def summarize(root: Path, *, sig_p: float = 0.05) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for h5_path in sorted(root.rglob("*.h5")):
        tag = next(
            (part for part in h5_path.parts if TAG_RE.match(part)),
            None,
        )
        if tag is None:
            continue
        match = TAG_RE.match(tag)
        assert match is not None
        variance = _parse_float_tag(match.group("var"))
        C = _parse_float_tag(match.group("C"))
        tmin = _parse_float_tag(match.group("tmin"))
        tmax = _parse_float_tag(match.group("tmax"))

        # .../sub-X/(decode)feat/file.h5
        subject = h5_path.parts[-3].removeprefix("sub-")
        datatype = h5_path.parts[-2].removeprefix("(decode)")
        name = h5_path.name
        description = "Repeat"
        if "_desc-" in name:
            description = name.split("_desc-")[1].split("_")[0]

        with h5py.File(h5_path, "r") as handle:
            accuracy = float(handle["accuracy_stable"][()])
            p_value = float(handle["p_value"][()])
            fold_acc = np.asarray(handle["accuracy"][()], dtype=float)
            classes = np.asarray(handle["classes"][()])
            n_classes = int(len(classes))

        chance = 1.0 / n_classes
        rows.append(
            {
                "subject": subject,
                "description": description,
                "datatype": datatype,
                "window": f"[{tmin:g}, {tmax:g}]",
                "tmin": tmin,
                "tmax": tmax,
                "variance": variance,
                "C": C,
                "accuracy": accuracy,
                "p_value": p_value,
                "chance": chance,
                "n_classes": n_classes,
                "fold_std": float(np.std(fold_acc, ddof=1)) if fold_acc.size > 1 else 0.0,
                "significant": bool(p_value < sig_p),
                "delta_chance": accuracy - chance,
                "path": str(h5_path),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Sweep results root (contains var*C*t*to* folders)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: <root>_summary.csv next to root)",
    )
    parser.add_argument("--sig-p", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = summarize(args.root, sig_p=args.sig_p)
    out = args.out
    if out is None:
        out = args.root.parent / f"{args.root.name}_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    n_sig = int(frame["significant"].sum()) if not frame.empty else 0
    print(f"Wrote {len(frame)} rows ({n_sig} significant) -> {out}")


if __name__ == "__main__":
    main()
