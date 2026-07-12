#!/usr/bin/env python3
"""Summarize cross-ROI window decoding results (Aim 3.3)."""

import h5py
import numpy as np
from pathlib import Path

REF = "bipolar"
TRAIN_ROI = "AICl"
PARTNERS = ["IFGl", "STGl", "MFGl", "SMCl"]
PARTNER_LABEL = {"IFGl": "IFG", "STGl": "STG", "MFGl": "MFG", "SMCl": "SMC"}
PHASES = ["Stimulus", "Delay", "Go", "Response"]
CHANCE = {"lexicality": 0.5, "articulator": 0.25}

TASKS = [
    ("LexicalDelay", "lexicality"),
    ("PhonemeSequence", "articulator"),
]


def find_h5(partner_dir, phase):
    dtype_dirs = list(partner_dir.glob("(cross)(window)*"))
    if not dtype_dirs:
        return None
    matches = list(dtype_dirs[0].glob(f"*_proc-{phase}_desc-Repeat_*.h5"))
    return matches[0] if matches else None


def load_row(h5_path):
    with h5py.File(h5_path, "r") as f:
        scores = np.asarray(f["scores"][()], dtype=float)
        p = float(f["p_value"][()])
    acc = float(np.mean(scores))
    bad = not np.isfinite(scores).all() or scores.max() > 1.5 or scores.min() < -0.5
    return acc, p, bad


def summarize_task(task, datatype, results_root):
    root = results_root / f"{task}(cross_roi)({REF})"
    chance = CHANCE[datatype]
    rows = []
    missing = []

    for partner in PARTNERS:
        partner_dir = root / f"sub-{TRAIN_ROI}2{partner}"
        label = PARTNER_LABEL[partner]
        for phase in PHASES:
            h5 = find_h5(partner_dir, phase) if partner_dir.is_dir() else None
            if h5 is None:
                missing.append(f"{label}-{phase}")
                continue
            acc, p, bad = load_row(h5)
            rows.append({
                "partner": label,
                "phase": phase,
                "acc": acc,
                "p": p,
                "sig": (p < 0.05) and not bad,
                "bad": bad,
            })

    return rows, missing, chance


def main():
    results_root = Path("results")
    if not results_root.is_dir():
        results_root = Path(__file__).resolve().parents[1] / "results"

    total_sig = 0
    total_valid = 0

    for task, datatype in TASKS:
        rows, missing, chance = summarize_task(task, datatype, results_root)
        n_sig = sum(r["sig"] for r in rows)
        n_valid = len(rows)
        total_sig += n_sig
        total_valid += n_valid

        print("=" * 60)
        print(f"{task} | {datatype} | chance={chance}")
        print("=" * 60)
        if missing:
            print(f"MISSING ({len(missing)}): {', '.join(missing)}")
        print(f"Significant (p<0.05, valid): {n_sig} / {n_valid}")
        print()
        print(f"{'Partner':<8} {'Phase':<10} {'Acc':>6} {'p':>8} {'Sig':>4}")
        print("-" * 40)
        for r in rows:
            flag = "YES" if r["sig"] else ("BAD" if r["bad"] else "no")
            print(f"{r['partner']:<8} {r['phase']:<10} {r['acc']:>6.3f} {r['p']:>8.4f} {flag:>4}")
        print()
        print("By partner (any phase sig):")
        for label in ["IFG", "STG", "MFG", "SMC"]:
            sig_phases = [r["phase"] for r in rows if r["partner"] == label and r["sig"]]
            print(f"  {label}: {', '.join(sig_phases) if sig_phases else 'none'}")
        print()

    n_h5 = sum(
        1 for task, _ in TASKS
        for _ in (results_root / f"{task}(cross_roi)({REF})").rglob("*.h5")
        if "(cross)(window)" in str(_)
    ) if results_root.is_dir() else 0
    print(f"Total h5 files: {n_h5}")
    print(f"Total significant across both tasks: {total_sig} / {total_valid}")


if __name__ == "__main__":
    main()
