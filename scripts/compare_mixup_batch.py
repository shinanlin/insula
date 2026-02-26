#!/usr/bin/env python3
"""Batch comparison of row-level mixup vs per-feature feature_mixup across ROIs.

Processes multiple ROI h5 files, runs CV decoding with both NaN-filling strategies,
saves per-ROI results to JSON, and generates a comprehensive cross-ROI report.
"""

import rootutils
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
import json
import os
import time as _time
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import get_scorer
from sklearn.base import clone
from mne.decoding import Vectorizer
from ieeg.calc.oversample import MinimumNaNSplit
from ieeg.calc.fast import mixup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── NaN-filling strategies ──────────────────────────────────────────────────

def feature_mixup(x_cls, alpha=1.0, rng=None):
    """Per-feature NaN interpolation."""
    if rng is None:
        rng = np.random.RandomState()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.RandomState(rng)

    n_samples = x_cls.shape[0]
    x_2d = x_cls.reshape(n_samples, -1)
    nan_mask = np.isnan(x_2d)
    if not nan_mask.any():
        return

    for f in range(x_2d.shape[1]):
        col = x_2d[:, f]
        nan_idx = np.where(nan_mask[:, f])[0]
        if len(nan_idx) == 0:
            continue
        valid_idx = np.where(~nan_mask[:, f])[0]
        if len(valid_idx) == 0:
            x_2d[nan_idx, f] = rng.normal(0, 1, len(nan_idx))
            continue
        if len(valid_idx) == 1:
            x_2d[nan_idx, f] = col[valid_idx[0]]
            continue
        n_nan = len(nan_idx)
        idx1 = rng.choice(valid_idx, size=n_nan, replace=True)
        idx2 = rng.choice(valid_idx, size=n_nan, replace=True)
        lam = rng.beta(alpha, alpha, size=n_nan)
        lam = np.maximum(lam, 1.0 - lam)
        x_2d[nan_idx, f] = lam * col[idx1] + (1.0 - lam) * col[idx2]


def sample_fold_original(X, y, train_idx, test_idx):
    """Original sample_fold using row-level mixup."""
    X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
    y_train, y_test = y[train_idx].copy(), y[test_idx].copy()
    for cls in np.unique(y_train):
        idx = y_train == cls
        x_cls = X_train[idx]
        mixup(x_cls, obs_axis=0, rng=42)
        X_train[idx] = x_cls
    m = np.isnan(X_test)
    if m.any():
        X_test[m] = np.random.normal(0, 1, int(np.sum(m)))
    return X_train, X_test, y_train, y_test


def sample_fold_feature(X, y, train_idx, test_idx):
    """New sample_fold using per-feature feature_mixup."""
    X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
    y_train, y_test = y[train_idx].copy(), y[test_idx].copy()
    for cls in np.unique(y_train):
        idx = y_train == cls
        x_cls = X_train[idx]
        feature_mixup(x_cls, alpha=1.0, rng=42)
        X_train[idx] = x_cls
    m = np.isnan(X_test)
    if m.any():
        X_test[m] = np.random.normal(0, 1, int(np.sum(m)))
    return X_train, X_test, y_train, y_test


# ── CV decoding ─────────────────────────────────────────────────────────────

def run_cv_decoding(X, y, cv, pipeline, sample_fold_fn, n_permutations=50,
                    random_state=42, method_name=""):
    scorer = get_scorer("accuracy")
    splits = list(cv.split(X, y))
    n_total = len(splits)
    fold_scores, perm_scores_all = [], []
    t_start = _time.time()
    for fold_idx, (tr, te) in enumerate(splits):
        t_fold = _time.time()
        dec = clone(pipeline)
        X_train, X_test, y_train, y_test = sample_fold_fn(X, y, tr, te)
        dec.fit(X_train, y_train)
        score = scorer(dec, X_test, y_test)
        fold_scores.append(score)
        rng_fold = np.random.RandomState(random_state + fold_idx)
        seeds = rng_fold.randint(0, 2**31 - 1, size=n_permutations)
        fold_perm = []
        for s in seeds:
            r = np.random.RandomState(s)
            y_perm = y_train.copy()
            r.shuffle(y_perm)
            dec_p = clone(pipeline)
            dec_p.fit(X_train, y_perm)
            fold_perm.append(scorer(dec_p, X_test, y_test))
        perm_scores_all.append(fold_perm)
        elapsed = _time.time() - t_start
        fold_time = _time.time() - t_fold
        logger.info(f"    [{method_name}] Fold {fold_idx+1}/{n_total}: "
                    f"acc={score:.3f}, fold_time={fold_time:.0f}s, "
                    f"total_elapsed={elapsed:.0f}s")
        sys.stdout.flush()
    return fold_scores, np.array(perm_scores_all)


# ── Process one ROI ─────────────────────────────────────────────────────────

def process_one_roi(h5_path, roi_name, pipeline, tmin, tmax,
                    n_folds, n_repeats, n_permutations):
    """Run both methods on one ROI and return results dict."""
    logger.info("=" * 70)
    logger.info(f"  Processing ROI: {roi_name}")
    logger.info(f"  File: {h5_path}")
    logger.info("=" * 70)

    with h5py.File(h5_path, "r") as f:
        X = f["X"][()]
        y = f["y"][()]
        fs = f.attrs["fs"]

    # Crop
    t_start = -1.0
    si = int(fs * (tmin - t_start))
    ei = int(fs * (tmax - t_start))
    X = X[:, :, si:ei]

    if y.dtype.kind in ("S", "O"):
        y = np.array([s.decode() if isinstance(s, bytes) else s for s in y])

    # NaN info
    total = X.size
    n_nan = int(np.isnan(X).sum())
    pct_nan = 100 * n_nan / total
    n_nan_trials = int(np.isnan(X).any(axis=(1, 2)).sum())
    n_nan_ch = int(np.isnan(X).any(axis=(0, 2)).sum())
    nan_heatmap = np.isnan(X).any(axis=2)

    logger.info(f"  X: {X.shape}, NaN: {pct_nan:.1f}%, "
                f"NaN trials: {n_nan_trials}/{X.shape[0]}, "
                f"NaN channels: {n_nan_ch}/{X.shape[1]}, "
                f"Classes: {len(np.unique(y))}")

    cv = MinimumNaNSplit(n_splits=n_folds, n_repeats=n_repeats)

    # Original
    t0 = _time.time()
    scores_orig, perm_orig = run_cv_decoding(
        X, y, cv, pipeline, sample_fold_original,
        n_permutations=n_permutations, method_name=f"{roi_name}/Original")
    t_orig = _time.time() - t0
    logger.info(f"  Original: {np.mean(scores_orig):.4f} ± {np.std(scores_orig):.4f} ({t_orig:.0f}s)")

    # Feature
    t0 = _time.time()
    scores_feat, perm_feat = run_cv_decoding(
        X, y, cv, pipeline, sample_fold_feature,
        n_permutations=n_permutations, method_name=f"{roi_name}/Feature")
    t_feat = _time.time() - t0
    logger.info(f"  Feature : {np.mean(scores_feat):.4f} ± {np.std(scores_feat):.4f} ({t_feat:.0f}s)")

    t_stat, p_val = stats.ttest_rel(scores_orig, scores_feat)
    logger.info(f"  Paired t: t={t_stat:.3f}, p={p_val:.4f}")

    return {
        "roi": roi_name,
        "shape": list(X.shape),
        "n_classes": int(len(np.unique(y))),
        "pct_nan": round(pct_nan, 2),
        "n_nan_trials": n_nan_trials,
        "n_nan_channels": n_nan_ch,
        "nan_heatmap": nan_heatmap,
        "scores_orig": scores_orig,
        "scores_feat": scores_feat,
        "perm_orig": perm_orig,
        "perm_feat": perm_feat,
        "t_stat": t_stat,
        "p_val": p_val,
        "time_orig": t_orig,
        "time_feat": t_feat,
    }


# ── Cross-ROI visualization ────────────────────────────────────────────────

def plot_batch_report(all_results, save_path):
    """Generate a comprehensive multi-ROI comparison figure."""
    n_rois = len(all_results)
    colors = ["#4C72B0", "#DD8452"]
    roi_names = [r["roi"] for r in all_results]

    fig = plt.figure(figsize=(22, 20), facecolor="white")
    gs = GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.30,
                  left=0.07, right=0.95, top=0.92, bottom=0.05)

    # ── Panel 1: Grouped bar chart – mean accuracy per ROI ───────────────
    ax1 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(n_rois)
    width = 0.32
    means_o = [np.mean(r["scores_orig"]) for r in all_results]
    means_f = [np.mean(r["scores_feat"]) for r in all_results]
    sems_o = [np.std(r["scores_orig"]) / np.sqrt(len(r["scores_orig"])) for r in all_results]
    sems_f = [np.std(r["scores_feat"]) / np.sqrt(len(r["scores_feat"])) for r in all_results]

    b1 = ax1.bar(x_pos - width/2, means_o, width, yerr=sems_o, color=colors[0],
                 edgecolor="black", linewidth=0.6, capsize=4, label="Original mixup")
    b2 = ax1.bar(x_pos + width/2, means_f, width, yerr=sems_f, color=colors[1],
                 edgecolor="black", linewidth=0.6, capsize=4, label="Feature mixup")

    # annotate significance
    for i, r in enumerate(all_results):
        p = r["p_val"]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ymax = max(means_o[i] + sems_o[i], means_f[i] + sems_f[i])
        ax1.text(i, ymax + 0.008, star, ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color="crimson" if p < 0.05 else "gray")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(roi_names, fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Mean CV Accuracy ± SEM per ROI", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: Paired difference (Feature – Original) per ROI ──────────
    ax2 = fig.add_subplot(gs[0, 1])
    diffs = [np.mean(r["scores_feat"]) - np.mean(r["scores_orig"]) for r in all_results]
    diff_colors = ["#DD8452" if d > 0 else "#4C72B0" for d in diffs]
    bars = ax2.bar(roi_names, diffs, color=diff_colors, edgecolor="black", linewidth=0.6)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    for i, (bar, d, r) in enumerate(zip(bars, diffs, all_results)):
        p = r["p_val"]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        y = d + 0.002 if d >= 0 else d - 0.005
        ax2.text(bar.get_x() + bar.get_width()/2, y, f"{d:+.3f}{star}",
                 ha="center", va="bottom" if d >= 0 else "top", fontsize=10,
                 fontweight="bold")
    ax2.set_ylabel("Δ Accuracy (Feature − Original)", fontsize=12)
    ax2.set_title("Accuracy Difference per ROI", fontsize=14, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Panel 3: Paired fold scatter per ROI ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for ri, r in enumerate(all_results):
        orig = np.array(r["scores_orig"])
        feat = np.array(r["scores_feat"])
        for j in range(len(orig)):
            ax3.plot([ri - 0.15, ri + 0.15], [orig[j], feat[j]], "-",
                     color="gray", alpha=0.15, linewidth=0.5)
        ax3.scatter(np.full(len(orig), ri - 0.15), orig, s=15, c=colors[0],
                    alpha=0.5, edgecolor="none")
        ax3.scatter(np.full(len(feat), ri + 0.15), feat, s=15, c=colors[1],
                    alpha=0.5, edgecolor="none")
        ax3.plot(ri - 0.15, orig.mean(), "s", color=colors[0], markersize=9,
                 markeredgecolor="black", zorder=5)
        ax3.plot(ri + 0.15, feat.mean(), "s", color=colors[1], markersize=9,
                 markeredgecolor="black", zorder=5)
    ax3.set_xticks(range(n_rois))
    ax3.set_xticklabels(roi_names, fontsize=11)
    ax3.set_ylabel("Accuracy", fontsize=12)
    ax3.set_title("Per-Fold Scores (dots) & Means (squares)", fontsize=14,
                  fontweight="bold")
    ax3.spines[["top", "right"]].set_visible(False)
    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[0],
                       markeredgecolor="black", markersize=8, label="Original"),
               Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[1],
                       markeredgecolor="black", markersize=8, label="Feature")]
    ax3.legend(handles=handles, fontsize=10, loc="upper right")

    # ── Panel 4: NaN % per ROI bar ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    nan_pcts = [r["pct_nan"] for r in all_results]
    nan_trial_pcts = [100 * r["n_nan_trials"] / r["shape"][0] for r in all_results]
    b4a = ax4.bar(x_pos - width/2, nan_pcts, width, color="#e74c3c", alpha=0.7,
                  edgecolor="black", linewidth=0.6, label="% NaN elements")
    b4b = ax4.bar(x_pos + width/2, nan_trial_pcts, width, color="#e67e22", alpha=0.7,
                  edgecolor="black", linewidth=0.6, label="% Trials with NaN")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(roi_names, fontsize=11)
    ax4.set_ylabel("Percentage (%)", fontsize=12)
    ax4.set_title("NaN Distribution per ROI", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.spines[["top", "right"]].set_visible(False)

    # ── Panel 5: Aggregated across all ROIs – pooled difference ──────────
    ax5 = fig.add_subplot(gs[2, 0])
    all_orig = np.concatenate([r["scores_orig"] for r in all_results])
    all_feat = np.concatenate([r["scores_feat"] for r in all_results])
    all_diff = all_feat - all_orig
    ax5.hist(all_diff, bins=25, color="#95a5a6", edgecolor="black", linewidth=0.5,
             alpha=0.8)
    ax5.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax5.axvline(all_diff.mean(), color="crimson", linewidth=2, linestyle="-",
                label=f"Mean Δ = {all_diff.mean():+.4f}")
    t_all, p_all = stats.ttest_rel(all_orig, all_feat)
    ax5.set_xlabel("Δ Accuracy (Feature − Original)", fontsize=12)
    ax5.set_ylabel("Count", fontsize=12)
    ax5.set_title(f"Pooled Fold Differences (N={len(all_diff)}, "
                  f"paired t={t_all:.2f}, p={p_all:.4f})",
                  fontsize=13, fontweight="bold")
    ax5.legend(fontsize=10)
    ax5.spines[["top", "right"]].set_visible(False)

    # ── Panel 6: Summary table ───────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    header = ["ROI", "Shape", "NaN%", "Orig Acc", "Feat Acc", "Δ", "t", "p"]
    table_data = [header]
    for r in all_results:
        mo = np.mean(r["scores_orig"])
        mf = np.mean(r["scores_feat"])
        table_data.append([
            r["roi"],
            f'{r["shape"][0]}×{r["shape"][1]}×{r["shape"][2]}',
            f'{r["pct_nan"]:.1f}%',
            f"{mo:.4f}",
            f"{mf:.4f}",
            f"{mf - mo:+.4f}",
            f"{r['t_stat']:.2f}",
            f"{r['p_val']:.4f}",
        ])
    # Add pooled row
    table_data.append([
        "POOLED", f"N={len(all_diff)}", "",
        f"{all_orig.mean():.4f}", f"{all_feat.mean():.4f}",
        f"{all_diff.mean():+.4f}", f"{t_all:.2f}", f"{p_all:.4f}",
    ])

    table = ax6.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.8)
    # header row
    for j in range(len(header)):
        table[0, j].set_facecolor("#2a2a2a")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # pooled row
    last = len(table_data) - 1
    for j in range(len(header)):
        table[last, j].set_facecolor("#f0e68c")
        table[last, j].set_text_props(fontweight="bold")
    # highlight significant p-values
    for i in range(1, len(table_data)):
        try:
            p = float(table_data[i][-1])
            if p < 0.05:
                table[i, -1].set_text_props(color="crimson", fontweight="bold")
        except ValueError:
            pass

    ax6.set_title("Summary Statistics", fontsize=14, fontweight="bold", pad=15)

    fig.suptitle("Batch Mixup Comparison: Row-level vs Feature-level\n"
                 "across Multiple ROIs (Stimulus, PhonemeSequence)",
                 fontsize=16, fontweight="bold", y=0.97)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Report saved to {save_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch compare mixup strategies")
    parser.add_argument("--bids_deriv", type=str,
                        default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/"
                                "derivatives/decoding(bipolar)")
    parser.add_argument("--rois", nargs="+",
                        default=["STGl", "SMCl", "PICl", "AICl"])
    parser.add_argument("--tmin", type=float, default=-0.5)
    parser.add_argument("--tmax", type=float, default=0.5)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--variance", type=float, default=0.85)
    parser.add_argument("--n_permutations", type=int, default=50)
    args = parser.parse_args()

    pipeline = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        PCA(n_components=args.variance, random_state=42),
        SVC(kernel="linear", random_state=42),
    )

    all_results = []
    for roi in args.rois:
        h5_path = (f"{args.bids_deriv}/sub-{roi}/phoneme/"
                    f"sub-{roi}_task-PhonemeSequence_proc-Stimulus"
                    f"_recording-1_desc-Repeat_highgamma.h5")
        if not Path(h5_path).exists():
            logger.warning(f"File not found, skipping: {h5_path}")
            continue

        result = process_one_roi(
            h5_path, roi, pipeline,
            tmin=args.tmin, tmax=args.tmax,
            n_folds=args.n_folds, n_repeats=args.n_repeats,
            n_permutations=args.n_permutations,
        )
        all_results.append(result)

    if not all_results:
        logger.error("No ROIs were processed!")
        return

    # Save report
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "mixup_batch_comparison.png")
    plot_batch_report(all_results, save_path)

    # Save numerical results to JSON
    json_path = os.path.join(save_dir, "mixup_batch_results.json")
    json_results = []
    for r in all_results:
        json_results.append({
            "roi": r["roi"],
            "shape": r["shape"],
            "n_classes": r["n_classes"],
            "pct_nan": r["pct_nan"],
            "n_nan_trials": r["n_nan_trials"],
            "n_nan_channels": r["n_nan_channels"],
            "mean_orig": float(np.mean(r["scores_orig"])),
            "std_orig": float(np.std(r["scores_orig"])),
            "mean_feat": float(np.mean(r["scores_feat"])),
            "std_feat": float(np.std(r["scores_feat"])),
            "diff": float(np.mean(r["scores_feat"]) - np.mean(r["scores_orig"])),
            "t_stat": float(r["t_stat"]),
            "p_val": float(r["p_val"]),
            "scores_orig": [float(s) for s in r["scores_orig"]],
            "scores_feat": [float(s) for s in r["scores_feat"]],
        })
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Numerical results saved to {json_path}")

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    for r in all_results:
        mo = np.mean(r["scores_orig"])
        mf = np.mean(r["scores_feat"])
        sig = "***" if r["p_val"] < 0.001 else "**" if r["p_val"] < 0.01 \
              else "*" if r["p_val"] < 0.05 else "n.s."
        logger.info(f"  {r['roi']:>5s}: Orig={mo:.4f}  Feat={mf:.4f}  "
                    f"Δ={mf-mo:+.4f}  t={r['t_stat']:.2f}  p={r['p_val']:.4f} {sig}")

    all_o = np.concatenate([r["scores_orig"] for r in all_results])
    all_f = np.concatenate([r["scores_feat"] for r in all_results])
    t_pool, p_pool = stats.ttest_rel(all_o, all_f)
    logger.info("-" * 70)
    logger.info(f"  POOLED: Orig={all_o.mean():.4f}  Feat={all_f.mean():.4f}  "
                f"Δ={all_f.mean()-all_o.mean():+.4f}  t={t_pool:.2f}  p={p_pool:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
