#!/usr/bin/env python3
"""Compare original row-level mixup vs per-feature feature_mixup for NaN filling.

Loads a single h5 file, runs CV decoding with both NaN-filling strategies
under identical conditions, and produces a visual comparison report.
"""

import rootutils
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
import time as _time
import logging
import sys

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
    """Per-feature NaN interpolation (from decoder (1).py)."""
    if rng is None:
        rng = np.random.RandomState()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.RandomState(rng)

    n_samples = x_cls.shape[0]
    x_2d = x_cls.reshape(n_samples, -1)
    n_features = x_2d.shape[1]

    nan_mask = np.isnan(x_2d)
    if not nan_mask.any():
        return

    for f in range(n_features):
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
    X_train = X[train_idx].copy()
    X_test = X[test_idx].copy()
    y_train = y[train_idx].copy()
    y_test = y[test_idx].copy()

    unique_classes = np.unique(y_train)
    for cls in unique_classes:
        idx = y_train == cls
        x_cls = X_train[idx]
        mixup(x_cls, obs_axis=0, rng=42)
        X_train[idx] = x_cls

    is_nan_test = np.isnan(X_test)
    if is_nan_test.any():
        X_test[is_nan_test] = np.random.normal(0, 1, int(np.sum(is_nan_test)))

    return X_train, X_test, y_train, y_test


def sample_fold_feature(X, y, train_idx, test_idx):
    """New sample_fold using per-feature feature_mixup."""
    X_train = X[train_idx].copy()
    X_test = X[test_idx].copy()
    y_train = y[train_idx].copy()
    y_test = y[test_idx].copy()

    unique_classes = np.unique(y_train)
    for cls in unique_classes:
        idx = y_train == cls
        x_cls = X_train[idx]
        feature_mixup(x_cls, alpha=1.0, rng=42)
        X_train[idx] = x_cls

    is_nan_test = np.isnan(X_test)
    if is_nan_test.any():
        X_test[is_nan_test] = np.random.normal(0, 1, int(np.sum(is_nan_test)))

    return X_train, X_test, y_train, y_test


# ── CV decoding helper ──────────────────────────────────────────────────────

def run_cv_decoding(X, y, cv, pipeline, sample_fold_fn, n_permutations=50,
                    random_state=42):
    """Run CV decoding with the given sample_fold function.

    Returns
    -------
    fold_scores : list[float]
        Observed accuracy per fold.
    perm_scores : ndarray, shape (n_folds, n_permutations)
        Permutation accuracy per fold.
    """
    scorer = get_scorer("accuracy")
    splits = list(cv.split(X, y))
    fold_scores = []
    perm_scores_all = []

    for fold_idx, (tr, te) in enumerate(splits):
        dec = clone(pipeline)
        X_train, X_test, y_train, y_test = sample_fold_fn(X, y, tr, te)

        dec.fit(X_train, y_train)
        score = scorer(dec, X_test, y_test)
        fold_scores.append(score)

        # permutation scores for this fold
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

    return fold_scores, np.array(perm_scores_all)


# ── NaN diagnostics ─────────────────────────────────────────────────────────

def nan_diagnostics(X, y):
    """Print and return NaN distribution info."""
    total = X.size
    n_nan = np.isnan(X).sum()
    pct = 100 * n_nan / total

    # per-trial: does this trial have any NaN?
    has_nan_trial = np.isnan(X).any(axis=(1, 2))
    n_nan_trials = has_nan_trial.sum()

    # per-channel: does this channel have any NaN across all trials?
    has_nan_ch = np.isnan(X).any(axis=(0, 2))
    n_nan_ch = has_nan_ch.sum()

    # NaN heatmap: for each (trial, channel) does any timepoint have NaN?
    nan_heatmap = np.isnan(X).any(axis=2)  # (trial, channel)

    info = {
        "shape": X.shape,
        "total_nan": int(n_nan),
        "pct_nan": pct,
        "n_trials": X.shape[0],
        "n_nan_trials": int(n_nan_trials),
        "n_channels": X.shape[1],
        "n_nan_channels": int(n_nan_ch),
        "n_classes": len(np.unique(y)),
        "class_counts": {str(c): int((y == c).sum()) for c in np.unique(y)},
        "nan_heatmap": nan_heatmap,
    }

    logger.info("=" * 60)
    logger.info("NaN Diagnostics")
    logger.info("=" * 60)
    logger.info(f"  X shape            : {info['shape']}")
    logger.info(f"  Total NaN elements : {info['total_nan']:,} / {total:,} ({pct:.2f}%)")
    logger.info(f"  Trials with NaN    : {info['n_nan_trials']} / {info['n_trials']}")
    logger.info(f"  Channels with NaN  : {info['n_nan_channels']} / {info['n_channels']}")
    logger.info(f"  Classes            : {info['n_classes']}")
    for c, cnt in info["class_counts"].items():
        cls_mask = y == int(c) if y.dtype.kind in ('i', 'u') else y == c
        cls_nan = np.isnan(X[cls_mask]).any(axis=(1, 2)).sum()
        logger.info(f"    class {c}: {cnt} total, {cls_nan} with NaN")
    logger.info("=" * 60)

    return info


# ── Visualization ────────────────────────────────────────────────────────────

def plot_comparison(fold_scores_orig, fold_scores_feat, perm_orig, perm_feat,
                    nan_info, save_path):
    """Generate a multi-panel comparison figure."""
    fig = plt.figure(figsize=(18, 14), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    methods = ["Row-level mixup\n(original)", "Feature-level mixup\n(new)"]
    colors = ["#4C72B0", "#DD8452"]

    orig_arr = np.array(fold_scores_orig)
    feat_arr = np.array(fold_scores_feat)

    # ── Panel 1: Bar chart of mean accuracy ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    means = [orig_arr.mean(), feat_arr.mean()]
    sems = [orig_arr.std() / np.sqrt(len(orig_arr)),
            feat_arr.std() / np.sqrt(len(feat_arr))]
    bars = ax1.bar(methods, means, yerr=sems, color=colors, capsize=6,
                   edgecolor="black", linewidth=0.8, width=0.5)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Mean CV Accuracy ± SEM", fontsize=13, fontweight="bold")
    for bar, m in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{m:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(means) * 1.25)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: Paired fold scores ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    n_folds = len(orig_arr)
    for i in range(n_folds):
        ax2.plot([0, 1], [orig_arr[i], feat_arr[i]], "o-", color="gray",
                 alpha=0.5, markersize=5)
    ax2.plot(0, orig_arr.mean(), "s", color=colors[0], markersize=10,
             zorder=5, markeredgecolor="black")
    ax2.plot(1, feat_arr.mean(), "s", color=colors[1], markersize=10,
             zorder=5, markeredgecolor="black")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Original\nmixup", "Feature\nmixup"], fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Paired Fold Comparison", fontsize=13, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    # paired t-test
    t_stat, p_val = stats.ttest_rel(orig_arr, feat_arr)
    sig_str = f"p = {p_val:.4f}" if p_val >= 0.001 else f"p = {p_val:.2e}"
    ax2.text(0.5, 0.95, f"Paired t-test: t={t_stat:.2f}, {sig_str}",
             transform=ax2.transAxes, ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                       edgecolor="gray"))

    # ── Panel 3: Box + strip of per-fold scores ─────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    bp = ax3.boxplot([orig_arr, feat_arr], labels=["Original", "Feature"],
                     patch_artist=True, widths=0.4)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    jitter1 = 1 + np.random.normal(0, 0.04, len(orig_arr))
    jitter2 = 2 + np.random.normal(0, 0.04, len(feat_arr))
    ax3.scatter(jitter1, orig_arr, c=colors[0], edgecolor="black", s=30, zorder=3)
    ax3.scatter(jitter2, feat_arr, c=colors[1], edgecolor="black", s=30, zorder=3)
    ax3.set_ylabel("Accuracy", fontsize=12)
    ax3.set_title("Score Distribution", fontsize=13, fontweight="bold")
    ax3.spines[["top", "right"]].set_visible(False)

    # ── Panel 4: NaN heatmap ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    hm = nan_info["nan_heatmap"].astype(float)
    im = ax4.imshow(hm, aspect="auto", cmap="Reds", interpolation="nearest")
    ax4.set_xlabel("Channel", fontsize=11)
    ax4.set_ylabel("Trial", fontsize=11)
    ax4.set_title("NaN Distribution (trial × channel)", fontsize=13,
                  fontweight="bold")
    plt.colorbar(im, ax=ax4, label="Has NaN", shrink=0.8)

    # ── Panel 5: Permutation null distributions ──────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    perm_mean_orig = perm_orig.mean(axis=0)
    perm_mean_feat = perm_feat.mean(axis=0)
    ax5.hist(perm_mean_orig, bins=20, alpha=0.6, color=colors[0],
             label="Original perm", edgecolor="black", linewidth=0.5)
    ax5.hist(perm_mean_feat, bins=20, alpha=0.6, color=colors[1],
             label="Feature perm", edgecolor="black", linewidth=0.5)
    ax5.axvline(orig_arr.mean(), color=colors[0], linestyle="--", linewidth=2,
                label=f"Obs orig ({orig_arr.mean():.3f})")
    ax5.axvline(feat_arr.mean(), color=colors[1], linestyle="--", linewidth=2,
                label=f"Obs feat ({feat_arr.mean():.3f})")
    ax5.set_xlabel("Accuracy", fontsize=11)
    ax5.set_ylabel("Count", fontsize=11)
    ax5.set_title("Permutation Null vs Observed", fontsize=13, fontweight="bold")
    ax5.legend(fontsize=8)
    ax5.spines[["top", "right"]].set_visible(False)

    # ── Panel 6: Summary table ───────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    # p-values from permutation
    p_orig = (np.sum(perm_mean_orig >= orig_arr.mean()) + 1) / (len(perm_mean_orig) + 1)
    p_feat = (np.sum(perm_mean_feat >= feat_arr.mean()) + 1) / (len(perm_mean_feat) + 1)

    table_data = [
        ["", "Original mixup", "Feature mixup"],
        ["Mean accuracy", f"{orig_arr.mean():.4f}", f"{feat_arr.mean():.4f}"],
        ["Std", f"{orig_arr.std():.4f}", f"{feat_arr.std():.4f}"],
        ["Min / Max", f"{orig_arr.min():.3f} / {orig_arr.max():.3f}",
                      f"{feat_arr.min():.3f} / {feat_arr.max():.3f}"],
        ["Perm p-value", f"{p_orig:.4f}", f"{p_feat:.4f}"],
        ["Paired t-test", f"t = {t_stat:.3f}", f"p = {p_val:.4f}"],
    ]

    table = ax6.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    # header row styling
    for j in range(3):
        table[0, j].set_facecolor("#2a2a2a")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax6.set_title("Summary Statistics", fontsize=13, fontweight="bold", pad=20)

    fig.suptitle("Mixup Strategy Comparison: Row-level vs Feature-level",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Figure saved to {save_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(h5_path, tmin=-0.5, tmax=0.5, n_folds=3, n_repeats=3,
         variance=0.85, n_permutations=50, n_jobs=1):
    """Load data, run both methods, compare."""

    logger.info(f"Loading data from {h5_path}")
    with h5py.File(h5_path, "r") as f:
        X = f["X"][()]
        y = f["y"][()]
        fs = f.attrs["fs"]
    logger.info(f"Raw X shape: {X.shape}, fs={fs}")

    # Crop to [tmin, tmax]
    t_start = -1.0
    start_idx = int(fs * (tmin - t_start))
    end_idx = int(fs * (tmax - t_start))
    X = X[:, :, start_idx:end_idx]
    logger.info(f"Cropped X shape: {X.shape} (tmin={tmin}, tmax={tmax})")

    # Decode y if bytes
    if y.dtype.kind == 'S' or y.dtype.kind == 'O':
        y = np.array([s.decode() if isinstance(s, bytes) else s for s in y])

    # NaN diagnostics
    nan_info = nan_diagnostics(X, y)

    # Pipeline
    pipeline = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        PCA(n_components=variance, random_state=42),
        SVC(kernel="linear", random_state=42),
    )

    # CV splitter
    cv = MinimumNaNSplit(n_splits=n_folds, n_repeats=n_repeats)

    # ── Run original mixup ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running ORIGINAL row-level mixup ...")
    logger.info("=" * 60)
    t0 = _time.time()
    scores_orig, perm_orig = run_cv_decoding(
        X, y, cv, pipeline, sample_fold_original,
        n_permutations=n_permutations, random_state=42,
    )
    t_orig = _time.time() - t0
    logger.info(f"Original mixup: mean={np.mean(scores_orig):.4f}, "
                f"std={np.std(scores_orig):.4f}, time={t_orig:.1f}s")

    # ── Run feature mixup ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running NEW feature-level mixup ...")
    logger.info("=" * 60)
    t0 = _time.time()
    scores_feat, perm_feat = run_cv_decoding(
        X, y, cv, pipeline, sample_fold_feature,
        n_permutations=n_permutations, random_state=42,
    )
    t_feat = _time.time() - t0
    logger.info(f"Feature mixup : mean={np.mean(scores_feat):.4f}, "
                f"std={np.std(scores_feat):.4f}, time={t_feat:.1f}s")

    # ── Comparison ───────────────────────────────────────────────────────
    t_stat, p_val = stats.ttest_rel(scores_orig, scores_feat)
    logger.info("=" * 60)
    logger.info("COMPARISON")
    logger.info(f"  Original : {np.mean(scores_orig):.4f} ± {np.std(scores_orig):.4f}")
    logger.info(f"  Feature  : {np.mean(scores_feat):.4f} ± {np.std(scores_feat):.4f}")
    logger.info(f"  Diff     : {np.mean(scores_orig) - np.mean(scores_feat):.4f}")
    logger.info(f"  Paired t : t={t_stat:.3f}, p={p_val:.4f}")
    logger.info("=" * 60)

    # ── Visualization ────────────────────────────────────────────────────
    save_path = "results/mixup_comparison.png"
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_comparison(scores_orig, scores_feat, perm_orig, perm_feat,
                    nan_info, save_path)

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare mixup strategies")
    parser.add_argument(
        "--h5_path", type=str,
        default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/"
                "decoding(bipolar)/sub-AICl/phoneme/"
                "sub-AICl_task-PhonemeSequence_proc-Stimulus_recording-1_desc-Repeat_highgamma.h5",
    )
    parser.add_argument("--tmin", type=float, default=-0.5)
    parser.add_argument("--tmax", type=float, default=0.5)
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--variance", type=float, default=0.85)
    parser.add_argument("--n_permutations", type=int, default=50)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))
