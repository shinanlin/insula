"""Load semantic ridge-encoding H5 results into tidy tables for visualization."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.paths import RESULTS_ROOT

# Avoid BlockingIOError when Slurm jobs are still writing results.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

DEFAULT_RESULTS_DIR = RESULTS_ROOT / "semantic" / "LexicalDelay"
DEFAULT_PARC_ROOT = Path(
    "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/parcellation"
)


def _decode_str_array(arr: np.ndarray) -> list[str]:
    out = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _open_h5_readonly(path: Path):
    """Open H5 for read without file locking (safe while jobs write other files)."""
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        # Older h5py without locking= kwarg
        return h5py.File(path, "r")


def load_parcellation(subject: str, parc_root: Path = DEFAULT_PARC_ROOT) -> pd.DataFrame:
    subject = subject.replace("sub-", "")
    path = parc_root / f"sub-{subject}" / "bipolar" / f"sub-{subject}_hammers.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


MULTI_MODELS = (
    "semantic",
    "phon",
    "acoustic",
    "full_perm_semantic",
)


def _model_from_h5_name(path: Path) -> str:
    """Parse model tag from ``..._ridge_{model}.h5`` filename."""
    stem = path.stem
    marker = "_ridge_"
    if marker not in stem:
        return "unknown"
    return stem.split(marker, 1)[1]


def iter_encoding_h5(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    model: str = "glove",
):
    """Yield (subject, path) for ridge H5 files of one model.

    Parameters
    ----------
    model
        ``glove`` (legacy) or a multi-block tag such as ``full_perm_semantic``.
    """
    results_dir = Path(results_dir)
    pattern = f"sub-*/sub-*_ridge_{model}.h5"
    for path in sorted(results_dir.glob(pattern)):
        if path.stat().st_size == 0:
            continue
        if model == "glove" and "_perm" in path.name:
            continue
        subject = path.parent.name.replace("sub-", "")
        yield subject, path


def iter_encoding_h5_multi(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    models: tuple[str, ...] | list[str] = MULTI_MODELS,
):
    """Yield (subject, path, model) for multi-block ridge H5 files."""
    for model in models:
        for subject, path in iter_encoding_h5(results_dir, model=model):
            yield subject, path, model


def load_encoding_long(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    parc_root: Path = DEFAULT_PARC_ROOT,
    model: str = "glove",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return channel-level summary and long channel×time table.

    Parameters
    ----------
    model
        Encoding model tag in the H5 filename (``glove``, ``semantic``,
        ``phon``, ``acoustic``, ``full_perm_semantic``, ...).

    Returns
    -------
    channels : one row per subject×channel with coords and ROI
    long : one row per subject×channel×time
    """
    ch_rows: list[dict] = []
    long_rows: list[dict] = []
    skipped: list[str] = []

    for subject, h5path in iter_encoding_h5(results_dir, model=model):
        parc = load_parcellation(subject, parc_root=parc_root)
        parc = parc.rename(columns={"name": "channel"})
        parc_lookup = parc.set_index("channel")

        try:
            f = _open_h5_readonly(h5path)
        except OSError as exc:
            skipped.append(f"{h5path.name} ({exc.__class__.__name__})")
            continue

        try:
            if "r" not in f or "times" not in f or "channel" not in f:
                skipped.append(f"{h5path.name} (incomplete)")
                continue
            r = f["r"][()]
            times = f["times"][()]
            channels = _decode_str_array(f["channel"][:])
            attrs = dict(f.attrs)
            model_tag = str(attrs.get("model", model) or model)
            has_mask = "mask" in f
            if has_mask:
                mask = f["mask"][()].astype(bool)
                p_values = f["p_values"][()] if "p_values" in f else None
            else:
                mask = None
                p_values = None
        except OSError as exc:
            skipped.append(f"{h5path.name} ({exc.__class__.__name__})")
            continue
        finally:
            f.close()

        for i, ch in enumerate(channels):
            abs_r = np.abs(r[i])
            ch_mean = float(np.nanmean(abs_r))
            ch_max = float(np.nanmax(abs_r))
            signed_mean = float(np.nanmean(r[i]))
            if mask is not None:
                ch_sig_any = bool(mask[i].any())
                ch_sig_frac = float(mask[i].mean())
            else:
                ch_sig_any = np.nan
                ch_sig_frac = np.nan

            if ch in parc_lookup.index:
                prow = parc_lookup.loc[ch]
            else:
                prow = None

            base = {
                "subject": subject,
                "phase": str(attrs.get("phase", "")),
                "description": str(attrs.get("description", "")),
                "model": model_tag,
                "channel": ch,
                "ch_mean_abs_r": ch_mean,
                "ch_max_abs_r": ch_max,
                "ch_mean_r": signed_mean,
                "ch_sig_any": ch_sig_any,
                "ch_sig_frac": ch_sig_frac,
                "has_significance": has_mask,
                "n_trials": int(attrs.get("n_trials", np.nan)),
                "n_tokens": int(attrs.get("n_tokens", np.nan)),
            }
            if prow is not None:
                base.update(
                    {
                        "roi": prow["roi"],
                        "roi_group": roi_group_label(prow["roi"], bool(prow["mix"])),
                        "mix": bool(prow["mix"]),
                        "hemi": prow["hemi"],
                        # native tkRAS (subject FS space) — not for group CVS plots
                        "x": float(prow["x"]),
                        "y": float(prow["y"]),
                        "z": float(prow["z"]),
                        # CVS template tkRAS — use for cvs_avg35_inMNI152 brain display
                        "x_t": float(prow["x_t"]),
                        "y_t": float(prow["y_t"]),
                        "z_t": float(prow["z_t"]),
                    }
                )
            else:
                base.update(
                    {
                        "roi": np.nan,
                        "roi_group": np.nan,
                        "mix": np.nan,
                        "hemi": np.nan,
                        "x": np.nan,
                        "y": np.nan,
                        "z": np.nan,
                        "x_t": np.nan,
                        "y_t": np.nan,
                        "z_t": np.nan,
                    }
                )
            ch_rows.append(base)

            for t_idx, t in enumerate(times):
                sig = bool(mask[i, t_idx]) if mask is not None else np.nan
                pval = float(p_values[i, t_idx]) if p_values is not None else np.nan
                long_rows.append(
                    {
                        "subject": subject,
                        "phase": base["phase"],
                        "description": base["description"],
                        "model": model_tag,
                        "channel": ch,
                        "time": float(t),
                        "r": float(r[i, t_idx]),
                        "abs_r": float(abs_r[t_idx]),
                        "significant": sig,
                        "p_value": pval,
                        "roi": base["roi"],
                        "roi_group": roi_group_label(base["roi"], base["mix"]),
                        "mix": base["mix"],
                        "hemi": base["hemi"],
                    }
                )

    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} H5 file(s) (locked/incomplete while jobs write): "
            + "; ".join(skipped[:5])
            + (" ..." if len(skipped) > 5 else ""),
            RuntimeWarning,
            stacklevel=2,
        )

    channels = pd.DataFrame(ch_rows)
    long = pd.DataFrame(long_rows)
    return channels, long


def load_encoding_long_multi(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    parc_root: Path = DEFAULT_PARC_ROOT,
    models: tuple[str, ...] | list[str] = MULTI_MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate channel/long tables for several encoding models."""
    ch_parts: list[pd.DataFrame] = []
    long_parts: list[pd.DataFrame] = []
    for model in models:
        ch, long = load_encoding_long(
            results_dir=results_dir,
            parc_root=parc_root,
            model=model,
        )
        if not ch.empty:
            ch_parts.append(ch)
            long_parts.append(long)
    if not ch_parts:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(ch_parts, ignore_index=True), pd.concat(long_parts, ignore_index=True)


def roi_group_label(roi: str, mix: bool) -> str | float:
    """Map Hammers ROI to vizpub-style groups (fig2 convention)."""
    if not isinstance(roi, str) or mix:
        return np.nan
    if roi in {"AIC", "PIC", "IFG", "MFG"}:
        return roi
    if roi in {"STGp", "STGa", "HG"}:
        return "STG"
    if roi in {"PrG", "PoG", "Subcentral"}:
        return "SMC"
    return np.nan


def build_roi_timeseries(long: pd.DataFrame) -> pd.DataFrame:
    """Subject×time×roi_group mean |r| after averaging channels.

    Preserves ``phase`` / ``description`` in the group keys when present.
    """
    df = long.copy()
    if "roi_group" not in df.columns:
        df["roi_group"] = [
            roi_group_label(r, m) for r, m in zip(df["roi"], df["mix"])
        ]
    df = df[df["roi_group"].notna()]
    keys = ["subject"]
    for col in ("description", "phase"):
        if col in df.columns:
            keys.append(col)
    keys.extend(["time", "roi_group"])
    return (
        df.groupby(keys, as_index=False)["abs_r"]
        .mean()
        .rename(columns={"abs_r": "value"})
    )
