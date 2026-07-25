"""Prototype diagnostic figure from serialized metric outputs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _read_pair_table(metric_dir: Path) -> pd.DataFrame:
    parquet = sorted(metric_dir.glob("*_pairs.parquet"))
    if parquet:
        return pd.read_parquet(parquet[0])
    csv = sorted(metric_dir.glob("*_pairs.csv.gz"))
    if csv:
        return pd.read_csv(csv[0])
    raise FileNotFoundError(f"No pair table in {metric_dir}")


def _read_detail(metric_dir: Path) -> xr.Dataset:
    files = sorted(metric_dir.glob("*_detail.nc"))
    if not files:
        raise FileNotFoundError(f"No detail NetCDF in {metric_dir}")
    return xr.load_dataset(files[0], engine="h5netcdf")


def _representative_pair_index(table: pd.DataFrame) -> int:
    pair_summary = (
        table.groupby("pair_id", sort=False)["p_uncorrected"]
        .min()
        .reset_index()
    )
    values = pair_summary["p_uncorrected"].to_numpy(dtype=float)
    if np.any(np.isfinite(values)):
        return int(np.nanargmin(values))
    return 0


def _distribution_panel(axis, table: pd.DataFrame, title: str) -> None:
    observed = table["stat"].to_numpy(dtype=float)
    null_column = (
        "null_mean"
        if "null_mean" in table
        else "null_mean_stat"
    )
    null = table[null_column].to_numpy(dtype=float)
    finite_observed = observed[np.isfinite(observed)]
    finite_null = null[np.isfinite(null)]
    bins = min(30, max(5, int(np.sqrt(max(len(table), 1)))))
    axis.hist(
        finite_null,
        bins=bins,
        color="#9aa0a6",
        alpha=0.65,
        label="null center",
    )
    axis.hist(
        finite_observed,
        bins=bins,
        color="#1565c0",
        alpha=0.65,
        label="observed",
    )
    axis.set_title(title)
    axis.set_ylabel("count")
    axis.legend(frameon=False, fontsize=8)


def _significance_panel(axis, table: pd.DataFrame, title: str) -> None:
    if "band" in table:
        grouped = table.groupby("band", sort=False)[["sig_fdr", "sig_fwer"]]
        values = grouped.mean().reindex(
            ["theta", "alpha", "beta", "broadband"]
        )
        x = np.arange(len(values))
        axis.bar(x - 0.18, values["sig_fdr"], 0.36, label="BH-FDR")
        axis.bar(x + 0.18, values["sig_fwer"], 0.36, label="max-stat")
        axis.set_xticks(x, values.index)
    else:
        values = [
            float(pd.to_numeric(table["sig_fdr"]).mean()),
            float(pd.to_numeric(table["sig_fwer"]).mean()),
        ]
        axis.bar(["BH-FDR", "max-stat"], values, color=["#2e7d32", "#ef6c00"])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("significant fraction")
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=8) if "band" in table else None


def create_prototype_diagnostic(
    entity_dir: str | Path,
    output: str | Path | None = None,
) -> Path:
    """Create a three-metric evidence/QC figure for one entity."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(entity_dir)
    tables = {
        metric: _read_pair_table(root / metric)
        for metric in ("xcorr", "oaec", "wpli")
    }
    details = {
        metric: _read_detail(root / metric)
        for metric in ("xcorr", "oaec", "wpli")
    }
    figure, axes = plt.subplots(
        3, 3, figsize=(15, 11), constrained_layout=True
    )

    xcorr_table = tables["xcorr"]
    _distribution_panel(axes[0, 0], xcorr_table, "xcorr: pair statistics")
    _significance_panel(axes[0, 1], xcorr_table, "xcorr: discoveries")
    xcorr_index = _representative_pair_index(xcorr_table)
    xcorr_detail = details["xcorr"]
    lag = xcorr_detail["lag"].to_numpy()
    axes[0, 2].plot(
        lag,
        xcorr_detail["observed_r"].isel(pair=xcorr_index),
        color="#1565c0",
        label="observed HGA amplitude r",
    )
    axes[0, 2].fill_between(
        lag,
        np.tanh(
            xcorr_detail["null_lower_fisher_z"].isel(pair=xcorr_index)
        ),
        np.tanh(
            xcorr_detail["null_upper_fisher_z"].isel(pair=xcorr_index)
        ),
        color="#9aa0a6",
        alpha=0.35,
        label="trial-shuffle pointwise band",
    )
    axes[0, 2].axvline(0.0, color="black", lw=0.8)
    axes[0, 2].set(
        title=f"xcorr evidence: {xcorr_table.iloc[xcorr_index]['pair_id']}",
        xlabel="lag (s; negative = Insula/source earlier)",
        ylabel="signed Pearson r",
    )
    axes[0, 2].legend(frameon=False, fontsize=8)

    oaec_table = tables["oaec"]
    _distribution_panel(axes[1, 0], oaec_table, "OAEC: pair statistics")
    _significance_panel(axes[1, 1], oaec_table, "OAEC: discoveries")
    oaec_index = _representative_pair_index(oaec_table)
    oaec_detail = details["oaec"]
    frequencies = oaec_detail["hga_frequency"].to_numpy()
    directional = oaec_detail["observed_fisher_z"].isel(
        pair=oaec_index
    )
    for direction in directional["direction"].to_numpy():
        axes[1, 2].plot(
            frequencies,
            directional.sel(direction=direction),
            marker="o",
            ms=3,
            label=str(direction).replace("_", " "),
        )
    axes[1, 2].set(
        title=f"OAEC evidence: {oaec_table.iloc[oaec_index]['pair_id']}",
        xlabel="HGA filterbank center (Hz)",
        ylabel="orthogonalized envelope Fisher z",
    )
    axes[1, 2].legend(frameon=False, fontsize=8)

    wpli_table = tables["wpli"]
    _distribution_panel(axes[2, 0], wpli_table, "TF-dwPLI: pair×band statistics")
    _significance_panel(axes[2, 1], wpli_table, "TF-dwPLI: discoveries")
    wpli_index = _representative_pair_index(wpli_table)
    representative_pair = (
        wpli_table.groupby("pair_id", sort=False)["p_uncorrected"]
        .min()
        .reset_index()
        .iloc[wpli_index]["pair_id"]
    )
    wpli_detail = details["wpli"]
    pair_ids = wpli_detail["pair_id"].astype(str).to_numpy()
    detail_index = int(np.flatnonzero(pair_ids == representative_pair)[0])
    image = axes[2, 2].pcolormesh(
        wpli_detail["time"],
        wpli_detail["frequency"],
        wpli_detail["observed_tf_wpli2_debiased"].isel(pair=detail_index),
        shading="auto",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
    )
    axes[2, 2].axhline(8.0, color="black", lw=0.6)
    axes[2, 2].axhline(13.0, color="black", lw=0.6)
    axes[2, 2].set(
        title=f"TF-dwPLI phase evidence: {representative_pair}",
        xlabel="phase time (s)",
        ylabel="frequency (Hz)",
    )
    figure.colorbar(image, ax=axes[2, 2], label="debiased squared wPLI")
    figure.suptitle(
        "Strict Hammers Insula-to-all connectivity prototype diagnostics",
        fontsize=15,
    )

    destination = (
        Path(output) if output is not None else root / "diag_prototype.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    figure.savefig(temporary, format="png", dpi=180)
    plt.close(figure)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return destination
