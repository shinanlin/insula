"""Summarize insula RT ridge HDF5 files into coverage/electrode/cluster CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.reaction_time.insula_ridge import iter_true_runs
from src.reaction_time.insula_rt_io import DEFAULT_RT_OUTPUT_ROOT, decode_strings


def _channel_metadata(h5: h5py.File) -> pd.DataFrame:
    group = h5["channels"]
    payload: dict[str, object] = {}
    for name, dataset in group.items():
        values = dataset[()]
        if values.dtype.kind in {"O", "S", "U"}:
            payload[name] = decode_strings(values)
        else:
            payload[name] = values
    return pd.DataFrame(payload)


def _functional_assignments(path: Path | None) -> pd.Series:
    if path is None or not path.is_file():
        return pd.Series(dtype=object)
    frame = pd.read_csv(path)
    if not {"channel", "functional_cluster"}.issubset(frame.columns):
        raise ValueError(f"Functional assignment schema invalid: {path}")
    return (
        frame.drop_duplicates("channel", keep="first")
        .set_index("channel")["functional_cluster"]
    )


def summarize_results(
    output_root: Path | str = DEFAULT_RT_OUTPUT_ROOT,
    *,
    assignments_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_root = Path(output_root)
    assignments = _functional_assignments(assignments_path)
    coverage_rows: list[dict] = []
    electrode_rows: list[dict] = []
    cluster_rows: list[dict] = []

    for path in sorted(output_root.glob("task-*/sub-*/*_rt-ridge.h5")):
        with h5py.File(path, "r") as h5:
            task = str(h5.attrs["task"])
            subject = str(h5.attrs["subject"])
            phase = str(h5.attrs["phase"])
            channel_meta = _channel_metadata(h5)
            score_r = h5["scores/r"][:]
            score_r2 = h5["scores/r2"][:]
            score_mae = h5["scores/mae"][:]
            significant = h5["inference/sig_mask_fwer"][:].astype(bool)
            cluster_p = h5["inference/cluster_p_fwer"][:]
            center = h5["windows/center"][:]
            start = h5["windows/start"][:]
            end = h5["windows/end"][:]
            items = decode_strings(h5["trials/item_id"][:])
            coverage_rows.append(
                {
                    "task": task,
                    "subject": subject,
                    "phase": phase,
                    "n_trials": len(items),
                    "n_items": len(set(items)),
                    "n_electrodes": len(channel_meta),
                    "n_significant_electrodes": int(significant.any(axis=1).sum()),
                    "n_windows": len(center),
                    "source_h5": str(path),
                }
            )
            for channel_index, meta in channel_meta.iterrows():
                finite = np.isfinite(score_r[channel_index])
                peak_index = (
                    int(np.nanargmax(score_r[channel_index])) if finite.any() else None
                )
                row = meta.to_dict()
                row.update(
                    {
                        "task": task,
                        "subject": subject,
                        "phase": phase,
                        "significant": bool(significant[channel_index].any()),
                        "n_significant_windows": int(significant[channel_index].sum()),
                        "n_significant_clusters": len(
                            iter_true_runs(significant[channel_index])
                        ),
                        "peak_r": (
                            float(score_r[channel_index, peak_index])
                            if peak_index is not None
                            else np.nan
                        ),
                        "peak_r2": (
                            float(score_r2[channel_index, peak_index])
                            if peak_index is not None
                            else np.nan
                        ),
                        "peak_mae": (
                            float(score_mae[channel_index, peak_index])
                            if peak_index is not None
                            else np.nan
                        ),
                        "peak_time": (
                            float(center[peak_index]) if peak_index is not None else np.nan
                        ),
                        "source_h5": str(path),
                    }
                )
                row["functional_cluster"] = assignments.get(
                    str(row["channel"]), np.nan
                )
                electrode_rows.append(row)

                for cluster_index, (first, stop) in enumerate(
                    iter_true_runs(significant[channel_index]), start=1
                ):
                    local = score_r[channel_index, first:stop]
                    peak = first + int(np.nanargmax(local))
                    cluster_rows.append(
                        {
                            "task": task,
                            "subject": subject,
                            "phase": phase,
                            "channel": row["channel"],
                            "roi": row["roi"],
                            "hemi": row["hemi"],
                            "functional_cluster": row["functional_cluster"],
                            "cluster_index": cluster_index,
                            "start_time": float(start[first]),
                            "end_time": float(end[stop - 1]),
                            "peak_time": float(center[peak]),
                            "peak_r": float(score_r[channel_index, peak]),
                            "peak_r2": float(score_r2[channel_index, peak]),
                            "cluster_p_fwer": float(
                                np.nanmin(cluster_p[channel_index, first:stop])
                            ),
                            "x_template": row.get("x_template", np.nan),
                            "y_template": row.get("y_template", np.nan),
                            "z_template": row.get("z_template", np.nan),
                            "source_h5": str(path),
                        }
                    )

    coverage = pd.DataFrame(coverage_rows)
    electrodes = pd.DataFrame(electrode_rows)
    clusters = pd.DataFrame(cluster_rows)
    summary_dir = output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(summary_dir / "coverage.csv", index=False)
    electrodes.to_csv(summary_dir / "electrodes.csv", index=False)
    clusters.to_csv(summary_dir / "significant_clusters.csv", index=False)
    return coverage, electrodes, clusters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RT_OUTPUT_ROOT)
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path(
            "/hpc/group/coganlab/nanlinshi/insula-functional/"
            "results/nmf/channel_assignments.csv"
        ),
    )
    args = parser.parse_args()
    coverage, electrodes, clusters = summarize_results(
        args.output_root, assignments_path=args.assignments
    )
    print(
        f"coverage={len(coverage)} rows | electrodes={len(electrodes)} rows | "
        f"clusters={len(clusters)} rows"
    )


if __name__ == "__main__":
    main()
