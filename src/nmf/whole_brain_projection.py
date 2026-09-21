"""Project whole-brain HGA waveforms onto frozen Insula temporal motifs.

The canonical Insula NMF factorizes electrode-by-time data as ``X ~= W @ H``.
This module freezes the three temporal rows of ``H`` and estimates one
non-negative loading vector per sampled whole-brain electrode.  It is waveform
matching, not pairwise connectivity: no second electrode is required in the
same participant.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from scipy.optimize import nnls

from src.nmf.waveform_analysis import (
    CLUSTER_ORDER,
    FUNCTION_COLORS,
    TASKS,
    discover_paths,
    prepare_shape_matrix,
)
from src.paths import (
    RESULTS_ROOT,
    hga_results_dir,
    img_dir,
    nmf_assignments_path,
    nmf_results_dir,
    save_svg,
)

LOGGER = logging.getLogger(__name__)

PHASES = ("stimulus", "delay", "go", "response")
DESCRIPTION = "Repeat"
DEFAULT_EXCLUDE_SUBJECTS = ("D0121",)
SIGNIFICANCE_MODES = ("mask-any", "union")
SignificanceMode = Literal["mask-any", "union"]
TIME_DECIMALS = 8

HGA_COLUMNS = (
    "time",
    "channel",
    "value",
    "mask",
    "subject",
    "description",
    "task",
    "phase",
    "modality",
    "label",
    "roi",
    "hemi",
    "x",
    "y",
    "z",
    "mix",
)
METADATA_COLUMNS = (
    "subject",
    "roi",
    "hemi",
    "x",
    "y",
    "z",
    "label",
    "mix",
)


def _normalize_subject(value: object) -> str:
    text = str(value)
    return text[4:] if text.startswith("sub-") else text


def _normalize_phase(value: object) -> str:
    text = str(value).lower()
    return {"audio": "stimulus", "resp": "response"}.get(text, text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    truthy = {"1", "true", "t", "yes", "y"}
    return series.fillna(False).astype(str).str.lower().isin(truthy)


@dataclass(frozen=True)
class TemporalBasis:
    """Frozen temporal NMF dictionary in component-by-feature orientation."""

    component_names: tuple[str, ...]
    phases: tuple[str, ...]
    phase_times: dict[str, np.ndarray]
    H: np.ndarray
    source_path: Path
    source_sha256: str

    @property
    def n_features(self) -> int:
        return int(self.H.shape[1])

    @property
    def phase_boundaries(self) -> np.ndarray:
        sizes = [len(self.phase_times[phase]) for phase in self.phases]
        return np.cumsum(sizes)[:-1]


def load_temporal_basis(path: Path | None = None) -> TemporalBasis:
    """Read canonical ``H_by_phase.csv`` and return unit-norm temporal rows."""

    source = Path(path or (nmf_results_dir() / "H_by_phase.csv"))
    frame = pd.read_csv(source)
    required = {
        "functional_cluster",
        "phase",
        "time",
        "normalized_H",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"temporal basis missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["phase"] = frame["phase"].map(_normalize_phase)
    present_components = set(frame["functional_cluster"].astype(str))
    component_names = tuple(
        name for name in CLUSTER_ORDER if name in present_components
    )
    if len(component_names) != len(present_components):
        extras = sorted(present_components - set(component_names))
        component_names = component_names + tuple(extras)
    phases = tuple(phase for phase in PHASES if phase in set(frame["phase"]))
    if not component_names or not phases:
        raise ValueError("temporal basis contains no recognized components or phases")

    phase_times: dict[str, np.ndarray] = {}
    rows: list[np.ndarray] = []
    for name in component_names:
        segments: list[np.ndarray] = []
        for phase in phases:
            subset = frame.loc[
                frame["functional_cluster"].astype(str).eq(name)
                & frame["phase"].eq(phase)
            ].sort_values("time")
            if subset.empty:
                raise ValueError(f"basis missing component={name}, phase={phase}")
            times = subset["time"].to_numpy(dtype=float)
            if phase not in phase_times:
                phase_times[phase] = times
            elif not np.allclose(times, phase_times[phase]):
                raise ValueError(f"component time grids differ in phase={phase}")
            segments.append(subset["normalized_H"].to_numpy(dtype=float))
        rows.append(np.concatenate(segments))

    H = np.vstack(rows)
    if not np.isfinite(H).all() or (H < -1e-12).any():
        raise ValueError("temporal basis must be finite and non-negative")
    H = np.clip(H, 0.0, None)
    norms = np.linalg.norm(H, axis=1)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("every temporal basis row must have non-zero norm")
    H = H / norms[:, None]
    return TemporalBasis(
        component_names=component_names,
        phases=phases,
        phase_times=phase_times,
        H=H,
        source_path=source.resolve(),
        source_sha256=_sha256(source),
    )


def discover_hga_path(
    task: str,
    subject: str,
    phase: str,
    *,
    description: str = DESCRIPTION,
    results_root: Path = RESULTS_ROOT,
) -> BIDSPath | None:
    """Return one packaged HGA CSV using a BIDSPath query."""

    task_root = (
        hga_results_dir(task)
        if Path(results_root) == RESULTS_ROOT
        else Path(results_root) / "hga" / task
    )
    query = BIDSPath(
        root=str(task_root),
        subject=_normalize_subject(subject),
        task=task,
        processing=phase.capitalize(),
        description=description,
        datatype="HGA",
        suffix="time",
        extension=".csv",
        check=False,
    )
    matches = query.match()
    if not matches:
        return None
    # Some tasks package several presentation modalities as separate recording
    # entities.  The canonical Insula NMF uses sound rows only.
    if len(matches) > 1:
        sound_matches = [path for path in matches if path.recording == "sound"]
        if len(sound_matches) == 1:
            matches = sound_matches
    if len(matches) > 1:
        raise ValueError(
            f"Expected one HGA file for {subject}/{task}/{phase}, got {matches}"
        )
    return matches[0]


def discover_cohort_subjects(
    tasks: Sequence[str] = TASKS,
    *,
    results_root: Path = RESULTS_ROOT,
    exclude_subjects: Iterable[str] = DEFAULT_EXCLUDE_SUBJECTS,
) -> tuple[str, ...]:
    """Unique subjects with Repeat HGA CSVs, minus ``exclude_subjects``."""

    paths = discover_paths(Path(results_root), tuple(tasks))
    exclude = {_normalize_subject(name) for name in exclude_subjects}
    found: list[str] = []
    for path in paths:
        for part in Path(path).parts:
            if part.startswith("sub-"):
                found.append(_normalize_subject(part))
                break
    return tuple(sorted({name for name in found if name not in exclude}))


def _discovery_channels(assignments_path: Path | None = None) -> set[str]:
    path = Path(assignments_path or nmf_assignments_path())
    if not path.is_file():
        return set()
    frame = pd.read_csv(path, usecols=["channel"])
    return set(frame["channel"].astype(str))


def load_subject_hga_rows(
    subject: str,
    tasks: Sequence[str],
    phases: Sequence[str],
    *,
    description: str = DESCRIPTION,
    results_root: Path = RESULTS_ROOT,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load available packaged HGA rows for one participant."""

    frames: list[pd.DataFrame] = []
    used_paths: list[str] = []
    missing_inputs: list[str] = []
    for task in tasks:
        for phase in phases:
            path = discover_hga_path(
                task,
                subject,
                phase,
                description=description,
                results_root=results_root,
            )
            input_id = f"{task}/{phase}"
            if path is None:
                missing_inputs.append(input_id)
                continue
            frame = pd.read_csv(
                path.fpath,
                usecols=lambda column: column in HGA_COLUMNS,
            )
            required = {"time", "channel", "value", "mask", "phase", "task"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{path.fpath} missing columns: {sorted(missing)}")
            if "subject" not in frame:
                frame["subject"] = _normalize_subject(subject)
            else:
                frame["subject"] = frame["subject"].map(_normalize_subject)
            frame["phase"] = frame["phase"].map(_normalize_phase)
            if "modality" in frame:
                frame = frame.loc[frame["modality"].eq("sound")].copy()
            frames.append(frame)
            used_paths.append(str(path.fpath.resolve()))
    if not frames:
        raise FileNotFoundError(
            f"No packaged HGA inputs found for subject={_normalize_subject(subject)}"
        )
    return pd.concat(frames, ignore_index=True), used_paths, missing_inputs


def _phase_matrix(
    rows: pd.DataFrame,
    phase: str,
    target_times: np.ndarray,
    *,
    min_coverage: float,
) -> pd.DataFrame:
    phase_rows = rows.loc[rows["phase"].eq(phase)].copy()
    if phase_rows.empty:
        raise ValueError(f"no rows for phase={phase}")
    target_keys = np.round(target_times, TIME_DECIMALS)
    tolerance = 0.5 * np.min(np.diff(target_times)) if len(target_times) > 1 else 1e-6
    phase_rows = phase_rows.loc[
        phase_rows["time"].between(
            float(target_times.min() - tolerance),
            float(target_times.max() + tolerance),
        )
    ].copy()
    phase_rows["_time_key"] = np.round(
        phase_rows["time"].to_numpy(dtype=float), TIME_DECIMALS
    )
    matrix = (
        phase_rows.groupby(["channel", "_time_key"], sort=True)["value"]
        .mean()
        .unstack("_time_key")
        .reindex(columns=target_keys)
    )
    matrix = matrix.loc[matrix.notna().mean(axis=1) >= min_coverage]
    return matrix.interpolate(axis=1, limit_direction="both")


def build_subject_waveform_matrix(
    rows: pd.DataFrame,
    basis: TemporalBasis,
    *,
    significance_mode: SignificanceMode = "mask-any",
    min_coverage: float = 0.95,
    min_tasks: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build channel-by-feature waveforms and aligned electrode metadata."""

    if significance_mode not in SIGNIFICANCE_MODES:
        raise ValueError(
            f"significance_mode must be one of {SIGNIFICANCE_MODES}, "
            f"got {significance_mode!r}"
        )
    if min_tasks < 1:
        raise ValueError("min_tasks must be at least 1")
    rows = rows.copy()
    rows["channel"] = rows["channel"].astype(str)
    rows["phase"] = rows["phase"].map(_normalize_phase)
    rows["mask"] = _coerce_bool(rows["mask"])

    per_phase = [
        _phase_matrix(
            rows,
            phase,
            basis.phase_times[phase],
            min_coverage=min_coverage,
        )
        for phase in basis.phases
    ]
    common = per_phase[0].index
    for matrix in per_phase[1:]:
        common = common.intersection(matrix.index)
    if len(common) == 0:
        raise ValueError("no channels have sufficient coverage in every basis phase")

    included_phase_rows: list[pd.DataFrame] = []
    for phase in basis.phases:
        times = basis.phase_times[phase]
        tolerance = 0.5 * np.min(np.diff(times)) if len(times) > 1 else 1e-6
        included_phase_rows.append(
            rows.loc[
                rows["phase"].eq(phase)
                & rows["time"].between(
                    float(times.min() - tolerance),
                    float(times.max() + tolerance),
                )
            ]
        )
    included_rows = pd.concat(included_phase_rows, ignore_index=True)
    if significance_mode == "mask-any":
        significant = included_rows.groupby("channel")["mask"].any()
        common = common.intersection(significant.index[significant])
    task_counts = rows.groupby("channel")["task"].nunique()
    common = common.intersection(task_counts.index[task_counts >= min_tasks])
    if len(common) == 0:
        raise ValueError(
            f"no channels remain after significance_mode={significance_mode}, "
            f"min_tasks={min_tasks}"
        )

    concat = pd.concat([matrix.loc[common] for matrix in per_phase], axis=1)
    concat.columns = pd.MultiIndex.from_tuples(
        [
            (phase, float(time))
            for phase in basis.phases
            for time in basis.phase_times[phase]
        ],
        names=("phase", "time"),
    )

    available_meta = [column for column in METADATA_COLUMNS if column in rows]
    aggregations = {column: "first" for column in available_meta}
    metadata = rows.groupby("channel", sort=True).agg(aggregations).loc[common]
    metadata["n_tasks_available"] = (
        rows.groupby("channel")["task"].nunique().reindex(common).fillna(0).astype(int)
    )
    metadata["n_mask_timepoints"] = (
        included_rows.groupby("channel")["mask"]
        .sum()
        .reindex(common)
        .fillna(0)
        .astype(int)
    )
    significant_phases = (
        included_rows.loc[included_rows["mask"]]
        .groupby("channel")["phase"]
        .agg(lambda values: "|".join(sorted(set(values), key=PHASES.index)))
    )
    metadata["significant_phases"] = significant_phases.reindex(common).fillna("")
    positive = np.clip(concat.to_numpy(dtype=float), 0.0, None)
    metadata["positive_l2_norm"] = np.linalg.norm(positive, axis=1)
    metadata["positive_peak"] = positive.max(axis=1)
    metadata["positive_mean"] = positive.mean(axis=1)
    return concat, metadata


def project_fixed_temporal_basis(
    X: np.ndarray,
    H: np.ndarray,
    *,
    ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate non-negative motif loadings and reconstructed waveforms."""

    X = np.asarray(X, dtype=float)
    H = np.asarray(H, dtype=float)
    if X.ndim != 2 or H.ndim != 2 or X.shape[1] != H.shape[1]:
        raise ValueError(
            f"expected X=(electrode, feature), H=(component, feature); "
            f"got X={X.shape}, H={H.shape}"
        )
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if not np.isfinite(X).all() or not np.isfinite(H).all():
        raise ValueError("X and H must be finite")
    if (X < -1e-12).any() or (H < -1e-12).any():
        raise ValueError("X and H must be non-negative")

    design = H.T
    if ridge > 0:
        design = np.vstack([design, np.sqrt(ridge) * np.eye(H.shape[0])])
    weights = np.zeros((X.shape[0], H.shape[0]), dtype=float)
    for index, waveform in enumerate(X):
        target = waveform
        if ridge > 0:
            target = np.concatenate([waveform, np.zeros(H.shape[0])])
        weights[index], _ = nnls(design, target)
    return weights, weights @ H


def build_projection_table(
    metadata: pd.DataFrame,
    X: np.ndarray,
    weights: np.ndarray,
    reconstructed: np.ndarray,
    component_names: Sequence[str],
    *,
    pilot_min_explained_energy: float = 0.5,
) -> pd.DataFrame:
    """Combine loadings, fit diagnostics, and metadata for each electrode."""

    if len(metadata) != X.shape[0] or weights.shape[0] != X.shape[0]:
        raise ValueError("metadata, X, and weights have inconsistent row counts")
    residual = X - reconstructed
    sse = np.square(residual).sum(axis=1)
    energy = np.square(X).sum(axis=1)
    centered_energy = np.square(X - X.mean(axis=1, keepdims=True)).sum(axis=1)
    explained_energy = 1.0 - np.divide(
        sse,
        energy,
        out=np.full_like(sse, np.nan),
        where=energy > np.finfo(float).eps,
    )
    reconstruction_r2 = 1.0 - np.divide(
        sse,
        centered_energy,
        out=np.full_like(sse, np.nan),
        where=centered_energy > np.finfo(float).eps,
    )
    cosine_denom = np.linalg.norm(X, axis=1) * np.linalg.norm(
        reconstructed, axis=1
    )
    reconstruction_cosine = np.divide(
        (X * reconstructed).sum(axis=1),
        cosine_denom,
        out=np.zeros_like(sse),
        where=cosine_denom > np.finfo(float).eps,
    )
    weight_sum = weights.sum(axis=1)
    proportions = np.divide(
        weights,
        weight_sum[:, None],
        out=np.zeros_like(weights),
        where=weight_sum[:, None] > np.finfo(float).eps,
    )
    dominance = proportions.max(axis=1)
    safe = np.where(proportions > 0, proportions, 1.0)
    entropy = -(proportions * np.log(safe)).sum(axis=1) / np.log(weights.shape[1])
    best_index = weights.argmax(axis=1)
    names = np.asarray(component_names, dtype=object)
    best_component = names[best_index]

    table = metadata.reset_index().copy()
    for index, name in enumerate(component_names):
        table[f"loading_{name}"] = weights[:, index]
        table[f"proportion_{name}"] = proportions[:, index]
    table["best_component"] = best_component
    table["dominance"] = dominance
    table["loading_entropy"] = entropy
    table["residual_norm"] = np.sqrt(sse)
    table["explained_energy"] = explained_energy
    table["reconstruction_r2"] = reconstruction_r2
    table["reconstruction_cosine"] = reconstruction_cosine
    table["pilot_class"] = np.where(
        explained_energy >= pilot_min_explained_energy,
        best_component,
        "unmatched",
    )
    return table


def _qc_row_indices(
    table: pd.DataFrame,
    component_names: Sequence[str],
    *,
    per_class: int = 3,
) -> list[int]:
    chosen: list[int] = []
    for name in (*component_names, "unmatched"):
        subset = table.loc[table["pilot_class"].eq(name)]
        if subset.empty:
            continue
        subset = subset.sort_values("explained_energy", ascending=name != "unmatched")
        chosen.extend(subset.tail(per_class).index.tolist())
    return list(dict.fromkeys(chosen))


def plot_reconstruction_qc(
    table: pd.DataFrame,
    X: np.ndarray,
    reconstructed: np.ndarray,
    basis: TemporalBasis,
    path: Path,
    *,
    per_class: int = 3,
) -> Path:
    """Plot representative observed and reconstructed normalized waveforms."""

    indices = _qc_row_indices(table, basis.component_names, per_class=per_class)
    if not indices:
        raise ValueError("no electrodes available for reconstruction QC")
    n_cols = 3
    n_rows = int(np.ceil(len(indices) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 2.4 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    feature_axis = np.arange(basis.n_features)
    for axis, row_index in zip(axes.flat, indices):
        row = table.loc[row_index]
        axis.plot(feature_axis, X[row_index], color="0.15", linewidth=1.1, label="observed")
        component = str(row["best_component"])
        axis.plot(
            feature_axis,
            reconstructed[row_index],
            color=FUNCTION_COLORS.get(component, "#7A4EAB"),
            linewidth=1.2,
            label="reconstructed",
        )
        for boundary in basis.phase_boundaries:
            axis.axvline(boundary, color="0.75", linewidth=0.55)
        axis.set_title(
            f"{row['channel']} | {row.get('roi', 'NA')} | {row['pilot_class']}\n"
            f"energy={row['explained_energy']:.2f}, cos={row['reconstruction_cosine']:.2f}",
            fontsize=7,
        )
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.flat[len(indices) :]:
        axis.set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.supxlabel("Concatenated phase/time feature")
    fig.supylabel("Normalized positive HGA")
    fig.tight_layout()
    return save_svg(fig, path, close=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_whole_brain_projection(
    subjects: Iterable[str] | None = None,
    *,
    tasks: Sequence[str] = TASKS,
    phases: Sequence[str] = PHASES,
    description: str = DESCRIPTION,
    significance_mode: SignificanceMode = "mask-any",
    min_coverage: float = 0.95,
    min_tasks: int = 1,
    ridge: float = 0.0,
    pilot_min_explained_energy: float = 0.5,
    basis_path: Path | None = None,
    assignments_path: Path | None = None,
    results_root: Path = RESULTS_ROOT,
    output_dir: Path | None = None,
    image_dir: Path | None = None,
    make_plots: bool = True,
    exclude_subjects: Iterable[str] = DEFAULT_EXCLUDE_SUBJECTS,
) -> dict[str, object]:
    """Project requested subjects onto the frozen temporal NMF basis."""

    if not tasks:
        raise ValueError("at least one task is required")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    if min_tasks > len(tasks):
        raise ValueError("min_tasks cannot exceed the number of requested tasks")
    if not 0 <= pilot_min_explained_energy <= 1:
        raise ValueError("pilot_min_explained_energy must be in [0, 1]")
    phases = tuple(_normalize_phase(phase) for phase in phases)
    if subjects is None:
        normalized_subjects = discover_cohort_subjects(
            tasks,
            results_root=results_root,
            exclude_subjects=exclude_subjects,
        )
    else:
        exclude = {_normalize_subject(name) for name in exclude_subjects}
        normalized_subjects = tuple(
            dict.fromkeys(
                _normalize_subject(name)
                for name in subjects
                if _normalize_subject(name) not in exclude
            )
        )
    if not normalized_subjects:
        raise ValueError("at least one subject is required")
    basis = load_temporal_basis(basis_path)
    if phases != basis.phases:
        raise ValueError(
            f"projection phases must match frozen basis {basis.phases}, got {phases}"
        )
    discovery = _discovery_channels(assignments_path)
    output = Path(output_dir or (nmf_results_dir() / "whole_brain_projection"))
    images = Path(image_dir or (img_dir("nmf") / "whole_brain_projection"))
    output.mkdir(parents=True, exist_ok=True)
    if make_plots:
        images.mkdir(parents=True, exist_ok=True)

    subject_tables: list[pd.DataFrame] = []
    subject_detail: dict[str, object] = {}
    skipped: dict[str, str] = {}
    for subject in normalized_subjects:
        LOGGER.info("Loading whole-brain HGA for subject=%s", subject)
        try:
            rows, input_paths, missing_inputs = load_subject_hga_rows(
                subject,
                tasks,
                phases,
                description=description,
                results_root=results_root,
            )
            raw, metadata = build_subject_waveform_matrix(
                rows,
                basis,
                significance_mode=significance_mode,
                min_coverage=min_coverage,
                min_tasks=min_tasks,
            )
            X, keep = prepare_shape_matrix(raw.to_numpy(dtype=float))
            if not keep.any():
                raise ValueError("no electrodes remain after shape normalization")
            metadata = metadata.iloc[np.flatnonzero(keep)]
            weights, reconstructed = project_fixed_temporal_basis(
                X, basis.H, ridge=ridge
            )
            table = build_projection_table(
                metadata,
                X,
                weights,
                reconstructed,
                basis.component_names,
                pilot_min_explained_energy=pilot_min_explained_energy,
            )
        except (FileNotFoundError, ValueError) as exc:
            LOGGER.warning("Skipping subject=%s: %s", subject, exc)
            skipped[subject] = str(exc)
            subject_detail[subject] = {"skipped": True, "reason": str(exc)}
            continue
        table.insert(1, "significance_mode", significance_mode)
        table.insert(2, "in_discovery", table["channel"].astype(str).isin(discovery))
        subject_path = output / f"sub-{subject}_projection.csv"
        table.to_csv(subject_path, index=False)
        qc_path: str | None = None
        if make_plots:
            saved = plot_reconstruction_qc(
                table,
                X,
                reconstructed,
                basis,
                images / f"sub-{subject}_reconstruction.svg",
            )
            qc_path = str(saved.resolve())
        class_counts = table["pilot_class"].value_counts().to_dict()
        LOGGER.info(
            "Projected subject=%s electrodes=%d classes=%s",
            subject,
            len(table),
            class_counts,
        )
        subject_tables.append(table)
        subject_detail[subject] = {
            "n_electrodes": int(len(table)),
            "n_discovery": int(table["in_discovery"].sum()),
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "n_input_files": len(input_paths),
            "input_paths": input_paths,
            "missing_inputs": missing_inputs,
            "table": str(subject_path.resolve()),
            "qc_svg": qc_path,
        }

    if not subject_tables:
        raise RuntimeError(
            "no subjects produced a projection table; "
            f"skipped={list(skipped)}"
        )
    combined = pd.concat(subject_tables, ignore_index=True)
    combined_path = output / "electrode_projection.csv"
    combined.to_csv(combined_path, index=False)
    used = [name for name in normalized_subjects if name not in skipped]
    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "fixed_temporal_basis_nnls",
        "interpretation": "waveform motif projection; not pairwise connectivity",
        "subjects": list(normalized_subjects),
        "subjects_used": used,
        "subjects_skipped": skipped,
        "n_subjects_used": len(used),
        "n_subjects_skipped": len(skipped),
        "tasks": list(tasks),
        "phases": list(phases),
        "description": description,
        "significance_mode": significance_mode,
        "significance_definition": (
            "at least one significant mask sample in included post-onset data"
            if significance_mode == "mask-any"
            else "packaged subject-by-modality significant-channel union"
        ),
        "min_coverage": min_coverage,
        "min_tasks": min_tasks,
        "ridge": ridge,
        "pilot_min_explained_energy": pilot_min_explained_energy,
        "component_names": list(basis.component_names),
        "n_features": basis.n_features,
        "basis_path": str(basis.source_path),
        "basis_sha256": basis.source_sha256,
        "n_discovery_channels": len(discovery),
        "combined_table": str(combined_path.resolve()),
        "output_dir": str(output.resolve()),
        "image_dir": str(images.resolve()) if make_plots else None,
        "n_electrodes": int(len(combined)),
        "n_electrodes_discovery": int(combined["in_discovery"].sum()),
        "subjects_detail": subject_detail,
        "caveat": (
            "pilot_class uses a diagnostic fit threshold, not an inferential "
            "significance threshold"
        ),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest
