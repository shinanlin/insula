"""Select OAEC electrodes to overlay on focused motif territories.

Significant Insula→partner pairs are reduced to unique electrodes.  Seeds are
sources; partners are targets that never appear as a source.  Focused-territory
membership is nearest pial vertex inside a focused display mask.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.nmf.waveform_analysis import CLUSTER_ORDER, FUNCTION_COLORS

DEFAULT_COMPONENTS = tuple(CLUSTER_ORDER)
HEMI_TO_SURF = {"L": "lh", "R": "rh", "lh": "lh", "rh": "rh"}
PHASE_ORDER = ("stimulus", "delay", "go", "response")
PHASE_EPOCHS = {
    "stimulus_delay": ("stimulus", "delay"),
    "go_response": ("go", "response"),
}
UNASSIGNED_SEED_COLOR = "#D4AF37"
UNASSIGNED_PARTNER_COLOR = "#BDBDBD"


def select_overlay_electrodes(
    pairs: pd.DataFrame,
    channel_meta: pd.DataFrame,
    projection: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per unique electrode on significant OAEC pairs.

    ``pairs`` must contain ``source_channel``, ``target_channel``, and
    ``pair_key``.  Optional ``subject`` is used only for coverage counts.
    """

    required = {"source_channel", "target_channel", "pair_key"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pair table missing columns: {sorted(missing)}")
    if "channel" not in channel_meta.columns:
        raise ValueError("channel_meta must contain a channel column")

    work = pairs.loc[:, list(required | ({"subject"} & set(pairs.columns)))].copy()
    work["source_channel"] = work["source_channel"].astype(str)
    work["target_channel"] = work["target_channel"].astype(str)
    work["pair_key"] = work["pair_key"].astype(str)
    work = work.drop_duplicates(subset=["source_channel", "target_channel", "pair_key"])

    sources = set(work["source_channel"])
    targets = set(work["target_channel"])
    seed_counts = (
        work.groupby("source_channel", sort=False)["pair_key"]
        .nunique()
        .rename("n_pairs")
    )
    partner_counts = (
        work.groupby("target_channel", sort=False)["pair_key"]
        .nunique()
        .rename("n_pairs")
    )
    seed_subjects = (
        work.groupby("source_channel")["subject"].nunique()
        if "subject" in work.columns
        else None
    )
    partner_subjects = (
        work.groupby("target_channel")["subject"].nunique()
        if "subject" in work.columns
        else None
    )

    rows: list[dict[str, object]] = []
    for channel in sorted(sources):
        row: dict[str, object] = {
            "channel": channel,
            "role": "seed",
            "n_pairs": int(seed_counts.loc[channel]),
        }
        if seed_subjects is not None:
            row["n_subjects"] = int(seed_subjects.loc[channel])
        rows.append(row)
    for channel in sorted(targets - sources):
        row = {
            "channel": channel,
            "role": "partner",
            "n_pairs": int(partner_counts.loc[channel]),
        }
        if partner_subjects is not None:
            row["n_subjects"] = int(partner_subjects.loc[channel])
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table

    meta = channel_meta.copy()
    meta["channel"] = meta["channel"].astype(str)
    meta = meta.drop_duplicates(subset=["channel"], keep="first")
    keep_meta = [
        column
        for column in ("channel", "roi", "hemi", "x", "y", "z", "functional_cluster")
        if column in meta.columns
    ]
    table = table.merge(meta[keep_meta], on="channel", how="left")
    if "functional_cluster" in table.columns:
        table = table.rename(columns={"functional_cluster": "seed_cluster"})
    else:
        table["seed_cluster"] = pd.NA
    table.loc[table["role"].ne("seed"), "seed_cluster"] = pd.NA

    table["best_component"] = pd.NA
    table["explained_energy"] = np.nan
    table["in_discovery"] = False
    if projection is not None and not projection.empty and "channel" in projection.columns:
        proj = projection.copy()
        proj["channel"] = proj["channel"].astype(str)
        proj = proj.drop_duplicates(subset=["channel"], keep="first")
        proj_cols = [
            column
            for column in (
                "channel",
                "best_component",
                "explained_energy",
                "in_discovery",
            )
            if column in proj.columns
        ]
        table = table.drop(columns=["best_component", "explained_energy", "in_discovery"])
        table = table.merge(proj[proj_cols], on="channel", how="left")
        if "in_discovery" in table.columns:
            table["in_discovery"] = (
                table["in_discovery"]
                .astype("string")
                .str.lower()
                .isin({"true", "1", "yes"})
            )
        else:
            table["in_discovery"] = False
        if "explained_energy" not in table.columns:
            table["explained_energy"] = np.nan
        if "best_component" not in table.columns:
            table["best_component"] = pd.NA

    table.loc[table["role"].eq("seed"), "best_component"] = pd.NA
    return table.reset_index(drop=True)


def restrict_overlay_to_focused_partners(
    pairs: pd.DataFrame,
    electrodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep in-territory partners and insula seeds that connect to them.

    Partners outside the focused masks are dropped.  Seeds that only connect to
    those extra-cluster partners are dropped with them.  Pair counts are
    recomputed on the retained edges.
    """

    required = {"source_channel", "target_channel", "pair_key"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pair table missing columns: {sorted(missing)}")
    if "channel" not in electrodes.columns or "role" not in electrodes.columns:
        raise ValueError("electrode table must contain channel and role")
    if "in_focused" not in electrodes.columns:
        raise ValueError("electrode table must contain in_focused")

    focused_partners = set(
        electrodes.loc[
            electrodes["role"].eq("partner") & electrodes["in_focused"].eq(True),
            "channel",
        ].astype(str)
    )
    keep_pairs = pairs.loc[
        pairs["target_channel"].astype(str).isin(focused_partners)
    ].copy()
    keep_channels = set(keep_pairs["source_channel"].astype(str)) | set(
        keep_pairs["target_channel"].astype(str)
    )
    keep_electrodes = electrodes.loc[
        electrodes["channel"].astype(str).isin(keep_channels)
    ].copy()
    if keep_electrodes.empty:
        return keep_pairs.reset_index(drop=True), keep_electrodes.reset_index(drop=True)

    seed_counts = (
        keep_pairs.groupby("source_channel", sort=False)["pair_key"].nunique()
        if not keep_pairs.empty
        else pd.Series(dtype=int)
    )
    partner_counts = (
        keep_pairs.groupby("target_channel", sort=False)["pair_key"].nunique()
        if not keep_pairs.empty
        else pd.Series(dtype=int)
    )
    keep_electrodes["n_pairs"] = keep_electrodes.apply(
        lambda row: int(
            seed_counts.get(str(row["channel"]), 0)
            if row["role"] == "seed"
            else partner_counts.get(str(row["channel"]), 0)
        ),
        axis=1,
    )
    if "subject" in keep_pairs.columns:
        seed_subjects = keep_pairs.groupby("source_channel")["subject"].nunique()
        partner_subjects = keep_pairs.groupby("target_channel")["subject"].nunique()
        keep_electrodes["n_subjects"] = keep_electrodes.apply(
            lambda row: int(
                seed_subjects.get(str(row["channel"]), 0)
                if row["role"] == "seed"
                else partner_subjects.get(str(row["channel"]), 0)
            ),
            axis=1,
        )
    return keep_pairs.reset_index(drop=True), keep_electrodes.reset_index(drop=True)


def mark_partner_significance(pairs: pd.DataFrame, electrodes: pd.DataFrame) -> pd.DataFrame:
    """Flag extra-insula partners that have at least one significant OAEC pair."""

    if "target_channel" not in pairs.columns:
        raise ValueError("pair table must contain target_channel")
    if "sig_within_pair" not in pairs.columns:
        raise ValueError("pair table must contain sig_within_pair")
    out = electrodes.copy()
    out["has_sig"] = True
    partner = out["role"].eq("partner")
    sig_targets = set(
        pairs.loc[pairs["sig_within_pair"].astype(bool), "target_channel"].astype(str)
    )
    out.loc[partner, "has_sig"] = out.loc[partner, "channel"].astype(str).isin(sig_targets)
    return out


def _phase_column(pairs: pd.DataFrame) -> str:
    if "phase_norm" in pairs.columns:
        return "phase_norm"
    if "phase" in pairs.columns:
        return "phase"
    raise ValueError("pair table must contain phase_norm or phase")


def filter_pairs_for_seed_cluster(
    pairs: pd.DataFrame,
    cluster: str,
    *,
    phase: str | None = None,
    sig_only: bool = True,
    components: Sequence[str] = DEFAULT_COMPONENTS,
) -> pd.DataFrame:
    """Keep Insula→partner rows for one seed cluster, optionally one phase."""

    required = {"source_channel", "target_channel"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pair table missing columns: {sorted(missing)}")
    work = pairs.copy()
    if "source_cluster" not in work.columns:
        raise ValueError("pair table must contain source_cluster")
    work["source_cluster"] = work["source_cluster"].astype("string")
    if cluster not in set(components):
        raise ValueError(f"unknown cluster {cluster!r}; expected one of {tuple(components)}")
    work = work.loc[work["source_cluster"].eq(cluster)].copy()
    if phase is not None:
        column = _phase_column(work)
        work = work.loc[work[column].astype(str).str.lower().str.strip().eq(phase.lower())]
    if sig_only and "sig_within_pair" in work.columns:
        work = work.loc[work["sig_within_pair"].astype(bool)].copy()
    return work.reset_index(drop=True)


def filter_pairs_for_any_insula_cluster(
    pairs: pd.DataFrame,
    *,
    phase: str | None = None,
    sig_only: bool = True,
    components: Sequence[str] = DEFAULT_COMPONENTS,
) -> pd.DataFrame:
    """Keep Insula→partner rows to any labeled insula cluster, optionally one phase."""

    required = {"source_channel", "target_channel"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pair table missing columns: {sorted(missing)}")
    work = pairs.copy()
    if "source_cluster" not in work.columns:
        raise ValueError("pair table must contain source_cluster")
    work["source_cluster"] = work["source_cluster"].astype("string")
    work = work.loc[work["source_cluster"].isin(tuple(components))].copy()
    if phase is not None:
        column = _phase_column(work)
        work = work.loc[work[column].astype(str).str.lower().str.strip().eq(phase.lower())]
    if sig_only and "sig_within_pair" in work.columns:
        work = work.loc[work["sig_within_pair"].astype(bool)].copy()
    return work.reset_index(drop=True)


def assign_territory_plot_colors(electrodes: pd.DataFrame) -> pd.DataFrame:
    """Color seeds by insula cluster and partners by the territory they sit in."""

    out = electrodes.copy()
    colors: list[str] = []
    for _, row in out.iterrows():
        if row["role"] == "seed":
            cluster = row.get("seed_cluster")
            if pd.notna(cluster) and str(cluster) in FUNCTION_COLORS:
                colors.append(FUNCTION_COLORS[str(cluster)])
            else:
                colors.append(UNASSIGNED_SEED_COLOR)
            continue
        name = None
        for column in ("focused_component", "best_component"):
            if column in row.index and pd.notna(row[column]):
                value = str(row[column])
                if value in FUNCTION_COLORS:
                    name = value
                    break
        colors.append(FUNCTION_COLORS[name] if name else UNASSIGNED_PARTNER_COLOR)
    out["plot_color"] = colors
    return out


def electrodes_for_cluster_phase(
    electrodes: pd.DataFrame,
    pairs: pd.DataFrame,
    cluster: str,
    phase: str,
    *,
    focused_partners_only: bool = True,
) -> pd.DataFrame:
    """All seeds of ``cluster`` plus partners with a significant pair in ``phase``.

    A partner that also couples to another insula cluster is still kept.  Seeds
    are stable across phases so the four panels share the same insula dots.
    """

    if "channel" not in electrodes.columns or "role" not in electrodes.columns:
        raise ValueError("electrode table must contain channel and role")
    if "seed_cluster" not in electrodes.columns:
        raise ValueError("electrode table must contain seed_cluster")

    seeds = electrodes.loc[
        electrodes["role"].eq("seed")
        & electrodes["seed_cluster"].astype(str).eq(cluster)
    ].copy()
    phase_pairs = filter_pairs_for_seed_cluster(
        pairs, cluster, phase=phase, sig_only=True
    )
    partner_ids = set(phase_pairs["target_channel"].astype(str))
    partners = electrodes.loc[
        electrodes["role"].eq("partner")
        & electrodes["channel"].astype(str).isin(partner_ids)
    ].copy()
    if focused_partners_only and "in_focused" in partners.columns:
        partners = partners.loc[partners["in_focused"].eq(True)].copy()
    if not partners.empty:
        partners["has_sig"] = True
    out = pd.concat([seeds, partners], ignore_index=True)
    if out.empty:
        return out
    return assign_territory_plot_colors(out)


def electrodes_for_any_insula_phase(
    electrodes: pd.DataFrame,
    pairs: pd.DataFrame,
    phase: str,
    *,
    focused_partners_only: bool = True,
) -> pd.DataFrame:
    """Focused partners with a significant pair to any labeled insula seed in ``phase``.

    Insula seeds are omitted.  Partner color follows the waveform territory.
    """

    if "channel" not in electrodes.columns or "role" not in electrodes.columns:
        raise ValueError("electrode table must contain channel and role")
    phase_pairs = filter_pairs_for_any_insula_cluster(pairs, phase=phase, sig_only=True)
    partner_ids = set(phase_pairs["target_channel"].astype(str))
    partners = electrodes.loc[
        electrodes["role"].eq("partner")
        & electrodes["channel"].astype(str).isin(partner_ids)
    ].copy()
    if focused_partners_only and "in_focused" in partners.columns:
        partners = partners.loc[partners["in_focused"].eq(True)].copy()
    if partners.empty:
        return partners
    partners["has_sig"] = True
    return assign_territory_plot_colors(partners)


def union_cluster_phase_partners(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Merge per-cluster phase electrode tables into unique partners per phase."""

    frames = []
    for table in tables:
        if table is None or table.empty:
            continue
        if "role" not in table.columns:
            raise ValueError("cluster phase table must contain role")
        if "phase" not in table.columns:
            raise ValueError("cluster phase table must contain phase")
        frames.append(table.loc[table["role"].eq("partner")].copy())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["channel"] = out["channel"].astype(str)
    out["phase"] = out["phase"].astype(str).str.lower().str.strip()
    out = out.drop_duplicates(subset=["phase", "channel"], keep="first")
    if out.empty:
        return out
    return assign_territory_plot_colors(out)


def collapse_phase_partners(
    table: pd.DataFrame,
    epochs: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Union unique partners across phases inside each epoch."""

    if table.empty:
        return table.copy()
    if "phase" not in table.columns:
        raise ValueError("table must contain phase")
    groups = epochs if epochs is not None else PHASE_EPOCHS
    work = table.copy()
    if "role" in work.columns:
        work = work.loc[work["role"].ne("seed")].copy()
    work["phase"] = work["phase"].astype(str).str.lower().str.strip()
    work["channel"] = work["channel"].astype(str)
    rows = []
    for name, phases in groups.items():
        wanted = {str(phase).lower() for phase in phases}
        part = work.loc[work["phase"].isin(wanted)].copy()
        if part.empty:
            continue
        part = part.drop_duplicates(subset=["channel"], keep="first")
        part["phase"] = name
        rows.append(part)
    if not rows:
        return work.iloc[0:0].copy()
    out = pd.concat(rows, ignore_index=True)
    return assign_territory_plot_colors(out)


def assign_partner_insula_cluster(
    pairs: pd.DataFrame,
    electrodes: pd.DataFrame,
    *,
    components: Sequence[str] = DEFAULT_COMPONENTS,
) -> pd.DataFrame:
    """Label extra-insula partners by the insula NMF cluster they couple to.

    Significant pairs take precedence.  If a partner has significant OAEC to
    more than one insula cluster, the cluster with the most unique pairs wins;
    ties follow ``components`` order.
    """

    required = {"source_channel", "target_channel", "pair_key"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pair table missing columns: {sorted(missing)}")
    out = electrodes.copy()
    out["insula_cluster"] = pd.NA
    if out.empty or pairs.empty:
        return out

    work = pairs.copy()
    work["source_channel"] = work["source_channel"].astype(str)
    work["target_channel"] = work["target_channel"].astype(str)
    if "source_cluster" not in work.columns:
        seed_map = (
            out.loc[out["role"].eq("seed"), ["channel", "seed_cluster"]]
            .dropna(subset=["seed_cluster"])
            .drop_duplicates("channel")
            .set_index("channel")["seed_cluster"]
        )
        work["source_cluster"] = work["source_channel"].map(seed_map)
    work["source_cluster"] = work["source_cluster"].astype("string")
    work = work.loc[work["source_cluster"].isin(tuple(components))].copy()
    if work.empty:
        return out
    if "sig_within_pair" in work.columns:
        work["sig_within_pair"] = work["sig_within_pair"].astype(bool)
    else:
        work["sig_within_pair"] = False

    def cluster_mode(frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            return pd.Series(dtype="string")
        counts = (
            frame.groupby(["target_channel", "source_cluster"], observed=True)["pair_key"]
            .nunique()
            .reset_index(name="n")
        )
        counts["source_cluster"] = pd.Categorical(
            counts["source_cluster"], list(components), ordered=True
        )
        counts = counts.sort_values(
            ["target_channel", "n", "source_cluster"],
            ascending=[True, False, True],
        )
        return (
            counts.drop_duplicates("target_channel")
            .set_index("target_channel")["source_cluster"]
            .astype("string")
        )

    preferred = cluster_mode(work.loc[work["sig_within_pair"]])
    fallback = cluster_mode(work)
    partner_idx = out.index[out["role"].eq("partner")]
    channels = out.loc[partner_idx, "channel"].astype(str)
    out.loc[partner_idx, "insula_cluster"] = channels.map(fallback).to_numpy()
    sig_mapped = channels.map(preferred)
    take = sig_mapped.notna().to_numpy()
    if take.any():
        out.loc[partner_idx[take], "insula_cluster"] = sig_mapped.to_numpy()[take]
    return out


def assign_focused_membership(
    electrodes: pd.DataFrame,
    *,
    vertices_by_hemi: Mapping[str, np.ndarray],
    mask_by_hemi: Mapping[str, np.ndarray],
    specificity_by_hemi: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    max_distance: float = 15.0,
) -> pd.DataFrame:
    """Mark electrodes whose nearest pial vertex lies in a focused mask."""

    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    out = electrodes.copy()
    if "hemi" not in out.columns:
        out["hemi"] = pd.NA
    out["in_focused"] = False
    out["focused_component"] = pd.NA
    out["vertex_index"] = -1
    out["vertex_distance"] = np.nan
    out["surf_hemi"] = out["hemi"].map(
        lambda value: HEMI_TO_SURF.get(str(value).strip()) if pd.notna(value) else None
    )
    if out.empty:
        return out

    trees = {
        hemi: cKDTree(np.asarray(vertices, dtype=float))
        for hemi, vertices in vertices_by_hemi.items()
    }
    xyz_ok = out["x"].notna() & out["y"].notna() & out["z"].notna()
    missing_hemi = xyz_ok & out["surf_hemi"].isna()
    out.loc[missing_hemi & out["x"].lt(0), "surf_hemi"] = "lh"
    out.loc[missing_hemi & out["x"].ge(0), "surf_hemi"] = "rh"
    for row_index, row in out.loc[xyz_ok].iterrows():
        hemi = row.get("surf_hemi")
        if hemi not in trees or hemi not in mask_by_hemi:
            continue
        xyz = np.asarray([row["x"], row["y"], row["z"]], dtype=float)
        distance, vertex = trees[hemi].query(xyz)
        out.at[row_index, "vertex_index"] = int(vertex)
        out.at[row_index, "vertex_distance"] = float(distance)
        if not np.isfinite(distance) or distance > max_distance:
            continue
        mask = np.asarray(mask_by_hemi[hemi], dtype=bool)
        if vertex >= len(mask) or not mask[vertex]:
            continue
        out.at[row_index, "in_focused"] = True
        if specificity_by_hemi is None:
            continue
        best_name = None
        best_value = -np.inf
        for component in components:
            component_mask = np.asarray(
                specificity_by_hemi[hemi].get(f"{component}_mask", mask),
                dtype=bool,
            )
            values = specificity_by_hemi[hemi].get(component)
            if values is None or vertex >= len(values) or not component_mask[vertex]:
                continue
            value = float(values[vertex])
            if np.isfinite(value) and value > best_value:
                best_value = value
                best_name = component
        if best_name is not None:
            out.at[row_index, "focused_component"] = best_name
    return out


def focused_mask_payload(
    payload: Mapping[str, np.ndarray],
    *,
    components: Sequence[str] = DEFAULT_COMPONENTS,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Union display masks and per-template specificity from a focused cache."""

    mask_by_hemi: dict[str, np.ndarray] = {}
    specificity_by_hemi: dict[str, dict[str, np.ndarray]] = {}
    for hemi in ("lh", "rh"):
        masks = []
        spec: dict[str, np.ndarray] = {}
        for component in components:
            key = f"{hemi}_{component}_display_mask"
            spec_key = f"{hemi}_{component}_specificity"
            if key not in payload:
                continue
            mask = np.asarray(payload[key], dtype=bool)
            masks.append(mask)
            spec[f"{component}_mask"] = mask
            if spec_key in payload:
                spec[component] = np.asarray(payload[spec_key], dtype=float)
        if not masks:
            continue
        mask_by_hemi[hemi] = np.logical_or.reduce(masks)
        specificity_by_hemi[hemi] = spec
    return mask_by_hemi, specificity_by_hemi


def annotate_motif_pairs(
    pairs: pd.DataFrame,
    electrodes: pd.DataFrame,
    projection: pd.DataFrame | None = None,
    *,
    components: Sequence[str] = DEFAULT_COMPONENTS,
) -> pd.DataFrame:
    """Label Insula→partner pairs by seed NMF cluster and partner motif.

    Keeps pairs whose target is a focused extra-insula partner and whose
    source cluster and partner ``best_component`` are in ``components``.
    """

    required = {"source_channel", "target_channel", "pair_key"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pair table missing columns: {sorted(missing)}")
    focused = set(
        electrodes.loc[
            electrodes["role"].eq("partner") & electrodes["in_focused"].eq(True),
            "channel",
        ].astype(str)
    )
    work = pairs.copy()
    work["source_channel"] = work["source_channel"].astype(str)
    work["target_channel"] = work["target_channel"].astype(str)
    work = work.loc[work["target_channel"].isin(focused)].copy()
    if "source_cluster" not in work.columns:
        seed_map = (
            electrodes.loc[electrodes["role"].eq("seed"), ["channel", "seed_cluster"]]
            .dropna(subset=["seed_cluster"])
            .drop_duplicates("channel")
            .set_index("channel")["seed_cluster"]
        )
        work["source_cluster"] = work["source_channel"].map(seed_map)
    if projection is not None and "best_component" in projection.columns:
        proj = projection.copy()
        proj["channel"] = proj["channel"].astype(str)
        proj = proj.drop_duplicates("channel")
        work = work.merge(
            proj[["channel", "best_component"]].rename(
                columns={
                    "channel": "target_channel",
                    "best_component": "partner_component",
                }
            ),
            on="target_channel",
            how="left",
        )
    if "partner_component" not in work.columns:
        work["partner_component"] = pd.NA
    if "best_component" in electrodes.columns:
        partner_map = (
            electrodes.loc[electrodes["role"].eq("partner"), ["channel", "best_component"]]
            .drop_duplicates("channel")
            .set_index("channel")["best_component"]
        )
        missing = work["partner_component"].isna()
        work.loc[missing, "partner_component"] = work.loc[missing, "target_channel"].map(
            partner_map
        )
    components = tuple(components)
    work["source_cluster"] = work["source_cluster"].astype("string")
    work["partner_component"] = work["partner_component"].astype("string")
    work = work.loc[
        work["source_cluster"].isin(components)
        & work["partner_component"].isin(components)
    ].copy()
    return work.reset_index(drop=True)


def subject_motif_coupling(
    pairs: pd.DataFrame,
    *,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    min_pairs: int = 1,
    stat_column: str = "stat",
) -> pd.DataFrame:
    """Per-subject hit-rate and mean Fisher-z for each seed × partner motif cell."""

    if min_pairs < 1:
        raise ValueError("min_pairs must be at least 1")
    required = {
        "subject",
        "source_cluster",
        "partner_component",
        "pair_key",
        "sig_within_pair",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"motif pair table missing columns: {sorted(missing)}")
    work = pairs.copy()
    work["sig_within_pair"] = work["sig_within_pair"].astype(bool)
    grouped = work.groupby(
        ["subject", "source_cluster", "partner_component"], observed=True, sort=False
    )
    n_pairs = grouped["pair_key"].nunique().rename("n_pairs")
    n_sig = (
        work.loc[work["sig_within_pair"]]
        .groupby(["subject", "source_cluster", "partner_component"], observed=True)[
            "pair_key"
        ]
        .nunique()
        .rename("n_sig_pairs")
    )
    out = n_pairs.to_frame().join(n_sig, how="left")
    out["n_sig_pairs"] = out["n_sig_pairs"].fillna(0).astype(int)
    out["pair_hit"] = out["n_sig_pairs"] / out["n_pairs"]
    if stat_column in work.columns:
        mean_z = grouped[stat_column].mean().rename("mean_z")
        out = out.join(mean_z, how="left")
    else:
        out["mean_z"] = np.nan
    out = out.reset_index()
    out = out.loc[out["n_pairs"] >= min_pairs].copy()
    out["source_cluster"] = pd.Categorical(
        out["source_cluster"], list(components), ordered=True
    )
    out["partner_component"] = pd.Categorical(
        out["partner_component"], list(components), ordered=True
    )
    out["within_motif"] = out["source_cluster"].astype(str).eq(
        out["partner_component"].astype(str)
    )
    return out.sort_values(["subject", "source_cluster", "partner_component"]).reset_index(
        drop=True
    )


def group_motif_coupling(subject_cells: pd.DataFrame) -> pd.DataFrame:
    """Average subject-level cell summaries (unweighted across subjects)."""

    if subject_cells.empty:
        return subject_cells.copy()
    grouped = subject_cells.groupby(
        ["source_cluster", "partner_component"], observed=True, sort=True
    )
    out = grouped.agg(
        n_subjects=("subject", "nunique"),
        n_pairs=("n_pairs", "sum"),
        n_sig_pairs=("n_sig_pairs", "sum"),
        mean_pair_hit=("pair_hit", "mean"),
        median_pair_hit=("pair_hit", "median"),
        mean_z=("mean_z", "mean"),
    ).reset_index()
    out["within_motif"] = out["source_cluster"].astype(str).eq(
        out["partner_component"].astype(str)
    )
    return out


def subject_within_cross(pairs: pd.DataFrame) -> pd.DataFrame:
    """Pool within-motif vs cross-motif pairs inside each subject."""

    required = {"subject", "source_cluster", "partner_component", "pair_key", "sig_within_pair"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"motif pair table missing columns: {sorted(missing)}")
    work = pairs.copy()
    work["within_motif"] = work["source_cluster"].astype(str).eq(
        work["partner_component"].astype(str)
    )
    work["sig_within_pair"] = work["sig_within_pair"].astype(bool)
    rows = []
    for subject, sub in work.groupby("subject", sort=True):
        row: dict[str, object] = {"subject": subject}
        for name, mask in (
            ("within", sub["within_motif"]),
            ("cross", ~sub["within_motif"]),
        ):
            part = sub.loc[mask]
            n_pairs = int(part["pair_key"].nunique())
            n_sig = int(part.loc[part["sig_within_pair"], "pair_key"].nunique())
            row[f"{name}_n_pairs"] = n_pairs
            row[f"{name}_n_sig"] = n_sig
            row[f"{name}_hit"] = n_sig / n_pairs if n_pairs else np.nan
            if "stat" in part.columns and n_pairs:
                row[f"{name}_z"] = float(part["stat"].mean())
            else:
                row[f"{name}_z"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def partners_by_motif_cell(pairs: pd.DataFrame) -> pd.DataFrame:
    """Unique focused partners in each insula-cluster × partner-motif cell.

    If ``sig_within_pair`` is present, only significant pairs are kept.
    """

    required = {
        "source_cluster",
        "partner_component",
        "target_channel",
        "pair_key",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"motif pair table missing columns: {sorted(missing)}")
    work = pairs.copy()
    if "sig_within_pair" in work.columns:
        work = work.loc[work["sig_within_pair"].astype(bool)].copy()
    grouped = work.groupby(
        ["source_cluster", "partner_component", "target_channel"],
        observed=True,
        sort=True,
    )
    agg = {"n_pairs": ("pair_key", "nunique")}
    if "subject" in work.columns:
        agg["n_subjects"] = ("subject", "nunique")
    return grouped.agg(**agg).reset_index()
