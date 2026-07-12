#!/usr/bin/env python3
"""QA summary for HGA Explorer exports (split and multi-atlas layouts)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

VIEWER_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = VIEWER_ROOT / "public" / "data"
NATIVE_MESH_DIR = VIEWER_ROOT / "public" / "assets" / "native"


def summarize_electrodes(electrodes: list[dict], meta: dict, label: str) -> list[str]:
    issues: list[str] = []
    null_hga = sum(
        1
        for electrode in electrodes
        if all(value is None for value in (electrode.get("hga_by_task") or {}).values())
    )
    missing_endpoints = sum(
        1
        for electrode in electrodes
        if not electrode.get("contact_1") or electrode.get("x1_native") is None
    )

    print(f"=== {label} ===")
    print(f"Subjects: {meta.get('subjects')}")
    print(f"Tasks: {meta.get('tasks')}")
    print(f"Electrodes: {len(electrodes)}")
    print(f"HGA scale: {meta.get('hga_size_scale')}")
    print(f"Null hga_by_task electrodes: {null_hga}")
    print(f"Missing endpoint fields: {missing_endpoints}")
    if missing_endpoints:
        issues.append(f"{missing_endpoints} electrodes missing endpoint coords")

    maper_hits = [
        electrode["id"]
        for electrode in electrodes
        if any(key.startswith("maper_") for key in electrode)
    ]
    if maper_hits:
        issues.append(f"maper_* fields present on {len(maper_hits)} electrodes")

    return issues


def qa_hammers_insula_spot_check(electrodes: list[dict], atlas_id: str) -> list[str]:
    issues: list[str] = []
    if atlas_id != "hammers":
        return issues
    aic_pic = [
        electrode
        for electrode in electrodes
        if electrode.get("roi") in {"AIC", "PIC"} and not electrode.get("mix", False)
    ]
    mixed = sum(1 for electrode in electrodes if electrode.get("mix"))
    print(f"  hammers insula (AIC/PIC, not mix): {len(aic_pic)} electrodes")
    print(f"  hammers mixed contacts: {mixed}")
    if not aic_pic:
        issues.append("hammers: no pure AIC/PIC electrodes found")
    return issues


def qa_task_all_spot_check(electrodes: list[dict], tasks: list[str]) -> list[str]:
    issues: list[str] = []
    partial = [
        electrode
        for electrode in electrodes
        if sum(1 for task in tasks if electrode.get("hga_by_task", {}).get(task) is not None) == 1
    ]
    if not partial:
        print("task=all spot check: no partial-task electrodes found")
        return issues

    sample = partial[0]
    values = [
        sample.get("hga_by_task", {}).get(task)
        for task in tasks
        if sample.get("hga_by_task", {}).get(task) is not None
    ]
    if not values:
        issues.append("task=all spot check: sample electrode has no HGA values")
        return issues

    expected_mean = sum(values) / len(values)
    print(
        f"task=all spot check: {sample['id']} partial tasks={len(values)} "
        f"mean_hga={expected_mean:.4f}"
    )
    return issues


def qa_native_meshes(subjects: list[str]) -> list[str]:
    issues: list[str] = []
    if not NATIVE_MESH_DIR.is_dir():
        issues.append(f"missing native mesh directory: {NATIVE_MESH_DIR}")
        return issues

    index_path = NATIVE_MESH_DIR / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        available = set(index.get("subjects") or [])
    else:
        available = {
            path.name.replace("_pial.glb", "")
            for path in NATIVE_MESH_DIR.glob("*_pial.glb")
        }

    missing = [subject for subject in subjects if subject not in available]
    print(f"Native meshes available: {len(available)} | missing for export subjects: {len(missing)}")
    if missing:
        print(f"  missing: {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")
    return issues


def qa_shared_assets(data_dir: Path, manifest: dict, meta: dict) -> list[str]:
    issues: list[str] = []
    subjects = meta.get("subjects") or []
    shared_files = manifest.get("shared", {}).get("files") or manifest.get("files") or {}

    for subject in subjects:
        trace_rel = shared_files["traces"][subject]
        trace_path = data_dir / trace_rel
        trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
        n_traces = len(trace_payload.get("traces") or {})
        print(f"  {trace_rel}: {n_traces} electrodes")

        for phase in meta.get("phases") or []:
            anim_rel = shared_files["animation"][subject][phase]
            anim_path = data_dir / anim_rel
            anim_payload = json.loads(anim_path.read_text(encoding="utf-8"))
            bundle = anim_payload["bundles"]["all|Repeat"]
            print(
                f"  {anim_rel}: "
                f"{len(bundle.get('times') or [])} frames, "
                f"{len(bundle.get('electrode_ids') or [])} electrodes"
            )

    return issues


def qa_multi_atlas_layout(data_dir: Path) -> int:
    issues: list[str] = []
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    meta = manifest["metadata"]
    tasks = meta.get("tasks") or []
    subjects = meta.get("subjects") or []

    print(
        f"Manifest version: {manifest.get('version')} | "
        f"layout: {manifest.get('layout')} | "
        f"default_atlas: {manifest.get('default_atlas')}"
    )
    print(f"Atlases: {manifest.get('atlases')}")

    issues.extend(qa_native_meshes(subjects))

    template_mesh = VIEWER_ROOT / "public" / "assets" / "cvs_avg35_pial.glb"
    if not template_mesh.is_file():
        issues.append(f"missing template mesh: {template_mesh}")
    else:
        print(f"Template mesh: {template_mesh} ({template_mesh.stat().st_size / 1e6:.2f} MB)")

    print("\n--- Shared traces/animation ---")
    issues.extend(qa_shared_assets(data_dir, manifest, meta))

    for atlas_id, atlas_entry in (manifest.get("atlas") or {}).items():
        electrodes_rel = atlas_entry["files"]["electrodes"]
        electrodes_payload = json.loads((data_dir / electrodes_rel).read_text(encoding="utf-8"))
        electrodes = electrodes_payload["electrodes"]
        atlas_meta = {**meta, **(atlas_entry.get("metadata") or {})}
        issues.extend(
            summarize_electrodes(electrodes, atlas_meta, f"Atlas {atlas_id}: {data_dir}")
        )
        issues.extend(qa_task_all_spot_check(electrodes, tasks))
        issues.extend(qa_hammers_insula_spot_check(electrodes, atlas_id))

        for subject in subjects:
            kde_rel = atlas_entry["files"]["kde_roi_mean"][subject]
            kde_payload = json.loads((data_dir / kde_rel).read_text(encoding="utf-8"))
            print(f"  {kde_rel}: {len(kde_payload.get('sources') or [])} ROI sources")

    if issues:
        print("\nQA issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("\nQA passed.")
    return 0


def qa_split_layout(data_dir: Path) -> int:
    issues: list[str] = []
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    electrodes_payload = json.loads((data_dir / "electrodes.json").read_text(encoding="utf-8"))
    meta = manifest["metadata"]
    electrodes = electrodes_payload["electrodes"]
    tasks = meta.get("tasks") or []

    issues.extend(summarize_electrodes(electrodes, meta, f"Split layout: {data_dir}"))
    issues.extend(qa_task_all_spot_check(electrodes, tasks))
    issues.extend(qa_native_meshes(meta.get("subjects") or []))

    template_mesh = VIEWER_ROOT / "public" / "assets" / "cvs_avg35_pial.glb"
    if not template_mesh.is_file():
        issues.append(f"missing template mesh: {template_mesh}")
    else:
        print(f"Template mesh: {template_mesh} ({template_mesh.stat().st_size / 1e6:.2f} MB)")

    subjects = meta.get("subjects") or []
    print(f"Manifest version: {manifest.get('version')} | layout: {manifest.get('layout')}")
    for subject in subjects:
        trace_path = data_dir / manifest["files"]["traces"][subject]
        trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
        n_traces = len(trace_payload.get("traces") or {})
        print(f"  traces/{subject}.json: {n_traces} electrodes")

        for phase in meta.get("phases") or []:
            anim_path = data_dir / manifest["files"]["animation"][subject][phase]
            anim_payload = json.loads(anim_path.read_text(encoding="utf-8"))
            bundle = anim_payload["bundles"]["all|Repeat"]
            print(
                f"  animation/{subject}/{phase}.json: "
                f"{len(bundle.get('times') or [])} frames, "
                f"{len(bundle.get('electrode_ids') or [])} electrodes"
            )

        kde_path = data_dir / manifest["files"]["kde_roi_mean"][subject]
        kde_payload = json.loads(kde_path.read_text(encoding="utf-8"))
        print(f"  kde/roi/{subject}/mean.json: {len(kde_payload.get('sources') or [])} ROI sources")

    if issues:
        print("\nQA issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("\nQA passed.")
    return 0


def main():
    target = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR)
    if not (target / "manifest.json").is_file():
        raise FileNotFoundError(f"No manifest.json under {target}")
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") == 2 and manifest.get("layout") == "split-multi-atlas":
        raise SystemExit(qa_multi_atlas_layout(target))
    raise SystemExit(qa_split_layout(target))


if __name__ == "__main__":
    main()
