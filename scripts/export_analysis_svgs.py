#!/usr/bin/env python
"""Export analysis SVGs for notebooks whose ``img/<name>/`` folders were empty.

Runs lightweight matplotlib exports (no full nbconvert). Uses notebook code
cells for loaders/plotters, with results roots pointed at data that actually
exists (often the sibling ``insula`` checkout).
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.paths import PROJECT_ROOT, RESULTS_ROOT, img_dir, save_svg

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("export_analysis_svgs")

MAIN_RESULTS = Path("/hpc/group/coganlab/nanlinshi/insula/results")


def _strip_magics(src: str) -> str:
    lines = []
    for ln in src.splitlines():
        s = ln.lstrip()
        if s.startswith("%") or s.startswith("!"):
            continue
        lines.append(ln)
    return "\n".join(lines)


def exec_notebook(
    nb_path: Path,
    *,
    stop_before: tuple[str, ...] = (),
    skip_containing: tuple[str, ...] = (),
    extra: dict | None = None,
) -> dict:
    """Execute notebook code cells into a namespace until a stop marker."""
    nb = json.loads(nb_path.read_text())
    ns: dict = {"__name__": "__export__"}
    if extra:
        ns.update(extra)
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = _strip_magics("".join(cell.get("source", [])))
        if not src.strip():
            continue
        if any(marker in src for marker in stop_before):
            logger.info("Stop before cell %d in %s", i, nb_path.name)
            break
        if any(marker in src for marker in skip_containing):
            logger.info("Skip cell %d in %s", i, nb_path.name)
            continue
        if "RUN_BRAIN = True" in src:
            src = src.replace("RUN_BRAIN = True", "RUN_BRAIN = False")
        if "RUN_BRAIN_LH_GRID = True" in src:
            src = src.replace("RUN_BRAIN_LH_GRID = True", "RUN_BRAIN_LH_GRID = False")
        try:
            exec(compile(src, f"{nb_path.name}:cell{i}", "exec"), ns)
        except Exception:
            logger.exception("Failed executing %s cell %d — continue", nb_path.name, i)
    return ns


def export_decode() -> None:
    out = img_dir("decode")
    out.mkdir(parents=True, exist_ok=True)
    ns = exec_notebook(
        PROJECT_ROOT / "notebooks" / "decode.ipynb",
        stop_before=("for spec in FIGURE_SPECS", "preview_task, preview_decode"),
    )
    hemi = ns.get("hemi", "Left")
    results_root = ns.get("results_root", MAIN_RESULTS)
    ref = ns.get("ref", "bipolar")
    rois = ns.get("hue_order")
    for spec in ns.get("FIGURE_SPECS", []):
        task, decode_type, spec_description = ns["parse_figure_spec"](spec)
        scores = ns["load_windowed_decoding"](
            task,
            decode_type,
            ref=ref,
            rois=rois,
            description=spec_description,
            hemi=hemi,
            articulator_phoneme_filter=ns.get("articulator_phoneme_filter", "1"),
            results_root=results_root,
        )
        if scores is None or getattr(scores, "empty", False):
            logger.warning("decode: empty %s %s %s", task, decode_type, spec_description)
            continue
        n_classes = int(scores["n_classes"].mode().iloc[0])
        chance = ns["chance_level"](decode_type, task, n_classes=n_classes)
        fig = ns["plot_window_accuracy_overview"](
            scores,
            phases=ns["phases"],
            roi_order=ns["hue_order"],
            hemi_label=hemi,
            chance=chance,
            title=f"{task} | {decode_type} | {spec_description} | {hemi}",
        )
        name = ns["bar_output_name"](task, decode_type, spec_description, hemi)
        path = save_svg(fig, out / name, close=True)
        logger.info("Saved %s", path)


def export_decode_functional() -> None:
    out = img_dir("decode_functional")
    out.mkdir(parents=True, exist_ok=True)
    # Skip preview + batch loops; keep helper defs (including resolved).
    ns = exec_notebook(
        PROJECT_ROOT / "notebooks" / "decode_functional.ipynb",
        skip_containing=(
            "preview_scores = load_windowed_decoding",
            "resolved_scores = load_resolved_decoding",
            "scores = load_windowed_decoding(\n        task,",
        ),
        stop_before=("scores = load_resolved_decoding(\n        task,",),
    )
    hemi = ns.get("hemi", "Left")
    roots = ns.get("results_roots", [RESULTS_ROOT, MAIN_RESULTS])
    ref = ns.get("ref", "bipolar")
    rois = ns.get("hue_order")
    load_fn = ns["load_windowed_decoding"]
    for spec in ns.get("FIGURE_SPECS", []):
        task, decode_type, spec_description = ns["parse_figure_spec"](spec)
        scores = load_fn(
            task,
            decode_type,
            ref=ref,
            rois=rois,
            description=spec_description,
            hemi=hemi,
            articulator_phoneme_filter=ns.get("articulator_phoneme_filter", "1"),
            results_roots=roots,
        )
        if scores is None or getattr(scores, "empty", False):
            logger.warning(
                "decode_functional: empty %s %s %s", task, decode_type, spec_description
            )
            continue
        n_classes = int(scores["n_classes"].mode().iloc[0])
        chance = ns["chance_level"](decode_type, task, n_classes=n_classes)
        fig = ns["plot_window_accuracy_overview"](
            scores,
            phases=ns["phases"],
            roi_order=ns["hue_order"],
            hemi_label=hemi,
            chance=chance,
            title=f"{task} | {decode_type} | {spec_description} | {hemi}",
        )
        name = ns["bar_output_name"](task, decode_type, spec_description, hemi)
        path = save_svg(fig, out / name, close=True)
        logger.info("Saved %s", path)

    if "load_resolved_decoding" in ns and "plot_phase_accuracy" in ns:
        for spec in ns.get("FIGURE_SPECS", []):
            task, decode_type, spec_description = ns["parse_figure_spec"](spec)
            try:
                scores = ns["load_resolved_decoding"](
                    task,
                    decode_type,
                    ref=ref,
                    rois=rois,
                    description=spec_description,
                    hemi=hemi,
                    articulator_phoneme_filter=ns.get("articulator_phoneme_filter", "1"),
                    smooth_sigma=ns.get("smooth_sigma", 2),
                    results_roots=roots,
                )
                if scores is None or getattr(scores, "empty", False):
                    continue
                ymin, ymax = ns["phase_accuracy_data_ylim"](
                    scores, phases=ns["phases"], hue_order=ns["hue_order"], hemi=hemi
                )
                pad, bar_h = ns["phase_accuracy_bar_params"](decode_type, task)
                fig, _axes = ns["plot_phase_accuracy"](
                    scores,
                    phases=ns["phases"],
                    hue_order=ns["hue_order"],
                    hemi=hemi,
                    decode_type=decode_type,
                    task=task,
                    chance=ns["chance_level"](decode_type, task),
                    ymin=ymin,
                    ymax=ymax,
                    pad=pad,
                    bar_h=bar_h,
                )
                stem = f"resolved_{task}_{decode_type}"
                if spec_description != "Repeat":
                    stem += f"_{spec_description}"
                path = save_svg(fig, out / stem, close=True)
                logger.info("Saved %s", path)
            except Exception:
                logger.exception(
                    "decode_functional resolved failed %s %s", task, decode_type
                )


def export_connectivity() -> None:
    out = img_dir("connectivity")
    out.mkdir(parents=True, exist_ok=True)
    # Drop stale misnamed / leftover exports before regenerating.
    for stale in (
        "connectivity_fig.svg",
        "connectivity_partner_composition.svg",  # previously mislabeled heatmap
    ):
        p = out / stale
        if p.exists():
            p.unlink()
            logger.info("Removed stale %s", p.name)
    # RUN_BRAIN / RUN_BRAIN_LH_GRID forced False inside exec_notebook.
    # Notebook cells now save:
    #   partner_composition, partner_heatmap, lag_strip_delay_sustained, lag_bins
    exec_notebook(PROJECT_ROOT / "notebooks" / "connectivity.ipynb")
    expected = [
        "connectivity_partner_composition.svg",
        "connectivity_partner_heatmap.svg",
        "connectivity_lag_strip_delay_sustained.svg",
        "connectivity_lag_bins.svg",
    ]
    for name in expected:
        path = out / name
        if path.exists():
            logger.info("OK %s", path)
        else:
            logger.error("MISSING %s", path)
    logger.info("connectivity: %d SVGs in %s", len(list(out.glob("*.svg"))), out)


def export_modulation() -> None:
    out = img_dir("modulation")
    ns = exec_notebook(
        PROJECT_ROOT / "notebooks" / "modulation.ipynb",
    )
    n = len(list(out.glob("*.svg")))
    logger.info("modulation: %d SVGs in %s", n, out)
    if n == 0 and "plot_condition_waveforms" in ns and "plot_df" in ns:
        try:
            fig = ns["plot_condition_waveforms"](
                ns["plot_df"],
                save_name="modulation_pooled",
            )
            if fig is not None:
                plt.close(fig)
        except Exception:
            logger.exception("modulation fallback plot failed")
        n = len(list(out.glob("*.svg")))
        logger.info("modulation after fallback: %d SVGs", n)


EXPORTERS = {
    "decode": export_decode,
    "decode_functional": export_decode_functional,
    "connectivity": export_connectivity,
    "modulation": export_modulation,
}


def main(argv: list[str] | None = None) -> None:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    names = args or list(EXPORTERS)
    for name in names:
        if name not in EXPORTERS:
            raise SystemExit(f"Unknown exporter {name!r}; choose from {list(EXPORTERS)}")
        logger.info("===== %s =====", name)
        try:
            EXPORTERS[name]()
        except Exception:
            logger.error("%s failed:\n%s", name, traceback.format_exc())
    for sub in names:
        files = sorted((PROJECT_ROOT / "img" / sub).glob("**/*.*"))
        logger.info("img/%s: %d files", sub, len(files))
        for f in files[:30]:
            logger.info("  %s", f.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
