#!/usr/bin/env python3
"""Stage 3 parcellation visual QC: native MRI slices for pure insula electrodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util

_SLICE_RENDER = REPO_ROOT / "pipeline" / "05_visual_qc" / "slice_render.py"
_SPEC = importlib.util.spec_from_file_location("slice_render", _SLICE_RENDER)
_SLICE_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _SLICE_MODULE
_SPEC.loader.exec_module(_SLICE_MODULE)

DEFAULT_FOV_MM = _SLICE_MODULE.DEFAULT_FOV_MM
SliceCase = _SLICE_MODULE.SliceCase
draw_slice_case = _SLICE_MODULE.draw_slice_case
load_subject_brain_and_maper_insula_mask = _SLICE_MODULE.load_subject_brain_and_maper_insula_mask
load_subject_brain_and_aparc_insula_mask = _SLICE_MODULE.load_subject_brain_and_aparc_insula_mask
MAPER_CONTOUR = _SLICE_MODULE.MAPER_CONTOUR
APARC_CONTOUR = _SLICE_MODULE.APARC_CONTOUR

from src.paths import parcellation_qc_dir  # noqa: E402

HAMMERS_INSULA_ROIS = frozenset({"AIC", "PIC"})
APARC_INSULA_ROIS = frozenset({"INS", "Insula"})
DEFAULT_RECON_ROOT = Path("/cwork/ns458/ECoG_Recon")
DEFAULT_MAPER_ROOT = Path("/cwork/ns458/maper_run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parcellation-csv", type=Path, required=True)
    parser.add_argument("--atlas", choices=("hammers", "aparc2009s"), required=True)
    parser.add_argument("--recon-dir", type=Path, default=DEFAULT_RECON_ROOT)
    parser.add_argument("--fused")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fov-mm", type=float, default=DEFAULT_FOV_MM)
    return parser.parse_args()


def recon_subject_id(subject: str) -> str:
    return f"D{int(subject.lstrip('D0'))}"


def safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def is_mix_false(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return not value
    text = str(value).strip().lower()
    return text in {"false", "0", "no", ""}


def insula_row_mask(df: pd.DataFrame, atlas: str) -> pd.Series:
    mix_ok = df["mix"].map(is_mix_false)
    if atlas == "hammers":
        return mix_ok & df["roi"].isin(HAMMERS_INSULA_ROIS)
    if atlas == "aparc2009s":
        return mix_ok & df["roi"].isin(APARC_INSULA_ROIS)
    raise ValueError(f"unsupported atlas: {atlas!r}")


def default_fused_path(subject: str, maper_root: Path = DEFAULT_MAPER_ROOT) -> Path:
    return maper_root / subject / "output" / f"f30-seg95-{subject}.nii.gz"


def default_orig_path(subject: str, recon_root: Path) -> Path:
    return recon_root / recon_subject_id(subject) / "mri" / "orig.mgz"


def row_to_case(row: pd.Series, orig: Path, fused: Path | None) -> SliceCase:
    return SliceCase(
        subject=str(row["subject"]),
        name=str(row["name"]),
        roi=str(row["roi"]),
        center_label="" if pd.isna(row.get("center")) else str(row["center"]),
        mix=not is_mix_false(row["mix"]),
        contact_1=str(row["contact_1"]),
        contact_2=str(row["contact_2"]),
        contact_1_xyz=(float(row["x1"]), float(row["y1"]), float(row["z1"])),
        center_xyz=(float(row["x"]), float(row["y"]), float(row["z"])),
        contact_2_xyz=(float(row["x2"]), float(row["y2"]), float(row["z2"])),
        orig=orig,
        fused=fused,
    )


def infer_subject(df: pd.DataFrame, csv_path: Path) -> str:
    if "subject" in df.columns and not df["subject"].empty:
        return str(df["subject"].iloc[0])
    stem = csv_path.stem
    if stem.startswith("sub-"):
        return stem.removeprefix("sub-").split("_", 1)[0]
    raise ValueError(f"cannot infer subject from {csv_path}")


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.parcellation_csv)
    subject = infer_subject(df, args.parcellation_csv)

    orig = default_orig_path(subject, args.recon_dir)
    fused = Path(args.fused) if args.fused else default_fused_path(subject)
    if not orig.exists():
        raise FileNotFoundError(f"missing recon MRI: {orig}")
    if args.atlas == "hammers" and not fused.exists():
        raise FileNotFoundError(f"missing MAPER fused volume: {fused}")

    insula_df = df.loc[insula_row_mask(df, args.atlas)].copy()
    if insula_df.empty:
        print(f"{subject}: no pure insula electrodes (mix=False) for atlas={args.atlas}; skipping output")
        return 0

    output_dir = args.output_dir or parcellation_qc_dir(args.atlas, subject)
    png_dir = output_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    if args.atlas == "hammers":
        orig_img, brain, insula_mask = load_subject_brain_and_maper_insula_mask(orig, fused)
        contour = MAPER_CONTOUR
        fused_for_case = fused
    else:
        orig_img, brain, insula_mask = load_subject_brain_and_aparc_insula_mask(orig)
        contour = APARC_CONTOUR
        fused_for_case = None
    index_rows: list[dict[str, object]] = []
    pdf_path = output_dir / f"{safe_name(subject)}_{args.atlas}_insula_slices.pdf"

    with PdfPages(pdf_path) as pdf:
        for _, row in insula_df.iterrows():
            case = row_to_case(row, orig, fused_for_case)
            figure = draw_slice_case(
                case, orig_img, brain, insula_mask, fov_mm=args.fov_mm, contour=contour,
            )
            png_path = png_dir / f"{safe_name(case.name)}.png"
            figure.savefig(png_path, dpi=150, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            index_rows.append({
                "subject": case.subject,
                "name": case.name,
                "roi": case.roi,
                "mix": case.mix,
                "center": case.center_label,
                "contact_1": case.contact_1,
                "contact_2": case.contact_2,
                "png": str(png_path.relative_to(output_dir)),
                "pdf": str(pdf_path.relative_to(output_dir)),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"wrote {png_path}")

    pd.DataFrame(index_rows).to_csv(output_dir / "index.csv", index=False)
    print(f"{subject}: wrote {len(index_rows)} slice PNGs and {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
