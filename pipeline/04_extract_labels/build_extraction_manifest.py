#!/usr/bin/env python3
"""Build task-specific MAPER extraction manifests across all BIDS roots."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TASK_ROOTS = {
    "LexicalDecRepDelay": Path("/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"),
    "LexicalDecRepNoDelay": Path("/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS"),
    "Phoneme_sequencing": Path("/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"),
    "PictureNaming": Path("/cwork/ns458/BIDS-1.3_PictureNaming/BIDS"),
    "SentenceRep": Path("/cwork/ns458/BIDS-1.4_SentenceRep/BIDS"),
}
MAPER_ROOT = Path("/cwork/ns458/maper_run")
RECON_ROOT = Path("/cwork/ns458/ECoG_Recon")


def recon_orig(subject: str) -> Path:
    return RECON_ROOT / f"D{int(subject.lstrip('D0'))}" / "mri" / "orig.mgz"


def identical_contacts(paths: list[Path]) -> bool:
    tables = [
        pd.read_csv(path, sep="\t")[["name", "x", "y", "z"]]
        .sort_values("name")
        .reset_index(drop=True)
        for path in paths
    ]
    return all(tables[0].equals(table) for table in tables[1:])


def contacts_tsv(bids_root: Path, subject: str) -> tuple[Path | None, str | None]:
    paths = sorted(
        (bids_root / "derivatives" / "clean" / f"sub-{subject}")
        .glob("**/*electrodes.tsv")
    )
    if not paths:
        return None, "missing_contacts"
    if len(paths) > 1 and not identical_contacts(paths):
        return None, "nonidentical_contacts"
    return paths[0], None


def output_paths(bids_root: Path, subject: str) -> tuple[Path, Path]:
    directory = (
        bids_root / "derivatives" / "faillenot" / f"sub-{subject}" / "bipolar"
    )


def choose_parcellation(directory: Path, subject: str) -> Path | None:
    """Choose one task/reference table without treating QC variants as runs."""
    canonical = directory / f"sub-{subject}_aparc2009s.csv"
    if canonical.is_file():
        return canonical
    legacy = sorted(directory.glob(f"sub-{subject}_*_aparc2009s.csv"))
    return legacy[0] if len(legacy) == 1 else None
    return (
        directory / f"sub-{subject}_desc-maper95_bipolar.csv",
        directory / f"sub-{subject}_desc-maper95Sphere2mm_bipolar.csv",
    )


def build() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for task, bids_root in TASK_ROOTS.items():
        pattern = "derivatives/parcellation/sub-D*/bipolar"
        for directory in sorted(bids_root.glob(pattern)):
            subject = directory.parts[-2].replace("sub-", "")
            # The unqualified file is the canonical three-point parcellation.
            # Historical products such as *_proc-3mm_aparc2009s.csv are QC
            # derivatives, not additional task/reference identities, and must
            # never compete for the same MAPER output path.
            parcellation = choose_parcellation(directory, subject)
            if parcellation is None:
                continue
            run = MAPER_ROOT / subject
            fused = run / "output" / f"f30-seg95-{subject}.nii.gz"
            tissue = run / "output" / f"f30-seg95-{subject}-tc3crisp.nii.gz"
            propagated = run / "output" / subject
            orig = recon_orig(subject)
            contacts, contacts_error = contacts_tsv(bids_root, subject)
            output, sensitivity = output_paths(bids_root, subject)

            status = "ready"
            checks = [
                (not fused.is_file(), "missing_fused"),
                (not tissue.is_file(), "missing_tissue"),
                (not orig.is_file(), "missing_orig"),
                (contacts_error is not None, contacts_error or "missing_contacts"),
                (
                    len(list(propagated.glob(f"*-{subject}/seg/seg95.nii.gz"))) != 30,
                    "missing_propagated",
                ),
            ]
            for failed, label in checks:
                if failed:
                    status = label
                    break

            rows.append({
                "task": task,
                "bids_root": str(bids_root),
                "subject": subject,
                "parcellation_csv": str(parcellation),
                "contacts_tsv": str(contacts or ""),
                "fused": str(fused),
                "tissue": str(tissue),
                "propagated_dir": str(propagated),
                "orig": str(orig),
                "output": str(output),
                "sensitivity_output": str(sensitivity),
                "status": status,
            })
    return pd.DataFrame(rows).sort_values(["task", "subject"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build()
    ready = manifest[manifest["status"] == "ready"].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.ready_output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, sep="\t", index=False)
    ready.to_csv(args.ready_output, sep="\t", index=False)
    print(f"Wrote {args.output}: {len(manifest)} combinations")
    print(f"Wrote {args.ready_output}: {len(ready)} ready")
    print(manifest["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
