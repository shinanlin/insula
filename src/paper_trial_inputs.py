"""Read-only restored-input views for the approved Figure 4/5 refresh."""
from pathlib import Path
from src.connectivity.pairwise.config import DEFAULT_DATASETS

ROOT=Path(__file__).resolve().parents[1]
VIEW=ROOT/'results/paper_trial_inputs'


def paper_bids_root(task):
    path=VIEW/task/'BIDS'
    return str(path) if (VIEW/'input_view_manifest.json').is_file() else DEFAULT_DATASETS[task]


def regenerated_parcellation_root(task):
    """Original parcellation entrypoint outputs, isolated from native BIDS."""
    return ROOT/'results/parcellation_input_restore_2026-10-18'/task/'BIDS/derivatives/parcellation'
