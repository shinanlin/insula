from pathlib import Path

import numpy as np
import pandas as pd

from src.connectivity.pairwise.config import phase_time_mask
from src.connectivity.pairwise.io import discover_manifest
from src.connectivity.pairwise.io import _channel_is_usable
from src.connectivity.pairwise.pairs import (
    enumerate_insula_to_all_pairs,
    shares_physical_contact,
)
from src.connectivity.pairwise.permutation import (
    generate_derangements,
    scalar_permutation_inference,
    stable_seed,
)
from src.connectivity.pairwise.seeds import (
    HAMMERS_INSULA_IDS,
    strict_hammers_seed_frame,
)


def test_strict_seed_uses_both_hammers_endpoints_not_derived_roi():
    parcellation = pd.DataFrame(
        {
            "name": ["S1-2", "S2-3", "N1-2"],
            "contact_1_label": [
                "insula anterior short gyrus L",
                "insula anterior short gyrus L",
                "frontal lobe L",
            ],
            "contact_2_label": [
                "insula posterior long gyrus L",
                "white matter L",
                "temporal lobe L",
            ],
            "center": [
                "not an insula label",
                "insula anterior short gyrus L",
                "insula anterior pole L",
            ],
            "roi": ["not_insula", "AIC", "PIC"],
            "mix": [False, False, False],
        }
    )
    seeds = strict_hammers_seed_frame(parcellation)
    assert seeds["channel"].tolist() == ["S1-2"]
    assert bool(seeds.loc[0, "seed_subregion_mix"])
    assert seeds.loc[0, "contact_1_subregion"] == "ASG"
    assert seeds.loc[0, "contact_2_subregion"] == "PLG"
    assert HAMMERS_INSULA_IDS == {
        "ASG": (86, 87),
        "MSG": (88, 89),
        "PSG": (90, 91),
        "AP": (92, 93),
        "ALG": (94, 95),
        "PLG": (20, 21),
    }


def test_pair_family_is_seed_to_all_and_removes_shared_contacts():
    channels = ["X1-2", "X2-3", "Y1-2", "Z1-2"]
    parcellation = pd.DataFrame(
        {
            "name": channels,
            "contact_1_label": [
                "insula anterior pole L",
                "insula anterior pole L",
                "frontal lobe L",
                "insula anterior long gyrus R",
            ],
            "contact_2_label": [
                "insula anterior pole L",
                "white matter L",
                "frontal lobe L",
                "insula posterior long gyrus R",
            ],
            "center": ["insula anterior pole L", "", "", ""],
            "roi": ["PIC", "AIC", "other", "other"],
        }
    )
    seeds = strict_hammers_seed_frame(parcellation, channels)
    pairs = enumerate_insula_to_all_pairs(channels, seeds, parcellation)
    assert set(pairs["pair_id"]) == {
        "X1-2__Y1-2",
        "X1-2__Z1-2",
        "Z1-2__X2-3",
        "Z1-2__Y1-2",
    }
    assert pairs["source_is_seed"].all()
    assert shares_physical_contact("X1-2", "X2-3")
    assert not shares_physical_contact("X1-2", "Y1-2")


def test_half_open_phase_window():
    times = np.asarray([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    assert phase_time_mask(times, "Stimulus").tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    assert phase_time_mask(times, "Delay").tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert phase_time_mask(times, "Response").tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_wpli_bands_include_broadband():
    from src.connectivity.pairwise.config import WPLI_BANDS

    assert list(WPLI_BANDS) == ["theta", "alpha", "beta", "broadband"]
    assert WPLI_BANDS["broadband"] == (4.0, 30.0)


def test_shared_derangements_seed_and_corrections_are_reproducible():
    seed = stable_seed(42, ["D0092", "LexicalDelay", "Response", "Repeat"])
    first = generate_derangements(40, 31, seed)
    second = generate_derangements(40, 31, seed)
    assert np.array_equal(first, second)
    assert not np.any(first == np.arange(31))
    assert np.all(
        np.sort(first, axis=1) == np.arange(31)[None, :]
    )

    rng = np.random.default_rng(4)
    null = rng.normal(size=(200, 5))
    observed = np.asarray([3.0, 1.0, 0.0, -1.0, np.nan])
    one = scalar_permutation_inference(
        observed, null, tail="two-sided", alpha=0.05
    )
    two = scalar_permutation_inference(
        observed, null, tail="two-sided", alpha=0.05
    )
    assert np.allclose(
        one["p_fwer_maxstat"], two["p_fwer_maxstat"], equal_nan=True
    )
    finite = np.isfinite(one["p_uncorrected"])
    assert np.all(
        one["p_fwer_maxstat"][finite] >= one["p_uncorrected"][finite]
    )


def test_manifest_discovers_all_channel_zscore_not_effective(tmp_path):
    root = tmp_path / "BIDS"
    epoch_root = (
        root
        / "derivatives"
        / "epoch(bipolar)"
        / "sub-DTEST"
    )
    zscore = (
        epoch_root
        / "epoch(band)(zscore)"
        / "sub-DTEST_task-Demo_proc-Response_desc-Repeat_highgamma.h5"
    )
    raw = (
        epoch_root
        / "epoch(raw)"
        / "sub-DTEST_task-Demo_proc-Response_desc-Repeat_raw.h5"
    )
    hammers = (
        root
        / "derivatives"
        / "parcellation"
        / "sub-DTEST"
        / "bipolar"
        / "sub-DTEST_hammers.csv"
    )
    zscore.parent.mkdir(parents=True)
    raw.parent.mkdir(parents=True)
    hammers.parent.mkdir(parents=True)
    zscore.touch()
    raw.touch()
    hammers.touch()
    frame = discover_manifest({"Demo": str(root)})
    assert len(frame) == 1
    assert frame.loc[0, "zscore_path"] == str(zscore)
    assert frame.loc[0, "status"] == "ready"
    assert not frame.loc[0, "effective_annotation_available"]


def test_channel_qc_accepts_nonconstant_voltage_in_volts():
    hga = np.arange(40, dtype=np.float32).reshape(4, 1, 10)
    raw = (hga * 1e-6).astype(np.float32)
    assert _channel_is_usable(hga, raw).tolist() == [True]


def test_d0092_strict_seed_and_pair_count_when_inputs_available():
    import mne

    bids_root = Path("/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS")
    hammers_path = (
        bids_root
        / "derivatives/parcellation/sub-D0092/bipolar/"
        "sub-D0092_hammers.csv"
    )
    zscore_path = (
        bids_root
        / "derivatives/epoch(bipolar)/sub-D0092/"
        "epoch(band)(zscore)/"
        "sub-D0092_task-LexicalDecRepDelay_proc-Response_"
        "desc-Repeat_highgamma.h5"
    )
    raw_path = (
        bids_root
        / "derivatives/epoch(bipolar)/sub-D0092/epoch(raw)/"
        "sub-D0092_task-LexicalDecRepDelay_proc-Response_"
        "desc-Repeat_raw.h5"
    )
    if not all(path.exists() for path in (hammers_path, zscore_path, raw_path)):
        return
    zscore = mne.read_epochs(zscore_path, preload=False, verbose="error")
    raw = mne.read_epochs(raw_path, preload=False, verbose="error")
    assert zscore.ch_names == raw.ch_names
    assert np.array_equal(zscore.events, raw.events)
    assert zscore.event_id == raw.event_id
    parcellation = pd.read_csv(hammers_path)
    seeds = strict_hammers_seed_frame(parcellation, zscore.ch_names)
    pairs = enumerate_insula_to_all_pairs(
        zscore.ch_names, seeds, parcellation
    )
    assert len(zscore.ch_names) == 142
    assert len(seeds) == 9
    assert len(pairs) == 1_224


def test_connectivity_paths_use_bidspath_layout():
    from pathlib import Path

    from src.connectivity.pairwise.output import (
        DEFAULT_OUTPUT_ROOT,
        connectivity_bids_path,
        entity_basename,
        entity_output_dir,
        failure_bids_path,
        metric_output_dir,
    )

    entities = {
        "dataset": "LexicalDelay",
        "subject": "D0092",
        "task": "LexicalDelay",
        "phase": "Response",
        "description": "Repeat",
        "recording": "highgamma",
        "run": "",
        "acquisition": "",
    }
    output_root = DEFAULT_OUTPUT_ROOT.parent / "connectivity-test"

    metric_dir = metric_output_dir(output_root, entities, "xcorr")
    assert metric_dir == (
        output_root / "LexicalDelay" / "sub-D0092" / "xcorr"
    )
    assert entity_output_dir(output_root, entities) == (
        output_root / "LexicalDelay" / "sub-D0092"
    )

    pairs = connectivity_bids_path(
        output_root,
        entities,
        metric="xcorr",
        suffix="pairs",
        extension=".parquet",
    )
    detail = connectivity_bids_path(
        output_root,
        entities,
        metric="oaec",
        suffix="detail",
        extension=".nc",
    )
    clusters = connectivity_bids_path(
        output_root,
        entities,
        metric="xcorr",
        suffix="clusters",
        extension=".parquet",
    )
    failure = failure_bids_path(output_root, entities)

    assert Path(pairs.fpath).name.endswith("_pairs.parquet")
    assert Path(detail.fpath).name.endswith("_detail.nc")
    assert Path(clusters.fpath).name.endswith("_clusters.parquet")
    assert Path(failure.fpath).name.endswith("_failure.json")
    assert Path(pairs.fpath).parent.name == "xcorr"
    assert Path(detail.fpath).parent.name == "oaec"

    basename = entity_basename(entities)
    assert Path(pairs.fpath).name == f"{basename}_pairs.parquet"

    for path in (pairs, detail, clusters, failure):
        filename = Path(path.fpath).name
        stem = filename.split(".", maxsplit=1)[0]
        entity_tokens = [
            token for token in stem.split("_") if "-" in token
        ]
        assert entity_tokens
        for token in entity_tokens:
            _, value = token.split("-", 1)
            assert "_" not in value

