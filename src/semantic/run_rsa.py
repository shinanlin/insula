#!/usr/bin/env python3
"""CLI scaffold for Lexical Delay semantic RSA (Tier B).

v1 builds the stimulus table and model RDMs. Neural RDM loading from
epoch/decoding derivatives will be wired next.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import rootutils

path = rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
    cwd=True,
)

from src.paths import RESULTS_ROOT
from src.semantic.features import (
    DEFAULT_BIDS_ROOT,
    build_stimulus_table,
    orthographic_rdm,
    phonological_rdm,
    scalar_diff_rdm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(
    bids_root: str,
    out_dir: str,
    frequency_csv: str | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    table = build_stimulus_table(
        bids_root=bids_root,
        frequency_csv=frequency_csv,
    )
    table_path = out / "stimulus_table.csv"
    # embedding column may be object; drop for csv friendliness
    table.drop(columns=["embedding"], errors="ignore").to_csv(table_path, index=False)
    logger.info("Wrote %s (%d tokens)", table_path, len(table))

    words = table[table["lexicality"] == "Word"].reset_index(drop=True)
    if words.empty:
        raise RuntimeError("No Word tokens found; check events parsing")

    tokens = words["token"].tolist()
    rdm_phon = phonological_rdm(tokens)
    rdm_orth = orthographic_rdm(tokens)
    np.save(out / "rdm_phonology_word.npy", rdm_phon)
    np.save(out / "rdm_orthography_word.npy", rdm_orth)

    if words["log_freq"].notna().any():
        rdm_freq = scalar_diff_rdm(words["log_freq"].to_numpy())
        np.save(out / "rdm_frequency_word.npy", rdm_freq)
        logger.info("Wrote frequency RDM")
    else:
        logger.info("No frequency column filled; skip frequency RDM")

    meta = {
        "n_word": int(len(words)),
        "n_nonword": int((table["lexicality"] == "Nonword").sum()),
        "tokens_word": tokens,
    }
    np.save(out / "word_token_order.npy", np.array(tokens, dtype=object))
    logger.info(
        "Model RDMs ready for %d words. Next: attach embeddings + neural RDM.",
        meta["n_word"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bids_root",
        type=str,
        default=str(DEFAULT_BIDS_ROOT),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(RESULTS_ROOT / "semantic" / "LexicalDelay"),
    )
    parser.add_argument(
        "--frequency_csv",
        type=str,
        default=None,
        help="Optional CSV with token,log_freq columns",
    )
    args = parser.parse_args()
    main(
        bids_root=args.bids_root,
        out_dir=args.out_dir,
        frequency_csv=args.frequency_csv,
    )
