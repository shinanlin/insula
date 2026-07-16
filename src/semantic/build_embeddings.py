#!/usr/bin/env python3
"""Build Word-token GloVe matrices under src/semantic/embedding/.

Reads Lexical Delay BIDS events, looks up GloVe 300d vectors for Word tokens
only, and writes:

  embedding/stimulus_tokens_word.npy
  embedding/embeddings_glove300.npy
  embedding/embeddings_meta.json

Full GloVe lives under the group cache (not in this package). Override with
``--glove_txt`` or ``--glove_zip`` / ``--cache_dir``.
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
    cwd=True,
)

logger = logging.getLogger(__name__)

DEFAULT_BIDS_ROOT = Path("/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS")
DEFAULT_CACHE_DIR = Path("/hpc/group/coganlab/nanlinshi/cache/embeddings/glove")
DEFAULT_HF_HOME = Path("/hpc/group/coganlab/nanlinshi/cache/huggingface")
GLOVE_HF_REPO = "stanfordnlp/glove"
GLOVE_ZIP_NAME = "glove.6B.zip"
GLOVE_MEMBER = "glove.6B.300d.txt"
GLOVE_DIM = 300

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = PACKAGE_DIR / "embedding"


def tokens_from_events(bids_root: Path) -> pd.DataFrame:
    """Unique tokens with lexicality from BIDS events.tsv files."""
    rows = []
    for path in sorted(bids_root.glob("sub-*/ieeg/*_events.tsv")):
        df = pd.read_csv(path, sep="\t")
        if "trial_type" not in df.columns:
            continue
        for tt in df["trial_type"].astype(str):
            parts = tt.split("/")
            if len(parts) < 4:
                continue
            if parts[0] not in {"Auditory_stim", "Cue", "Delay", "Go", "Resp"}:
                continue
            rows.append(
                {
                    "lexicality": parts[2],
                    "token": parts[3].lower(),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No events parsed under {bids_root}")
    return (
        pd.DataFrame(rows)
        .groupby("token", as_index=False)
        .agg(lexicality=("lexicality", lambda s: s.mode().iloc[0]))
        .sort_values("token")
        .reset_index(drop=True)
    )


def ensure_glove_txt(
    cache_dir: Path,
    glove_txt: Path | None,
    glove_zip: Path | None,
    hf_home: Path,
) -> Path:
    """Return path to glove.6B.300d.txt, downloading the zip if needed."""
    if glove_txt is not None:
        glove_txt = Path(glove_txt)
        if not glove_txt.is_file():
            raise FileNotFoundError(glove_txt)
        return glove_txt

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    txt_path = cache_dir / GLOVE_MEMBER
    if txt_path.is_file():
        logger.info("Using existing GloVe txt: %s", txt_path)
        return txt_path

    zip_path = Path(glove_zip) if glove_zip is not None else cache_dir / GLOVE_ZIP_NAME
    if not zip_path.is_file():
        logger.info(
            "Downloading %s from Hugging Face (%s) into %s",
            GLOVE_ZIP_NAME,
            GLOVE_HF_REPO,
            cache_dir,
        )
        hf_home = Path(hf_home)
        hf_home.mkdir(parents=True, exist_ok=True)
        import os

        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=GLOVE_HF_REPO,
            filename=GLOVE_ZIP_NAME,
            local_dir=str(cache_dir),
        )
        zip_path = Path(downloaded)
        logger.info("Downloaded zip to %s", zip_path)

    logger.info("Extracting %s from %s", GLOVE_MEMBER, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(GLOVE_MEMBER, path=cache_dir)
    if not txt_path.is_file():
        raise FileNotFoundError(f"Failed to extract {GLOVE_MEMBER} into {cache_dir}")
    return txt_path


def lookup_glove(tokens: list[str], glove_txt: Path) -> tuple[np.ndarray, list[str]]:
    """Return (n_tokens, 300) matrix and list of missing tokens (zero rows)."""
    want = {t: i for i, t in enumerate(tokens)}
    mat = np.zeros((len(tokens), GLOVE_DIM), dtype=np.float64)
    found = np.zeros(len(tokens), dtype=bool)
    with open(glove_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < GLOVE_DIM + 1:
                continue
            word = parts[0]
            idx = want.get(word)
            if idx is None:
                continue
            mat[idx] = np.asarray(parts[1 : GLOVE_DIM + 1], dtype=np.float64)
            found[idx] = True
            if found.all():
                break
    missing = [t for t, ok in zip(tokens, found) if not ok]
    return mat, missing


def build(
    bids_root: Path,
    out_dir: Path,
    cache_dir: Path,
    hf_home: Path,
    glove_txt: Path | None = None,
    glove_zip: Path | None = None,
) -> None:
    table = tokens_from_events(bids_root)
    words = table.loc[table["lexicality"] == "Word", "token"].tolist()
    if not words:
        raise RuntimeError("No Word tokens found in events")

    txt = ensure_glove_txt(cache_dir, glove_txt, glove_zip, hf_home)
    emb, missing = lookup_glove(words, txt)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    token_arr = np.asarray(words, dtype=object)
    np.save(out_dir / "stimulus_tokens_word.npy", token_arr)
    np.save(out_dir / "embeddings_glove300.npy", emb)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bids_root": str(bids_root),
        "glove_txt": str(txt),
        "glove_source": GLOVE_HF_REPO,
        "glove_file": GLOVE_MEMBER,
        "dim": GLOVE_DIM,
        "n_word": len(words),
        "n_missing": len(missing),
        "missing_tokens": missing,
        "files": [
            "stimulus_tokens_word.npy",
            "embeddings_glove300.npy",
            "embeddings_meta.json",
        ],
    }
    with open(out_dir / "embeddings_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    logger.info(
        "Wrote %d Word embeddings to %s (missing=%d: %s)",
        len(words),
        out_dir,
        len(missing),
        missing if missing else "none",
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids_root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--hf_home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument(
        "--glove_txt",
        type=Path,
        default=None,
        help="Existing glove.6B.300d.txt (skip download)",
    )
    parser.add_argument(
        "--glove_zip",
        type=Path,
        default=None,
        help="Existing glove.6B.zip (skip download, still extract txt if needed)",
    )
    args = parser.parse_args()
    build(
        bids_root=args.bids_root,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        hf_home=args.hf_home,
        glove_txt=args.glove_txt,
        glove_zip=args.glove_zip,
    )


if __name__ == "__main__":
    main()
