"""Shared insula parcel patterns (aparc.a2009s) — keep in sync with src/constants/insula.js."""

INSULA_PATTERNS = [
    "G_insular_short",
    "G_Ins_lg_and_S_cent_ins",
    "S_circular_insula_ant",
    "S_circular_insula_inf",
    "S_circular_insula_sup",
]


def is_insula_label(name: str) -> bool:
    if not name:
        return False
    return any(pattern in name for pattern in INSULA_PATTERNS)
