"""Shared insula definitions — keep in sync with src/constants/insula.js."""

INSULA_PATTERNS = [
    "G_insular_short",
    "G_Ins_lg_and_S_cent_ins",
    "S_circular_insula_ant",
    "S_circular_insula_inf",
    "S_circular_insula_sup",
]

APARC_INSULA_ROIS = {"INS", "Insula"}
HAMMERS_INSULA_ROIS = {"AIC", "PIC"}


def is_insula_label(name: str) -> bool:
    if not name:
        return False
    return any(pattern in name for pattern in INSULA_PATTERNS)


def electrode_in_insula(roi: str, label: str, atlas: str = "hammers", mix: bool = False) -> bool:
    if atlas == "hammers":
        if roi not in HAMMERS_INSULA_ROIS:
            return False
        return not mix
    if roi in APARC_INSULA_ROIS:
        return True
    return is_insula_label(label)
