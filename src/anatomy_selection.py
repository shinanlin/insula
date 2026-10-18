"""Anatomical exceptions explicitly accepted for the paper cohort."""


def accepted_unknown_endpoint(point):
    """Retain an unresolved zero endpoint without relabeling it as white matter.

    The native parcellation, aseg and ribbon must all be unassigned. Known
    tissue, missing images, and disagreements still follow the usual review.
    """
    return (
        bool(point.get('native_available'))
        and not point.get('native_error')
        and not point.get('native_archived_FS_conflict')
        and point.get('maper_tissue_code') is None
        and point.get('native_FS_label') == 'Unknown'
        and point.get('native_aseg_label') == 'Unknown'
        and point.get('native_ribbon_code') == 0
    )
