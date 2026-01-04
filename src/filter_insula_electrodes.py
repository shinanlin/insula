"""
Filter Insula Electrodes - Quality Control Script

This script performs quality control on sEEG electrode positions to filter out
Insula electrodes that may be contaminated by IFG (Inferior Frontal Gyrus).

Two exclusion rules are applied:
1. Proximity Rule: Exclude if electrode midpoint is < 3mm from IFG gray matter
2. Cross-Sulcus Rule: Exclude if bipolar pair spans Insula and IFG regions
"""

import argparse
import logging
import sys
import os
import re
from typing import Dict, Tuple, Optional, Set

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from mne_bids import BIDSPath
import os
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants: Label keywords for IFG and Insula regions
# ============================================================================
IFG_KEYWORDS = (
    'G_front_inf-Opercular',
    'G_front_inf-Triangul',
    'G_front_inf-Orbital',
    'Lat_Fis-ant-Vertical',
)

INSULA_KEYWORDS = (
    'G_insular_short',
    'S_circular_insula',
    'G_Ins_lg_and_S_cent_ins',
)

# Proximity threshold in mm
PROXIMITY_THRESHOLD_MM = 5.0


def load_lut(lut_path: str) -> Dict[int, str]:
    """
    Load FreeSurfer Color LUT file and return a mapping of ID -> label name.
    
    Parameters
    ----------
    lut_path : str
        Path to FreeSurferColorLUT.txt
        
    Returns
    -------
    Dict[int, str]
        Mapping from label ID to label name
    """
    id_to_name = {}
    if not os.path.isfile(lut_path):
        logger.warning(f"LUT file not found: {lut_path}")
        return id_to_name
    
    with open(lut_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) >= 2 and parts[0].isdigit():
                id_to_name[int(parts[0])] = parts[1]
    
    logger.info(f"Loaded LUT with {len(id_to_name)} entries from {lut_path}")
    return id_to_name


def get_ifg_label_ids(id_to_name: Dict[int, str]) -> Set[int]:
    """
    Find all label IDs that belong to IFG regions.
    
    Parameters
    ----------
    id_to_name : Dict[int, str]
        LUT mapping from ID to label name
        
    Returns
    -------
    Set[int]
        Set of label IDs belonging to IFG
    """
    ifg_ids = set()
    for label_id, label_name in id_to_name.items():
        for keyword in IFG_KEYWORDS:
            if keyword in label_name:
                ifg_ids.add(label_id)
                break
    logger.info(f"Found {len(ifg_ids)} IFG label IDs")
    return ifg_ids


def is_insula_label(label: str) -> bool:
    """Check if a label belongs to Insula region."""
    if not label or label == 'Unknown':
        return False
    return any(kw in label for kw in INSULA_KEYWORDS)


def is_ifg_label(label: str) -> bool:
    """Check if a label belongs to IFG region."""
    if not label or label == 'Unknown':
        return False
    return any(kw in label for kw in IFG_KEYWORDS)


def compute_ifg_distance_map(
    atlas_data: np.ndarray,
    ifg_ids: Set[int],
    voxel_size_mm: float = 1.0
) -> np.ndarray:
    """
    Compute Euclidean distance transform from IFG gray matter.
    
    Parameters
    ----------
    atlas_data : np.ndarray
        3D atlas volume with integer labels
    ifg_ids : Set[int]
        Set of label IDs belonging to IFG
    voxel_size_mm : float
        Voxel size in mm (assumes isotropic)
        
    Returns
    -------
    np.ndarray
        Distance map where each voxel contains distance to nearest IFG voxel (in mm)
    """
    # Create binary mask: 1 = IFG, 0 = other
    ifg_mask = np.isin(atlas_data, list(ifg_ids)).astype(np.uint8)
    n_ifg_voxels = np.sum(ifg_mask)
    logger.info(f"IFG mask contains {n_ifg_voxels} voxels")
    
    if n_ifg_voxels == 0:
        logger.warning("No IFG voxels found in atlas! Distance map will be infinite.")
        return np.full(atlas_data.shape, np.inf)
    
    # Compute EDT: distance from non-IFG voxels to nearest IFG voxel
    # We invert the mask so that IFG voxels have distance 0
    dist_map = distance_transform_edt(~ifg_mask.astype(bool)) * voxel_size_mm
    
    logger.info(f"Distance map computed. Range: [{dist_map.min():.2f}, {dist_map.max():.2f}] mm")
    return dist_map


def ras_to_voxel(ras: np.ndarray) -> Tuple[int, int, int]:
    """
    Convert RAS coordinates (mm) to FreeSurfer atlas voxel indices.
    
    This follows the same transformation as in parcellation.py:
    - Adaptive unit conversion: if |value| < 10, assume meters and convert to mm
    - RAS -> atlas voxel index using FreeSurfer tkr space constants
    
    Parameters
    ----------
    ras : np.ndarray
        (x, y, z) coordinates in RAS space
        
    Returns
    -------
    Tuple[int, int, int]
        (i, j, k) voxel indices
    """
    ras = np.array(ras, dtype=float)
    
    # Adaptive unit conversion: if coordinates look like meters, convert to mm
    if np.max(np.abs(ras)) < 10:
        ras = ras * 1000
    
    # RAS -> atlas voxel index (matching parcellation.py)
    i = int(np.round(128 - ras[0]))  # X -> first index (with flip)
    j = int(np.round(128 - ras[2]))  # Z -> second index (with flip)
    k = int(126 + np.round(ras[1]))  # Y -> third index (offset by 126)
    
    return i, j, k


def clip_to_bounds(i: int, j: int, k: int, shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Clip voxel indices to atlas bounds."""
    i = int(np.clip(i, 0, shape[0] - 1))
    j = int(np.clip(j, 0, shape[1] - 1))
    k = int(np.clip(k, 0, shape[2] - 1))
    return i, j, k


def parse_bipolar_name(name: str) -> Tuple[str, str]:
    """
    Parse a bipolar electrode name into its two constituent monopolar names.
    
    Examples:
        'D0019_ROG1-2' -> ('D0019_ROG1', 'D0019_ROG2')
        'D0019_LAI10-11' -> ('D0019_LAI10', 'D0019_LAI11')
    
    Parameters
    ----------
    name : str
        Bipolar electrode name (e.g., 'D0019_ROG1-2')
        
    Returns
    -------
    Tuple[str, str]
        (deep_contact, shallow_contact) names
    """
    # Pattern: prefix + number1 + '-' + number2
    # e.g., D0019_ROG1-2 -> prefix=D0019_ROG, num1=1, num2=2
    match = re.match(r'^(.+?)(\d+)-(\d+)$', name)
    if not match:
        logger.warning(f"Could not parse bipolar name: {name}")
        return (name, name)
    
    prefix = match.group(1)
    num1 = match.group(2)
    num2 = match.group(3)
    
    contact1 = f"{prefix}{num1}"
    contact2 = f"{prefix}{num2}"
    
    return contact1, contact2


def build_car_lookup(car_df: pd.DataFrame, dist_map: np.ndarray, atlas_shape: Tuple[int, int, int]) -> Dict[str, Dict]:
    """
    Build a lookup table from monopolar (CAR) electrode data.
    
    Parameters
    ----------
    car_df : pd.DataFrame
        Monopolar electrode DataFrame with columns: name, x, y, z, label
    dist_map : np.ndarray
        Distance-to-IFG map
    atlas_shape : Tuple[int, int, int]
        Shape of the atlas volume
        
    Returns
    -------
    Dict[str, Dict]
        Lookup table: {name: {'label': str, 'dist_to_IFG_mm': float}}
    """
    lookup = {}
    
    for _, row in car_df.iterrows():
        name = row['name']
        ras = np.array([row['x'], row['y'], row['z']])
        label = row.get('label', 'Unknown')
        
        # Convert to voxel and get distance
        i, j, k = ras_to_voxel(ras)
        i, j, k = clip_to_bounds(i, j, k, atlas_shape)
        dist_to_ifg = dist_map[i, j, k]
        
        lookup[name] = {
            'label': label,
            'dist_to_IFG_mm': dist_to_ifg,
        }
    
    logger.info(f"Built CAR lookup with {len(lookup)} entries")
    return lookup


def process_bipolar_qc(
    bipolar_df: pd.DataFrame,
    car_lookup: Dict[str, Dict],
    dist_map: np.ndarray,
    atlas_shape: Tuple[int, int, int],
    proximity_threshold: float = PROXIMITY_THRESHOLD_MM
) -> pd.DataFrame:
    """
    Apply QC rules to bipolar electrode data.
    
    Parameters
    ----------
    bipolar_df : pd.DataFrame
        Bipolar electrode DataFrame
    car_lookup : Dict[str, Dict]
        Monopolar lookup table
    dist_map : np.ndarray
        Distance-to-IFG map
    atlas_shape : Tuple[int, int, int]
        Shape of the atlas volume
    proximity_threshold : float
        Distance threshold in mm for proximity rule
        
    Returns
    -------
    pd.DataFrame
        Bipolar DataFrame with added QC columns
    """
    # Initialize new columns
    bipolar_df = bipolar_df.copy()
    bipolar_df['dist_to_IFG_mm'] = np.nan
    bipolar_df['QC_Status'] = ''
    bipolar_df['QC_Reason'] = ''
    
    for idx, row in bipolar_df.iterrows():
        name = row['name']
        label = row.get('label', 'Unknown')
        ras = np.array([row['x'], row['y'], row['z']])
        
        # Compute distance to IFG for this bipolar midpoint
        i, j, k = ras_to_voxel(ras)
        i, j, k = clip_to_bounds(i, j, k, atlas_shape)
        dist_to_ifg = dist_map[i, j, k]
        bipolar_df.at[idx, 'dist_to_IFG_mm'] = dist_to_ifg
        
        # Check if this is an Insula electrode
        if not is_insula_label(label):
            bipolar_df.at[idx, 'QC_Status'] = 'IGNORE'
            bipolar_df.at[idx, 'QC_Reason'] = 'Not Insula'
            continue
        
        # Rule A: Proximity check
        if dist_to_ifg < proximity_threshold:
            bipolar_df.at[idx, 'QC_Status'] = 'EXCLUDE'
            bipolar_df.at[idx, 'QC_Reason'] = f'Proximity < {proximity_threshold}mm (dist={dist_to_ifg:.2f}mm)'
            logger.info(f"{name}: EXCLUDE - Proximity {dist_to_ifg:.2f}mm < {proximity_threshold}mm")
            continue
        
        # Rule B: Cross-Sulcus check
        contact1, contact2 = parse_bipolar_name(name)
        
        info1 = car_lookup.get(contact1, {})
        info2 = car_lookup.get(contact2, {})
        
        label1 = info1.get('label', 'Unknown')
        label2 = info2.get('label', 'Unknown')
        
        is_insula1 = is_insula_label(label1)
        is_insula2 = is_insula_label(label2)
        is_ifg1 = is_ifg_label(label1)
        is_ifg2 = is_ifg_label(label2)
        
        # Cross-sulcus: one is Insula, the other is IFG
        if (is_insula1 and is_ifg2) or (is_ifg1 and is_insula2):
            bipolar_df.at[idx, 'QC_Status'] = 'EXCLUDE'
            bipolar_df.at[idx, 'QC_Reason'] = f'Cross-Sulcus (Insula-IFG): {contact1}={label1}, {contact2}={label2}'
            logger.info(f"{name}: EXCLUDE - Cross-Sulcus ({contact1}:{label1} vs {contact2}:{label2})")
            continue
        
        # Passed all checks
        bipolar_df.at[idx, 'QC_Status'] = 'KEEP'
        bipolar_df.at[idx, 'QC_Reason'] = ''
        logger.info(f"{name}: KEEP (dist={dist_to_ifg:.2f}mm)")
    
    return bipolar_df


def main(
    bids_root: str,
    recon_dir: str,
    subject: str,
    lut_path: str,
    proximity_threshold: float = PROXIMITY_THRESHOLD_MM,
    **kwargs
):
    """
    Main function to filter Insula electrodes for a given subject.
    
    Parameters
    ----------
    bids_root : str
        Root of the BIDS dataset
    recon_dir : str
        Root of the FreeSurfer / ECoG_Recon directory
    subject : str
        BIDS subject ID (e.g., 'D0019')
    lut_path : str
        Path to FreeSurferColorLUT.txt
    proximity_threshold : float
        Distance threshold in mm for proximity rule
    """
    logger.info(f"Processing subject: {subject}")
    
    # =========================================================================
    # 1. Load FreeSurfer LUT and identify IFG labels
    # =========================================================================
    id_to_name = load_lut(lut_path)
    ifg_ids = get_ifg_label_ids(id_to_name)
    
    # =========================================================================
    # 2. Load subject-specific atlas volume
    # =========================================================================
    # Convert BIDS subject ID (e.g., "D0019") to FreeSurfer ID (e.g., "D19")
    subject_id = f"D{int(subject[1:])}"
    subj_dir = os.path.join(recon_dir, subject_id)
    atlas_path = os.path.join(subj_dir, 'mri', 'aparc.a2009s+aseg.mgz')
    
    logger.info(f"Loading atlas: {atlas_path}")
    try:
        atlas_img = nib.load(atlas_path)
    except Exception as e:
        raise FileNotFoundError(f"Atlas not found: {atlas_path}") from e
    
    atlas_data = atlas_img.get_fdata().astype('int32')
    logger.info(f"Atlas loaded. Shape: {atlas_data.shape}")
    
    # =========================================================================
    # 3. Compute distance-to-IFG map
    # =========================================================================
    dist_map = compute_ifg_distance_map(atlas_data, ifg_ids)
    
    # =========================================================================
    # 4. Load CAR (monopolar) electrode CSV
    # =========================================================================
    car_path = BIDSPath(
        root=os.path.join(bids_root, 'derivatives', 'parcellation'),
        subject=subject,
        suffix='aparc2009s',
        datatype='car',
        processing='3mm',
        extension='.csv',
        check=False
    ).match()
    
    if not car_path:
        raise FileNotFoundError(f"CAR parcellation CSV not found for subject {subject}")
    
    logger.info(f"Loading CAR CSV: {car_path[0]}")
    car_df = pd.read_csv(car_path[0])
    
    # Build lookup table
    car_lookup = build_car_lookup(car_df, dist_map, atlas_data.shape)
    
    # =========================================================================
    # 5. Load Bipolar electrode CSV
    # =========================================================================
    bipolar_path = BIDSPath(
        root=os.path.join(bids_root, 'derivatives', 'parcellation'),
        subject=subject,
        suffix='aparc2009s',
        datatype='bipolar',
        processing='3mm',
        extension='.csv',
        check=False
    ).match()
    
    if not bipolar_path:
        raise FileNotFoundError(f"Bipolar parcellation CSV not found for subject {subject}")
    
    logger.info(f"Loading Bipolar CSV: {bipolar_path[0]}")
    bipolar_df = pd.read_csv(bipolar_path[0])
    
    # =========================================================================
    # 6. Apply QC rules
    # =========================================================================
    qc_df = process_bipolar_qc(
        bipolar_df,
        car_lookup,
        dist_map,
        atlas_data.shape,
        proximity_threshold
    )
    
    # =========================================================================
    # 7. Summary statistics
    # =========================================================================
    n_total = len(qc_df)
    n_insula = len(qc_df[qc_df['QC_Status'] != 'IGNORE'])
    n_keep = len(qc_df[qc_df['QC_Status'] == 'KEEP'])
    n_exclude = len(qc_df[qc_df['QC_Status'] == 'EXCLUDE'])
    
    logger.info("=" * 60)
    logger.info(f"QC Summary for {subject}:")
    logger.info(f"  Total electrodes: {n_total}")
    logger.info(f"  Insula electrodes: {n_insula}")
    logger.info(f"  KEEP: {n_keep}")
    logger.info(f"  EXCLUDE: {n_exclude}")
    logger.info("=" * 60)
    
    # =========================================================================
    # 8. Save results using BIDSPath
    # =========================================================================
    task = 'LexicalDelay'
    ref = 'bipolar'
    save_path = BIDSPath(
        root=os.path.join('results', f'{task}({ref})'),
        subject=subject,
        suffix='qc',
        datatype='parcellation',
        processing='3mm',
        extension='.csv',
        check=False
    )
    save_path.mkdir(exist_ok=True)
    qc_df.to_csv(save_path, index=False)
    logger.info(f"Saved QC results to: {save_path}")
    
    return qc_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter Insula electrodes - Quality Control for sEEG"
    )
    parser.add_argument(
        "--bids_root",
        default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS",
        type=str,
        help="Root directory of the BIDS dataset"
    )
    parser.add_argument(
        "--recon_dir",
        default="/cwork/ns458/ECoG_Recon/",
        type=str,
        help="Root directory containing FreeSurfer/ECoG_Recon subjects"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default='D0103',
        help="BIDS subject ID, e.g., 'D0019'"
    )
    parser.add_argument(
        "--lut_path",
        default=None,
        type=str,
        help="Path to FreeSurferColorLUT.txt (default: <script_dir>/FreeSurferColorLUT.txt)"
    )
    parser.add_argument(
        "--proximity_threshold",
        default=PROXIMITY_THRESHOLD_MM,
        type=float,
        help=f"Distance threshold in mm for proximity rule (default: {PROXIMITY_THRESHOLD_MM})"
    )
    
    args = parser.parse_args()
    
    # Default LUT path to same directory as this script
    if args.lut_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.lut_path = os.path.join(script_dir, 'FreeSurferColorLUT.txt')
    
    main(**vars(args))
