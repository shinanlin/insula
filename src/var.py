#!/usr/bin/env python3
"""Clean and efficient connectivity analysis implementation."""

import rootutils
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
from pathlib import Path
import h5py
import numpy as np
from mne_bids import BIDSPath
import logging
import sys
from scot.var import VAR
from scot.connectivity import Connectivity
import mne
import scipy as sp
import json
from joblib import Parallel, delayed

# SciPy compatibility patch for SCOT (which uses deprecated top-level scipy aliases)
# Safe no-ops if already present in your SciPy build
if not hasattr(sp, 'shape'):
    sp.shape = np.shape
if not hasattr(sp, 'ceil'):
    sp.ceil = np.ceil
if not hasattr(sp, 'atleast_3d'):
    sp.atleast_3d = np.atleast_3d
if not hasattr(sp, 'sign'):
    sp.sign = np.sign
if not hasattr(sp, 'sqrt'):
    sp.sqrt = np.sqrt
if not hasattr(sp, 'zeros'):
    sp.zeros = np.zeros
if not hasattr(sp, 'cov'):
    sp.cov = np.cov
if not hasattr(sp, 'sum'):
    sp.sum = np.sum
if not hasattr(sp, 'eye'):
    sp.eye = np.eye
if not hasattr(sp, 'random'):
    sp.random = np.random


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _load_subject_data(bids_root, subject, band, description):
    epoch_path = BIDSPath(
        root=bids_root,
        subject=subject,
        suffix=band,
        description=description,
        datatype='epoch(band)(sig)',
        extension='.fif',
        check=False
    )
    
    epoch_path = epoch_path.match()[0]
    epoch = mne.read_epochs(epoch_path, verbose='error')
    
    return epoch
            
            
def _permute_pdc(seed: int, segment: np.ndarray, p: int) -> np.ndarray:
    """Generate a circular-shift surrogate, fit VAR, and return mean PDC over freq.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility across workers.
    segment : np.ndarray
        Data segment with shape (n_epochs, n_channels, n_samples).
    p : int
        VAR model order.

    Returns
    -------
    np.ndarray
        Mean PDC over frequency with shape (m, m).
    """
    rng = np.random.default_rng(seed)
    n_tr, n_ch, L = segment.shape
    sur = np.empty_like(segment)
    for tr in range(n_tr):
        shifts = rng.integers(1, L, size=n_ch)
        for ch in range(n_ch):
            sur[tr, ch, :] = np.roll(segment[tr, ch, :], int(shifts[ch]))

    m_sur = VAR(model_order=p)
    m_sur.fit(sur)
    conn_sur = Connectivity(m_sur.coef, c=m_sur.rescov, nfft=64)
    pdc_sur = conn_sur.PDC()
    return pdc_sur.mean(axis=-1)


def convert_ch_names_to_roi(recon_dir, subject, info):
    from ieeg.viz.mri import gen_labels
    import pandas as pd

    # Normalize subject ID (e.g., D0096 -> D96, D0100 -> D100)
    sub_id = 'D' + subject.lstrip('D0') if subject.startswith('D0') else subject

    # Generate anatomical labels per channel
    labels = gen_labels(info, sub=sub_id, subj_dir=recon_dir, atlas='.BN_atlas')

    # Build mapping dictionary from atlas.csv: fine label -> gross label
    atlas_df = pd.read_csv('src/atlas.csv')
    mapping = {
        str(row['Anatomical and modified Cyto-architectonic descriptions']).split(',')[0]:
        str(row['Left and Right Hemisphere']).split('_')[0]
        for _, row in atlas_df.iterrows()
    }
    # Manual override
    mapping['TE1.0/TE1.2'] = 'STG'

    channel_names = []
    for ch in info.ch_names:
        label_full = labels.get(ch, '')
        parts = label_full.rsplit('_', 1)
        label = parts[0] if len(parts) == 2 else label_full
        hemi = parts[1].lower() if len(parts) == 2 else 'unknown'
        gross = mapping.get(label, label)
        roi = (gross + hemi).replace(' ', '')
        channel_names.append(f"{subject}_{roi}_{ch}")

    return channel_names
    
def main(
    bids_root,
    recon_dir,
    subject,
    band,
    n_permutations,
    n_jobs,
    description,
    window,
    step,
    p=4,
    ):
    
    epochs = _load_subject_data(bids_root, subject, band, description)
    
    # get ROI
    ch_names = convert_ch_names_to_roi(recon_dir, subject, epochs.info)
    
    tmin,tmax, fs = epochs.tmin, epochs.tmax, epochs.info['sfreq']
    
    time_points = np.arange(tmin + window,
                            tmax + step,
                            step)
    window_samples = int(window * fs)
    step_samples = int(step * fs)

    # Compute candidate window sample indices
    end_samples = (np.round((time_points - tmin) * fs).astype(int)) + 1
    start_samples = end_samples - window_samples

    # Determine valid windows within bounds
    n_samples_total = epochs._data.shape[-1]
    valid = (start_samples >= 0) & (end_samples <= n_samples_total)
    start_samples = start_samples[valid]
    end_samples = end_samples[valid]
    n_windows = end_samples.size

    # Determine channel count (m) and set model order (fixed p)
    _, m, _ = epochs._data.shape  # (n_epochs, n_channels, n_samples)
    p = 4

    # Pre-allocate result arrays
    coefs = np.empty((n_windows, m, m * p), dtype=float)
    rescovs = np.empty((n_windows, m, m), dtype=float)
    pdcs = np.empty((n_windows, m, m), dtype=float)
    null = np.empty((n_permutations, n_windows, m, m), dtype=float)
    pvals = np.ones((n_windows, m, m), dtype=float)

    # Fit VAR per valid window and fill arrays
    for wi, (ss, es) in enumerate(zip(start_samples, end_samples)):
        # make a segment copy, shape (n_epochs, n_channels, n_samples)
        segment = epochs._data[..., ss:es].copy()
        model = VAR(model_order=p)
        model.fit(segment)
        coefs[wi] = model.coef
        rescovs[wi] = model.rescov

        # Observed PDC averaged over frequency (fixed internal nfft)
        conn = Connectivity(model.coef, c=model.rescov, nfft=64)
        pdc = conn.PDC()  # (m, m, 64)
        pdc_mean_obs = pdc.mean(axis=-1)
        pdcs[wi] = pdc_mean_obs
        
        # Optional: permutation-based null with circular shifts (max-T correction)
        # Create independent seeds per permutation, varied by window index for diversity
        seed_rng = np.random.default_rng(42 + wi)
        seeds = seed_rng.integers(0, 2**32 - 1, size=n_permutations, dtype=np.uint64)

        # Run permutations in parallel using joblib
        null[:,wi] = np.asarray(Parallel(n_jobs=n_jobs, prefer='processes')(
                delayed(_permute_pdc)(int(seeds[r]), segment, p) for r in range(n_permutations)
            ))

        # One-sided p-values: proportion of null >= observed
        pvals[wi] = (1 + (null[:, wi] >= pdcs[wi]).sum(axis=0)) / (1 + n_permutations)
    
    out_root = BIDSPath(
        root='results/PhonemeSequence(subject)',
        subject=subject,
        suffix='var',
        datatype='connectivity',
        description=description,
        extension='.h5',
        check=False
    )
    out_root.mkdir(parents=True, exist_ok=True)

    # Save HDF5 with clean structure and metadata
    with h5py.File(out_root, 'w') as f:
        f.create_dataset('start_samples', data=start_samples)
        f.create_dataset('end_samples', data=end_samples)
        f.create_dataset('coef', data=coefs)
        f.create_dataset('rescov', data=rescovs)
        f.create_dataset('pdc', data=pdcs)
        f.create_dataset('pvals', data=pvals)
        f.create_dataset('null', data=null)

        # Global metadata
        f.attrs['subject'] = subject
        f.attrs['band'] = band
        f.attrs['description'] = description
        f.attrs['fs'] = float(fs)
        f.attrs['tmin'] = float(tmin)
        f.attrs['tmax'] = float(tmax)
        f.attrs['window'] = float(window)
        f.attrs['step'] = float(step)
        f.attrs['model_order'] = int(p)
        f.attrs['data_shape'] = np.array(epochs._data.shape)  # (n_epochs, n_channels, n_samples)
        # Store channel names as JSON to avoid variable-length string dtype issues
        f.attrs['channel_names_json'] = json.dumps(epochs.ch_names)

    logger.info(f'Saved connectivity results to: {out_root}')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/epoch(WM)")
    parser.add_argument("--recon_dir", type=str,
                        default="/cwork/ns458/ECoG_Recon/")
    parser.add_argument("--subject", type=str, default="D0088")
    parser.add_argument("--band", type=str, default='highgamma')
    parser.add_argument("--window", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--description", type=str, default='production')
    parser.add_argument("--n_permutations", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))