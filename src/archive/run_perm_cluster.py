#!/usr/bin/env python3
"""
Permutation test (time cluster) with channel-averaged trials per ROI.
- Loads z-scored epochs from derivatives/epoch(bipolar)
- Averages channels within ROI per trial, keeps trials as observations
- Runs time_perm_cluster between Decision and Repeat per phase
- Saves results with ROI column
"""

import argparse
import logging
import os
from typing import Dict, List

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from ieeg.calc.stats import time_perm_cluster

# ---- ROI helpers (adapted from condition_decoding.py + notebook mapping) ----
EXCLUDE_CHANNELS = [
    'D0040_L1IF2-3', 'D0040_L1IF3-4', 'D0079_LFAI1-2', 'D0079_LFAI2-3',
    'D0079_LFMI1-2', 'D0079_LPI9-10', 'D0103_LAI7-8', 'D0103_LAI3-4',
    'D0103_LAI5-6', 'D0103_LAI6-7',
]

# Fixed phases (loop, no user input)
PHASES = ['Stimulus', 'Delay', 'Go', 'Response']


def split_insula_ap(parcs_df: pd.DataFrame, y_threshold: float = 0) -> pd.DataFrame:
    """Add insula_region column with AIC/PIC labels based on parcellation rules."""
    coords_classified = parcs_df.copy()
    coords_classified['insula_region'] = None

    aic_conditions = (
        coords_classified['label'].str.contains('G_insular_short', na=False)
        | coords_classified['label'].str.contains('S_circular_insula_ant', na=False)
        | (
            coords_classified['label'].str.contains('S_circular_insula_sup', na=False)
            & (coords_classified['y'] > y_threshold)
        )
        | (
            coords_classified['label'].str.contains('S_circular_insula_inf', na=False)
            & (coords_classified['y'] > y_threshold)
        )
    )

    pic_conditions = (
        coords_classified['label'].str.contains('G_Ins_lg_and_S_cent_ins', na=False)
        | (
            coords_classified['label'].str.contains('S_circular_insula_sup', na=False)
            & (coords_classified['y'] <= y_threshold)
        )
        | (
            coords_classified['label'].str.contains('S_circular_insula_inf', na=False)
            & (coords_classified['y'] <= y_threshold)
        )
    )

    coords_classified.loc[aic_conditions, 'insula_region'] = 'AIC'
    coords_classified.loc[pic_conditions, 'insula_region'] = 'PIC'

    mask = coords_classified['insula_region'].notna()
    coords_classified.loc[mask, 'roi'] = coords_classified.loc[mask, 'insula_region']
    coords_classified = coords_classified.drop(columns=['insula_region'])
    return coords_classified


def get_roi_channels(
    bids_root: str,
    ref: str,
    roi: str,
    hemi: str = 'B',
    y_threshold: float = 0,
) -> Dict[str, List[str]]:
    
    """Return mapping subj -> channels for given ROI (with AIC/PIC split)."""
    parc_paths = BIDSPath(
        root=os.path.join(bids_root, 'derivatives', 'parcellation'),
        datatype=ref,
        task=None,
        description=None,
        recording=None,
        processing='3mm',
        suffix='aparc2009s',
        extension='.csv',
        check=False,
    ).match()

    if len(parc_paths) == 0:
        raise FileNotFoundError('No parcellation CSV found under derivatives/parcellation')

    parcs = pd.concat([pd.read_csv(p) for p in parc_paths], ignore_index=True)
    parcs = parcs[~parcs['name'].isin(EXCLUDE_CHANNELS)]
    parcs = split_insula_ap(parcs, y_threshold=y_threshold)

    # Map to match notebook conventions
    parcs.loc[parcs.roi == 'PrG', 'roi'] = 'SMC'
    parcs.loc[parcs.roi == 'PoG', 'roi'] = 'SMC'
    parcs.loc[parcs.roi == 'Subcentral', 'roi'] = 'SMC'
    parcs.loc[parcs.roi == 'CG', 'roi'] = 'ACC'

    if hemi != 'B':
        parcs = parcs[parcs['hemi'] == hemi]

    parcs = parcs[parcs['roi'] == roi]
    subj_to_channels: Dict[str, List[str]] = {}
    for sub, df_sub in parcs.groupby('subject'):
        subj_to_channels[sub] = df_sub['name'].unique().tolist()
    return subj_to_channels


# ---- Data collection ----


def main(
    bids_root: str,
    roi: str,
    hemi: str,
    band: str,
    ref: str,
):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Collecting epochs (channel-averaged) for ROI=%s', roi)
    subj_to_channels = get_roi_channels(bids_root, ref, roi, hemi)
    epos = []

    for phase in PHASES:
        for sub, picks in subj_to_channels.items():
            if len(picks) == 0:
                continue
            base = BIDSPath(
                root=os.path.join(bids_root, 'derivatives', 'epoch(bipolar)'),
                subject=sub,
                datatype='epoch(band)(zscore)',
                processing=phase,
                suffix=band,
                extension='.h5',
                check=False,
            )
            for condition in ['Decision', 'Repeat']:
                matches = base.update(description=condition).match()
                if len(matches) == 0:
                    logging.warning('No epoch found for %s %s %s', sub, phase, condition)
                    continue
                epo_path = matches[0]
                dc_epo = mne.read_epochs(epo_path, verbose='error')
                dc_epo.pick(picks)
                data = dc_epo.get_data()  # (trial, channel, time)
                if data.size == 0:
                    continue
                mean_ch = data.mean(axis=1)  # (trial, time)
                times = dc_epo.times
                trial_ids = [f"{sub}_trial{i}" for i in range(mean_ch.shape[0])]
                df = pd.DataFrame(mean_ch, columns=times)
                df['trial'] = trial_ids
                df = df.melt(id_vars=['trial'], var_name='time', value_name='value')
                df['condition'] = condition
                df['phase'] = phase.lower()
                df['subject'] = sub
                df['roi'] = roi
                epos.append(df)

    if not epos:
        raise RuntimeError('No epochs collected; check inputs and ROI/channel mapping.')

    epos = pd.concat(epos, ignore_index=True)

    logging.info('Running permutation tests (trial-level, time cluster)')
    perms = []
    
    for phase in PHASES:
        dec_df = epos[(epos.condition == 'Decision') & (epos.phase == phase.lower())]
        rep_df = epos[(epos.condition == 'Repeat') & (epos.phase == phase.lower())]
        if dec_df.empty or rep_df.empty:
            logging.warning('No data for phase %s; skipping', phase)
            continue
        decision = dec_df.pivot(index='trial', columns='time', values='value').sort_index(axis=1)
        repeat = rep_df.pivot(index='trial', columns='time', values='value').sort_index(axis=1)
        
        mask, pvals = time_perm_cluster(
            decision.values,
            repeat.values,
            p_thresh=0.1,
            n_perm=5000,
            n_jobs=-1,
        )
        
        times = decision.columns.values
        res = pd.DataFrame({
            'time': times,
            'mask': mask,
            'pval': pvals,
            'phase': phase.lower(),
            'roi': roi,
        })
        perms.append(res)

    perms = pd.concat(perms, ignore_index=True)

    task = epo_path.task
    out_path = BIDSPath(
        root=os.path.join('results', f'{task}(roi)(bipolar)'),
        subject=roi,
        suffix='perm',
        extension='.csv',
        check=False
    )
    perms.to_csv(out_path, index=False)
    logging.info('Saved permutation results to %s', out_path)
    
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--bids_root', 
                        default='/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/', 
                        help='Path to BIDS root')
    parser.add_argument('--roi', default='AIC', 
                        choices=['AIC', 'PIC', 'STG', 'IFG', 'SMC', 'ACC'], help='ROI to analyze')
    parser.add_argument('--hemi', default='B', 
                        choices=['L', 'R', 'B'], help='Hemisphere (default: both)')
    parser.add_argument('--band', 
                        default='highgamma', 
                        help='Band suffix in epoch files')
    parser.add_argument('--ref', 
                        default='bipolar', 
                        help='Reference name used in derivatives')

    args = parser.parse_args()
    main(**vars(args))