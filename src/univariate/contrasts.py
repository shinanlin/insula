"""Univariate HGA contrasts: Decision vs Repeat AND Word vs Nonword.

Uses time_perm_cluster from ieeg to compute per-channel cluster
permutation tests on epoch-level HGA data.

Usage:
    python src/univariate/contrasts.py --bids_root /cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/
"""
import numpy as np
import pandas as pd
import mne
import h5py
import os
import argparse
import logging
from mne_bids import BIDSPath
from ieeg.calc.stats import time_perm_cluster
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EXCLUDE_CHN = [
    'D0040_L1IF3-4', 'D0084_RFAI1-2', 'D0086_LTPI2-3', 'D0032_LAI4-5',
    'D0090_RIA4-5', 'D0096_LFAI2-3', 'D0096_LFAI4-5', 'D0102_RFAI2-3',
    'D0106_LTAS2-3', 'D0121_LFMI3-4', 'D0122_LFAI3-4', 'D0125_LIA4-5',
    'D0125_LIA7-8',
]


def run_contrast(group_a, group_b, label_a, label_b, times, channels, n_perm=2000):
    """Run time_perm_cluster on two groups of trial-level data.
    
    Parameters
    ----------
    group_a, group_b : xarray.DataArray, shape (trial, channel, time)
    label_a, label_b : str, labels for the two conditions
    
    Returns
    -------
    pd.DataFrame with columns: channel, time, mask, p, direction
    """
    mask, p = time_perm_cluster(
        group_a.values,
        group_b.values,
        p_thresh=0.05,
        n_perm=n_perm,
        ignore_adjacency=1,
        n_jobs=-1,
        tails=2,
    )
    
    # Compute per-channel×time means for both groups
    mean_a = np.nanmean(group_a.values, axis=0)  # (channel, time)
    mean_b = np.nanmean(group_b.values, axis=0)  # (channel, time)
    mean_diff = mean_a - mean_b                   # (channel, time)
    
    df = pd.DataFrame({
        'channel': np.repeat(channels, len(times)),
        'time': np.tile(times, len(channels)),
        'mask': mask.ravel(),
        'p': p.ravel(),
        'mean_a': mean_a.ravel(),
        'mean_b': mean_b.ravel(),
        'mean_diff': mean_diff.ravel(),
    })
    
    # Determine direction for significant channels
    hit = df['mask'].astype(bool) & (df['p'] < 0.05)
    df['significant'] = hit.groupby(df['channel']).transform('any')
    
    # Compute overall direction per channel
    diff_per_ch = np.nanmean(mean_diff, axis=1)  # (channel,)
    ch_direction = {ch: label_a if d > 0 else label_b 
                    for ch, d in zip(channels, diff_per_ch)}
    df['direction'] = df['channel'].map(ch_direction)
    df.loc[~df['significant'], 'direction'] = np.nan
    
    return df


def main(bids_root, band, reference='bipolar', n_perm=2000, subject=None):
    
    pt = BIDSPath(
        root=os.path.join(bids_root, 'derivatives', f'epoch({reference})'),
        datatype='epoch(band)(zscore)',
        suffix=band,
        subject=subject,
        extension='.h5',
        check=False,
    )
    
    pts = pt.match()
    subjects = sorted(set([p.subject for p in pts]))
    logger.info(f"Found {len(subjects)} subjects")
    
    logger.info(f"Processing {subject}")
    
    this_sub_pts = pt.copy().update(subject=subject).match()
    
    # Load parcellation
    parc_path = this_sub_pts[0].copy().update(
        root=str(this_sub_pts[0].root).replace(f'epoch({reference})', 'parcellation'),
        datatype=reference,
        task=None, description=None, recording=None,
        processing='3mm', suffix='aparc2009s', extension='.csv',
    ).match()[0]
    parc = pd.read_csv(parc_path)
    parc = parc[~parc['roi'].str.contains('white|intersection|unknown|WM', case=False, na=False)]
    this_sub_channel = parc.name.unique().tolist()
    this_sub_channel = [ch for ch in this_sub_channel if ch not in EXCLUDE_CHN]
    
    # Load all epochs for this subject
    this_sub_df = []
    for this_sub_pt in this_sub_pts:
        try:
            dc_epo = mne.read_epochs(this_sub_pt, verbose='error')
            include = set(this_sub_channel) & set(dc_epo.ch_names)
            if not include:
                continue
            dc_epo.pick_channels(list(include))
            df = dc_epo.to_data_frame(long_format=True, scalings={'seeg': 1}, verbose=False)
            
            # Parse condition string: task/modality/lexicality/word/remark
            parts = df['condition'].str.split('/')
            df['lexicality'] = parts.str[2]
            df['word'] = parts.str[3]
            df['remark'] = parts.str[4:].str.join('/')
            df['condition_short'] = parts.str[2:4].str.join('/')
            df['description'] = this_sub_pt.description
            df['phase'] = str(this_sub_pt.processing).lower()
            df['subject'] = this_sub_pt.subject
            
            # Trial ID
            trial_map = (
                df[['epoch', 'condition_short']]
                .drop_duplicates()
                .assign(_idx=lambda d: d.groupby('condition_short').cumcount() + 1)
                .assign(trial=lambda d: d['condition_short'] + '_' + d['_idx'].astype(str))
                [['epoch', 'condition_short', 'trial']]
            )
            df = df.merge(trial_map, on=['epoch', 'condition_short'], how='left')
            this_sub_df.append(df)
        except Exception as e:
            logger.error(f"Error loading {this_sub_pt}: {e}")
            continue
    
    if not this_sub_df:
        logger.warning(f"No data for {subject}, skipping")
        return
        
    epos = pd.concat(this_sub_df, ignore_index=True)
    
    # Filter to CORRECT trials only
    epos = epos[epos.remark == 'CORRECT']
    logger.info(f"  {subject}: {len(epos)} rows after filtering correct trials")
    
    # Collect results per contrast type
    contrast_results = {
        'DecisionVsRepeat': [],
        'WordVsNonwordDecision': [],
        'WordVsNonwordRepeat': [],
    }
    
    for phase in tqdm(epos.phase.unique(), desc=f"{subject} phases"):
        phase = str(phase).lower()
        phase_data = epos[epos.phase == phase]
        
        # ───── Contrast 1: Decision vs Repeat ─────
        dec = phase_data[phase_data.description == 'Decision']
        rep = phase_data[phase_data.description == 'Repeat']
        
        if len(dec) > 0 and len(rep) > 0:
            dec_da = dec.groupby(['trial', 'channel', 'time'])['value'].mean().to_xarray()
            rep_da = rep.groupby(['trial', 'channel', 'time'])['value'].mean().to_xarray()
            
            common_ch = sorted(set(dec_da.channel.values) & set(rep_da.channel.values))
            if len(common_ch) > 0:
                dec_da = dec_da.sel(channel=common_ch)
                rep_da = rep_da.sel(channel=common_ch)
                
                result = run_contrast(
                    dec_da, rep_da, 'Decision', 'Repeat',
                    dec_da.time.values, common_ch, n_perm=n_perm
                )
                result['phase'] = phase
                contrast_results['DecisionVsRepeat'].append(result)
        
        # ───── Contrast 2: Word vs Nonword (within Decision) ─────
        dec_word = phase_data[(phase_data.description == 'Decision') & (phase_data.lexicality == 'Word')]
        dec_nw = phase_data[(phase_data.description == 'Decision') & (phase_data.lexicality == 'Nonword')]
        
        if len(dec_word) > 0 and len(dec_nw) > 0:
            w_da = dec_word.groupby(['trial', 'channel', 'time'])['value'].mean().to_xarray()
            nw_da = dec_nw.groupby(['trial', 'channel', 'time'])['value'].mean().to_xarray()
            
            common_ch = sorted(set(w_da.channel.values) & set(nw_da.channel.values))
            if len(common_ch) > 0:
                w_da = w_da.sel(channel=common_ch)
                nw_da = nw_da.sel(channel=common_ch)
                
                result = run_contrast(
                    w_da, nw_da, 'Word', 'Nonword',
                    w_da.time.values, common_ch, n_perm=n_perm
                )
                result['phase'] = phase
                contrast_results['WordVsNonwordDecision'].append(result)
        
        # ───── Contrast 3: Word vs Nonword (within Repeat) ─────
        rep_word = phase_data[(phase_data.description == 'Repeat') & (phase_data.lexicality == 'Word')]
        rep_nw = phase_data[(phase_data.description == 'Repeat') & (phase_data.lexicality == 'Nonword')]
        
        if len(rep_word) > 0 and len(rep_nw) > 0:
            w_da = rep_word.groupby(['trial', 'channel', 'time'])['value'].mean().to_xarray()
            nw_da = rep_nw.groupby(['trial', 'channel', 'time'])['value'].mean().to_xarray()
            
            common_ch = sorted(set(w_da.channel.values) & set(nw_da.channel.values))
            if len(common_ch) > 0:
                w_da = w_da.sel(channel=common_ch)
                nw_da = nw_da.sel(channel=common_ch)
                
                result = run_contrast(
                    w_da, nw_da, 'Word', 'Nonword',
                    w_da.time.values, common_ch, n_perm=n_perm
                )
                result['phase'] = phase
                contrast_results['WordVsNonwordRepeat'].append(result)
    
    # Save each contrast as a separate CSV
    task_name = this_sub_pts[0].task
    for contrast_name, dfs in contrast_results.items():
        if not dfs:
            continue
        df_out = pd.concat(dfs, ignore_index=True)
        save_path = BIDSPath(
            root=f'results/{task_name}({reference})',
            datatype='univariate',
            suffix=band,
            task=task_name,
            subject=subject,
            description=contrast_name,
            extension=".csv",
            check=False,
        )
        save_path.mkdir(exist_ok=True)
        df_out.to_csv(save_path, index=False)
        logger.info(f"  Saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/",
                        help="Root directory of the BIDS dataset")
    parser.add_argument("--band", type=str, default='highgamma',
                        choices=['highgamma', 'gamma', 'beta', 'alpha', 'theta', 'lowband'])
    parser.add_argument("--n_perm", type=int, default=2000)
    parser.add_argument("--subject", type=str, default='D0024',
                        help="Process a single subject (e.g. D0023). If not set, processes all.")
    args = parser.parse_args()
    main(**vars(args))
