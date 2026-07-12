"""Cross-task univariate HGA contrasts: LexicalDelay vs LexicalNoDelay.

Compares response-phase HGA between Delay and NoDelay tasks for common electrodes.
Outputs both mean_diff and peak_diff metrics.

Usage:
    python src/archive/cross_task_contrast.py --subject D0024
"""
import numpy as np
import pandas as pd
import mne
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

DELAY_ROOT = "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"
NODELAY_ROOT = "/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/"


def run_contrast(group_a, group_b, label_a, label_b, times, channels, n_perm=2000):
    """Run time_perm_cluster on two groups of trial-level data.
    
    Returns DataFrame with mean_diff and peak_diff.
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
    
    mean_a = np.nanmean(group_a.values, axis=0)  # (channel, time)
    mean_b = np.nanmean(group_b.values, axis=0)
    mean_diff = mean_a - mean_b
    
    # Peak difference: max(abs(a)) - max(abs(b)) per channel
    peak_a = np.nanmax(np.abs(group_a.values), axis=(0, 2))  # (channel,)
    peak_b = np.nanmax(np.abs(group_b.values), axis=(0, 2))
    peak_diff_per_ch = peak_a - peak_b
    
    df = pd.DataFrame({
        'channel': np.repeat(channels, len(times)),
        'time': np.tile(times, len(channels)),
        'mask': mask.ravel(),
        'p': p.ravel(),
        'mean_a': mean_a.ravel(),
        'mean_b': mean_b.ravel(),
        'mean_diff': mean_diff.ravel(),
    })
    
    # Add peak_diff (same value for all time points of a channel)
    ch_peak_map = dict(zip(channels, peak_diff_per_ch))
    df['peak_diff'] = df['channel'].map(ch_peak_map)
    
    # Significance
    hit = df['mask'].astype(bool) & (df['p'] < 0.05)
    df['significant'] = hit.groupby(df['channel']).transform('any')
    
    # Direction based on mean_diff
    diff_per_ch = np.nanmean(mean_diff, axis=1)
    ch_direction = {ch: label_a if d > 0 else label_b 
                    for ch, d in zip(channels, diff_per_ch)}
    df['direction'] = df['channel'].map(ch_direction)
    df.loc[~df['significant'], 'direction'] = np.nan
    
    return df


def load_epochs_for_task(bids_root, subject, band, reference, phase, description):
    """Load epoch data for a specific task/phase/description."""
    pt = BIDSPath(
        root=os.path.join(bids_root, 'derivatives', f'epoch({reference})'),
        datatype='epoch(band)(zscore)',
        suffix=band,
        subject=subject,
        processing=phase,
        description=description,
        extension='.h5',
        check=False,
    )
    
    matches = pt.match()
    if not matches:
        return None, None
    
    # Load parcellation for channel filtering
    parc_path = matches[0].copy().update(
        root=str(matches[0].root).replace(f'epoch({reference})', 'parcellation'),
        datatype=reference,
        task=None, description=None, recording=None, processing='3mm',
        suffix='aparc2009s', extension='.csv',
    )
    parc_matches = parc_path.match()
    
    valid_channels = None
    if parc_matches:
        parc = pd.read_csv(parc_matches[0])
        parc = parc[~parc['roi'].str.contains('white|intersection|unknown|WM', case=False, na=False)]
        valid_channels = [ch for ch in parc.name.unique() if ch not in EXCLUDE_CHN]
    
    all_dfs = []
    for match in matches:
        try:
            epo = mne.read_epochs(match, verbose='error')
            if valid_channels:
                include = set(valid_channels) & set(epo.ch_names)
                if not include:
                    continue
                epo.pick_channels(list(include))
            
            df = epo.to_data_frame(long_format=True, scalings={'seeg': 1}, verbose=False)
            
            # Parse condition
            parts = df['condition'].str.split('/')
            df['lexicality'] = parts.str[2]
            df['word'] = parts.str[3]
            df['remark'] = parts.str[4:].str.join('/')
            df['condition_short'] = parts.str[2:4].str.join('/')
            
            # Trial ID
            trial_map = (
                df[['epoch', 'condition_short']]
                .drop_duplicates()
                .assign(_idx=lambda d: d.groupby('condition_short').cumcount() + 1)
                .assign(trial=lambda d: d['condition_short'] + '_' + d['_idx'].astype(str))
                [['epoch', 'condition_short', 'trial']]
            )
            df = df.merge(trial_map, on=['epoch', 'condition_short'], how='left')
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {match}: {e}")
            continue
    
    if not all_dfs:
        return None, None
    
    epos = pd.concat(all_dfs, ignore_index=True)
    epos = epos[epos.remark == 'CORRECT']
    
    return epos, valid_channels


def main(subject, band='highgamma', reference='bipolar', n_perm=2000):
    """Run cross-task contrast for a single subject."""
    
    logger.info(f"Processing {subject}")
    
    results = []
    
    for desc in ['Decision', 'Repeat']:
        logger.info(f"  Loading {desc} Response data...")
        
        # Load Delay task
        delay_df, delay_ch = load_epochs_for_task(
            DELAY_ROOT, subject, band, reference, 'Response', desc
        )
        
        # Load NoDelay task
        nodelay_df, nodelay_ch = load_epochs_for_task(
            NODELAY_ROOT, subject, band, reference, 'Response', desc
        )
        
        if delay_df is None or nodelay_df is None:
            logger.warning(f"  Missing data for {desc}, skipping")
            continue
        
        # Find common channels
        delay_channels = set(delay_df['channel'].unique())
        nodelay_channels = set(nodelay_df['channel'].unique())
        common_ch = sorted(delay_channels & nodelay_channels)
        
        if not common_ch:
            logger.warning(f"  No common channels for {desc}, skipping")
            continue
        
        logger.info(f"  Found {len(common_ch)} common channels for {desc}")
        
        # Filter to common channels
        delay_df = delay_df[delay_df['channel'].isin(common_ch)]
        nodelay_df = nodelay_df[nodelay_df['channel'].isin(common_ch)]
        
        # Convert to xarray
        delay_da = delay_df.groupby(['trial', 'channel', 'time'])['value'].mean().to_xarray()
        nodelay_da = nodelay_df.groupby(['trial', 'channel', 'time'])['value'].mean().to_xarray()
        
        # Align channels
        delay_da = delay_da.sel(channel=common_ch)
        nodelay_da = nodelay_da.sel(channel=common_ch)
        
        # Run contrast: Delay - NoDelay
        # Positive = Delay > NoDelay (vocal stronger)
        # Negative = NoDelay > Delay (button stronger for Decision)
        result = run_contrast(
            delay_da, nodelay_da, 'Delay', 'NoDelay',
            delay_da.time.values, common_ch, n_perm=n_perm
        )
        result['description'] = desc
        result['phase'] = 'response'
        results.append(result)
    
    if not results:
        logger.warning(f"No results for {subject}")
        return
    
    df_out = pd.concat(results, ignore_index=True)
    
    # Save
    save_path = BIDSPath(
        root='results/CrossTask(bipolar)',
        datatype='univariate',
        suffix=band,
        task='CrossTask',
        subject=subject,
        description='DelayVsNoDelay',
        extension=".csv",
        check=False,
    )
    save_path.mkdir(exist_ok=True)
    df_out.to_csv(save_path, index=False)
    logger.info(f"Saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default='D0024',
                        help="Subject ID (e.g. D0024)")
    parser.add_argument("--band", type=str, default='highgamma',
                        choices=['highgamma', 'gamma', 'beta', 'alpha', 'theta', 'lowband'])
    parser.add_argument("--n_perm", type=int, default=2000)
    args = parser.parse_args()
    main(**vars(args))
