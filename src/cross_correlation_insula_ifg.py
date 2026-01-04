# Cross-correlation analysis between Insula and IFG electrodes
# to detect volume conduction (peak at lag=0)

import argparse
from typing import List, Tuple
import numpy as np
from mne_bids import BIDSPath
import pandas as pd
import mne
import logging
from tqdm import tqdm
from scipy.signal import correlate
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Label keywords for IFG and Insula regions
IFG_KEYWORDS = (
    'G_front_inf-Opercular',
    'G_front_inf-Triangul',
    'G_front_inf-Orbital',
)

INSULA_KEYWORDS = (
    'G_insular_short',
    'S_circular_insula',
    'G_Ins_lg_and_S_cent_ins',
)


def is_insula_label(label: str) -> bool:
    """Check if label belongs to Insula."""
    if not label or label == 'Unknown':
        return False
    return any(kw in str(label) for kw in INSULA_KEYWORDS)


def is_ifg_label(label: str) -> bool:
    """Check if label belongs to IFG."""
    if not label or label == 'Unknown':
        return False
    return any(kw in str(label) for kw in IFG_KEYWORDS)


def plot_xcorr(xcorrs: pd.DataFrame):
    """
    Plot cross-correlation matrix for a subject-phase combination.
    
    Parameters
    ----------
    xcorrs : pd.DataFrame
        Cross-correlation data with columns: subject, phase, insula_channel, ifg_channel, lag, correlation
    """
    # Convert cm to inches for matplotlib
    cm = 1/2.54
    
    n_ins = len(xcorrs.insula_channel.unique())
    n_ifg = len(xcorrs.ifg_channel.unique())
    
    fig, ax = plt.subplots(n_ins, n_ifg, figsize=(n_ifg*3*cm, n_ins*3*cm))
    
    # Handle case where only one row or column
    if n_ins == 1 and n_ifg == 1:
        ax = np.array([[ax]])
    elif n_ins == 1:
        ax = ax.reshape(1, -1)
    elif n_ifg == 1:
        ax = ax.reshape(-1, 1)
    
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i, ins_ch in enumerate(xcorrs.insula_channel.unique()):
        ins_ch_short = ins_ch.split("_")[-1]
        
        ax[i,0].set_ylabel(ins_ch_short, fontsize=7)
        
        for j, ifg_ch in enumerate(xcorrs.ifg_channel.unique()):
            ifg_ch_short = ifg_ch.split("_")[-1]
            
            # Set title for top row
            if i == 0:
                ax[0,j].set_title(ifg_ch_short, fontsize=7)
            
            # Plot cross-correlation
            subset = xcorrs[(xcorrs.insula_channel==ins_ch) & (xcorrs.ifg_channel==ifg_ch)]
            if len(subset) > 0:
                sns.lineplot(data=subset, x='lag', y='correlation', ax=ax[i,j], lw=1)
            
            ax[i,j].axvline(0, color='r', linewidth=0.5, linestyle='--')
            ax[i,j].set_xlabel('')
            ax[i,-1].set_ylabel('')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            
            ax[i,j].spines['top'].set_visible(False)
            ax[i,j].spines['right'].set_visible(False)
            ax[i,j].spines['left'].set_visible(False)
            ax[i,j].tick_params(labelsize=7, width=0.5, length=2, which='both')
            plt.setp(ax[i,j].spines.values(), linewidth=0.75)
    
    plt.suptitle(f'{xcorrs.subject.unique()[0]}-{xcorrs.phase.unique()[0]}', fontsize=10)
    
    return fig


def compute_xcorr(x: np.ndarray, y: np.ndarray, dt: float) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Compute cross-correlation between two time series.
    
    Returns
    -------
    zero_lag_corr : float
        Pearson correlation at zero lag
    peak_corr : float
        Peak correlation value (normalized)
    peak_lag : float
        Time lag at peak correlation (seconds)
    lags : np.ndarray
        All lag values
    xcorr : np.ndarray
        Full cross-correlation function (normalized)
    """
    # Zero-lag correlation
    zero_lag_corr, _ = pearsonr(x, y)
    
    # Normalize for cross-correlation
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)
    
    # Cross-correlation
    xcorr = correlate(x_norm, y_norm, mode='full') / len(x)
    
    # Lags
    lags = np.arange(-len(x) + 1, len(x)) * dt
    
    # Peak
    peak_idx = np.argmax(np.abs(xcorr))
    peak_corr = xcorr[peak_idx]
    peak_lag = lags[peak_idx]
    
    return zero_lag_corr, peak_corr, peak_lag, lags, xcorr


def main(
    bids_root: str,
    band: str,
    ref: str,
):
    epoch_paths = BIDSPath(
        root=bids_root + f"derivatives/epoch({ref})",
        suffix=band,
        datatype='epoch(band)(sig)(effective)',
        extension=".h5",
        check=False,
    )
    
    for epoch_path in tqdm(epoch_paths.match(), desc='Processing subjects'):
        
        results = []
        
        subject = epoch_path.subject
        phase = epoch_path.processing
        desc = epoch_path.description
        phase = 'Response' if desc == 'production' else 'Audio'
        
        # Load parcellation
        parc_path = epoch_path.copy().update(
            root=str(epoch_path.root).replace(f'epoch({ref})', 'parcellation'),
            datatype=ref,
            task=None,
            description=None,
            processing='3mm',
            suffix='aparc2009s',
            extension='.csv',
        ).match()
        
        if not parc_path:
            logger.warning(f"No parcellation for {subject}, skipping")
            continue
        
        parc = pd.read_csv(parc_path[0])
        parc.rename(columns={'name': 'channel'}, inplace=True)
        
        # Find Insula and IFG channels
        insula_channels = parc[parc['label'].apply(is_insula_label)]['channel'].tolist()
        ifg_channels = parc[parc['label'].apply(is_ifg_label)]['channel'].tolist()
        
        if not insula_channels or not ifg_channels:
            logger.info(f"{subject}: Insula={len(insula_channels)}, IFG={len(ifg_channels)}, skipping")
            continue
        
        logger.info(f"{subject}: Insula={len(insula_channels)}, IFG={len(ifg_channels)}")
        
        # Load epochs
        epochs = mne.read_epochs(epoch_path, preload=True)
        evoked = epochs.average()
        times = evoked.times
        dt = times[1] - times[0] if len(times) > 1 else 0.001
        
        # Get data as DataFrame
        df = evoked.to_data_frame(long_format=True, scalings={'seeg': 1})
        
        # Filter to only Insula and IFG channels
        available_channels = set(df['channel'].unique())
        insula_channels = [ch for ch in insula_channels if ch in available_channels]
        ifg_channels = [ch for ch in ifg_channels if ch in available_channels]
        
        if not insula_channels or not ifg_channels:
            continue
        
        # Compute cross-correlation for each pair
        for ins_ch in insula_channels:
            ins_data = df[df['channel'] == ins_ch].sort_values('time')['value'].values
            ins_label = parc[parc['channel'] == ins_ch]['label'].values[0]
            
            for ifg_ch in ifg_channels:
                ifg_data = df[df['channel'] == ifg_ch].sort_values('time')['value'].values
                ifg_label = parc[parc['channel'] == ifg_ch]['label'].values[0]
                
                if len(ins_data) != len(ifg_data) or len(ins_data) < 10:
                    continue
                
                zero_lag_corr, peak_corr, peak_lag, lags, xcorr = compute_xcorr(ins_data, ifg_data, dt)
                
                # Store full cross-correlation
                for lag, corr in zip(lags, xcorr):
                    results.append({
                        'subject': subject,
                        'phase': phase,
                        'task': epoch_path.task,
                        'insula_channel': ins_ch,
                        'ifg_channel': ifg_ch,
                        'insula_label': ins_label,
                        'ifg_label': ifg_label,
                        'lag': lag,
                        'correlation': corr,
                        'zero_lag_corr': zero_lag_corr,
                        'peak_corr': peak_corr,
                        'peak_lag': peak_lag,
                        'phase': phase,
                        'n_timepoints': len(ins_data),
                    })
    
        results_df = pd.DataFrame(results)
        

        save_path = epoch_path.copy().update(
            root = f'results/{epoch_path.task}({ref})',
            extension='.csv',
            datatype='xcorr',
            suffix='ifg',
            check=False,
        )
        save_path.mkdir(exist_ok=True)
        results_df.to_csv(save_path, index=False)
        logger.info(f"Saved {len(results_df)} rows to {save_path}")
        
        if len(results_df) > 0:
            # Plot cross-correlation matrix for this subject-phase
            fig = plot_xcorr(results_df)
            fig.savefig(save_path.copy().update(extension='.png', suffix='ifg'), dpi=300, bbox_inches='tight')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/", type=str)
    parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/", type=str)
    parser.add_argument("--band", type=str, default="highgamma", choices=['highgamma', 'gamma', 'beta', 'alpha', 'theta'])
    parser.add_argument("--ref", type=str, default='bipolar', choices=['bipolar', 'car'])
    args = parser.parse_args()
    main(**vars(args))
