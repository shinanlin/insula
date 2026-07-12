import argparse
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath
import logging
from tqdm import tqdm
import re
import os
import sys

# Add the directory containing run_xcorr to path so we can import it
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.xcorr.run_xcorr import compute_xcorr_matrix
from ieeg.viz.mri import force2frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(bids_root: str, band: str, reference: str, recon_dir: str):
    datatype = 'epoch(raw)' if band=='raw' else 'epoch(band)(raw)'
    raw_pts = BIDSPath(
        root=bids_root + f"derivatives/epoch({reference})",
        datatype=datatype,
        suffix=band,
        extension='.h5',
        check=False
    )
    
    task_name = bids_root.rstrip('/').split('_')[-1].split('/')[0] # Usually LexicalDecRepDelay etc.
    if 'Lexical' in bids_root:
        if 'NoDelay' in bids_root: task_name = 'LexicalNoDelay'
        else: task_name = 'LexicalDelay'
    elif 'Phoneme' in bids_root: task_name = 'PhonemeSequence'
    elif 'Sentence' in bids_root: task_name = 'SentenceRep'
    elif 'Picture' in bids_root: task_name = 'PictureNaming'
    
    all_rows = []
    
    for raw_pt in tqdm(raw_pts.match(), desc='Processing subjects'):
        subject = raw_pt.subject
        phase = raw_pt.processing
        
        raw = mne.read_epochs(raw_pt, preload=True)
        
        try:
            parc_path = raw_pt.copy().update(
                root=str(raw_pt.root).replace(f'epoch({reference})', 'parcellation'),
                datatype=reference, task=None, description=None, recording=None,
                processing='3mm', suffix='aparc2009s', extension='.csv',
            ).match()[0]
            parc = pd.read_csv(parc_path)
        except IndexError:
            logger.warning(f"No parcellation file found for subject {subject}")
            continue
            
        parc.rename(columns={'name': 'channel'}, inplace=True)
        parc_sub = parc[['channel', 'label','roi','hemi']]
        
        montage = raw.get_montage()
        sub_id = re.sub(r'^D0+', 'D', subject)
        to_fsaverage = mne.read_talxfm(sub_id, recon_dir)
        trans = mne.transforms.Transform(fro='head', to='mri', trans=to_fsaverage['trans'])
        force2frame(montage, trans.from_str)  
        montage.apply_trans(trans) 
        pos_m = montage.get_positions()['ch_pos']
        
        cord_df = pd.DataFrame(pos_m).T
        cord_df.columns = ['x', 'y', 'z']
        cord_df = cord_df.reset_index().rename(columns={'index': 'channel'})
        cord_df = cord_df.merge(parc_sub, on='channel', how='left')
        
        # Target ROIs
        rois_of_interest = ['AIC', 'PIC', 'STGl', 'STGr', 'SMCl', 'SMCr', 'IFG', 'dACC']
        
        ch_to_roi = {}
        for ch in raw.ch_names:
            roi_str = str(cord_df.loc[cord_df['channel'] == ch, 'roi'].values)
            for tgt in rois_of_interest:
                if tgt.lower() in roi_str.lower():
                    ch_to_roi[ch] = tgt
                    break
                    
        eval_channels = list(ch_to_roi.keys())
        if not eval_channels: continue
        if 'AIC' not in ch_to_roi.values() and 'PIC' not in ch_to_roi.values(): continue
        
        raw.pick_channels(eval_channels)
        xdata = raw.get_data()
        
        # Compute exact xcorr matrix (Focus on inner 0.5s for clean visualization)
        xcorr, lag_times = compute_xcorr_matrix(xdata, raw.info['sfreq'], max_lag_s=0.5)
        n_trials = xcorr.shape[0]
        
        unique_rois = list(set(ch_to_roi.values()))
        
        for idx1, roi1 in enumerate(unique_rois):
            ch_idx1 = [i for i, ch in enumerate(raw.ch_names) if ch_to_roi.get(ch) == roi1]
            for idx2, roi2 in enumerate(unique_rois):
                if idx1 == idx2: continue
                ch_idx2 = [i for i, ch in enumerate(raw.ch_names) if ch_to_roi.get(ch) == roi2]
                if not ch_idx1 or not ch_idx2: continue
                
                # Extract and average the block over channels to get (trials, lags)
                block = xcorr[:, ch_idx1][:, :, ch_idx2]
                mean_wave = block.mean(axis=(1, 2)) # Shape: (trials, lags)
                
                # Add condition from event_id if available (using events array)
                # event is typically [time, 0, event_id]
                events = raw.events
                inv_events = {v: k for k, v in raw.event_id.items()}
                
                for t in range(n_trials):
                    cond_id = events[t, 2]
                    condition = inv_events.get(cond_id, str(cond_id))
                    
                    row = {
                        'subject': subject,
                        'phase': phase,
                        'task': task_name,
                        'condition': condition,
                        'trial': t,
                        'source_roi': roi1,
                        'target_roi': roi2,
                    }
                    for i, lag in enumerate(lag_times):
                        row[f"lag_{lag:.3f}"] = mean_wave[t, i]
                    all_rows.append(row)
                    
    if all_rows:
        df = pd.DataFrame(all_rows)
        save_dir = Path(f'results/{task_name}({reference})')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'roi_mean_xcorr_waveforms_{band}.parquet'
        df.to_parquet(save_path, engine='pyarrow')
        logger.info(f"Saved {len(df)} ROI-averaged xcorr traces to {save_path}")
    else:
        logger.warning(f"No valid ROI pairs found for {bids_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", type=str, required=True)
    parser.add_argument("--band", type=str, default="highgamma")
    parser.add_argument("--reference", type=str, default="bipolar")
    parser.add_argument("--recon_dir", type=str, default="/cwork/ns458/ECoG_Recon/")
    args = parser.parse_args()
    main(**vars(args))
