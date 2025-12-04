# Compute ROI-level significance mask by loading pre-organized ROI epochs
# and running time_perm_cluster against baseline
# Data is organized as: derivatives/roi({ref})/sub-{ROI}{hemi}/epoch(band)(power)/

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import h5py
import numpy as np
import sys
import copy
from mne_bids import BIDSPath
import pandas as pd
import mne
import logging
from tqdm import tqdm
import os
from ieeg.calc.stats import time_perm_cluster


def main(
    bids_root: str,
    description: str,
    band: str,
    ref: str,
    p_threshold: float,
    n_perm: int,
):
    
    # ============ ROI list to focus on (without hemi suffix) ============
    ROI_LIST = ['INS', 'SMC', 'STG', 'STS', 'MTG']
    
    # ROI folder names: left = {roi}l, right = {roi}r
    # e.g., INSl, INSr -> combine to INS
    
    roi_root = os.path.join(bids_root, f"derivatives/roi({ref})")
    
    times = None
    task_name = None
    
    for roi in tqdm(ROI_LIST, desc='Processing ROIs'):
        
        results = []
        
        # ============ load left and right hemisphere epochs ============
        left_name = f"{roi}l"
        right_name = f"{roi}r"
        
        task_data_list = []
        baseline_data_list = []
        
        for hemi_name in [left_name, right_name]:
            
            task_path = BIDSPath(
                root=roi_root,
                subject=hemi_name,
                datatype='epoch(band)(power)',
                suffix=band,
                description=description,
                extension='.fif',
                check=False,
            )
            
            baseline_path = BIDSPath(
                root=roi_root,
                subject=hemi_name,
                datatype='epoch(band)(power)',
                suffix=band,
                description='baseline',
                extension='.fif',
                check=False,
            )
            
            try:
                task_match = task_path.match()[0]
                baseline_match = baseline_path.match()[0]
                
                task_epochs = mne.read_epochs(task_match, preload=True, verbose='error')
                baseline_epochs = mne.read_epochs(baseline_match, preload=True, verbose='error')
                
                if times is None:
                    times = task_epochs.times
                if task_name is None:
                    task_name = task_match.task
                
                # get data: (n_trials, n_channels, n_times)
                task_data_list.append(task_epochs.get_data())
                baseline_data_list.append(baseline_epochs.get_data())
                
            except (IndexError, FileNotFoundError) as e:
                logging.warning(f"Could not load {hemi_name} for {roi}: {e}")
                continue
        
        if len(task_data_list) == 0:
            logging.warning(f"No data found for ROI {roi}, skipping")
            continue
        
        # ============ concatenate left and right hemispheres ============
        # concat along channel axis: (n_trials, n_channels_L + n_channels_R, n_times)
        task_concat = np.concatenate(task_data_list, axis=1)
        baseline_concat = np.concatenate(baseline_data_list, axis=1)
        
        n_channels = task_concat.shape[1]
        
        baseline_concat = np.nanmean(baseline_concat, axis=1,keepdims=True)
        task_concat = np.nanmean(task_concat, axis=1,keepdims=True)
        
        # ============ time_perm_cluster ============
        mask, pvals = time_perm_cluster(
            task_concat, 
            baseline_concat,
            p_thresh=0.1,
            ignore_adjacency=1,
            n_perm=n_perm, 
            n_jobs=-1
        )
        
        # mask shape: (n_channels, n_times) -> average across channels for ROI-level
        # or take any channel since ignore_adjacency=1 treats them independently
        # Actually with ignore_adjacency=1, each channel gets its own mask
        # We want ROI-level, so we check if ANY channel is significant at each time
        roi_mask = mask.any(axis=0)  # (n_times,)
        roi_pvals = pvals.min(axis=0)  # take min p-value across channels
        
        # apply p_threshold
        sig_mask = roi_mask & (roi_pvals < p_threshold)
        
        # structure into DataFrame rows
        for t_idx, t in enumerate(times):
            results.append({
                'time': t,
                'mask': sig_mask[t_idx],
                'pval': roi_pvals[t_idx],
                'roi': roi,
                'description': description,
                'task': task_name,
            })
        
        # ============ save results ============
        df = pd.DataFrame(results)
        
        save_path = BIDSPath(
            root=f'results/{task_name}(roi)({ref})',
            description=description,
            datatype='HGA',
            suffix='roimask',
            task=task_name,
            subject=roi,
            extension=".csv",
            check=False,
        )
        save_path.mkdir(exist_ok=True)
        df.to_csv(save_path, index=False)
    
    print(f"Saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/", type=str)
    parser.add_argument("--description", type=str, default="production")
    parser.add_argument("--band", type=str, default="highgamma", 
                        choices=['highgamma','gamma','beta','alpha','theta'],
                        help='which frequency band to use')
    parser.add_argument("--ref", type=str, default='bipolar',
                        choices=['bipolar','car'],
                        help='reference channel')
    parser.add_argument("--p_threshold", type=float, default=0.05,
                        help='p-value threshold for significance')
    parser.add_argument("--n_perm", type=int, default=5000,
                        help='number of permutations')
    args = parser.parse_args()
    main(**vars(args))
