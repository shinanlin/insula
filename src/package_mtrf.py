#!/usr/bin/env python3
"""
Package TRF results into CSV format with electrode positions and ROI labels.
"""
import rootutils
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import h5py
import numpy as np
import sys
from mne_bids import BIDSPath
import pandas as pd
import mne
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def main(
    bids_root: str,
    task: str,
    ref: str,
):
    """
    Package TRF results into CSV format with electrode positions and ROI labels.
    
    Parameters
    ----------
    bids_root : str
        Root directory of the BIDS dataset
    ref : str
        Reference type ('bipolar' or 'car')
    """
    
    # Get all TRF result files
    trf_paths = BIDSPath(
        root=f'results/{task}({ref})',
        datatype='mtrf',
        suffix='e',
        extension='.h5',
        check=False,
    ).match()
    
    for trf_path in tqdm(trf_paths, desc='Processing subjects'):
        
        subject = trf_path.subject
        
        # Load TRF results
        performance = h5py.File(trf_path, 'r')

        # pearsonr arrays are shape (n_folds, n_channel)
        fdr_mask = performance['fdr_mask'][:]
        coef = performance['pearsonr'][:].mean(axis=0)
        weight = performance['weights'][:].mean(axis=0)
        mask = performance['mask'][:]
        chn_names = np.array([chn.decode() for chn in performance['chn_names']])
        times = performance['times'][:]

        performance.close()
        
        # Load parcellation info
        try:
            parc_path = trf_path.copy().update(
                root=bids_root + 'derivatives/parcellation',
                datatype=ref,
                task=None,
                description=None,
                recording=None,
                processing='3mm',
                suffix='aparc2009s',
                extension='.csv',
            ).match()[0]
            parc = pd.read_csv(parc_path)
        except IndexError:
            logger.warning(f"No parcellation file found for subject {subject}")
            continue
        
        # Get feature type from task name
        feature_type = trf_path.task.split('(')[0] if '(' in trf_path.task else trf_path.task
        
        # Create dataframe for this subject
        n_chan, n_feat, n_time = weight.shape
        dfs = []
        for fidx in range(n_feat):
            d = (
                pd.DataFrame(weight[:, fidx, :], index=chn_names, columns=times)
                .stack()
                .rename_axis(['channel', 'time'])
                .reset_index(name='weight')
            )
            channel_coef_map = dict(zip(chn_names, coef))
            d['pearsonr'] = d['channel'].map(channel_coef_map)
            
            d['feature'] = f'feature_{fidx}'
            d['subject'] = subject
            d['task'] = trf_path.task
            d['description'] = trf_path.description
            d['processing'] = trf_path.processing
            d['significant'] = d['channel'].map(lambda ch: fdr_mask[chn_names.tolist().index(ch)])
            dfs.append(d)

        df_long = pd.concat(dfs, ignore_index=True)
        
        # Merge with parcellation info
        parc.rename(columns={'name': 'channel'}, inplace=True)
        df_long = df_long.merge(parc[['channel', 'roi', 'hemi', 'x', 'y', 'z']], on='channel', how='left')
        
        # Clean up ROI names (similar to package_HGA.py)
        df_long.loc[df_long['roi'] == 'PrG', 'roi'] = 'SMC'
        df_long.loc[df_long['roi'] == 'PoG', 'roi'] = 'SMC'
        df_long.loc[df_long['roi'] == 'Subcentral', 'roi'] = 'SMC'
        df_long[['x','y','z']] *= 1000
        # Save each subject's dataframe separately
        save_path = trf_path.copy().update(
            extension='.csv',
        )
        save_path.mkdir(exist_ok=True)
        df_long.to_csv(save_path, index=False)
        
        logger.info(f"Saved TRF results for subject {subject} to {save_path}")
        logger.info(f"Rows: {len(df_long)}, Features: {df_long['feature'].nunique()}")
        logger.info(f"Significant electrodes: {df_long['significant'].sum()}, Total electrodes: {len(chn_names)}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package TRF results into CSV format")
    
    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/",
                        help="Root directory of the BIDS dataset")
    parser.add_argument("--task", type=str,
                        default="PhonemeSequence",
                        help="Task name")
    parser.add_argument("--ref", type=str, default='bipolar',
                        choices=['bipolar','car'],
                        help="Reference type")
    
    args = parser.parse_args()
    main(**vars(args))
