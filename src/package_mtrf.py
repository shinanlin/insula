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
import re
from tqdm import tqdm
from ieeg.viz.mri import force2frame


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def main(
    bids_root: str,
    ref: str,
    recon_dir: str,
    feature: str,
):
    """
    Package TRF results into CSV format with electrode positions and ROI labels.
    
    Parameters
    ----------
    bids_root : str
        Root directory of the BIDS dataset
    ref : str
        Reference type ('bipolar' or 'car')
    recon_dir: str,
    """

    # get task name from a artibutary bids file
    
    task_path = BIDSPath(
        root=bids_root,
        suffix='highgamma',
        extension='.h5',
        check=False,
    ).match()[0]
    task = task_path.task
    
    # Get all TRF result files
    trf_paths = BIDSPath(
        root=f'results/{task}({ref})',
        datatype='mtrf',
        suffix=feature,
        extension='.h5',
        check=False,
    ).match()
    
    for trf_path in tqdm(trf_paths, desc='Processing subjects'):
        
        subject = trf_path.subject
        phase = trf_path.processing
        description = trf_path.description
        
        # load the any epoch file of this subject to convert the montage
        epoch_path = BIDSPath(
            root=bids_root + 'derivatives/epoch(bipolar)',
            subject=subject,
            datatype='epoch(band)(sig)(effective)',
            suffix='highgamma',
            processing=phase,
            description=description,
            extension='.h5',
            check=False,
        ).match()
        try:
            epoch_path = epoch_path[0]
            epochs = mne.read_epochs(epoch_path, verbose='error')
            montage = epochs.get_montage()
        except IndexError:
            logger.warning(f'No epoch file found for subject {subject}')
            continue
        
        # Load parcellation info
        try:
            parc_path = epoch_path.copy().update(
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
        
        # Merge with parcellation info
        parc.rename(columns={'name': 'channel'}, inplace=True)
        
        # add channel (x, y, z)
        sub_id = re.sub(r'^D0+', 'D', subject)
        to_fsaverage = mne.read_talxfm(sub_id, recon_dir)
        trans = mne.transforms.Transform(fro='head', to='mri',
                                        trans=to_fsaverage['trans'])
        force2frame(montage, trans.from_str)  
        montage.apply_trans(trans) 
        pos_m = montage.get_positions()['ch_pos']
        cord_df = pd.DataFrame(pos_m).T
        cord_df.columns = ['x', 'y', 'z']
        cord_df[['x','y','z']] *= 1000
        cord_df = cord_df.reset_index().rename(columns={'index': 'channel'})
        
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
        
        # Create dataframe for this subject
        n_chan, n_feat, n_time = weight.shape
        significant_map = dict(zip(chn_names, fdr_mask.astype(bool)))
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
            d['significant'] = d['channel'].map(significant_map)
            dfs.append(d)

        df_long = pd.concat(dfs, ignore_index=True)
        
        parc_sub = parc[['channel', 'label','roi','hemi']]
        df_long = df_long.merge(parc_sub, on='channel', how='left')
        # Merge with parcellation info
        df_long = df_long.merge(cord_df[['channel', 'x', 'y', 'z']], on='channel', how='left')
        # Clean up ROI names (similar to package_HGA.py)
        df_long.loc[df_long['roi'] == 'PrG', 'roi'] = 'SMC'
        df_long.loc[df_long['roi'] == 'PoG', 'roi'] = 'SMC'
        df_long.loc[df_long['roi'] == 'Subcentral', 'roi'] = 'SMC'
        df_long['suffix'] = trf_path.suffix
        # Save each subject's dataframe separately
        save_path = trf_path.copy().update(
            extension='.csv',
        )
        save_path.mkdir(exist_ok=True)
        df_long.to_csv(save_path, index=False)
        
        logger.info(f"Saved TRF results for subject {subject} to {save_path}")
        logger.info(f"Rows: {len(df_long)}, Features: {df_long['feature'].nunique()}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package TRF results into CSV format")
    
    # parser.add_argument("--bids_root", type=str,
    #                     default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/",
    #                     help="Root directory of the BIDS dataset")
    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/",
                        help="Root directory of the BIDS dataset")
    parser.add_argument("--ref", type=str, default='bipolar',
                        choices=['bipolar','car'],
                        help="Reference type")
    parser.add_argument('--feature', type=str, default='em',
                        choices=['m', 'e', 'em'],
                        help='feature type')
    parser.add_argument('--recon_dir', type=str, default=r'/cwork/ns458/ECoG_Recon/',
                        help='path to the recon-all directory')
    args = parser.parse_args()
    main(**vars(args))
