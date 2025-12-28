# project coord to average space

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import h5py
import numpy as np
import sys
import copy
from mne_bids import BIDSPath
import pandas as pd
import torch
import mne
import logging
from tqdm import tqdm
from ieeg.viz.mri import gen_labels
from ieeg.io import get_elec_volume_labels
import re
import json
from ieeg.viz.mri import force2frame
import os

def main(
    bids_root: str,
    band: str,
    recon_dir: str,
    radius: float,
    ref: str,
):
    
    epoch_paths = BIDSPath(
        root=os.path.join(bids_root,f"derivatives/epoch({ref})"),
        suffix=band,
        datatype='epoch(band)(power)',
        extension=".fif",
        check=False,
    )
    
    # what this need is [subject, electrode name, significant or not, ]
    
    for epoch_path in tqdm(epoch_paths.match(), desc='Processing subjects'):
        
        if epoch_path.description == 'baseline' or epoch_path.processing == 'baseline':
            continue
        
        epochs = mne.read_epochs(epoch_path, preload=True)
        
        logging.info(f'Processing {epoch_path.subject}')
        # get the ROI label
        try:
            montage = epochs.get_montage()
            sub_id = re.sub(r'^D0+', 'D', epoch_path.subject)
            to_fsaverage = mne.read_talxfm(sub_id, recon_dir)
            trans = mne.transforms.Transform(fro='head', to='mri',
                                            trans=to_fsaverage['trans'])
            force2frame(montage, trans.from_str)  
            montage.apply_trans(trans) 
            pos_m = montage.get_positions()['ch_pos']
        except FileNotFoundError:
            logging.warning(f"Talxfm file not found for {epoch_path.subject}, skipping")
            continue
        
        df = pd.DataFrame(pos_m).T
        df.columns = ['x', 'y', 'z']
        df[['x','y','z']] *= 1000
        df = df.reset_index().rename(columns={'index': 'channel'})
        
        # perception: 0-1s, production: -0.5-0.5s
        # tmin = 0 if description == 'perception' else -0.5
        # tmax = 1 if description == 'perception' else 0.5
        
        tmin = epochs.tmin+0.5
        tmax = epochs.tmax-0.5
        
        df['HGA'] = epochs.crop(tmin, tmax).get_data().mean(axis=(0,-1))
        
        try:
            sig_path = epoch_path.copy().update(
                datatype='epoch(band)(sig)',
                extension='.fif',
            ).match()[0]
            sig_epochs = mne.read_epochs(sig_path, preload=True)
            sig_mask = np.isin(montage.ch_names, sig_epochs.ch_names)
        except IndexError:
            sig_mask = np.zeros(len(montage.ch_names), dtype=bool)
        
        # read parcellation result
        parc_path = epoch_path.copy().update(
            root=str(epoch_path.root).replace(f'epoch({ref})', 'parcellation'),
            datatype=ref,
            task=None,
            description=None,
            processing=f'{int(radius)}mm',
            suffix='aparc2009s',
            extension='.csv',
        ).match()[0]
        parc = pd.read_csv(parc_path)
        
        cols_need = ['name', 'label', 'roi', 'hemi']
        if not all(c in parc.columns for c in cols_need):
            logging.warning(
                f"Skip {epoch_path.subject}: parc file {parc_path} "
                f"missing columns {set(cols_need) - set(parc.columns)}"
            )
            continue
        
        df['significant'] = sig_mask
        df['subject'] = epoch_path.subject
        df['task'] = epoch_path.task
        df['band'] = band
        df['description'] = epoch_path.description
        df['phase'] = epoch_path.processing
        
        df_merged = df.merge(
            parc[['name', 'label', 'roi', 'hemi']],
            left_on='channel',
            right_on='name',
            how='left'
        )
        
        # remove name column
        df_merged = df_merged.drop(columns=['name'])

        # combine PrG and PoG to SM
        df_merged.loc[df_merged['roi'] == 'PrG', 'roi'] = 'SMC'
        df_merged.loc[df_merged['roi'] == 'PoG', 'roi'] = 'SMC'
        df_merged.loc[df_merged['roi'] == 'Subcentral', 'roi'] = 'SMC'
        
        save_path = BIDSPath(
            root=f'results/{epoch_path.task}({ref})',
            description=epoch_path.description,
            datatype='HGA',
            suffix='coord',
            task=epoch_path.task,
            subject=epoch_path.subject,
            processing=epoch_path.processing,
            extension=".csv",
            check=False,
        )
        save_path.mkdir(exist_ok=True)
        df_merged.to_csv(save_path, index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/", type=str)
    parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_TIMIT/BIDS/", type=str)
    parser.add_argument("--band", type=str, default="highgamma", choices=['highgamma','gamma','beta','alpha','theta'],
                        help='which frequency band to use')
    parser.add_argument("--recon_dir", type=str, default=r'/cwork/ns458/ECoG_Recon/',
                        help='path to the recon-all directory')
    parser.add_argument("--radius", type=int, default=3,
                        help='radius of the sphere in mm')
    parser.add_argument("--ref", type=str, default='bipolar',
                        choices=['bipolar','car'],
                        help='reference channel')

    args = parser.parse_args()
    main(**vars(args))