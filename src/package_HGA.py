# take HGA signal and save it to pandas

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


def main(
    bids_root: str,
    description: str,
    band: str,
    recon_dir: str,
    ref: str,
):
    
    epoch_paths = BIDSPath(
        root=bids_root + f"derivatives/epoch({ref})",
        suffix=band,
        datatype='epoch(band)(sig)(effective)',
        description=description,
        extension=".h5",
        check=False,
    )
    
    for epoch_path in tqdm(epoch_paths.match(), desc='Processing subjects'):
        
        epochs = mne.read_epochs(epoch_path, preload=True)
        evoked = epochs.average()
        # save epochs to pandas
        df = evoked.to_data_frame(
            long_format=True,
            scalings={'seeg':1},
        )
        
        # ============ 加载 stats mask ============
        stats_path = epoch_path.copy().update(
            root=str(epoch_path.root).replace(f'epoch({ref})', 'statistics'),
            datatype=ref,
            extension='.h5',
        )
        try:
            with h5py.File(stats_path, 'r') as stats:
                mask_data = stats['mask'][:]
                ch_names_stats = [chn.decode('utf-8') for chn in stats['ch_names'][:]]
            
            # 转成 long format
            mask_df = pd.DataFrame(
                index=ch_names_stats,
                columns=epochs.times,
                data=mask_data,
            )
            mask_long = (
                mask_df.reset_index()
                .melt(id_vars='index', var_name='time', value_name='mask')
                .rename(columns={'index': 'channel'})
            )
            # take only the channel in the HGA
            mask_long = mask_long[mask_long['channel'].isin(df['channel'])]
            # merge mask 到 HGA dataframe
            df = df.merge(mask_long, on=['channel', 'time'], how='left')
            df['mask'] = df['mask'].fillna(False).astype(bool)
        except FileNotFoundError:
            logging.warning(f"Stats file not found: {stats_path}, setting mask to False")
            df['mask'] = False
        # attach ROI column from epochs.info['description'] JSON if present
        roi_map = {}
        desc = epochs.info.get('description')
        if isinstance(desc, str) and len(desc):
            try:
                payload = json.loads(desc)
                if isinstance(payload, dict):
                    roi_map = payload.get('roi_map', {}) or {}
            except Exception:
                roi_map = {}
        if roi_map:
            df['roi'] = df['channel'].map(lambda ch: roi_map.get(ch, 'Unknown'))
        # delete the 'ch_names' column
        tmp = df['roi'].str.extract(r'^(.*)\((L|R)\)$')
        df['hemi'] = tmp[1].fillna('')
        df['roi'] = tmp[0].fillna(df['roi']).str.strip()
        df.drop(columns=['ch_type'], inplace=True)
        df['subject'] = epoch_path.subject
        df['description'] = epoch_path.description
        df['task'] = epoch_path.task

        
        df.loc[df['roi'] == 'PrG', 'roi'] = 'SMC'
        df.loc[df['roi'] == 'PoG', 'roi'] = 'SMC'
        df.loc[df['roi'] == 'Subcentral', 'roi'] = 'SMC'
        
        save_path = BIDSPath(
            root=f'results/{epoch_path.task}({ref})',
            description=description,
            datatype='HGA',
            suffix='time',
            task=epoch_path.task,
            subject=epoch_path.subject,
            extension=".csv",
            check=False,
        )
        save_path.mkdir(exist_ok=True)
        df.to_csv(save_path, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/", type=str)
    parser.add_argument("--description", type=str, default="production")
    parser.add_argument("--band", type=str, default="highgamma", choices=['highgamma','gamma','beta','alpha','theta'],
                        help='which frequency band to use')
    parser.add_argument("--recon_dir", type=str, default=r'/cwork/ns458/ECoG_Recon/',
                        help='path to the recon-all directory')
    parser.add_argument("--ref", type=str, default='bipolar',
                        choices=['bipolar','car'],
                        help='reference channel')
    args = parser.parse_args()
    main(**vars(args))