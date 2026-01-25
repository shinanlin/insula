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
from ieeg.viz.mri import force2frame

def main(
    bids_root: str,
    band: str,
    ref: str,
    recon_dir: str,
):
    
    epoch_paths = BIDSPath(
        root=bids_root + f"derivatives/epoch({ref})",
        suffix=band,
        datatype='epoch(band)(zscore)',
        extension=".h5",
        check=False,
    )
    
    # 记录失败的subjects
    failed_subjects = []
    
    for epoch_path in tqdm(epoch_paths.match(), desc='Processing subjects'):
        
        try:
        # load parc file for this subject
            parc_path = epoch_path.copy().update(
                root=str(epoch_path.root).replace(f'epoch({ref})', 'parcellation'),
                datatype=ref,
                task=None,
                description=None,
                recording=None,
                processing='3mm',
                suffix='aparc2009s',
                extension='.csv',
            ).match()[0]
            parc = pd.read_csv(parc_path)
        except Exception as e:
            logging.warning(f"Parc file not found for subject {epoch_path.subject}: {str(e)}, skipping this subject")
            failed_subjects.append(epoch_path.subject)
            continue
        
        try:
            epochs = mne.read_epochs(epoch_path, preload=True)
            evoked = epochs.average(method=lambda x: np.nanmean(x, axis=0))
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

            # delete the 'ch_names' column
            df.drop(columns=['ch_type'], inplace=True)
            df['subject'] = epoch_path.subject
            df['description'] = epoch_path.description
            df['task'] = epoch_path.task
            df['phase'] = epoch_path.processing
            df['modality'] = epoch_path.recording if epoch_path.recording is not None else 'sound'
            
            
            # rename column name
            parc.rename(columns={'name': 'channel'}, inplace=True)
            parc_sub = parc[['channel', 'label','roi','hemi']]
            df = df.merge(parc_sub, on='channel', how='left')

            # remove White matter, Intersection and unknown electrodes (more flexible matching)
            df = df[~df['roi'].str.contains('white|intersection|unknown', case=False, na=False)]
            df = df.dropna(subset=['roi'])
            df.loc[df['roi'] == 'PrG', 'roi'] = 'SMC'
            df.loc[df['roi'] == 'PoG', 'roi'] = 'SMC'
            df.loc[df['roi'] == 'Subcentral', 'roi'] = 'SMC'
            # add channel (x, y, z)
            montage = epochs.get_montage()
            
            sub_id = re.sub(r'^D0+', 'D', epoch_path.subject)
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
            cord_df = cord_df[cord_df.channel.isin(df.channel)]
            
            # merge cord_df and df
            df = df.merge(cord_df, on='channel', how='left')
            
            save_path = BIDSPath(
                root=f'results/{epoch_path.task}({ref})',
                description=epoch_path.description,
                datatype='LBA',
                suffix='zscore',
                recording=epoch_path.recording,
                task=epoch_path.task,
                subject=epoch_path.subject,
                processing=epoch_path.processing,
                extension=".csv",
                check=False,
            )
            save_path.mkdir(exist_ok=True)
            df.to_csv(save_path, index=False)
            
        except Exception as e:
            logging.error(f"Failed to process subject {epoch_path.subject}: {str(e)}")
            failed_subjects.append(epoch_path.subject)
            continue
    
    # 输出处理结果总结
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total subjects processed: {len(epoch_paths.match())}")
    print(f"Failed: {len(failed_subjects)}")
    
    if failed_subjects:
        print(f"\nFailed subjects: {', '.join(failed_subjects)}")
    else:
        print("\nAll subjects processed successfully!")
    
    print("="*50)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/", type=str)
    parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_TIMIT/BIDS/", type=str)
    parser.add_argument("--band", type=str, default="beta", choices=['highgamma','gamma','beta','alpha','theta'],
                        help='which frequency band to use')
    parser.add_argument("--ref", type=str, default='bipolar',
                        choices=['bipolar','car'],
                        help='reference channel')
    parser.add_argument('--recon_dir', type=str, default=r'/cwork/ns458/ECoG_Recon/',
                        help='path to the recon-all directory')
    args = parser.parse_args()
    main(**vars(args))