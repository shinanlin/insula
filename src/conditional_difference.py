import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from IPython.display import Audio
import IPython.display as ipd
from scipy.io import wavfile
import tempfile
import os
import librosa
import pandas as pd
import seaborn as sns
import h5py
import mne
from scipy.stats import zscore
from mne_bids import BIDSPath, read_raw_bids
from ieeg.calc.stats import time_perm_cluster
from tqdm import tqdm
import argparse



def main(
    bids_root,
    band,
):
    
    
    pt = BIDSPath(
            root=os.path.join(bids_root, 'derivatives', 'epoch(bipolar)'),
            datatype='epoch(band)(zscore)',
            suffix=band,
            extension='.h5',
            check=False
        )
    
    pts = pt.match()
    subjects = list(set([p.subject for p in pts]))
    reference = 'bipolar'
    for subject in subjects:
        
        this_sub_pts = pt.copy().update(subject=subject).match()
        
        # load parc files
        parc_path = this_sub_pts[0].copy().update(
            root=str(this_sub_pts[0].root).replace(f'epoch(bipolar)', 'parcellation'),
            datatype='bipolar',
            task=None,
            description=None,
            recording=None,
            processing='3mm',
            suffix='aparc2009s',
            extension='.csv',
        ).match()[0]
        parc = pd.read_csv(parc_path)
        
        parc = parc[~parc['roi'].str.contains('white|intersection|unknown|WM', case=False, na=False)]
        parc = parc[~parc['roi'].isin([
            'OPC', 'sOccG', 'OccS', 'mOccG',
            'WM','LinG','PhG','Cb','iOccGs',
            'GRect','Amyg','Hipp','LinGs',
            'CG','OFCs','CGs',
            'Thal','Right-Pallidum','Left-Pallidum',
            'Calcarine','CollatAnt','VDC','InfLatV','LatV',
            'CC_Central','BrainStem','CollatPost','Caud'
        ])]
        this_sub_channel = parc.name.unique().tolist()
        
        this_sub_df = []
        for this_sub_pt in this_sub_pts:
            # load epoch
            dc_epo = mne.read_epochs(this_sub_pt, verbose='error')
            dc_epo.pick_channels(this_sub_channel)
            df = dc_epo.to_data_frame(long_format=True, verbose='error')
            df['condition'] = df['condition'].str.split('/').str[2:4].str.join('/')
            df['description'] = this_sub_pt.description
            df['phase'] = str(this_sub_pt.processing).lower()
            df['subject'] = this_sub_pt.subject
            this_sub_df.append(df)
            
        epos = pd.concat(this_sub_df, ignore_index=True)

        # do perm
        perms = []

        for phase in tqdm(epos.phase.unique()):
            phase = str(phase).lower()
            # 保留 trial，先按 channel 平均得到 (trial, time)
            dec = epos[(epos.description == 'Decision') & (epos.phase == phase)]
            rep = epos[(epos.description == 'Repeat') & (epos.phase == phase)]
            # pivot to (trial x time)
            decision_da = (dec
                .groupby(['condition','channel','time'])['value']
                .mean() 
                .to_xarray()
            )
            repeat_da = (rep
                .groupby(['condition','channel','time'])['value']
                .mean() 
                .to_xarray()
            )

            mask, p = time_perm_cluster(
                decision_da.values,
                repeat_da.values,
                p_thresh=0.05,
                n_perm=2000,
                n_jobs=-1,
                tails=2,
            )
            times = decision_da.time.values
            channels = repeat_da.channel.values

            df = pd.DataFrame({
                'channel': np.repeat(channels, len(times)),
                'time': np.tile(times, len(channels)),
                'mask': mask.ravel(),
                'p': p.ravel(),
                'phase': phase
            })
            perms.append(df)
            
        perms = pd.concat(perms, ignore_index=True)
        hit = perms['mask'].astype(bool) & (perms['p'] < 0.05)
        perms['significant'] = hit.groupby([perms['phase'], perms['channel']]).transform('any')

        agg = epos.groupby(['phase','description','channel','time'], as_index=False)['value'].mean()
        use_mask = (
            (agg['phase'].isin(['stimulus', 'delay']) & (agg['time'] > 0))
            | (agg['phase'].isin(['go', 'response']) & (agg['time'] < 0))
        )
        phase_mean = (
            agg.loc[use_mask]
            .groupby(['phase', 'description', 'channel'], as_index=False)['value']
            .mean()
        )

        phase_wide = phase_mean.pivot_table(
            index=['phase', 'channel'],
            columns='description',
            values='value'
        ).reset_index()
        phase_wide = phase_wide.rename(columns={
            'Decision': 'decision_mean',
            'Repeat': 'repeat_mean',
        })
        phase_wide['diff'] = phase_wide['decision_mean'] - phase_wide['repeat_mean']
        phase_wide['direction'] = np.where(phase_wide['diff'] > 0, 'Decision', 'Repeat')

        perms = perms.merge(phase_wide, on=['phase', 'channel'], how='left')
        perms.loc[~perms['significant'], 'direction'] = np.nan

        # save path
        save_path = BIDSPath(
            root=f'results/{this_sub_pt.task}({reference})',
            datatype='condition',
            suffix=band,
            task=this_sub_pt.task,
            subject=this_sub_pt.subject,
            extension=".csv",
            check=False,
        )
        save_path.mkdir(exist_ok=True)
        perms.to_csv(save_path, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/",
                        help="Root directory of the BIDS dataset")
    parser.add_argument("--band", type=str, default='highgamma',
                        help="highgamma or other band of neural signal")

    args = parser.parse_args()
    main(**vars(args))