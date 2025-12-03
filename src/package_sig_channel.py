import pandas as pd
import numpy as np
import argparse
import logging
from mne_bids import BIDSPath
import mne
import os

def main(
    bids_root: str,
    band: str,
    radius: int,
    ref: str,
):
    
    bids_path = BIDSPath(
        root=os.path.join(bids_root, f"derivatives/epoch({ref})"),
        suffix=band,
        datatype='epoch(band)(sig)(effective)',
        description='perception',
        extension=".h5",
        check=False,
    )
    
    for fpath in bids_path.match():
            
        perception_path = fpath.copy().update(description='perception').match()[0]
        production_path = fpath.copy().update(description='production').match()[0]

        perception_epochs = mne.read_epochs(perception_path)
        production_epochs = mne.read_epochs(production_path)
        
        unique_sig_channels = np.unique(np.concatenate([perception_epochs.ch_names, production_epochs.ch_names]))
        
        # load parcelation
        parc_path = perception_path.copy().update(
            root=str(perception_path.root).replace(f'epoch({ref})', 'parcellation'),
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
            missing = set(cols_need) - set(parc.columns)
            logging.warning(
                f"Skip {fpath.subject}: parc file {parc_path} "
                f"missing columns {missing}"
            )
            continue

        # per-electrode metadata
        electrodes = []
        categories = []
        rois = []
        hemis = []

        for ch in unique_sig_channels:
            is_in_perc = ch in perception_epochs.ch_names
            is_in_prod = ch in production_epochs.ch_names

            this_row = parc[parc['name'] == ch]
            if this_row.empty:
                logging.warning(f"Skip {ch}: not found in parc file {parc_path}")
                continue

            this_roi = this_row['roi'].values[0]
            this_hemi = this_row['hemi'].values[0]

            if is_in_perc and is_in_prod:
                cat = 'Both'
            elif is_in_perc:
                cat = 'Perception Only'
            elif is_in_prod:
                cat = 'Production Only'
            else:
                cat = 'None'

            electrodes.append(ch)
            categories.append(cat)
            rois.append(this_roi)
            hemis.append(this_hemi)

        df = pd.DataFrame({
            'electrode': electrodes,
            'category': categories,
            'roi': rois,
            'hemi': hemis,
            'band': band,
        })
        
        df.loc[df['roi'] == 'PrG', 'roi'] = 'SMC'
        df.loc[df['roi'] == 'PoG', 'roi'] = 'SMC'
        
        # save as csv
        save_path = BIDSPath(
            root=f'results/{fpath.task}',
            datatype="statistics",
            suffix='sig',
            subject=fpath.subject,
            description=None,
            extension='.csv',
            check=False
        )
        save_path.mkdir(exist_ok=True)
        
        df.to_csv(save_path.fpath, index=False)
    
    return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/", type=str)
    parser.add_argument("--band", type=str, default='highgamma',
                        help="Band to process")
    parser.add_argument("--radius", type=int, default=10,
                        help='radius of the sphere in mm')
    parser.add_argument("--ref", type=str, default='bipolar',
                        choices=['bipolar','car'],
                        help='reference channel')

    args = parser.parse_args()
    main(**vars(args))