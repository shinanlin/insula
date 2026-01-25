#!/usr/bin/env python3
"""Decoding script for phoneme classification using PCA-SVM."""

import rootutils
# add the root path to the python path for importing
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import h5py
import numpy as np
from mne_bids import BIDSPath
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score, accuracy_score
)
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from decoder import decode_permutation_scores
from ieeg.calc.oversample import MinimumNaNSplit
from mne.decoding import Vectorizer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import sys
import logging
import os
import pandas as pd
import mne
import pickle
from ieeg.arrays.label import LabeledArray
from tqdm import tqdm

# Simple logging: everything INFO and above to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


EXCLUDE_CHANNELS = [
    'D0040_L1IF2-3',
    'D0040_L1IF3-4',
    'D0079_LFAI1-2',
    'D0079_LFAI2-3',
    'D0079_LFMI1-2',
    'D0079_LPI9-10',
    'D0103_LAI7-8',
    'D0103_LAI3-4',
    'D0103_LAI5-6',
    'D0103_LAI6-7',
]


def split_insula_ap(parcs_df, y_threshold=0):
    """Add insula_region column with AIC/PIC labels based on parcellation rules."""
    coords_classified = parcs_df.copy()
    coords_classified['insula_region'] = None

    aic_conditions = (
        (coords_classified['label'].str.contains('G_insular_short', na=False)) |
        (coords_classified['label'].str.contains('S_circular_insula_ant', na=False)) |
        ((coords_classified['label'].str.contains('S_circular_insula_sup', na=False)) &
         (coords_classified['y'] > y_threshold)) |
        ((coords_classified['label'].str.contains('S_circular_insula_inf', na=False)) &
         (coords_classified['y'] > y_threshold))
    )

    pic_conditions = (
        (coords_classified['label'].str.contains('G_Ins_lg_and_S_cent_ins', na=False)) |
        ((coords_classified['label'].str.contains('S_circular_insula_sup', na=False)) &
         (coords_classified['y'] <= y_threshold)) |
        ((coords_classified['label'].str.contains('S_circular_insula_inf', na=False)) &
         (coords_classified['y'] <= y_threshold))
    )

    coords_classified.loc[aic_conditions, 'insula_region'] = 'AIC'
    coords_classified.loc[pic_conditions, 'insula_region'] = 'PIC'

    mask = coords_classified['insula_region'].notna()
    coords_classified.loc[mask, 'roi'] = coords_classified.loc[mask, 'insula_region']
    coords_classified = coords_classified.drop(columns=['insula_region'])
    return coords_classified


def cluster_correction(scores, baseline, p_thresh=0.05, tails=1):
    from scipy.stats import permutation_test
    from ieeg.calc.stats import time_cluster, proportion, tail_compare

    pvals = np.zeros(scores.shape)
    n_perm = baseline.shape[0]

    diff = baseline - scores[None, :]
    p_act = (np.sum(diff >= 0, axis=0) + 1) / (diff.shape[0] + 1)
    p_perm = proportion(diff, tail=tails, axis=0)
    b_act = tail_compare(1. - p_act, 1. - p_thresh, tails)
    b_perm = tail_compare(p_perm, 1. - p_thresh, tails)
    mask = time_cluster(b_act, b_perm, 1 - p_thresh, tails)

    return mask, p_act


def load_data(
    bids_root,
    roi,
    hemi,
    ref,
    phase,
    band,
):
    
    

    # get the parcellation file
    parc_paths = BIDSPath(
                root=os.path.join(bids_root, 'derivatives', 'parcellation'),
                datatype=ref,
                task=None,
                description=None,
                recording=None,
                processing='3mm',
                suffix='aparc2009s',
                extension='.csv',
                check=False
            ).match()
    
    parcs = pd.concat([pd.read_csv(path) for path in parc_paths], ignore_index=True)
    parcs = parcs[~parcs['name'].isin(EXCLUDE_CHANNELS)]
    parcs = split_insula_ap(parcs)

    parcs.loc[parcs.roi=='PrG', 'roi'] = 'SMC'
    parcs.loc[parcs.roi=='PrG', 'roi'] = 'SMC'
    parcs.loc[parcs.roi=='Subcentral', 'roi'] = 'SMC'

    if hemi == 'B':
        this_parcs = parcs[parcs['roi'] == roi]
    else:
        this_parcs = parcs[(parcs['roi'] == roi) & (parcs['hemi'] == hemi)]

    # load nested dict to form LabeledArray
    Xs = []
    
    for this_subject in tqdm(this_parcs.subject.unique()):
        logger.info(f"Processing subject: {this_subject}")
        pick_chns = this_parcs[this_parcs['subject'] == this_subject]['name'].unique().tolist()
        # load BIDS path
        epo_paths = BIDSPath(
            root=os.path.join(bids_root, 'derivatives', f'epoch({ref})'),
            subject=this_subject,
            datatype='epoch(band)(labelarray)',
            processing=phase,
            suffix=band,
            extension='.pkl',
            check=False
        ).match()
        
        this_sub_X = []
        
        for epo_pt in epo_paths:
            # load epoch nested epoch from pkl
            with open(epo_pt, 'rb') as f:
                epoch = pickle.load(f)
            # index: (subject, event_type, trial, channel, time)
            # take channel first
            la = epoch[:, pick_chns, :]
            y = np.tile(epo_pt.description, la.shape[0])
            # relabel trial axis with description (y)
            labels = list(la.labels)
            labels[0] = tuple(y)
            la = LabeledArray(np.asarray(la), labels=tuple(labels))
            this_sub_X.append(la)
            
        # LabeledArray.concatenate is an instance method; reduce pairwise
        this_sub_X = LabeledArray.concatenate(*this_sub_X, axis=0, mismatch='raise')
        Xs.append(this_sub_X)
    
    # combine all subjects
    logger.info(f"Making LabeledArray from {len(Xs)} subjects")
    xs = Xs[0]
    for x in Xs[1:]:
        xs = xs.concatenate(x, axis=1, mismatch='expand')  
    Xs = xs

    # map trial labels to numeric labels
    trial_labels = list(Xs.labels[0])
    base_labels = [
        lab.rsplit('-', 1)[0] if str(lab).rsplit('-', 1)[-1].isdigit() else str(lab)
        for lab in trial_labels
    ]
    unique_labels = sorted(set(base_labels))
    label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
    y_numeric = np.array([label_to_int[lab] for lab in base_labels])

    return Xs, y_numeric, label_to_int


def main(
    bids_root,
    roi,
    hemi,
    ref,
    phase,
    band,
    variance,
    window,
    step,
    n_perm,
    n_folds,
    n_jobs,
    tmin=-0.5,
    tmax=1.5,
):

    X, y, label_to_int = load_data(
        bids_root=bids_root,
        roi=roi,
        hemi=hemi,
        ref=ref,
        phase=phase,
        band=band,
    )

    cv = MinimumNaNSplit(n_splits=n_folds, n_repeats=1)
    logger.info('Making pipeline with variance %f', variance)

    decoder = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        PCA(n_components=variance, random_state=42),
        SVC(kernel='linear', random_state=42)
    )

    fs = 128
    time_points = np.arange(tmin + window, tmax + step, step)
    window_samples = int(window * fs)
    step_samples = int(step * fs)

    accuracies = np.zeros((len(time_points), n_folds))
    baseline_accuracies = np.zeros((len(time_points), n_folds, n_perm))

    X_np = np.asarray(X)

    for t_idx, time_end in enumerate(time_points):
        end_sample = int((time_end - tmin) * fs) + 1
        start_sample = end_sample - window_samples

        if start_sample < 0 or end_sample > X_np.shape[-1]:
            logger.warning(
                "Window out of bounds for time %.3fs, skipping",
                time_end,
            )
            continue

        X_segment = X_np[..., start_sample:end_sample]
        logger.info(
            "Processing time window: %.3fs, samples %d:%d",
            time_end,
            start_sample,
            end_sample,
        )

        score, permutation_scores, _ = decode_permutation_scores(
            X_segment,
            y,
            cv,
            decoder,
            n_jobs=n_jobs,
            n_permutations=n_perm,
            random_state=42,
        )

        accuracies[t_idx] = score
        baseline_accuracies[t_idx] = permutation_scores

    mask, p_values = cluster_correction(
        accuracies.mean(axis=-1),
        baseline_accuracies.mean(axis=1).T,
    )


    temp_pt = BIDSPath(
        root=os.path.join(bids_root, 'derivatives', f'epoch({ref})'),
        suffix=band,
        extension='.h5',
        check=False
    ).match()
    task = temp_pt[0].task
    save_path = BIDSPath(
        root=os.path.join('results', f'{task}(roi)({ref})'),
        datatype='(decode)(resolved)condition',
        subject=roi+hemi.lower(),
        suffix=band,
        processing=phase,
        extension='.h5',
        check=False
    )
    save_path.mkdir(exist_ok=True)

    logger.info('Saving results to %s', save_path)
    with h5py.File(save_path, "w") as f:
        f.create_dataset(name='accuracy', data=accuracies)
        f.create_dataset(name='baseline', data=baseline_accuracies)
        f.create_dataset(name='time', data=time_points)
        f.create_dataset(name='mask', data=mask)
        f.create_dataset(name='p_values', data=p_values)

        f.attrs["fs"] = fs
        f.attrs["tmin"] = tmin
        f.attrs["tmax"] = tmax

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/",
                        help="Root directory of the BIDS dataset")
    parser.add_argument("--roi", type=str, default='HG',
                        help="ROI to process")
    parser.add_argument("--hemi", type=str, default='L',
                        choices=['L', 'R', 'B'],
                        help="Hemisphere to process")
    parser.add_argument("--ref", type=str, default='bipolar',
                        choices=['car', 'bipolar'],
                        help="Reference type")
    parser.add_argument("--phase", type=str, default='Stimulus',
                        choices=['Stimulus', 'Go', 'Delay','Response'],
                        help="phase of the experiment")
    parser.add_argument("--band", type=str, default='highgamma',
                        help="highgamma or other band of neural signal")
    parser.add_argument("--variance", type=float, default=0.85,
                        help="number of variance")
    parser.add_argument("--window", type=float, default=0.6,
                        help="window length in seconds")
    parser.add_argument("--step", type=float, default=0.5,
                        help="step size in seconds")
    parser.add_argument("--n_perm", type=int, default=2,
                        help="number of permutations")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="number of folds")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="number of jobs")

    args = parser.parse_args()
    main(**vars(args))