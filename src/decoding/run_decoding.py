#!/usr/bin/env python3
"""Windowed ROI decoding with PCA-LinearSVC and permutation testing."""

import rootutils
# add the root path to the python path for importing
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time as _time

import h5py
import numpy as np
from ieeg.calc.oversample import MinimumNaNSplit
from mne.decoding import Vectorizer
from mne_bids import BIDSPath
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.decoding.decoder import (
    decode_cv_scores,
    decode_permutation_scores,
    get_cv_predict,
)
from src.paths import decoding_task_dir, decoding_input_dir
from src.decoding.repeated_cv import (
    METHOD as REPEATED_METHOD, SCHEMA_VERSION, SE_DESCRIPTION,
    repeated_cv_permutation,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42

# Fixed decoding windows per phase (seconds, relative to phase onset).
PHASE_WINDOWS = {
    'Stimulus': (0.0, 0.5),
    'Delay': (0.0, 0.7),
    'Go': (0.0, 0.5),
    # Locked after PhonemeSequence Response sweep (SHI-12): post-onset [0, 1].
    'Response': (0.0, 1.0),
}

# Feature-specific overrides (LexicalDelay lexicality Response sweep):
# peri-onset [-0.5, 0.5] recovers Sensory/Sustain Repeat significance that
# the PS phoneme/articulator lock [0, 1] removes.
FEATURE_PHASE_WINDOWS = {
    'lexicality': {
        'Response': (-0.5, 0.5),
    },
}


def load_roi_data(
    bids_root,
    ref,
    roi,
    description,
    phase,
    band,
    datatype,
    tmin,
    tmax,
    recording=None,
):
    """Load and preprocess neural data from a specific brain region (ROI).

    Loads neural time series data from HDF5 files following BIDS structure,
    then crops the temporal window to the requested interval.

    Parameters
    ----------
    bids_root : str or Path
        Root directory of BIDS dataset containing neural data
    ref : str
        Reference scheme (e.g., 'car', 'bipolar')
    roi : str
        Region of interest identifier (e.g., 'PrGl', 'STGl')
    description : str
        Task description (e.g., 'production', 'perception')
    band : str
        Frequency band (e.g., 'highgamma', 'beta')
    datatype : str
        Data type (e.g., 'phoneme', 'word')
    tmin : float
        Start time of the temporal window in seconds
    tmax : float
        End time of the temporal window in seconds

    Returns
    -------
    Xs : list of ndarray
        Neural time series per matched file, cropped to the window
    ys : list of ndarray
        Class labels per file
    paths : list
        Matched BIDS paths

    Raises
    ------
    FileNotFoundError
        If no matching files found for the specified ROI
    """
    root = BIDSPath(
        root=str(decoding_input_dir(bids_root, ref)),
        datatype=datatype,
        description=description,
        suffix=band,
        processing=phase,
        extension='.h5',
        check=False
    )
    roi_path = root.copy().update(subject=roi)
    if recording is not None:
        roi_path.update(recording=str(recording), check=False)
    roi_files = roi_path.match()

    if not roi_files:
        raise FileNotFoundError(f"No files found for ROI {roi}")

    Xs, ys, paths = [], [], []

    for roi_file in roi_files:
        data = h5py.File(roi_file, 'r')
        X = data['X'][:]
        y = data['y'][:]

        t_start = data.attrs['tmin']
        fs = data.attrs['fs']

        start_idx = int(fs * (tmin - t_start))
        end_idx = int(fs * (tmax - t_start))
        X = X[:, :, start_idx:end_idx]

        data.close()
        Xs.append(X)
        ys.append(y)
        paths.append(roi_file)

    return Xs, ys, paths


def first_repeat_inference(X, y, decoder, n_folds, n_repeats, n_perm, n_jobs):
    """Historical first-repeat inference, retained for explicitly legacy callers."""
    scores = np.empty((n_repeats, n_folds))
    first_predictions = None
    for repeat in range(n_repeats):
        seed = RANDOM_SEED + repeat
        cv = MinimumNaNSplit(n_splits=n_folds, n_repeats=1, random_state=seed)
        if repeat == 0:
            observed, permutations, p = decode_permutation_scores(
                X, y, cv, decoder, n_jobs=n_jobs, n_permutations=n_perm,
                scoring='balanced_accuracy', random_state=seed)
            scores[repeat] = observed
            first_predictions = get_cv_predict(X, y, cv, decoder,
                n_jobs=n_jobs, random_state=seed)
        else:
            scores[repeat] = decode_cv_scores(X, y, cv, decoder,
                n_jobs=n_jobs, scoring='balanced_accuracy', random_state=seed)
    classes = np.unique(y)
    return dict(accuracy=scores[0], accuracy_repeats=scores,
        accuracy_stable=float(scores.mean()), perm_scores=permutations, p_value=p,
        confusion=confusion_matrix(y, first_predictions, labels=classes),
        confusion_norm=confusion_matrix(y, first_predictions, labels=classes, normalize='true'),
        classes=classes, first_repeat_predictions=first_predictions)


def save_window_result(destination, result, *, metadata, input_metadata, inference):
    """Write one analysis schema atomically; plots consume it without overrides."""
    destination = Path(destination)
    temporary = destination.with_suffix('.pending.h5')
    with h5py.File(temporary, 'w') as f:
        for key in ['accuracy', 'accuracy_repeats', 'accuracy_stable', 'perm_scores',
                    'p_value', 'confusion', 'confusion_norm', 'classes', 'first_repeat_predictions']:
            f.create_dataset(key, data=result[key])
        for key, value in metadata.items():
            f.attrs[key] = value
        text_dtype = h5py.string_dtype('utf-8')
        for key in ['channel', 'input_channel', 'excluded_channel', 'trial', 'condition']:
            f.create_dataset(key, data=np.asarray(input_metadata[key], dtype=text_dtype))
        f.attrs['inference_method'] = inference
        f.attrs['n_trials'] = len(input_metadata['trial'])
        f.attrs['n_electrodes'] = len(input_metadata['channel'])
        f.attrs['n_subjects'] = len({c.split('_')[0] for c in input_metadata['channel']})
        if inference == REPEATED_METHOD:
            f.attrs['result_schema_version'] = SCHEMA_VERSION
            f.attrs['accuracy_unit'] = 'CV-repeat mean of fold balanced accuracies'
            f.attrs['p_value_statistic'] = 'mean across all CV repeats and folds'
            f.attrs['errorbar_definition'] = SE_DESCRIPTION
            f.attrs['permutation_unit'] = 'whole stimulus blocks within equal retained-repeat-count strata'
            f.attrs['permutation_labels'] = 'one global relabeling applied to both train and test in every repeat/fold'
            f.attrs['exclusion_stage'] = 'after reference-input trial QC, CV and imputation; before training-fold transform'
            f.attrs['confusion_definition'] = 'first-repeat out-of-fold predictions'
            for key in ['accuracy_se', 'permutation_means', 'observed_iterations',
                        'permutation_iterations', 'pca_component_counts']:
                f.create_dataset(key, data=result[key])
            f.create_dataset('accuracy_per_repeat', data=result['accuracy'])
            f.attrs['null_exceedances'] = result['null_exceedances']
            f.attrs['null_test_folds_missing_any_class'] = result['null_test_folds_missing_any_class']
            f.attrs['observed_reference_check'] = json.dumps(result['reference_check'])
            for key, value in result['permutation_plan'].items():
                if key == 'group_names':
                    value = np.asarray(value, dtype=text_dtype)
                f.create_dataset('permutation_plan/'+key, data=value)
            for item in result['folds']:
                group = f.create_group(f'cv/repeat_{item.repeat:02d}/fold_{item.fold}')
                group.create_dataset('train', data=item.train)
                group.create_dataset('test', data=item.test)
                group.attrs['full_train_digest'] = item.full_train_digest
                group.attrs['full_test_digest'] = item.full_test_digest
        else:
            f.attrs['result_schema_version'] = 1
            f.attrs['accuracy_unit'] = 'fold in first CV repeat'
            f.attrs['p_value_statistic'] = 'first-repeat five-fold mean'
    temporary.replace(destination)


def main(
    bids_root,
    subject,
    ref,
    description,
    phase,
    band,
    datatype,
    variance,
    n_perm,
    n_folds,
    n_repeats,
    n_jobs,
    C=1.0,
    tmin=None,
    tmax=None,
    results_root=None,
    inference='first_repeat',
    exclude_channel=None,
    observed_reference=None,
    recording=None,
):
    if tmin is None or tmax is None:
        try:
            default_tmin, default_tmax = FEATURE_PHASE_WINDOWS.get(datatype, {}).get(
                phase, PHASE_WINDOWS[phase]
            )
        except KeyError as exc:
            raise ValueError(f"Unknown phase: {phase}") from exc
        tmin = default_tmin if tmin is None else tmin
        tmax = default_tmax if tmax is None else tmax
    if float(tmin) >= float(tmax):
        raise ValueError(f"Require tmin < tmax; got [{tmin}, {tmax}]")

    logger.info(
        'Phase window: %s [%.2f, %.2f] s (C=%g, variance=%g, datatype=%s)',
        phase, tmin, tmax, C, variance, datatype,
    )

    Xs, ys, paths = load_roi_data(
        bids_root,
        ref,
        subject,
        description,
        phase,
        band,
        datatype,
        tmin,
        tmax,
        recording=recording,
    )

    n_files = len(Xs)
    logger.info(f"Loaded {n_files} files to process")

    for i in range(n_files):
        X, y, path = Xs[i], ys[i], paths[i]
        _t0 = _time.time()
        logger.info(f"Processing file {i}/{n_files - 1}: {path}")

        logger.info('Making pipeline with variance %f C=%g', variance, C)
        decoder = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=variance, random_state=42),
            LinearSVC(C=float(C), random_state=42, max_iter=5000),
        )

        decode_strings = lambda values: [v.decode() if isinstance(v, bytes) else str(v) for v in values]
        with h5py.File(path.fpath, 'r') as input_file:
            channels = decode_strings(input_file['channel'][:])
            trials = decode_strings(input_file['trial'][:])
            conditions = decode_strings(input_file['condition'][:])
            assignment_sha = str(input_file.attrs.get('assignment_sha256', ''))
            fs = float(input_file.attrs['fs'])
        exclusions = list(exclude_channel or [])
        unknown = set(exclusions)-set(channels)
        if unknown:
            raise ValueError(f'Excluded channels are absent from input: {sorted(unknown)}')
        keep = [i for i, channel in enumerate(channels) if channel not in exclusions]
        input_metadata = dict(channel=[channels[i] for i in keep], input_channel=channels,
            excluded_channel=exclusions, trial=trials, condition=conditions)
        if inference == REPEATED_METHOD:
            groups = np.asarray([trial.rsplit('_', 1)[0] for trial in trials])
            result = repeated_cv_permutation(X, y, groups, decoder[:-1], decoder[-1],
                keep_channels=keep, n_splits=n_folds, n_repeats=n_repeats,
                n_permutations=n_perm, random_state=RANDOM_SEED, n_jobs=n_jobs,
                observed_reference=observed_reference)
        elif inference == 'first_repeat':
            if exclusions:
                raise ValueError('Channel exclusions require explicit repeated-CV inference')
            result = first_repeat_inference(X, y, decoder, n_folds, n_repeats, n_perm, n_jobs)
        else:
            raise ValueError(f'Unknown inference method: {inference}')

        # Sweep runs nest hyperparameters in the results root. BIDSPath
        # datatype cannot contain ``-`` or ``_``.
        if results_root is not None:
            tmin_tag = f'{float(tmin):g}'.replace('-', 'm').replace('.', 'p')
            tmax_tag = f'{float(tmax):g}'.replace('-', 'm').replace('.', 'p')
            var_tag = f'{float(variance):g}'.replace('.', 'p')
            c_tag = f'{float(C):g}'.replace('.', 'p')
            out_root = os.path.join(
                results_root,
                f'var{var_tag}C{c_tag}t{tmin_tag}to{tmax_tag}',
                str(path.task),
            )
        else:
            out_root = str(decoding_task_dir(str(path.task)))
        datatype_out = '(decode)' + str(datatype)

        save_path = BIDSPath(
            root=out_root,
            datatype=datatype_out,
            subject=subject,
            suffix=band,
            processing=path.processing,
            description=path.description,
            recording=path.recording,
            extension='.h5',
            check=False
        )
        save_path.mkdir(exist_ok=True)

        logger.info('Saving results to %s', save_path)
        save_window_result(save_path.fpath, result, metadata=dict(fs=fs,
            tmin=float(tmin), tmax=float(tmax), variance=float(variance), C=float(C),
            n_perm=n_perm, n_folds=n_folds, n_repeats=n_repeats, n_jobs=n_jobs,
            cv_random_state=RANDOM_SEED, assignment_sha256=assignment_sha,
            input_path=str(path.fpath), input_sha256=hashlib.sha256(Path(path.fpath).read_bytes()).hexdigest()),
            input_metadata=input_metadata, inference=inference)

        logger.info(f"File {i} completed in {_time.time() - _t0:.2f}s")
        Xs[i] = None
        ys[i] = None
        del X, y, decoder, result
        gc.collect()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/",
                        help="Root directory of the BIDS dataset")
    parser.add_argument("--subject", type=str, default='AICl',
                        help="Subject to process")
    parser.add_argument("--ref", type=str, default='bipolar',
                        choices=['car', 'bipolar'],
                        help="Reference scheme")
    parser.add_argument("--description", type=str, default='Repeat',
                        choices=['Repeat', 'Passive', 'Decision'],
                        help="Repeat, Passive, or Decision")
    parser.add_argument("--phase", type=str, default='Stimulus',
                        choices=['Stimulus', 'Delay', 'Go', 'Response'],
                        help="Stimulus, Delay, Go, or Response")
    parser.add_argument("--band", type=str, default='highgamma',
                        help="highgamma or other band of neural signal")
    parser.add_argument("--datatype", type=str, default='phoneme',
                        choices=['phoneme', 'articulator', 'token', 'lexicality'],
                        help="what to classify? can be phoneme, articulator, token, or lexicality")
    parser.add_argument("--variance", type=float, default=0.85,
                        help="PCA variance explained (0-1)")
    parser.add_argument("--C", type=float, default=1.0,
                        help="LinearSVC regularization strength C")
    parser.add_argument("--tmin", type=float, default=None,
                        help="Override phase crop start (seconds); default from PHASE_WINDOWS")
    parser.add_argument("--tmax", type=float, default=None,
                        help="Override phase crop end (seconds); default from PHASE_WINDOWS")
    parser.add_argument("--results_root", type=str, default=None,
                        help="Optional alternate results root (e.g. sweep directory)")
    parser.add_argument("--n_perm", type=int, default=2,
                        help="number of permutations")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="number of folds")
    parser.add_argument("--n_repeats", type=int, default=1,
                        help="number of CV-seed repeats")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="number of jobs")
    parser.add_argument('--inference', choices=['first_repeat', REPEATED_METHOD], default='first_repeat',
                        help='Explicit inference statistic and null construction')
    parser.add_argument('--exclude_channel', action='append', default=[],
                        help='Analysis exclusion after reference-input CV and imputation')
    parser.add_argument('--observed_reference', default=None,
                        help='Optional archived result for exact observed-fold regression validation')
    parser.add_argument('--recording', default=None,
                        help='Optional recording selector, e.g. first phoneme recording 1')

    args = parser.parse_args()
    main(**vars(args))
