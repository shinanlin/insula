"""Read windowed decoder results according to their analysis schema."""
from pathlib import Path

import h5py
import numpy as np

from src.decoding.repeated_cv import METHOD, SCHEMA_VERSION
from src.decoding.formal_cv import METHOD as FORMAL_METHOD, SCHEMA_VERSION as FORMAL_SCHEMA


def load_windowed_result(path, *, expected_method=None):
    """Return the stored score units and their matched inference, without overrides."""
    path = Path(path)
    with h5py.File(path, 'r') as f:
        method = str(f.attrs.get('inference_method', 'first_repeat'))
        if expected_method is not None and method != expected_method:
            raise ValueError(f'{path}: inference {method!r}, expected {expected_method!r}')
        scores = np.asarray(f['accuracy'][()], dtype=float)
        p = float(f['p_value'][()])
        if method in (METHOD, FORMAL_METHOD):
            required_schema = FORMAL_SCHEMA if method == FORMAL_METHOD else SCHEMA_VERSION
            if int(f.attrs['result_schema_version']) != required_schema:
                raise ValueError(f'{path}: unsupported repeated-CV schema')
            if method == FORMAL_METHOD and not f.attrs.get('all_fits_converged', False):
                raise ValueError(f'{path}: formal result failed the convergence gate')
            repetitions = f['accuracy_repeats'][:]
            permutations = f['perm_scores'][:]
            shape = (int(f.attrs['n_repeats']), int(f.attrs['n_folds']))
            if repetitions.shape != shape or permutations.shape != (int(f.attrs['n_perm']), *shape):
                raise ValueError(f'{path}: scores/null do not cover the same repeats and folds')
            if not np.array_equal(scores, repetitions.mean(axis=1)):
                raise ValueError(f'{path}: plotted score units are not CV-repeat means')
            null = permutations.mean(axis=(1, 2))
            expected_p = (1.+np.count_nonzero(null >= repetitions.mean()))/(len(null)+1.)
            if p != expected_p:
                raise ValueError(f'{path}: p does not test the stored overall mean')
            se = float(scores.std(ddof=1)/np.sqrt(len(scores)))
            if not np.isclose(se, float(f['accuracy_se'][()]), rtol=0, atol=1e-14):
                raise ValueError(f'{path}: SE is not calculated across CV-repeat means')
            unit = 'CV-repeat mean'
        else:
            se = float(scores.std(ddof=1)/np.sqrt(len(scores)))
            unit = 'first-repeat fold'
        return dict(scores=scores, mean=float(scores.mean()), SE=se,
            accuracy_stable=float(f['accuracy_stable'][()]), p_value=p,
            n_classes=len(f['classes']), inference_method=method, score_unit=unit,
            n_repeats=int(f.attrs.get('n_repeats', 1)), n_folds=int(f.attrs.get('n_folds', len(scores))),
            n_permutations=int(f.attrs.get('n_perm', 0)),
            n_electrodes=int(f.attrs.get('n_electrodes', -1)), n_trials=int(f.attrs.get('n_trials', -1)),
            n_subjects=int(f.attrs.get('n_subjects', -1)), h5_path=str(path),
            p_value_fdr=float(f['p_value_fdr'][()]) if 'p_value_fdr' in f else np.nan,
            correction_family_size=int(f.attrs.get('correction_family_size', 0)))
