"""Cross-domain connectivity analysis with CCA-aligned neural decoders.

This module implements cross-domain decoding for neural connectivity analysis,
where a classifier is trained on one brain region (X1) and evaluated on another
region (X2). Canonical Correlation Analysis (CCA) is used to align feature
spaces between regions before training/testing.

Key components:
- ``CrossDecoder``: CCA-based cross-domain decoder that aligns X1 and X2, then
  trains an sklearn pipeline on the aligned source domain.
- ``cross_domain_resolved_permutation_scores``: time-resolved decoding with
  cross-validation and permutation testing across temporal windows.
- Data utilities: ``load_roi_data`` (BIDS/HDF5 loader with temporal cropping),
  ``_balance_datasets`` (class balancing across domains), ``sample_fold`` (per-fold
  sampling and basic data hygiene).

Typical workflow:
1. Load neural data from two ROIs (regions of interest) for specified time windows.
2. Balance datasets to ensure equal trials per class across regions.
3. Build an sklearn pipeline (e.g., Vectorizer -> StandardScaler -> PCA -> SVC).
4. Use ``CrossDecoder`` with CCA alignment and run time-resolved permutation testing
   via ``cross_domain_resolved_permutation_scores``.

Example:
    >>> from mne.decoding import Vectorizer
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.svm import SVC, LinearSVC
    >>> from ieeg.calc.oversample import MinimumNaNSplit
    >>> 
    >>> X1, y1 = load_roi_data(bids_root, 'PrGl', 'perception', 'highgamma', 'phoneme',
    ...                        tmin=0.0, tmax=0.5)
    >>> X2, y2 = load_roi_data(bids_root, 'STGl', 'perception', 'highgamma', 'phoneme',
    ...                        tmin=0.0, tmax=0.5)
    >>> 
    >>> # Balance datasets across classes present in both ROIs
    >>> X1, X2, y1, y2 = _balance_datasets(X1, y1, X2, y2)
    >>> 
    >>> estimator = make_pipeline(Vectorizer(), StandardScaler(), PCA(0.85), SVC(kernel='linear'))
    >>> decoder = CrossDecoder(estimator, n_components=5)
    >>> cv = MinimumNaNSplit(n_splits=6, n_repeats=1)
    >>> 
    >>> obs_scores, perm_scores, pvals_fdr = cross_domain_resolved_permutation_scores(
    ...     X1, y1, X2, y2, cv, decoder, n_permutations=100, window=0.2, step=0.1, fs=128,
    ...     tmin=0.0, tmax=0.5,
    ... )
"""

import rootutils
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
from pathlib import Path
import h5py
import numpy as np
from mne_bids import BIDSPath
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import get_scorer
from sklearn.base import clone
from mne.decoding import Vectorizer
from sklearn.cross_decomposition import CCA
from einops import rearrange
from joblib import Parallel, delayed
import logging
import sys
from ieeg.calc.fast import mixup
from sklearn.base import BaseEstimator, ClassifierMixin
from ieeg.calc.oversample import MinimumNaNSplit
from tqdm import tqdm
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_roi_data(
    bids_root, 
    ref,
    roi, 
    description, 
    band, 
    datatype,
    tmin,
    tmax,
):
    """Load and preprocess neural data from a specific brain region (ROI).
    
    Loads neural time series data from HDF5 files following BIDS structure,
    then crops the temporal window to -0.5 to 0.5 seconds relative to stimulus.
    
    Parameters
    ----------
    bids_root : str or Path
        Root directory of BIDS dataset containing neural data
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
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Neural time series data cropped to 1-second window
    y : ndarray, shape (n_epochs,)
        Class labels for each epoch
        
    Raises
    ------
    FileNotFoundError
        If no matching files found for the specified ROI
    """
    # Construct BIDS path for the ROI data file
    root = BIDSPath(
        root=os.path.join(bids_root, 'derivatives', f'decoding({ref})'), 
        datatype=datatype, 
        description=description,
        suffix=band, 
        extension='.h5', 
        check=False
    )
    roi_path = root.copy().update(subject=roi)
    roi_files = roi_path.match()
    
    if not roi_files:
        raise FileNotFoundError(f"No files found for ROI {roi}")
        
    # Load neural data and metadata from HDF5 file
    with h5py.File(roi_files[0], 'r') as data:
        X = data['X'][()]  # Neural time series: (epochs, channels, times)
        y = data['y'][()]  # Class labels: (epochs,)
        fs = data.attrs['fs']  # Sampling frequency
        t_end = data.attrs['tmax']
        t_start = data.attrs['tmin']
        
    data.close()
    
    # Crop temporal window to tmin to tmax seconds relative to stimulus onset
    # Assumes original data spans t_start to t_end seconds
    start_idx = int(fs * (tmin - t_start))  # Start at tmin seconds
    end_idx = int(fs * (tmax - t_start))    # End at tmax seconds
    X = X[:, :, start_idx:end_idx]
    
    return X, y
    
    
class CrossDecoder(BaseEstimator, ClassifierMixin):
    """Cross-domain neural decoder with CCA-based feature space alignment.

    This class implements connectivity analysis between brain regions by training
    a classifier on one region (source) and testing on another (target). The key
    innovation is using Canonical Correlation Analysis (CCA) to find a shared
    representational space between the two regions before classification.
    
    The workflow is:
    1. (Balance datasets to ensure equal trials per class across regions)
    2. Apply CCA to find maximally correlated components between regions
    3. Transform both regions' data to the aligned CCA space
    4. Train classifier on source region's aligned data
    5. Test classifier on target region's aligned data
    
    This approach tests the hypothesis that information can transfer between
    brain regions through a shared representational space, indicating functional
    connectivity for the task.

    Parameters
    ----------
    estimator : Pipeline
        Scikit-learn compatible estimator pipeline (e.g., Vectorizer + PCA + SVM).
        Must implement `fit` and `predict` methods.
    n_components : int, default=10
        Number of CCA components to extract. Higher values capture more variance
        but may include noise. Typically 5-20 for neural data.
    random_state : int, default=42
        Random seed for reproducible results.
        
    Attributes
    ----------
    cca : CCA
        Fitted CCA transformer for aligning feature spaces.
    estimator : Pipeline
        Fitted classifier pipeline.
        
    Examples
    --------
    >>> # Create pipeline for neural decoding
    >>> estimator = make_pipeline(
    ...     Vectorizer(),           # Flatten time series
    ...     StandardScaler(),       # Normalize features
    ...     PCA(n_components=0.85), # Dimensionality reduction
    ...     SVC(kernel='linear')    # Linear classifier
    ... )
    >>> 
    >>> # Initialize cross-decoder with CCA alignment
    >>> decoder = CrossDecoder(estimator, n_components=5)
    >>> 
    >>> # Fit on region 1, test on region 2
    >>> decoder.fit(X1, y1, X2, y2)
    >>> predictions = decoder.predict(X2_new)
    """
     
    def __init__(
        self,
        estimator: Pipeline,
        n_components: int = 10, # Number of CCA components
        random_state: int = 42, # Random seed
    ):
         
        self.n_components = n_components
        self.random_state = random_state
        self.estimator = estimator
         
    def fit(self, X1, y1, X2, y2):
        """Fit the cross-domain decoder with CCA alignment.
        
        This method performs the core cross-domain training procedure:
        1. Validates input data dimensions
        2. Balances datasets to have equal trials per class
        3. Fits CCA to find shared representational space
        4. Transforms both regions' data to CCA space
        5. Trains classifier on source region's aligned data

        Parameters
        ----------
        X1 : ndarray, shape (n_epochs, n_channels, n_times)
            Source region neural data for training the classifier.
            Typically high-gamma power or other neural features.
        y1 : ndarray, shape (n_epochs,)
            Class labels for X1 (e.g., phoneme categories, word types).
        X2 : ndarray, shape (n_epochs, n_channels, n_times)
            Target region neural data for CCA alignment.
            Must have same temporal structure as X1.
        y2 : ndarray, shape (n_epochs,)
            Class labels for X2. Used only for dataset balancing;
            must correspond to same trials/conditions as y1.

        Returns
        -------
        self : CrossDecoder
            Fitted decoder ready for cross-domain prediction.
            
        Raises
        ------
        ValueError
            If input arrays are not 3D (epochs, channels, times).
        """
        # Validate input dimensions - must be 3D neural time series
        if X1.ndim != 3 or X2.ndim != 3:
            raise ValueError("X1 and X2 must be 3D arrays in (epoch, channel, time) format")
        
        # Balance datasets to ensure equal trials per class across regions
        # This is crucial for fair cross-domain comparison
        assert np.array_equal(y1, y2), "Labels must be aligned"
        y = y1
        
        # Fit CCA and transform both regions to aligned space
        # Output shape: (epoch, component, time) where components are CCA dimensions
        X1_cca, X2_cca = self.fit_and_transform_cca(X1, X2)
        
        # Train classifier on source region's CCA-aligned data
        # This learns the mapping from aligned neural patterns to class labels
        self.estimator.fit(X1_cca, y)
        
        return self
        
    def predict(self, X2):
        """Predict class labels for target region data.
        
        Transforms target region data to the CCA-aligned space learned during
        fitting, then applies the trained classifier to make predictions.
        This tests whether information from the source region can successfully
        decode target region activity.

        Parameters
        ----------
        X2 : ndarray, shape (n_epochs, n_channels, n_times)
            Target region neural data to classify.
            Must have same channel and time structure as training data.

        Returns
        -------
        predicted : ndarray, shape (n_epochs,)
            Predicted class labels for each epoch in X2.
            
        Notes
        -----
        This method requires the decoder to be fitted first. The CCA transformation
        uses the alignment learned during fit() to project X2 into the shared space.
        """
        # Transform target region data to CCA-aligned space
        X2_cca = self.transform_cca(X2)
        
        # Apply trained classifier to make predictions
        predicted = self.estimator.predict(X2_cca)
        
        return predicted
        
    def fit_and_transform_cca(self, X1, X2):
        """Fit CCA alignment and transform both domains' data.
        
        Canonical Correlation Analysis finds linear combinations of features
        in each domain that are maximally correlated. This creates a shared
        representational space for cross-domain analysis.
        
        The method reshapes 3D neural data (epochs, channels, times) to 3D
        (epochs, components, times) for CCA fitting, then reshapes back to preserve
        temporal structure.
        
        Parameters
        ----------
        X1 : ndarray, shape (n_epochs, n_channels, n_times)
            Source domain neural data
        X2 : ndarray, shape (n_epochs, n_channels, n_times)  
            Target domain neural data
            
        Returns
        -------
        X1_cca : ndarray, shape (n_epochs, n_components, n_times)
            Source domain data in CCA-aligned space
        X2_cca : ndarray, shape (n_epochs, n_components, n_times)
            Target domain data in CCA-aligned space
            
        Notes
        -----
        CCA is fitted on the concatenated time series across all epochs,
        treating each time point as an independent sample. This captures
        temporal dynamics in the correlation structure.
        """
        # Store original dimensions for reshaping
        n_epochs, n_channels, n_time = X1.shape
        
        # Reshape to 2D for CCA: (samples, features) where samples = epochs × times
        # This treats each time point as an independent observation
        X1 = rearrange(X1, 'epoch channel time -> (epoch time) channel')
        X2 = rearrange(X2, 'epoch channel time -> (epoch time) channel')
        
        # Fit CCA to find maximally correlated linear combinations
        # n_components determines dimensionality of shared space
        components = min(self.n_components, min(X1.shape[-1], X2.shape[-1]))        
        cca = CCA(n_components=components)
        cca.fit(X1, X2)
        self.cca = cca  # Store for later use in predict()
        
        # Transform both regions to CCA-aligned space
        X1_cca, X2_cca = self.cca.transform(X1, X2)
        
        # compute the covariance matrix
        cov_X1 = np.cov(X1.T)
        cov_X2 = np.cov(X2.T)
        
        # compute the weights
        U, V = X1 @ cca.x_weights_, X2 @ cca.y_weights_
        Cov_U, Cov_V = np.cov(U.T), np.cov(V.T)
        P1 = cov_X1 @ cca.x_weights_ @ np.linalg.inv(Cov_U)
        P2 = cov_X2 @ cca.y_weights_ @ np.linalg.inv(Cov_V)
        
        self.x_pattern_ = P1
        self.y_pattern_ = P2
        
        # Reshape back to 3D: (epochs, components, times)
        # Components replace channels as the feature dimension
        X1_cca = rearrange(X1_cca, '(epoch time) component -> epoch component time', epoch=n_epochs)
        X2_cca = rearrange(X2_cca, '(epoch time) component -> epoch component time', epoch=n_epochs)
        
        return X1_cca, X2_cca

    def transform_source_cca(self, X1):
        """Transform source domain data using fitted CCA alignment."""
        n_epochs, n_channels, n_time = X1.shape
        X1 = rearrange(X1, 'epoch channel time -> (epoch time) channel')
        X1_cca = X1 @ self.cca.x_weights_
        X1_cca = rearrange(X1_cca, '(epoch time) component -> epoch component time', epoch=n_epochs)
        return X1_cca
    
    def transform_cca(self, X2):
        """Transform target domain data using fitted CCA alignment.
        
        Applies the CCA transformation learned during fit() to new target
        domain data. This projects the data into the shared representational
        space for classification.
        
        Parameters
        ----------
        X2 : ndarray, shape (n_epochs, n_channels, n_times)
            Target domain neural data to transform
            
        Returns
        -------
        X2_cca : ndarray, shape (n_epochs, n_components, n_times)
            Transformed data in CCA-aligned space
            
        Notes
        -----
        This method uses a dummy X1 array since CCA.transform() expects
        both domains, but we only need the X2 transformation. The dummy
        array has the correct feature dimension but random values.
        """
        # Store dimensions and reshape for CCA transformation
        n_epochs, n_channels, n_time = X2.shape
        X2 = rearrange(X2, 'epoch channel time -> (epoch time) channel')
        
        # Create dummy X1 array since CCA.transform() expects both inputs
        # Only the X2 transformation is used, so X1 values don't matter
        X1_dummy = np.random.randn(X2.shape[0], self.cca.x_weights_.shape[0])
        
        # Apply CCA transformation (only X2_cca is meaningful)
        _, X2_cca = self.cca.transform(X1_dummy, X2)
        
        # Reshape back to 3D temporal structure
        X2_cca = rearrange(X2_cca, '(epoch time) component -> epoch component time', epoch=n_epochs)
        
        return X2_cca

    
    def _balance_datasets(self, X1, y1, X2, y2):
        """Balance datasets to ensure equal trials per class across domains.
        
        Cross-domain analysis requires balanced datasets to avoid bias.
        This method finds the minimum number of trials per class across
        both domains and subsamples to that number.
        
        Parameters
        ----------
        X1, X2 : ndarray, shape (n_epochs, n_channels, n_times)
            Neural data from source and target domains
        y1, y2 : ndarray, shape (n_epochs,)
            Class labels for each domain
            
        Returns
        -------
        X1_balanced, X2_balanced : ndarray
            Balanced neural data with equal trials per class
        y_balanced : ndarray
            Aligned labels (same for both domains after balancing)
            
        Notes
        -----
        Only classes present in both domains are retained. The method
        ensures that y1_balanced == y2_balanced after balancing.
        """
        # Find classes present in both regions
        unique_classes = np.intersect1d(np.unique(y1), np.unique(y2))
        
        # Balance each class separately
        balanced_data = []
        for cls in unique_classes:
            # Extract trials for this class from both regions
            X1_cls, X2_cls = X1[y1 == cls], X2[y2 == cls]
            y1_cls, y2_cls = y1[y1 == cls], y2[y2 == cls]
            
            # Use minimum number of trials across regions
            min_trials = min(len(X1_cls), len(X2_cls))
            balanced_data.append((X1_cls[:min_trials], X2_cls[:min_trials], 
                               y1_cls[:min_trials], y2_cls[:min_trials]))
        
        # Concatenate balanced data across classes
        X1_balanced = np.concatenate([x1 for x1, _, _, _ in balanced_data])
        X2_balanced = np.concatenate([x2 for _, x2, _, _ in balanced_data])
        y1_balanced = np.concatenate([y1 for _, _, y1, _ in balanced_data])
        y2_balanced = np.concatenate([y2 for _, _, _, y2 in balanced_data])
        
        # Verify labels are aligned after balancing
        assert np.array_equal(y1_balanced, y2_balanced), "Labels must be aligned"
        logger.info(f"Balanced dataset: {len(y1_balanced)} trials total")
        
        return X1_balanced, X2_balanced, y1_balanced


def cross_domain_cv_predict(
    X1,
    y1,
    X2,
    y2,
    cv,
    cross_decoder,
    n_jobs: int = -1,
    predict_method: str = "predict",
):
    """
    Cross-domain OOF-style predictions on XB: fit on A_train, predict on B_test.

    Parameters are analogous to cross_domain_cv_score, with:
    - predict_method: 'predict' | 'predict_proba' | 'decision_function'

    Returns
    - y_pred: array aligned to y (index-wise), containing predictions for XB at test folds.
              For 'predict', shape (n_samples,);
              For 'predict_proba' or 'decision_function', shape (n_samples, n_outputs).
    """
    # Only 'predict' is currently supported by CrossDecoder
    if predict_method != "predict":
        raise ValueError("Only 'predict' is supported by CrossDecoder in CV mode.")

    splits = list(cv.split(X1, y1))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")

    # Pre-allocate output (labels)
    y_pred = np.empty_like(y2)

    def one_fold(train_idx, test_idx):
        # Create a fresh CrossDecoder per fold to avoid shared state
        dec = clone(cross_decoder)
        X1_train, X1_test, y1_train, y1_test = sample_fold(
            X1,
            y1,
            train_idx,
            test_idx,
        )
        X2_train, X2_test, y2_train, y2_test = sample_fold(
            X2,
            y2,
            train_idx,
            test_idx,
        )
        
        dec.fit(
            X1_train,
            y1_train,
            X2_train,
            y2_train,
        )
        pred = dec.predict(X2_test)
        return test_idx, pred

    # Run cross-validation with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(one_fold)(tr, te) for tr, te in tqdm(splits, desc="CV folds")
    )

    for te, pred in results:
        y_pred[te] = pred

    return y_pred

def cross_domain_cv_score(
    X1,
    y1,
    X2,
    y2,
    cv,
    cross_decoder,
    n_jobs: int = -1,
    predict_method: str = "predict",
):
    """Cross-domain OOF-style predictions on XB: fit on A_train, predict on B_test."""
    
    from sklearn.metrics import accuracy_score
    
    y_pred = cross_domain_cv_predict(
        X1,
        y1,
        X2,
        y2,
        cv,
        cross_decoder,
        n_jobs=n_jobs,
        predict_method=predict_method,
    )
    
    return accuracy_score(y2, y_pred)
    
def cross_domain_permutation_scores(
    X1,
    y1,
    X2,
    y2,
    cv,
    cross_decoder,
    scoring: str = "accuracy",
    n_permutations: int = 10,
    n_jobs: int = -1,
    random_state: int = 42,
):
    
    scorer = get_scorer(scoring)
    splits = list(cv.split(X1, y1))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")
    
    # Observed
    obs_scores = []
    perm_scores = []
    for tr, te in tqdm(splits, desc="Cross-validation"):
        dec = clone(cross_decoder)
        X1_train, X1_test, y1_train, y1_test = sample_fold(
            X1,
            y1,
            tr,
            te,
        )
        X2_train, X2_test, y2_train, y2_test = sample_fold(
            X2,
            y2,
            tr,
            te,
        )
        
        dec.fit(
            X1_train,
            y1_train,
            X2_train,
            y2_train,
        )
        observed_score = scorer(dec, X2_test, y2_test)
        obs_scores.append(observed_score)
        
        # permute
        X1_train_cca = dec.transform_source_cca(X1_train)
        X2_test_cca = dec.transform_cca(X2_test)
        
        rng_fold = np.random.RandomState(random_state)
        seeds_fold = rng_fold.randint(0, 2**31 - 1, size=n_permutations)

        def one_perm(seed):
            r = np.random.RandomState(seed)
            y1_train_perm = y1_train.copy()
            r.shuffle(y1_train_perm)
            est_p = clone(dec.estimator)
            est_p.fit(X1_train_cca, y1_train_perm)
            return scorer(est_p, X2_test_cca, y2_test)

        perm_score = np.asarray(Parallel(n_jobs=n_jobs)(delayed(one_perm)(s) for s in tqdm(seeds_fold, desc="Permutations")))
        
        perm_scores.append(perm_score)
        
    score = np.mean(obs_scores)
    perm_scores = np.stack(perm_scores)
    
    # p-value (greater is better metric)
    p_value = (np.sum(perm_scores.mean(axis=0) >= score) + 1.0) / (n_permutations + 1.0)
    
    return obs_scores, perm_scores, p_value
    
    

def cross_domain_resolved_permutation_scores(
    X1,
    y1,
    X2,
    y2,
    cv,
    cross_decoder,
    scoring: str = "accuracy",
    n_permutations: int = 10,
    n_jobs: int = -1,
    random_state: int = 42,
    window: float = 0.2,
    step: float = 0.1,
    fs: int = 128,
    tmin: float = 0,
    tmax: float = 0.5,
):
    """Time-resolved cross-domain decoding with permutation testing and FDR.

    For each CV fold, this function:
    - Fits a fresh ``CrossDecoder`` to align X1/X2 via CCA using the training split.
    - Slides a window over the temporal axis [tmin, tmax] with length ``window`` and step ``step``.
    - For each time window, trains the estimator on aligned X1 (train) and evaluates on aligned X2 (test).
    - Builds a permutation baseline by shuffling y1 in the training split ``n_permutations`` times.

    After all folds finish, observed fold scores and permuted fold scores are averaged across folds
    to obtain per-time observed statistics and the corresponding permutation distribution. Per-time
    one-sided p-values are computed and corrected using FDR-BH.

    Parameters
    ----------
    X1, X2 : ndarray, shape (n_epochs, n_channels, n_times)
        Source and target ROI data.
    y1, y2 : ndarray, shape (n_epochs,)
        Class labels. After balancing upstream, they should be aligned across domains.
    cv : CV splitter
        Any sklearn-compatible splitter with ``split(X1, y1)`` yielding (train_idx, test_idx).
    cross_decoder : CrossDecoder
        A configured ``CrossDecoder`` instance wrapping an sklearn estimator pipeline.
    scoring : str, default='accuracy'
        Scorer name accepted by sklearn's ``get_scorer``.
    n_permutations : int, default=10
        Number of label permutations per fold per time window.
    n_jobs : int, default=-1
        Parallel jobs for permutation evaluations within each time window.
    random_state : int, default=42
        Base seed for reproducibility. Seeds are expanded per fold/time internally.
    window : float, default=0.2
        Window length (seconds) for time-resolved decoding.
    step : float, default=0.1
        Step size (seconds) for sliding window.
    fs : int, default=128
        Sampling rate used to convert seconds to samples.
    tmin, tmax : float, default=(0, 0.5)
        Temporal range (seconds) over which windows are evaluated.

    Returns
    -------
    obs_scores : ndarray, shape (T, n_folds)
        Observed accuracy (or other ``scoring``) per time window per fold.
    perm_scores : ndarray, shape (T, n_permutations, n_folds)
        Permutation baseline scores per time window, permutation, and fold.
    pvals_fdr : ndarray, shape (T,)
        FDR-BH corrected per-time p-values (one-sided; greater-is-better).

    Notes
    -----
    - Parallelization happens at the permutation level within each time window. For large T × folds,
      consider parallelizing at the fold level externally to reduce overhead.
    - For stricter family-wise error control across time, consider adding a max-T correction using
      the per-time fold-averaged permutation distribution.
    - To avoid unnecessary memory pressure, keep ``perm_scores`` only if downstream analysis requires it.
    """
    from statsmodels.stats.multitest import multipletests
    
    scorer = get_scorer(scoring)
    splits = list(cv.split(X1, y1))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")
    
    time_points = np.arange(tmin + window,
                            tmax + step,
                            step)
    window_samples = int(window * fs)
    step_samples = int(step * fs)
    
    # Observed
    n_folds = len(splits)
    obs_scores = np.empty((len(time_points), n_folds))
    perm_scores = np.empty((len(time_points), n_permutations, n_folds))
    
    for fold_idx, (tr, te) in enumerate(tqdm(splits, desc="Cross-validation")):
        
        dec = clone(cross_decoder)
        
        X1_train, X1_test, y1_train, y1_test = sample_fold(
            X1,
            y1,
            tr,
            te,
        )
        X2_train, X2_test, y2_train, y2_test = sample_fold(
            X2,
            y2,
            tr,
            te,
        )
        
        X1_train_cca, X2_train_cca = dec.fit_and_transform_cca(X1_train, X2_train)
        X2_test_cca = dec.transform_cca(X2_test)

        # Generate permutation seeds once per fold
        rng_fold = np.random.RandomState(random_state + fold_idx)
        seeds_fold = rng_fold.randint(0, 2**31 - 1, size=n_permutations)

        # time resolved decoding - for resolved, train_time == test_time
        # so we still need to fit per time window, but parallelize permutations efficiently
        for t_idx, time_end in enumerate(tqdm(time_points, desc="Sliding windows")):
            
            end_sample = int((time_end - tmin) * fs) + 1
            start_sample = end_sample - window_samples
            
            if start_sample < 0 or end_sample > X1_train_cca.shape[-1]:
                logger.warning(f"Window out of bounds for time {time_end:.3f}s, skipping")
                continue
            
            x1_train_s = X1_train_cca[..., start_sample:end_sample]
            x2_train_s = X2_train_cca[..., start_sample:end_sample]
            x2_test_s = X2_test_cca[..., start_sample:end_sample]
            
            dec.estimator.fit(x1_train_s, y1_train)
            observed_score = scorer(dec.estimator, x2_test_s, y2_test)

            def one_perm(seed):
                r = np.random.RandomState(seed)
                y1_train_perm = y1_train.copy()
                r.shuffle(y1_train_perm)
                est_p = clone(dec.estimator)
                est_p.fit(x1_train_s, y1_train_perm)
                return scorer(est_p, x2_test_s, y2_test)

            perm_score = np.asarray(Parallel(n_jobs=n_jobs, batch_size=40)(delayed(one_perm)(s) for s in seeds_fold))
            
            obs_scores[t_idx, fold_idx] = observed_score
            perm_scores[t_idx, :, fold_idx] = perm_score
            
    observed_t = obs_scores.mean(axis=1)     # (T,)
    perm_t = perm_scores.mean(axis=2)        # (T, n_perm)

    pvals_pt = ((perm_t >= observed_t[:, None]).sum(axis=1) + 1) / (perm_t.shape[1] + 1)
    _, pvals_fdr, _, _ = multipletests(pvals_pt, alpha=0.05, method='fdr_bh')

    return obs_scores, perm_scores, pvals_fdr
    
    

def cross_domain_generalized_permutation_scores(
    X1,
    y1,
    X2,
    y2,
    cv,
    cross_decoder,
    scoring: str = "accuracy",
    n_permutations: int = 10,
    n_jobs: int = -1,
    random_state: int = 42,
    window: float = 0.2,
    step: float = 0.1,
    fs: int = 128,
    train_tmin: float = 0,
    train_tmax: float = 0.5,
    test_tmin: float = 0,
    test_tmax: float = 0.5,
):
    """Temporal generalization (train-time × test-time) decoding with permutations and FDR.

    For each CV fold, this function:
    - Fits a fresh ``CrossDecoder`` (CCA alignment + sklearn pipeline) on the training split.
    - Constructs two sliding-window grids: a train-time grid within [train_tmin, train_tmax] and
      a test-time grid within [test_tmin, test_tmax], each with window length ``window`` and step ``step``.
    - For every cell (t_train, t_test), trains the estimator on X1 (aligned, training split) using the
      train window and evaluates on X2 (aligned, test split) using the test window.
    - Builds a permutation baseline at each cell by shuffling y1 in the training split ``n_permutations`` times.

    After all folds finish, scores are averaged across folds to obtain a 2D observed map and a corresponding
    permutation distribution per cell. One-sided per-cell p-values are computed from the permutation null and
    then corrected across the 2D field using FDR-BH.

    Parameters
    ----------
    X1, X2 : ndarray, shape (n_epochs, n_channels, n_times)
        Source (train) and target (test) ROI data.
    y1, y2 : ndarray, shape (n_epochs,)
        Class labels for X1/X2. Upstream balancing should align label distributions.
    cv : CV splitter
        Any sklearn-compatible splitter yielding (train_idx, test_idx) on (X1, y1).
    cross_decoder : CrossDecoder
        Configured cross-decoder (CCA + estimator pipeline).
    scoring : str, default='accuracy'
        Scorer name accepted by sklearn's ``get_scorer``.
    n_permutations : int, default=10
        Number of label permutations per fold per (t_train, t_test) cell.
    n_jobs : int, default=-1
        Parallel jobs for permutation evaluations within each cell.
    random_state : int, default=42
        Base seed for reproducibility (expanded per fold/cell internally as needed).
    window : float, default=0.2
        Window length in seconds for temporal slicing.
    step : float, default=0.1
        Step size in seconds between adjacent windows.
    fs : int, default=128
        Sampling rate to convert seconds to sample indices.
    train_tmin, train_tmax : float, default=(0, 0.5)
        Temporal range (seconds) for train-time windows (applied to X1 on the training split).
    test_tmin, test_tmax : float, default=(0, 0.5)
        Temporal range (seconds) for test-time windows (applied to X2 on the test split).

    Returns
    -------
    obs_scores : ndarray, shape (T_train, T_test, n_folds)
        Observed score (``scoring``) per (train-time, test-time) cell and per fold.
    perm_scores : ndarray, shape (T_train, T_test, n_permutations, n_folds)
        Permutation baseline scores per cell, permutation, and fold.
    pvals_fdr : ndarray, shape (T_train, T_test)
        FDR-BH corrected per-cell p-values (one-sided; greater-is-better).

    Notes
    -----
    - Indexing: indices are derived with rounding to samples (``round(seconds * fs)``) and slicing
      uses half-open ranges [start:end], ensuring a constant window length of ``window * fs`` samples.
    - Parallelization occurs at the permutation level inside each cell; consider adjusting the
      granularity (e.g., over folds) if overhead becomes significant for large grids.
    - To implement FWER control, you can add a 2D max-field correction or a 2D cluster-based
      permutation procedure on the fold-averaged statistic map.
    """
    from statsmodels.stats.multitest import multipletests
    
    scorer = get_scorer(scoring)
    splits = list(cv.split(X1, y1))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")
    
    
    # format to .2f 
    train_tmin, train_tmax, test_tmin, test_tmax = [round(t, 2) for t in [train_tmin, train_tmax, test_tmin, test_tmax]]
    
    train_time_points = np.arange(train_tmin + window,
                            train_tmax + step,
                            step)
    test_time_points = np.arange(test_tmin + window,
                            test_tmax + step,
                            step)
    window_samples = int(window * fs)
    step_samples = int(step * fs)
    
    # Observed
    n_folds = len(splits)
    obs_scores = np.empty((len(train_time_points), len(test_time_points), n_folds))
    perm_scores = np.empty((len(train_time_points), len(test_time_points), n_permutations, n_folds))
    
    for fold_idx, (tr, te) in enumerate(tqdm(splits, desc="Cross-validation")):
        
        dec = clone(cross_decoder)
        
        X1_train, X1_test, y1_train, y1_test = sample_fold(
            X1,
            y1,
            tr,
            te,
        )
        X2_train, X2_test, y2_train, y2_test = sample_fold(
            X2,
            y2,
            tr,
            te,
        )
        
        X1_train_cca, X2_train_cca = dec.fit_and_transform_cca(X1_train, X2_train)
        X2_test_cca = dec.transform_cca(X2_test)

        # Generate permutation seeds once per fold (shared across all time windows)
        rng_fold = np.random.RandomState(random_state + fold_idx)
        seeds_fold = rng_fold.randint(0, 2**31 - 1, size=n_permutations)
        
        # time generalized decoding - batch permutation fits per train_time
        for train_t_idx, train_time_end in enumerate(train_time_points):
            
            end_train = int(round((train_time_end - train_tmin) * fs))
            start_train = end_train - window_samples
            
            if start_train < 0 or end_train > X1_train_cca.shape[-1]:
                logger.warning(f"Window out of bounds for time {train_time_end:.3f}s, skipping")
                continue
            
            x1_train_s = X1_train_cca[..., start_train:end_train]
            
            # Fit observed estimator once for this train_time
            dec.estimator.fit(x1_train_s, y1_train)
            
            # Pre-fit all permutation estimators for this train_time (batch optimization)
            def fit_one_perm(seed):
                r = np.random.RandomState(seed)
                y1_train_perm = y1_train.copy()
                r.shuffle(y1_train_perm)
                est_p = clone(dec.estimator)
                est_p.fit(x1_train_s, y1_train_perm)
                return est_p
            
            perm_estimators = Parallel(n_jobs=n_jobs, batch_size=40)(
                delayed(fit_one_perm)(s) for s in seeds_fold
            )
            
            # Now evaluate on all test_times (only predict, no fit - much faster)
            for test_t_idx, test_time_end in enumerate(test_time_points):
                
                end_test = int(round((test_time_end - test_tmin) * fs))
                start_test = end_test - window_samples
            
                x2_test_s = X2_test_cca[..., start_test:end_test]
                
                # Observed score
                observed_score = scorer(dec.estimator, x2_test_s, y2_test)
                obs_scores[train_t_idx, test_t_idx, fold_idx] = observed_score
                
                # Permutation scores (only scoring, estimators already fitted)
                perm_score = np.array([scorer(est_p, x2_test_s, y2_test) for est_p in perm_estimators])
                perm_scores[train_t_idx, test_t_idx, :, fold_idx] = perm_score
            
    observed_mean = obs_scores.mean(axis=-1)        # (Ttr, Tte)
    perm_mean = perm_scores.mean(axis=-1)           # (Ttr, Tte, n_perm)
    
    P = n_permutations
    pvals_pt = ( (perm_mean >= observed_mean[..., None]).sum(axis=2) + 1 ) / (P + 1)

    p_flat = pvals_pt.ravel()
    _, pvals_corrected, _, _ = multipletests(p_flat, alpha=0.05, method='fdr_bh')
    pvals_corrected = pvals_corrected.reshape(pvals_pt.shape)

    return obs_scores, perm_scores, pvals_corrected
    

def _balance_datasets(
    X1,
    y1,
    X2,
    y2,
):
    """Balance datasets to have same trials per class."""
    unique_classes = np.intersect1d(np.unique(y1), np.unique(y2))
    
    balanced_data = []
    for cls in unique_classes:
        X1_cls, X2_cls = X1[y1 == cls], X2[y2 == cls]
        y1_cls, y2_cls = y1[y1 == cls], y2[y2 == cls]
        
        # Shuffle X1 and X2 independently to break trial correspondence
        perm1 = np.random.permutation(len(X1_cls))
        perm2 = np.random.permutation(len(X2_cls))
        
        X1_cls = X1_cls[perm1]
        X2_cls = X2_cls[perm2]
        y1_cls = y1_cls[perm1]
        y2_cls = y2_cls[perm2]
        
        min_trials = min(len(X1_cls), len(X2_cls))
        balanced_data.append((X1_cls[:min_trials], X2_cls[:min_trials], 
                           y1_cls[:min_trials], y2_cls[:min_trials]))
    
    X1_balanced = np.concatenate([x1 for x1, _, _, _ in balanced_data])
    X2_balanced = np.concatenate([x2 for _, x2, _, _ in balanced_data])
    y1_balanced = np.concatenate([y1 for _, _, y1, _ in balanced_data])
    y2_balanced = np.concatenate([y2 for _, _, _, y2 in balanced_data])
    
    assert np.array_equal(y1_balanced, y2_balanced), "Labels must be aligned"
    
    return X1_balanced, X2_balanced, y1_balanced, y2_balanced

def sample_fold(
    X,
    y,
    train_idx,
    test_idx,
):
    """Sample a fold of data for cross-validation."""
    X_, y_ = X.copy(), y.copy()
    X_train, X_test = X_[train_idx], X_[test_idx]
    y_train, y_test = y_[train_idx], y_[test_idx]
    
    unique_classes = np.unique(y_train)
    for cls in unique_classes:
        idx = (y_train == cls)
        # observer axis is the epoch axis
        x_cls = X_train[idx]
        mixup(x_cls, obs_axis=0, rng=42)
        X_train[idx] = x_cls
    
    is_nan_test = np.isnan(X_test)
    if is_nan_test.any():
        X_test[is_nan_test] = np.random.normal(0, 1, int(np.sum(is_nan_test)))
    
    return X_train, X_test, y_train, y_test


def main(
    bids_root,
    ref,
    train_roi,
    test_roi,
    description,
    band,
    datatype,
    tmin,
    tmax,
    variance,
    n_components,
    n_permutations,
    n_jobs,
    n_folds,
):
    # Create estimator pipeline
    estimator = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        PCA(n_components=variance, random_state=42),
        SVC(kernel='linear', random_state=42)
    )
    
    # load X1 and X2
    X1, y1 = load_roi_data(bids_root, ref, train_roi, description, band, datatype, tmin, tmax)
    X2, y2 = load_roi_data(bids_root, ref, test_roi, description, band, datatype, tmin, tmax)
    
    
    # balance datasets
    X1, X2, y1, y2 = _balance_datasets(X1, y1, X2, y2)
    
    cross_decoder = CrossDecoder(
        estimator=estimator,
        n_components=n_components,
        random_state=42,
    )
    
    msn = MinimumNaNSplit(n_splits=n_folds, n_repeats=1)
    
    obs_scores, perm_scores, p_value = cross_domain_permutation_scores(
        X1 = X1,
        X2 = X2,
        y1 = y1,
        y2 = y2,
        cv = msn,
        cross_decoder = cross_decoder,
        n_jobs=n_jobs,
        n_permutations=n_permutations,
    )
    
    # score = cross_domain_cv_score(
    #     X1 = X1,
    #     X2 = X2,
    #     y1 = y1,
    #     y2 = y2,
    #     cv = msn,
    #     cross_decoder = cross_decoder,
    #     n_jobs=n_jobs,
    # )
    
    print('obs_scores:', np.mean(obs_scores))
    print('perm_scores:', np.mean(perm_scores))
    print('p_value:', p_value)
    
    
    # cross_decoder.fit(X1, y1, X2, y2)
    
    # # predict
    # predicted = cross_decoder.predict(X2)
    
    # # calculate accuracy
    # accuracy = np.mean(predicted == y2)
    
    return
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS")
    parser.add_argument("--ref", type=str, default='ROI')
    parser.add_argument("--train_roi", type=str, default="STGl")
    parser.add_argument("--test_roi", type=str, default="PrGl")
    parser.add_argument("--description", type=str, default='production')
    parser.add_argument("--band", type=str, default='highgamma')
    parser.add_argument("--datatype", type=str, default='phoneme')
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=0.5)
    parser.add_argument("--variance", type=float, default=0.85)
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--n_permutations", type=int, default=300)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--n_folds", type=int, default=10)
    args = parser.parse_args()
    main(**vars(args))
