import argparse
import logging
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mne_bids import BIDSPath
from scipy import ndimage
import warnings
import sys
from ieeg.calc.stats import time_cluster

def main(
    bids_root: str,
    band: str,
    ref: str
):
    
    # 默认写死的参数 (Hardcoded logic mapping)
    datatype = "lexicality"
    
    # Extract the task name natively from the bids_root directory name heuristically,
    # or just mapping it for the results root. Match logic from other pipeline scripts:
    task_root = "LexicalDelay"  # fallback

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Target directory structure follows the saving convention of direct_cross_decoder results
    # e.g.: results/LexicalDelay(roi)(bipolar)
    root_dir = f"results/{task_root}(roi)({ref})"
    
    # Setup BIDSPath for finding all generalized decoding results
    search_path = BIDSPath(
        root=root_dir,
        datatype=f"(cross)(generalized){datatype}",
        suffix=band,
        extension=".h5",
        check=False,
    )
    
    matched_files = search_path.match()
    if not matched_files:
        logging.warning(f"No match found for {search_path}")
        return
        
    logging.info(f"Found {len(matched_files)} 2D generalized decoding files. Starting MNE cluster testing...")
    
    for file_path in tqdm(matched_files, desc='Processing 2D Decoding results'):
        
        try:
            with h5py.File(file_path.fpath, 'r+') as mat:
                
                # Check if 'scores' and 'baseline' exists
                if 'scores' not in mat or 'baseline' not in mat:
                    logging.warning(f"Required datasets not found in {file_path.fpath}, skipping.")
                    continue
                
                scores = mat['scores'][()] # Shape: (Ttr, Tte, n_folds)
                baseline = mat['baseline'][()] # Shape: (Ttr, Tte, n_perm, n_folds)
                
                # ============================================================
                # Cluster-based permutation test using EXISTING permutation data
                # Uses ieeg.calc.stats.time_cluster for 2D cluster correction
                # ============================================================
                # scores: (Ttr, Tte, n_folds) - observed decoding accuracy
                # baseline: (Ttr, Tte, n_perm, n_folds) - permuted decoding accuracy
                # ============================================================
                
                Ttr, Tte, n_folds = scores.shape
                n_perm = baseline.shape[2]
                
                # Step 1: Average across folds (folds are CV splits, not independent samples)
                obs_mean = scores.mean(axis=-1)  # (Ttr, Tte)
                perm_mean = baseline.mean(axis=-1)  # (Ttr, Tte, n_perm)
                
                # Step 2: Compute point-wise p-values for observed data
                # p = (# of perms >= observed + 1) / (n_perm + 1)
                p_obs = ((perm_mean >= obs_mean[..., None]).sum(axis=-1) + 1) / (n_perm + 1)
                
                # Step 3: Compute point-wise p-values for each permutation (leave-one-out)
                # For each perm p, compare against all OTHER perms
                p_perm = np.zeros((n_perm, Ttr, Tte))
                for p in range(n_perm):
                    perm_slice = perm_mean[:, :, p]  # (Ttr, Tte)
                    other_perms = np.delete(perm_mean, p, axis=-1)  # (Ttr, Tte, n_perm-1)
                    p_perm[p] = ((other_perms >= perm_slice[..., None]).sum(axis=-1) + 1) / (n_perm)
                
                # Step 4: Threshold to get binary masks (cluster-forming threshold)
                threshold = 0.05
                obs_mask = (p_obs < threshold).astype(int)  # (Ttr, Tte)
                perm_masks = (p_perm < threshold).astype(int)  # (n_perm, Ttr, Tte)
                
                # Step 5: Use time_cluster for 2D cluster correction
                # time_cluster expects: act (Ttr, Tte), perm (n_perm, Ttr, Tte)
                # Returns: proportion of perms where observed cluster is LARGER than null max cluster
                # So higher value = more significant. Convert to standard p-value (lower = more significant)
                cluster_proportion = time_cluster(obs_mask, perm_masks, p_val=None, tails=1)
                p_map = 1 - cluster_proportion  # Convert to p-value format
                        
                # Create a NEW dataset without overwriting original p-values
                target_col = 'p_values_cluster'
                if target_col in mat:
                    del mat[target_col]
                    
                mat.create_dataset(target_col, data=p_map)
                
        except Exception as e:
            logging.error(f"Error processing {file_path.fpath}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_TIMIT/BIDS/", type=str)
    
    parser.add_argument("--band", type=str, default="highgamma", choices=['highgamma','gamma','beta','alpha','theta'],
                        help='which frequency band to use')
    parser.add_argument("--ref", type=str, default='bipolar',
                        choices=['bipolar','car'],
                        help='reference channel')
    
    args = parser.parse_args()
    main(**vars(args))
