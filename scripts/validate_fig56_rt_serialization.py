"""Preflight real RT metadata serialization and parallel execution identity."""
from dataclasses import fields
from pathlib import Path
import json
import os
import tempfile

import h5py
import numpy as np

from scripts.refresh_fig56_manual14 import RUN, configure_rt, write
from src.reaction_time.insula_rt_data import load_strict_insula_parcellation, load_phase_data
from src.reaction_time.insula_rt_io import _write_numeric
from src.reaction_time.insula_ridge import fit_window_scores, make_group_splits, make_permutation_seeds
from src.reaction_time.run_insula_rt_ridge import sliding_windows


assert os.environ.get('SLURM_JOB_ID')
configure_rt()
plan=json.loads((RUN/'rt_run_plan.json').read_text())
with tempfile.TemporaryDirectory(dir=RUN) as temp:
    for row in plan['models'][:96]:
        meta=load_strict_insula_parcellation(row['bids_root'],subject=row['subject'],task=row['task'])
        assert meta['mix'].dtype==bool
        with h5py.File(Path(temp)/'metadata.h5','w') as f:
            _write_numeric(f,'mix',meta['mix'],dtype=np.uint8)
            np.testing.assert_array_equal(f['mix'][:],meta['mix'].to_numpy(dtype=np.uint8))
row=plan['models'][25]
data=load_phase_data(row['bids_root'],subject=row['subject'],task=row['task'],phase='Delay')
groups=data.trial_meta.item_id.astype(str).to_numpy()
splits,_=make_group_splits(groups,n_splits=10,random_state=42)
seeds=make_permutation_seeds(len(splits),1000,random_state=42)
starts,stops,*_=sliding_windows(data.times,data.sfreq,window_s=.2,step_s=.02,max_windows=1)
args=(data.X[...,starts[0]:stops[0]],data.trial_meta.rt_log.to_numpy(),groups,splits,seeds)
baseline=fit_window_scores(*args,inner_splits=5,random_state=42,n_jobs=4)
parallel=fit_window_scores(*args,inner_splits=5,random_state=42,n_jobs=12)
for field in fields(baseline):
    np.testing.assert_array_equal(getattr(baseline,field.name),getattr(parallel,field.name))
report=dict(passed=True,metadata_groups_checked=96,boolean_mix_serialization_verified=True,
            validation_subject=row['subject'],validation_task=row['task'],phase='Delay',
            channels=data.X.shape[1],n_perm=1000,outer_folds=10,inner_folds=5,
            compared_workers=[4,12],observed_predictions_scores_and_all_permutations_identical=True,
            numerical_engine_changed=False,job_id=os.environ['SLURM_JOB_ID'])
write(RUN/'RT_serialization_and_parallel_validation.json',report)
print(json.dumps(report,indent=2),flush=True)
