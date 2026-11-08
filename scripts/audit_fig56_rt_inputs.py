"""Verify phase eligibility and source chronology for the frozen RT run plan."""
from pathlib import Path
import json
import os

import h5py
import mne
import pandas as pd

from scripts.refresh_fig56_manual14 import ROOT, RUN, METADATA, COHORT, PHASES, sha, write


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run via Slurm'
    plan_path=RUN/'rt_run_plan_before_trial_restore.json'
    if not plan_path.exists():plan_path=RUN/'rt_run_plan.json'
    plan=json.loads(plan_path.read_text())
    cohort=pd.read_csv(COHORT/'sampled_sites_by_subject_task.csv')
    metadata=pd.read_csv(METADATA)
    sources=pd.read_csv(RUN/'native_RT_epoch_sources.csv').dropna(subset=['native_path'])
    rows=[]
    for model in plan['models'][:96]:
        task,subject=model['task'],model['subject']
        group=cohort[cohort.task.eq(task)&cohort.subject.eq(subject)]
        accepted=set(group.loc[~group.native_QC_excluded & ~group.subject_preprocessing_deferred,'site_id'])
        phase_sets={phase:set(accepted) for phase in PHASES}
        bad=set()
        native=sources[sources.task.eq(task)&sources.subject.eq(subject)]
        for row in native.itertuples():
            ep=mne.read_epochs(row.native_path,preload=False,verbose='ERROR')
            phase_sets[row.phase].intersection_update(ep.ch_names)
            bad.update(ep.info['bads'])
        expected=set.union(*phase_sets.values())-bad
        actual=set(metadata.loc[metadata.task.eq(task)&metadata.subject.eq(subject),'channel'])
        assert expected==actual,(task,subject,'phase eligibility',expected-actual,actual-expected)
        later=[]
        if not model['refit']:
            for original in model['original_models']:
                model_path=Path(original['path'])
                with h5py.File(model_path) as f:
                    paths={Path(x.decode() if isinstance(x,bytes) else x).resolve() for x in f['trials/source_file'][:]}
                later.extend(str(p) for p in paths if p.stat().st_mtime_ns>model_path.stat().st_mtime_ns)
            assert not later,(task,subject,'cache sources postdate fit',later)
        rows.append(dict(task=task,subject=subject,phase_eligible_sites=len(expected),
                         phase_union_matches_frozen_metadata=True,cache_reused=not model['refit'],
                         cache_sources_postdate_model=False))
    pd.DataFrame(rows).to_csv(RUN/'RT_phase_eligibility_and_cache_source_audit.csv',index=False)
    report=dict(passed=True,subject_task_groups=len(rows),
                phase_specific_union_preserved=True,reused_cache_sources_predate_fitting=True,
                cohort_sha256=sha(COHORT/'sampled_sites_by_subject_task.csv'),
                metadata_sha256=sha(METADATA),job_id=os.environ['SLURM_JOB_ID'])
    write(RUN/'RT_phase_eligibility_and_cache_source_audit.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
