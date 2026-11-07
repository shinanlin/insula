"""Publish stage BH tests and the original time-permutation temporal masks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src.decoding.formal_cv import METHOD, inference, sha256_file, unit_solver_specification
from src.decoding.figure4_formal import (
    ROOT, AUDIT, load_plan, strings, unit_identity, verify_protected, write_json,
)
from src.decoding.windowed_results import load_windowed_result
from src.decoding.run_decoding_resolved import cluster_correction


def resolved_policy():
    """Reject any replacement of the author's original temporal test."""
    policy = json.loads((ROOT / 'config/figure4_phonetic_decoding.json').read_text())['multiple_comparisons']
    assert policy['resolved_method'] == 'ieeg_time_cluster'
    assert policy['resolved_function'] == 'src.decoding.run_decoding_resolved.cluster_correction'
    assert policy['resolved_p_thresh'] == .05 and policy['resolved_tails'] == 1
    return policy


def verify_inputs(plan):
    for path,digest in {m['path']:m['input_sha256'] for m in plan['models']}.items():
        if sha256_file(path)!=digest:
            raise ValueError(f'Frozen single-trial input changed: {path}')


def validate_unit(path, model, bounds, plan, spec, mode="stage"):
    spec, amendment = unit_solver_specification(spec, plan, model, mode)
    loaded=load_windowed_result(path,expected_method=METHOD)
    assert loaded["n_repeats"]==30 and loaded["n_folds"]==5 and loaded["n_permutations"]==500
    with h5py.File(path,"r") as f:
        assert f.attrs["complete"] and f.attrs["all_fits_converged"]
        assert f.attrs["config_sha256"]==plan["specification_sha256"]
        assert int(f.attrs["max_iter"]) == spec["max_iter"]
        assert str(f.attrs.get("solver_amendment_sha256", "")) == (amendment["approval_sha256"] if amendment else "")
        assert f.attrs["input_sha256"]==model["input_sha256"]
        assert strings(f["channel"][:])==model["kept_channels"]
        assert strings(f["input_channel"][:])==model["channels"]
        assert len(f["trial"])==model["n_trials"]
        assert f.attrs["sample_first"]==bounds["first"] and f.attrs["sample_last_exclusive"]==bounds["last"]
        assert f["permutation_completed"][:].all()
        true_iter=f["observed_iterations"][:]
        null_iter=f["permutation_iterations"][:]
        assert np.all((true_iter>0)&(true_iter<spec["max_iter"]))
        assert np.all((null_iter>0)&(null_iter<spec["max_iter"]))
        assert not np.any(f["observed_fold_assignment"][:]==255)
        assert not np.any(f["permutation_fold_assignment"][:]==255)
        loaded.update(max_observed_iterations=int(true_iter.max()),max_null_iterations=int(null_iter.max()),
            max_iter=spec["max_iter"])
    return loaded


def require_complete(mode, plan):
    count=plan["stage_units"] if mode=="stage" else plan["resolved_units"]
    missing=[]
    for i in range(count):
        _,_,path=unit_identity(mode,i,plan)
        if not path.is_file():
            receipt=path.with_suffix(".status.json")
            missing.append(dict(index=i,path=str(path),
                status=json.loads(receipt.read_text()) if receipt.exists() else "not started"))
    if missing:
        write_json(AUDIT/f"{mode}_incomplete.json",dict(complete=False,missing=missing))
        raise ValueError(f"{mode}: {len(missing)}/{count} units incomplete; see {AUDIT}/{mode}_incomplete.json")


def primary_path(model):
    return ROOT/f"results/decoding/PhonemeSequence/sub-{model['roi']}/(decode)articulator/sub-{model['roi']}_proc-{model['phase']}_recording-1_desc-Repeat_highgamma.h5"


def validate_stage():
    """Publication gate: all nine files and their common FDR family must match."""
    plan,spec=load_plan()
    verify_protected()
    verify_inputs(plan)
    values=[]
    for model in plan["models"][:9]:
        path=primary_path(model)
        value=validate_unit(path,model,model["stage"],plan,spec)
        assert value["correction_family_size"]==9
        values.append(value)
    expected=multipletests([v["p_value"] for v in values],alpha=.05,method="fdr_bh")[1]
    np.testing.assert_array_equal(expected,[v["p_value_fdr"] for v in values])
    return values


def collect_stage():
    plan,spec=load_plan()
    verify_protected()
    verify_inputs(plan)
    require_complete("stage",plan)
    rows=[]
    for model in plan["models"]:
        _,bounds,path=unit_identity("stage",model["index"],plan)
        v=validate_unit(path,model,bounds,plan,spec)
        rows.append(dict(roi=model["roi"],phase=model["phase"],role=model["role"],
            n_sites=v["n_electrodes"],n_subjects=v["n_subjects"],n_pooled_trials=v["n_trials"],
            mean_accuracy=v["mean"],SE=v["SE"],p_value=v["p_value"],
            max_iter=v["max_iter"],
            max_observed_iterations=v["max_observed_iterations"],max_null_iterations=v["max_null_iterations"],
            tmin=spec["phase_windows"][model["phase"]][0],tmax=spec["phase_windows"][model["phase"]][1],
            source=str(path)))
    corrected=multipletests([r["p_value"] for r in rows[:9]],alpha=.05,method="fdr_bh")[1]
    for row,q in zip(rows[:9],corrected):
        row.update(p_value_fdr=float(q),significant_fdr=bool(q<.05),correction_family_size=9)
        with h5py.File(row["source"],"r+") as f:
            if "p_value_fdr" in f:del f["p_value_fdr"]
            f.create_dataset("p_value_fdr",data=q)
            f.attrs.update(correction_family_size=9,correction="fdr_bh",correction_alpha=.05,
                correction_family="all nine primary phonetic-category cluster-by-phase stage tests")
    # The 13/14 comparison must use identical full-reference CV and numeric
    # imputation for every observed and permuted fit, before channel removal.
    with h5py.File(rows[4]["source"],"r") as a,h5py.File(rows[9]["source"],"r") as b:
        for key in ("trial","permutation_plan/labels","observed_fold_assignment",
                "permutation_fold_assignment","permutation_imputation_sha256"):
            np.testing.assert_array_equal(a[key][:],b[key][:])
        assert a.attrs["observed_imputation_sha256"]==b.attrs["observed_imputation_sha256"]
    destination=ROOT/"results/fig4"
    summary=pd.DataFrame(rows[:9])
    summary.to_csv(Path(plan["output"])/"stage/summary.csv",index=False)
    pd.DataFrame([rows[4],rows[9]]).to_csv(Path(plan["output"])/"stage/sustain_delay_sensitivity.csv",index=False)
    publication=[]
    for model,row in zip(plan["models"][:9],rows[:9]):
        target=primary_path(model)
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(".new.h5")
        shutil.copy2(row["source"],temporary)
        temporary.replace(target)
        publication.append(dict(path=str(target),sha256=sha256_file(target)))
    summary.to_csv(destination/"phonetic_category_summary.csv",index=False)
    pd.DataFrame([rows[4],rows[9]]).to_csv(destination/"phonetic_sustain_delay_sensitivity.csv",index=False)
    counts=[];members=[];repeats=[]
    for model in plan["models"][:9]:
        counts.append(dict(task="PhonemeSequence",roi=model["roi"],phase=model["phase"],
            n_sites=model["n_electrodes"],n_subjects=model["n_subjects"],n_pooled_trials=model["n_trials"],
            n_repeats=30,n_folds=5,n_permutations=500,inference_method=METHOD))
        for c in model["channels"]:
            members.append(dict(task="PhonemeSequence",roi=model["roi"],phase=model["phase"],
                subject=c.split("_")[0],channel=c,in_model=c in model["kept_channels"]))
        value=load_windowed_result(primary_path(model),expected_method=METHOD)
        for r,score in enumerate(value["scores"]):
            repeats.append(dict(roi=model["roi"],phase=model["phase"],repeat=r,seed=42+r,accuracy=score))
    pd.DataFrame(counts).to_csv(destination/"PhonemeSequence_model_counts.csv",index=False)
    pd.DataFrame(members).to_csv(destination/"PhonemeSequence_model_membership.csv",index=False)
    pd.DataFrame(repeats).to_csv(destination/"phonetic_category_repeat_scores.csv",index=False)
    (destination/"phonetic_decoding_caption.txt").write_text(
        "Balanced accuracy is the mean of 30 stratified five-fold CV repeats. "
        "SE across repeat means reflects split variability conditional on this dataset, not participant uncertainty. "
        "Each of 500 whole-stimulus permutations repeats label-dependent CV and train-only scaling/PCA. "
        "Stars indicate BH-FDR q<0.05 across nine primary tests. "
        "Stimulus [0,0.6), Delay [0,1), Response [0,1) s. "
        "Sustain–Delay uses 13 electrodes after diagnostic exclusion of D0084_RAI4-5; "
        "the full14 sensitivity is reported separately, and post-selection inference remains a limitation.\n")
    validate_stage()
    receipt=dict(complete=True,publication=publication,fdr_method="fdr_bh",family_size=9,
        paired_full14_sensitivity_verified=True,all_fits_converged=True,
        protected_count=verify_protected(),completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(AUDIT/"stage_publication.json",receipt)
    print(summary.to_string(index=False),flush=True)


def collect_resolved():
    plan,spec=load_plan()
    verify_protected();verify_inputs(plan);require_complete("resolved",plan)
    policy=resolved_policy()
    rows=[]
    for index in range(plan["resolved_units"]):
        model,bounds,path=unit_identity("resolved",index,plan)
        v=validate_unit(path,model,bounds,plan,spec,mode="resolved")
        rows.append(dict(roi=model["roi"],phase=model["phase"],window_index=bounds["index"],
            time=bounds["time"],actual_start=bounds["start"],actual_end_exclusive=bounds["end_exclusive"],
            mean_accuracy=v["mean"],SE=v["SE"],p_value=v["p_value"],n_sites=v["n_electrodes"],
            n_subjects=v["n_subjects"],n_trials=v["n_trials"],source=str(path),
            max_observed_iterations=v["max_observed_iterations"],max_null_iterations=v["max_null_iterations"]))
    for row in rows:
        row.update(significant_timeperm=False,correction='ieeg_time_cluster',
                   correction_family_size=plan['windows_per_curve'])
    summary=pd.DataFrame(rows)
    curves=Path(plan["output"])/"resolved/curves";curves.mkdir(parents=True,exist_ok=True)
    publication=[]
    for model in plan["models"][:9]:
        subset=summary[(summary.roi==model["roi"])&(summary.phase==model["phase"])].sort_values("window_index")
        assert len(subset)==plan["windows_per_curve"]
        target=curves/f"{model['roi']}_{model['phase']}.h5"
        temp=target.with_suffix(".pending.h5")
        with h5py.File(temp,"w") as f:
            f.attrs.update(complete=True,inference_method=METHOD,config_sha256=plan["specification_sha256"],
                n_repeats=30,n_folds=5,n_perm=500,n_electrodes=model["n_electrodes"],n_trials=model["n_trials"],
                n_subjects=model["n_subjects"],roi=model["roi"],phase=model["phase"],all_fits_converged=True,
                correction="ieeg_time_cluster",correction_alpha=.05,
                correction_family_size=len(subset),
                correction_family="original temporal correction within each complete curve",
                correction_function=policy['resolved_function'],
                cluster_forming_p=policy['resolved_p_thresh'],tails=policy['resolved_tails'],
                errorbar_definition=spec["SE"],window=spec["time_resolved"]["window"],step=spec["time_resolved"]["step"],
                time_axis="actual first/last sample midpoint",input_sha256=model["input_sha256"])
            for col in ("time","actual_start","actual_end_exclusive","mean_accuracy","SE","p_value"):
                f.create_dataset(col,data=subset[col].to_numpy())
            st=h5py.string_dtype("utf-8")
            f.create_dataset("channel",data=np.asarray(model["kept_channels"],dtype=st))
            f.create_dataset("unit_paths",data=np.asarray(subset.source,dtype=st))
            stacked={}
            for key in ("accuracy_repeats","perm_scores"):
                items=[]
                for path in subset.source:
                    with h5py.File(path,"r") as unit:items.append(unit[key][:])
                stacked[key]=np.stack(items)
                f.create_dataset(key,data=stacked[key],compression="gzip")
            # Exactly the original temporal correction; only the already
            # approved matched 30-repeat statistic is supplied from the cache.
            observed=stacked['accuracy_repeats'].mean(axis=(1,2))
            baseline=stacked['perm_scores'].mean(axis=(2,3)).T
            mask,p_values=cluster_correction(observed,baseline,
                p_thresh=policy['resolved_p_thresh'],tails=policy['resolved_tails'])
            np.testing.assert_array_equal(p_values,subset.p_value.to_numpy())
            f.create_dataset('mask',data=np.asarray(mask,dtype=bool))
            f.create_dataset('p_values',data=p_values)
            summary.loc[subset.index,'significant_timeperm']=np.asarray(mask,dtype=bool)
            reference_trials=None;reference_labels=None
            for path in subset.source:
                with h5py.File(path,"r") as unit:
                    if reference_trials is None:
                        reference_trials=unit["trial"][:];reference_labels=unit["permutation_plan/labels"][:]
                        f.create_dataset("trial",data=reference_trials)
                    else:
                        np.testing.assert_array_equal(reference_trials,unit["trial"][:])
                        np.testing.assert_array_equal(reference_labels,unit["permutation_plan/labels"][:])
        temp.replace(target)
        publication.append(dict(path=str(target),sha256=sha256_file(target)))
    summary.to_csv(Path(plan["output"])/"resolved/summary.csv",index=False)
    summary.to_csv(ROOT/"results/fig4/phonetic_time_resolved_summary.csv",index=False)
    write_json(AUDIT/"resolved_publication.json",dict(complete=True,
        n_windows_total=len(rows),n_curves=9,windows_per_curve=plan['windows_per_curve'],
        correction="ieeg_time_cluster",correction_function=policy['resolved_function'],
        p_thresh=.05,tails=1,extra_across_curve_correction=False,
        source_specification=str(ROOT/'config/figure4_phonetic_decoding.json'),
        source_specification_sha256=sha256_file(ROOT/'config/figure4_phonetic_decoding.json'),
        curves=publication,protected_count=verify_protected(),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    print(summary.groupby(["roi","phase"]).agg(windows=("time","size"),significant=("significant_timeperm","sum")),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("stage","resolved","validate-stage"))
    args=parser.parse_args()
    if args.mode=="stage":collect_stage()
    elif args.mode=="resolved":collect_resolved()
    else:validate_stage()


if __name__=="__main__":main()
