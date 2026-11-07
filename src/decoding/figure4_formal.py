"""Versioned Figure 4 workflow: one source specification and one CV engine.

All numerical work is launched through Slurm. Unit results remain separate
from the publication inputs until a complete correction family is validated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import time

import h5py
from joblib import Parallel, delayed
import numpy as np
import scipy
import sklearn
from sklearn.metrics import confusion_matrix

from src.decoding.formal_cv import (
    METHOD, SCHEMA_VERSION, SE_DESCRIPTION, evaluate_labels, inference,
    phase_window, sample_windows, sha256_file, unit_solver_specification,
)
from src.decoding.repeated_cv import whole_stimulus_permutations

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "results/audit/figure4_formal_2026-10-20"
SPEC = AUDIT / "analysis_specification.json"
ROIS = ("Sensory", "Sustain", "Motor")
PHASES = ("Stimulus", "Delay", "Response")
LOG = logging.getLogger(__name__)


def strings(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def verify_protected():
    protected = json.loads((AUDIT / "protected_main_inputs.json").read_text())
    changed = [p for p, digest in protected.items() if sha256_file(p) != digest]
    if changed:
        raise ValueError(f"Protected Figure 1/2 or NMF changed: {changed}")
    return len(protected)


def prepare():
    """Freeze identities, complete input rows, specification and sample windows."""
    verify_protected()
    spec = json.loads((ROOT / "config/figure4_phonetic_decoding.json").read_text())
    declared_amendments = spec.pop("solver_overrides", [])
    assert spec["inference"] == METHOD and spec["n_permutations"] == 500
    assert spec["n_repeats"] == 30 and spec["n_folds"] == 5
    if SPEC.exists() and json.loads(SPEC.read_text()) != spec:
        raise ValueError("The frozen run specification already differs")
    if (AUDIT/"run_plan.json").exists():
        existing, _ = load_plan()
        fields = ("mode", "roi", "phase", "max_iter", "approved_date", "reason")
        recorded = [{k:a[k] for k in fields} for a in existing.get("solver_amendments", [])]
        if declared_amendments != recorded:
            raise ValueError("Source solver amendments do not match their frozen approvals")
        return existing
    if declared_amendments:
        raise ValueError("Solver amendments require a frozen base run and an approval record")
    write_json(SPEC, spec)
    models = []
    for roi in ROIS:
        for phase in PHASES:
            path = ROOT / f"results/decoding_inputs/PhonemeSequence/sub-{roi}/articulator/sub-{roi}_task-PhonemeSequence_proc-{phase}_recording-1_desc-Repeat_highgamma.h5"
            with h5py.File(path, "r") as f:
                channels = strings(f["channel"][:])
                trials = strings(f["trial"][:])
                shape = list(f["X"].shape)
                fs, origin = float(f.attrs["fs"]), float(f.attrs["tmin"])
                assignment_sha = str(f.attrs["assignment_sha256"])
            exclusions = [x["channel"] for x in spec["analysis_exclusions"]
                if x["roi"] == roi and x["phase"] == phase]
            if not set(exclusions).issubset(channels):
                raise ValueError("Approved exclusion is absent from full reference")
            kept = [c for c in channels if c not in exclusions]
            old = AUDIT / f"before/results/decoding/PhonemeSequence/sub-{roi}/(decode)articulator/sub-{roi}_proc-{phase}_recording-1_desc-Repeat_highgamma.h5"
            with h5py.File(old, "r") as f:
                assert channels == strings(f["input_channel"][:])
                assert trials == strings(f["trial"][:])
                assert kept == strings(f["channel"][:])
            if roi == "Sustain" and phase == "Delay":
                assert len(trials) == 203 and len(kept) == 13
            models.append(dict(index=len(models), roi=roi, phase=phase,
                role="primary", path=str(path), input_sha256=sha256_file(path),
                channels=channels, kept_channels=kept, excluded_channels=exclusions,
                n_trials=len(trials), n_electrodes=len(kept),
                n_subjects=len({c.split("_")[0] for c in kept}),
                trial_sha256=__import__("hashlib").sha256(json.dumps(trials).encode()).hexdigest(),
                shape=shape, fs=fs, origin=origin, assignment_sha256=assignment_sha,
                stage=phase_window(origin, fs, shape[-1], spec["phase_windows"][phase]),
                resolved=sample_windows(origin, fs, shape[-1],
                    **{k:spec["time_resolved"][k] for k in ("tmin","tmax","window","step")}),
                previous_result=str(old)))
    sensitivity = dict(models[4], index=9, role="sensitivity_full14",
        kept_channels=models[4]["channels"], excluded_channels=[],
        n_electrodes=14,
        n_subjects=len({c.split("_")[0] for c in models[4]["channels"]}))
    models.append(sensitivity)
    n_windows = len(models[0]["resolved"])
    assert all(len(m["resolved"]) == n_windows for m in models)
    assert all(m["fs"] == models[0]["fs"] for m in models)
    plan = dict(approved_date=spec["approved_date"], specification=str(SPEC),
        specification_sha256=sha256_file(SPEC), models=models,
        stage_units=10, primary_stage_units=9, resolved_units=9*n_windows,
        windows_per_curve=n_windows, output=str(ROOT/spec["versioned_output"]),
        environment=dict(python=platform.python_version(), numpy=np.__version__,
            scipy=scipy.__version__, sklearn=sklearn.__version__),
        protected_count=verify_protected())
    write_json(AUDIT/"run_plan.json", plan)
    print(json.dumps({k:v for k,v in plan.items() if k!="models"},indent=2),flush=True)
    return plan


def load_plan():
    plan = json.loads((AUDIT/"run_plan.json").read_text())
    if sha256_file(SPEC) != plan["specification_sha256"]:
        raise ValueError("Frozen specification was modified")
    for amendment in plan.get("solver_amendments", []):
        if sha256_file(amendment["approval_file"]) != amendment["approval_sha256"]:
            raise ValueError("Approved solver amendment was modified")
        frozen = json.loads(Path(amendment["approval_file"]).read_text())
        expected = {k:v for k,v in amendment.items() if k not in ("approval_file", "approval_sha256")}
        if frozen != expected or frozen["base_specification_sha256"] != plan["specification_sha256"]:
            raise ValueError("Solver amendment does not match its approved base run")
    return plan, json.loads(SPEC.read_text())


def unit_identity(mode, index, plan):
    if mode == "stage":
        model, window = plan["models"][index], None
    else:
        mi, wi = divmod(index, plan["windows_per_curve"])
        if mi >= 9:
            raise ValueError("Resolved primary model index is outside the family")
        model, window = plan["models"][mi], plan["models"][mi]["resolved"][wi]
    bounds = model["stage"] if mode == "stage" else window
    name = f"{model['roi']}_{model['phase']}"
    if mode == "stage":
        name += "_full14" if model["role"] != "primary" else ""
        output = Path(plan["output"])/"stage/raw"/f"{name}.h5"
    else:
        output = Path(plan["output"])/"resolved/units"/name/f"window_{window['index']:03d}.h5"
    return model, bounds, output


def fit_unit(mode, index, *, n_jobs, preflight_only=False):
    """Resume complete global draws; never publish partial/mixed inference."""
    plan, spec = load_plan()
    model, bounds, output = unit_identity(mode, index, plan)
    spec, solver_amendment = unit_solver_specification(spec, plan, model, mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    status = output.with_suffix(".status.json")
    pending = output.with_suffix(".pending.h5")
    identity = dict(config_sha256=plan["specification_sha256"],
        input_sha256=model["input_sha256"], mode=mode, index=index,
        first=bounds["first"], last=bounds["last"], channels=model["kept_channels"])
    if solver_amendment is not None:
        identity.update(solver_amendment_sha256=solver_amendment["approval_sha256"],
            effective_max_iter=spec["max_iter"])
    identity_text = json.dumps(identity, sort_keys=True)
    if output.exists():
        with h5py.File(output,"r") as f:
            assert f.attrs["identity"] == identity_text and f.attrs["complete"]
        LOG.info("Already complete: %s",output)
        return
    if sha256_file(model["path"]) != model["input_sha256"]:
        raise ValueError("Frozen single-trial input changed")
    with h5py.File(model["path"],"r") as f:
        X = f["X"][:,:,bounds["first"]:bounds["last"]]
        y = f["y"][:]
        trials, conditions = strings(f["trial"][:]), strings(f["condition"][:])
    keep = [model["channels"].index(c) for c in model["kept_channels"]]
    groups = np.asarray([t.rsplit("_",1)[0] for t in trials])
    null_plan = whole_stimulus_permutations(y, groups, spec["n_permutations"],spec["random_state"])
    started = time.monotonic()
    nr,nf,np_ = spec["n_repeats"],spec["n_folds"],spec["n_permutations"]
    state = dict(mode=mode,index=index,roi=model["roi"],phase=model["phase"],
        role=model["role"],output=str(output),job_id=os.environ.get("SLURM_JOB_ID"),
        array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),max_iter=spec["max_iter"])
    f = None
    try:
        f = h5py.File(pending,"a")
        if "identity" in f.attrs:
            assert f.attrs["identity"] == identity_text,"Partial result belongs to a different analysis"
        else:
            f.attrs.update(identity=identity_text,complete=False,inference_method=METHOD,
                result_schema_version=SCHEMA_VERSION,config_sha256=plan["specification_sha256"],
                fs=model["fs"],tmin=bounds["nominal_start"],tmax=bounds["nominal_end"],
                time=bounds["time"],actual_start=bounds["start"],actual_end_exclusive=bounds["end_exclusive"],
                sample_first=bounds["first"],sample_last_exclusive=bounds["last"],
                variance=spec["variance"],C=spec["C"],max_iter=spec["max_iter"],tol=spec["tol"],
                n_perm=np_,n_folds=nf,n_repeats=nr,cv_random_state=spec["random_state"],
                n_electrodes=model["n_electrodes"],n_subjects=model["n_subjects"],n_trials=len(y),
                input_path=model["path"],input_sha256=model["input_sha256"],
                assignment_sha256=model["assignment_sha256"],role=model["role"],roi=model["roi"],phase=model["phase"],
                errorbar_definition=SE_DESCRIPTION,
                accuracy_unit="CV-repeat mean of fold balanced accuracies",
                p_value_statistic="mean across all CV repeats and folds",
                cv_split_policy="re-stratify separately for observed and each globally relabeled target with the same seed table",
                transform_policy="refit StandardScaler and PCA inside each draw's training fold",
                permutation_unit="whole stimulus blocks within equal retained-repeat-count strata",
                exclusion_stage="after full-reference CV and original imputation, before training transform")
            if solver_amendment is not None:
                f.attrs["solver_amendment_sha256"] = solver_amendment["approval_sha256"]
                f.attrs["solver_amendment_file"] = solver_amendment["approval_file"]
            st = h5py.string_dtype("utf-8")
            for name, values in dict(channel=model["kept_channels"],input_channel=model["channels"],
                    excluded_channel=model["excluded_channels"],trial=trials,condition=conditions).items():
                f.create_dataset(name,data=np.asarray(values,dtype=st))
            f.create_dataset("classes",data=np.unique(y))
            for name,values in null_plan.items():
                if name=="group_names":values=np.asarray(values,dtype=st)
                f.create_dataset("permutation_plan/"+name,data=values)
            f.create_dataset("perm_scores",shape=(np_,nr,nf),dtype="f8",fillvalue=np.nan)
            f.create_dataset("permutation_iterations",shape=(np_,nr,nf),dtype="i4",fillvalue=-1)
            f.create_dataset("permutation_pca_component_counts",shape=(np_,nr,nf),dtype="i2",fillvalue=-1)
            f.create_dataset("permutation_fold_assignment",shape=(np_,nr,len(y)),dtype="u1",
                chunks=(1,nr,len(y)),compression="gzip",fillvalue=255)
            f.create_dataset("permutation_imputation_sha256",shape=(np_,),dtype="S64")
            f.create_dataset("permutation_completed",data=np.zeros(np_,dtype=bool))
            f.flush()
        if "accuracy_repeats" not in f:
            write_json(status,dict(state,status="observed_fits",completed_permutations=0))
            observed = evaluate_labels(X,y,keep,spec)
            f.create_dataset("accuracy_repeats",data=observed["scores"])
            for key,value in observed.items():
                if key=="scores":continue
                if key=="imputation_sha256":f.attrs["observed_imputation_sha256"]=value
                else:f.create_dataset("observed_"+key,data=value)
            classes = np.unique(y)
            predictions=observed["first_repeat_predictions"]
            f.create_dataset("first_repeat_predictions",data=predictions)
            f.create_dataset("confusion",data=confusion_matrix(y,predictions,labels=classes))
            f.create_dataset("confusion_norm",data=confusion_matrix(y,predictions,labels=classes,normalize="true"))
            # Two unchanged, previously converged Response models must still
            # reproduce every observed score; other windows intentionally change.
            if mode=="stage" and model["role"]=="primary" and model["phase"]=="Response" and model["roi"]!="Motor":
                with h5py.File(model["previous_result"],"r") as previous:
                    assert np.array_equal(observed["scores"],previous["accuracy_repeats"][:]),"Unchanged converged Response reference failed"
                f.attrs["unchanged_observed_reference_identical"]=True
            f.flush()
        def draw(index):
            try:
                return index,evaluate_labels(X,null_plan["labels"][index],keep,spec)
            except Exception as exc:
                raise type(exc)(f"null_draw={index}: {exc}") from exc
        def run_draws(indices):
            if not indices:return
            for pi,value in Parallel(n_jobs=n_jobs,prefer="threads",batch_size=1,
                    pre_dispatch="n_jobs",return_as="generator_unordered")(
                        delayed(draw)(pi) for pi in indices):
                f["perm_scores"][pi]=value["scores"]
                f["permutation_iterations"][pi]=value["iterations"]
                f["permutation_pca_component_counts"][pi]=value["components"]
                f["permutation_fold_assignment"][pi]=value["fold_assignment"]
                f["permutation_imputation_sha256"][pi]=value["imputation_sha256"].encode()
                f["permutation_completed"][pi]=True
                done=int(f["permutation_completed"][:].sum())
                f.flush()
                if done%10==0 or done==np_:
                    elapsed=time.monotonic()-started
                    LOG.info("%s %s %s: %d/%d global draws, %.1fs",mode,model["roi"],model["phase"],done,np_,elapsed)
                    write_json(status,dict(state,status="permutations",completed_permutations=done,elapsed_seconds=elapsed))
        initial=spec["preflight_permutations"]
        run_draws([i for i in range(initial) if not f["permutation_completed"][i]])
        f.attrs["preflight_passed"]=True
        f.flush()
        write_json(status,dict(state,status="preflight_passed",completed_permutations=int(f["permutation_completed"][:].sum()),
            observed_max_iterations=int(f["observed_iterations"][:].max()),
            preflight_null_max_iterations=int(f["permutation_iterations"][:initial].max()),
            elapsed_seconds=time.monotonic()-started))
        if preflight_only:
            LOG.info("Preflight passed: %s",output)
            return
        run_draws([i for i in range(initial,np_) if not f["permutation_completed"][i]])
        assert f["permutation_completed"][:].all()
        result=inference(f["accuracy_repeats"][:],f["perm_scores"][:])
        for key,value in result.items():f.create_dataset(key,data=value)
        f["pca_component_counts"]=f["observed_components"]
        f.attrs.update(complete=True,all_fits_converged=True,
            null_test_folds_missing_any_class=0,observed_reference_check="train-only full pipeline; original input identities verified",
            completed_utc=datetime.now(timezone.utc).isoformat())
        f.flush();f.close();f=None
        pending.replace(output)
        write_json(status,dict(state,status="complete",completed_permutations=np_,
            accuracy=result["accuracy_stable"],SE=result["accuracy_se"],p_value=result["p_value"],
            elapsed_seconds=time.monotonic()-started))
    except Exception as exc:
        write_json(status,dict(state,status="failed",error=repr(exc),elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        if f is not None:f.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("prepare","fit"))
    parser.add_argument("--mode",choices=("stage","resolved"),default="stage")
    parser.add_argument("--index",type=int,default=0)
    parser.add_argument("--n-jobs",type=int,default=int(os.environ.get("SLURM_CPUS_PER_TASK","1")))
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    if args.command=="prepare":prepare()
    else:fit_unit(args.mode,args.index,n_jobs=args.n_jobs,preflight_only=args.preflight_only)


if __name__=="__main__":main()
