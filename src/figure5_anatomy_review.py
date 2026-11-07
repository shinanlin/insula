"""Apply explicit author retention decisions without rewriting anatomical labels."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DECISIONS=ROOT/"config/figure5_anatomy_retention_2026-10-20.json"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_anatomy_review(comparison_path, decision_path=DEFAULT_DECISIONS):
    """Release only the exact task/site conflicts covered by the saved decision."""
    comparison_path=Path(comparison_path)
    frame=pd.read_csv(comparison_path,keep_default_na=False)
    flagged=frame.loc[frame.review_required.astype(str).str.lower().eq("true")]
    if flagged.empty:
        return {"status":"no_conflict","retained_sites":[]}
    if not Path(decision_path).is_file():
        raise RuntimeError("Anatomical retention decision is missing")
    decision=json.loads(Path(decision_path).read_text())
    if decision.get("decision")!="retain_frozen_cohort":
        raise RuntimeError("No approved retain-frozen-cohort decision")
    if sha(decision["cohort_path"])!=decision["cohort_sha256"]:
        raise RuntimeError("Frozen cohort differs from the reviewed retention decision")
    records={(r["task"],r["subject"],r["site_id"]):r for r in decision["records"]}
    retained=[]
    for row in flagged.to_dict("records"):
        key=(str(row["task"]),str(row["subject"]),str(row["site_id"]))
        approved=records.get(key)
        if approved is None:
            raise RuntimeError("New anatomical conflict has no author decision: "+str(key))
        if approved["decision"]!="retain":
            raise RuntimeError("Conflict was not approved for retention: "+str(key))
        for field in ["review_reason","frozen_midpoint_label","new_midpoint_label"]:
            if str(row[field])!=approved[field]:
                raise RuntimeError("Anatomical evidence differs from reviewed decision: "+str(key)+" "+field)
        if sha(comparison_path)!=approved["comparison_sha256"]:
            raise RuntimeError("Anatomy comparison differs from the reviewed file: "+str(key))
        if sha(approved["parcellation_CSV"])!=approved["parcellation_sha256"]:
            raise RuntimeError("Parcellation differs from the reviewed file: "+str(key))
        retained.append(key[2])
    return {"status":"author_retained","retained_sites":sorted(retained),
            "decision_file":str(decision_path),"decision_sha256":sha(decision_path),
            "sampling_membership_changed":False,"anatomical_labels_rewritten":False}

