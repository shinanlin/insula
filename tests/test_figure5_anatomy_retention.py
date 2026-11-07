import json
from pathlib import Path
import pandas as pd
import pytest
from src.figure5_anatomy_review import validate_anatomy_review, sha

def fixture_files(tmp_path):
    cohort=tmp_path/"cohort.csv";cohort.write_text("site_id\nD0001_AI1-2\n")
    parcellation=tmp_path/"parcellation.csv";parcellation.write_text("name,center\nD0001_AI1-2,0\n")
    comparison=tmp_path/"comparison.csv"
    row={"task":"Task","subject":"D0001","site_id":"D0001_AI1-2","review_required":True,
         "review_reason":"midpoint_not_Hammers_insula","frozen_midpoint_label":"insula","new_midpoint_label":"0"}
    pd.DataFrame([row]).to_csv(comparison,index=False)
    record={k:row[k] for k in ["task","subject","site_id","review_reason","frozen_midpoint_label","new_midpoint_label"]}
    record.update(decision="retain",comparison_sha256=sha(comparison),parcellation_CSV=str(parcellation),parcellation_sha256=sha(parcellation))
    decision=tmp_path/"decision.json"
    decision.write_text(json.dumps({"decision":"retain_frozen_cohort","cohort_path":str(cohort),"cohort_sha256":sha(cohort),"records":[record]}))
    return comparison,decision,parcellation,cohort

def test_exact_reviewed_record_passes_without_relabeling(tmp_path):
    comparison,decision,parcellation,cohort=fixture_files(tmp_path)
    before=(sha(comparison),sha(parcellation),sha(cohort))
    result=validate_anatomy_review(comparison,decision)
    assert result["status"]=="author_retained"
    assert result["retained_sites"]==["D0001_AI1-2"]
    assert (sha(comparison),sha(parcellation),sha(cohort))==before

@pytest.mark.parametrize("change",["new_site","new_reason","new_CSV","new_cohort"])
def test_unreviewed_or_changed_evidence_still_blocks(tmp_path,change):
    comparison,decision,parcellation,cohort=fixture_files(tmp_path)
    frame=pd.read_csv(comparison,keep_default_na=False)
    if change=="new_site":frame["site_id"]="D0001_AI2-3";frame.to_csv(comparison,index=False)
    if change=="new_reason":frame["review_reason"]="AI1:new_MAPER_CSF";frame.to_csv(comparison,index=False)
    if change=="new_CSV":parcellation.write_text("changed")
    if change=="new_cohort":cohort.write_text("changed")
    with pytest.raises(RuntimeError):
        validate_anatomy_review(comparison,decision)

