"""Assigned-only analysis views of the completed, native-audited Figure 3 tests."""
from pathlib import Path
import json
import pandas as pd
from src.main_analysis import assigned_functional_sites

ROOT=Path('/cwork/ns458/insula-paper')
OUT=ROOT/'results/supplementary/fig3_assigned_only_2026-11-01'


def load_cohort(key):
    from scripts.refresh_lexical_condition_sets import sha,write
    folder=ROOT/'results/audit'/f"{ 'picture_stimulus' if key=='picture' else 'lexical_delay'}_responsive_union_2026-11-01"
    assert json.loads((folder/'completion.json').read_text())['passed']
    source=folder/'all_eligible_electrodes_reconstructed.csv'
    original=pd.read_csv(source)
    table=assigned_functional_sites(original)
    conditions=('sound','image') if key=='picture' else ('decision','repeat')
    for condition in conditions:
        table[condition+'_responsive']=table[condition+'_native_mask_p05_any'].astype(bool)
    a,b=conditions
    table['baseline_class']='Neither'
    table.loc[table[a+'_responsive']&~table[b+'_responsive'],'baseline_class']=a.capitalize()+' only'
    table.loc[table[b+'_responsive']&~table[a+'_responsive'],'baseline_class']=b.capitalize()+' only'
    table.loc[table[a+'_responsive']&table[b+'_responsive'],'baseline_class']='Both'
    folder_out=OUT/key;folder_out.mkdir(parents=True,exist_ok=True)
    table.to_csv(folder_out/'eligible_assigned_electrodes.csv',index=False)
    excluded=original.loc[~original.electrode.isin(table.electrode)].copy()
    excluded['exclusion_reason']='No frozen Sensory/Sustained/Motor functional assignment'
    excluded.to_csv(folder_out/'excluded_unassigned_electrodes.csv',index=False)
    response=table[a+'_responsive']|table[b+'_responsive']
    before_response=original['pool_mask_p05' if key=='picture' else 'response_union_mask_p05']
    excluded.loc[excluded.electrode.isin(original.loc[before_response,'electrode'])].to_csv(
        folder_out/'excluded_unassigned_responsive_electrodes.csv',index=False)
    stages=[('Anatomical paired inventory',original),('Assigned statistical cohort',table),
        ('Either-condition active assigned pool',table.loc[response])]
    pd.DataFrame([dict(stage=label,n_sites=len(frame),n_participants=frame.subject.nunique())
        for label,frame in stages]).to_csv(folder_out/'cohort_flow.csv',index=False)
    write(folder_out/'policy_provenance.json',dict(passed=True,policy='Exclude functional Unassigned from statistical denominators, counts, curves, SEM, selections and component summaries',
        source=str(source),source_sha256=sha(source),anatomical_inventory=len(original),
        assigned_n=len(table),assigned_subjects=table.subject.nunique(),responsive_n=int(response.sum()),
        responsive_subjects=table.loc[response,'subject'].nunique(),unassigned_excluded=len(excluded),
        baseline_rule='Native mask AND pointwise p<.05',no_per_electrode_tests_rerun=True,
        no_NMF_refit_or_reassignment=True,anatomical_unknown_policy_unchanged=True))
    return table,original
