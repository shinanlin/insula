"""Refresh active condition comparisons and RT reporting after author exclusion."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import nbformat as nbf

ROOT=Path('/cwork/ns458/insula-paper')
sys.path.insert(0,str(ROOT))
AUDIT=ROOT/'results/audit/exclude_unassigned_2026-11-01'


def archive(path):
    dest=AUDIT/'before'/path.relative_to(ROOT)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copy2(path,dest)


def imports():
    return nbf.v4.new_code_cell("import sys\nsys.path.insert(0,'/cwork/ns458/insula-paper')\nfrom IPython.display import SVG,display\nimport pandas as pd\n")


def cached_svg(source,path):
    cell=nbf.v4.new_code_cell(source)
    cell.outputs=[nbf.v4.new_output('display_data',data={'image/svg+xml':Path(path).read_text(),'text/plain':'<IPython.core.display.SVG object>'},metadata={})]
    return cell


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    from scripts.refresh_lexical_condition_sets import sha,write
    from src.main_analysis import FUNCTIONAL_COMPONENTS,COHORT_DIR
    from src.supplementary.assigned_condition_cohorts import OUT,load_cohort
    from src.supplementary.assigned_comparison_traces import render_picture,render_lexical
    from src.supplementary.assigned_condition_overviews import render as overview
    if os.environ.get('ASSIGNED_FINISH_REPORTS_ONLY')=='1':
        for key in ('picture','lexical'):
            assert json.loads((OUT/key/'overview/completion.json').read_text())['passed']
        paths=[ROOT/'vizpub/supplementary'/n for n in ('sound_image_test.ipynb','decision_repeat_test.ipynb','fig3_picture_stimulus_condition_sets.ipynb','fig3_lexical_delay_condition_sets.ipynb')]
        return finish_reports(pd.read_csv(OUT/'cohort_counts.csv').to_dict('records'),paths)
    for folder in ('img/supplementary/sound_image_test','img/supplementary/decision_repeat_test',
        'img/supplementary/fig3_picture_stimulus_condition_sets','img/supplementary/fig3_lexical_delay_condition_sets','img/fig5'):
        for path in (ROOT/folder).glob('*.svg'):archive(path)
    owned=[ROOT/'vizpub/supplementary'/n for n in ('sound_image_test.ipynb','decision_repeat_test.ipynb','fig3_picture_stimulus_condition_sets.ipynb','fig3_lexical_delay_condition_sets.ipynb')]
    hashes={p:sha(p) for p in owned}
    stats=[];subject_rows=[]
    for key in ('picture','lexical'):
        t,parent=load_cohort(key)
        fields=('sound_responsive','image_responsive') if key=='picture' else ('decision_responsive','repeat_responsive')
        t['response_union']=t[fields[0]]|t[fields[1]]
        for subject,part in t.groupby('subject'):
            subject_rows.append(dict(comparison=key,subject=subject,assigned_n=len(part),responsive_n=int(part.response_union.sum()),
                assigned_channels=';'.join(part.channel),responsive_channels=';'.join(part.loc[part.response_union,'channel'])))
        stats.append(dict(comparison=key,anatomical_inventory_n=len(parent),assigned_n=len(t),assigned_subjects=t.subject.nunique(),
            responsive_n=int(t.response_union.sum()),responsive_subjects=t.loc[t.response_union,'subject'].nunique()))
    pd.DataFrame(subject_rows).to_csv(OUT/'subject_task_counts_and_lists.csv',index=False)
    pd.DataFrame(stats).to_csv(OUT/'cohort_counts.csv',index=False)
    # The same render functions appear in runnable notebook cells. The already
    # generated SVG is attached as the output; no figures are recomputed just
    # to populate duplicate notebook outputs.
    picture_paths={key:render_picture(key) for key in ('sound_gt_image','image_gt_sound','no_difference_sound_minus_image','no_difference_image_minus_sound')}
    lexical_path=render_lexical()
    pn=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# Picture Naming · assigned-only Sound/Image comparisons

Author-approved 2026-11-01: exclude functional Unassigned from every statistical denominator and displayed group. Anatomical inventory128/27 is coverage metadata; assigned cohort81/24; either-condition Stimulus[0,1] response pool44/21. Upper curves and SEM use these44 electrodes. Original5000-draw subject-block time_perm_cluster bars have been recomputed for44, with unchanged .05/.05 settings and original phase windows (Stimulus upper inference remains[-.5,.6]).

Lower selections use unchanged original per-electrode2000-draw two-sided .05/.05 time_perm_cluster: Sound>Image19 (includes bidirectional1), Image>Sound5 (includes the same1), no detected difference21 (two signed displays). Every lower bin retains its original own-phase/direction condition-difference mask, unchanged values, row-order subset,1-mm rows,proportional axes and0–.5 mapping. No new per-electrode test or NMF fit. “No detected difference” is not equivalence. The current mutually exclusive classification and brain maps are in `fig3_directional_difference_venn_draft.ipynb`.

Prior mixed-cohort notebook/SVGs and all old scientific results remain archived. Current source: `src/supplementary/assigned_comparison_traces.py`.
'''),imports(),nbf.v4.new_code_cell('from src.supplementary.assigned_comparison_traces import render_picture\n')])
    for key,path in picture_paths.items():pn.cells += [nbf.v4.new_markdown_cell('## '+key.replace('_',' ')),cached_svg(f"display(SVG(filename=str(render_picture('{key}'))))\n",path)]
    ld=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# Lexical Delay · assigned-only Decision/Repeat comparisons

Author-approved 2026-11-01: remove Unassigned from statistics. Anatomical inventory187/35 is separate coverage metadata. Upper curves use135 assigned electrodes/35 participants, electrode mean±SEM. Recompute their original5000-draw subject-block time_perm_cluster, .05/.05, original separate phase ranges. Lower heatmaps use the29 assigned electrodes from the original full-cohort31 Decision>Repeat set; retain original mean differences, Decision-baseline display mask, row order,1-mm rows,component hues and0–.5 mapping. No new per-electrode tests or NMF fit.

This full-assigned-cohort panel is distinct from the Delay-active71-site classification (27 Decision>Repeat /43 no difference /1 Repeat>Decision) and its27-site temporal portrait in `fig3_directional_difference_venn_draft.ipynb`. Do not mix those denominators. Original full-cohort tests and prior displays remain archived.
'''),imports(),nbf.v4.new_code_cell('from src.supplementary.assigned_comparison_traces import render_lexical\n'),cached_svg('display(SVG(filename=str(render_lexical())))\n',lexical_path)])
    notebooks={owned[0]:pn,owned[1]:ld}
    for key,p in [('picture',owned[2]),('lexical',owned[3])]:
        paths=overview(key)
        note=f'''# {key.capitalize()} condition sets · assigned-only statistical cohort

Current policy (2026-11-01): every displayed count, denominator, participant list, mean/SEM and functional map excludes functionally Unassigned electrodes. Anatomical inventories are retained separately as coverage metadata. Native baseline responsiveness requires the original mask AND pointwise p<.05; original per-electrode differences and NMF assignments remain unchanged.

The literal baseline-response Venn below is separate from the current three-region condition-difference schematic in `fig3_directional_difference_venn_draft.ipynb`; its center means both conditions baseline-responsive. Functional component summaries use absolute electrode counts. Sources and excluded IDs: `results/supplementary/fig3_assigned_only_2026-11-01/{key}/`.

The previous notebook, including historical exploratory weight-map drafts, is preserved under `results/audit/exclude_unassigned_2026-11-01/before/vizpub/supplementary/`. Current canonical statistics supersede its mixed-cohort counts.
'''
        cells=[nbf.v4.new_markdown_cell(note),imports(),nbf.v4.new_code_cell(f"from src.supplementary.assigned_condition_overviews import render\nfrom src.supplementary.assigned_condition_cohorts import OUT\npaths=render('{key}')\ndisplay(pd.read_csv(OUT/'{key}'/'cohort_flow.csv'))\ndisplay(pd.read_csv(OUT/'{key}'/'overview'/'set_counts.csv'))\ndisplay(pd.read_csv(OUT/'{key}'/'overview'/'component_composition.csv'))\n")]
        for name,path in paths.items():cells += [nbf.v4.new_markdown_cell('## '+name.replace('_',' ')),cached_svg(f"display(SVG(filename=paths['{name}']))\n",path)]
        notebooks[p]=nbf.v4.new_notebook(cells=cells)
    for path,nb in notebooks.items():
        assert sha(path)==hashes[path],f'Concurrent author edit: {path}'
        nbf.write(nb,path)
    finish_reports(stats,list(notebooks))


def finish_reports(stats,notebooks):
    from scripts.refresh_lexical_condition_sets import sha,write
    from src.main_analysis import FUNCTIONAL_COMPONENTS,COHORT_DIR
    from src.supplementary.assigned_condition_cohorts import OUT
    # Report-only RT restriction. Do not alter models, shuffle arrays, original
    # across-channel correction families or already-assigned encoding inference.
    from src.reaction_time.summarize_insula_rt_ridge import summarize_results
    coverage,electrodes,clusters=summarize_results(ROOT/'results/rt',assignments_path=ROOT/'results/nmf/channel_assignments.csv')
    before=AUDIT/'before/results/rt/summaries'
    removed=[]
    for name,frame in [('electrodes',electrodes),('significant_clusters',clusters)]:
        old=pd.read_csv(before/f'{name}.csv',float_precision='round_trip')
        kept=old.loc[old.functional_cluster.isin(FUNCTIONAL_COMPONENTS)].copy()
        assert frame.functional_cluster.isin(FUNCTIONAL_COMPONENTS).all()
        serialized=pd.read_csv(ROOT/'results/rt/summaries'/f'{name}.csv',float_precision='round_trip')
        pd.testing.assert_frame_equal(serialized.reset_index(drop=True),kept.reset_index(drop=True),check_dtype=False,check_exact=False,rtol=1e-14,atol=1e-14)
        old.loc[~old.functional_cluster.isin(FUNCTIONAL_COMPONENTS)].to_csv(OUT/f'rt_excluded_unassigned_{name}.csv',index=False)
        removed.append(dict(table=name,before=len(old),after=len(frame),removed=len(old)-len(frame)))
    pd.DataFrame(removed).to_csv(OUT/'rt_count_changes.csv',index=False)
    encoding_checks={}
    for path in (ROOT/'results/rt/summaries').glob('hga_rt_*.csv'):
        frame=pd.read_csv(path)
        if 'functional_cluster' in frame:assert frame.functional_cluster.isin(FUNCTIONAL_COMPONENTS).all()
        encoding_checks[path.name]=sha(path)==sha(before/path.name)
    assert all(encoding_checks.values())
    assignment=pd.read_csv(ROOT/'results/nmf/channel_assignments.csv')
    assert assignment.functional_cluster.isin(FUNCTIONAL_COMPONENTS).all()
    responsive=pd.read_csv(COHORT_DIR/'responsive_sites_by_subject_task.csv')
    main_missing=responsive.loc[~responsive.site_id.isin(assignment.channel)]
    main_missing.to_csv(OUT/'main_responsive_sites_without_assignments.csv',index=False)
    assert main_missing.empty,'Review any main Figure1 responsive/NMF mismatch before changing its curves'
    protected=json.loads((AUDIT/'protected_main.json').read_text())
    assert all(sha(p)==h for p,h in protected.items())
    write(AUDIT/'statistics_completion.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        completed_utc=datetime.now(timezone.utc).isoformat(),cohorts=stats,rt_tables=removed,
        rt_original_model_scores_and_correction_unchanged=True,rt_encoding_already_assigned_and_unchanged=encoding_checks,
        main_responsive_sites_all_assigned=True,main_frozen_files_unchanged=len(protected),
        notebooks=list(map(str,notebooks)),no_new_NMF_or_per_electrode_tests=True))
    print('COMPLETE assigned-only comparisons, overviews and RT statistics',flush=True)


if __name__=='__main__':main()
