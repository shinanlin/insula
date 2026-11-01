"""Verify assigned-only cohorts and exact retention of existing heatmap values."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import pandas as pd
import nbformat

ROOT=Path('/cwork/ns458/insula-paper')
sys.path.insert(0,str(ROOT))


def main():
    assert os.environ.get('SLURM_JOB_ID')
    from scripts.refresh_lexical_condition_sets import sha,write
    from src.main_analysis import FUNCTIONAL_COMPONENTS
    from src.supplementary.assigned_condition_cohorts import OUT
    from src.supplementary.assigned_comparison_traces import OLD_PN,OLD_LD
    audit=ROOT/'results/audit/exclude_unassigned_2026-11-01'
    assert json.loads((audit/'statistics_completion.json').read_text())['passed']
    pairs=[(OLD_PN/f'{p}_cluster_masked_matrices.npz',OUT/'picture/comparison_traces'/f'{p}_matrices.npz')
        for p in ('sound_gt_image','image_gt_sound','no_difference_sound_minus_image','no_difference_image_minus_sound')]
    pairs += [(OLD_LD/'decision_minus_repeat_heatmap_matrices.npz',OUT/'lexical/comparison_traces/decision_minus_repeat_heatmap_matrices.npz')]
    for key in ('picture','lexical'):
        pairs.append((ROOT/'results/supplementary/fig3_condition_time_profiles_2026-11-01'/key/'heatmap_matrices.npz',OUT/'time_profiles'/key/'heatmap_matrices.npz'))
    verified=0
    for previous,current in pairs:
        with np.load(previous,allow_pickle=False) as old,np.load(current,allow_pickle=False) as new:
            assert not any('Unassigned' in k for k in new.files)
            for name in new.files:
                # Three assigned component blocks are unchanged; only the
                # fourth, unassigned block has disappeared.
                np.testing.assert_array_equal(new[name],old[name]);verified+=1
    for key,expected in [('picture',[18,21,4,1]),('lexical',[27,43,1,0])]:
        folder=OUT/'regions'/key
        groups=['first_only','no_detected_difference','second_only','bidirectional']
        counts=pd.read_csv(folder/'group_counts.csv').set_index('group')
        assert counts.loc[groups,'n_electrodes'].tolist()==expected
        pool=pd.read_csv(folder/'responsive_pool_electrodes.csv')
        assert pool.functional_cluster.isin(FUNCTIONAL_COMPONENTS).all()
        selected=[]
        for group in groups:
            t=pd.read_csv(folder/f'{group}_electrodes.csv');selected+=t.electrode.tolist()
            assert t.functional_cluster.isin(FUNCTIONAL_COMPONENTS).all()
        assert len(selected)==len(set(selected))==len(pool) and set(selected)==set(pool.electrode)
        shown=json.loads((folder/'diagram_area_provenance.json').read_text())
        assert shown['passed']
    coverage=pd.read_csv(ROOT/'results/rt/summaries/coverage.csv')
    rt=pd.read_csv(ROOT/'results/rt/summaries/electrodes.csv')
    reported=coverage.set_index(['task','subject','phase']).n_electrodes.sort_index()
    actual=rt.groupby(['task','subject','phase']).size().sort_index()
    np.testing.assert_array_equal(reported.index.to_numpy(),actual.index.to_numpy())
    np.testing.assert_array_equal(reported.to_numpy(),actual.to_numpy())
    notebooks=[ROOT/'vizpub/supplementary'/name for name in ('fig3_directional_difference_venn_draft.ipynb',
        'sound_image_test.ipynb','decision_repeat_test.ipynb','fig3_picture_stimulus_condition_sets.ipynb','fig3_lexical_delay_condition_sets.ipynb')]
    svgs=0
    for path in notebooks:
        nb=nbformat.read(path,as_version=4)
        for cell in nb.cells:
            for output in cell.get('outputs',[]):
                assert output.get('output_type')!='error'
                assert 'image/png' not in output.get('data',{})
                svg=output.get('data',{}).get('image/svg+xml')
                if svg:
                    assert 'Unassigned' not in svg and 'unassigned' not in svg
                    svgs+=1
    protected=json.loads((audit/'protected_main.json').read_text())
    assert all(sha(p)==h for p,h in protected.items())
    write(audit/'validation.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        original_heatmap_arrays_and_masks_identical=verified,matrix_files_checked=len(pairs),
        condition_partitions_exhaustive_and_disjoint=True,rt_coverage_matches_assigned_rows=True,
        no_unassigned_legend_or_rows_in_active_notebook_svg_outputs=svgs,
        protected_main_files_unchanged=len(protected)))
    print('PASS: assigned-only statistics; all retained heatmap arrays/masks identical;',verified,'arrays;',svgs,'SVG notebook outputs',flush=True)


if __name__=='__main__':main()
