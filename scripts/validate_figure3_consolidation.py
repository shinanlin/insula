"""Validate main/SI cohort separation and preservation of reviewed values/masks."""
from pathlib import Path
import ast
import hashlib
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
    from src.figure3 import MAIN_RESULTS,SAMPLING_RESULTS,HGA_RESULTS,write,sha
    audit=ROOT/'results/audit/figure3_consolidation_2026-11-01'
    assert json.loads((audit/'execution.json').read_text())['passed']
    main_manifest=json.loads((MAIN_RESULTS/'manifest.json').read_text())
    assert len(main_manifest['svg_sha256'])==11
    assert all(sha(p)==h for p,h in main_manifest['svg_sha256'].items())
    assert {str(p) for p in (ROOT/'img/fig3').rglob('*') if p.is_file()}==set(main_manifest['svg_sha256'])
    old=ROOT/'results/supplementary/fig3_assigned_only_2026-11-01'
    arrays=0
    for key in ('picture','lexical'):
        with np.load(old/'time_profiles'/key/'heatmap_matrices.npz',allow_pickle=False) as before,np.load(MAIN_RESULTS/'time_profiles'/key/'heatmap_matrices.npz',allow_pickle=False) as after:
            assert set(before.files)==set(after.files)
            for name in before.files:np.testing.assert_array_equal(before[name],after[name]);arrays+=1
        for group in ('first_only','no_detected_difference','second_only','bidirectional'):
            a=pd.read_csv(old/'regions'/key/f'{group}_electrodes.csv',float_precision='round_trip')
            b=pd.read_csv(MAIN_RESULTS/'regions'/key/f'{group}_electrodes.csv',float_precision='round_trip')
            pd.testing.assert_frame_equal(a,b)
        for group in ('first_only','no_detected_difference','second_only')+ (('bidirectional',) if key=='picture' else ()):
            a=pd.read_csv(old/'regions'/key/f'{group}_display_coordinates.csv')
            b=pd.read_csv(MAIN_RESULTS/'regions'/key/f'{group}_display_coordinates.csv')
            pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=0,atol=1e-8)
    expected={'picture':[162,128,81,44],'lexical':[214,187,135,71]}
    counts=pd.read_csv(SAMPLING_RESULTS/'cohort_counts.csv')
    for key,values in expected.items():assert counts.loc[counts.comparison.eq(key),'n_electrodes'].tolist()==values
    for key,regions in [('picture',[23,9,12]),('lexical',[13,10,48])]:
        meta=json.loads((SAMPLING_RESULTS/key/'overview/completion.json').read_text())
        assert meta['baseline_venn_regions']==regions and meta['unassigned_in_statistics']==0
    proof=json.loads((audit/'legacy_hga_extraction.json').read_text())
    source=(ROOT/'src/figure3_legacy_hga.py').read_text()
    functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    assert all(hashlib.sha256(functions[name].encode()).hexdigest()==digest for name,digest in proof['functions'].items())
    # Verify every displayed supplementary matrix against its condition's
    # original values and native baseline mask, using the exported row order.
    checked=0;memberships={}
    for key in ('sound_image','decision_repeat','repeat_passive'):
        meta=json.loads((HGA_RESULTS/f'{key}_manifest.json').read_text());memberships[key]=meta['n_electrodes']
        frame=pd.read_csv(meta['source'],float_precision='round_trip')
        if key=='repeat_passive':frame=frame.loc[frame.contrast.eq(key)].copy()
        else:frame['level']=frame.condition;frame['mask']=frame.baseline_mask
        levels=meta['levels']
        with np.load(HGA_RESULTS/f'{key}_matrices.npz',allow_pickle=False) as f:
            for phase in ('stimulus','delay','response'):
                part=frame.loc[frame.phase.eq(phase)&frame.time.between(*meta['phase_windows'][phase])].copy()
                for c in ('sensory','sustain','motor'):
                    prefix=f'{phase}_{c}';ids=f[prefix+'_electrodes'].tolist();times=f[prefix+'_time']
                    part_selected=part.loc[part.electrode.isin(ids)].copy()
                    part_selected.loc[~part_selected['mask'],'value']=np.nan
                    matrices=[]
                    for condition in reversed(levels):
                        matrix=part_selected.loc[part_selected.level.eq(condition)].pivot_table(index='electrode',columns='time',values='value',aggfunc='mean').reindex(index=ids,columns=times).to_numpy(float)
                        matrices.append(matrix)
                    np.testing.assert_allclose(f[prefix+'_matrix'],np.concatenate(matrices),rtol=0,atol=0,equal_nan=True)
                    checked+=1
    assert memberships==dict(sound_image=81,decision_repeat=135,repeat_passive=64)
    total_svgs=0
    for relative in ('vizpub/fig3.ipynb','vizpub/supplementary/fig3_sampling_responsiveness.ipynb','vizpub/supplementary/fig3_all_condition_hga.ipynb'):
        nb=nbformat.read(ROOT/relative,as_version=4)
        for cell in nb.cells:
            for output in cell.get('outputs',[]):
                assert output.get('output_type')!='error'
                assert 'image/png' not in output.get('data',{})
                svg=output.get('data',{}).get('image/svg+xml')
                if svg:
                    assert 'Unassigned' not in svg
                    total_svgs+=1
    assert total_svgs==20
    protected=json.loads((audit/'protected.json').read_text())
    assert all(sha(p)==h for p,h in protected.items())
    write(audit/'validation.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        main_retained_numeric_arrays_identical=arrays,main_spatial_lists_and_projections_identical=True,
        original_HGA_function_bodies_identical=len(proof['functions']),
        supplementary_phase_component_matrices_match_native_values_and_masks=checked,
        main_svg_subpanels=11,supplementary_sampling_svg_subpanels=6,supplementary_hga_svg_subpanels=3,
        supplementary_HGA_memberships=memberships,protected_other_figures_and_NMF_unchanged=len(protected),
        no_new_inferential_tests=True))
    print('PASS Figure3 consolidation:',arrays,'main arrays unchanged;',checked,'SI matrices match original baseline-masked data;',total_svgs,'SVGs',flush=True)


if __name__=='__main__':main()
