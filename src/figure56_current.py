"""Execute and validate the current224-site Figure5/6 publication notebooks."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT=Path('/cwork/ns458/insula-paper')
RUN=ROOT/'results/figure56_manual14_2026-11-07'
NMF=ROOT/'results/insula_manual14_2026-11-07/nmf'
RT=RUN/'rt'
PROJ=RUN/'projection'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def tree(source):
    return ast.dump(ast.parse('\n'.join(x for x in source.splitlines() if not x.lstrip().startswith('%'))),include_attributes=False)


def expanded_hga_ylim(original=(0.,2.)):
    """Keep the shared scale while including the refreshed mean+SEM envelope."""
    import math
    import pandas as pd
    peak=float(original[1])
    for phase,limits in (('delay',(0.,1.)),('go',(-1.,1.5))):
        frame=pd.read_csv(RUN/f'{phase}_electrode_mean_sem.csv')
        frame=frame[frame.time.between(*limits)]
        peak=max(peak,float((frame['mean']+frame['sem'].fillna(0.)).max()))
    upper=original[1] if peak<=original[1] else math.ceil(peak*1.03*2)/2
    result=(float(original[0]),float(upper))
    write(RUN/'fig5/display_hga_ylim.json',dict(original=list(original),current=list(result),
          observed_mean_plus_SEM_max=peak,values_or_SEM_changed=False))
    return result


def expand_shared_y(axes):
    """Expand only the common upper display limit to contain plotted artists."""
    import math
    import numpy as np
    lower=min(ax.get_ylim()[0] for ax in axes)
    original=max(ax.get_ylim()[1] for ax in axes)
    maxima=[ax.dataLim.ymax for ax in axes if np.isfinite(ax.dataLim.ymax)]
    peak=max(maxima,default=original)
    upper=original if peak<=original else math.ceil(peak*1.03*10)/10
    for ax in axes:ax.set_ylim(lower,upper)
    write(RUN/'fig5/display_rt_r_ylim.json',dict(original=[lower,original],current=[lower,upper],
          observed_mean_plus_SEM_max=float(peak),values_or_SEM_changed=False))


def prepare_notebook(figure):
    import nbformat
    path=ROOT/f'vizpub/fig{figure}.ipynb'
    nb=nbformat.read(path,as_version=4)
    before=nbformat.read(RUN/f'before/vizpub/fig{figure}.ipynb',as_version=4)
    original={c.id:c for c in before.cells}
    # Keep original paths explicit and captions accurate. The only extra
    # display adjustment expands shared upper limits to contain new mean+SEM;
    # statistical calls, values, palettes and drawing geometry stay intact.
    for cell in nb.cells:
        if cell.cell_type=='markdown':
            if cell.id==f'fig{figure}-title':
                if 'Current version: figure56_manual14_2026-11-07' not in cell.source:
                    note=('Current version: figure56_manual14_2026-11-07. Current224-site assignments; '
                          'original anatomical RT modeling and correction families, with updated native-good contacts. '
                          'Unassigned and the14 analysis exclusions do not enter functional reporting. '
                          'Go panels retain the original fixed0.55/0.93-s visual reference lines; '
                          'these are not estimates of the refreshed cohort means.' if figure==5 else
                          'Current version: figure56_manual14_2026-11-07. Whole-brain projection is restricted to '
                          'the67 reviewed insula-sampled subjects and uses the current224-site three-phase H. '
                          'NNLS, surface smoothing and top50% display are recomputed using the original rules. '
                          'Discovery means membership in the224-site template fit; other eligible whole-brain contacts '
                          'remain continuous projections without receiving NMF discovery labels.')
                    cell.source += '\n\n'+note
            if figure==5:
                cell.source=cell.source.replace('Sensory has one subject, so $t$ is undefined and has no null.',
                    'A group with fewer than two subjects has undefined $t$ and no null; current counts are printed below.')
            if figure==6:
                cell.source=cell.source.replace('`results/nmf/whole_brain_projection/visualization/surface_motif_maps_focused.npz`',
                    '`results/figure56_manual14_2026-11-07/projection/visualization/surface_motif_maps_focused.npz`')
            continue
        if figure==5 and cell.id=='fig5-style':
            cell.source=cell.source.replace('RT_ROOT = RESULTS_ROOT / "rt"',
                'from src.main_analysis import FIG5_RT_ROOT\nRT_ROOT = FIG5_RT_ROOT')
            if 'expanded_hga_ylim' not in cell.source:
                cell.source+='\n\nfrom src.figure56_current import expanded_hga_ylim\nHGA_YLIM = expanded_hga_ylim(HGA_YLIM)'
        raster_injection='    from src.figure56_current import expand_shared_y\n    expand_shared_y(axes_r)\n\n'
        if figure==5 and cell.id=='fig5-raster' and 'expand_shared_y' not in cell.source:
            anchor='    axes_r[0].set_ylabel("mean OOF r", fontsize=fontsize)'
            assert anchor in cell.source
            cell.source=cell.source.replace(anchor,raster_injection+anchor,1)
        if figure==5 and cell.id=='fig5-encoding':
            cell.source=cell.source.replace('# Group mean RTs (go onset to response): fast 0.55 s, slow 0.93 s.',
                '# Original fixed visual references:0.55/0.93 s, not refreshed-cohort mean estimates.')
        if figure==6 and cell.id=='fig6-imports':
            old='from src.paths import img_dir, nmf_wholebrain_dir, save_svg'
            new=('from src.paths import img_dir, save_svg\n'
                 'from src.main_analysis import FIG6_PROJECTION_DIR, FIG6_NMF_DIR\n'
                 'def nmf_wholebrain_dir():\n    return FIG6_PROJECTION_DIR')
            cell.source=cell.source.replace(old,new)
        if figure==6 and cell.id=='928218f3':
            cell.source=cell.source.replace('H_PATH = ROOT / "results" / "nmf" / "H_by_phase.csv"',
                                           'H_PATH = FIG6_NMF_DIR / "H_by_phase.csv"')
        if figure==6 and cell.id=='fig6-roi':
            cell.source=cell.source.replace('from src.paths import nmf_wholebrain_dir',
                                           '# Current projection path is imported in fig6-imports.')
        allowed={5:{'fig5-style'},6:{'fig6-imports','928218f3','fig6-roi'}}[figure]
        if cell.id in original and cell.id not in allowed:
            normalized=cell.source.replace(raster_injection,'') if figure==5 and cell.id=='fig5-raster' else cell.source
            assert tree(normalized)==tree(original[cell.id].source),f'Unexpected drawing/statistic change:{cell.id}'
    audit_id=f'fig{figure}-current-version-audit'
    if not any(c.id==audit_id for c in nb.cells):
        cell=nbformat.v4.new_code_cell(f'from src.figure56_current import export_audit\nexport_audit({figure}, globals())')
        cell.id=audit_id;nb.cells.append(cell)
    return nb


def export_audit(figure, namespace):
    import h5py
    import numpy as np
    import pandas as pd
    out=RUN/f'fig{figure}';out.mkdir(exist_ok=True)
    assignments=pd.read_csv(NMF/'channel_assignments.csv')
    labels=assignments.set_index('channel').functional_cluster
    excluded=set(json.loads((ROOT/'config/insula_manual14_2026-11-07.json').read_text())['excluded_sites'])
    assert len(labels)==224
    if figure==5:
        e=namespace['electrodes'];c=namespace['clusters'];traces=namespace['traces']
        for frame in (e,c,namespace['elec_enc'],traces,namespace['delay_traces']):
            assert set(frame.channel)<=set(labels.index)
            assert not set(frame.channel)&excluded
            assert frame.functional_cluster.eq(frame.channel.map(labels)).all()
        assert set(e.source_h5.astype(str).map(lambda p:Path(p).is_relative_to(RT)))=={True}
        main=c[((c.phase=='Delay')&(c.end_time>0))|((c.phase=='Go')&(c.start_time<0))]
        e[['task','subject','phase','channel','functional_cluster','source_h5']].to_csv(out/'modeled_functional_membership.csv',index=False)
        main.to_csv(out/'delay_go_hit_clusters.csv',index=False)
        c.to_csv(out/'all_phase_hit_clusters.csv',index=False)
        namespace['raster_rows'].to_csv(out/'raster_rows.csv',index=False)
        namespace['r_traces'].to_csv(out/'raster_r_values.csv.gz',index=False)
        comparisons=[]
        for phase,key in [('go','traces'),('delay','delay_traces')]:
            expected=pd.read_csv(RUN/f'{phase}_electrode_mean_sem.csv')
            expected.rt_split=expected.rt_split.map({'hga_fast':'fast','hga_slow':'slow'})
            for cluster in ('sensory','sustain','motor'):
                long=namespace['_hga_rt_long'](namespace[key][namespace[key].functional_cluster.eq(cluster)])
                actual=long.groupby(['rt_split','time']).hga.agg(['mean','sem','count']).reset_index()
                ref=expected[expected.functional_cluster.eq(cluster)]
                merged=actual.merge(ref,on=['rt_split','time'],suffixes=('_actual','_expected'))
                assert len(merged)==len(actual)==len(ref)
                for field in ('mean','sem','count'):
                    np.testing.assert_allclose(merged[field+'_actual'],merged[field+'_expected'],rtol=1e-10,atol=1e-12,equal_nan=True)
                comparisons.append(dict(phase=phase,cluster=cluster,samples=len(actual)))
        nulls=namespace['enc_nulls'];null_rows=[]
        for cluster,r in nulls.items():
            np.savez_compressed(out/f'{cluster}_encoding_null.npz',observed_r=r['observed_r'],null_t=r['null_t'])
            null_rows.append({k:r[k] for k in ('n_subjects','observed_t','p_left','n_perm')}|dict(cluster=cluster))
        pd.DataFrame(null_rows).to_csv(out/'encoding_null_summary.csv',index=False)
        record=dict(passed=True,n_modeled_functional_sites=e.channel.nunique(),n_modeled_subjects=e.subject.nunique(),
            n_Delay_Go_hit_sites=main.channel.nunique(),n_Delay_Go_hit_subjects=main.subject.nunique(),
            n_all_phase_hit_sites=c.channel.nunique(),n_all_phase_hit_subjects=c.subject.nunique(),
            native_task_QC_preserved=True,manual14_and_Unassigned_absent_from_all_functional_reports=True,
            assignment_sha256=sha(NMF/'channel_assignments.csv'),HGA_electrode_means_and_SEM_verified=comparisons,
            statistical_methods_and_drawing_geometry_unchanged=True,original_fixed_RT_guide_lines=[.55,.93],
            display_y_limits={name:json.loads((out/f'display_{name}_ylim.json').read_text()) for name in ('hga','rt_r')},
            exported_HGA_components=[name for name in ('sensory','sustain','motor')
                if namespace['subjects_enc'].functional_cluster.eq(name).any() and traces.functional_cluster.eq(name).any()])
    else:
        manifest=json.loads((PROJ/'manifest.json').read_text());spec=json.loads((RUN/'figure6_spec.json').read_text())
        raw=pd.read_csv(PROJ/'electrode_projection.csv')
        assert set(raw.subject)==set(spec['subjects']) and raw.subject.nunique()==67
        np.testing.assert_array_equal(raw.in_discovery.to_numpy(),raw.channel.isin(labels.index).to_numpy())
        assert manifest['basis_sha256']==sha(NMF/'H_by_phase.csv') and manifest['n_features']==399
        pd.testing.assert_frame_equal(namespace['H_by_phase'].drop(columns=['cluster_label']).reset_index(drop=True),
            pd.read_csv(NMF/'H_by_phase.csv').loc[lambda f:f.functional_cluster.isin(namespace['COMPONENTS'])].reset_index(drop=True))
        for key in ('electrodes','marked','in_patch','counts','plot_frame'):
            namespace[key].to_csv(out/f'{key}.csv',index=False)
        marked=namespace['marked'];assert set(marked.subject)<=set(spec['subjects'])
        absent=assignments[~assignments.channel.isin(raw.channel)]
        absent.to_csv(out/'discovery_sites_not_in_projection.csv',index=False)
        raw[['channel','subject','roi','in_discovery']].assign(manual_analysis_excluded=raw.channel.isin(excluded)).to_csv(out/'projection_membership.csv',index=False)
        np.savez_compressed(out/'current_display_payload.npz',**namespace['payload'])
        in_patch=namespace['in_patch']
        record=dict(passed=True,cohort_subjects=67,n_projected_sites=len(raw),n_projected_subjects=raw.subject.nunique(),
            n_discovery_sites_projected=int(raw.in_discovery.sum()),n_discovery_missing=len(absent),
            n_heldout_map_sites=len(namespace['electrodes']),n_heldout_map_subjects=namespace['electrodes'].subject.nunique(),
            n_in_patch_noninsula_sites=len(in_patch),n_in_patch_noninsula_subjects=in_patch.subject.nunique(),
            phases=manifest['phases'],n_features=399,basis_sha256=manifest['basis_sha256'],
            assignment_sha256=sha(NMF/'channel_assignments.csv'),surface_cache_sha256=sha(PROJ/'visualization/surface_motif_maps_focused.npz'),
            original_fixed_H_NNLS_and_spatial_rules_preserved=True,keep_top=.5,
            not_pairwise_connectivity=True,trial_counts='Unavailable in trial-averaged HGA sources; none invented.')
    # Pandas count scalars need native JSON numbers.
    record={k:int(v) if isinstance(v,np.integer) else v for k,v in record.items()}
    write(out/'method_validation.json',record)
    print(json.dumps(record,indent=2),flush=True)


def execute(figure):
    import nbformat
    from nbclient import NotebookClient
    assert os.environ.get('SLURM_JOB_ID')
    if figure==5:
        assert (RUN/'rt_data_completed.json').is_file()
    else:
        assert (PROJ/'visualization/surface_motif_maps_focused.npz').is_file()
    path=ROOT/f'vizpub/fig{figure}.ipynb';before_sha=sha(path)
    nb=prepare_notebook(figure);out=RUN/f'fig{figure}';out.mkdir(exist_ok=True)
    figures=ROOT/f'img/fig{figure}';figures.mkdir(exist_ok=True)
    if figure==5 and (out/'execution_validation.json').exists() and not (out/'before_display_limit_expansion').exists():
        archive=out/'before_display_limit_expansion';archive.mkdir()
        shutil.copy2(path,archive/path.name)
        for p in figures.glob('*.svg'):shutil.copy2(p,archive/p.name)
        for name in ('execution_validation.json','method_validation.json'):shutil.copy2(out/name,archive/name)
    for p in figures.glob('*.svg'):p.unlink()  # Archived before this task.
    kernel=out/f'jupyter/kernels/current-fig{figure}';kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name=f'Current Figure{figure}',language='python')))
    os.environ['JUPYTER_PATH']=str(out/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    os.environ['NUMBA_CACHE_DIR']=str(out/'numba_cache');os.environ['MNE_CONFIG_PATH']=str(out/'mne_config')
    def started(cell,cell_index,**kwargs):
        print(datetime.now(timezone.utc).isoformat(),'cell',cell_index,cell.get('id'),flush=True)
    try:
        NotebookClient(nb,kernel_name=f'current-fig{figure}',timeout=3600,iopub_timeout=180,
            raise_on_iopub_timeout=True,record_timing=True,on_cell_start=started,
            resources={'metadata':{'path':str(ROOT/'vizpub')}}).execute()
    except Exception:
        nbformat.write(nb,out/f'fig{figure}_failed.ipynb');raise
    assert not [o for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    assert sha(path)==before_sha,'Concurrent author notebook edit; preserve it'
    nb.metadata['insula_refresh']=dict(version='figure56_manual14_2026-11-07',figure=figure,
        assignment_sha256=sha(NMF/'channel_assignments.csv'),NMF_input_sites=224,
        figure6_cohort_subjects=67,svg_inline_only=True,executed_utc=datetime.now(timezone.utc).isoformat())
    nbformat.write(nb,path)
    result=json.loads((out/'method_validation.json').read_text())
    exports=sorted(figures.glob('*.svg'))
    expected_exports=3 if figure==6 else 5+len(result['exported_HGA_components'])
    assert len(exports)==expected_exports,f'Unexpected current export inventory:{exports}'
    validation=dict(passed=True,figure=figure,version='figure56_manual14_2026-11-07',
        code_cells=sum(c.cell_type=='code' for c in nb.cells),all_code_cells_executed=True,svg_inline_only=True,
        notebook_sha256=sha(path),figures=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in exports],
        job_id=os.environ['SLURM_JOB_ID'],method_validation=result)
    write(out/'execution_validation.json',validation)
    formal=ROOT/f'results/fig{figure}';formal.mkdir(exist_ok=True)
    for name in ('method_validation.json','execution_validation.json'):shutil.copy2(out/name,formal/name)
    cfg=dict(version='figure56_manual14_2026-11-07',status='complete',notebook=str(path),NMF_dir=str(NMF),
             data_dir=str(RT if figure==5 else PROJ),validation=str(out/'execution_validation.json'),counts=result)
    write(ROOT/f'config/figure{figure}_manual14_2026-11-07.json',cfg)
    # The final registry update for both figures is performed after both jobs.
    print('COMPLETED current Figure',figure,'exports',len(exports),flush=True)


def finalize():
    import pandas as pd
    import numpy as np
    protected=json.loads((RUN/'protected_files.json').read_text())
    changed=[p for p,h in protected.items() if sha(p)!=h]
    assert not changed,changed
    reports={i:json.loads((RUN/f'fig{i}/execution_validation.json').read_text()) for i in (5,6)}
    assert all(r['passed'] for r in reports.values())
    assert json.loads((RUN/'RT_phase_eligibility_and_cache_source_audit.json').read_text())['passed']
    assert json.loads((RUN/'RT_serialization_and_parallel_validation.json').read_text())['passed']
    plan=json.loads((RUN/'rt_run_plan.json').read_text())
    assert len(plan['models'])==99 and plan['restored_groups_appended']
    assert sha(plan['complete_metadata_path'])==plan['complete_metadata_sha256']
    checked_inputs={}
    for name in ('native_input_hashes.json','restored_native_input_hashes.json','projection_input_hashes.json'):
        for path,digest in json.loads((RUN/name).read_text()).items():
            assert sha(path)==digest,f'Input changed during analysis:{path}'
            checked_inputs[path]=digest
    write(RUN/'verified_final_input_hashes.json',checked_inputs)
    # The numerical signal/RT functions are byte-for-byte AST-identical to the
    # original; the one additional branch supplies reviewed coordinate metadata.
    old=ast.parse((RUN/'before/src/reaction_time/insula_rt_data.py').read_text())
    new=ast.parse((ROOT/'src/reaction_time/insula_rt_data.py').read_text())
    old_functions={n.name:n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    new_functions={n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for key,node in new_functions.items():
        if key=='load_strict_insula_parcellation':
            node.body=[n for n in node.body if not (
                isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='reviewed_metadata' for t in n.targets))
                and not (isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='reviewed_metadata')]
        assert ast.dump(node,include_attributes=False)==ast.dump(old_functions[key],include_attributes=False),key
    engine_paths=[p for p in (ROOT/'src/reaction_time').glob('*.py') if p.name!='insula_rt_data.py']
    engine_paths += [ROOT/'src/nmf/whole_brain_projection.py',ROOT/'src/nmf/whole_brain_projection_viz.py',
                     ROOT/'scripts/plot_nmf_whole_brain_projection.py']
    engine_hashes={}
    for path in engine_paths:
        committed=subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)
        assert path.read_bytes()==committed,f'Unexpected numerical engine change:{path}'
        engine_hashes[str(path)]=sha(path)
    write(RUN/'unchanged_numerical_engine_hashes.json',engine_hashes)
    registry=ROOT/'config/current_insula_analysis.json';data=json.loads(registry.read_text())
    data['current_figures'].pop('fig4_to_fig6',None)
    data['current_figures']['fig4']='Preserved prior formal model/results; electrode-removal comparison remains diagnostic only.'
    for i in (5,6):
        cfg=json.loads((ROOT/f'config/figure{i}_manual14_2026-11-07.json').read_text())
        data['current_figures'][f'fig{i}']=cfg
    data['status']='adopted_manual14_and_refreshed_figures1_to3_and5_to6'
    write(registry,data)
    result=dict(passed=True,protected_files_unchanged=len(protected),figures=[5,6],
        native_and_projection_inputs_unchanged=len(checked_inputs),
        current_NMF_sites=224,Figure6_cohort_subjects=67,original_numerical_RT_and_projection_engines_preserved=True,
        RT_metadata_provider_extension_only=True,job_id=os.environ['SLURM_JOB_ID'],
        complete_utc=datetime.now(timezone.utc).isoformat())
    write(RUN/'completed.json',result)
    # Raster previews are app-cache only; publication exports remain SVG.
    import cairosvg
    import io
    from PIL import Image,ImageDraw
    cache=Path('/hpc/home/ns458/.codex/visualizations/2026/11/07/01a1b77b-8243-7f90-a68e-a0cfb4ba70d4')
    cache.mkdir(parents=True,exist_ok=True)
    for number in (5,6):
        paths=sorted((ROOT/f'img/fig{number}').glob('*.svg'))
        sheet=Image.new('RGB',(1600,((len(paths)+1)//2)*650),'white');draw=ImageDraw.Draw(sheet)
        for index,path in enumerate(paths):
            tile=Image.open(io.BytesIO(cairosvg.svg2png(url=str(path),output_width=1200))).convert('RGB');tile.thumbnail((780,600))
            x,y=(index%2)*800,(index//2)*650
            draw.text((x+10,y+8),path.stem,fill='black');sheet.paste(tile,(x+(800-tile.width)//2,y+35))
        sheet.save(cache/f'fig{number}_manual14_contact_sheet.png')
    print(json.dumps(result,indent=2),flush=True)
