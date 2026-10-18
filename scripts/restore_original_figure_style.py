#!/usr/bin/env python3
"""Restore original figure renderers while keeping reviewed sound-only inputs."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=Path('/hpc/group/coganlab/nanlinshi/insula-functional')


def source(cell):return ''.join(cell['source'])


def set_source(cell,text):cell['source']=text.splitlines(keepends=True)


def original_trace(text,table,clusters=False):
 """Original axes, canvas, scale bar and palette; retain reviewed mean/SEM."""
 text=text.replace('phase_df = subset[','phase_df = '+table+'[').replace('(subset.phase == phase) & (subset.time >= lo) & (subset.time <= hi)',
  f'({table}.phase == phase) & ({table}.time >= lo) & ({table}.time <= hi)')
 start=text.index('    sns.lineplot(');end=text.index('    ax.set_xlim',start)
 if clusters:
  lines='''    for cluster in CLUSTERS:
        row = phase_df.loc[phase_df.functional_cluster.eq(cluster)].sort_values('time')
        color = FUNCTION_COLORS[cluster]
        ax.plot(row.time, row['mean'], color=color, lw=1)
        ax.fill_between(row.time, row['mean']-row['sem'], row['mean']+row['sem'],
                        color=color, alpha=.2, linewidth=0)
'''
 else:
  lines='''    phase_df = phase_df.sort_values('time')
    ax.plot(phase_df.time, phase_df['mean'], color='k', lw=1)
    ax.fill_between(phase_df.time, phase_df['mean']-phase_df['sem'],
                    phase_df['mean']+phase_df['sem'], color='k', alpha=.2, linewidth=0)
'''
 return text[:start]+lines+text[end:]


def main():
 references={name:json.loads((REFERENCE/f'vizpub/{name}.ipynb').read_text()) for name in ['fig1','fig2']}
 notebooks={name:json.loads((ROOT/f'vizpub/{name}.ipynb').read_text()) for name in ['fig1','fig2']}
 old={name:{c.get('id'):source(c) for c in nb['cells']} for name,nb in references.items()}
 cells={name:{c.get('id'):c for c in nb['cells']} for name,nb in notebooks.items()}
 records=[]
 def restore(name,key,text=None,note='Original plotting cell; current data supplied upstream'):
  code=old[name][key] if text is None else text
  set_source(cells[name][key],code)
  records.append(dict(figure=name,cell=key,reference_source_sha256=hashlib.sha256(old[name][key].encode()).hexdigest(),
   restored_source_sha256=hashlib.sha256(code.encode()).hexdigest(),adaptation=note))
 # Original heatmap: mask=False -> NaN; first significant time per phase.
 heatmap=old['fig2']['6c3ec19c']
 proof='''
assert agg.loc[~agg['mask'],'value'].isna().all()
heatmap_order=[]
for cluster in HEATMAP_CLUSTERS:
    for phase in TRACE_PHASES:
        ordered=cluster_phase_channel_order_crop(agg,phase,cluster)
        for rank,channel in enumerate(ordered):
            selected=agg.loc[agg.channel.eq(channel)&agg.phase.eq(phase)&agg['mask']]
            lo,hi=FIG2_TRACE_XLIMS[phase]
            times=selected.loc[selected.time.between(lo,hi),'time']
            heatmap_order.append(dict(cluster=cluster,phase=phase,row=rank,channel=channel,
                first_significant_time=float(times.min()) if len(times) else None))
pd.DataFrame(heatmap_order).to_csv(FIG2_DATA/'heatmap_row_order.csv',index=False)
heatmap_validation=dict(non_significant_points_blank=True,order='first significant time separately per phase',
    n_significant_points=int(agg['mask'].sum()),n_blank_points=int(agg.value.isna().sum()),
    pointwise_source='existing saved sound Repeat masks; any across contributing tasks')
(FIG2_DATA/'heatmap_plot_validation.json').write_text(json.dumps(heatmap_validation,indent=2)+'\\n')
'''
 heatmap=heatmap.replace('save_aligned_svg(fig, "fig2_hga_heatmap.svg")',proof+'\nsave_aligned_svg(fig, "fig2_hga_heatmap.svg")')
 restore('fig2','6c3ec19c',heatmap,'Original masking, row ordering and renderer; add saved verification tables')
 mask_load='''
point_masks=pd.read_csv(FIG2_DATA/'site_significance_masks.csv.gz')
subset=subset.merge(point_masks[['channel','phase','time','mask','mask_available']],
    on=['channel','phase','time'],validate='one_to_one')
assert subset['mask'].notna().all()
'''
 text=source(cells['fig2']['fig2-load'])
 if 'point_masks=pd.read_csv' not in text:text+=mask_load
 set_source(cells['fig2']['fig2-load'],text)
 pooled='''pooled = subset.groupby(['phase','time']).value.agg(['mean','std','count']).reset_index()
pooled['sem'] = pooled['std']/np.sqrt(pooled['count'])
pooled.to_csv(FIG2_DATA/'pooled_waveform_summary.csv',index=False)
'''
 restore('fig2','fig2-avg-plot',pooled+original_trace(old['fig2']['fig2-avg-plot'],'pooled'),
  'Original curve canvas, axes, scale bar; approved electrode mean and SEM')
 restore('fig2','fig2-hga-plot',original_trace(old['fig2']['fig2-hga-plot'],'cluster_trace',True),
  'Original curve canvas, axes, scale bar and colors; approved electrode mean and SEM')
 restore('fig2','fig2-nmf-plot',old['fig2']['fig2-nmf-plot'].replace(
  '# Concat-NMF H is post-onset only (no pre-event samples). Go cannot use the\n# HGA crop (-1, 0); that window is empty. Keep the same column *width* as HGA go.',
  '# Retain the original display context. Current fitted phases and samples come from H_by_phase.'))
 rank=old['fig2']['fig2-rank-plot'].replace('k_nmf = int(chosen["k"])','k_nmf = int(manifest["k"])')
 rank=rank.replace('print(f"chosen k={k_nmf}  rule={chosen[\'rule\']}  near-ties={chosen[\'near_tie_ks\']}")',
  'print(f"fitted k={k_nmf}; cophenetic maximum k={chosen[\'k\']}; near ties={chosen[\'near_tie_ks\']}")')
 restore('fig2','fig2-rank-plot',rank,'Original cosine silhouette plot; highlight actual fitted k=3')
 restore('fig2','fig2-pca-kmeans-plot',old['fig2']['fig2-pca-kmeans-plot']+'''
km_table = scores[['channel','subject','functional_cluster']].copy()
km_table['kmeans_cluster'] = km3_named
km_table.to_csv(FIG2_DATA/'pca_kmeans_membership.csv',index=False)
''')
 restore('fig2','3b264e16',old['fig2']['3b264e16']+"\nqc.to_csv(FIG2_DATA/'loading_specificity.csv',index=False)\n")
 restore('fig2','fig2-spatial-plot')
 # Keep the reviewed sampling denominator; use original laterality renderer.
 lat=old['fig2']['fig2-laterality-plot'];start=lat.index('n_imp_l = ')
 prefix='''from scipy.stats import chi2_contingency, fisher_exact
implanted = pd.read_csv(ROOT/'results/fig1/insula_plot_coordinates.csv')
assign_lr = assignments.copy()
n_imp = implanted.groupby('hemi').channel.nunique().reindex(['L','R'])
n_cluster = assignments.groupby(['functional_cluster','hemi']).size().unstack(fill_value=0).reindex(index=CLUSTERS,columns=['L','R'],fill_value=0)
n_rate = n_cluster.div(n_imp,axis=1)
coverage = implanted.groupby(['subject','hemi']).channel.nunique().rename('n_sampled_sites').reset_index()
coverage.to_csv(FIG2_DATA/'hemisphere_sampling_by_subject.csv',index=False)
n_cluster.to_csv(FIG2_DATA/'hemisphere_cluster_counts.csv')
n_rate.to_csv(FIG2_DATA/'hemisphere_cluster_fractions.csv')
'''
 restore('fig2','fig2-laterality-plot',(prefix+lat[start:]).replace('implanted AIC/PIC','reviewed sampled').replace('fraction of implanted','fraction of sampled'),
  'Original plot and exploratory electrode-count tests; reviewed sound sampling denominator retained')
 manifest=source(cells['fig2']['e4f7f44d']).replace("heatmap='unthresholded site means; same site order across phases'",
  "heatmap='saved nonsignificant points blank; first significant time order separately per phase'").replace(
  "laterality='descriptive confirmed cluster fraction of all reviewed sampled sites; no electrode-level inference'",
  "laterality='original electrode-count Fisher/chi-square summaries of confirmed observations; denominator includes incomplete response evidence; not participant-level tests'")
 set_source(cells['fig2']['e4f7f44d'],manifest)
 set_source(notebooks['fig2']['cells'][11],
  '## 4. HGA heatmap\n\nOriginal display: blank nonsignificant saved mask points and order electrodes by first significant time separately within each phase. Missing masks remain blank and are reported separately.')
 # Figure 1 laterality returns to the original line plot with current counts.
 lat=old['fig1']['4de8e393']
 append='''
assert n_samp.sum()==N_SAMPLED and n_sig.sum()==N_RESPONSIVE
laterality_counts=pd.DataFrame({'sampled':n_samp,'confirmed_responsive':n_sig})
laterality_counts.to_csv(FIG1_DATA/'insula_laterality_counts.csv')
(FIG1_DIR/'fig1_insula_laterality_counts.svg').unlink(missing_ok=True)
(ROOT/'docs/evidence/figure1_sound_only_2026-10-17/figures/fig1_insula_laterality_counts.svg').unlink(missing_ok=True)
'''
 restore('fig1','4de8e393',lat+append,'Original laterality plot; current reviewed counts and CSV export')
 # Original SEEG example includes the D0075 MRI/MAPER slice. Its native MAPER
 # is no longer available; retain the published SVG exactly rather than replace
 # its subject, geometry, tissue map or contact drawing.
 asset=REFERENCE/'img/fig1/fig1_insula_electrode_schematic_seeg.svg'
 seeg=f'''# Preserve the original D0064/D0075 physical-contact illustration unchanged.
import shutil
from IPython.display import SVG
original_seeg=Path({str(asset)!r})
assert sha256_file(original_seeg)=={hashlib.sha256(asset.read_bytes()).hexdigest()!r}
shutil.copy2(original_seeg,FIG1_DIR/'fig1_insula_electrode_schematic_seeg.svg')
display(SVG(filename=str(FIG1_DIR/'fig1_insula_electrode_schematic_seeg.svg')))
'''
 restore('fig1','fig1-seeg-plot',seeg,'Original illustrative SVG reused byte-for-byte; original D0075 MAPER input is unavailable')
 average=source(cells['fig1']['fig1-average'])
 marker='FIG2_TRACE_XLIMS=TRACE_XLIMS' if 'FIG2_TRACE_XLIMS=TRACE_XLIMS' in average else 'width_ratios = [hi-lo for lo,hi in TRACE_XLIMS.values()]'
 prefix=average[:average.index(marker)]
 constants='''FIG2_TRACE_XLIMS=TRACE_XLIMS
FIG2_TRACE_XTICKS={'stimulus':(-.5,0,.6),'delay':(0,1),'response':(-.5,0,1.5)}
TRACE_PHASES=list(phase_configs)
FIG2_TRACE_SIZE=(8*cm,3*cm)
FIG2_MARGINS=dict(left=.16,right=.94,wspace=.2)
'''
 plot=original_trace(old['fig2']['fig2-avg-plot'],'group_mean').replace(
  'save_aligned_svg(fig, "fig2_insula_average_traces.svg")',
  "fig.savefig(FIG1_DIR/'fig1_insula_average_traces.svg',bbox_inches=None,pad_inches=0)")
 set_source(cells['fig1']['fig1-average'],prefix+constants+plot)
 records.append(dict(figure='fig1',cell='fig1-average',adaptation='Original pooled HGA canvas, axes and scale bar; approved electrode mean/SEM and existing site waveforms'))
 for name,nb in notebooks.items():
  for c in nb['cells']:
   if c['cell_type']=='code':c['execution_count']=None;c['outputs']=[]
  (ROOT/f'vizpub/{name}.ipynb').write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n')
 out=ROOT/'docs/evidence/original_figure_style_2026-10-17';out.mkdir(exist_ok=True)
 (out/'restored_cells.json').write_text(json.dumps(records,indent=2)+'\n')
 print(f'Restored {len(records)} plotting cells from original notebooks.')


if __name__=='__main__':main()
