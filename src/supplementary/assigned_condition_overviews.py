"""Current descriptive condition-set panels, excluding functional Unassigned."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2,venn2_circles

from src.main_analysis import FUNCTIONAL_COMPONENTS
from src.supplementary.assigned_condition_cohorts import ROOT,OUT,load_cohort
from src.supplementary.assigned_comparison_traces import override


def render(key, *, core_only=False, image_dir=None):
    from scripts.refresh_lexical_condition_sets import write
    from src.supplementary import lexical_condition_figures as legacy
    from src.figure3_brain import brain_pair
    from src.supplementary.condition_difference_group_panels import COLORS as functional_colors,LABELS
    table,parent=load_cohort(key)
    a,b=('sound','image') if key=='picture' else ('decision','repeat')
    title='Picture Naming · Stimulus 0–1 s' if key=='picture' else 'Lexical Delay · Delay 0–1 s'
    dest=OUT/key/'overview';dest.mkdir(parents=True,exist_ok=True)
    if image_dir is None:
        image_dir=ROOT/'img/supplementary'/f'fig3_{key}_stimulus_condition_sets' if key=='picture' else ROOT/'img/supplementary/fig3_lexical_delay_condition_sets'
    image_dir=Path(image_dir)
    image_dir.mkdir(parents=True,exist_ok=True)
    prefix='fig3_picture_stimulus' if key=='picture' else 'fig3_lexical_delay'
    paths={}
    palette={a.capitalize()+' only':'#2166AC',b.capitalize()+' only':'#D95F02','Both':'#762A83'}
    def save(fig,name):
        path=image_dir/f'{prefix}_{name}.svg';fig.savefig(path,bbox_inches='tight',pad_inches=.03);plt.close(fig)
        paths[name]=str(path);return path
    first=set(table.loc[table[a+'_responsive'],'electrode']);second=set(table.loc[table[b+'_responsive'],'electrode'])
    responsive=table[a+'_responsive']|table[b+'_responsive']
    counts=(len(first-second),len(second-first),len(first&second))
    fig,ax=plt.subplots(figsize=(7.2/2.54,5.4/2.54));fig.subplots_adjust(left=.08,right=.92,bottom=.20,top=.80)
    v=venn2(subsets=counts,set_labels=(f'{a.capitalize()}\n(n={len(first)})',f'{b.capitalize()}\n(n={len(second)})'),ax=ax)
    colors=['#2166AC','#D95F02','#762A83']
    for region,color in zip(('10','01','11'),colors):
        patch=v.get_patch_by_id(region)
        if patch is not None:patch.set_facecolor(color);patch.set_alpha(.40)
    for circle,color in zip(venn2_circles(subsets=counts,ax=ax,lw=.8),colors):circle.set_edgecolor(color)
    for label in v.subset_labels:
        if label is not None:label.set_fontsize(8)
    for label,color in zip(v.set_labels,colors):
        if label is not None:label.set_fontsize(7);label.set_color(color)
    fig.suptitle(title,fontsize=8,y=.98)
    fig.text(.5,.025,f'{len(table)} assigned electrodes / {table.subject.nunique()} participants\nNeither detected: {int((~responsive).sum())}',ha='center',fontsize=6.5)
    save(fig,'0_1_venn')
    original_renderer=legacy.brain_renderer()
    for kind in ('sampling_response','baseline_classes'):
        points=table.copy() if kind=='sampling_response' else table.loc[responsive].copy()
        if kind=='sampling_response':
            points['active']=points.electrode.isin(table.loc[responsive,'electrode']);points=points.sort_values('active',kind='stable')
            points['display_group']=np.where(points.active,'Significant, either condition','Sampled assigned, n.s.')
            pal={'Sampled assigned, n.s.':'#BDBDBD','Significant, either condition':'#762A83'}
        else:points['display_group']=points.baseline_class;pal=palette
        fig,audit=original_renderer['brain_pair'](points,list(pal),pal,title+f'\nAssigned statistical cohort: {len(table)} sites')
        save(fig,kind+'_brain');pd.DataFrame(audit).to_csv(dest/f'{kind}_display_coordinates.csv',index=False)
    rows=[]
    sets=[('All assigned',table)]+[(name,table.loc[table.baseline_class.eq(name)]) for name in [a.capitalize()+' only','Both',b.capitalize()+' only','Neither']]
    sets += [(a.capitalize()+' > '+b.capitalize(),table.loc[table[f'{a}_gt_{b}']]),(b.capitalize()+' > '+a.capitalize(),table.loc[table[f'{b}_gt_{a}']])]
    for label,part in sets:
        for c in FUNCTIONAL_COMPONENTS:
            selected=part.loc[part.functional_cluster.eq(c)]
            rows.append(dict(set=label,component=c,n_electrodes=len(selected),n_participants=selected.subject.nunique(),
                component_denominator=int(table.functional_cluster.eq(c).sum())))
    composition=pd.DataFrame(rows);composition['fraction_within_component']=composition.n_electrodes/composition.component_denominator
    composition.to_csv(dest/'component_composition.csv',index=False)
    pd.DataFrame([dict(set=label,n_electrodes=len(part),n_participants=part.subject.nunique(),participants=';'.join(sorted(part.subject.unique()))) for label,part in sets]).to_csv(dest/'set_counts.csv',index=False)
    if core_only:
        write(dest/'completion.json',dict(passed=True,anatomical_inventory_n=len(parent),
            anatomical_inventory_subjects=parent.subject.nunique(),statistical_n=len(table),
            statistical_subjects=table.subject.nunique(),responsive_n=int(responsive.sum()),
            responsive_subjects=table.loc[responsive,'subject'].nunique(),baseline_venn_regions=list(counts),
            unassigned_in_statistics=0,paths=paths,no_new_tests=True,labels_and_coordinates_unchanged=True))
        return paths
    # Original absolute-count stacked-bar convention; never hide small counts in percentages.
    names=[name for name,_ in sets];fig,ax=plt.subplots(figsize=(8.5/2.54,6/2.54));fig.subplots_adjust(left=.33,right=.96,bottom=.22,top=.85)
    left=np.zeros(len(names));y=np.arange(len(names))
    for c in FUNCTIONAL_COMPONENTS:
        values=composition.loc[composition.component.eq(c)].set_index('set').loc[names,'n_electrodes'].to_numpy()
        ax.barh(y,values,left=left,color=functional_colors[c],label=LABELS[c],height=.65)
        for i,(value,start) in enumerate(zip(values,left)):
            if value:ax.text(start+value/2,i,str(value),ha='center',va='center',fontsize=6)
        left+=values
    ax.set_yticks(y,labels=names,fontsize=6);ax.invert_yaxis();ax.set_xlabel('Electrodes',fontsize=7)
    ax.spines[['top','right']].set_visible(False);fig.suptitle(title,fontsize=8)
    fig.legend(ncol=3,loc='lower center',frameon=False,fontsize=6)
    save(fig,'component_composition')
    for group in palette:
        points=table.loc[table.baseline_class.eq(group)].copy();points['display_group']=points.functional_cluster
        if points.empty:continue
        fig,projection,contours=brain_pair(points,list(FUNCTIONAL_COMPONENTS),functional_colors,LABELS,title=title+' · '+group)
        save(fig,'functional_'+group.lower().replace(' ','_'))
        pd.DataFrame(projection).to_csv(dest/f'{group.replace(" ","_")}_projection.csv',index=False)
        pd.DataFrame(contours).to_csv(dest/f'{group.replace(" ","_")}_contours.csv',index=False)
    if key=='lexical':
        waves=pd.read_csv(legacy.OUT/'waveforms.csv.gz');waves=waves.loc[waves.electrode.isin(table.electrode)]
        with override(legacy,OUT=dest,IMG=image_dir,COMPONENT_COLORS=functional_colors,LABELS=LABELS):
            for i,path in enumerate(legacy.draw_waveform_sets(table,waves)):paths[f'waveforms_{i}']=str(path)
    write(dest/'completion.json',dict(passed=True,anatomical_inventory_n=len(parent),statistical_n=len(table),
        statistical_subjects=table.subject.nunique(),responsive_n=int(responsive.sum()),
        responsive_subjects=table.loc[responsive,'subject'].nunique(),baseline_venn_regions=list(counts),
        unassigned_in_statistics=0,paths=paths,no_new_tests=True,labels_and_coordinates_unchanged=True))
    return paths
