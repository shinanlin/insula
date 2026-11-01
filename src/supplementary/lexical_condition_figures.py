"""Descriptive displays for the approved 187-site Lexical comparison."""
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib_venn import venn2, venn2_circles
import numpy as np
import pandas as pd

ROOT = Path('/cwork/ns458/insula-paper')
BASE = ROOT/'results/supplementary/fig3_lexical_delay_condition_sets'
OUT = BASE/'full_cohort_2026-10-27'
IMG = ROOT/'img/supplementary/fig3_lexical_delay_condition_sets'
AUDIT = ROOT/'results/audit/lexical_condition_sets_full_2026-10-27'
COLORS = {'Decision only':'#2166AC','Repeat only':'#D95F02','Both':'#762A83'}
COMPONENT_COLORS = {'sensory':'#2369BD','sustain':'#C4A35A','motor':'#A9373B','Unassigned':'#BDBDBD'}
LABELS = {'sensory':'Sensory','sustain':'Sustained','motor':'Motor','Unassigned':'Unassigned'}
GROUPS = ('Decision only','Both','Repeat only','Neither')
PHASES = ('stimulus','delay','response')
XLIMS = {'stimulus':(-.5,.6),'delay':(0,1),'response':(-.5,1)}
CM = 1/2.54


def load():
    plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none'})
    table = pd.read_csv(OUT/'electrodes.csv')
    waves = pd.read_csv(OUT/'waveforms.csv.gz')
    counts = pd.read_csv(OUT/'set_counts.csv')
    composition = pd.read_csv(OUT/'component_composition.csv')
    summary = json.loads((OUT/'summary.json').read_text())
    assert len(table)==187 and table.subject.nunique()==35
    return table,waves,counts,composition,summary


def save(fig,name):
    path = IMG/name
    assert path.suffix=='.svg'
    fig.savefig(path,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)
    return path


def draw_venn(table):
    decision=set(table.loc[table.decision_responsive,'electrode'])
    repeat=set(table.loc[table.repeat_responsive,'electrode'])
    numbers=(len(decision-repeat),len(repeat-decision),len(decision&repeat))
    assert numbers==(19,11,48)
    fig,ax=plt.subplots(figsize=(7.2*CM,5.4*CM))
    fig.subplots_adjust(left=.08,right=.92,bottom=.20,top=.80)
    v=venn2(subsets=numbers,set_labels=(f'Decision\n(n={len(decision)})',f'Repeat\n(n={len(repeat)})'),ax=ax)
    for key,color in zip(('10','01','11'),(COLORS['Decision only'],COLORS['Repeat only'],COLORS['Both'])):
        patch=v.get_patch_by_id(key)
        if patch is not None:patch.set_facecolor(color);patch.set_alpha(.40)
    for circle,color in zip(venn2_circles(subsets=numbers,ax=ax,lw=.8),(COLORS['Decision only'],COLORS['Repeat only'])):
        circle.set_edgecolor(color)
    for label in v.subset_labels:
        if label is not None:label.set_fontsize(8)
    for label,color in zip(v.set_labels,(COLORS['Decision only'],COLORS['Repeat only'])):
        if label is not None:label.set_fontsize(7);label.set_color(color)
    fig.suptitle('Lexical decision/repetition · Delay 0–1 s',fontsize=8,y=.98)
    fig.text(.5,.025,'187 electrodes / 35 participants\nNeither detected: 109',ha='center',fontsize=6.5)
    return save(fig,'fig3_lexical_delay_0_1_venn.svg')


def brain_renderer():
    # Reuse this notebook's accepted point display verbatim, including the
    # exact Figure 1 surface, camera-facing snap, 4-mm lift and marker size.
    from src.figure3_legacy_brain import renderer_namespace
    return renderer_namespace(ROOT)


def draw_brains(table):
    namespace=brain_renderer()
    paths=[]
    points=table.loc[table.baseline_class.ne('Neither')].copy()
    points['display_group']=points.baseline_class
    assert len(points)==78
    fig,audit=namespace['brain_pair'](points,('Decision only','Repeat only','Both'),COLORS,'Lexical · Delay 0–1 s')
    paths.append(save(fig,'fig3_lexical_delay_decision_repeat_brain.svg'))
    pd.DataFrame(audit).to_csv(OUT/'lexical_brain_display_coordinates.csv',index=False)
    for field,label,color in [('decision_gt_repeat','Decision > Repeat','#762A83'),('repeat_gt_decision','Repeat > Decision','#D95F02')]:
        points=table.loc[table[field]].copy()
        points['display_group']=label
        fig,audit=namespace['brain_pair'](points,[label],{label:color},label+' · Delay 0–1 s')
        path=save(fig,f'fig3_lexical_delay_{field}_brain.svg')
        paths.append(path)
        pd.DataFrame(audit).to_csv(OUT/f'{field}_brain_display_coordinates.csv',index=False)
    (OUT/'brain_display_provenance.json').write_text(json.dumps(dict(
        renderer=namespace['renderer_proof'],contour_helpers=namespace['contour_proof'],
        original_condition_set_brain_pair_reused_verbatim=True,
        note='Preserves current notebook point display; contours are not added by this update.',
        image_files=list(map(str,paths))),indent=2)+'\n')
    return paths


def summarize_traces(table,waves,sets):
    records=[]
    for name,selected in sets:
        identities=set(selected.electrode)
        for (phase,condition),frame in waves.loc[waves.electrode.isin(identities)].groupby(['phase','condition'],sort=False):
            values=frame.pivot(index='electrode',columns='time',values='value').reindex(sorted(identities))
            n=values.notna().sum(axis=0)
            means=values.mean(axis=0)
            sems=values.std(axis=0,ddof=1)/np.sqrt(n)
            records.append(pd.DataFrame(dict(set=name,phase=phase,condition=condition,time=values.columns.to_numpy(float),
                mean=means.to_numpy(),sem=sems.to_numpy(),n_finite_electrodes=n.to_numpy(),
                n_selected_electrodes=len(selected),n_participants=selected.subject.nunique())))
    return pd.concat(records,ignore_index=True)


def draw_traces(table,waves,sets,name,title):
    sets=[(label,part) for label,part in sets if len(part)]
    summary=summarize_traces(table,waves,sets)
    summary.to_csv(OUT/name.replace('.svg','_mean_sem.csv'),index=False)
    fig,axes=plt.subplots(len(sets),3,squeeze=False,figsize=(12.8*CM,(1.05+len(sets)*2.0)*CM),
        gridspec_kw={'width_ratios':[1.1,1.,1.5]})
    fig.subplots_adjust(left=.22,right=.90,bottom=.10,top=.90,hspace=.72,wspace=.13)
    condition_colors={'Decision':'#6A5ACD','Repeat':'#000000'}
    # Preserve a common amplitude scale across response groups/components,
    # as well as across phases, so weak responses are not visually inflated.
    bounds_sem=summary['sem'].fillna(0.)
    low=float(np.nanmin((summary['mean']-bounds_sem).to_numpy()))
    high=float(np.nanmax((summary['mean']+bounds_sem).to_numpy()))
    span=max(high-low,.10)
    ylimits=(min(0.,low)-.06*span,max(.05,high)+.08*span)
    for i,(label,selected) in enumerate(sets):
        row=summary.loc[summary['set'].eq(label)]
        for j,phase in enumerate(PHASES):
            ax=axes[i,j]
            for condition in ('Decision','Repeat'):
                data=row.loc[row.phase.eq(phase)&row.condition.eq(condition)].sort_values('time')
                t=data.time.to_numpy();m=data['mean'].to_numpy();se=data['sem'].to_numpy()
                ax.plot(t,m,color=condition_colors[condition],lw=1,label=condition)
                ax.fill_between(t,m-se,m+se,color=condition_colors[condition],alpha=.18,lw=0)
            ax.axhline(0,color='0.2',ls='--',lw=.7)
            ax.axvline(0,color='0.2',ls='--',lw=.7)
            ax.set_xlim(*XLIMS[phase]);ax.set_ylim(*ylimits)
            ticks={'stimulus':[-.5,0,.6],'delay':[0,1],'response':[-.5,0,1]}[phase]
            ax.set_xticks(ticks,labels=[f'{v:g}' for v in ticks])
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(labelsize=6,width=.75,length=2,labelbottom=i==len(sets)-1)
            if j:ax.spines['left'].set_visible(False);ax.tick_params(left=False,labelleft=False)
            if i==0:ax.set_title(phase.capitalize(),fontsize=7)
            if j==0:
                ax.set_ylabel('HGA (z)',fontsize=6)
                fig.text(.01,(ax.get_position().y0+ax.get_position().y1)/2,
                    f'{label}\nn={len(selected)}\n{selected.subject.nunique()} participants',va='center',fontsize=6.5)
        scales=[ax.get_position().width/np.diff(ax.get_xlim())[0] for ax in axes[i]]
        np.testing.assert_allclose(scales,scales[0],rtol=1e-12)
    axes[0,-1].legend(frameon=False,fontsize=6,loc='upper left',bbox_to_anchor=(1.02,1))
    axes[-1,1].set_xlabel('Time (s)',fontsize=7)
    fig.suptitle(title,fontsize=8,y=.99)
    return save(fig,name)


def draw_waveform_sets(table,waves):
    baseline_sets=[('All eligible',table)]+[(name,table.loc[table.baseline_class.eq(name)]) for name in GROUPS]
    first=draw_traces(table,waves,baseline_sets,'fig3_lexical_response_class_waveforms.svg','Baseline-response classes · native electrode mean ± SEM')
    differences=[(label,table.loc[table[field]]) for label,field in [('Decision > Repeat','decision_gt_repeat'),('Repeat > Decision','repeat_gt_decision')]]
    second=draw_traces(table,waves,differences,'fig3_lexical_condition_difference_waveforms.svg','Delay-selected sites · descriptive electrode mean ± SEM')
    component_sets=[(LABELS[c],table.loc[table.functional_cluster.eq(c)]) for c in COMPONENT_COLORS]
    third=draw_traces(table,waves,component_sets,'fig3_lexical_component_waveforms.svg','Frozen NMF labels · electrode mean ± SEM')
    return [first,second,third]


def draw_components(table,composition):
    names=[f'All {len(table)}',*GROUPS,'Decision > Repeat','Repeat > Decision']
    fig,axes=plt.subplots(1,2,figsize=(15.5*CM,7.6*CM),gridspec_kw={'width_ratios':[1.25,1.]})
    fig.subplots_adjust(left=.21,right=.98,bottom=.25,top=.82,wspace=.50)
    y=np.arange(len(names));left=np.zeros(len(names))
    for component,color in COMPONENT_COLORS.items():
        vals=composition.loc[composition.component.eq(component)].set_index('set').reindex(names).n_electrodes.to_numpy()
        axes[0].barh(y,vals,left=left,color=color,label=LABELS[component],height=.65)
        for i,(n,start) in enumerate(zip(vals,left)):
            if n:axes[0].text(start+n/2,i,str(int(n)),ha='center',va='center',fontsize=6,color='black')
        left+=vals
    axes[0].set_yticks(y,labels=names,fontsize=6.5);axes[0].invert_yaxis()
    axes[0].set_xlabel('Electrodes',fontsize=7);axes[0].set_title('Component composition',fontsize=7)
    selected=composition.loc[composition['set'].isin(['Decision > Repeat','Repeat > Decision'])]
    for i,(label,hatch) in enumerate([('Decision > Repeat',None),('Repeat > Decision','///')]):
        part=selected.loc[selected['set'].eq(label)].set_index('component').reindex(COMPONENT_COLORS)
        axes[1].bar(np.arange(len(COMPONENT_COLORS))+(i-.5)*.35,part.fraction_within_component,width=.35,
            color=list(COMPONENT_COLORS.values()),hatch=hatch,edgecolor='0.4' if hatch else 'none',lw=.5,label=label)
        for j,r in enumerate(part.itertuples()):
            axes[1].text(j+(i-.5)*.35,r.fraction_within_component+.015,f'{r.n_electrodes}/{r.component_denominator}',
                ha='center',va='bottom',fontsize=5.5,rotation=90)
    axes[1].set_xticks(range(len(COMPONENT_COLORS)),labels=[LABELS[c] for c in COMPONENT_COLORS],rotation=30,ha='right',fontsize=6)
    top=max(.3,float(selected.fraction_within_component.max())+.18)
    axes[1].set_ylim(0,top);axes[1].set_ylabel('Fraction within component',fontsize=7)
    axes[1].set_title('Detected condition differences',fontsize=7)
    axes[1].legend(frameon=False,fontsize=5.8,loc='upper left',bbox_to_anchor=(-.06,1.04))
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=6,width=.75,length=2)
    fig.legend(handles=[plt.Rectangle((0,0),1,1,color=color) for color in COMPONENT_COLORS.values()],
        labels=list(LABELS.values()),ncol=4,loc='lower center',bbox_to_anchor=(.57,.015),frameon=False,fontsize=6)
    fig.suptitle(f'{len(table)} eligible assigned sites · frozen component assignments; descriptive counts',fontsize=8,y=.99)
    return save(fig,'fig3_lexical_component_composition.svg')


def assemble(panels,name,width=650.):
    """Stack SVG panels without changing their native aspect ratios."""
    ns='http://www.w3.org/2000/svg';xl='http://www.w3.org/1999/xlink'
    ET.register_namespace('',ns);ET.register_namespace('xlink',xl)
    root=ET.Element(f'{{{ns}}}svg')
    y=0.
    for i,(path,x,w) in enumerate(panels):
        source=ET.fromstring(Path(path).read_text())
        view=list(map(float,source.attrib['viewBox'].split()))
        h=view[3]/view[2]*w
        identifiers={e.attrib['id']:f'p{i}_{e.attrib["id"]}' for e in source.iter() if 'id' in e.attrib}
        for e in source.iter():
            if 'id' in e.attrib:e.set('id',identifiers[e.attrib['id']])
            for key,value in list(e.attrib.items()):
                value=re.sub(r'url\(#([^)]*)\)',lambda m:f'url(#{identifiers.get(m.group(1),m.group(1))})',value)
                if key.endswith('href') and value.startswith('#'):value='#'+identifiers.get(value[1:],value[1:])
                e.set(key,value)
            href=e.attrib.get(f'{{{xl}}}href')
            if href and href.startswith('data:'):
                header,payload=href.split(',',1);href=header+','+''.join(payload.split())
                e.set(f'{{{xl}}}href',href);e.set('href',href)
        source.set('x',str(x));source.set('y',str(y));source.set('width',str(w));source.set('height',str(h))
        source.set('preserveAspectRatio','xMidYMid meet');root.append(source)
        y+=h+12.
    root.set('width',f'{width}pt');root.set('height',f'{y}pt');root.set('viewBox',f'0 0 {width} {y}')
    background=ET.Element(f'{{{ns}}}rect',{'width':str(width),'height':str(y),'fill':'white'})
    root.insert(0,background)
    path=IMG/name;ET.ElementTree(root).write(path,encoding='unicode',xml_declaration=True)
    return path


def assemble_condition_pair(venn,brain):
    """Refresh the original two-panel filename with the original layout."""
    nb=json.loads((AUDIT/'before/fig3_lexical_delay_condition_sets.ipynb').read_text())
    source=''.join(nb['cells'][13]['source']).split('\nsheet = assemble')[0]
    namespace=dict(ET=ET,Path=Path,re=re,IMG=IMG)
    exec(compile(source,str(AUDIT/'before/fig3_lexical_delay_condition_sets.ipynb')+':cell13','exec'),namespace)
    return namespace['assemble'](venn,[brain],'fig3_lexical_delay_0_1_condition_sets.svg')


def refresh_compatibility_exports(table):
    """Keep every existing current-result filename on the same denominator."""
    from itertools import product
    regions=[]
    for decision,repeat,difference in product((False,True),repeat=3):
        part=table.loc[table.decision_responsive.eq(decision)&table.repeat_responsive.eq(repeat)&table.decision_gt_repeat.eq(difference)]
        regions.append(dict(region=f'{int(decision)}{int(repeat)}{int(difference)}',decision=decision,repeat=repeat,
            decision_gt_repeat=difference,n_electrodes=len(part),n_participants=part.subject.nunique()))
    assert sum(r['n_electrodes'] for r in regions)==187
    pd.DataFrame(regions).to_csv(BASE/'lexical_venn_regions.csv',index=False)
    sources=pd.read_csv(OUT/'native_sources.csv')
    sources=sources.loc[sources.phase.eq('delay')].copy()
    sources['n_sites']=sources.subject.map(table.groupby('subject').size())
    sources['window_start']=0.;sources['window_end']=1.;sources['n_bins']=129
    sources.to_csv(BASE/'native_mask_sources.csv',index=False)
    for field in ('decision_gt_repeat','repeat_gt_decision'):
        source=OUT/f'{field}_brain_display_coordinates.csv'
        (BASE/f'lexical_{field}_brain_display_coordinates.csv').write_bytes(source.read_bytes())
