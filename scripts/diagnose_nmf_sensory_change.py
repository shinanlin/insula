#!/usr/bin/env python3
"""Trace old sensory sites and separate cohort, signal averaging, and NMF-label changes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import pickle
import re
import sys
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/hpc/group/coganlab/nanlinshi/insula-functional')
OUT=ROOT/'results/diagnostics/nmf_sensory_comparison_2026-10-17'
TASKS=['PhonemeSequence','LexicalDelay','PictureNaming','SentenceRep']
COLORS={'sustain':'#C4A35A','motor':'#A9373B','sensory':'#2369BD'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curve(frame, *, participant=False, site=False):
    if participant or site:
        frame=frame.groupby(['subject','channel','time'],as_index=False).value.mean()
    if participant:
        frame=frame.groupby(['subject','time'],as_index=False).value.mean()
    return frame.groupby('time').value.mean().rename('value').reset_index()


def peak(frame, lo=0, hi=.6):
    segment=frame.loc[frame.time.between(lo,hi)]
    return float(segment.value.max()) if len(segment) else None


def site_peak(frame, prefix):
    rows=[]
    for channel, data in frame.groupby('channel'):
        d=data.groupby('time',as_index=False).value.mean()
        within=d.loc[d.time.between(0,.6)]
        extended=d.loc[d.time.between(0,1)]
        at=within.loc[within.value.idxmax()]
        rows.append(dict(channel=channel,**{prefix+'_peak_0_0p6':float(at.value),
            prefix+'_peak_time':float(at.time),prefix+'_mean_0_0p6':float(within.value.mean()),
            prefix+'_peak_0_1':(float(extended.value.max()) if d.time.max() >= .99 else np.nan)}))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protected=[ROOT/'vizpub/fig1.ipynb',ROOT/'vizpub/fig2.ipynb',SOURCE/'vizpub/fig2.ipynb']
    protected += [p for root in [ROOT/'results/nmf',SOURCE/'results/nmf'] for p in root.glob('*') if p.is_file()]
    before={str(p):sha(p) for p in protected}
    old=pd.read_csv(SOURCE/'results/nmf/channel_assignments.csv')
    new=pd.read_csv(ROOT/'results/nmf/channel_assignments.csv')
    short=pd.read_csv(ROOT/'docs/evidence/figure2_refresh_2026-10-17/channel_assignments.csv')
    old_sens=set(old.loc[old.functional_cluster.eq('sensory'),'channel'])
    current_ids=set(new.channel)
    wanted=set(old.channel)|current_ids
    paths=[]
    for task in TASKS:
        paths.extend(p for p in (SOURCE/'results/hga'/task).glob('sub-*/HGA/*desc-Repeat_time.csv')
                     if any(token in p.name.lower() for token in ['_proc-stimulus_','_proc-audio_']))
    def read(path):
        cols={'channel','subject','time','value','modality','task','description','roi','mix','label'}
        frame=pd.read_csv(path,usecols=lambda c:c in cols)
        return frame.loc[frame.channel.isin(wanted)&frame.time.between(-.5,1)&frame.description.eq('Repeat')]
    frames=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,frame in enumerate(pool.map(read,paths),1):
            if len(frame): frames.append(frame)
            if i%50==0: print(f'Loaded {i}/{len(paths)} stimulus files',flush=True)
    raw=pd.concat(frames,ignore_index=True)
    raw.to_pickle(OUT/'stimulus_source_rows.pkl.gz')
    sound=raw.loc[raw.modality.eq('sound')].copy()
    assert old_sens <= set(sound.channel)
    # Reproduce the current all-modality preparation, including the Figure 1 epoch supplements.
    with (ROOT/'results/fig1/available_evidence_stream.pkl').open('rb') as f:
        cached=pickle.load(f)['trace_rows']
    current_rows=cached.loc[cached.phase.eq('stimulus')].copy()
    current_rows['subject']=current_rows.channel.str.split('_').str[0]
    for item in json.loads((ROOT/'results/fig1/average_hga_supplemented_epochs.json').read_text()):
        if item['phase']!='stimulus': continue
        path=Path(item['path'])
        epochs=mne.read_epochs(path,preload=False,verbose='error')
        epochs.load_data();epochs.pick(item['channels'])
        evoked=epochs.average(method=lambda data:np.nanmean(data,axis=0))
        keep=(evoked.times>=-.5)&(evoked.times<=.6)
        frame=pd.DataFrame(evoked.data[:,keep],index=evoked.ch_names,columns=evoked.times[keep])
        frame=frame.rename_axis('channel').reset_index().melt(id_vars='channel',var_name='time',value_name='value')
        modality=re.search(r'_recording-([^_]+)',path.name)
        frame=frame.assign(subject=frame.channel.str.split('_').str[0],task=re.search(r'_task-([^_]+)',path.name)[1],
                           phase='stimulus',modality=modality[1] if modality else 'sound')
        current_rows=pd.concat([current_rows,frame],ignore_index=True)
    current_rows=current_rows.loc[current_rows.channel.isin(current_ids)]
    current_rows=current_rows.groupby(['subject','channel','task','modality','time'],as_index=False).value.mean()
    new_task=current_rows.groupby(['subject','channel','task','time'],as_index=False).value.mean()
    new_site=new_task.groupby(['subject','channel','time'],as_index=False).value.mean()
    new_sound=current_rows.loc[current_rows.modality.eq('sound')].groupby(['subject','channel','time'],as_index=False).value.mean()
    saved=pd.read_csv(ROOT/'results/nmf/site_waveforms.csv.gz')
    saved=saved.loc[saved.phase.eq('stimulus')]
    common=saved.merge(new_site,on=['subject','channel','time'],suffixes=('_saved','_rebuilt'),validate='one_to_one')
    assert len(common)==len(saved) and np.allclose(common.value_saved,common.value_rebuilt,atol=1e-12,rtol=0)
    new_site=saved[['subject','channel','time','value']]
    new_sound=new_sound.merge(new_site[['subject','channel','time']],on=['subject','channel','time'],validate='one_to_one')
    current_rows.to_csv(OUT/'current_stimulus_source_rows.csv.gz',index=False)
    # One trace record per old site, with current disposition and anatomy reason.
    sites=old.rename(columns={'functional_cluster':'old_cluster'})[['channel','subject','old_cluster']]
    sites=sites.merge(new[['channel','functional_cluster']].rename(columns={'functional_cluster':'new_cluster'}),on='channel',how='left',validate='one_to_one')
    sites=sites.merge(short[['channel','functional_cluster']].rename(columns={'functional_cluster':'short_response_cluster'}),on='channel',how='left',validate='one_to_one')
    anatomy=pd.read_csv(ROOT/'results/audit/four_task_selection_2026-10-17/site_anatomy_audit.csv')
    sites=sites.merge(anatomy[['channel','status','primary_reason','reason_zh','old_midpoint_labels']],on='channel',how='left',validate='one_to_one')
    response=pd.read_csv(ROOT/'results/audit/four_task_selection_2026-10-17/task_site_response_audit.csv')
    response_status=response.groupby('channel').response_status.agg(lambda x:'|'.join(sorted(set(x))))
    sites['response_status']=sites.channel.map(response_status)
    sites['disposition']=np.where(sites.new_cluster.notna(),'retained_'+sites.new_cluster.fillna(''),
        np.where(sites.status.eq('include'),'anatomy_eligible_without_confirmed_response',sites.status.fillna('not_in_audit')))
    sites=sites.merge(site_peak(sound,'old_sound'),on='channel',how='left')
    sites=sites.merge(site_peak(new_site,'new_all'),on='channel',how='left')
    sites=sites.merge(site_peak(new_sound,'new_sound'),on='channel',how='left')
    sites['old_sensory']=sites.old_cluster.eq('sensory')
    sites['new_all_minus_sound_peak']=sites.new_all_peak_0_0p6-sites.new_sound_peak_0_0p6
    data=np.load(ROOT/'results/nmf/input_matrix.npz')
    features=pd.read_csv(ROOT/'results/nmf/input_features.csv')
    for phase in ['stimulus','delay','response']:
        idx=features.phase.eq(phase).to_numpy()
        energy=np.square(data['X'][:,idx]).sum(axis=1)
        sites['new_'+phase+'_energy_fraction']=sites.channel.map(dict(zip(data['channels'],energy)))
    sites.sort_values('old_sound_peak_0_0p6',ascending=False).to_csv(OUT/'all_old_sites_signal_and_disposition.csv',index=False)
    trace=sites.loc[sites.old_sensory].sort_values('old_sound_peak_0_0p6',ascending=False)
    trace.to_csv(OUT/'old_sensory_sites_trace.csv',index=False)
    print('Old sensory dispositions:',trace.disposition.value_counts().to_dict(),flush=True)
    print(trace[['channel','old_sound_peak_0_0p6','new_all_peak_0_0p6','new_sound_peak_0_0p6','new_cluster','disposition','reason_zh']].head(15).to_string(index=False),flush=True)
    # A sequential display comparison; changing stages can interact, so these are not independent effect estimates.
    retained=old_sens&current_ids
    current_sens=set(new.loc[new.functional_cluster.eq('sensory'),'channel'])
    curves={
        'old Figure 2: sound, record rows, old sensory39':curve(sound.loc[sound.channel.isin(old_sens)]),
        'sound, equal sites, old sensory39':curve(sound.loc[sound.channel.isin(old_sens)],site=True),
        'sound, equal participants, old sensory39':curve(sound.loc[sound.channel.isin(old_sens)],participant=True),
        'sound, equal participants, retained old sensory':curve(new_sound.loc[new_sound.channel.isin(retained)],participant=True),
        'all modalities, equal participants, retained old sensory':curve(new_site.loc[new_site.channel.isin(retained)],participant=True),
        'current Figure 2: all modalities, new sensory44':curve(new_site.loc[new_site.channel.isin(current_sens)],participant=True),
    }
    controls=[]
    for name,frame in curves.items():
        f=frame.copy();f['comparison']=name;controls.append(f)
    pd.concat(controls).to_csv(OUT/'averaging_comparison_curves.csv',index=False)
    peaks={name:peak(frame) for name,frame in curves.items()}
    # Component H is L2-normalized and cannot be interpreted as the raw HGA amplitude.
    h_peaks=[]
    for version,folder in [('old_four_phase',SOURCE/'results/nmf'),('short_response',ROOT/'docs/evidence/figure2_refresh_2026-10-17'),('current',ROOT/'results/nmf')]:
        h=pd.read_csv(folder/'H_by_phase.csv')
        for cluster,group in h.groupby('functional_cluster'):
            stim=group.loc[group.phase.eq('stimulus')&group.time.between(0,.6)]
            h_peaks.append(dict(version=version,cluster=cluster,stimulus_H_peak=float(stim.normalized_H.max()),
                full_H_l2_norm=float(np.linalg.norm(group.normalized_H)),
                stimulus_energy_fraction=float(np.square(stim.normalized_H).sum()/np.square(group.normalized_H).sum())))
    pd.DataFrame(h_peaks).to_csv(OUT/'component_scale_comparison.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(10,3),sharex=True)
    names=list(curves)
    panels=[(names[:3],'Averaging the same old sensory sites'),(names[2:5],'Retained sites and modality changes'),([names[0],names[-1]],'Original and current Figure 2')]
    for ax,(selected,title) in zip(axes,panels):
        for name in selected:
            d=curves[name];ax.plot(d.time,d.value,lw=1.3,label=name.replace(', ','\n'))
        ax.set_xlim(0,.6);ax.set_title(title,fontsize=8);ax.set_xlabel('Stimulus time (s)',fontsize=8)
        ax.axhline(0,color='.7',ls='--',lw=.5);ax.legend(fontsize=5.5,frameon=False,loc='upper left',bbox_to_anchor=(0,-.3))
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel('Raw mean HGA (z)',fontsize=8)
    fig.subplots_adjust(bottom=.43,wspace=.3,left=.07,right=.99,top=.88)
    plt.rcParams['svg.fonttype']='none'
    fig.savefig(OUT/'sensory_peak_averaging_comparison.svg',bbox_inches='tight');plt.close(fig)
    summary=dict(n_old_sites=len(old),n_new_sites=len(new),n_shared=len(set(old.channel)&current_ids),
        n_old_sensory=len(old_sens),n_old_sensory_retained=len(retained),n_current_sensory=len(current_sens),
        old_sensory_dispositions=trace.disposition.value_counts().to_dict(),
        raw_group_peak_0_0p6=peaks,
        old_sensory_anatomy_reasons=trace.loc[trace.new_cluster.isna(),'reason_zh'].value_counts().to_dict(),
        fit_response_window=json.loads((ROOT/'results/nmf/nmf_manifest.json').read_text())['fit_windows']['response'],
        display_response_window=[-.5,1.5],current_waveform_reproduction_passed=True,
        component_scale_comparison=h_peaks,
        source_stimulus_csv_count=len(paths),current_sound_site_count=int(new_sound.channel.nunique()),
        protected_files_unchanged=all(sha(Path(p))==h for p,h in before.items()))
    assert summary['protected_files_unchanged']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'protected_hashes.json').write_text(json.dumps(before,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
