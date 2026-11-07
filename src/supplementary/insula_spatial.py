"""Frozen Hammers sampling counts and window-mean HGA for supplementary maps."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.main_analysis import NMF_WINDOWS

GYRI={
    'insula anterior pole':('Anterior pole','#E76F51'),
    'insula anterior short gyrus':('Anterior short gyrus','#F4A261'),
    'insula middle short gyrus':('Middle short gyrus','#E9C46A'),
    'insula posterior short gyrus':('Posterior short gyrus','#2A9D8F'),
    'insula anterior long gyrus':('Anterior long gyrus','#457B9D'),
    'insula posterior long gyrus':('Posterior long gyrus','#7B6DAD'),
}


def prepare_spatial_tables(root,output):
    root,output=Path(root),Path(output)
    sites=pd.read_csv(root/'results/fig1/insula_plot_coordinates.csv')
    sites=sites.rename(columns={'channel':'site_id'})
    assert sites.site_id.nunique()==len(sites)==348
    sites['hammers_gyrus']=sites.label.str.replace(r' [LR]$','',regex=True)
    sites['hemisphere']=sites.label.str.extract(r' ([LR])$',expand=False)
    assert sites.hammers_gyrus.isin(GYRI).all() and sites.hemisphere.notna().all()
    sites['gyrus_name']=sites.hammers_gyrus.map({k:v[0] for k,v in GYRI.items()})
    sites['gyrus_color']=sites.hammers_gyrus.map({k:v[1] for k,v in GYRI.items()})
    sites.to_csv(output/'hammers_sampled_site_membership.csv',index=False)
    counts=sites.groupby(['hammers_gyrus','hemisphere']).size().rename('n_sites').reindex(
        pd.MultiIndex.from_product([list(GYRI),['L','R']],names=['hammers_gyrus','hemisphere']),fill_value=0).reset_index()
    assert counts.n_sites.sum()==348
    counts['gyrus_name']=counts.hammers_gyrus.map({k:v[0] for k,v in GYRI.items()})
    counts['color']=counts.hammers_gyrus.map({k:v[1] for k,v in GYRI.items()})
    counts.to_csv(output/'hammers_gyrus_counts.csv',index=False)
    wave=pd.read_csv(root/'results/fig1/average_hga_site_waveforms.csv.gz')
    assert wave.channel.nunique()==231 and set(wave.phase)==set(NMF_WINDOWS)
    selected=[]
    for phase,(lo,hi) in NMF_WINDOWS.items():
        frame=wave.loc[wave.phase.eq(phase)&wave.time.ge(lo-1e-9)&wave.time.le(hi+1e-9)]
        mean=frame.groupby(['subject','channel']).value.agg(mean_hga='mean',n_time_samples='count').reset_index()
        assert len(mean)==231 and np.isfinite(mean.mean_hga).all()
        mean=mean.rename(columns={'channel':'site_id'}).assign(phase=phase,window_start=lo,window_end=hi)
        selected.append(mean)
    hga=pd.concat(selected,ignore_index=True)
    # Original local_insula.ipynb 2D mapping: 5..60 pt^2, zero to P95.
    # Keep a separate explicit scale in each phase, just as that reference.
    for phase,frame in hga.groupby('phase'):
        upper=float(np.percentile(frame.mean_hga,95))
        assert upper>0
        fraction=np.clip(frame.mean_hga.to_numpy()/upper,0,1)
        hga.loc[frame.index,'display_p95']=upper
        hga.loc[frame.index,'display_normalized_hga']=fraction
        hga.loc[frame.index,'marker_area_pt2']=5+fraction*55
    hga.to_csv(output/'stage_window_mean_hga.csv',index=False)
    report=dict(sampled_sites=348,hga_sites_per_phase=231,windows=NMF_WINDOWS,
        source_waveforms=str(root/'results/fig1/average_hga_site_waveforms.csv.gz'),
        waveform_scope='Same 231 sound-responsive frozen site means as Figure 1; no refitting or new subject-first averaging',
        hammers_label_source='Unique frozen Figure 1 coordinate row, with its recorded Hammers midpoint label',
        color_and_size='Both map mean HGA; original local_insula zero-to-phase-P95 clipping and 5..60 pt^2 size rule',
        scale_note='Phase-specific P95; use each phase legend for magnitude comparisons. Negative values remain in CSV and use the display minimum.',
        plotting_reference='/hpc/group/coganlab/nanlinshi/insula-aicl-delay/notebooks/local_insula.ipynb, cells 23/28',
        counts_by_gyrus=counts.groupby('gyrus_name').n_sites.sum().to_dict())
    (output/'insula_spatial_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return sites,counts,hga,report
