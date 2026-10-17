"""Audit and plot the requested D0153 RIA5-6 task HGA; run on Slurm."""
from pathlib import Path
import json
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import mne
from matplotlib import font_manager
import matplotlib.pyplot as plt
import audit_grant_d0153 as source
import plot_grant_subject_spatiotemporal as spatial
import plot_grant_ccep_receivers as ccep

ROOT=source.ROOT
OUT=ROOT/'results/grant_d0153_ria56'
FIG=ROOT/'img/grant_d0153_ria56'
CHANNEL='D0153_RIA5-6'
CONDITION='Repeat'

def condition_paths(condition):
    assert condition in ('Repeat','Decision')
    suffix='' if condition=='Repeat' else '_decision'
    return ROOT/('results/grant_d0153_ria56'+suffix),ROOT/('img/grant_d0153_ria56'+suffix)

def audit_and_summarize(condition='Repeat'):
    CONDITION=condition
    OUT,_=condition_paths(condition)
    OUT.mkdir(parents=True,exist_ok=True)
    inventory=[]
    # Check the same upstream stages before reading target epochs/statistics.
    for phase in ('baseline',)+source.PHASES:
        kinds=('raw','power') if phase=='baseline' else ('raw','power','zscore','sig')
        paths=[source.epoch_path(kind,phase,CONDITION) for kind in kinds]
        if phase!='baseline':
            paths.append(source.EPOCH/'epoch(band)(raw)'/source.epoch_path('raw',phase,CONDITION).name.replace('_raw.h5','_highgamma.h5'))
            paths.append(source.stats_path(phase,CONDITION))
        for path in paths:
            assert path.is_file() and path.stat().st_size>0,str(path)
            stat=path.stat()
            inventory.append(dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    pipeline=ccep.load_pipeline(ccep.UPSTREAM)
    anatomy,pial,anatomy_hashes=ccep.get_anatomy('D0153',['RIA5–RIA6'],pipeline)
    assert anatomy.canonical_channel.tolist()==[CHANNEL]
    pairs=pd.read_csv(source.DERIV/'qc/bipolar_pairing/sub-D0153_pairs.tsv',sep='\t')
    pair=pairs.set_index('bipolar_name').loc[CHANNEL]
    assert (pair.ch1,pair.ch2)==('RIA5','RIA6')
    anatomy.to_csv(OUT/'electrode_anatomy.csv',index=False)
    summaries=[];stats=[];intervals=[];hashes=dict(anatomy_hashes)
    for phase in source.PHASES:
        zpath=source.epoch_path('zscore',phase,CONDITION)
        spath=source.stats_path(phase,CONDITION)
        epochs=mne.read_epochs(zpath,preload=True,verbose='error')
        mask,pvals,names,sig_names=source.read_stats(spath)
        assert names==epochs.ch_names and CHANNEL in names
        assert mask.shape==pvals.shape==(len(names),len(epochs.times))
        assert epochs.info['sfreq']==128 and len(epochs)==168
        assert np.isfinite(pvals).all() and ((pvals>=0)&(pvals<=1)).all()
        idx=names.index(CHANNEL)
        x=epochs.get_data(picks=[CHANNEL])[:,0,:]
        n=np.isfinite(x).sum(axis=0)
        assert (n>=2).all() and not np.isinf(x).any()
        mean=np.nanmean(x,axis=0);sem=np.nanstd(x,axis=0,ddof=1)/np.sqrt(n)
        start,stop=epochs.time_as_index(source.TEST_WINDOWS[phase])
        in_test=np.zeros(len(epochs.times),bool);in_test[start:stop]=True
        significant=mask[idx]&(pvals[idx]<.05)&in_test
        assert bool(significant.any())==(CHANNEL in sig_names)
        sig_epochs=mne.read_epochs(source.epoch_path('sig',phase,CONDITION),preload=True,verbose='error')
        assert sig_epochs.ch_names==sig_names and len(sig_epochs)==len(epochs)
        np.testing.assert_allclose(sig_epochs.times,epochs.times)
        if CHANNEL in sig_names:
            np.testing.assert_allclose(sig_epochs.get_data(picks=[CHANNEL])[:,0,:],x,equal_nan=True)
        summaries.append(pd.DataFrame(dict(subject='D0153',channel=CHANNEL,roi='WM',condition=CONDITION,
            phase=phase,time=epochs.times,mean=mean,sem=sem,n_valid=n,n_trials=len(epochs),
            cluster_mask=mask[idx],pvalue=pvals[idx],in_analysis_window=in_test,significant=significant)))
        stats.append(dict(phase=phase,condition=CONDITION,n_trials=len(epochs),min_valid_trials=int(n.min()),
            max_valid_trials=int(n.max()),significant=bool(significant.any())))
        intervals.extend(dict(phase=phase,start_s=float(a),stop_exclusive_s=float(b))
            for a,b in spatial.intervals(significant,epochs.times))
        hashes.update({str(p):source.digest(p) for p in [zpath,spath,source.epoch_path('sig',phase,CONDITION)]})
        print(phase,stats[-1],flush=True)
        del epochs,sig_epochs
    summary=pd.concat(summaries,ignore_index=True)
    assert np.isfinite(summary[['mean','sem']]).all().all()
    assert not summary.duplicated(['phase','time']).any()
    summary.to_csv(OUT/'timecourse_mean_sem.csv',index=False)
    pd.DataFrame(stats).to_csv(OUT/'significance_summary.csv',index=False)
    pd.DataFrame(intervals,columns=['phase','start_s','stop_exclusive_s']).to_csv(OUT/'significant_intervals.csv',index=False)
    for item in inventory:
        stat=Path(item['path']).stat()
        assert (stat.st_size,stat.st_mtime_ns)==(item['bytes'],item['mtime_ns'])
    pd.DataFrame(inventory).to_csv(OUT/'file_inventory.csv',index=False)
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),subject='D0153',channel=CHANNEL,
        task='LexicalDelay',condition=CONDITION,status='passed_for_requested_visualization',
        upstream_inventory_complete=True,existing_significance_verified=True,statistics=stats,
        input_sha256=hashes,summary='Trial mean ± SEM; NaNs omitted samplewise; no additional smoothing or trial selection',
        significance_rule='Stored cluster mask AND p < .05 within upstream half-open test windows; no new tests',
        localization='Native virtual bipolar midpoint; WM, nearest gray matter MFGs; coordinates not snapped',
        scope='Task HGA response, not stimulation-locked CCEP voltage')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def render(condition='Repeat'):
    CONDITION=condition
    OUT,FIG=condition_paths(condition)
    report=json.loads((OUT/'audit.json').read_text())
    assert report['status']=='passed_for_requested_visualization'
    assert report['condition']==CONDITION
    for path,digest in report['input_sha256'].items():assert source.digest(path)==digest,path
    anatomy=pd.read_csv(OUT/'electrode_anatomy.csv')
    data=pd.read_csv(OUT/'timecourse_mean_sem.csv')
    assert set(data.condition)=={CONDITION}
    panel=ccep.render_lateral(ccep.RECON/'D153/surf/rh.pial',anatomy)
    for font in (Path(sys.prefix)/'fonts').glob('arial*.ttf'):font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none','axes.linewidth':.75})
    subject=dict(subject='D0153',anatomy=anatomy.rename(columns={'canonical_channel':'name'}),
        data=data,conditions=(CONDITION,),points=anatomy,panel=panel,cortical_localization=True)
    FIG.mkdir(parents=True,exist_ok=True)
    previous=spatial.FIG
    try:
        spatial.FIG=FIG
        path,layout=spatial.make_main(subject,channels=[CHANNEL],single_condition=CONDITION)
    finally:spatial.FIG=previous
    assert layout['ylim'][0]<=layout['envelope_min'] and layout['ylim'][1]>=layout['envelope_max']
    # Check embedded image dimensions, not just the outer SVG canvas size.
    import base64
    import struct
    import xml.etree.ElementTree as ET
    images=list(ET.parse(path).getroot().iter('{http://www.w3.org/2000/svg}image'))
    assert len(images)==1
    png=base64.b64decode(images[0].attrib['{http://www.w3.org/1999/xlink}href'].split(',')[1])
    pixels=struct.unpack('>II',png[16:24])
    assert pixels==(panel['rgb'].shape[1],panel['rgb'].shape[0]),'Brain downsampled during SVG export'
    layout.update(figure_sha256=source.digest(path),localization=report['localization'],windows=spatial.WINDOWS,
                  brain_display_zoom=1.,embedded_brain_pixels=list(pixels),native_brain_resolution_preserved=True)
    (OUT/'validation.json').write_text(json.dumps(layout,indent=2)+'\n')
    return path,layout
