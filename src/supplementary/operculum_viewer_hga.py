"""Read existing bipolar HGA for anatomical review, without fitting new tests."""
import hashlib
import json
from pathlib import Path
import warnings

import mne
import numpy as np
import pandas as pd

from src.main_analysis import TASK_DISPLAY_NAMES


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def summary(times,data,n_trials,source,kind):
    display=(times>=-.5-1e-8)&(times<=1.5+1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        wave=np.nanmean(data,axis=0)[display]
    return dict(times=times[display].tolist(),mean=[float(v) if np.isfinite(v) else None for v in wave],
        n_trials=int(n_trials),source=str(source),source_kind=kind,
        condition='Repeat',modality='sound',units='HGA z score',available=True)


def subject_hga(root,subject,native_contacts,anatomy):
    """Include every recorded pair on these shafts, independent of ROI labels."""
    root=Path(root);base=root/'results/supplementary/operculum'
    receipts=[json.loads(p.read_text()) for p in sorted((base/'pairs').glob('*/completed.json'))]
    receipts=[r for r in receipts if r['pair']['subject']==subject]
    contexts={r['pair']['task']:r['pair'] for r in receipts}
    frame=pd.read_csv(root/'results/audit/sound_only_cohort_2026-10-17/coordinate_variants.csv',keep_default_na=False)
    frame=frame.loc[frame.subject.eq(subject)&frame.description.eq('Repeat')&frame.modality.eq('sound')&frame.task.isin(contexts)]
    contact_set=set(native_contacts.contact);bipoles={};membership={};issues=[]
    for r in frame.drop_duplicates(['task','channel']).itertuples():
        if r.contact_1 not in contact_set or r.contact_2 not in contact_set:continue
        if r.contact_1.rstrip('0123456789')!=r.contact_2.rstrip('0123456789'):continue
        a=np.array([r.x1_native,r.y1_native,r.z1_native],float)
        b=np.array([r.x2_native,r.y2_native,r.z2_native],float)
        xyz=np.array([r.x_native,r.y_native,r.z_native],float)
        if not (np.allclose(a,anatomy['recon_contacts'][r.contact_1],atol=1e-3) and
                np.allclose(b,anatomy['recon_contacts'][r.contact_2],atol=1e-3) and
                np.allclose((a+b)/2,xyz,atol=1e-5)):
            issues.append(dict(task=r.task,channel=r.channel,reason='native_coordinate_version_conflict'));continue
        if r.channel in bipoles:np.testing.assert_allclose(bipoles[r.channel]['xyz'],xyz,atol=1e-3)
        ijk=np.rint(np.append(xyz,1)@anatomy['inverse'].T)[:3].astype(int)
        label=anatomy['lut'].get(int(anatomy['aparc'][tuple(ijk)]),'Unknown')
        region='Frontal operculum' if label.endswith('_G_front_inf-Opercular') else 'Insula' if '_insul' in label or '_Ins_' in label else 'Other'
        bipoles[r.channel]=dict(id=subject+':midpoint:'+r.channel,name=r.channel.split('_',1)[1],channel=r.channel,
            kind='midpoint',shaft=r.contact_1.rstrip('0123456789'),xyz=xyz.tolist(),
            endpoints=[r.contact_1,r.contact_2],atlas_label=label,atlas_region=region)
        membership.setdefault(r.task,set()).add(r.channel)
    values={channel:{} for channel in bipoles};sources={};availability=[]
    for task,pair in contexts.items():
        channels=sorted(membership.get(task,set()))
        for channel in channels:values[channel][task]={}
        for phase in ['Stimulus','Delay','Response']:
            raw=Path(pair[f'{phase}_path'])
            path=Path(str(raw).replace('epoch(raw)','epoch(band)(zscore)').replace('_raw.h5','_highgamma.h5'))
            if not path.is_file():
                availability.append(dict(task=task,phase=phase,path=str(path),status='native_zscore_missing'));continue
            try:
                ep=mne.read_epochs(path,preload=False,verbose='error')
                good_codes={v for k,v in ep.event_id.items() if not k.lower().startswith('bad ')}
                if set(ep.events[:,2])-good_codes:raise ValueError('bad event retained')
                if any('image' in k.lower() or 'picture' in k.lower() or '/text/' in k.lower() for k in ep.event_id):raise ValueError('non-sound events')
                keep=[ch for ch in channels if ch in ep.ch_names and ch not in ep.info['bads']]
                for ch in channels:
                    if ch not in keep:
                        reason='原生坏道 QC 已排除' if ch in ep.info['bads'] else '原生 HGA 未保留该通道'
                        values[ch][task][phase]=dict(available=False,reason=reason)
                if keep:
                    data=ep.get_data(picks=keep,copy=True)
                    sources[str(path)]=dict(sha256=sha(path),bytes=path.stat().st_size,kind='native_zscore')
                    for i,channel in enumerate(keep):values[channel][task][phase]=summary(ep.times,data[:,i],len(ep),path,'native_zscore')
                availability.append(dict(task=task,phase=phase,path=str(path),status='read',n_channels=len(keep),n_trials=len(ep)))
                del ep
            except (OSError,ValueError,RuntimeError) as error:
                availability.append(dict(task=task,phase=phase,path=str(path),status='unavailable',reason=str(error)))
    # Retain the exact pre-orthogonalization curve when an analyzed example
    # lacks a native zscore cache. Never substitute the neighbor's waveform.
    for receipt in receipts:
        pair=receipt['pair'];task=pair['task'];pair_dir=base/'pairs'/receipt['pair_id']
        for phase in ['Stimulus','Delay','Response']:
            path=pair_dir/f'{phase}.npz'
            if not path.is_file():continue
            with np.load(path) as data:
                for channel,index in [(pair['insula_site'],0),(pair['operculum_site'],2)]:
                    if channel not in values or phase in values[channel].get(task,{}):continue
                    values[channel].setdefault(task,{})[phase]=summary(data['times'],data['zscore'][:,index],len(data['zscore']),path,'saved_before_orthogonalization')
                    sources[str(path)]=dict(sha256=sha(path),bytes=path.stat().st_size,kind='saved_before_orthogonalization')
    out=root/'results/supplementary/operculum/viewer'
    (out/f'{subject}_HGA_sources.json').write_text(json.dumps(dict(subject=subject,condition='Repeat',modality='sound',
        task_names=TASK_DISPLAY_NAMES,source_files=sources,phase_availability=availability,geometry_conflicts=issues,
        n_recorded_bipoles=len(bipoles),n_with_HGA=sum(any(any(v.get('available',False) for v in phases.values()) for phases in tasks.values()) for tasks in values.values()),
        no_new_significance_or_orthogonalization=True),indent=2)+'\n')
    return list(bipoles.values()),values
