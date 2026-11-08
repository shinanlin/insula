"""Restore only the three missing Figure5 trial-level HGA checkpoints.

Uses existing clean derivatives (not raw BIDS), unchanged preprocessing
entrypoints and isolated output directories. D0107 is explicitly excluded.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import hashlib

ROOT=Path('/cwork/ns458/insula-paper')
RUN=ROOT/'results/figure56_manual14_2026-11-07'
OUT=RUN/'preprocessing'
BASE=Path('/hpc/group/coganlab/nanlinshi/seeg-preprocessing-worktrees')
CONFIG={'LexicalDelay':('BIDS-1.0_LexicalDecRepDelay','lexical_delay'),
        'PhonemeSequence':('BIDS-1.4_Phoneme_sequencing','phoneme_seq'),
        'PictureNaming':('BIDS-1.3_PictureNaming','picture_naming')}
TARGETS={('LexicalDelay','D0086'),('PhonemeSequence','D0086'),('PictureNaming','D0139')}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch():
    from globus_cli.login_manager import LoginManager
    from globus_sdk import TransferData
    client=LoginManager().get_transfer_client()
    folder=OUT/'transfer';folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'receipt.json'
    if not receipt.exists():
        files=[]
        for task in ('LexicalDelay','PhonemeSequence'):
            dataset,_=CONFIG[task]
            source=f'/C/Users/ns458/Box/CoganLab/{dataset}/BIDS/derivatives/clean/sub-D0086/ieeg/'
            entries=client.operation_ls('bfd3e08e-22c9-11f0-9e52-0ed8719d94b5',path=source)['DATA']
            for e in entries:
                name=e['name']
                if e['type']!='file' or Path(name).suffix not in ('.edf','.json','.tsv'):continue
                dest=OUT/task/'D0086'/'BIDS/derivatives/clean/sub-D0086/ieeg'/name
                files.append(dict(task=task,source=source+name,destination=str(dest),bytes=int(e['size'])))
            assert any(x['task']==task and x['source'].endswith('.edf') for x in files),f'No clean EDF for {task}/D0086'
        write(folder/'file_manifest.json',files)
        pending=[x for x in files if not Path(x['destination']).is_file() or Path(x['destination']).stat().st_size!=x['bytes']]
        if pending:
            t=TransferData('bfd3e08e-22c9-11f0-9e52-0ed8719d94b5','1ad66c7c-4f60-11e8-900c-0a6d4e044368',
                           label='Restore D0086 clean checkpoints for current Figure5 RT')
            for x in pending:
                Path(x['destination']).parent.mkdir(parents=True,exist_ok=True)
                t.add_item(x['source'],x['destination'],recursive=False)
            result=client.submit_transfer(t)
            write(receipt,dict(task_id=result['task_id'],files=len(pending),bytes=sum(x['bytes'] for x in pending)))
        else:write(receipt,dict(task_id=None,files=0,bytes=0))
    task_id=json.loads(receipt.read_text())['task_id']
    if task_id:
        for _ in range(720):
            state=client.get_task(task_id).data
            write(folder/'status.json',{k:state.get(k) for k in ['task_id','status','files_transferred','bytes_transferred','nice_status','fatal_error']})
            print(state['status'],state.get('files_transferred'),state.get('bytes_transferred'),flush=True)
            if state['status']=='SUCCEEDED':break
            if state['status'] not in ('ACTIVE','INACTIVE'):raise RuntimeError('Clean checkpoint transfer failed')
            time.sleep(10)
        else:raise RuntimeError('Transfer pending; resume existing receipt')
    files=json.loads((folder/'file_manifest.json').read_text())
    assert all(Path(x['destination']).is_file() and Path(x['destination']).stat().st_size==x['bytes'] for x in files)
    write(folder/'completed.json',dict(files=len(files),bytes=sum(x['bytes'] for x in files),job_id=os.environ['SLURM_JOB_ID']))


def stage(task,subject):
    dataset,_=CONFIG[task];source=Path('/cwork/ns458')/dataset/'BIDS'
    dest=OUT/task/subject/'BIDS';dest.mkdir(parents=True,exist_ok=True)
    for top in (Path('.'),Path('derivatives/clean')):
        for p in (source/top).glob('*'):
            if p.is_file() and p.suffix in ('.json','.tsv'):
                target=dest/top/p.name;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():shutil.copy2(p,target)
    native=source/'derivatives/clean'/f'sub-{subject}'/'ieeg'
    for p in native.glob('*'):
        if not p.is_file():continue
        target=dest/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():continue
        if p.suffix=='.edf':target.symlink_to(p.resolve())
        elif p.suffix in ('.json','.tsv'):shutil.copy2(p,target)
    if (source/'stimuli').exists() and not (dest/'stimuli').exists():
        (dest/'stimuli').symlink_to(source/'stimuli',target_is_directory=True)
    assert list((dest/'derivatives/clean'/f'sub-{subject}'/'ieeg').glob('*.edf'))
    return dest


def run(task,subject):
    assert (task,subject) in TARGETS and subject!='D0107'
    dest=stage(task,subject);_,profile=CONFIG[task];code=BASE/profile
    commands=[]
    def cmd(relative,*args):commands.append([sys.executable,str(code/relative),'--bids_root',str(dest),'--subject',subject,*args])
    if task in ('PictureNaming','PhonemeSequence'):
        cmd('task/save_delay_derivative.py');source_pipeline='delay'
    else:source_pipeline='clean'
    cmd('common/save_bipolar_derivatives.py','--source_pipeline',source_pipeline)
    if task=='PictureNaming':
        cmd('task/extract_ieeg_epochs.py','--conditions','Repeat','--phases','Delay','GoCue','Response')
    else:cmd('task/extract_ieeg_epochs.py')
    directory=dest.parent
    write(directory/'run_manifest.json',dict(task=task,subject=subject,source='Existing clean derivative checkpoints',
        commands=commands,entrypoint_sha256={c[1]:sha(c[1]) for c in commands},
        original_default_preprocessing_parameters=True,original_entrypoints_modified=False,
        new_time_permutation_tests=False,RT_requires_no_HGA_significance_gate=True,
        job_id=os.environ['SLURM_JOB_ID']))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for i,c in enumerate(commands):
        done=directory/f'step_{i:02d}.json'
        if done.exists():
            prior=json.loads(done.read_text());assert prior['command']==c and prior['entrypoint_sha256']==sha(c[1]);continue
        print('RUN',c,flush=True);subprocess.run(c,check=True,cwd=code,env=env)
        write(done,dict(command=c,entrypoint_sha256=sha(c[1]),completed_utc=datetime.now(timezone.utc).isoformat()))
    write(directory/'completed.json',dict(task=task,subject=subject,job_id=os.environ['SLURM_JOB_ID'],BIDS=str(dest)))


def repair_montage():
    """Fill only missing staged PN coordinates from the same subject's LD table.

    The author guaranteed contact identity/coordinate equality across tasks.
    Both source sidecars explicitly declare ACPC metres. Present PN values,
    clean EDF samples, events and native bad-channel flags are not modified.
    """
    import numpy as np
    import pandas as pd
    subject='D0139';dest=OUT/'PictureNaming'/subject/'BIDS'
    source=Path('/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/clean/sub-D0139/ieeg')
    src=next(source.glob('*electrodes.tsv'));src_system=json.loads(next(source.glob('*coordsystem.json')).read_text())
    reference=pd.read_csv(src,sep='\t').set_index('name')
    receipt=[];backup=OUT/'montage_recovery/D0139';backup.mkdir(parents=True,exist_ok=True)
    assert dest.is_relative_to(OUT), 'Restoration may edit only its isolated inputs'
    for pipeline in ('clean','delay'):
        folder=dest/'derivatives'/pipeline/f'sub-{subject}'/'ieeg'
        for target in folder.glob('*electrodes.tsv'):
            system=json.loads(next(folder.glob('*coordsystem.json')).read_text())
            assert system['iEEGCoordinateSystem']==src_system['iEEGCoordinateSystem']=='ACPC'
            assert system['iEEGCoordinateUnits']==src_system['iEEGCoordinateUnits']=='m'
            table=pd.read_csv(target,sep='\t');before=table.copy(deep=True)
            missing=table[['x','y','z']].isna().any(axis=1)
            changed=[]
            for index,row in table.loc[missing].iterrows():
                if row['name'] not in reference.index:continue
                xyz=reference.loc[row['name'],['x','y','z']].to_numpy(dtype=float)
                if not np.isfinite(xyz).all():continue
                for axis,value in zip('xyz',xyz):
                    if pd.isna(row[axis]):table.loc[index,axis]=value
                changed.append(str(row['name']))
            old=backup/f'{pipeline}_{target.name}'
            if not old.exists():shutil.copy2(target,old)
            pd.testing.assert_frame_equal(before.loc[~missing],table.loc[~missing])
            table.to_csv(target,sep='\t',index=False,na_rep='n/a')
            receipt.append(dict(pipeline=pipeline,target=str(target),before_sha256=sha(old),after_sha256=sha(target),filled_contacts=changed))
    bipolar=dest/'derivatives/bipolar'
    if bipolar.exists():
        previous=dest.parent/'incomplete_bipolar_before_coordinate_restore'
        assert not previous.exists();bipolar.rename(previous)
    for derived in ('epoch(bipolar)','statistics','qc'):
        path=dest/'derivatives'/derived
        if path.exists():path.rename(dest.parent/f'prior_{derived}_before_coordinate_restore')
    for index in (1,2):
        marker=dest.parent/f'step_{index:02d}.json'
        if marker.exists():marker.rename(backup/f'prior_step_{index:02d}.json')
    write(backup/'receipt.json',dict(source=str(src),source_sha256=sha(src),coordinate_system='ACPC',units='m',
        author_guaranteed_same_contact_coordinates=True,only_missing_coordinate_fields_filled=True,
        EDF_events_bad_flags_changed=False,records=receipt,job_id=os.environ['SLURM_JOB_ID']))
    print('Restored missing D0139 coordinates in isolated PN clean/delay sidecars',flush=True)


def normalize(task,subject):
    """Apply the original z-score/outlier/save steps without a new HGA test."""
    import ast
    import mne
    from ieeg.calc.scaling import rescale
    from ieeg.navigate import outliers_to_nan
    assert (task,subject) in TARGETS
    directory=OUT/task/subject
    assert (directory/'completed.json').is_file()
    bids=directory/'BIDS'
    source=BASE/CONFIG[task][1]/'lib/time_perm_stats.py'
    native=ast.parse(source.read_text())
    function=next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name=='run_stats_for_condition_phase')
    defaults=dict(zip([a.arg for a in function.args.args][-len(function.args.defaults):],function.args.defaults))
    assert ast.literal_eval(defaults['outlier_thresh'])==15
    expected=["rescale(task_epochs, baseline_epochs, mode='zscore', copy=False)",
              "outliers_to_nan(task_epochs, outliers=outlier_thresh)"]
    native_calls=[n.value for n in function.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)]
    assert all(any(ast.dump(ast.parse(call).body[0].value)==ast.dump(n) for n in native_calls) for call in expected)
    power=bids/'derivatives/epoch(bipolar)'/f'sub-{subject}'/'epoch(band)(power)'
    outputs=[]
    for phase in ('Delay','Go','Response'):
        files=sorted(power.glob(f'*_proc-{phase}_*desc-Repeat_highgamma.h5'))
        if task=='PictureNaming':files=[p for p in files if '_recording-sound_' in p.name]
        assert files,(task,subject,phase)
        for path in files:
            baseline=path.with_name(path.name.replace(f'_proc-{phase}_','_proc-baseline_'))
            assert baseline.is_file(),baseline
            task_epochs=mne.read_epochs(path,preload=True,verbose='ERROR')
            baseline_epochs=mne.read_epochs(baseline,preload=True,verbose='ERROR')
            assert task_epochs.ch_names==baseline_epochs.ch_names
            rescale(task_epochs,baseline_epochs,mode='zscore',copy=False)
            outliers_to_nan(task_epochs,outliers=15)
            target=power.parent/'epoch(band)(zscore)'/path.name
            target.parent.mkdir(parents=True,exist_ok=True)
            assert not target.exists(),f'Preserve existing normalized output:{target}'
            task_epochs.save(target,overwrite=False,verbose='ERROR')
            outputs.append(dict(path=str(target),sha256=sha(target),power_source=str(path),
                                power_sha256=sha(path),baseline=str(baseline),baseline_sha256=sha(baseline),
                                trials=len(task_epochs),channels=len(task_epochs.ch_names)))
            print('Saved original baseline z-score/outlier derivative',target,flush=True)
    write(directory/'normalization_completed.json',dict(task=task,subject=subject,
        native_source=str(source),native_source_sha256=sha(source),original_normalization_calls_verified=True,
        mode='zscore',copy=False,outlier_threshold=15,HGA_time_permutation_tests_run=False,
        outputs=outputs,job_id=os.environ['SLURM_JOB_ID']))


if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID')
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fetch','run','repair_montage','normalize']);p.add_argument('--task',choices=CONFIG);p.add_argument('--subject');a=p.parse_args()
    if a.mode=='fetch':fetch()
    elif a.mode=='repair_montage':repair_montage()
    elif a.mode=='normalize':normalize(a.task,a.subject)
    else:run(a.task,a.subject)
