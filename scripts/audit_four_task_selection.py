#!/usr/bin/env python3
"""Apply the agreed four-task anatomy/response rules; no figures or NMF.

Hammers is the primary anatomical atlas. A zero endpoint is not presumed WM.
Native FreeSurfer WM must agree across aparc, aseg, and ribbon; available
MAPER tissue must also support WM. Unknown/conflicting evidence is deferred.
All cross-task variants are retained and same-name identity conflicts deferred.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

from audit_zero_endpoints import ROOTS, TASK_KEYS, INSULA, classify, load_lut, read_recon_contact_positions

BASE = Path('/cwork/ns458/insula-paper')
SOURCE = Path('/hpc/group/coganlab/nanlinshi/insula-functional')
OUT = BASE / 'results/audit/four_task_selection_2026-10-17'
COMMON = Path('/hpc/group/coganlab/nanlinshi/seeg-preprocessing/common')
TASKS = list(ROOTS)
WINDOWS = {'stimulus': (0., .6), 'delay': (0., 1.), 'response': (-.5, .5)}
GEO = ['x_native','y_native','z_native'] + [a+str(i)+'_native' for i in [1,2] for a in 'xyz']
TOL = 1e-3
REASON_ZH = {
 'excluded_D0121_noise': 'D0121：用户确认强噪声，排除',
 'excluded_LIA6_user_CSF': 'D0125 含 LIA6：用户已确认排除；MAPER 为 CSF',
 'invalid_native_coordinates': '原生中点或端点坐标不完整／中点不是端点均值',
 'inconsistent_contact_identity': '触点名称在记录间不一致或与双极名称不符',
 'cross_task_native_coordinate_conflict': '同名双极对跨记录／任务的原生位置不一致',
 'multiple_implant_versions': '同名双极对涉及不同植入版本，暂不合并纳入',
 'unresolved_implant_version': '植入版本未确定',
 'recon_coordinate_mismatch': '历史端点坐标与现有重建触点文件不符',
 'hemisphere_conflict': '端点名称或 Hammers 标签与原生坐标左右侧冲突',
 'midpoint_noninsula': '中点的 Hammers 分区为非岛叶',
 'midpoint_CSF': '中点组织为 CSF，未满足岛叶灰质要求',
 'midpoint_WM': '中点组织为白质，未满足岛叶灰质要求',
 'midpoint_zero_unknown': '中点 Hammers 为 0，不能确认岛叶灰质',
 'midpoint_label_conflict': '中点 Hammers 是否属于岛叶的记录不一致',
 'endpoint_CSF': '至少一个端点被原生组织分割判为 CSF／脑室',
 'endpoint_noninsula_gray': '至少一个端点为非岛叶灰质',
 'endpoint_other_tissue': '至少一个端点为其他非允许组织',
 'endpoint_zero_unresolved': '端点为 0，现有组织证据不足或互相冲突',
 'endpoint_label_conflict': '同一端点的 Hammers 标签类别不一致',
 'missing_native_tissue_evidence': '端点为 0，缺少可核验的原生组织图像',
 'native_vs_archived_FS_conflict': '当前原生 FreeSurfer 与坐标匹配的历史标签不一致',
 'not_responsive': 'Repeat 三窗口均有记录，但未见显著 mask',
 'response_evidence_incomplete': '尚无显著响应证据，且 Repeat 三窗口记录不完整',
 'eligible': '满足当前解剖纳入规则',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_label(value):
    text = str(value).strip()
    return '0' if text in {'0','0.0','nan','None','','Unknown'} else text


def category(label):
    label = normalize_label(label)
    return 'insula' if label.startswith('insula ') else 'zero' if label == '0' else 'other'


def bare(value, subject):
    return str(value).removeprefix(subject + '_')


def expected_contacts(channel, subject):
    text = channel.removeprefix(subject + '_')
    if '-' not in text:
        return None
    first, second = text.rsplit('-', 1)
    if second.isdigit():
        match = re.fullmatch(r'(.*?)(\d+)', first)
        if not match:
            return None
        second = match[1] + second
    return first, second


class Evidence:
    def __init__(self, frame):
        self.lut = load_lut(COMMON / 'FreeSurferColorLUT.txt')
        self.hlut = {int(i): n for i,n in re.findall(r'<index>(\d+)</index><name>(.*?)</name>',
            (COMMON / 'Hammers_mith_atlases_n30r95_label_indices_SPM12_20160111.txt').read_text())}
        self.hlut[0] = '0'
        manual = pd.read_csv(COMMON.parent / 'qc/recon_subject_map.tsv', sep='\t')
        self.manual = {(r.task,r.subject): r.recon_subject for r in manual.itertuples()}
        self.cached = defaultdict(list)
        self.contacts = {}
        self.images = {}
        self.mapers = {}
        self.files = {}
        for task, rows in frame.groupby('task'):
            for subject in sorted(rows.subject.unique()):
                folder = ROOTS[task] / 'derivatives/parcellation' / f'sub-{subject}' / 'bipolar'
                for path in [folder/f'sub-{subject}_{suffix}.csv' for suffix in ['aparc2009s','xtractaf']]:
                    if not path.is_file():
                        continue
                    table = pd.read_csv(path, usecols=lambda c: c in {'name','x','y','z','x1','y1','z1',
                        'x2','y2','z2','contact_1','contact_2','contact_1_label','contact_2_label','center'})
                    self.record_file(path)
                    for row in table.itertuples(index=False):
                        data = row._asdict()
                        channel = str(data['name'])
                        if not channel.startswith(subject + '_'):
                            channel = subject + '_' + channel
                        self.cached[(task,channel)].append((data,str(path)))

    def record_file(self, path):
        path = Path(path)
        if str(path) not in self.files:
            self.files[str(path)] = dict(path=str(path),bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns,sha256=sha(path))

    def folder_contacts(self, folder):
        key = str(folder)
        if key not in self.contacts:
            self.contacts[key] = read_recon_contact_positions(folder)
            for p in (folder/'elec_recon').glob('*elec_locations_RAS*.txt'):
                self.record_file(p)
        return self.contacts[key]

    def identify(self, subject, task, contacts, xyz):
        mapped = self.manual.get((TASK_KEYS[task], subject))
        number = int(subject[1:].rstrip('AB'))
        ambiguous_variant = subject in {'D0107','D0139'}
        candidates = [mapped] if mapped else [f'D{number}A',f'D{number}B'] if ambiguous_variant else [f'D{number}']
        matches, present = [], []
        for recon in candidates:
            for root in [Path('/cwork/ns458/ECoG_Recon'),Path('/hpc/group/coganlab/ECoG_Recon')]:
                folder = root / recon
                positions = self.folder_contacts(folder)
                if positions:
                    present.append(str(folder))
                if all(any(np.linalg.norm(p[0]-point) <= TOL for p in positions.get(name,[]))
                       for name,point in zip(contacts,xyz)):
                    matches.append((recon,folder))
        if len({m[0] for m in matches}) > 1:
            return mapped or '',None,'unresolved_implant_version', '|'.join(present)
        if matches:
            return matches[0][0], matches[0][1], '', '|'.join(present)
        nominal = mapped or ('' if ambiguous_variant else f'D{number}')
        if present:
            return nominal,None,'recon_coordinate_mismatch','|'.join(present)
        # A consistent, ordinary reconstruction can still be identified from
        # the historical coordinates; zero endpoints will require native MRI.
        return nominal,None,('unresolved_implant_version' if not nominal else ''),''

    def image_set(self, recon, coordinate_folder):
        folders = ([coordinate_folder] if coordinate_folder else []) + [
            Path('/cwork/ns458/ECoG_Recon')/recon,Path('/hpc/group/coganlab/ECoG_Recon')/recon]
        for folder in dict.fromkeys(folders):
            if not recon or not (folder/'mri/orig.mgz').is_file() or not (folder/'mri/aparc.a2009s+aseg.mgz').is_file():
                continue
            key = str(folder)
            if key not in self.images:
                orig = nib.load(folder/'mri/orig.mgz')
                data = {'orig':orig,'inverse_tkr':np.linalg.inv(orig.header.get_vox2ras_tkr()),'folder':key}
                for name,file in [('aparc','aparc.a2009s+aseg'),('aseg','aseg'),('ribbon','ribbon')]:
                    path = folder/'mri'/f'{file}.mgz'
                    if not path.is_file():
                        continue
                    image = nib.load(path)
                    if image.shape != orig.shape or not np.allclose(image.affine,orig.affine,atol=1e-4):
                        raise ValueError(f'Geometry mismatch: {path}')
                    data[name] = np.asarray(image.dataobj)
                    self.record_file(path)
                self.record_file(folder/'mri/orig.mgz')
                self.images[key] = data
            return self.images[key]
        return None

    def maper_set(self, subject, recon, native):
        if native is None:
            return None
        key = (subject,recon,native['folder'])
        if key in self.mapers:
            return self.mapers[key]
        roots = ([Path('/cwork/ns458/maper_run/restore_20261017/D0125/output')]
                 if subject == 'D0125' and recon == 'D125' else [])
        roots += [Path('/cwork/ns458/maper_run')/subject/'output']
        for root in roots:
            seg,tissue = [root/f'f30-seg95-{subject}{suffix}.nii.gz' for suffix in ['', '-tc3crisp']]
            if seg.is_file() and tissue.is_file():
                result = {}
                for name,path in [('seg',seg),('tissue',tissue)]:
                    image = nib.load(path)
                    if image.shape[:3] != native['orig'].shape or not np.allclose(
                        image.affine,native['orig'].affine,atol=1e-4):
                        raise ValueError(f'MAPER/native geometry mismatch: {path}')
                    result[name] = np.asarray(image.dataobj).squeeze()
                    self.record_file(path)
                self.mapers[key] = result
                return result
        self.mapers[key] = None
        return None

    def point(self, subject, channel, rows, xyz, position, recon, folder):
        native = self.image_set(recon,folder)
        archived = set()
        paths = set()
        suffix = '' if position == 'midpoint' else position[-1]
        for task in rows.task.unique():
            for data,path in self.cached[(task,channel)]:
                coord = np.array([data.get(a+suffix,np.nan) for a in 'xyz'],float)
                if np.isfinite(coord).all() and np.linalg.norm(coord-xyz) <= TOL:
                    archived.add(normalize_label(data.get('center' if not suffix else f'contact_{suffix}_label','')))
                    paths.add(path)
        record = dict(channel=channel,subject=subject,position=position,
            contact=('midpoint' if position=='midpoint' else bare(rows.iloc[0][f'contact_{suffix}'],subject)),
            x_native=float(xyz[0]),y_native=float(xyz[1]),z_native=float(xyz[2]),
            recon_subject=recon,archived_FS_labels='|'.join(sorted(archived)),
            archived_FS_sources='|'.join(sorted(paths)),native_available=native is not None,
            native_FS_label='',native_FS_class='',native_aseg_label='',native_ribbon_code=None,
            native_archived_FS_conflict=False,maper_tissue_code=None,maper_raw_label='',native_source='')
        if native is not None:
            ijk = np.rint(nib.affines.apply_affine(native['inverse_tkr'],xyz)).astype(int)
            if not ((ijk>=0).all() and (ijk<np.array(native['orig'].shape)).all()):
                record['native_error'] = 'outside_image'
                return record
            index = tuple(ijk)
            label = self.lut.get(int(native['aparc'][index]),str(int(native['aparc'][index])))
            record.update(native_FS_label=label,native_FS_class=classify(label),
                native_archived_FS_conflict=bool(archived and archived!={normalize_label(label)}),native_source=native['folder'])
            if 'aseg' in native:
                record['native_aseg_label'] = self.lut.get(int(native['aseg'][index]),str(int(native['aseg'][index])))
            if 'ribbon' in native:
                record['native_ribbon_code'] = int(native['ribbon'][index])
            maper = self.maper_set(subject,recon,native)
            if maper is not None:
                record.update(maper_tissue_code=int(maper['tissue'][index]),
                              maper_raw_label=self.hlut[int(maper['seg'][index])])
        return record


def endpoint_decision(point, labels):
    cats = {category(v) for v in labels}
    tissue = point['maper_tissue_code']
    if tissue == 1:
        return 'exclude','endpoint_CSF','native_MAPER_CSF'
    if tissue == 2:
        if category(point['maper_raw_label']) == 'insula':
            return 'include','','native_MAPER_insula_GM'
        if category(point['maper_raw_label']) == 'other':
            return 'exclude','endpoint_noninsula_gray','native_MAPER_noninsula_GM'
        return 'pending','endpoint_zero_unresolved','MAPER_GM_without_region'
    if point['native_archived_FS_conflict']:
        return 'pending','native_vs_archived_FS_conflict','native_and_archived_FS_disagree'
    if tissue is not None and tissue not in {2,3}:
        return 'pending','endpoint_zero_unresolved','MAPER_background_or_invalid'
    if tissue is None and cats == {'insula'}:
        return 'include','','consistent_archived_Hammers_insula_GM'
    if tissue is None and cats == {'other'}:
        return 'exclude','endpoint_noninsula_gray','consistent_archived_Hammers_noninsula_GM'
    if tissue is None and cats != {'zero'}:
        return 'pending','endpoint_label_conflict','archived_Hammers_categories_disagree'
    if not point['native_available'] or point.get('native_error'):
        return 'pending','missing_native_tissue_evidence','native_tissue_unavailable'
    fs,aseg,ribbon = point['native_FS_class'],point['native_aseg_label'],point['native_ribbon_code']
    confirmed_wm = (fs=='white_matter' and 'Cerebral-White-Matter' in aseg and ribbon in {2,41})
    if confirmed_wm:
        return 'include','','native_MAPER_and_FS_WM' if tissue==3 else 'native_aparc_aseg_ribbon_WM'
    if tissue == 3:
        return 'pending','endpoint_zero_unresolved','MAPER_WM_without_concordant_native_FS_WM'
    if fs=='csf_or_ventricle' and any(v in aseg.lower() for v in ['ventricle','csf','choroid']):
        return 'exclude','endpoint_CSF','native_FS_CSF'
    if fs in {'opercular_gray','peri_insular_gray','other_cortical_gray'} and ribbon in {3,42} and 'Cortex' in aseg:
        return 'exclude','endpoint_noninsula_gray','native_FS_noninsula_cortical_GM'
    if point['native_FS_label'] in {'Left-Thalamus','Right-Thalamus','Left-Caudate','Right-Caudate',
                                  'Left-Putamen','Right-Putamen','Left-Pallidum','Right-Pallidum'}:
        return 'exclude','endpoint_other_tissue','native_FS_subcortical_GM'
    return 'pending','endpoint_zero_unresolved','zero_Hammers_not_confirmed_WM'


def analyze_anatomy():
    data = pd.read_pickle(OUT/'all_coordinate_rows.pkl.gz')
    data['subject'] = data.subject.astype(str).str.removeprefix('sub-')
    point_labels = ['label','contact_1_label','contact_2_label']
    potential = pd.concat([data[c].astype(str).str.startswith('insula ') for c in point_labels],axis=1).any(axis=1)
    potential |= data.roi.astype(str).str.contains('AIC|PIC',regex=True)
    known_LIA6 = data.subject.eq('D0125') & (data.contact_1.eq('LIA6') | data.contact_2.eq('LIA6'))
    names = set(data.loc[potential|known_LIA6,'channel'])
    # Once a name is a candidate in any task, retain ALL its task records.
    frame = data.loc[data.channel.isin(names)].copy()
    outside = data.loc[~data.channel.isin(names),['task','subject','channel','label','contact_1_label','contact_2_label']].drop_duplicates()
    outside['reason_zh'] = '当前三位置分区及 ROI 均无岛叶证据，未进入岛叶候选池'
    outside.to_csv(OUT/'outside_insula_candidate_pool.csv',index=False)
    fields = ['task','subject','channel','contact_1','contact_2','label','contact_1_label','contact_2_label','hemi']+GEO
    frame = frame.drop_duplicates(subset=fields)
    frame.to_csv(OUT/'candidate_coordinate_evidence.csv',index=False)
    evidence = Evidence(frame)
    site_rows,point_rows,task_rows = [],[],[]
    old = set(pd.read_csv(BASE/'results/audit/fig1_nmf_selection_2026-10-17/sampled_464_coordinates.csv').channel)
    manual26 = set((SOURCE/'results/nmf/exclude_channels.txt').read_text().split())
    print(f'Candidates: {len(names)} channel names, {len(frame)} anatomy variants',flush=True)
    for channel,rows in frame.groupby('channel',sort=True):
        first = rows.iloc[0]
        subject = first.subject
        tasks = sorted(set(rows.task),key=TASKS.index)
        g = rows[GEO].to_numpy(float)
        issue = []
        valid = np.isfinite(g).all()
        if valid:
            valid = np.allclose(g[:,:3],(g[:,3:6]+g[:,6:9])/2,atol=1e-3,rtol=0)
        if not valid:
            issue.append('invalid_native_coordinates')
        if valid and np.max(np.ptp(g,axis=0))>TOL:
            issue.append('cross_task_native_coordinate_conflict')
        expected = expected_contacts(channel,subject)
        contacts = [bare(first[f'contact_{i}'],subject) for i in [1,2]]
        if expected != tuple(contacts) or any(
            len({bare(v,subject) for v in rows[f'contact_{i}']})!=1 for i in [1,2]):
            issue.append('inconsistent_contact_identity')
        recon,folder,identity_evidence = '',None,[]
        if valid:
            for task,group in rows.groupby('task'):
                item=group.iloc[0]
                xyz=np.array([[item[a+str(i)+'_native'] for a in 'xyz'] for i in [1,2]],float)
                r,f,why,sources=evidence.identify(subject,task,contacts,xyz)
                identity_evidence.append(dict(task=task,recon=r,folder=str(f) if f else '',reason=why,
                                               checked_coordinate_folders=sources))
                if why:
                    issue.append(why)
                if r and not recon:
                    recon,folder=r,f
                elif r and recon!=r:
                    issue.append('multiple_implant_versions')
                elif folder is None and f is not None:
                    folder=f
            for i in [1,2]:
                name=contacts[i-1]
                x=float(first[f'x{i}_native'])
                if (name.startswith('L') and x>1) or (name.startswith('R') and x< -1):
                    issue.append('hemisphere_conflict')
            for column,xx in [('label','x_native'),('contact_1_label','x1_native'),('contact_2_label','x2_native')]:
                for item in rows.itertuples():
                    label=str(getattr(item,column)); x=float(getattr(item,xx))
                    if category(label)!='zero' and ((label.endswith(' L') and x>1) or (label.endswith(' R') and x< -1)):
                        issue.append('hemisphere_conflict')
        issue=list(dict.fromkeys(issue))
        is_LIA6=subject=='D0125' and 'LIA6' in contacts
        record=dict(channel=channel,subject=subject,tasks='|'.join(tasks),n_tasks=len(tasks),
            historical_464=channel in old,old_manual26=channel in manual26,
            contact_1=contacts[0],contact_2=contacts[1],recon_subject=recon,
            identity_evidence=json.dumps(identity_evidence),identity_issues='|'.join(issue),
            old_midpoint_labels='|'.join(sorted(set(map(normalize_label,rows.label)))),
            status='include',first_stage='included',primary_reason='eligible',all_reasons='',
            endpoint_1_status='',endpoint_2_status='',endpoint_1_evidence='',endpoint_2_evidence='',
            template_coordinates_available=bool(rows[['x','y','z']].notna().all(axis=1).any()))
        if subject=='D0121':
            record.update(status='exclude',first_stage='subject_QC',primary_reason='excluded_D0121_noise')
        elif is_LIA6:
            record.update(status='exclude',first_stage='known_CSF_contact',primary_reason='excluded_LIA6_user_CSF')
        elif issue:
            record.update(status='pending',first_stage='identity',primary_reason=issue[0])
        else:
            points={}
            for position,suffix in [('midpoint',''),('contact_1','1'),('contact_2','2')]:
                xyz=first[[a+suffix+'_native' for a in 'xyz']].to_numpy(float)
                p=evidence.point(subject,channel,rows,xyz,position,recon,folder)
                points[position]=p
                point_rows.append(p)
            midpoint=points['midpoint']
            cats={category(v) for v in rows.label}
            if midpoint['maper_tissue_code'] is not None:
                mid_ok=midpoint['maper_tissue_code']==2 and category(midpoint['maper_raw_label'])=='insula'
                mid_pending=not mid_ok and (midpoint['maper_tissue_code']==0 or category(midpoint['maper_raw_label'])=='zero')
            else:
                mid_ok=cats=={'insula'}
                mid_pending=cats!={'other'} and not mid_ok
            if not mid_ok:
                reason='midpoint_zero_unknown' if cats=={'zero'} else 'midpoint_label_conflict' if mid_pending else 'midpoint_noninsula'
                if midpoint['maper_tissue_code'] in {1,3}:
                    reason='midpoint_CSF' if midpoint['maper_tissue_code']==1 else 'midpoint_WM'
                record.update(status='pending' if mid_pending else 'exclude',first_stage='midpoint',primary_reason=reason)
            else:
                decisions=[]
                for i in [1,2]:
                    dec=endpoint_decision(points[f'contact_{i}'],set(rows[f'contact_{i}_label']))
                    decisions.append(dec)
                    record[f'endpoint_{i}_status']=dec[0]
                    record[f'endpoint_{i}_evidence']=dec[2]
                fail=[d for d in decisions if d[0]=='exclude']
                pending=[d for d in decisions if d[0]=='pending']
                if fail or pending:
                    chosen=(fail or pending)[0]
                    record.update(status=chosen[0],first_stage='endpoints',primary_reason=chosen[1])
                    issue.extend(d[1] for d in decisions if d[1])
        reasons=list(dict.fromkeys([record['primary_reason']]+issue))
        record['all_reasons']='|'.join(r for r in reasons if r!='eligible')
        record['reason_zh']=REASON_ZH[record['primary_reason']]
        site_rows.append(record)
        for task in tasks:
            task_rows.append(dict(task=task,**record))
    sites=pd.DataFrame(site_rows)
    task_sites=pd.DataFrame(task_rows)
    sites.to_csv(OUT/'site_anatomy_audit.csv',index=False)
    task_sites.to_csv(OUT/'task_site_anatomy_audit.csv',index=False)
    pd.DataFrame(point_rows).to_csv(OUT/'point_tissue_evidence.csv',index=False)
    sites.loc[sites.status.eq('include')].to_csv(OUT/'included_anatomy_sites.csv',index=False)
    sites.loc[sites.status.eq('exclude')].to_csv(OUT/'excluded_anatomy_sites.csv',index=False)
    sites.loc[sites.status.eq('pending')].to_csv(OUT/'pending_anatomy_sites.csv',index=False)
    # Include all raw LIA6-bearing pairs, even those never insular at midpoint.
    lia6=data.loc[known_LIA6,['task','channel','contact_1','contact_2','label','contact_1_label','contact_2_label']].drop_duplicates()
    lia6.to_csv(OUT/'D0125_LIA6_all_affected_pairs.csv',index=False)
    (OUT/'anatomy_input_manifest.json').write_text(json.dumps(list(evidence.files.values()),indent=2)+'\n')
    return data,sites,task_sites,evidence


def response_file(path, targets):
    columns=['channel','time','mask','description','phase','modality']
    frame=pd.read_csv(path,usecols=columns)
    frame=frame.loc[frame.description.eq('Repeat') & frame.channel.isin(targets)]
    frame['phase']=frame.phase.astype(str).str.lower().replace({'audio':'stimulus','resp':'response'})
    frame['time']=pd.to_numeric(frame.time,errors='coerce')
    records=[]
    for phase,(lo,hi) in WINDOWS.items():
        window=frame.loc[frame.phase.eq(phase)&frame.time.gt(lo)&frame.time.lt(hi)]
        if not len(window):
            continue
        old_window=frame.loc[frame.phase.eq(phase)&frame['mask'].eq(True)
            &frame.time.gt(0. if phase=='stimulus' else lo)&frame.time.lt(1. if phase=='stimulus' else hi)]
        old_significant_pairs=set(zip(old_window.channel,old_window.modality))
        for (channel,modality),rows in window.groupby(['channel','modality'],dropna=False):
            records.append(dict(channel=channel,phase=phase,modality=str(modality),
                significant=bool(rows['mask'].eq(True).any()),n_time_samples=int(rows.time.nunique()),
                old_window_significant=(channel,modality) in old_significant_pairs,
                t_min=float(rows.time.min()),t_max=float(rows.time.max()),source_path=str(path),evidence='packaged_mask'))
    return records


def analyze_response(task_sites):
    all_records=[]
    paths=[]
    expected_cells=defaultdict(set)
    manifests=[]
    for task in TASKS:
        for path in sorted((SOURCE/'results/hga'/task).glob('sub-*/HGA/*desc-Repeat_time.csv')):
            paths.append((task,path))
            phase_match=re.search(r'_proc-([^_]+)_',path.name)
            phase=phase_match[1].lower() if phase_match else ''
            if phase in WINDOWS:
                modality_match=re.search(r'_recording-([^_]+)_',path.name)
                modality=modality_match[1] if modality_match else 'sound'
                subject=path.name.split('_')[0].removeprefix('sub-')
                expected_cells[(task,subject)].add((phase,modality))
    def process(item):
        task,path=item
        records=response_file(path,set(task_sites.channel))
        for r in records:r['task']=task
        return records
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(item,result) in enumerate(zip(paths,pool.map(process,paths)),1):
            task,path=item
            all_records.extend(result)
            manifests.append(dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns))
            if i%100==0 or i==len(paths):print(f'Response evidence {i}/{len(paths)}',flush=True)
    response=pd.DataFrame(all_records)
    candidate_names=set(task_sites.channel)
    response=response.loc[response.channel.isin(candidate_names)]
    # Existing full-channel statistics can recover channels omitted by old
    # significant-union packaging, when the matching time grid is available.
    stats=json.loads((OUT/'statistics_inventory.json').read_text())
    stats_added=[]
    for item in stats:
        path=Path(item['path'])
        phase_match=re.search(r'_proc-([^_]+)_',path.name)
        if not phase_match:continue
        phase=phase_match[1].lower()
        if phase not in WINDOWS:continue
        task=item['task']
        subject=path.name.split('_')[0]
        matches=[p for t,p in paths if t==task and p.name==path.name.removesuffix('_highgamma.h5')+'_time.csv']
        if not matches:continue
        grid=pd.read_csv(matches[0],usecols=['time']).time.unique()
        grid=np.sort(grid.astype(float))
        with h5py.File(path,'r') as f:
            mask=f['mask'][:]
            if mask.shape[1]!=len(grid):continue
            names=[v.decode() if isinstance(v,bytes) else str(v) for v in f['ch_names'][:]]
            lo,hi=WINDOWS[phase]
            window=(grid>lo)&(grid<hi)
            if not window.any():continue
            modality_match=re.search(r'_recording-([^_]+)_',path.name)
            modality=modality_match[1] if modality_match else 'sound'
            for i,channel in enumerate(names):
                if channel not in candidate_names:continue
                stats_added.append(dict(task=task,channel=channel,phase=phase,modality=modality,
                    significant=bool(mask[i,window].any()),n_time_samples=int(window.sum()),
                    old_window_significant=bool(mask[i,(grid>(0. if phase=='stimulus' else lo))
                        &(grid<(1. if phase=='stimulus' else hi))].any()),
                    t_min=float(grid[window].min()),t_max=float(grid[window].max()),
                    source_path=str(path),evidence='full_channel_statistics_mask'))
        manifests.append(dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns))
    if stats_added:response=pd.concat([response,pd.DataFrame(stats_added)],ignore_index=True)
    response['evidence_conflict']=False
    conflicts=[]
    for keys,one in response.groupby(['task','channel','phase','modality']):
        if one.significant.nunique()>1:
            response.loc[one.index,'evidence_conflict']=True
            conflicts.append(dict(zip(['task','channel','phase','modality'],keys),
                sources='|'.join(one.source_path),reason='现有全通道统计与历史打包 mask 不一致'))
    pd.DataFrame(conflicts,columns=['task','channel','phase','modality','sources','reason']).to_csv(
        OUT/'response_mask_conflicts.csv',index=False)
    response.to_csv(OUT/'response_evidence.csv',index=False)
    records=[]
    for item in task_sites.itertuples():
        rows=response.loc[response.task.eq(item.task)&response.channel.eq(item.channel)]
        record=dict(task=item.task,channel=item.channel,subject=item.subject,anatomy_status=item.status)
        expected=expected_cells[(item.task,item.subject)]
        # Completeness is checked by phase x modality, so a negative result
        # in sound cannot mask a missing image recording, for example.
        observed={(r.phase,r.modality) for r in rows.itertuples() if not r.evidence_conflict}
        for phase in WINDOWS:
            one=rows.loc[rows.phase.eq(phase)&~rows.evidence_conflict]
            needed={cell for cell in expected if cell[0]==phase}
            record[f'{phase}_available']=bool(needed) and needed.issubset(observed)
            record[f'{phase}_significant']=bool(one.significant.any())
        any_sig=any(record[f'{p}_significant'] for p in WINDOWS)
        complete=all(record[f'{p}_available'] for p in WINDOWS)
        record.update(response_status='responsive' if any_sig else 'not_responsive' if complete else 'incomplete',
                      all_three_phases_available=complete,any_significant=any_sig,
                      old_window_any_significant=bool(rows.loc[~rows.evidence_conflict,'old_window_significant'].any()),
                      modalities='|'.join(sorted(set(rows.modality))),
                      missing_phase_modality_cells='|'.join('/'.join(c) for c in sorted(expected-observed)),
                      any_mask_conflict=bool(rows.evidence_conflict.any()))
        records.append(record)
    result=pd.DataFrame(records)
    result.to_csv(OUT/'task_site_response_audit.csv',index=False)
    (OUT/'response_input_manifest.json').write_text(json.dumps(manifests,indent=2)+'\n')
    return result


def summaries(data,sites,task_sites,response,evidence,protected):
    stages=['all_recorded','insula_related_candidates','after_subject_QC','after_LIA6_exclusion',
            'after_identity_checks','after_midpoint_rule','eligible_anatomy',
            'responsive_any_phase']
    stages_zh=['全脑有坐标表记录','涉及岛叶或指定 LIA6 的候选','排除 D0121 后','排除含 LIA6 配对后',
               '坐标／身份可统一','中点确认岛叶灰质','端点合格：解剖纳入','至少一阶段显著响应']
    counts=[]
    for task in TASKS+['ALL_UNIQUE']:
        raw=data if task=='ALL_UNIQUE' else data.loc[data.task.eq(task)]
        table=sites if task=='ALL_UNIQUE' else task_sites.loc[task_sites.task.eq(task)]
        reps=response.loc[response.anatomy_status.eq('include')]
        if task!='ALL_UNIQUE':reps=reps.loc[reps.task.eq(task)]
        values=[raw.channel.nunique(),len(table)]
        for removed in [{'subject_QC'},{'subject_QC','known_CSF_contact'},
                        {'subject_QC','known_CSF_contact','identity'},
                        {'subject_QC','known_CSF_contact','identity','midpoint'}]:
            values.append(int((~table.first_stage.isin(removed)).sum()))
        values += [int(table.status.eq('include').sum()),reps.loc[reps.any_significant,'channel'].nunique()]
        previous=None
        for stage,label,count in zip(stages,stages_zh,values):
            counts.append(dict(task=task,stage=stage,stage_zh=label,remaining=count,
                               removed_from_previous=None if previous is None else previous-count))
            previous=count
    pd.DataFrame(counts).to_csv(OUT/'stage_counts.csv',index=False)
    reason_counts=(task_sites.groupby(['task','status','first_stage','primary_reason','reason_zh'])
                   .size().reset_index(name='n_sites'))
    reason_counts.to_csv(OUT/'exclusion_counts_by_task.csv',index=False)
    phase_counts=[]
    included=set(sites.loc[sites.status.eq('include'),'channel'])
    for task in TASKS+['ALL_UNIQUE']:
        table=response.loc[response.channel.isin(included)]
        if task!='ALL_UNIQUE':table=table.loc[table.task.eq(task)]
        available=set(table.channel)
        significant=set(table.loc[table.any_significant,'channel'])
        complete=set(table.loc[table.all_three_phases_available,'channel'])
        # A fully observed task without significance cannot rule out a
        # response in another task with missing data.
        incomplete_names=set(table.loc[~table.all_three_phases_available,'channel'])-significant
        negative=(complete-significant)-incomplete_names
        row=dict(task=task,anatomy_eligible=len(available),responsive=len(significant),
                 confirmed_not_responsive=len(negative),response_incomplete=len(available-significant-negative))
        for p in WINDOWS:row[p]=table.loc[table[f'{p}_significant'],'channel'].nunique()
        phase_counts.append(row)
    pd.DataFrame(phase_counts).to_csv(OUT/'response_counts.csv',index=False)
    sites26=sites.loc[sites.old_manual26]
    sites26.to_csv(OUT/'old_manual26_reassessment.csv',index=False)
    eligible=set(sites.loc[sites.status.eq('include'),'channel'])
    responsive=set(response.loc[response.any_significant & response.channel.isin(eligible),'channel'])
    sites.loc[sites.channel.isin(responsive)].to_csv(OUT/'responsive_included_sites.csv',index=False)
    old_removed=sites.loc[sites.historical_464 & ~sites.status.eq('include')]
    old_removed.to_csv(OUT/'historical_464_removed_or_pending.csv',index=False)
    sites.loc[~sites.historical_464 & sites.status.eq('include')].to_csv(OUT/'newly_eligible_vs_464.csv',index=False)
    changed=[p for p,digest in protected.items() if sha(Path(p))!=digest]
    if changed:raise ValueError(f'Protected Figure1/NMF sources changed: {changed}')
    changed_inputs=[p for p,r in evidence.files.items()
                    if (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=(r['bytes'],r['mtime_ns'])]
    if changed_inputs:raise ValueError(f'Anatomy source inputs changed: {changed_inputs}')
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),raw_unique_names=data.channel.nunique(),
        candidate_unique_names=len(sites),candidate_task_sites=len(task_sites),
        anatomy_status=sites.status.value_counts().to_dict(),
        primary_reason_counts=sites.primary_reason.value_counts().to_dict(),
        stages=counts,response_counts=phase_counts,
        old_464_included=int((sites.historical_464 & sites.status.eq('include')).sum()),
        old_464_removed_or_pending=len(old_removed),
        newly_eligible=int((~sites.historical_464 & sites.status.eq('include')).sum()),
        old_manual26_status=sites26.status.value_counts().to_dict(),
        windows_seconds=WINDOWS,window_boundaries='Open intervals, matching historical implementation',
        response_definition='Repeat; union across input modalities; any saved significant mask in any selected phase',
        primary_atlas='Hammers; all insula labels allowed across AIC/PIC boundaries',
        zero_endpoint_rule='Native aparc+aseg+ribbon concordant WM; when available MAPER must also be WM. Missing/conflicting tissue is pending.',
        cross_task_identity_rule='Same subject and bipolar name; all geometry/version conflicts deferred rather than choosing first task',
        source_anatomy='Archived Hammers; existing native MAPER when available; native FreeSurfer for zero-endpoint tissue verification',
        figures_generated=False,nmf_rerun=False,original_site_lists_modified=False,
        protected_source_sha256=protected,input_anatomy_files=len(evidence.files),
        script_sha256=sha(Path(__file__)))
    (OUT/'anatomy_input_manifest.json').write_text(json.dumps(list(evidence.files.values()),indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


def main():
    protected_paths=[SOURCE/'vizpub/fig1.ipynb',SOURCE/'results/nmf/channel_assignments.csv',
                     SOURCE/'results/nmf/nmf_manifest.json',SOURCE/'results/nmf/exclude_channels.txt']
    protected={str(p):sha(p) for p in protected_paths}
    data,sites,task_sites,evidence=analyze_anatomy()
    response=analyze_response(task_sites)
    summaries(data,sites,task_sites,response,evidence,protected)


if __name__=='__main__':main()
