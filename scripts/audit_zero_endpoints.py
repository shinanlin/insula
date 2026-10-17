#!/usr/bin/env python3
"""Audit the 132 bipolar pairs with Hammers-zero endpoint labels.

Original native positions are immutable inputs. Archived aparc labels are used
only after coordinate matching; surviving native volumes provide an additional
voxel check. Missing MAPER products remain explicitly missing. No NMF is run.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np
import pandas as pd

BASE = Path('/cwork/ns458/insula-paper')
SOURCE = BASE / 'results/audit/fig1_nmf_selection_2026-10-17/sampled_464_coordinates.csv'
OUT = BASE / 'results/audit/zero_endpoints_2026-10-17'
UPSTREAM = Path('/hpc/group/coganlab/nanlinshi/seeg-preprocessing')
ROOTS = {
    'PhonemeSequence': Path('/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS'),
    'LexicalDelay': Path('/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS'),
    'PictureNaming': Path('/cwork/ns458/BIDS-1.3_PictureNaming/BIDS'),
    'SentenceRep': Path('/cwork/ns458/BIDS-1.4_SentenceRep/BIDS'),
}
TASK_KEYS = {'LexicalDelay': 'lexical_delay', 'PictureNaming': 'picture_naming',
             'SentenceRep': 'sentence_rep', 'PhonemeSequence': 'phoneme_seq'}
INSULA = {'G_insular_short', 'G_Ins_lg_and_S_cent_ins', 'S_circular_insula_ant',
          'S_circular_insula_inf', 'S_circular_insula_sup'}
OPERCULUM = {'G_front_inf-Opercular', 'G_and_S_subcentral'}
PERI_INSULA = {'G_precentral', 'G_postcentral', 'G_pariet_inf-Supramar',
              'G_temp_sup-Lateral', 'G_temp_sup-Plan_polar', 'G_temp_sup-Plan_tempo',
              'G_front_inf-Triangul', 'S_precentral-inf-part', 'S_postcentral'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(label):
    text = str(label)
    if text in {'', 'nan', 'Unknown', '0', '0.0'}:
        return 'unknown'
    if 'Cerebral-White-Matter' in text:
        return 'white_matter'
    fine = re.sub(r'^ctx_[lr]h_', '', text)
    if fine in INSULA:
        return 'insula_gray'
    if fine in OPERCULUM:
        return 'opercular_gray'
    if fine in PERI_INSULA:
        return 'peri_insular_gray'
    if text.startswith('ctx_'):
        return 'other_cortical_gray'
    if any(w in text.lower() for w in ['ventricle', 'csf', 'choroid']):
        return 'csf_or_ventricle'
    return 'other_tissue'


def load_lut(path):
    d = {}
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) > 1 and p[0].isdigit():
            d[int(p[0])] = p[1]
    return d


def recon_for(subject, task, manual):
    key = (TASK_KEYS[task], subject)
    if key in manual:
        return manual[key]
    if subject in {'D0107', 'D0139'}:
        return None
    return f'D{int(subject[1:])}'


def read_recon_contact_positions(folder):
    result = defaultdict(list)
    elec = folder / 'elec_recon'
    for path in sorted(elec.glob('*elec_locations_RAS*.txt')):
        for line in path.read_text(errors='replace').splitlines():
            p = line.split()
            try:
                if len(p) >= 5 and p[1].isdigit():
                    name, xyz = p[0] + p[1], np.array(p[2:5], dtype=float)
                elif len(p) >= 4:
                    name, xyz = p[0], np.array(p[1:4], dtype=float)
                else:
                    continue
                result[name].append((xyz, str(path)))
            except ValueError:
                continue
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x = pd.read_csv(SOURCE)
    x = x.loc[x.label.astype(str).str.startswith('insula ')].copy()
    z1 = pd.to_numeric(x.contact_1_label, errors='coerce').eq(0)
    z2 = pd.to_numeric(x.contact_2_label, errors='coerce').eq(0)
    pairs = x.loc[z1 | z2].copy()
    assert len(pairs) == 132
    pairs.to_csv(OUT / 'target_132_pairs.csv', index=False)
    manual_table = pd.read_csv(UPSTREAM / 'qc/recon_subject_map.tsv', sep='\t')
    manual = {(r.task, r.subject): r.recon_subject for r in manual_table.itertuples()}
    lut = load_lut(UPSTREAM / 'common/FreeSurferColorLUT.txt')
    hammers_lut = {int(k): v for k, v in re.findall(
        r'<index>(\d+)</index><name>(.*?)</name>',
        (UPSTREAM / 'common/Hammers_mith_atlases_n30r95_label_indices_SPM12_20160111.txt').read_text())}
    manifests = []
    cached = defaultdict(list)
    subjects = sorted(set(pairs.subject))
    for subject in subjects:
        for task, root in ROOTS.items():
            folder = root / 'derivatives/parcellation' / f'sub-{subject}' / 'bipolar'
            candidates = [folder / f'sub-{subject}_{a}.csv' for a in ['aparc2009s', 'xtractaf']]
            for path in candidates:
                if not path.is_file():
                    continue
                table = pd.read_csv(path, usecols=lambda c: c in {
                    'name', 'x', 'y', 'z', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2',
                    'contact_1', 'contact_2', 'contact_1_label', 'contact_2_label', 'center'})
                manifests.append(dict(path=str(path), bytes=path.stat().st_size,
                                      mtime_ns=path.stat().st_mtime_ns, sha256=sha(path)))
                for _, row in table.iterrows():
                    for i in [1, 2]:
                        xyz = row.reindex([f'{a}{i}' for a in 'xyz']).to_numpy(float)
                        name = str(row.get(f'contact_{i}', ''))
                        if name.startswith(subject + '_'):
                            name = name[len(subject) + 1:]
                        if np.isfinite(xyz).all():
                            cached[(subject, name)].append(dict(
                                task=task, path=str(path), xyz=xyz,
                                label=str(row.get(f'contact_{i}_label', ''))))

    # A row per subject and native reconstruction variant; missing inputs are
    # recorded without silently substituting a different A/B reconstruction.
    availability = []
    combinations = sorted({(r.subject, recon_for(r.subject, r.task, manual))
                           for r in pairs.itertuples()}, key=str)
    for subject, recon in combinations:
        cwork = Path('/cwork/ns458/ECoG_Recon') / str(recon)
        shared = Path('/hpc/group/coganlab/ECoG_Recon') / str(recon)
        maper = Path('/cwork/ns458/maper_run') / subject / 'output'
        entry = dict(subject=subject, recon_subject=recon,
                     n_target_pairs=int(pairs.subject.eq(subject).sum()))
        for tag, folder in [('cwork', cwork), ('shared', shared)]:
            for filename, col in [('orig.mgz', 'orig'), ('brainmask.mgz', 'brainmask'),
                                  ('aparc.a2009s+aseg.mgz', 'aparc'), ('aseg.mgz', 'aseg'),
                                  ('wm.mgz', 'wm'), ('ribbon.mgz', 'ribbon')]:
                path = folder / 'mri' / filename
                entry[f'{tag}_{col}'] = path.is_file()
                entry[f'{tag}_{col}_path'] = str(path)
            entry[f'{tag}_electrode_RAS'] = any((folder / 'elec_recon').glob('*elec_locations_RAS*.txt'))
        for suffix, col in [('', 'maper_seg95'), ('-tc3crisp', 'maper_tissue')]:
            path = maper / f'f30-seg95-{subject}{suffix}.nii.gz'
            entry[col] = path.is_file()
            entry[col + '_path'] = str(path)
        for task, root in ROOTS.items():
            entry['raw_T1_' + task] = any((root / f'sub-{subject}' / 'anat').glob('*T1w.nii*'))
        availability.append(entry)
    avail = pd.DataFrame(availability)
    avail.to_csv(OUT / 'anatomy_availability.csv', index=False)

    all_records = []
    by_recon = defaultdict(list)
    for _, pair in pairs.iterrows():
        by_recon[(pair.subject, recon_for(pair.subject, pair.task, manual))].append(pair)
    for (subject, recon), records in by_recon.items():
        # Prefer the original cwork tree, then a matching shared recon folder.
        available = []
        for folder in [Path('/cwork/ns458/ECoG_Recon') / str(recon),
                       Path('/hpc/group/coganlab/ECoG_Recon') / str(recon)]:
            if (folder / 'mri/orig.mgz').is_file() and (folder / 'mri/aparc.a2009s+aseg.mgz').is_file():
                available.append(folder)
        folder = available[0] if available else None
        orig = aparc = adata = recon_positions = None
        if folder:
            orig = nib.load(folder / 'mri/orig.mgz')
            aparc = nib.load(folder / 'mri/aparc.a2009s+aseg.mgz')
            if orig.shape != aparc.shape or not np.allclose(orig.affine, aparc.affine, atol=1e-4):
                raise ValueError(f'Native FreeSurfer geometry mismatch for {subject}/{recon}')
            adata = np.asarray(aparc.dataobj)
            inv_tkr = np.linalg.inv(orig.header.get_vox2ras_tkr())
            recon_positions = read_recon_contact_positions(folder)
            for path in [folder / 'mri/orig.mgz', folder / 'mri/aparc.a2009s+aseg.mgz']:
                manifests.append(dict(path=str(path), bytes=path.stat().st_size,
                                      mtime_ns=path.stat().st_mtime_ns, sha256=sha(path)))
        maper_path = Path('/cwork/ns458/maper_run') / subject / 'output' / f'f30-seg95-{subject}.nii.gz'
        tissue_path = maper_path.with_name(f'f30-seg95-{subject}-tc3crisp.nii.gz')
        hdata = tissue = None
        if orig and maper_path.is_file() and tissue_path.is_file():
            for path in [maper_path, tissue_path]:
                vol = nib.load(path)
                if vol.shape[:3] != orig.shape[:3] or not np.allclose(vol.affine, orig.affine, atol=1e-2):
                    raise ValueError(f'MAPER/native geometry mismatch: {path}')
            hdata = np.asarray(nib.load(maper_path).dataobj).squeeze()
            tissue = np.asarray(nib.load(tissue_path).dataobj).squeeze()
        for pair in records:
            for i in [1, 2]:
                if not pd.to_numeric(pd.Series([pair[f'contact_{i}_label']]), errors='coerce').eq(0).iloc[0]:
                    continue
                xyz = pair[[f'{a}{i}_native' for a in 'xyz']].to_numpy(float)
                contact = str(pair[f'contact_{i}']).removeprefix(subject + '_')
                result = dict(channel=pair.channel, subject=subject, source_task=pair.task,
                              recon_subject=recon, endpoint=i, contact=contact,
                              x_native=xyz[0], y_native=xyz[1], z_native=xyz[2],
                              old_hammers_label='0',
                              hemisphere_name_vs_x_disagreement=(
                                  (contact.startswith('L') and xyz[0] > 0) or
                                  (contact.startswith('R') and xyz[0] < 0)),
                              native_source_path=pair.source_path)
                candidates = []
                for c in cached[(subject, contact)]:
                    distance = float(np.linalg.norm(xyz - c['xyz']))
                    if distance <= 1e-3:
                        candidates.append((c, distance))
                preferred = [c for c in candidates if c[0]['task'] == pair.task] or candidates
                labels = sorted({c[0]['label'] for c in preferred})
                label = labels[0] if len(labels) == 1 else ''
                result.update(cached_aparc_label=label,
                              cached_label_alternatives='|'.join(labels),
                              cached_class=classify(label),
                              cached_coord_match=bool(preferred),
                              cached_sources='|'.join(sorted({c[0]['path'] for c in preferred})),
                              cached_tasks='|'.join(sorted({c[0]['task'] for c in preferred})),
                              cached_max_coord_delta_mm=max([c[1] for c in preferred], default=None))
                result.update(native_volume_available=bool(folder), native_aparc_label='',
                              native_class='unavailable', native_voxel_checked=False,
                              backup_contact_coord_delta_mm=None, backup_coordinate_verified=False,
                              native_aparc_path=str(folder / 'mri/aparc.a2009s+aseg.mgz') if folder else '',
                              maper_tissue_class=None, maper_ungated_label='',
                              maper_rechecked=False)
                if folder:
                    ijk_cont = nib.affines.apply_affine(inv_tkr, xyz)
                    ijk = np.rint(ijk_cont).astype(int)
                    if (ijk >= 0).all() and (ijk < np.asarray(orig.shape)).all():
                        lab = lut.get(int(adata[tuple(ijk)]), str(int(adata[tuple(ijk)])))
                        distances = [float(np.linalg.norm(xyz - q[0])) for q in recon_positions.get(contact, [])]
                        delta = min(distances, default=None)
                        result.update(native_aparc_label=lab, native_class=classify(lab),
                                      native_voxel_checked=True,
                                      voxel_i=int(ijk[0]), voxel_j=int(ijk[1]), voxel_k=int(ijk[2]),
                                      backup_contact_coord_delta_mm=delta,
                                      backup_coordinate_verified=(delta is not None and delta <= 1e-3),
                                      native_matches_cached=(lab == label if label else None))
                        if hdata is not None:
                            raw = int(hdata[tuple(ijk)])
                            result.update(maper_tissue_class=int(tissue[tuple(ijk)]),
                                          maper_ungated_label=hammers_lut.get(raw, str(raw)),
                                          maper_rechecked=True)
                # Label uncertainty is preserved: a matching historical label
                # is evidence, not a replacement for missing MAPER tissue maps.
                effective = (result['native_class'] if result['backup_coordinate_verified']
                             else result['cached_class'])
                result['best_available_class'] = effective
                result['evidence_level'] = ('native_voxel_and_contact_coordinate_match'
                    if result['backup_coordinate_verified'] else
                    'coordinate_matched_archived_aparc_label' if label else 'unresolved')
                result['requires_maper_or_image_review'] = not result['maper_rechecked']
                all_records.append(result)
        print(f'{subject}/{recon}: sampled {sum(r["subject"] == subject for r in all_records)} zero-endpoint occurrences', flush=True)
    endpoints = pd.DataFrame(all_records)
    assert len(endpoints) == 144
    endpoints.to_csv(OUT / 'zero_endpoint_occurrences.csv', index=False)
    # Anatomical points are deduplicated by contact name AND native position.
    endpoints['physical_point_key'] = endpoints.apply(lambda r:
        f'{r.subject}_{r.contact}:' + ','.join(f'{r[a + "_native"]:.4f}' for a in 'xyz'), axis=1)
    unique = []
    for key, group in endpoints.groupby('physical_point_key', sort=True):
        record = group.iloc[0].to_dict()
        record.update(physical_point_key=key, channels='|'.join(sorted(set(group.channel))),
                      n_pair_occurrences=len(group), source_tasks='|'.join(sorted(set(group.source_task))))
        classes = set(group.best_available_class)
        if len(classes) > 1:
            record['best_available_class'] = 'conflicting_evidence'
        unique.append(record)
    contacts = pd.DataFrame(unique)
    contacts.to_csv(OUT / 'unique_zero_contacts.csv', index=False)
    pair_records = []
    for channel, group in endpoints.groupby('channel', sort=True):
        classes = list(group.best_available_class)
        pair_records.append(dict(channel=channel, subject=group.subject.iloc[0],
             n_zero_endpoints=len(group), zero_contacts='|'.join(group.contact),
             zero_classes='|'.join(classes),
             all_zero_endpoints_WM_by_available_evidence=all(c == 'white_matter' for c in classes),
             any_opercular_gray_evidence=any(c == 'opercular_gray' for c in classes),
             any_other_noninsula_gray_evidence=any(c in {'peri_insular_gray','other_cortical_gray'} for c in classes),
             all_zero_endpoints_native_verified=bool(group.backup_coordinate_verified.all()),
             maper_tissue_verified=bool(group.maper_rechecked.all())))
    pair_summary = pd.DataFrame(pair_records)
    pair_summary.to_csv(OUT / 'pair_summary.csv', index=False)
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), model_fitted=False,
        target_pairs=len(pairs), target_subjects=len(subjects), zero_endpoint_occurrences=len(endpoints),
        unique_subject_contacts=endpoints[['subject','contact']].drop_duplicates().shape[0],
        unique_native_points=len(contacts),
        cached_coord_matched_occurrences=int(endpoints.cached_coord_match.sum()),
        cached_classes=endpoints.cached_class.value_counts().to_dict(),
        best_available_classes_unique_points=contacts.best_available_class.value_counts().to_dict(),
        native_voxel_checked_unique_points=int(contacts.native_voxel_checked.sum()),
        native_coord_verified_unique_points=int(contacts.backup_coordinate_verified.sum()),
        maper_verified_unique_points=int(contacts.maper_rechecked.sum()),
        native_vs_cached_label_disagreements=endpoints.loc[
            endpoints.native_voxel_checked & endpoints.cached_aparc_label.ne('') &
            endpoints.native_aparc_label.ne(endpoints.cached_aparc_label),
            ['channel','contact','cached_aparc_label','native_aparc_label']].to_dict('records'),
        pairs_WM_by_available_evidence=int(pair_summary.all_zero_endpoints_WM_by_available_evidence.sum()),
        pairs_opercular_gray_evidence=int(pair_summary.any_opercular_gray_evidence.sum()),
        pairs_other_noninsula_gray_evidence=int(pair_summary.any_other_noninsula_gray_evidence.sum()),
        cwork_inputs_subject_variants=avail[['subject','recon_subject']].to_dict('records'),
        availability_counts={c:int(avail[c].sum()) for c in avail.columns
                             if avail[c].dtype == bool},
        source_table_sha256=sha(SOURCE), script_sha256=sha(Path(__file__)))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'input_manifest.json').write_text(json.dumps(manifests, indent=2) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
