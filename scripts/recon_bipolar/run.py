"""Standalone recon-only bipolar anatomy; run on Slurm, never on signals."""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from schema import MAIN_COLUMNS

PROJECT = Path(__file__).resolve().parents[2]
UPSTREAM = Path('/hpc/group/coganlab/nanlinshi/seeg-preprocessing')
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(UPSTREAM))
from screen_recon_aic_contacts import read_contacts, load_pipeline
from lib.shaft_bipolar import build_bipolar_pairs

SUBJECTS = (148, 149, 150, 152, 153, 154, 157, 158)
INSULA_SUFFIXES = {'G_Ins_lg_and_S_cent_ins', 'G_insular_short',
                   'S_circular_insula_ant', 'S_circular_insula_inf',
                   'S_circular_insula_sup'}


def is_insula(label):
    return label.startswith(('ctx_lh_', 'ctx_rh_')) and label[7:] in INSULA_SUFFIXES


def merge_label(aparc, hammers):
    """Aparc defines the replacement mask; Hammers never expands that mask."""
    h_insula = hammers.startswith('insula ')
    if not is_insula(aparc):
        return aparc, 'aparc2009s', 'hammers_insula_outside_aparc' if h_insula else ''
    hemi = 'L' if aparc.startswith('ctx_lh_') else 'R'
    if h_insula and hammers.endswith(' ' + hemi):
        return hammers, 'hammers', ''
    return aparc, 'aparc2009s', 'insula_subdivision_unresolved'


def make_pairs(subject, contacts):
    groups, coords = {}, {}
    for c in contacts:
        if c['contact_type'] != 'D':
            raise ValueError('Non-depth contacts require an explicit pairing policy')
        groups.setdefault(c['shaft'], []).append((c['contact'], c['contact_number']))
        # Canonical library expects montage meters. Explicit conversion prevents
        # its legacy magnitude heuristic from rescaling native-mm points near 0.
        coords[c['contact']] = tuple(np.array([c['x'], c['y'], c['z']]) / 1000)
    for group in groups.values():
        group.sort(key=lambda x: x[1])
    return build_bipolar_pairs(subject, groups, coords, bads=set(),
                               max_dist_mm=20, max_turn_deg=60)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def select_recons(root):
    selected, inventory = {}, []
    for number in SUBJECTS:
        candidates = []
        for folder in sorted(root.glob(f'D{number}*')):
            if re.fullmatch(fr'D{number}[AB]?', folder.name):
                contacts, path, _ = read_contacts(folder)
                candidates.append((folder.name, contacts, path))
        if not candidates:
            raise FileNotFoundError(f'No recon for D{number}')
        chosen = max(candidates, key=lambda x: (len(x[1]), x[0]))
        selected[f'D{number:04d}'] = chosen
        inventory.extend(dict(subject=f'D{number:04d}', recon_subject=name,
                              n_contacts=len(rows), selected=name == chosen[0])
                         for name, rows, _ in candidates)
    return selected, inventory


def process(pipeline, subject, selected, root, maper, output):
    recon, contacts, coord_path = selected
    maper_id = 'D0152A' if recon in {'D152A', 'D152B'} else subject
    if recon == 'D152B':
        for name in ('orig.mgz', 'brainmask.mgz'):
            if sha(root / 'D152A/mri' / name) != sha(root / 'D152B/mri' / name):
                raise ValueError('Cannot share D152A segmentation: MRI identity changed')
    common = str(UPSTREAM / 'common')
    aparc = pipeline._resolve_atlas('aparc2009s', str(root), recon, subject, common)
    hammers = pipeline._resolve_atlas('hammers', str(root), recon, maper_id,
                                      common, maper_run=str(maper))
    crosswalk = dict(aparc[3])
    # The original aparc crosswalk omits the long insular gyrus. Handle all
    # five Destrieux insular parcels explicitly when Hammers cannot refine it.
    for side in ('lh', 'rh'):
        for suffix in INSULA_SUFFIXES:
            crosswalk[f'ctx_{side}_{suffix}'] = 'INS'
    crosswalk.update({k: v for k, v in hammers[3].items() if k.startswith('insula ')})

    def label(xyz):
        a = pipeline._label_coordinate(xyz, aparc[1], aparc[0], aparc[2])
        h = pipeline._label_coordinate(xyz, hammers[1], hammers[0], hammers[2])
        merged, source, flag = merge_label(a, h)
        roi, _ = pipeline.consensus_roi([merged], crosswalk)
        return dict(label=merged, roi=roi, source=source, qc=flag,
                    aparc_label=a, hammers_label=h)

    lookup = {c['contact']: c for c in contacts}
    point_labels = {name: label(np.array([c['x'], c['y'], c['z']]))
                    for name, c in lookup.items()}
    build = make_pairs(subject, contacts)
    rows = []
    for (first, second), pair in zip(build.pairs, build.pair_rows):
        c1, c2 = lookup[first], lookup[second]
        if c1['recon_hemi'] != c2['recon_hemi']:
            raise ValueError(f'Cross-hemisphere pair: {first}, {second}')
        p1, p2 = (np.array([c['x'], c['y'], c['z']]) for c in (c1, c2))
        mid = (p1 + p2) / 2
        if not np.isclose(np.linalg.norm(p2 - p1), pair['dist_mm'], atol=1e-8):
            raise ValueError('Pair spacing unit mismatch')
        points = [point_labels[first], label(mid), point_labels[second]]
        roi, mix = pipeline.consensus_roi([p['label'] for p in points], crosswalk)
        # Full, lossless anatomy names; EDF-compatible upstream alias is audit only.
        name = f'{subject}_{c1["shaft"]}{c1["contact_number"]}-{c2["contact_number"]}'
        row = dict(subject=subject, recon_subject=recon, name=name,
                   contact_1=first, contact_2=second,
                   contact_1_label=points[0]['label'], center=points[1]['label'],
                   contact_2_label=points[2]['label'], roi=roi, mix=mix,
                   hemi=c1['recon_hemi'], spacing_mm=pair['dist_mm'],
                   shaft=c1['shaft'], segment_id=pair['segment_id'],
                   contact_number_gap=c2['contact_number'] - c1['contact_number'],
                   upstream_edf_name=pair['bipolar_name'],
                   aic_consensus=roi == 'AIC' and not mix,
                   aic_any_point=any(p['roi'] == 'AIC' for p in points),
                   aic_midpoint=points[1]['roi'] == 'AIC',
                   aic_all_three=all(p['roi'] == 'AIC' for p in points))
        for suffix, xyz in (('', mid), ('1', p1), ('2', p2)):
            row.update({axis + suffix: float(val) for axis, val in zip('xyz', xyz)})
        for prefix, point in zip(('contact_1', 'midpoint', 'contact_2'), points):
            row.update({f'{prefix}_{key}': val for key, val in point.items()
                        if key != 'label'})
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty or df.name.duplicated().any():
        raise ValueError('Empty or duplicate bipolar names')
    destination = output / subject
    destination.mkdir()
    qc = destination / 'qc'
    qc.mkdir()
    public = df.rename(columns={'center': 'midpoint_label'})
    public[MAIN_COLUMNS].to_csv(destination / f'sub-{subject}_bipolar_parcellation.csv', index=False)
    audit_columns = ['subject', 'name'] + [c for c in public.columns if c not in MAIN_COLUMNS]
    public[audit_columns].to_csv(qc / 'bipolar_pair_audit.csv', index=False)
    for name, records, cols in (
        ('segments', build.segments, ['subject', 'prefix', 'segment_id', 'flagged']),
        ('skipped', build.skipped_breaks, ['subject', 'prefix', 'ch1', 'ch2', 'reason']),
        ('orphans', build.orphans, ['subject', 'prefix', 'segment_id', 'contact']),
    ):
        pd.DataFrame(records, columns=None if records else cols).to_csv(qc / f'{name}.csv', index=False)
    pd.DataFrame([dict(**c, **point_labels[c['contact']]) for c in contacts]).to_csv(
        qc / 'contact_label_audit.csv', index=False)
    summary = dict(subject=subject, recon_subject=recon, n_contacts=len(contacts),
                   n_bipolar=len(df), n_aic_consensus=int(df.aic_consensus.sum()),
                   n_aic_any_point=int(df.aic_any_point.sum()),
                   n_aic_midpoint=int(df.aic_midpoint.sum()),
                   n_aic_all_three=int(df.aic_all_three.sum()),
                   n_mixed=int(df.mix.sum()), n_skipped_breaks=len(build.skipped_breaks),
                   n_orphans=len(build.orphans),
                   n_flagged_segments=sum(s['flagged'] for s in build.segments),
                   n_number_gaps=int(df.contact_number_gap.ne(1).sum()),
                   n_unresolved_insula_pairs=int(df.filter(regex='_qc$').eq('insula_subdivision_unresolved').any(axis=1).sum()),
                   n_hammers_outside_aparc_pairs=int(df.filter(regex='_qc$').eq('hammers_insula_outside_aparc').any(axis=1).sum()),
                   aic_consensus_pairs=';'.join(df.loc[df.aic_consensus, 'name']))
    paths = [coord_path, root / recon / 'mri/orig.mgz',
             root / recon / 'mri/aparc.a2009s+aseg.mgz']
    paths += [maper / maper_id / 'output' / f'f30-seg95-{maper_id}{suffix}.nii.gz'
              for suffix in ('', '-tc3crisp')]
    return summary, dict(recon_subject=recon, maper_id=maper_id,
                         source_sha256={str(p): sha(p) for p in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pipeline = load_pipeline(UPSTREAM / 'common/parcellation.py')
    root, maper = Path('/cwork/ns458/ECoG_Recon'), Path('/cwork/ns458/maper_run')
    selected, inventory = select_recons(root)
    pd.DataFrame(inventory).to_csv(args.output / 'recon_selection.csv', index=False)
    summaries, sources = [], []
    for subject, candidate in selected.items():
        summary, source = process(pipeline, subject, candidate, root, maper, args.output)
        summaries.append(summary)
        sources.append(source)
        print(json.dumps(summary), flush=True)
    pd.DataFrame(summaries).to_csv(args.output / 'coverage_summary.csv', index=False)
    code = [Path(__file__), Path(__file__).with_name('schema.py'), PROJECT / 'scripts/screen_recon_aic_contacts.py',
            UPSTREAM / 'lib/shaft_bipolar.py', UPSTREAM / 'common/parcellation.py',
            UPSTREAM / 'common/a2009s.csv', UPSTREAM / 'common/hammers.csv',
            UPSTREAM / 'common/FreeSurferColorLUT.txt',
            UPSTREAM / 'common/Hammers_mith_atlases_n30r95_label_indices_SPM12_20160111.txt']
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Recon-only anatomical bipolar pairs; no signals, task selection, or bad-channel exclusions.',
        coordinates='Native FreeSurfer tkRAS millimeters; no MNI/CVS coordinates.',
        pairing='Canonical 3D shaft pairing: 20 mm distance, 60 degree turn. Sorted adjacent available contacts; numeric gaps retained and flagged.',
        atlas='aparc2009s base; only its five insular parcels can be refined by same-hemisphere GM-gated Hammers insula. Unresolved points retain aparc label and INS ROI.',
        consensus='Canonical endpoint1/midpoint/endpoint2 consensus; WM and Unknown ignored. AIC consensus is not equivalent to all three points in AIC.',
        sources=sources, code_sha256={str(p): sha(p) for p in code})
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (args.output / 'COMPLETE').write_text('All eight subjects completed.\n')


if __name__ == '__main__':
    main()
