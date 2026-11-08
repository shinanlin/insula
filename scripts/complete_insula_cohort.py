"""Complete the author-reviewed insula cohort from saved native evidence.

Run through Slurm. This builds an executable new cohort/NMF input version;
it does not run new time-permutation tests or rewrite existing figures.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import sys
import warnings

import h5py
import mne
import numpy as np
import pandas as pd

ROOT = Path('/cwork/ns458/insula-paper')
sys.path.insert(0, str(ROOT))
from src.main_analysis import CONDITION, MODALITY, NMF_WINDOWS, RESPONSE_WINDOWS, TASKS

OLD = ROOT/'docs/evidence/sound_only_cohort_2026-10-17'
OUT = ROOT/'docs/evidence/insula_completion_2026-11-07'
RUN = ROOT/'results/insula_completion_2026-11-07'
NMF = RUN/'nmf'
DATASETS = {'PhonemeSequence':'BIDS-1.4_Phoneme_sequencing',
            'LexicalDelay':'BIDS-1.0_LexicalDecRepDelay',
            'PictureNaming':'BIDS-1.3_PictureNaming',
            'SentenceRep':'BIDS-1.4_SentenceRep'}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def entity(path, key):
    match = re.search(r'(?:^|_)'+key+r'-([^_.]+)', path.name)
    return match[1] if match else ''


def phase_name(path):
    phase = entity(path, 'proc').lower()
    return {'audio':'stimulus','resp':'response'}.get(phase, phase)


def name(value, subject):
    value = value.decode() if isinstance(value, bytes) else str(value)
    return value if value.startswith(subject+'_') else subject+'_'+value


def native_product(task, subject, phase):
    roots = []
    if task == 'PictureNaming' and phase == 'stimulus':
        roots.append(ROOT/'results/native_stimulus_event_repair_2026-10-21/PictureNaming'/subject/'BIDS')
    roots.extend([RUN/'restored_native'/task/subject/'BIDS',
                  ROOT/'results/native_preprocessing_restore_2026-10-18'/task/subject/'BIDS',
                  Path('/cwork/ns458')/DATASETS[task]/'BIDS'])
    for bids in roots:
        def selected(p):
            return (p.is_file() and 'highgamma' in p.name and entity(p, 'desc') == CONDITION
                and phase_name(p) == phase
                and (task != 'PictureNaming' or entity(p, 'recording') == MODALITY))
        eps = [p for p in (bids/'derivatives/epoch(bipolar)'/f'sub-{subject}'/'epoch(band)(zscore)').glob('*') if selected(p)]
        sts = [p for p in (bids/'derivatives/statistics'/f'sub-{subject}'/'bipolar').glob('*') if selected(p)]
        if eps or sts:
            if len(eps) != 1 or len(sts) != 1:
                return None, None, 'partial_or_ambiguous_native_product:'+str(bids)
            return eps[0], sts[0], ''
    return None, None, 'no_native_product'


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run in Slurm'
    assert CONDITION == 'Repeat' and MODALITY == 'sound'
    for folder in (OUT, RUN, NMF):
        folder.mkdir(parents=True, exist_ok=True)
    snapshot = RUN/'before'
    if not snapshot.exists():
        snapshot.mkdir()
        shutil.copytree(OLD, snapshot/'cohort')
        (snapshot/'nmf').mkdir()
        for p in (ROOT/'results/nmf').iterdir():
            if p.is_file():
                shutil.copy2(p, snapshot/'nmf'/p.name)
        for p in [ROOT/'src/main_analysis.py', ROOT/'docs/MAIN_ANALYSIS.md']:
            shutil.copy2(p, snapshot/p.name)
        figures = [p for p in (ROOT/'img').glob('fig[1-6]/*') if p.is_file()]
        figures += list((ROOT/'vizpub').glob('fig[1-6].ipynb'))
        write(snapshot/'protected_figures.json', {str(p):sha(p) for p in figures})
    protected = json.loads((snapshot/'protected_figures.json').read_text())
    anatomy = pd.read_csv(OLD/'site_anatomy_audit.csv', keep_default_na=False)
    old_sample = pd.read_csv(OLD/'sampled_sites_by_subject_task.csv', keep_default_na=False)
    old_nmf = pd.read_csv(snapshot/'nmf/input_sites.csv', keep_default_na=False)
    review = json.loads((ROOT/'config/nmf_anatomy_review_2026-11-07.json').read_text())
    approved = set().union(*(set(value) for key, value in review.items() if key.startswith('accepted_') and key.endswith('_ids')))
    assert len(approved) == review['accepted_n'] == 30
    old_ids = set(old_sample.site_id)
    assert not approved & old_ids
    accepted = old_ids | approved
    mask = anatomy.channel.isin(approved)
    assert int(mask.sum()) == 30
    anatomy.loc[mask, 'status'] = 'include'
    anatomy.loc[mask, 'first_stage'] = 'author_review_2026-11-07'
    anatomy.loc[mask, 'primary_reason'] = 'author_approved_anatomical_inclusion'
    anatomy.loc[mask, 'reason_zh'] = '作者已明确确认纳入；保留原始端点标签及证据'
    anatomy['author_reviewed_inclusion'] = anatomy.channel.isin(approved)
    anatomy.to_csv(OUT/'site_anatomy_audit.csv', index=False)
    coords = pd.read_pickle(ROOT/'results/audit/sound_only_cohort_2026-10-17/all_coordinate_rows.pkl.gz')
    coords = coords.loc[coords.description.eq(CONDITION) & coords.modality.eq(MODALITY) & coords.channel.isin(accepted)]
    # The author confirmed coordinate/label identity; choose recorded rows without rechecking identity.
    pairs = coords.sort_values(['task','channel','phase']).drop_duplicates(['task','channel']).copy()
    assert set(pairs.channel) == accepted and len(accepted) == 378
    pairs = pairs.merge(anatomy[['channel','status','primary_reason','author_reviewed_inclusion',
        'has_accepted_unknown_endpoint','accepted_unknown_contacts']], on='channel', validate='many_to_one')
    pairs['old_midpoint_labels'] = pairs['label']
    pairs.to_csv(OUT/'task_site_anatomy_audit.csv', index=False)
    qc = pd.read_csv(ROOT/'results/audit/preprocessing_restore_2026-10-18/restored_site_eligibility.csv', keep_default_na=False)
    bad_keys = set(zip(qc.loc[qc.restore_status.eq('original_clean_bad_endpoint'),'task'],
                       qc.loc[qc.restore_status.eq('original_clean_bad_endpoint'),'site_id']))
    settings = json.loads((ROOT/'config/insula_completion_2026-11-07.json').read_text())
    deferred = set(settings['deferred_preprocessing_subjects'])
    additional_qc = []
    # The old QC registry covers the old anatomical sample. Add original clean
    # metadata evidence for newly accepted sites without changing any QC flag.
    for (task, subject), group in pairs.loc[pairs.channel.isin(approved)].groupby(['task','subject']):
        folder = Path('/cwork/ns458')/DATASETS[task]/'BIDS/derivatives/clean'/f'sub-{subject}'/'ieeg'
        paths = sorted(folder.glob('*channels.tsv'))
        restored_metadata = RUN/'preprocessed_recovery/checkpoint_metadata'/task/subject
        if restored_metadata.is_dir():
            paths = sorted(restored_metadata.glob('*channels.tsv'))
        tables = [(p,pd.read_csv(p,sep='\t',keep_default_na=False).set_index('name')) for p in paths]
        for site in group.itertuples():
            decisions = []
            for p, table in tables:
                contacts = [site.contact_1,site.contact_2]
                if not all(c in table.index for c in contacts):
                    continue
                bad = [c for c in contacts if str(table.loc[c,'status'])=='bad']
                decisions.append(bool(bad))
                additional_qc.append(dict(task=task,subject=subject,channel=site.channel,source_path=str(p),
                    bad_contacts='|'.join(bad),bad_reasons='|'.join(str(table.loc[c].get('status_description','')) for c in bad)))
            if decisions and all(decisions):
                bad_keys.add((task,site.channel))
    pd.DataFrame(additional_qc).to_csv(OUT/'approved30_original_clean_QC.csv',index=False)
    rows, wave_rows, sources, gaps = [], [], [], []
    input_hashes = {}

    for (task, subject), group in pairs.groupby(['task','subject'], sort=True):
        requested = set(group.channel)
        good = requested-{channel for t, channel in bad_keys if t == task}
        print('Collecting', task, subject, len(requested), 'anatomical sites', flush=True)
        if subject in deferred:
            for phase in RESPONSE_WINDOWS:
                for channel in requested:
                    rows.append(dict(task=task,subject=subject,channel=channel,phase=phase,modality=MODALITY,
                        available=False,significant=False,status='author_deferred_preprocessing',source_path='',
                        evidence='author_deferred_preprocessing'))
                    gaps.append(dict(task=task,subject=subject,channel=channel,phase=phase,reason='author_deferred_preprocessing'))
            continue
        for phase, selection_window in RESPONSE_WINDOWS.items():
            available = {}
            for channel in requested-good:
                rows.append(dict(task=task, subject=subject, channel=channel, phase=phase, modality=MODALITY,
                    available=False, significant=False, status='original_native_QC_exclusion', source_path='', evidence='native_QC'))
            if not good:
                continue
            epoch_path, stats_path, issue = native_product(task, subject, phase)
            if epoch_path is not None:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        ep = mne.read_epochs(epoch_path, preload=False, verbose='ERROR')
                    ep_names = [name(n, subject) for n in ep.ch_names]
                    times = ep.times.copy()
                    with h5py.File(stats_path, 'r') as h:
                        stat_names = [name(n, subject) for n in h['ch_names'][:]]
                        assert stat_names == ep_names and h['mask'].shape == h['pvals'].shape == (len(ep_names), len(times))
                        keep = (times > selection_window[0]) & (times < selection_window[1])
                        assert keep.any()
                        bad_names = {name(n, subject) for n in ep.info['bads']}
                        present = sorted(good & set(ep_names)-bad_names)
                        for channel in good:
                            if channel not in present:
                                available[channel] = (False, False, 'target_absent_from_native_good_channels')
                                continue
                            i = ep_names.index(channel)
                            # Preserve the original main-cohort rule. The Figure3
                            # pointwise p<.05 overlay is a separate analysis rule.
                            saved = np.asarray(h['mask'][i], bool)
                            available[channel] = (True, bool(saved[keep].any()), 'native_timeperm_H5')
                        # Load only required channels, retaining the original native trial/QC selection.
                        if present:
                            picks = [ep_names.index(channel) for channel in present]
                            data = ep.get_data(picks=picks, copy=True, verbose='ERROR')
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', RuntimeWarning)
                                means = np.nanmean(data, axis=0)
                            lo, hi = NMF_WINDOWS[phase]
                            display = (times >= min(-.5,lo)) & (times <= max(hi,1. if phase=='response' else hi))
                            for channel, values in zip(present, means):
                                wave_rows.append(pd.DataFrame(dict(subject=subject, channel=channel, task=task,
                                    phase=phase, modality=MODALITY, time=times[display], value=values[display])))
                            del data, means
                        sources.append(dict(task=task, subject=subject, phase=phase, modality=MODALITY,
                            epoch_path=str(epoch_path), statistics_path=str(stats_path), trials=len(ep),
                            source_type='native_timeperm_H5', n_requested=len(good), n_present=len(present),
                            included_channels='|'.join(present)))
                    for p in (epoch_path, stats_path):
                        input_hashes[str(p)] = sha(p)
                    del ep
                except Exception as exc:
                    raise RuntimeError(f'Native product failed: {task}/{subject}/{phase}: {exc}') from exc
            elif issue == 'no_native_product':
                # Existing sound-only packaged waveforms are retained only when no native product exists.
                folder = Path('/hpc/group/coganlab/nanlinshi/insula-functional/results/hga')/task/f'sub-{subject}'/'HGA'
                files = [p for p in folder.glob('*desc-Repeat_time.csv') if phase_name(p) == phase]
                for p in files:
                    columns = {'channel','time','value','mask','pvals','p_value','modality','description'}
                    parts = []
                    for chunk in pd.read_csv(p, usecols=lambda c:c in columns, chunksize=200000):
                        keep = chunk.channel.isin(good)
                        if 'modality' in chunk:
                            keep &= chunk.modality.eq(MODALITY)
                        if 'description' in chunk:
                            keep &= chunk.description.eq(CONDITION)
                        if keep.any():
                            parts.append(chunk.loc[keep].copy())
                    if not parts:
                        continue
                    frame = pd.concat(parts, ignore_index=True)
                    for channel, one in frame.groupby('channel'):
                        selected = one.time.gt(selection_window[0]) & one.time.lt(selection_window[1])
                        significant = one['mask'].eq(True)
                        available[channel] = (bool(selected.any()), bool((selected & significant).any()), 'retained_packaged_sound_mask')
                    wave_rows.append(frame[['channel','time','value']].assign(subject=subject, task=task, phase=phase, modality=MODALITY))
                    sources.append(dict(task=task, subject=subject, phase=phase, modality=MODALITY,
                        epoch_path='', statistics_path=str(p), trials='', source_type='retained_packaged_sound_mask',
                        n_requested=len(good), n_present=frame.channel.nunique(), included_channels='|'.join(sorted(frame.channel.unique()))))
                    input_hashes[str(p)] = sha(p)
            for channel in good:
                present, significant, status = available.get(channel, (False, False, issue or 'target_missing'))
                rows.append(dict(task=task, subject=subject, channel=channel, phase=phase, modality=MODALITY,
                    available=present, significant=significant, status=status,
                    source_path=str(stats_path) if stats_path else '', evidence=status))
                if not present:
                    gaps.append(dict(task=task, subject=subject, channel=channel, phase=phase, reason=status))

    evidence = pd.DataFrame(rows)
    assert not evidence.duplicated(['task','channel','phase']).any()
    evidence.to_csv(OUT/'response_evidence.csv', index=False)
    pd.DataFrame(sources).to_csv(OUT/'native_input_manifest.csv', index=False)
    pd.DataFrame(gaps, columns=['task','subject','channel','phase','reason']).to_csv(OUT/'missing_evidence.csv', index=False)
    response = []
    for (task, subject, channel), frame in evidence.groupby(['task','subject','channel']):
        one = dict(task=task, subject=subject, channel=channel, modality=MODALITY, anatomy_status='include')
        native_bad = (task, channel) in bad_keys
        for phase in RESPONSE_WINDOWS:
            part = frame.loc[frame.phase.eq(phase)]
            one[phase+'_available'] = bool(part.available.all())
            one[phase+'_significant'] = bool(part.significant.any())
        one['all_three_phases_available'] = all(one[p+'_available'] for p in RESPONSE_WINDOWS)
        one['any_significant'] = any(one[p+'_significant'] for p in RESPONSE_WINDOWS)
        one['native_QC_excluded'] = native_bad
        one['subject_preprocessing_deferred'] = subject in deferred
        one['response_status'] = ('deferred_preprocessing' if subject in deferred else 'native_QC_excluded' if native_bad else 'responsive' if one['any_significant'] else
            'not_responsive' if one['all_three_phases_available'] else 'incomplete')
        response.append(one)
    response = pd.DataFrame(response)
    response.to_csv(OUT/'task_site_response_audit.csv', index=False)
    sample = pairs[['subject','task','channel','contact_1','contact_2','old_midpoint_labels',
                    'has_accepted_unknown_endpoint','accepted_unknown_contacts','author_reviewed_inclusion']].merge(response, on=['task','subject','channel'], validate='one_to_one')
    sample = sample.rename(columns={'channel':'site_id','old_midpoint_labels':'hammers_midpoint_labels',
        'any_significant':'confirmed_responsive','response_status':'response_evidence_status',
        'all_three_phases_available':'response_evidence_complete'})
    sample.to_csv(OUT/'sampled_sites_by_subject_task.csv', index=False)
    sample.loc[sample.confirmed_responsive].to_csv(OUT/'responsive_sites_by_subject_task.csv', index=False)
    unique = []
    for site, g in sample.groupby('site_id'):
        responsive = g.loc[g.confirmed_responsive]
        unique.append(dict(subject=g.subject.iloc[0], site_id=site,
            sampled_in_tasks='|'.join(sorted(g.task.unique())), responsive_in_tasks='|'.join(sorted(responsive.task.unique())),
            confirmed_responsive=bool(len(responsive)), contact_1=g.contact_1.iloc[0], contact_2=g.contact_2.iloc[0],
            has_accepted_unknown_endpoint=bool(g.has_accepted_unknown_endpoint.any()),
            accepted_unknown_contacts=g.accepted_unknown_contacts.iloc[0],
            all_usable_tasks_complete=bool(g.loc[~g.native_QC_excluded].response_evidence_complete.all()) if (~g.native_QC_excluded).any() else False))
    unique = pd.DataFrame(unique)
    unique.to_csv(OUT/'sampled_sites_unique.csv', index=False)
    candidates = unique.loc[unique.confirmed_responsive].copy()
    candidates.to_csv(OUT/'nmf_candidate_sites_not_final_inputs.csv', index=False)
    task_counts = sample.groupby(['subject','task']).agg(n_sampled_bipolar_sites=('site_id','size'),
        n_confirmed_responsive_bipolar_sites=('confirmed_responsive','sum'), n_QC_excluded=('native_QC_excluded','sum')).reset_index()
    task_counts.to_csv(OUT/'subject_task_counts_and_lists.csv', index=False)
    old_nmf_ids = set(old_nmf.channel)
    delta = unique.copy()
    delta['in_previous_NMF'] = delta.site_id.isin(old_nmf_ids)
    delta['newly_anatomically_accepted'] = delta.site_id.isin(approved)
    delta['change'] = np.select([~delta.in_previous_NMF & delta.confirmed_responsive,
        delta.in_previous_NMF & ~delta.confirmed_responsive], ['new_responsive_site','previous_NMF_site_now_not_responsive'], default='retained_or_not_responsive')
    delta.to_csv(OUT/'cohort_changes.csv', index=False)
    all_waves = pd.concat(wave_rows, ignore_index=True)
    all_waves = all_waves.loc[all_waves.channel.isin(set(candidates.site_id))]
    all_waves = all_waves.groupby(['subject','channel','task','phase','modality','time'], as_index=False).value.mean()
    all_waves.groupby(['subject','channel','task','phase','modality']).agg(n_time_samples=('time','nunique'),
        n_finite_values=('value','count')).reset_index().to_csv(NMF/'waveform_source_membership.csv', index=False)
    site_waves = all_waves.groupby(['subject','channel','phase','time'], as_index=False).value.mean()
    site_waves.to_csv(NMF/'site_waveforms.csv.gz', index=False)
    channels = sorted(candidates.site_id)
    blocks, features, missing_wave = [], [], set()
    for phase, window in NMF_WINDOWS.items():
        part = site_waves.loc[site_waves.phase.eq(phase) & site_waves.time.between(*window)]
        times = np.sort(part.time.unique())
        # Original shared 128-Hz grid, including both approved window endpoints.
        expected = np.arange(np.ceil(window[0]*128), np.floor(window[1]*128)+1)/128
        times = expected
        matrix = part.pivot(index='channel', columns='time', values='value').reindex(index=channels, columns=times)
        absent = ~np.isfinite(matrix.to_numpy()).all(axis=1)
        missing_wave.update(np.asarray(channels)[absent])
        blocks.append(matrix.to_numpy(float))
        features.extend(dict(phase=phase, time=float(t)) for t in times)
    raw = np.concatenate(blocks, axis=1)
    finite = np.isfinite(raw).all(axis=1)
    positive = np.clip(raw[finite], 0, None)
    norms = np.linalg.norm(positive, axis=1)
    nonzero = norms > np.finfo(float).eps
    included = np.asarray(channels)[finite][nonzero]
    X = positive[nonzero]/norms[nonzero,None]
    final_raw = raw[finite][nonzero]
    metadata = pairs.sort_values(['task','channel']).drop_duplicates('channel').set_index('channel').loc[included].reset_index()
    metadata.to_csv(NMF/'input_sites.csv', index=False)
    pd.DataFrame(features).to_csv(NMF/'input_features.csv', index=False)
    np.savez_compressed(NMF/'input_matrix.npz', X=X, raw=final_raw, channels=included, positive_l2_norm=norms[nonzero])
    candidates['NMF_input_complete'] = candidates.site_id.isin(set(included))
    candidates['NMF_exclusion_reason'] = np.where(candidates.NMF_input_complete, '',
        np.where(candidates.site_id.isin(missing_wave), 'missing_or_nonfinite_waveform_in_approved_fit_window', 'zero_rectified_waveform'))
    candidates.to_csv(OUT/'nmf_candidate_sites_not_final_inputs.csv', index=False)
    candidates.loc[~candidates.NMF_input_complete].to_csv(OUT/'responsive_sites_missing_NMF_waveform.csv', index=False)
    manifest = dict(condition=CONDITION, modality=MODALITY, n_sites=len(included), n_subjects=metadata.subject.nunique(),
        n_features=X.shape[1], phases=list(NMF_WINDOWS), fit_windows=NMF_WINDOWS, significance_windows=RESPONSE_WINDOWS,
        weighting='equal available native-good task means within each site; electrode unit',
        preprocessing='negative HGA clipped at zero; full concatenated waveform L2 normalization',
        native_response_rule='original main-cohort saved time_perm mask in the approved open selection windows; no extra pointwise gate',
        native_QC_preserved=True, no_new_timeperm_tests=True,
        input_sha256=sha(NMF/'input_matrix.npz'), source_cohort=str(OUT), phase_samples=pd.DataFrame(features).groupby('phase').size().to_dict())
    write(NMF/'input_manifest.json', manifest)
    summary = dict(condition=CONDITION, modality=MODALITY, sampled_sites=len(unique), sampled_subjects=unique.subject.nunique(),
        previously_sampled_sites=len(old_ids), author_approved_added_anatomical_sites=len(approved),
        responsive_sites=len(candidates), responsive_subjects=candidates.subject.nunique(),
        previous_NMF_sites=len(old_nmf_ids), added_responsive_sites=len(set(candidates.site_id)-old_nmf_ids),
        previous_NMF_sites_no_longer_responsive=len(old_nmf_ids-set(candidates.site_id)),
        NMF_input_sites=len(included), NMF_input_subjects=metadata.subject.nunique(),
        responsive_sites_without_complete_NMF_waveform=int((~candidates.NMF_input_complete).sum()),
        native_QC_excluded_task_sites=int(response.native_QC_excluded.sum()),
        deferred_preprocessing_subjects=sorted(deferred),
        incomplete_response_sites=int((~unique.confirmed_responsive & ~unique.all_usable_tasks_complete).sum()),
        confirmed_nonresponsive_sites=int((~unique.confirmed_responsive & unique.all_usable_tasks_complete).sum()),
        phase_counts={p:int(sample.loc[sample[p+'_significant'],'site_id'].nunique()) for p in RESPONSE_WINDOWS},
        selection_windows=RESPONSE_WINDOWS, job_id=os.environ['SLURM_JOB_ID'], completed_utc=datetime.now(timezone.utc).isoformat())
    write(OUT/'summary.json', summary)
    write(RUN/'input_provenance_sha256.json', input_hashes)
    for p, digest in protected.items():
        assert sha(p) == digest, 'Existing figure changed: '+p
    assert len(included) > 0 and np.isfinite(X).all() and (X >= 0).all()
    assert np.allclose(np.linalg.norm(X,axis=1), 1) and 'D0121' not in set(metadata.subject)
    assert not (set(zip(sample.loc[sample.confirmed_responsive,'task'],sample.loc[sample.confirmed_responsive,'site_id'])) & bad_keys)
    with pd.ExcelWriter(OUT/'insula_completion_counts_and_changes.xlsx', engine='openpyxl') as writer:
        for title, frame in [('sampling',unique),('task_sites',sample),('changes',delta),('NMF_candidates',candidates),('missing_evidence',pd.DataFrame(gaps))]:
            frame.to_excel(writer, sheet_name=title, index=False)
    write(RUN/'preparation_completed.json', dict(passed=True, summary=summary, input_manifest=manifest,
        figures_unchanged=True, old_canonical_NMF_unchanged=True))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
