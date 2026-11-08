"""Refresh every Figure 1 panel for the author-selected 67-subject cohort.

Run via Slurm. Reuse reviewed anatomy, native QC, saved statistics and existing
sound waveforms. Figure 1 response selection remains independent of NMF.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import pickle
import shutil
import sys
import warnings

ROOT = Path('/cwork/ns458/insula-paper')
sys.path.insert(0, str(ROOT))
VERSION = 'figure1_cohort67_2026-11-07'
RUN = ROOT / 'results' / VERSION
COHORT = ROOT / 'docs/evidence' / VERSION
SOURCE_COHORT = ROOT / 'docs/evidence/insula_manual14_2026-11-07'
NB = ROOT / 'vizpub/fig1.ipynb'
OUT = ROOT / 'results/fig1'
FIG = ROOT / 'img/fig1'
DISPLAY = {'stimulus': (-.5, .6), 'delay': (0., 1.), 'response': (-.5, 1.5)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def snapshot():
    before = RUN / 'before'
    if before.exists():
        return
    before.mkdir(parents=True)
    targets = [NB, ROOT/'AGENTS.md', ROOT/'docs/MAIN_ANALYSIS.md',
               ROOT/'src/main_analysis.py', ROOT/'config/current_insula_analysis.json',
               ROOT/'scripts/execute_fig1_notebook.py']
    targets += [p for folder in (OUT, FIG) for p in folder.iterdir() if p.is_file()]
    for path in targets:
        dest = before / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    protected = []
    for pattern in ('img/fig[2-6]/**/*', 'vizpub/fig[2-6].ipynb',
                    'results/nmf/**/*', 'results/insula_completion_2026-11-07/nmf/**/*',
                    'results/insula_manual14_2026-11-07/nmf/**/*', 'results/fig2/**/*',
                    'docs/evidence/insula_manual14_2026-11-07/**/*'):
        protected += [p for p in ROOT.glob(pattern) if p.is_file()]
    source = Path('/hpc/group/coganlab/nanlinshi/insula-functional')
    protected += [source/'vizpub/fig1.ipynb']
    protected += [p for p in (source/'results/nmf').glob('*') if p.is_file()]
    write(RUN/'protected_files.json', {str(p): sha(p) for p in sorted(set(protected))})


def build_cohort():
    import pandas as pd
    from src.main_analysis import TASKS
    COHORT.mkdir(parents=True, exist_ok=True)
    for name in ('sampled_sites_unique.csv', 'sampled_sites_by_subject_task.csv',
                 'responsive_sites_by_subject_task.csv', 'task_site_response_audit.csv',
                 'site_anatomy_audit.csv', 'response_evidence.csv', 'coordinate_manifest.json'):
        shutil.copy2(SOURCE_COHORT/name, COHORT/name)
    unique = pd.read_csv(COHORT/'sampled_sites_unique.csv')
    sites = pd.read_csv(COHORT/'sampled_sites_by_subject_task.csv')
    subjects = sorted(unique.subject.unique())
    assert len(unique) == 378 and len(subjects) == 67
    assert unique.confirmed_responsive.sum() == 239
    rows = []
    for task in (*TASKS, 'ALL_UNIQUE'):
        data = sites[sites.task.eq(task)] if task != 'ALL_UNIQUE' else unique
        responsive = data[data.confirmed_responsive]
        rows.append(dict(task=task, n_subjects_with_sampled_sites=int(data.subject.nunique()),
            n_sampled_bipolar_sites=int(data.site_id.nunique()),
            n_subjects_with_responsive_sites=int(responsive.subject.nunique()),
            n_confirmed_responsive_bipolar_sites=int(responsive.site_id.nunique()),
            sampled_subject_ids='|'.join(sorted(data.subject.unique())),
            responsive_subject_ids='|'.join(sorted(responsive.subject.unique()))))
    pd.DataFrame(rows).to_csv(COHORT/'task_summary.csv', index=False)
    rows = []
    for (subject, task), data in sites.groupby(['subject', 'task']):
        positive = data[data.confirmed_responsive]
        rows.append(dict(subject=subject, task=task, n_sampled_bipolar_sites=len(data),
            n_confirmed_responsive_bipolar_sites=len(positive), n_QC_excluded=int(data.native_QC_excluded.sum()),
            sampled_site_ids='|'.join(sorted(data.site_id)),
            confirmed_responsive_site_ids='|'.join(sorted(positive.site_id))))
    pd.DataFrame(rows).to_csv(COHORT/'subject_task_counts_and_lists.csv', index=False)
    summary = json.loads((SOURCE_COHORT/'summary.json').read_text())
    summary.update(version=VERSION, source_cohort=str(SOURCE_COHORT),
        figure1_response_selection='Native confirmed sound response, independent of NMF eligibility',
        wholebrain_subjects=subjects, wholebrain_subject_count=67,
        manual14_exclusions_scope='Figure2 NMF analysis; native Figure1 response flags preserved')
    write(COHORT/'summary.json', summary)
    pd.DataFrame({'subject': subjects}).to_csv(COHORT/'wholebrain_subjects.csv', index=False)
    write(ROOT/'config'/f'{VERSION}.json', dict(version=VERSION, source_cohort=str(SOURCE_COHORT),
        cohort_dir=str(COHORT), sampled_subjects=67, sampled_insula_sites=378,
        confirmed_responsive_insula_sites=239, subject_ids=subjects,
        response_selection=summary['figure1_response_selection'],
        manual14_exclusions_scope=summary['manual14_exclusions_scope'],
        author_request='Use the completed 67-subject version and update all Figure1 panels.'))


def display_inputs():
    """Reuse 224-site task means and load the remaining 15 from identical sources."""
    import h5py
    import mne
    import numpy as np
    import pandas as pd
    from scripts.complete_insula_cohort import name
    if (RUN/'display_inputs.json').exists():
        print('Reusing verified Figure1 cohort67 display inputs.', flush=True)
        return
    unique = pd.read_csv(COHORT/'sampled_sites_unique.csv')
    wanted_all = set(unique.loc[unique.confirmed_responsive, 'site_id'])
    membership = pd.read_csv(ROOT/'results/insula_completion_2026-11-07/nmf/waveform_source_membership.csv')
    assert set(membership.channel) == wanted_all
    existing_path = ROOT/'results/insula_manual14_2026-11-07/display_task_waveforms.csv.gz'
    existing = pd.read_csv(existing_path)
    manifest = pd.read_csv(SOURCE_COHORT/'native_input_manifest.csv', keep_default_na=False)
    frames = [existing]; sources = []
    cached_keys = set(zip(existing.channel, existing.task, existing.phase))
    for source in manifest.itertuples(index=False):
        active = membership[(membership.task == source.task) & (membership.subject == source.subject)
                            & (membership.phase == source.phase)]
        wanted = sorted(ch for ch in set(active.channel) & wanted_all
                        if (ch, source.task, source.phase) not in cached_keys)
        if not wanted:
            continue
        lo, hi = DISPLAY[source.phase]
        if source.source_type == 'native_timeperm_H5':
            ep = mne.read_epochs(source.epoch_path, preload=False, verbose='ERROR')
            names = [name(n, source.subject) for n in ep.ch_names]
            picks = [names.index(ch) for ch in wanted]
            assert not (set(wanted) & {name(n, source.subject) for n in ep.info['bads']})
            grid = ep.times.copy(); selected = (grid >= lo) & (grid <= hi)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                values = np.nanmean(ep.get_data(picks=picks, copy=True, verbose='ERROR'), axis=0)[:, selected]
            with h5py.File(source.statistics_path, 'r') as handle:
                assert [name(n, source.subject) for n in handle['ch_names'][:]] == names
                masks = np.asarray(handle['mask'][:], bool)[picks][:, selected]
            part = pd.DataFrame(dict(channel=np.repeat(wanted, selected.sum()),
                time=np.tile(grid[selected], len(wanted)), value=values.ravel(), mask=masks.ravel()))
            del ep, values
        else:
            blocks = []
            for chunk in pd.read_csv(source.statistics_path, chunksize=200000):
                keep = chunk.channel.isin(wanted) & chunk.time.between(lo, hi)
                if 'modality' in chunk:
                    keep &= chunk.modality.eq('sound')
                if 'description' in chunk:
                    keep &= chunk.description.eq('Repeat')
                if keep.any():
                    blocks.append(chunk.loc[keep, ['channel', 'time', 'value', 'mask']])
            assert blocks, (source.task, source.subject, source.phase)
            part = pd.concat(blocks, ignore_index=True)
        part['subject'] = source.subject; part['task'] = source.task; part['phase'] = source.phase
        part['time'] = part.time.round(9); part['mask'] = part['mask'].eq(True)
        part = part.groupby(['subject','channel','task','phase','time'], as_index=False).agg(
            value=('value','mean'), mask=('mask','any'))
        frames.append(part)
        sources.append(dict(task=source.task, subject=source.subject, phase=source.phase,
            source_type=source.source_type, epoch_path=source.epoch_path,
            statistics_path=source.statistics_path, included_channels='|'.join(wanted)))
        print('Additional Figure1 display source', source.task, source.subject, source.phase, len(wanted), flush=True)
    tasks = pd.concat(frames, ignore_index=True)
    assert not tasks.duplicated(['channel','task','phase','time']).any()
    assert set(zip(tasks.channel,tasks.task,tasks.phase)) == set(zip(membership.channel,membership.task,membership.phase))
    assert set(tasks.channel) == wanted_all
    tasks['modality'] = 'sound'
    tasks.to_csv(RUN/'display_task_waveforms.csv.gz', index=False)
    waves = tasks.groupby(['subject','channel','phase','time'], as_index=False).value.mean()
    old_waves = pd.read_csv(ROOT/'results/insula_completion_2026-11-07/nmf/site_waveforms.csv.gz')
    # The input-building cache intentionally stopped native Response at1 s.
    # Compare the complete original fit domain; the1–1.5-s display tail instead
    # comes from the same full native products (as in the current Figure2).
    from src.main_analysis import NMF_WINDOWS
    fit_rows = pd.concat([waves[waves.phase.eq(phase) & waves.time.between(*window)]
                          for phase, window in NMF_WINDOWS.items()], ignore_index=True)
    compared = fit_rows.merge(old_waves, on=['subject','channel','phase','time'],
        suffixes=('_display','_completed'), validate='one_to_one')
    assert len(compared) == len(fit_rows) == 239 * 399
    assert np.allclose(compared.value_display, compared.value_completed, atol=1e-12, rtol=0)
    assert waves.groupby('phase').channel.nunique().eq(239).all()
    write(RUN/'display_inputs.json', dict(n_sites=239, n_subjects=int(waves.subject.nunique()),
        reused_224_site_task_cache=str(existing_path), reused_cache_sha256=sha(existing_path),
        new_loaded_channels=sorted(wanted_all - set(existing.channel)),
        additional_sources=sources, saved_waveforms_match_completed_cohort_on_fit_domain=True,
        verified_fit_samples=len(compared), response_display_tail='Same full native sources through1.4921875 s; historical fit cache stops at1 s',
        native_QC_preserved=True, no_new_timeperm_tests=True, display_windows=DISPLAY))


def patch_notebook():
    import nbformat
    nb = nbformat.read(NB, as_version=4)
    byid = {cell.id: cell for cell in nb.cells}
    imports = byid['fig1-imports']
    imports.source = imports.source.replace(
        "REVIEW_DIR = ROOT / 'docs/evidence/sound_only_cohort_2026-10-17'",
        "from src.main_analysis import FIG1_COHORT_DIR\nREVIEW_DIR = FIG1_COHORT_DIR")
    cohort = byid['43c0bb99']
    cohort.source = cohort.source.replace(
        '# Read the shared Figure 1 / supplementary-information cohort lists.',
        '# Read the versioned Figure1 cohort and exported participant/task lists.')
    anchor = "coords = coords.loc[~coords.subject.isin(EXCLUDED_SUBJECTS)].copy()\n"
    assert anchor in cohort.source
    cohort.source = cohort.source.replace(anchor, anchor + '''# Author-selected whole-brain scope: all 67 subjects with accepted insula sampling.
source_coordinate_subjects = sorted(coords.subject.unique())
WHOLEBRAIN_SUBJECT_IDS = frozenset(sampled_list.subject)
assert len(WHOLEBRAIN_SUBJECT_IDS) == COHORT_SUMMARY['sampled_subjects'] == 67
excluded_context_subjects = sorted(set(source_coordinate_subjects) - WHOLEBRAIN_SUBJECT_IDS)
coords = coords.loc[coords.subject.isin(WHOLEBRAIN_SUBJECT_IDS)].copy()
assert set(coords.subject) == set(WHOLEBRAIN_SUBJECT_IDS)
pd.DataFrame({'subject': sorted(WHOLEBRAIN_SUBJECT_IDS)}).to_csv(FIG1_DATA/'wholebrain_subjects.csv', index=False)
pd.DataFrame({'subject': excluded_context_subjects}).to_csv(FIG1_DATA/'excluded_noninsula_subjects.csv', index=False)
''')
    cohort.source += '''\ncoords_unique.to_csv(FIG1_DATA/'wholebrain_plot_coordinates.csv', index=False)
print(f'Whole-brain sampling cohort: {len(WHOLEBRAIN_SUBJECT_IDS)} participants; '
      f'{len(excluded_context_subjects)} outside-cohort source subjects omitted.')
'''
    load = byid['hga-phase-load']
    first = load.source.index('time_paths = []')
    last = load.source.index('def _remap_roi_column')
    load.source = load.source[:first] + '''# Reuse the existing whole-brain saved-mask union; membership below applies
# the same 67-subject scope before every regional count and pie chart.
with (ROOT/'results/figure1_cohort67_2026-11-07/before/results/fig1/available_evidence_stream.pkl').open('rb') as handle:
    saved_stream = pickle.load(handle)
raw_sig_ids = saved_stream['raw_sig_ids']
# Native-good task means have already been matched to the completed response
# inventory, including the 15 responsive sites outside the selected NMF model.
trace_rows = pd.read_csv(ROOT/'results/figure1_cohort67_2026-11-07/display_task_waveforms.csv.gz')
assert set(trace_rows.channel) == set(responsive_site_ids)
assert trace_rows.modality.eq(MODALITY).all()

''' + load.source[last:]
    load.source += '''\nassert set(roi_sampled.subject).issubset(WHOLEBRAIN_SUBJECT_IDS)
assert set(sig_union_df.subject).issubset(WHOLEBRAIN_SUBJECT_IDS)
roi_sampled.to_csv(FIG1_DATA/'wholebrain_regional_sampled_sites.csv', index=False)
pd.DataFrame({
    'plotted_sites': coords_unique.groupby('subject').channel.nunique(),
    'regional_sampled_sites': roi_sampled.groupby('subject').channel.nunique(),
    'confirmed_responsive_sites': sig_union_df.groupby('subject').channel.nunique(),
}).reindex(sorted(WHOLEBRAIN_SUBJECT_IDS)).fillna(0).astype(int).rename_axis('subject').to_csv(
    FIG1_DATA/'wholebrain_subject_counts.csv')
'''
    # Keep the complete original group averaging and drawing code; replace only
    # the obsolete fallback that could reintroduce native bad-channel task rows.
    hga = byid['ca82e415'] if 'ca82e415' in byid else nb.cells[26]
    cut = hga.source.index('# Sound-only waveforms; equal available task weights within each site.')
    hga.source = '''# Existing native-good Repeat+sound task means for all 239 confirmed-response sites.
wave_rows = trace_rows.copy()
supplemented_epochs = []  # sources recorded in the versioned display_inputs.json
''' + hga.source[cut:]
    # The accepted single-electrode example remains the same subject and raw TFR.
    # Its old NMF name is historical and must not imply membership in the new fit.
    tfr = byid['fig1-tfr-plot']
    tfr.source = tfr.source.replace("ROOT/'docs/evidence/sound_only_cohort_2026-10-17/sampled_sites_unique.csv'",
                                    "FIG1_COHORT_DIR/'sampled_sites_unique.csv'")
    manifest = byid['fig1-manifest']
    manifest.source = manifest.source.replace("cohort='sound_only_2026-10-17'", "cohort='figure1_cohort67_2026-11-07'")
    manifest.source = manifest.source.replace("tfr_example_component='Sustained',", "tfr_example_component='Sustained (historical author-selected example)',")
    target = "(FIG1_DATA/'figure_manifest.json').write_text"
    assert target in manifest.source
    manifest.source = manifest.source.replace(target, '''manifest.update(
    wholebrain_sampled_participants=len(WHOLEBRAIN_SUBJECT_IDS),
    wholebrain_subject_ids=sorted(WHOLEBRAIN_SUBJECT_IDS),
    source_coordinate_participants=len(source_coordinate_subjects),
    excluded_noninsula_subjects=excluded_context_subjects,
    wholebrain_plotted_sites=len(coords_unique),
    wholebrain_plotted_participants=int(coords_unique.subject.nunique()),
    sampling_scope='All whole-brain electrodes of the67 reviewed insula-sampled participants',
    response_selection='Figure1 native confirmed sound responses; independent of NMF eligibility',
    manual14_exclusions_scope='Figure2 NMF analysis; Figure1 native response evidence retained',
    deferred_preprocessing_subjects=COHORT_SUMMARY['deferred_preprocessing_subjects'],
    all_tasks_native_QC_excluded_sites=COHORT_SUMMARY['all_tasks_native_QC_excluded_sites'],
    native_good_HGA_source_manifest=str(ROOT/'results/figure1_cohort67_2026-11-07/display_inputs.json'),
    previous_version=str(ROOT/'results/figure1_cohort67_2026-11-07/before'))
''' + target)
    # Replace stale prose and counts while retaining the original panel layout.
    nb.cells[4].source = '''## Sampling and confirmed-response counts by task

The complete67-participant × task inventory is exported with this Figure1 version. Supplementary notebooks retain their separately recorded versions until refreshed.
'''
    byid['fig1-count-table'].source = byid['fig1-count-table'].source.replace(
        '# Figure 1 count table; identical input rows to the supplementary subject/task list.',
        '# Figure1 count table from the versioned67-participant subject/task inventory.')
    nb.cells[2].source = '''# Figure 1: sound-only sampling and confirmed responses

Author-selected completed cohort (2026-11-07): **67 participants with reviewed insula sampling**. The whole-brain map, regional sampling counts and all response pies use these same participants. There are **378 anatomically sampled insula sites**, of which **239 have confirmed Repeat+sound responses** across the four tasks.

Response selection retains the original saved cluster-mask rule in open windows: Stimulus (0,0.6) s, Delay (0,1) s and Response (−0.5,0.5) s. Native task bad channels stay excluded. D0107's seven accepted anatomical sites remain sampling coverage with preprocessing deferred. Missing statistics and QC exclusions are not negative responses. Figure1 selection is independent of NMF eligibility; the14 manual Figure2 analysis exclusions do not erase anatomical sampling or positive native response evidence. D0121 remains excluded.
'''
    nb.cells[8].source = '''# Insular sampling and confirmed sound responses

All **378 reviewed sound-sampled insula sites / 67 participants** are shown. Purple marks **239 confirmed sound-responsive sites / 61 participants**. Gray includes sites without a confirmed response, missing evidence and original QC exclusions; it is not uniformly a nonresponsive category. Hammers contours and the original display geometry are retained.
'''
    nb.cells[12].source = '''### Coverage and confirmed sound responses by hemisphere

Use the same378-site anatomical sampling denominator and239-site native sound-response list as the maps and count table. Retain the original descriptive Fisher comparison and identify missing/QC-excluded evidence separately in the exported inventory.
'''
    nb.cells[14].source = '''## Confirmed HGA responses by phase

All pies use the same67 insula-sampled participants. Native Repeat+sound mask-positive insula counts are **116 Stimulus**, **133 Delay** and **185 Response**, using the original open selection windows. Other regions keep their historical regional QC; unresolved/rejected insula candidates are not relabeled Other. Counts describe available confirmed responses, not complete response probabilities.
'''
    nb.cells[21].source += '\n\nBoth pies use only the same67 participants with accepted insula sampling. The regional-QC sampling denominator differs from the unfiltered finite-coordinate context on the whole-brain map; each is exported with its site IDs.'
    nb.cells[25].source = '''## Mean sound HGA at confirmed-responsive insular sites

Use existing native-good Repeat+sound task means for all239 confirmed-responsive sites /61 participants. Average available tasks equally within each site, then plot electrode mean±SEM with the original axes and phase windows. No NMF labels select these sites; native response masks and bad-channel decisions are reused without new tests.
'''
    byid['fig1-tfr-md'].source += '\n\nThis accepted illustration is retained for the67-subject Figure1 cohort. Its Sustained designation refers to the historical author-reviewed example; D0028_LAI3-4 is excluded from the newly adopted Figure2 model.'
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            ast.parse(cell.source, filename=f'fig1-cell-{i}')
    nbformat.validate(nb)
    nbformat.write(nb, NB)


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN/'prepared.json').exists():
        print('Figure1 cohort67 preparation already exists; reusing snapshot and notebook.', flush=True)
        return
    snapshot()
    build_cohort()
    patch_notebook()
    write(RUN/'prepared.json', dict(version=VERSION, prepared_utc=datetime.now(timezone.utc).isoformat(),
        n_subjects=67, sampled_sites=378, responsive_sites=239))


def execute():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    import nbformat
    import numpy as np
    import pandas as pd
    from nbclient import NotebookClient
    import xml.etree.ElementTree as ET
    protected = json.loads((RUN/'protected_files.json').read_text())
    assert all(sha(p) == digest for p, digest in protected.items())
    nb = nbformat.read(NB, as_version=4)
    original = nbformat.read(RUN/'before/vizpub/fig1.ipynb', as_version=4)
    allowed = {'fig1-imports','43c0bb99','hga-phase-load',nb.cells[26].id,'fig1-tfr-plot','fig1-manifest'}
    for old, new in zip(original.cells, nb.cells):
        if old.cell_type == 'code' and old.id not in allowed:
            assert ast.dump(ast.parse(old.source)) == ast.dump(ast.parse(new.source)), old.id
    kernel_name = 'insula-figure1-cohort67'
    jupyter = RUN/'jupyter'; spec = jupyter/'kernels'/kernel_name; spec.mkdir(parents=True, exist_ok=True)
    write(spec/'kernel.json', dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Insula Figure1 cohort67 (ieeg)', language='python'))
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH','')
    os.environ['JUPYTER_PATH'] = str(jupyter) + os.pathsep + os.environ.get('JUPYTER_PATH','')
    def started(cell, cell_index, **kwargs):
        print(datetime.now(timezone.utc).isoformat(), 'cell', cell_index, cell.id,
              cell.source.splitlines()[0][:100] if cell.source else '', flush=True)
    client = NotebookClient(nb, kernel_name=kernel_name, timeout=1800, iopub_timeout=180,
        raise_on_iopub_timeout=True, resources={'metadata':{'path':str(ROOT/'vizpub')}},
        on_cell_start=started, record_timing=True)
    try:
        client.execute()
    except Exception:
        nbformat.write(nb, RUN/'fig1_failed.ipynb')
        raise
    errors = [o for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type') == 'error']
    assert not errors
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
    assert not [o for c in nb.cells for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
    manifest = json.loads((OUT/'figure_manifest.json').read_text())
    cohort = pd.read_csv(COHORT/'sampled_sites_unique.csv')
    subjects = set(cohort.subject)
    plotted = pd.read_csv(OUT/'wholebrain_plot_coordinates.csv')
    regional = pd.read_csv(OUT/'wholebrain_regional_sampled_sites.csv')
    responsive = pd.read_csv(OUT/'wholebrain_confirmed_responsive_sites.csv')
    assert plotted.channel.is_unique and regional.channel.is_unique and responsive.channel.is_unique
    assert set(plotted.subject) == subjects and len(subjects) == 67
    assert set(regional.subject) <= subjects and set(responsive.subject) <= subjects
    assert manifest['sampled_sites'] == 378 and manifest['confirmed_responsive_sites'] == 239
    assert manifest['sampled_participants'] == manifest['wholebrain_sampled_participants'] == 67
    assert set(plotted.loc[plotted.is_insula, 'channel']) == set(cohort.site_id)
    before_responses = pd.read_csv(RUN/'before/results/fig1/wholebrain_confirmed_responsive_sites.csv')
    anatomy = pd.read_csv(COHORT/'site_anatomy_audit.csv')
    expected_other = set(before_responses.loc[before_responses.subject.isin(subjects), 'channel']) - set(anatomy.channel)
    expected_responsive = expected_other | set(cohort.loc[cohort.confirmed_responsive,'site_id'])
    assert set(responsive.channel) == expected_responsive
    assert len(regional) == manifest['roi_sampling_counts']['total']
    assert len(responsive) == manifest['roi_confirmed_counts']['total']
    assert manifest['phase_response_counts'] == {'stimulus':116,'delay':133,'response':185}
    waves = pd.read_csv(OUT/'average_hga_site_waveforms.csv.gz')
    assert set(waves.channel) == set(cohort.loc[cohort.confirmed_responsive, 'site_id'])
    assert waves.groupby('phase').channel.nunique().eq(239).all()
    mean = waves.groupby(['phase','time']).value.agg(['mean','std','count']).reset_index()
    actual = pd.read_csv(OUT/'average_hga_group_trace.csv').sort_values(['phase','time']).reset_index(drop=True)
    assert np.allclose(actual['mean'], mean['mean'])
    assert np.allclose(actual['sem'], mean['std']/np.sqrt(mean['count']))
    assert not pd.read_csv(OUT/'average_hga_missing_waveforms.csv').shape[0]
    assert manifest['tfr_channel'] == 'D0028_LAI3-4' and not manifest['tfr_recomputed']
    # The accepted single-subject illustrations use identical data and display code.
    for name in ('fig1_example_electrode_tfr.svg','fig1_example_electrode_tfr_raw.svg',
                 'fig1_insula_electrode_schematic_seeg.svg'):
        ET.parse(FIG/name)
    assert manifest['smc_export_color_verified'] and manifest['roi_colors']['SMC'] == '#F39B35'
    assert all(sha(p) == digest for p, digest in protected.items())
    for path in FIG.glob('*.svg'):
        assert ET.parse(path).getroot().tag.endswith('svg')
    assert len(manifest['figures']) == 14
    nb.metadata['insula_figure1'] = dict(cohort_version=VERSION, sampled_sites=378,
        sampled_participants=67, confirmed_responsive_sites=239, condition='Repeat',modality='sound',
        figures_svg_only=True, all_panels_refreshed=True, executed_utc=datetime.now(timezone.utc).isoformat())
    temp = NB.with_suffix('.executed.tmp.ipynb'); nbformat.write(nb,temp); temp.replace(NB)
    receipt = dict(passed=True, version=VERSION, slurm_job_id=os.environ['SLURM_JOB_ID'],
        code_cells=sum(c.cell_type == 'code' for c in nb.cells), all_code_cells_executed=True,
        output_errors=0, svg_panels=len(manifest['figures']), sampled_participants=67,
        sampled_insula_sites=378, confirmed_responsive_insula_sites=239,
        wholebrain_plotted_sites=len(plotted), regional_sampling_counts=manifest['roi_sampling_counts'],
        regional_response_counts=manifest['roi_confirmed_counts'], phase_response_counts=manifest['phase_response_counts'],
        native_QC_and_saved_statistics_preserved=True, new_timeperm_tests=False,
        group_average_unit='electrode mean +/- SEM', figures2_to6_and_models_unchanged=True,
        protected_files=len(protected), original_drawing_AST_preserved=True,
        notebook_sha256=sha(NB), completed_utc=datetime.now(timezone.utc).isoformat())
    write(OUT/'execution_validation.json', receipt); write(RUN/'completed.json',receipt)
    for name in ('figure_manifest.json','execution_validation.json','task_summary.csv',
                 'subject_task_counts_and_lists.csv','wholebrain_subjects.csv','wholebrain_subject_counts.csv',
                 'phase_roi_counts.csv','roi_sampled_and_confirmed_counts.csv','insula_laterality_counts.csv',
                 'average_hga_counts.csv'):
        shutil.copy2(OUT/name,COHORT/name)
    registry = json.loads((ROOT/'config/current_insula_analysis.json').read_text())
    registry['current_figures'].pop('fig1_and_fig3_to_fig6',None)
    registry['current_figures']['fig1'] = dict(version=VERSION, notebook=str(NB),
        cohort_dir=str(COHORT), n_subjects=67, sampled_sites=378, confirmed_responsive_sites=239,
        wholebrain_scope='Only subjects with reviewed insula sampling; all their brain regions')
    registry['current_figures']['fig3_to_fig6'] = 'Preserved prior outputs/models; not refreshed for the224-site version'
    write(ROOT/'config/current_insula_analysis.json', registry)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    parser = argparse.ArgumentParser(); parser.add_argument('mode',choices=('prepare','display-inputs','execute'))
    mode = parser.parse_args().mode
    {'prepare':prepare,'display-inputs':display_inputs,'execute':execute}[mode]()
