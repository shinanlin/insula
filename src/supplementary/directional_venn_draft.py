"""Direct-difference Venn drafts from unchanged, completed temporal tests.

Run under Slurm. Existing main/supplementary panels are not replaced.
"""
from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib_venn import venn2, venn2_circles
import pandas as pd

ROOT = Path('/cwork/ns458/insula-paper')
OUT = ROOT/'results/supplementary/fig3_directional_venn_draft_2026-10-28'
IMG = ROOT/'img/supplementary/fig3_directional_venn_draft'
COMPONENTS = ('sensory', 'sustain', 'motor', 'Unassigned')
SPECS = {
    'picture': dict(source='fig3_picture_stimulus_condition_sets',
        conditions=('sound', 'image'), names=('Sound > Image', 'Image > Sound'),
        title='Picture Naming · Stimulus 0–1 s', expected=(19, 4, 1, 25), pool=49),
    'lexical': dict(source='fig3_lexical_delay_condition_sets/full_cohort_2026-10-27',
        conditions=('decision', 'repeat'), names=('Decision > Repeat', 'Repeat > Decision'),
        title='Lexical Delay · Delay 0–1 s', expected=(28, 1, 0, 49), pool=78),
}


def prepare(key):
    """Read all anatomical sites, then retain either-condition baseline responders."""
    spec = SPECS[key]
    source = ROOT/'results/supplementary'/spec['source']
    table = pd.read_csv(source/'electrodes.csv')
    a, b = spec['conditions']
    ac, bc = f'{a}_gt_{b}', f'{b}_gt_{a}'
    assert table.electrode.is_unique and table.eligible.all()
    assert table.paired_statistics_complete.all() and not table.native_qc_bad.any()
    stats = pd.read_csv(source/'timeperm.csv.gz')
    stats = stats.loc[stats.time.between(0, 1)]
    assert stats.electrode.nunique() == len(table)
    for col in (ac, bc):
        mask = stats[col+'_mask']
        assert not (mask & ~stats.temporal_mask_two_sided).any()
        assert stats.loc[mask, 'pointwise_p_two_sided'].lt(.05).all()
        assert stats.loc[mask, 'observed'].all()
        expected_sign = 1 if col == ac else -1
        assert (stats.loc[mask, 'mean_difference'] * expected_sign).gt(0).all()
        ids = set(stats.loc[mask, 'electrode'])
        assert ids == set(table.loc[table[col], 'electrode'])
    responsive = table[f'{a}_responsive'] | table[f'{b}_responsive']
    pool = table.loc[responsive].copy()
    assert len(pool) == spec['pool']
    groups = {
        'first_only': pool[ac] & ~pool[bc],
        'second_only': pool[bc] & ~pool[ac],
        'bidirectional': pool[ac] & pool[bc],
        'no_detected_difference': ~pool[ac] & ~pool[bc],
        'first_direction_total': pool[ac],
        'second_direction_total': pool[bc],
    }
    partition = tuple(int(groups[g].sum()) for g in list(groups)[:4])
    assert partition == spec['expected'] and sum(partition) == len(pool)
    folder = OUT/key
    folder.mkdir(parents=True, exist_ok=True)
    pool.to_csv(folder/'responsive_pool_electrodes.csv', index=False)
    table.loc[~responsive].to_csv(folder/'excluded_nonresponsive_electrodes.csv', index=False)
    rows = []
    for name, selected in groups.items():
        sub = pool.loc[selected]
        sub.to_csv(folder/f'{name}_electrodes.csv', index=False)
        counts = sub.functional_cluster.fillna('Unassigned').value_counts()
        rows.append(dict(group=name, n_electrodes=len(sub), n_participants=sub.subject.nunique(),
            **{c:int(counts.get(c,0)) for c in COMPONENTS}))
    counts = pd.DataFrame(rows)
    counts.to_csv(folder/'group_counts.csv', index=False)
    proof = dict(title=spec['title'], parent_sites=len(table), parent_participants=table.subject.nunique(),
        responsive_sites=len(pool), responsive_participants=pool.subject.nunique(),
        excluded_nonresponsive=int((~responsive).sum()), exclusive_partition=partition,
        groups=rows, no_new_tests=True,
        pool_rule='Either condition has native baseline responsiveness in the target 0–1 s phase; no NMF gate',
        difference_rule='Saved per-electrode two-sided temporal cluster mask, pointwise p<.05, and effect sign',
        venn_intersection='Opposite significant signs at different times in the same target phase',
        outside_circles='Neither significant direction detected; not an equivalence test',
        original_corrected_tests_reused_and_verified=True)
    (folder/'summary.json').write_text(json.dumps(proof, indent=2)+'\n')
    return pool, counts, proof


def render(key, pool, counts, proof):
    spec = SPECS[key]
    p = proof['exclusive_partition']
    totals = (p[0]+p[2], p[1]+p[2])
    plt.rcParams.update({'font.family':'Arial', 'font.size':7, 'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(10/2.54, 8.5/2.54))
    fig.subplots_adjust(left=.09, right=.91, bottom=.35, top=.80)
    labels = tuple(f'{name}\n(n={n})' for name,n in zip(spec['names'],totals))
    v = venn2(subsets=p[:3], set_labels=labels, ax=ax)
    colors = ('#2166AC','#D95F02','#762A83')
    for region, color in zip(('10','01','11'),colors):
        patch = v.get_patch_by_id(region)
        if patch is not None:
            patch.set_facecolor(color)
            patch.set_alpha(.35)
    for circle,color in zip(venn2_circles(subsets=p[:3],ax=ax,lw=.9),colors[:2]):
        circle.set_edgecolor(color)
    for label in v.subset_labels:
        if label is not None: label.set_fontsize(9)
    for label,color in zip(v.set_labels,colors[:2]):
        if label is not None:
            label.set_fontsize(7)
            label.set_color(color)
    fig.add_artist(Rectangle((.035,.17),.93,.70,transform=fig.transFigure,
        fill=False,edgecolor='#999999',linewidth=.6))
    fig.suptitle(spec['title'],fontsize=9,y=.98)
    fig.text(.5,.895,f'Responsive pool: {len(pool)} electrodes / {pool.subject.nunique()} participants',
        ha='center',fontsize=7)
    fig.text(.5,.205,f'No detected difference: {p[3]}',ha='center',fontsize=8)
    note = 'Overlap = opposite effects at different times' if p[2] else 'No bidirectional electrodes'
    fig.text(.5,.11,note,ha='center',fontsize=6.5)
    fig.text(.5,.055,'Saved two-sided time_perm_cluster · p_thresh = p_cluster = 0.05\n'
        '2,000 permutations · no new tests · draft',ha='center',fontsize=6)
    IMG.mkdir(parents=True,exist_ok=True)
    path = IMG/f'fig3_{key}_direct_difference_venn_draft.svg'
    fig.savefig(path,format='svg',bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    return path


def run():
    assert os.environ.get('SLURM_JOB_ID'), 'Run scientific summaries and plots via Slurm'
    from scripts.refresh_lexical_condition_sets import protected, sha, write
    from scripts.refresh_picture_condition_sets import protected as pn_protected
    protected(); pn_protected()
    files = []
    for folder in ('results/nmf','img/fig1','img/fig2',
        'results/supplementary/fig3_picture_stimulus_condition_sets',
        'results/supplementary/fig3_lexical_delay_condition_sets/full_cohort_2026-10-27',
        'img/supplementary/sound_image_test'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    files.extend((ROOT/'vizpub').glob('fig[1-6].ipynb'))
    files.extend(p for p in (ROOT/'vizpub/supplementary').glob('*.ipynb')
        if p.name != 'fig3_directional_difference_venn_draft.ipynb')
    hashes = {str(p):sha(p) for p in sorted(set(files))}
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'protected_existing_files.json',hashes)
    results = {}
    for key in SPECS:
        pool,counts,proof = prepare(key)
        path = render(key,pool,counts,proof)
        results[key] = dict(**proof, image=str(path))
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'completion.json',dict(passed=True,existing_files_unchanged=len(hashes),
        main_guard=protected(),picture_guard=pn_protected(),comparisons=results))
    return results
