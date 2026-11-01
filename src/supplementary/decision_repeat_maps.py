"""Saved-mask Delay overlap and the approved Decision > Repeat spatial map.

No inferential test, trial processing, electrode selection or NMF is run here.
"""
from __future__ import annotations

import hashlib
import json
import ast
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.nmf.waveform_analysis import FUNCTION_COLORS

CLUSTERS = ('sensory', 'sustain', 'motor')
LABELS = dict(sensory='Sensory', sustain='Sustained', motor='Motor')
WINDOW = (0., 1.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_delay_overlap(root, destination):
    """Reuse the same 123-site cohort, native baseline masks and 29-site list."""
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    parent = root / 'results/supplementary/decision_repeat_test'
    source = root / 'results/supplementary/context_comparisons/waveforms.pkl'
    assignments_path = root / 'results/nmf/channel_assignments.csv'
    frozen_path = parent / 'frozen_heatmap_row_membership.csv'
    selection_path = parent / 'delay_electrode_selection/selected_electrodes.csv'
    selection_manifest = json.loads((selection_path.parent / 'selection_manifest.json').read_text())
    selection_input_audit = json.loads((selection_path.parent / 'input_audit.json').read_text())
    assert selection_manifest['completed'] and selection_manifest['n_tested'] == 123
    assert selection_manifest['window'] == [0, 1]
    assert sha(selection_path) == selection_manifest['outputs']['selected_electrodes.csv']
    assert sha(source) == selection_input_audit['source_cache_sha256']
    source_manifest = json.loads((source.parent / 'native_input_manifest.json').read_text())
    assert sha(assignments_path) == source_manifest['assignments_sha256']
    frozen = pd.read_csv(frozen_path)
    frozen = frozen.loc[frozen.phase.eq('delay'), ['electrode','subject','channel','functional_cluster']]
    assert len(frozen) == 123 and frozen.subject.nunique() == 34
    assert not frozen.electrode.duplicated().any()
    sites = set(frozen.electrode)
    frame = pd.read_pickle(source)
    frame = frame.loc[frame.contrast.eq('decision_repeat') & frame.phase.eq('delay')
        & frame.time.between(*WINDOW)].copy()
    assert set(frame.electrode) == sites
    assert frame.task.eq('LexicalDelay').all() and frame.modality.eq('sound').all()
    assert frame.mask_observed.all() and set(frame.level) == {'Decision','Repeat'}
    assert not frame.duplicated(['electrode','level','time']).any()
    assert frame.groupby(['electrode','level']).size().eq(129).all()
    assert frame.time.min() == 0 and frame.time.max() == 1
    masks = frame.groupby(['electrode','level'])['mask'].any().unstack('level')
    assert masks.index.is_unique and len(masks) == 123 and not masks.isna().any().any()
    membership = frozen.merge(masks.reset_index(), on='electrode', validate='one_to_one')
    membership = membership.rename(columns={'Decision':'decision_baseline_responsive',
                                            'Repeat':'repeat_baseline_responsive'})
    decision = set(membership.loc[membership.decision_baseline_responsive,'electrode'])
    repeat = set(membership.loc[membership.repeat_baseline_responsive,'electrode'])
    region_sets = {'Decision only':decision-repeat, 'Both':decision&repeat,
                   'Repeat only':repeat-decision, 'Neither':sites-(decision|repeat)}
    assert sum(map(len,region_sets.values())) == len(sites)
    region_for_site = {site:region for region,group in region_sets.items() for site in group}
    membership['venn_region'] = membership.electrode.map(region_for_site)
    selected = pd.read_csv(selection_path)
    assert len(selected) == 29 and selected.subject.nunique() == 18
    assert set(selected.electrode) <= sites
    membership['decision_gt_repeat_selected'] = membership.electrode.isin(selected.electrode)
    membership['phase'], membership['window_start'], membership['window_end'] = 'delay', 0., 1.
    membership.to_csv(destination / 'baseline_overlap_membership.csv',index=False)
    counts = pd.DataFrame([dict(region=region,n_sites=len(group),
        n_subjects=membership.loc[membership.electrode.isin(group),'subject'].nunique())
        for region,group in region_sets.items()])
    counts.to_csv(destination / 'baseline_overlap_region_counts.csv',index=False)
    component_counts = membership.groupby(['functional_cluster','venn_region']).size().rename('n_sites').reset_index()
    component_counts.to_csv(destination / 'baseline_overlap_component_counts.csv',index=False)

    assignments = pd.read_csv(assignments_path)
    assert not assignments.duplicated(['subject','channel']).any()
    points = frozen[['subject','channel','electrode']].merge(assignments,
        on=['subject','channel'],validate='one_to_one')
    assert len(points) == len(sites)
    assert np.isfinite(points[['x','y','z']].to_numpy()).all()
    assert (points.x != 0).all()
    assert set(points.functional_cluster) == set(CLUSTERS)
    label_check = membership.set_index('electrode').functional_cluster
    assert points.functional_cluster.eq(points.electrode.map(label_check)).all()
    # Retain the assignment's original flag explicitly; render only the
    # separately approved condition-difference selection as colored points.
    points = points.rename(columns={'significant':'original_assignment_significant'})
    points['significant'] = points.electrode.isin(selected.electrode)
    points['pattern'] = np.where(points.significant,1.,np.nan)
    points['map_phase'], points['map_window_start'], points['map_window_end'] = 'delay', 0., 1.
    points['selection'] = 'Saved per-electrode significant Decision > Repeat in Delay 0–1 s'
    assert points.significant.sum() == 29
    assert points.loc[points.significant].functional_cluster.value_counts().to_dict() == {
        'sustain':14,'motor':12,'sensory':3}
    points.to_csv(destination / 'decision_gt_repeat_brain_membership.csv',index=False)
    hemispheres = pd.DataFrame([dict(hemisphere=hemi,n_tested=int(side.sum()),
        n_selected=int(points.loc[side,'significant'].sum()))
        for hemi,side in (('Left',points.x.lt(0)),('Right',points.x.gt(0)))])
    hemispheres.to_csv(destination / 'brain_hemisphere_counts.csv',index=False)
    report = dict(window=[0.,1.],task='LexicalDelay',modality='sound',
        n_sites=123,n_subjects=34,decision_baseline_responsive=len(decision),
        repeat_baseline_responsive=len(repeat),region_counts={r:len(s) for r,s in region_sets.items()},
        selected_decision_gt_repeat=29,n_selected_subjects=18,
        brain_component_counts={c:int((points.significant & points.functional_cluster.eq(c)).sum()) for c in CLUSTERS},
        hemisphere_counts=hemispheres.to_dict(orient='records'),
        baseline_definition='Any existing saved native baseline time-permutation mask=True within Delay [0,1] s',
        venn_definition='Decision and Repeat baseline-responsive sets, each within the same 123 sites',
        venn_only_means='Significance detected in one baseline test only; not a test of a condition difference',
        brain_definition='Exactly the 29 previously selected per-electrode Delay Decision > Repeat sites',
        brain_background='Other 94 sites in the same tested cohort',
        new_statistics=False,selection_recomputed=False,coordinates_recomputed=False,
        source_sha256=sha(source),assignments_sha256=sha(assignments_path),
        frozen_membership_sha256=sha(frozen_path),selected_membership_sha256=sha(selection_path))
    (destination / 'input_and_membership_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return membership,counts,points,report


def draw_delay_venn(membership, report):
    from matplotlib_venn import venn2,venn2_circles
    decision = set(membership.loc[membership.decision_baseline_responsive,'electrode'])
    repeat = set(membership.loc[membership.repeat_baseline_responsive,'electrode'])
    counts = (len(decision-repeat),len(repeat-decision),len(decision&repeat))
    cm = 1/2.54
    fig,ax = plt.subplots(figsize=(7.2*cm,5.4*cm))
    fig.subplots_adjust(left=.08,right=.92,bottom=.23,top=.78)
    diagram = venn2(subsets=counts,
        set_labels=(f'Decision\n(n={len(decision)})',f'Repeat\n(n={len(repeat)})'),
        set_colors=('#6A5ACD','#000000'),alpha=.30,ax=ax)
    outlines = venn2_circles(subsets=counts,ax=ax,linewidth=.8)
    for artist,color in zip(outlines,('#6A5ACD','#000000')):
        artist.set_edgecolor(color)
    for label in diagram.subset_labels:
        if label is not None:label.set_fontsize(8)
    for label,color in zip(diagram.set_labels,('#6A5ACD','#000000')):
        if label is not None:label.set_fontsize(7);label.set_color(color)
    fig.suptitle('Delay 0–1 s\nBaseline-responsive sites',fontsize=7.5,y=.97)
    fig.text(.5,.025,f"Neither: n={report['region_counts']['Neither']} · Total: n=123",
        ha='center',va='bottom',fontsize=6.5)
    return fig


def _figure1_local_insula_renderer(root):
    """Load exact Figure 1 function bodies without executing its analysis cells."""
    import mne
    import pyvista as pv
    from mne.viz import Brain
    from scipy.spatial import cKDTree

    notebook_path = Path(root) / 'vizpub/fig1.ipynb'
    notebook = json.loads(notebook_path.read_text())
    cells = {cell.get('id'):''.join(cell['source']) for cell in notebook['cells']}
    source = cells['7731cdbd']
    tree = ast.parse(source)
    names = {'_insula_vertices','_faces_to_polydata','_insula_outward_normals',
             '_prepare_insula_surface','_hide_default_pial','_make_insula_brain',
             '_camera_xyz','_project_to_insula','_add_electrodes_fixed',
             '_points_to_pixels','_render_insula_hemi'}
    definitions = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names
    constants = {}
    for text in (source,cells['fig1-style'],cells['00031585']):
        for node in ast.parse(text).body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                name = node.targets[0].id
                if name in {'INSULA_MESH_COLOR','INSULA_MESH_ALPHA','ELECTRODE_OFFSET_MM',
                            'INSULA_PATTERNS','POINT_SIZE_SIG','POINT_SIZE_NSIG','NSIG_COLOR',
                            'recon_dir','FS_SUBJECT'}:
                    constants[name] = ast.literal_eval(node.value)
    assert constants['ELECTRODE_OFFSET_MM'] == 4.0
    assert constants['POINT_SIZE_SIG'] == 12 and constants['POINT_SIZE_NSIG'] == 7
    pv.OFF_SCREEN = True
    mne.viz.set_3d_backend('notebook')
    namespace = dict(np=np,pv=pv,Brain=Brain,cKDTree=cKDTree,**constants)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),
                 f'{notebook_path}:7731cdbd','exec'),namespace)
    namespace['labels'] = mne.read_labels_from_annot(subject=constants['FS_SUBJECT'],
        parc='aparc.a2009s',surf_name='pial',hemi='both',subjects_dir=constants['recon_dir'])
    for hemi in ('lh','rh'):
        pial,faces = mne.read_surface(f"{constants['recon_dir']}/{constants['FS_SUBJECT']}/surf/{hemi}.pial")
        namespace[f'{hemi}_pial_coords'] = pial
        surface = namespace['_prepare_insula_surface'](hemi,pial,faces)
        namespace[f'{hemi}_surf'] = surface
        namespace[f'{hemi}_insula'] = surface['mesh']
        namespace[f'{hemi}_insula_center'] = surface['center']
    proof = dict(source_notebook=str(notebook_path),source_cell='7731cdbd',
        source_cell_sha256=hashlib.sha256(source.encode()).hexdigest(),
        exact_source_function_sha256={node.name:hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()
                                    for node in definitions},
        constants=constants,functions_copied_verbatim=True,
        source_analysis_cells_executed=False,full_hemisphere_actors_hidden=True,
        surface='Figure 1 aparc.a2009s insula-only pial mesh',
        projection='Figure 1 camera-facing insula vertices plus original 4-mm camera-facing display lift',
        analytic_coordinates_changed=False)
    return namespace,proof


def draw_delay_difference_brain(points, crop, *, display_output=None):
    """Selected points with original Figure 2 contours; omit gray background sites."""
    from src.supplementary.selected_brain_contours import draw_selected_brain_contours
    return draw_selected_brain_contours(points, 'Decision > Repeat · Delay 0–1 s', display_output=display_output)
