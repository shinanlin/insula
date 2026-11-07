"""Exact spatial locations of the 25 responsive PN no-difference sites."""
from pathlib import Path
import inspect
import json
import os

ROOT=Path('/cwork/ns458/insula-paper')
POOL=ROOT/'results/supplementary/sound_image_test/responsive_pool_difference_masks_2026-10-28'
DATA=POOL/'no_stimulus_difference_brain_2026-10-28'
AUDIT=ROOT/'results/audit/pn_responsive_no_difference_brain_2026-10-28'
NOTEBOOK=ROOT/'vizpub/supplementary/sound_image_test.ipynb'
IMAGE=ROOT/'img/supplementary/sound_image_test/fig3_responsive_no_stimulus_difference_brain.svg'


def notebook_cells():
    import nbformat as nbf
    return [nbf.v4.new_markdown_cell('''## Locations of the 25 responsive sites without a detected Stimulus difference

These are exactly the **same 25 electrodes / 15 participants** in the two preceding no-difference heatmaps. Every site has a Sound OR Image baseline response in **Stimulus 0–1 s**, but neither Sound > Image nor Image > Sound passes the existing per-electrode condition-difference test in that window. This is not the 13-site Both baseline-response set, and does not establish equivalence.

Use the accepted Figure 1 local-insula surfaces, Hammers boundaries, cameras, surface display projection and 4-mm lift with original 24-pixel points. Reuse the Figure 2 spatial-density contour functions and their minimum of three sites per component/hemisphere. Colors match the heatmaps: Sensory 4 (blue), Sustained 8 (gold), Motor 9 (red), Unassigned 4 (gray). Gray sites are responsive electrodes lacking a frozen NMF assignment; no labels are refitted. Contours describe density, not significance boundaries. Native/template analytical coordinates, memberships, all HGA panels and statistical results are unchanged.
''',id='pn-responsive-no-difference-brain-note'),
        nbf.v4.new_code_cell('''from src.supplementary.pn_no_difference_brain import render as render_no_difference_brain, DATA as no_difference_brain_data
display(SVG(filename=str(render_no_difference_brain())))
display(pd.read_csv(no_difference_brain_data/'hemisphere_component_counts.csv'))
print('Exact 25-electrode list and display coordinates:', no_difference_brain_data)
''',id='pn-responsive-no-difference-brain-plot')]


def render():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from src.paths import save_svg
    from src.figure3_brain import brain_pair
    from src.supplementary.sound_image_cluster_difference import COLORS,LABELS,CLUSTERS,DISPLAY_OUT,TEST_OUT
    from src.supplementary.selected_brain_contours import _figure2_contour_helpers
    from scripts.refresh_lexical_condition_sets import sha,write
    DATA.mkdir(parents=True,exist_ok=True)
    pool=pd.read_csv(POOL/'responsive_pool_electrodes.csv')
    points=pool.loc[~pool.sound_gt_image & ~pool.image_gt_sound].copy()
    saved=pd.read_csv(POOL/'no_difference_electrodes.csv')
    assert len(pool)==49 and len(points)==25 and points.subject.nunique()==15
    assert points.electrode.is_unique and set(points.electrode)==set(saved.electrode)
    assert (points.sound_responsive | points.image_responsive).all()
    assert points.baseline_class.value_counts().to_dict()=={'Image only':9,'Both':8,'Sound only':8}
    expected={'sensory':4,'sustain':8,'motor':9,'Unassigned':4}
    assert points.functional_cluster.value_counts().to_dict()==expected
    assert np.isfinite(points[['x','y','z']]).all().all() and points.x.ne(0).all()
    for key in ['no_difference_sound_minus_image','no_difference_image_minus_sound']:
        shown=pd.read_csv(DISPLAY_OUT/f'{key}_selected_electrodes.csv')
        assert set(shown.electrode)==set(points.electrode)
    tests=pd.read_csv(TEST_OUT/'all_phase_timeperm.csv.gz')
    stimulus=tests.loc[tests.phase.eq('stimulus')&tests.electrode.isin(points.electrode)]
    assert stimulus.electrode.nunique()==25 and not stimulus.sound_gt_image_mask.any() and not stimulus.image_gt_sound_mask.any()
    _,contour_proof=_figure2_contour_helpers(ROOT)
    assert contour_proof['copied_verbatim'] and contour_proof['minimum_sites_per_component_hemisphere']==3
    for font in Path(sys.prefix,'fonts').glob('arial*.ttf'):font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none'})
    points['display_group']=points.functional_cluster
    fig,coordinates,contours=brain_pair(points,list(CLUSTERS),COLORS,LABELS,
        title='Responsive, no detected Sound–Image difference · n=25\nPicture Naming · Stimulus 0–1 s')
    path=save_svg(fig,IMAGE,close=True)
    projection=pd.DataFrame(coordinates)
    assert len(projection)==25 and set(projection.electrode)==set(points.electrode)
    reference=pd.read_csv(ROOT/'results/supplementary/fig3_picture_stimulus_condition_sets/baseline_classes_display_coordinates.csv')
    matched=projection.merge(reference,on='electrode',suffixes=('_new','_reference'),validate='one_to_one')
    assert len(matched)==25
    for col in ['pixel_x','pixel_y','surface_residual_mm','display_lift_mm']:
        np.testing.assert_allclose(matched[col+'_new'],matched[col+'_reference'],rtol=0,atol=1e-8)
    rings=pd.DataFrame(contours)
    assert rings.loc[rings.n_sites.lt(3),'n_contours'].eq(0).all()
    points['hemisphere']=np.where(points.x.lt(0),'lh','rh')
    counts=points.groupby(['hemisphere','functional_cluster']).agg(n_electrodes=('electrode','size'),n_participants=('subject','nunique')).reset_index()
    points.to_csv(DATA/'electrodes.csv',index=False)
    projection.to_csv(DATA/'display_coordinates.csv',index=False)
    counts.to_csv(DATA/'hemisphere_component_counts.csv',index=False)
    rings.to_csv(DATA/'contours.csv',index=False)
    write(DATA/'provenance.json',dict(passed=True,n_sites=25,n_participants=15,
        hemispheres={h:int(points.hemisphere.eq(h).sum()) for h in ['lh','rh']},component_counts=expected,
        selection='Within Stimulus Sound-or-Image baseline-response union, neither existing condition-difference direction in [0,1]',
        exact_same_sites_as_both_no_difference_heatmaps=True,coordinates_identical_to_existing_PN_map=True,
        analytical_coordinates_unchanged=True,display_lift_mm=4,point_size_pixels=24,
        renderer_source=inspect.getsourcefile(brain_pair),renderer_sha256=sha(inspect.getsourcefile(brain_pair)),
        contour_proof=contour_proof,palette=COLORS,new_statistics=False,new_NMF=False,
        source_hashes={str(POOL/'responsive_pool_electrodes.csv'):sha(POOL/'responsive_pool_electrodes.csv'),
            str(POOL/'no_difference_electrodes.csv'):sha(POOL/'no_difference_electrodes.csv'),
            str(ROOT/'results/nmf/channel_assignments.csv'):sha(ROOT/'results/nmf/channel_assignments.csv')},
        svg=str(path),svg_sha256=sha(path)))
    print('RENDERED exact25 no-difference sites:',points.hemisphere.value_counts().to_dict(),flush=True)
    return path
