"""Existing PN directional selections with the accepted local-insula map style."""
from pathlib import Path
import inspect
import json
import os

ROOT=Path('/cwork/ns458/insula-paper')
POOL=ROOT/'results/supplementary/sound_image_test/responsive_pool_difference_masks_2026-10-28'
DATA=POOL/'directional_brain_maps_2026-10-28'
AUDIT=ROOT/'results/audit/pn_responsive_directional_brains_2026-10-28'
NOTEBOOK=ROOT/'vizpub/supplementary/sound_image_test.ipynb'
IMG=ROOT/'img/supplementary/sound_image_test'
SPECS={
    'sound_gt_image':dict(title='Sound > Image',n=20,subjects=13,
        components={'sensory':7,'sustain':5,'motor':7,'Unassigned':1},
        svg='fig3_responsive_sound_gt_image_brain.svg'),
    'image_gt_sound':dict(title='Image > Sound',n=5,subjects=4,
        components={'sensory':1,'sustain':2,'motor':2,'Unassigned':0},
        svg='fig3_responsive_image_gt_sound_brain.svg'),
}


def notebook_cells():
    import nbformat as nbf
    cells=[nbf.v4.new_markdown_cell('''## Spatial comparison of the three Stimulus difference groups

Use the **same 49-site Stimulus response pool** as the HGA panels: Sound > Image **20 sites / 13 participants**, Image > Sound **5 / 4**, and no detected Stimulus difference **25 / 15**. Selection is by each electrode's existing **Stimulus 0–1 s** trial-level condition-difference test. One site has significant effects in opposite directions at different times and appears in both directional maps; it is not relabeled.

All three independent maps use the **same existing Figure 1 local-insula renderer, camera, 24-pixel points and 4-mm display lift**, and the original Figure 2 spatial-density contours. Colors match the HGA panels: Sensory blue, Sustained gold, Motor red, Unassigned gray. The existing rule requires **at least 3 sites in a component within one hemisphere** for a contour; sparse groups retain their points. Contours describe density and are not significance boundaries. Coordinates, assignments, memberships and statistics are unchanged. The original 25-site map is retained below without rerendering.
''',id='pn-responsive-directional-brains-note')]
    for key,spec in SPECS.items():
        cells.append(nbf.v4.new_code_cell(
            'from src.supplementary.pn_directional_brains import render as render_directional_brain\n'
            f"display(SVG(filename=str(render_directional_brain('{key}'))))\n",
            id='pn-responsive-directional-brain-'+key.replace('_','-')))
    return cells


def render(key):
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
    spec=SPECS[key]
    DATA.mkdir(parents=True,exist_ok=True)
    pool=pd.read_csv(POOL/'responsive_pool_electrodes.csv')
    points=pool.loc[pool[key]].copy()
    saved=pd.read_csv(POOL/f'{key}_electrodes.csv')
    shown=pd.read_csv(DISPLAY_OUT/f'{key}_selected_electrodes.csv')
    assert len(pool)==49 and len(points)==spec['n'] and points.subject.nunique()==spec['subjects']
    assert points.electrode.is_unique and set(points.electrode)==set(saved.electrode)==set(shown.electrode)
    assert (points.sound_responsive|points.image_responsive).all()
    actual={c:int(points.functional_cluster.eq(c).sum()) for c in CLUSTERS}
    assert actual==spec['components']
    assert np.isfinite(points[['x','y','z']]).all().all() and points.x.ne(0).all()
    tests=pd.read_csv(TEST_OUT/'all_phase_timeperm.csv.gz')
    stimulus=tests.loc[tests.phase.eq('stimulus')&tests.electrode.isin(points.electrode)]
    assert set(stimulus.loc[stimulus[key+'_mask'],'electrode'])==set(points.electrode)
    _,contour_proof=_figure2_contour_helpers(ROOT)
    assert contour_proof['copied_verbatim'] and contour_proof['minimum_sites_per_component_hemisphere']==3
    for font in Path(sys.prefix,'fonts').glob('arial*.ttf'):font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none'})
    points['display_group']=points.functional_cluster
    fig,coordinates,contours=brain_pair(points,list(CLUSTERS),COLORS,LABELS,
        title=f"{spec['title']} · n={spec['n']}\nPicture Naming · Stimulus 0–1 s")
    path=save_svg(fig,IMG/spec['svg'],close=True)
    projection=pd.DataFrame(coordinates)
    assert len(projection)==spec['n'] and set(projection.electrode)==set(points.electrode)
    reference=pd.read_csv(ROOT/'results/supplementary/fig3_picture_stimulus_condition_sets/baseline_classes_display_coordinates.csv')
    matched=projection.merge(reference,on='electrode',suffixes=('_new','_reference'),validate='one_to_one')
    assert len(matched)==spec['n']
    for col in ['pixel_x','pixel_y','surface_residual_mm','display_lift_mm']:
        np.testing.assert_allclose(matched[col+'_new'],matched[col+'_reference'],rtol=0,atol=1e-8)
    rings=pd.DataFrame(contours)
    assert rings.loc[rings.n_sites.lt(3),'n_contours'].eq(0).all()
    points['hemisphere']=np.where(points.x.lt(0),'lh','rh')
    points.to_csv(DATA/f'{key}_electrodes.csv',index=False)
    projection.to_csv(DATA/f'{key}_display_coordinates.csv',index=False)
    points.groupby(['hemisphere','functional_cluster']).agg(n_electrodes=('electrode','size'),n_participants=('subject','nunique')).reset_index().to_csv(
        DATA/f'{key}_hemisphere_component_counts.csv',index=False)
    rings.to_csv(DATA/f'{key}_contours.csv',index=False)
    write(DATA/f'{key}_provenance.json',dict(passed=True,n_sites=spec['n'],n_participants=spec['subjects'],
        hemispheres={h:int(points.hemisphere.eq(h).sum()) for h in ['lh','rh']},component_counts=actual,
        exact_same_sites_as_heatmap=True,coordinates_identical_to_existing_PN_map=True,analytical_coordinates_unchanged=True,
        display_lift_mm=4,point_size_pixels=24,contour_method=contour_proof,palette=COLORS,
        brain_renderer_source=inspect.getsourcefile(brain_pair),brain_renderer_sha256=sha(inspect.getsourcefile(brain_pair)),
        new_statistics=False,new_NMF=False,source_hashes={str(POOL/f'{key}_electrodes.csv'):sha(POOL/f'{key}_electrodes.csv'),
            str(ROOT/'results/nmf/channel_assignments.csv'):sha(ROOT/'results/nmf/channel_assignments.csv')},
        svg=str(path),svg_sha256=sha(path)))
    print('RENDERED',spec['title'],points.hemisphere.value_counts().to_dict(),flush=True)
    return path
