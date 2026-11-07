"""Additional functional-color display of the saved Decision > Repeat sites."""
import json

import numpy as np
import pandas as pd

from src.supplementary import lexical_condition_figures as plots
from scripts.refresh_lexical_condition_sets import sha

VARIANT = plots.BASE/'component_color_variant'
AUDIT = plots.ROOT/'results/audit/lexical_component_color_variant_2026-10-27'


def render():
    VARIANT.mkdir(parents=True,exist_ok=True)
    plots.plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none'})
    table=pd.read_csv(plots.OUT/'electrodes.csv')
    points=table.loc[table.decision_gt_repeat].copy()
    assert len(points)==31 and points.subject.nunique()==18
    expected={'sensory':3,'sustain':14,'motor':12,'Unassigned':2}
    assert points.functional_cluster.value_counts().to_dict()==expected
    points['display_group']=points.functional_cluster.map(plots.LABELS)
    assert points.display_group.notna().all()
    colors={plots.LABELS[c]:color for c,color in plots.COMPONENT_COLORS.items()}
    names=list(colors)
    namespace=plots.brain_renderer()
    fig,coordinates=namespace['brain_pair'](points,names,colors,'Decision > Repeat · Delay 0–1 s')
    path=plots.save(fig,'fig3_lexical_delay_decision_gt_repeat_brain_components.svg')
    coordinates=pd.DataFrame(coordinates)
    reference=pd.read_csv(plots.OUT/'decision_gt_repeat_brain_display_coordinates.csv')
    joined=coordinates.merge(reference,on='electrode',validate='one_to_one',suffixes=('_variant','_original'))
    assert len(joined)==31 and set(coordinates.electrode)==set(points.electrode)
    for column in ['pixel_x','pixel_y','surface_residual_mm','display_lift_mm']:
        np.testing.assert_allclose(joined[column+'_variant'],joined[column+'_original'],atol=1e-8,rtol=0)
    coordinates.to_csv(VARIANT/'display_coordinates.csv',index=False)
    points.to_csv(VARIANT/'electrodes.csv',index=False)
    proof=dict(image=str(path),image_sha256=sha(path),n_sites=31,n_participants=18,
        component_counts=expected,colors=colors,original_renderer_reused=True,
        display_coordinates_equal_original=True,marker_sizes_and_4mm_lift_unchanged=True,
        unassigned_gray_means='Significant sites without a frozen NMF assignment, not nonsignificant sites',
        input_table_sha256=sha(plots.OUT/'electrodes.csv'),
        nmf_assignment_sha256=sha(plots.ROOT/'results/nmf/channel_assignments.csv'),
        selection_or_inference_changed=False,original_figures_replaced=False)
    (VARIANT/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
    return path
