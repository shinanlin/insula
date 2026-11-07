"""Display all frozen Lexical sites and the native Delay response union."""
import json

import numpy as np
import pandas as pd

from src.supplementary import lexical_condition_figures as plots
from src.supplementary.picture_condition_figures import SAMPLING_COLORS
from scripts.refresh_lexical_condition_sets import sha

DATA=plots.BASE/'sampling_response_2026-10-27'
AUDIT=plots.ROOT/'results/audit/lexical_sampling_response_2026-10-27'
IMAGE=plots.IMG/'fig3_lexical_delay_sampling_response_brain.svg'


def render():
    DATA.mkdir(parents=True,exist_ok=True)
    plots.plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none'})
    table=pd.read_csv(plots.OUT/'electrodes.csv')
    assert len(table)==187 and table.subject.nunique()==35 and table.electrode.is_unique
    responsive=table.decision_responsive|table.repeat_responsive
    np.testing.assert_array_equal(responsive,table.baseline_class.ne('Neither'))
    assert int(responsive.sum())==78
    assert table.baseline_class.value_counts().to_dict()=={'Neither':109,'Both':48,'Decision only':19,'Repeat only':11}
    points=table.copy()
    points['responsive']=responsive
    points=points.sort_values('responsive',kind='stable')
    groups=['Sampled, n.s.','Significant, Decision or Repeat']
    colors={groups[0]:SAMPLING_COLORS['Sampled, n.s.'],groups[1]:SAMPLING_COLORS['Significant, Sound or Image']}
    points['display_group']=np.where(points.responsive,groups[1],groups[0])
    namespace=plots.brain_renderer()
    fig,coordinates=namespace['brain_pair'](points,groups,colors,
        'Lexical Delay · Delay 0–1 s\n187 sampled sites / 35 participants')
    path=plots.save(fig,IMAGE.name)
    frame=pd.DataFrame(coordinates)
    assert len(frame)==187 and set(frame.electrode)==set(table.electrode)
    assert frame.display_group.value_counts().to_dict()=={groups[0]:109,groups[1]:78}
    frame.to_csv(DATA/'display_coordinates.csv',index=False)
    columns=['subject','channel','electrode','decision_responsive','repeat_responsive','baseline_class','x','y','z']
    sites=table[columns].copy()
    sites['delay_responsive']=responsive
    sites.to_csv(DATA/'electrodes.csv',index=False)
    hemispheres={}
    for name,side in [('Left',table.x.lt(0)),('Right',table.x.gt(0))]:
        hemispheres[name]=dict(sampled=int(side.sum()),responsive=int((side&responsive).sum()),
                              neither=int((side&~responsive).sum()))
    proof=dict(image=str(path),image_sha256=sha(path),task='LexicalDelay',phase='Delay',window=[0,1],
        n_sites=187,n_participants=35,n_responsive=78,n_responsive_participants=int(table.loc[responsive].subject.nunique()),
        n_neither=109,hemispheres=hemispheres,colors=colors,
        rule='Decision OR Repeat retained native baseline mask hit in Delay [0,1] s',
        input_table=str(plots.OUT/'electrodes.csv'),input_table_sha256=sha(plots.OUT/'electrodes.csv'),
        renderer=namespace['renderer_proof'],original_brain_pair_reused_verbatim=True,
        marker_size=24.,display_lift_mm=4.,original_camera_and_hemisphere_spacing=True,
        analytic_coordinates_changed=False,response_masks_changed=False,new_tests_run=False)
    (DATA/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
    return path
