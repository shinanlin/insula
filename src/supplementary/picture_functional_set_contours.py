"""Frozen functional colors and original spatial-density contours for PN sets."""
import inspect
import json

import numpy as np
import pandas as pd

from scripts.refresh_lexical_condition_sets import sha
from src.supplementary import lexical_condition_figures as reference
from src.supplementary import picture_condition_figures as baseline
from src.supplementary.condition_response_maps import brain_pair
from src.supplementary.selected_brain_contours import _figure2_contour_helpers

ROOT=baseline.ROOT
DATA=baseline.OUT/'functional_set_contours_2026-10-27'
AUDIT=ROOT/'results/audit/picture_functional_set_contours_2026-10-27'
SPECS=[('picture_only','Image only','Picture-only',11,dict(sensory=1,sustain=2,motor=6,Unassigned=2)),
       ('sound_only','Sound only','Sound-only',25,dict(sensory=8,sustain=6,motor=9,Unassigned=2)),
       ('both','Both','Both',13,dict(sensory=2,sustain=7,motor=3,Unassigned=1))]
EXPORT_NAMES={key:f'fig3_picture_stimulus_{key}_functional_contours.svg' for key,*_ in SPECS}
SHEET='fig3_picture_stimulus_response_sets_functional_contours.svg'


def notebook_cells():
    import nbformat as nbf
    note=nbf.v4.new_markdown_cell('''## 3. Picture-only, Sound-only and Both by functional component

The same Stimulus **0–1 s** baseline-response sets are displayed separately: **Picture-only 11**, **Sound-only 25**, and **Both 13**. Picture-only is the Image-only set in the preceding Venn; all memberships are unchanged.

Colors reuse the frozen NMF assignments: **Sensory blue, Sustained gold, Motor red**. Picture-only contains 1 / 2 / 6 assigned sites, Sound-only 8 / 6 / 9, and Both 2 / 7 / 3, respectively. Their remaining **2 / 2 / 1 gray sites are responsive but Unassigned**, rather than nonsignificant context points. No labels are refitted or imputed.

Points retain the Figure 1 surface, cameras, 24-pixel size and 4-mm display lift. Contours reuse the exact Figure 2 spatial-density functions and parameters. Each functional group requires at least 3 sites within one hemisphere for a contour; sparse groups retain their points. Contours describe spatial density and are not significance boundaries. The two preceding figures and all saved statistical results remain unchanged.
''',id='picture-functional-set-contours-note')
    code=nbf.v4.new_code_cell('''from pathlib import Path
import sys
from IPython.display import SVG,display
_pn_paper_root=Path('/cwork/ns458/insula-paper')
if str(_pn_paper_root) not in sys.path:sys.path.insert(0,str(_pn_paper_root))
from src.supplementary.picture_functional_set_contours import render as render_picture_functional_sets,DATA as functional_set_data
functional_set_paths,functional_set_sheet=render_picture_functional_sets()
display(SVG(filename=str(functional_set_sheet)))
import pandas as pd
display(pd.read_csv(functional_set_data/'counts.csv'))
print('Functional-color contour SVGs:',functional_set_paths)
''',id='picture-functional-set-contours-maps')
    return [note,code]


def render():
    DATA.mkdir(parents=True,exist_ok=True)
    table,_counts=baseline.load()
    _,contour_proof=_figure2_contour_helpers(ROOT)
    assert contour_proof['copied_verbatim'] and contour_proof['minimum_sites_per_component_hemisphere']==3
    reference_positions=pd.read_csv(baseline.OUT/'baseline_classes_display_coordinates.csv')
    paths={};reports=[];count_rows=[];all_ids=[];all_contours=[]
    for key,source_class,title,total,expected in SPECS:
        selected=table.loc[table.baseline_class.eq(source_class)].copy()
        assert len(selected)==total and selected.electrode.is_unique
        assert selected.functional_cluster.value_counts().to_dict()==expected
        selected['display_group']=selected.functional_cluster
        colors=reference.COMPONENT_COLORS
        fig,coordinates,contours=brain_pair(selected,list(colors),colors,reference.LABELS,
            title=f'{title} · Picture Naming · Stimulus 0–1 s')
        paths[key]=baseline.save(fig,EXPORT_NAMES[key])
        projection=pd.DataFrame(coordinates)
        assert len(projection)==total and set(projection.electrode)==set(selected.electrode)
        joined=projection.merge(reference_positions,on='electrode',validate='one_to_one',suffixes=('_new','_baseline'))
        assert len(joined)==total
        for column in ['pixel_x','pixel_y','surface_residual_mm','display_lift_mm']:
            np.testing.assert_allclose(joined[column+'_new'],joined[column+'_baseline'],atol=1e-8,rtol=0)
        projection.to_csv(DATA/f'{key}_display_coordinates.csv',index=False)
        selected.to_csv(DATA/f'{key}_electrodes.csv',index=False)
        rings=pd.DataFrame(contours)
        assert rings.loc[rings.n_sites.lt(3),'n_contours'].eq(0).all()
        rings.insert(0,'response_set',title)
        all_contours.append(rings)
        row=dict(response_set=title,n_electrodes=total,n_participants=int(selected.subject.nunique()))
        row.update({reference.LABELS[c]:expected[c] for c in reference.COMPONENT_COLORS})
        count_rows.append(row)
        reports.append(dict(response_set=title,n_sites=total,n_participants=row['n_participants'],
            components=expected,source_baseline_class=source_class,
            hemispheres={h:dict(n_sites=int(projection.hemisphere.eq(h).sum()),
                n_contours=int(rings.loc[rings.hemisphere.eq(h),'n_contours'].sum())) for h in ['lh','rh']},
            original_pixel_coordinates_and_lift_unchanged=True,image=str(paths[key]),image_sha256=sha(paths[key])))
        all_ids.extend(selected.electrode.tolist())
    union=set(table.loc[table.baseline_class.ne('Neither'),'electrode'])
    assert len(all_ids)==49 and len(set(all_ids))==49 and set(all_ids)==union
    pd.DataFrame(count_rows).to_csv(DATA/'counts.csv',index=False)
    pd.concat(all_contours,ignore_index=True).to_csv(DATA/'contour_counts.csv',index=False)
    sheet=baseline.compose([(paths[key],25.,i*215.,370.,205.) for i,(key,*_) in enumerate(SPECS)],
        SHEET,420.,645.)
    proof=dict(task='PictureNaming',phase='Stimulus',window=[0,1],
        n_eligible_sites=128,n_eligible_participants=27,n_displayed_unique_sites=49,
        all_three_sets_disjoint_and_equal_existing_responsive_union=True,
        source_table=str(baseline.OUT/'electrodes.csv'),source_table_sha256=sha(baseline.OUT/'electrodes.csv'),
        nmf_assignment_sha256=sha(ROOT/'results/nmf/channel_assignments.csv'),
        functional_colors=reference.COMPONENT_COLORS,contour_method=contour_proof,
        brain_function_source=inspect.getsourcefile(brain_pair),brain_function_source_sha256=sha(inspect.getsourcefile(brain_pair)),
        panels=reports,sheet=str(sheet),sheet_sha256=sha(sheet),
        response_sets_changed=False,new_tests_run=False,nmf_changed=False,
        gray_meaning='Responsive sites without an existing NMF assignment')
    (DATA/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
    return paths,sheet
