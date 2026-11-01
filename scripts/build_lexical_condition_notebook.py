"""Update only the explicitly authorized supplementary notebook source."""
from pathlib import Path
import json

ROOT=Path('/cwork/ns458/insula-paper')
NB=ROOT/'vizpub/supplementary/fig3_lexical_delay_condition_sets.ipynb'


def sampling_response_cells():
    import nbformat as nbf
    note=nbf.v4.new_markdown_cell('''## Sampling and Delay-responsive sites

The map includes all **187 eligible paired insula sites / 35 participants**. Purple highlights the **78 sites** with a retained native baseline response to Decision OR Repeat in **Delay [0,1] s**. Gray denotes the remaining **109 sites**, for which neither condition detected a response in this interval. Each sampled site appears once; hemisphere counts refer to all sampled sites. These are the same 78 responsive sites subsequently separated into Decision-only 19, Both 48 and Repeat-only 11 in the Venn and condition-colored map.

This uses the same sampling/response colors, surface, cameras, 24-pixel markers and 4-mm display lift as the corresponding Picture Naming sampling panel. The saved response masks and analytic coordinates are reused unchanged.
''',id='lexical-sampling-response-note')
    code=nbf.v4.new_code_cell('''from pathlib import Path
import sys
from IPython.display import SVG,display
_lexical_paper_root=Path('/cwork/ns458/insula-paper')
if str(_lexical_paper_root) not in sys.path:sys.path.insert(0,str(_lexical_paper_root))
from src.supplementary.lexical_sampling_response import render as render_lexical_sampling_response
lexical_sampling_response_path=render_lexical_sampling_response()
display(SVG(filename=str(lexical_sampling_response_path)))
print('Sampling/response SVG:',lexical_sampling_response_path)
''',id='lexical-sampling-response-map')
    return [note,code]


def component_variant_cells():
    import nbformat as nbf
    note=nbf.v4.new_markdown_cell('''### Additional display: Decision > Repeat by functional component

The same **31 sites / 18 participants** are shown in the frozen NMF colors: Sensory blue (3), Sustained gold (14), Motor red (12). The two sites without an existing NMF assignment are gray and explicitly labeled Unassigned; they are significant sites. The original single-color maps remain available above. Electrode membership, coordinates, marker size, hemisphere geometry, surface projection and statistics are unchanged.
''',id='decision-functional-color-note')
    code=nbf.v4.new_code_cell('''from pathlib import Path
import sys
from IPython.display import SVG, display
_paper_root=Path('/cwork/ns458/insula-paper')
if str(_paper_root) not in sys.path: sys.path.insert(0,str(_paper_root))
from src.supplementary.lexical_component_color_variant import render as render_functional_color_variant
functional_color_variant=render_functional_color_variant()
display(SVG(filename=str(functional_color_variant)))
print('Additional SVG:',functional_color_variant)
''',id='decision-functional-color-map')
    return [note,code]


def build():
    import nbformat as nbf
    old=nbf.read(NB,as_version=4)
    md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    cells=[md('''# Lexical decision/repetition: complete insula response characterization

**187 anatomically accepted, native-good paired bipolar sites / 35 participants.**
The starting anatomical sample is 214 sites / 38 participants; 27 sites have existing native bad-channel/endpoint QC. No NMF membership or prior response gate defines this denominator.

The sequence is: complete sampling → baseline-response classes and corresponding locations/waveforms → direct condition differences → relation to the frozen NMF populations. **Figure 1, Figure 2, NMF, other main notebooks, and all original 123-site tests remain unchanged.**

## Definitions and inference

- Baseline classes use the retained native Decision and Repeat masks in **Delay [0,1] s**, inclusive, 129 samples. “Decision only” means a baseline response was detected only for Decision; it does not itself mean Decision > Repeat. Neither means neither baseline test detected a response, not absence of neural activity.
- Direct differences retain the original per-electrode **two-sided iEEG `time_perm_cluster`**, default t statistic, 2,000 permutations, `p_thresh=.05`, `p_cluster=.05`, independent trial permutations, seed 42. Exact original CORRECT-only trial labels and the two previously approved runtime NumPy sorting fixes are retained. The 123 completed tests are reused; only 64 additional sites are tested. Both directions are extracted from the same corrected two-sided mask and pointwise p<.05, according to the sign of Decision minus Repeat.
- Correction remains across time **within each electrode**, with no additional across-electrode correction. A site can have significant intervals in both directions; both flags and intervals are retained.
- Waveforms average all retained native epochs within each electrode, then equally weight electrodes, with SEM across electrode means, as in the previous figures. They are descriptive; old 123-site pooled significance bars are not transferred to new subsets. No smoothing is added. The 3 phase widths scale with their displayed durations: Stimulus −.5–.6 s, Delay 0–1 s, Response −.5–1 s. Time zero is stimulus onset, stimulus offset, and articulation onset, respectively. Each waveform figure uses one shared y scale across all rows and phases, covering the full mean ± SEM range.
- Component summaries reuse the existing assignments: **135 assigned sites**, plus **52 explicitly Unassigned sites**. All component counts/proportions and plotted effect summaries are descriptive; no NMF refit or new component-level significance claim is made.
- Numbers of participants overlap between groups and cannot be added. Selecting a set for its Delay response makes its waveform a descriptive characterization of that selection, not independent confirmation of the selection.
'''),code('''from pathlib import Path
import json
import sys
import pandas as pd
from IPython.display import SVG, display
ROOT=Path('/cwork/ns458/insula-paper')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.supplementary import lexical_condition_figures as plots
from scripts.refresh_lexical_condition_sets import protected
table,waves,counts,composition,summary=plots.load()
assert protected()>0
display(counts.drop(columns='participants'))
display(pd.DataFrame({'Stage':['Anatomical sample','Existing native QC excluded','Eligible paired sample','Previously tested','Newly tested','Frozen NMF assigned','Unassigned'], 'Electrodes':[214,27,187,123,64,135,52]}))
'''),md('''## 1. Baseline-response classes and matching anatomy

Decision 67; Repeat 59; Decision-only 19, Both 48, Repeat-only 11, Neither 109. The bilateral map shows the 78 sites in the union, once per site, colored exactly like the Venn. Neither is counted explicitly and is not drawn as gray context points. Brain locations retain the accepted Figure 1 surface, camera-facing snap and 4-mm display lift. Analytic coordinates remain unchanged.
'''),code('''venn_path=plots.draw_venn(table)
brain_paths=plots.draw_brains(table)
display(SVG(filename=str(venn_path)))
display(SVG(filename=str(brain_paths[0])))
'''),md('''## 2. Corresponding waveforms

Each row uses exactly the membership shown in its label, including a full 187-site reference and all four baseline classes. Decision is purple and Repeat black. Curves are electrode mean ± SEM, with the same temporal scale per unit axis width across all phases. These are response descriptions, without newly inferred group significance bars.
'''),code('''wave_paths=plots.draw_waveform_sets(table,waves)
display(SVG(filename=str(wave_paths[0])))
'''),md('''## 3. Direct condition differences in both directions

All 187 sites have now received the same two-sided test. Baseline responsiveness and direct condition modulation are reported separately. The original 123-site positive set is checked against its saved 29-site list; the 64 newly tested sites are reported separately in the exported table. “No detected difference” does not imply equivalent responses. Difference-selected curves illustrate the selected sites and are not independent tests of that contrast.
'''),code('''display(counts.loc[counts['set'].isin(['Decision > Repeat','Repeat > Decision'])].drop(columns='participants'))
display(pd.read_csv(plots.OUT/'response_difference_counts.csv'))
for p in brain_paths[1:]: display(SVG(filename=str(p)))
display(SVG(filename=str(wave_paths[1])))
display(table.loc[table.decision_gt_repeat|table.repeat_gt_decision,
    ['subject','channel','functional_cluster','baseline_class','test_origin','n_decision_correct','n_repeat_correct','decision_gt_repeat_intervals','repeat_gt_decision_intervals']])
'''),md('''## 4. Relation to the frozen temporal components

Condition-class colors and NMF-component colors describe different classifications. This summary retains all 187 sites, with the 52 sites outside the frozen NMF explicitly gray/Unassigned. Counts are shown both as composition of each set and as detected-difference fractions within each component's full eligible denominator. The accompanying component curves use every eligible site with that fixed label, not just the contrast-selected sites. These descriptive results do not establish a component × condition interaction.
'''),code('''component_path=plots.draw_components(table,composition)
display(SVG(filename=str(component_path)))
display(SVG(filename=str(wave_paths[2])))
display(composition)
'''),md('''## 5. Review sheet and reproducibility

Standalone SVG panels and a complete review sheet are exported to the existing supplementary image folder. Source file hashes, exact trial selection, all per-site time bins and both-direction intervals are retained under `results/supplementary/fig3_lexical_delay_condition_sets/full_cohort_2026-10-27/`. The previous notebook and outputs are archived under `results/audit/lexical_condition_sets_full_2026-10-27/before/`. The original 123-site pooled HGA curve and group significance remain available in `decision_repeat_test.ipynb`; their significance bars are not reused for this expanded cohort.
'''),code('''pair_sheet=plots.assemble_condition_pair(venn_path,brain_paths[0])
plots.refresh_compatibility_exports(table)
panels=[(venn_path,160.,280.),(brain_paths[0],110.,400.),(wave_paths[0],30.,590.),
    (component_path,20.,610.),(brain_paths[1],110.,400.),(brain_paths[2],110.,400.),
    (wave_paths[1],30.,590.),(wave_paths[2],30.,590.)]
sheet=plots.assemble(panels,'fig3_lexical_delay_full_summary.svg')
specification=dict(notebook=str(Path('/cwork/ns458/insula-paper/vizpub/supplementary/fig3_lexical_delay_condition_sets.ipynb')),
    lexical_window=[0.,1.],lexical_n_sites=187,lexical_n_participants=35,
    new_tests=64,reused_tests=123,parameters=summary['parameters'],
    direct_difference_counts=counts.loc[counts['set'].isin(['Decision > Repeat','Repeat > Decision'])].to_dict('records'),
    sheet=str(sheet),condition_set_pair=str(pair_sheet),colors=plots.COLORS,component_colors=plots.COMPONENT_COLORS,
    original_brain_display_preserved=True,new_component_inference=False,
    main_figures_and_nmf_unchanged=bool(protected()),source_results=str(plots.OUT))
(plots.BASE/'specification.json').write_text(json.dumps(specification,indent=2)+'\\n')
for filename in ['electrodes.csv','set_counts.csv','lexical_brain_display_coordinates.csv']:
    source=plots.OUT/filename
    destination=plots.BASE/('lexical_delay_0_1_electrodes.csv' if filename=='electrodes.csv' else filename)
    destination.write_bytes(source.read_bytes())
display(SVG(filename=str(sheet)))
print('Saved:',sheet)
print('Protected files verified:',protected())
''')]
    cells[8:8]=component_variant_cells()
    cells[2:2]=sampling_response_cells()
    new=nbf.v4.new_notebook(cells=cells,metadata=old.metadata)
    new.metadata['lexical_full_cohort_update']='Author approved 2026-10-27; no main-figure or NMF changes'
    nbf.write(new,NB)


if __name__=='__main__':build()
