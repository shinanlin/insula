"""Refresh only the two approved editable main Figure 3 schematics via Slurm."""
from pathlib import Path
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import nbformat as nbf
import numpy as np
import pandas as pd
from IPython.display import SVG

ROOT=Path('/cwork/ns458/insula-paper')
sys.path.insert(0,str(ROOT))
AUDIT=ROOT/'results/audit/figure3_venn_restore_overlap_2026-11-01'


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    from src.figure3 import MAIN_RESULTS,MAIN_IMG,sha,style,write
    from src.supplementary.condition_difference_group_panels import diagram
    from scripts.consolidate_figure3 import main_notebook
    style();AUDIT.mkdir(parents=True,exist_ok=True)
    notebook=ROOT/'vizpub/fig3.ipynb';manifest=MAIN_RESULTS/'manifest.json'
    diagrams={key:MAIN_IMG/f'fig3_{key}_condition_difference_schematic.svg' for key in ('picture','lexical')}
    provenance={key:MAIN_RESULTS/'regions'/key/'diagram_area_provenance.json' for key in diagrams}
    owned={*diagrams.values(),*provenance.values(),notebook,manifest}
    protected=[]
    for folder in ('results/nmf','results/fig3','img/fig1','img/fig2','img/fig3',
        'results/supplementary/fig3_sampling_responsiveness','results/supplementary/fig3_all_condition_hga',
        'img/supplementary/fig3_sampling_responsiveness','img/supplementary/fig3_all_condition_hga'):
        protected.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p not in owned)
    protected.extend(ROOT/f'vizpub/fig{i}.ipynb' for i in (1,2,4,5,6))
    protected.append(ROOT/'vizpub/supplementary/fig3_supplementary.ipynb')
    frozen={str(p):sha(p) for p in protected}
    write(AUDIT/'protected.json',frozen)
    before_hashes={str(p):sha(p) for p in owned}
    for path in owned:
        backup=AUDIT/'before'/path.relative_to(ROOT);backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():shutil.copy2(path,backup)
    nb=nbf.read(notebook,as_version=4)
    before_nb=nbf.reads(nbf.writes(nb),as_version=4)
    main_manifest=json.loads(manifest.read_text());proof={}
    svg='{http://www.w3.org/2000/svg}'
    for key,path in diagrams.items():
        folder=MAIN_RESULTS/'regions'/key
        pool=pd.read_csv(folder/'responsive_pool_electrodes.csv')
        groups={name:pd.read_csv(folder/f'{name}_electrodes.csv')
            for name in ('first_only','second_only','no_detected_difference','bidirectional')}
        assert sum(map(len,groups.values()))==len(pool)
        assert diagram(key,pool,groups)==path
        meta=json.loads(provenance[key].read_text())
        old_meta=json.loads((AUDIT/'before'/provenance[key].relative_to(ROOT)).read_text())
        for field in ('target_counts','recovered_counts_from_circle_geometry','circle_totals',
            'radii_data_units','common_area_per_electrode_data_units','pixels_per_data_unit'):
            np.testing.assert_allclose(meta[field],old_meta[field],rtol=0,atol=1e-12)
        tree=ET.parse(path);root=tree.getroot()
        previous=ET.parse(AUDIT/'before'/path.relative_to(ROOT)).getroot()
        old_axes=previous.find(f'.//{svg}g[@id="axes_1"]')
        if old_axes is not None:
            original_paths=[g.find(svg+'path').get('d').strip().removesuffix('z').rstrip()
                for g in old_axes if g.tag==svg+'g' and g.find(svg+'path') is not None][:3]
        else:
            ids=list(old_meta['editable_svg_groups'])
            original_paths=[previous.find(f'.//{svg}g[@id="{ids[i]}_shape"]/{svg}path').get('d').strip().removesuffix('z').rstrip()
                for i in (0,2,1)]
        region_ids=list(meta['editable_svg_groups'])
        # Match the original left/right/center path ordering exactly.
        region_paths={}
        for group_id,spec in meta['editable_svg_groups'].items():
            group=root.find(f'.//{svg}g[@id="{group_id}"]')
            assert group is not None
            assert group.get('transform') is None and spec['offset_mm']==0
            members={child.get('id') for child in group if child.tag==svg+'g'}
            assert members==set(spec['members'])
            shape=group.find(f'{svg}g[@id="{group_id}_shape"]/{svg}path')
            assert shape is not None and shape.get('d').strip().lower().endswith('z')
            assert 'stroke:' not in shape.get('style','')
            assert not any('clip-path' in element.attrib for element in group.iter())
            assert group.find(f'.//{svg}g[@id="{group_id}_count"]/{svg}text') is not None
            region_paths[group_id]=shape.get('d').strip()[:-1].rstrip()
        assert [region_paths[region_ids[i]] for i in (0,2,1)]==original_paths
        outlines=root.find(f'.//{svg}g[@id="circle_outlines"]')
        assert outlines is not None and len(outlines)==2
        assert all(outline.find(svg+'path') is not None for outline in outlines)
        assert not root.findall('.//'+svg+'image')
        assert len([e for e in root.iter() if e.get('id','').startswith('region_') and e.get('id') in region_ids])==3
        cells=[c for c in nb.cells if f"results['{key}']['diagram']" in c.source]
        assert len(cells)==1
        outputs=[o for o in cells[0].get('outputs',[]) if 'image/svg+xml' in o.get('data',{})]
        assert len(outputs)==1
        outputs[0]['data']['image/svg+xml']=SVG(filename=str(path)).data
        main_manifest['svg_sha256'][str(path)]=sha(path)
        proof[key]=dict(svg=str(path),svg_sha256=sha(path),independent_groups=meta['editable_svg_groups'],
            original_closed_paths_identical=True,area_scale_and_counts_identical=True,
            fills_counts_labels_grouped=True,shared_clipping_masks=False,
            conventional_overlapping_circle_outlines=True,no_display_offsets=True)
        print('UPDATED',key,meta['target_counts'],flush=True)
    # Retain every other notebook cell/output; only update the caption and two SVGs.
    nb.cells[0].source=main_notebook().cells[0].source
    for old,new in zip(before_nb.cells,nb.cells):
        if old.cell_type=='code':
            assert old.source==new.source
            if "['diagram']" not in old.source:assert old==new
    assert sha(notebook)==before_hashes[str(notebook)],'Concurrent author edit'
    nbf.validate(nb);nbf.write(nb,notebook);write(manifest,main_manifest)
    assert all(sha(p)==digest for p,digest in frozen.items())
    assert all(sha(p)==digest for p,digest in main_manifest['svg_sha256'].items())
    write(AUDIT/'completion.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        comparisons=proof,protected_other_files_unchanged=len(frozen),
        only_two_diagram_outputs_and_caption_changed=True,no_statistical_or_NMF_changes=True,
        illustrator_import_requires_local_user_check=True))
    print('PASS editable SVG groups, exact original paths/areas/counts;',len(frozen),'other files unchanged',flush=True)


if __name__=='__main__':main()
