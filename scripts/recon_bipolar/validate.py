"""Independently re-sample native atlas voxels and check delivered tables."""
import json
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from run import load_pipeline, UPSTREAM, merge_label
from schema import MAIN_COLUMNS

out = Path(sys.argv[1])
pipeline = load_pipeline(UPSTREAM / 'common/parcellation.py')
lut = pipeline._load_lut(UPSTREAM / 'common/FreeSurferColorLUT.txt')
hlut = pipeline._load_hammers_lut(UPSTREAM / 'common/Hammers_mith_atlases_n30r95_label_indices_SPM12_20160111.txt')
summary = pd.read_csv(out / 'coverage_summary.csv')
assert len(summary) == 8
checked = 0
for item in summary.itertuples():
    recon = Path('/cwork/ns458/ECoG_Recon') / item.recon_subject
    mid = 'D0152A' if item.subject == 'D0152' else item.subject
    maper = Path('/cwork/ns458/maper_run') / mid / 'output'
    orig = nib.load(recon / 'mri/orig.mgz')
    aparc = nib.load(recon / 'mri/aparc.a2009s+aseg.mgz')
    assert orig.shape == aparc.shape
    np.testing.assert_allclose(orig.affine, aparc.affine, atol=1e-4)
    adata = np.asarray(aparc.dataobj)
    hdata = np.asarray(nib.load(maper / f'f30-seg95-{mid}.nii.gz').dataobj).squeeze()
    tissue = np.asarray(nib.load(maper / f'f30-seg95-{mid}-tc3crisp.nii.gz').dataobj).squeeze()
    df = pd.read_csv(out / item.subject / f'sub-{item.subject}_bipolar_parcellation.csv', keep_default_na=False)
    assert df.columns.tolist() == MAIN_COLUMNS
    pair_audit = pd.read_csv(out / item.subject / 'qc/bipolar_pair_audit.csv', keep_default_na=False)
    assert set(zip(df.subject, df.name)) == set(zip(pair_audit.subject, pair_audit.name))
    df = df.merge(pair_audit, on=['subject', 'name'], validate='one_to_one')
    assert len(df) == item.n_bipolar and df.name.is_unique
    np.testing.assert_allclose(df[['x','y','z']],
        (df[['x1','y1','z1']].to_numpy() + df[['x2','y2','z2']].to_numpy()) / 2, atol=1e-12)
    distances = np.linalg.norm(df[['x1','y1','z1']].to_numpy() - df[['x2','y2','z2']].to_numpy(), axis=1)
    np.testing.assert_allclose(distances, df.spacing_mm)
    assert (distances <= 20).all()
    for prefix, suffix, labelcol in [('contact_1','1','contact_1_label'), ('midpoint','','midpoint_label'), ('contact_2','2','contact_2_label')]:
        xyz = df[[a + suffix for a in 'xyz']].to_numpy()
        ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(orig.header.get_vox2ras_tkr()), xyz)).astype(int)
        # The vectorized affine check is independent of the production point helper.
        assert (ijk >= 0).all() and (ijk < np.array(orig.shape)).all()
        indices = tuple(ijk.T)
        a = [lut.get(int(v), str(int(v))) for v in adata[indices]]
        h = [hlut.get(int(v), str(int(v))) for v in np.where(tissue[indices] == 2, hdata[indices], 0)]
        assert a == df[f'{prefix}_aparc_label'].tolist()
        assert h == df[f'{prefix}_hammers_label'].tolist()
        assert [merge_label(x,y)[0] for x,y in zip(a,h)] == df[labelcol].tolist()
        checked += len(df)
    audit = pd.read_csv(out / item.subject / 'qc/contact_label_audit.csv')
    orphans = pd.read_csv(out / item.subject / 'qc/orphans.csv')
    assert set(audit.contact) == set(df.contact_1) | set(df.contact_2) | set(orphans.contact)
    assert not ((set(df.contact_1) | set(df.contact_2)) & set(orphans.contact))
    for flag in ('aic_consensus','aic_any_point','aic_midpoint','aic_all_three'):
        assert int(df[flag].sum()) == getattr(item, 'n_' + flag)
report = dict(status='passed', subjects=len(summary), bipolar_pairs=int(summary.n_bipolar.sum()),
              independently_sampled_positions=checked,
              checks=['atlas geometry', 'native voxel labels', 'hybrid labels', 'midpoints and spacing',
                      'unique names', 'all contacts accounted for', 'coverage counts'])
(out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
