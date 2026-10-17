"""Anatomical screen safeguards, using the real canonical atlas helpers."""

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/screen_recon_aic_contacts.py'
spec = importlib.util.spec_from_file_location('screen_recon', SCRIPT)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


@pytest.fixture(scope='module')
def pipeline():
    return screen.load_pipeline(screen.UPSTREAM)


def case(tmp_path, points, labels, tissues, subject='D152B', prefix='D152'):
    recon_root = tmp_path / 'recon'
    recon = recon_root / subject
    elec = recon / 'elec_recon'
    elec.mkdir(parents=True)
    (recon / 'mri').mkdir()
    # An asymmetric voxel grid and nonidentity affine expose scanner-RAS/tkRAS confusion.
    affine = np.array([[0, 0, -1, 7], [-1, 0, 0, 8], [0, 1, 0, -9], [0, 0, 0, 1.]])
    orig = nib.MGHImage(np.zeros((7, 8, 9), dtype=np.float32), affine)
    nib.save(orig, recon / 'mri/orig.mgz')
    orig = nib.load(recon / 'mri/orig.mgz')
    xyz = nib.affines.apply_affine(orig.header.get_vox2ras_tkr(), np.array(points))
    rows = [f'LI {i+1} {x:.6f} {y:.6f} {z:.6f} L D' for i, (x, y, z) in enumerate(xyz)]
    (elec / f'{prefix}_elec_locations_RAS.txt').write_text('\n'.join(rows) + '\n')
    (elec / f'{prefix}.electrodeNames').write_text('01-Oct-2026\nName, Depth/Strip/Grid, Hem\n' +
                                                  ''.join(f'LI{i+1} D L\n' for i in range(len(xyz))))
    np.savetxt(elec / f'{prefix}.LEPTO', xyz, header='date\nR A S', comments='')
    maper_root = tmp_path / 'maper'
    output = maper_root / 'D0152B/output'
    output.mkdir(parents=True)
    seg = np.zeros(orig.shape, dtype=np.int16)
    tissue = np.zeros(orig.shape, dtype=np.int16)
    for point, label, kind in zip(points, labels, tissues):
        if all(0 <= v < dim for v, dim in zip(point, orig.shape)):
            seg[tuple(point)] = label
            tissue[tuple(point)] = kind
    nib.save(nib.Nifti1Image(seg, orig.affine), output / 'f30-seg95-D0152B.nii.gz')
    nib.save(nib.Nifti1Image(tissue, orig.affine), output / 'f30-seg95-D0152B-tc3crisp.nii.gz')
    return recon_root, maper_root, output


def test_native_lookup_gray_matter_gate_and_split_prefix(tmp_path, pipeline):
    recon, maper, _ = case(tmp_path, [(1, 1, 1), (2, 2, 2), (3, 3, 3)], [86, 86, 94], [2, 3, 2])
    summary, contacts = screen.screen_subject(pipeline, 'D152B', 'D0152B', recon, maper)
    assert summary['electrode_prefix'] == 'D152'
    assert summary['n_aic'] == 1 and summary['n_pic'] == 1
    assert [r['roi'] for r in contacts] == ['AIC', 'Unknown', 'PIC']
    assert summary['aic_contacts'] == 'LI1'
    assert summary['status'] == 'aic_detected_needs_qc'


def test_missing_tissue_cannot_be_reported_as_no_aic(tmp_path, pipeline):
    recon, maper, output = case(tmp_path, [(1, 1, 1)], [86], [2])
    (output / 'f30-seg95-D0152B-tc3crisp.nii.gz').unlink()
    summary, contacts = screen.screen_subject(pipeline, 'D152B', 'D0152B', recon, maper)
    assert summary['status'] == 'pending_inputs'
    assert summary['n_aic'] is None
    assert contacts[0]['status'] == 'not_evaluated'


def test_outside_atlas_cannot_support_negative_screen(tmp_path, pipeline):
    recon, maper, _ = case(tmp_path, [(20, 1, 1)], [0], [0])
    summary, _ = screen.screen_subject(pipeline, 'D152B', 'D0152B', recon, maper)
    assert summary['status'] == 'incomplete_contact_review'
    assert summary['n_invalid'] == 1


def test_geometry_mismatch_rejected(tmp_path, pipeline):
    recon, maper, output = case(tmp_path, [(1, 1, 1)], [86], [2])
    path = output / 'f30-seg95-D0152B.nii.gz'
    image = nib.load(path)
    affine = image.affine.copy()
    affine[0, 3] += 2
    nib.save(nib.Nifti1Image(np.asarray(image.dataobj), affine), path)
    with pytest.raises(ValueError, match='voxel geometry'):
        screen.screen_subject(pipeline, 'D152B', 'D0152B', recon, maper)


def test_negative_screen_still_requires_qc(tmp_path, pipeline):
    recon, maper, _ = case(tmp_path, [(1, 1, 1)], [94], [2])
    summary, _ = screen.screen_subject(pipeline, 'D152B', 'D0152B', recon, maper)
    assert summary['status'] == 'no_aic_at_contact_centers_needs_qc'
    assert summary['n_aic'] == 0
