#!/usr/bin/env python3
"""Prepare one isolated native D0125 MAPER run; never modify BIDS or NMF."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import nibabel as nib
import numpy as np

BASE = Path('/cwork/ns458/insula-paper')
RUN = Path('/cwork/ns458/maper_run/restore_20261017/D0125')
ANC = RUN.parent / 'ancillaries'
RECON = Path('/cwork/ns458/ECoG_Recon/D125')
OUT = BASE / 'results/audit/maper_D0125_2026-10-17'
SIF = Path('/hpc/group/coganlab/nanlinshi/maper_tool/maper.sif')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    expected = {r['path']: r['sha256'] for r in json.loads((BASE /
        'docs/evidence/globus_recon_restore_2026-10-17/restored_file_hashes.json').read_text())}
    inputs = [RECON / 'mri' / n for n in
              ['orig.mgz', 'brainmask.mgz', 'aparc.a2009s+aseg.mgz']]
    inputs += sorted((RECON / 'elec_recon').glob('*elec_locations_RAS*.txt'))
    for path in inputs:
        if sha(path) != expected[str(path)]:
            raise ValueError(f'Restored input changed since verification: {path}')
    atlas_manifest = json.loads((BASE /
        'docs/evidence/globus_recon_restore_2026-10-17/atlas_input_restore.json').read_text())
    for atlas in atlas_manifest['atlas_records']:
        a, b = nib.load(atlas['t1']), nib.load(atlas['labels'])
        if (a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-4)
                or sha(Path(atlas['labels'])) != atlas['labels_sha256']):
            raise ValueError(f"Atlas changed since verification: {atlas['atlas_id']}")
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / 'output'
    if output.exists() and any(output.iterdir()):
        raise ValueError('Run output already populated; inspect before preparing again')
    output.mkdir(exist_ok=True)
    (RUN / 'target').mkdir(exist_ok=True)
    link = RUN / 'ancillaries'
    if not link.exists():
        link.symlink_to(ANC, target_is_directory=True)
    if link.resolve() != ANC.resolve():
        raise ValueError('Unexpected ancillaries link')
    orig, brain = [nib.load(p) for p in inputs[:2]]
    if orig.shape != brain.shape or not np.allclose(orig.affine, brain.affine, atol=1e-4):
        raise ValueError('Native orig/brainmask geometry differs')
    target = RUN / 'target/sub-D0125_T1w_brain.nii.gz'
    nib.save(nib.Nifti1Image(np.asarray(brain.dataobj, dtype=np.float32), brain.affine), target)
    check = nib.load(target)
    if (check.shape != orig.shape or not np.allclose(check.affine, orig.affine, atol=1e-4)
            or not np.array_equal(np.asarray(check.dataobj), np.asarray(brain.dataobj))):
        raise ValueError('Target conversion changed MRI data or geometry')
    (RUN / 'target/tgt-description.csv').write_text('id, mri\nD0125, ' + target.name + '\n')
    subprocess.run(['bash', str(BASE / 'pipeline/03_run_maper/generate_launchlist.sh'),
                    'D0125', str(RUN), str(SIF)], check=True)
    launch = RUN / 'launchlist_D0125.sh'
    commands = launch.read_text().splitlines()
    if len(commands) != 30 or not all(c.startswith('/opt/maper/maper ') for c in commands):
        raise ValueError('Expected 30 full MAPER atlas pairings')
    # Retain every target crisp map to check consistency across pairings.
    # The first pairing also retains the native tissue probabilities.
    commands = [c + ' -tc3crisp' + (' -tc3out' if i == 0 else '')
                for i, c in enumerate(commands)]
    for i, command in enumerate(commands, 1):
        args = shlex.split(command)
        assert args[args.index('-srcid') + 1] == f'a{i}'
        assert args[args.index('-tgtid') + 1] == 'D0125'
        assert args[args.index('-output-dir') + 1] == str(output)
        assert '-quicktest' not in args and '-notc3' not in args
    launch.write_text('\n'.join(commands) + '\n')
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), subject='D0125',
        recon_subject='D125', run=str(RUN), target=str(target), launchlist=str(launch),
        container=str(SIF), n_atlases=30, launchlist_sha256=sha(launch),
        input_sha256={str(p): sha(p) for p in inputs}, target_sha256=sha(target),
        atlas_manifest_sha256=sha(BASE /
            'docs/evidence/globus_recon_restore_2026-10-17/atlas_input_restore.json'),
        coordinate_space='Native FreeSurfer tkRAS millimetres via inv(orig.vox2ras_tkr)',
        focus='LIA6 and LIA5-6 midpoint only; no cohort rerun, no NMF')
    (OUT / 'run_inputs.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
