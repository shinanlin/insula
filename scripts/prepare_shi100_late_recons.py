"""Validate and prepare D157/D158 using the existing, verified MAPER atlas set."""
import csv
import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from prepare_shi100_maper import ANC, REPO, ROOT, SIF, sha
from screen_recon_aic_contacts import read_contacts


def main():
    # Shared atlas inputs are read-only; do not rerun the original restoration.
    for i in range(1, 31):
        t1, label = (nib.load(ANC / name) for name in
                     (f'onepad/a{i}.nii.gz', f'seg/seg95/a{i}.nii.gz'))
        assert t1.shape[:3] == label.shape[:3]
        assert np.allclose(t1.affine, label.affine, atol=1e-3)
        assert (ANC / f'posnorm/a{i}.dof.gz').stat().st_size > 0

    inventory = json.loads((REPO / 'logs/slurm/shi100_box_inventory_20261006.json').read_text())
    jobs, records = [], []
    for subject in ('D157', 'D158'):
        maper_id = f'D{int(subject[1:]):04d}'
        recon = Path('/cwork/ns458/ECoG_Recon') / subject
        # Compare every directly inventoried file with the Windows listing.
        for folder, record in inventory['subjects'][subject].items():
            assert record['returncode'] == 0, (subject, folder)
            for entry in record['listing']['DATA']:
                if entry['type'] == 'file':
                    path = recon / folder / entry['name']
                    assert path.is_file() and path.stat().st_size == entry['size'], path
        assert (recon / 'scripts/recon-all.done').is_file()
        assert not list((recon / 'scripts').glob('IsRunning*'))
        assert not (recon / 'scripts/recon-all.error').exists()
        contacts, coordinate_file, prefix = read_contacts(recon)
        assert all(c['contact_type'] == 'D' for c in contacts), subject
        orig_path, brain_path = recon / 'mri/orig.mgz', recon / 'mri/brainmask.mgz'
        orig, brain = nib.load(orig_path), nib.load(brain_path)
        assert orig.shape == brain.shape and np.allclose(orig.affine, brain.affine, atol=1e-3)
        run = ROOT / maper_id
        run.mkdir(exist_ok=True)
        (run / 'target').mkdir(exist_ok=True)
        (run / 'output').mkdir(exist_ok=True)
        link = run / 'ancillaries'
        if not link.exists():
            link.symlink_to(ANC, target_is_directory=True)
        assert link.resolve() == ANC.resolve(), link
        target = run / f'target/sub-{maper_id}_T1w_brain.nii.gz'
        if not target.exists():
            nib.save(nib.Nifti1Image(np.asarray(brain.dataobj, dtype=np.float32), brain.affine), target)
        check = nib.load(target)
        assert check.shape == orig.shape and np.allclose(check.affine, orig.affine, atol=1e-3)
        assert np.array_equal(np.asarray(check.dataobj), np.asarray(brain.dataobj)), target
        (run / 'target/tgt-description.csv').write_text(f'id, mri\n{maper_id}, {target.name}\n')
        subprocess.run(['bash', str(REPO / 'pipeline/03_run_maper/generate_launchlist.sh'),
                        maper_id, str(run), str(SIF)], check=True)
        launch = run / f'launchlist_{maper_id}.sh'
        commands = launch.read_text().splitlines()
        assert len(commands) == 30 and all(c.startswith('/opt/maper/maper ') for c in commands)
        jobs.extend([subject, maper_id, str(launch), i] for i in range(1, 31))
        records.append(dict(subject=subject, maper_id=maper_id, n_contacts=len(contacts),
                            coordinate_file=str(coordinate_file), electrode_prefix=prefix,
                            coordinate_sha256=sha(coordinate_file), orig_sha256=sha(orig_path),
                            brainmask_sha256=sha(brain_path), target=str(target)))
        print('Prepared', subject, len(contacts), 'contacts', flush=True)
    manifest = REPO / 'logs/slurm/shi100_maper_20261006.tsv'
    with manifest.open('w', newline='') as stream:
        csv.writer(stream, delimiter='\t', lineterminator='\n').writerows(jobs)
    (REPO / 'logs/slurm/shi100_prepare_20261006.json').write_text(
        json.dumps(dict(subjects=records, ancillaries=str(ANC), manifest=str(manifest)), indent=2)+'\n')
    print('READY', manifest, 'tasks', len(jobs), flush=True)


if __name__ == '__main__':
    main()
