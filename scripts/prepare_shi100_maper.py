"""Restore MAPER inputs and prepare seven native targets on a compute node."""
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path('/hpc/group/coganlab/nanlinshi/insula-functional')
ROOT = Path('/cwork/ns458/maper_run')
ANC = ROOT / 'shi100_ancillaries_20261005'
CACHE = Path('/hpc/group/coganlab/nanlinshi/cache/maper')
SIF = Path('/hpc/group/coganlab/nanlinshi/maper_tool/maper.sif')
SUBJECTS = ['D148', 'D149', 'D150', 'D152A', 'D152B', 'D153', 'D154']
URL = 'https://soundray.org/maper/hammers_mith-ancillaries-n30r95.tar'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / 'hammers_mith-ancillaries-n30r95.tar'
    if not archive.is_file():
        partial = archive.with_suffix('.tar.part')
        with urllib.request.urlopen(URL, timeout=120) as response, partial.open('wb') as f:
            shutil.copyfileobj(response, f)
        # Validate before making this the reusable archive.
        with tarfile.open(partial):
            pass
        partial.rename(archive)
    ANC.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if not parts or parts[0] not in {'onepad', 'posnorm'} or not member.isfile():
                continue
            if '..' in parts:
                raise ValueError(f'Unsafe archive path: {member.name}')
            target = ANC / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('wb') as dest:
                shutil.copyfileobj(source, dest)
    (ANC / 'seg/seg95').mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / 'ancillaries/src-description.csv', ANC / 'src-description.csv')
    for i in range(1, 31):
        label = ROOT / f'ancillaries/seg/seg95/a{i}.nii.gz'
        target = ANC / f'seg/seg95/a{i}.nii.gz'
        shutil.copy2(label, target)
        t1 = ANC / f'onepad/a{i}.nii.gz'
        transform = ANC / f'posnorm/a{i}.dof.gz'
        if not transform.is_file() or transform.stat().st_size == 0:
            raise ValueError(f'Missing initial transform: {transform}')
        a, b = nib.load(t1), nib.load(target)
        if a.shape[:3] != b.shape[:3] or not np.allclose(a.affine, b.affine, atol=1e-3):
            raise ValueError(f'Atlas {i} T1/label geometry mismatch')
    print('Verified all 30 atlas T1/transform/corrected-label sets', flush=True)
    jobs = []
    records = []
    for subject in SUBJECTS:
        maper_id = 'D' + subject[1:].rstrip('AB').zfill(4) + (subject[-1] if subject[-1] in 'AB' else '')
        recon = Path('/cwork/ns458/ECoG_Recon') / subject / 'mri'
        orig_path, brain_path = recon / 'orig.mgz', recon / 'brainmask.mgz'
        orig, brain = nib.load(orig_path), nib.load(brain_path)
        if orig.shape != brain.shape or not np.allclose(orig.affine, brain.affine, atol=1e-3):
            raise ValueError(f'{subject}: orig/brainmask geometry mismatch')
        run = ROOT / maper_id
        (run / 'target').mkdir(parents=True, exist_ok=True)
        (run / 'output').mkdir(exist_ok=True)
        link = run / 'ancillaries'
        if not link.exists():
            link.symlink_to(ANC, target_is_directory=True)
        elif link.resolve() != ANC.resolve():
            raise ValueError(f'Existing ancillaries link points elsewhere: {link}')
        out = run / f'target/sub-{maper_id}_T1w_brain.nii.gz'
        nib.save(nib.Nifti1Image(np.asarray(brain.dataobj, dtype=np.float32), brain.affine), out)
        check = nib.load(out)
        if check.shape != orig.shape or not np.allclose(check.affine, orig.affine, atol=1e-3):
            raise ValueError('Converted target geometry differs from orig')
        (run / 'target/tgt-description.csv').write_text(f'id, mri\n{maper_id}, {out.name}\n')
        subprocess.run(['bash', str(REPO / 'pipeline/03_run_maper/generate_launchlist.sh'),
                        maper_id, str(run), str(SIF)], check=True)
        launch = run / f'launchlist_{maper_id}.sh'
        commands = launch.read_text().splitlines()
        if len(commands) != 30 or not all(c.startswith('/opt/maper/maper ') for c in commands):
            raise ValueError(f'Invalid launchlist: {launch}')
        for i in range(1, 31):
            jobs.append([subject, maper_id, str(launch), i])
        records.append(dict(subject=subject, maper_id=maper_id, orig_sha256=sha(orig_path),
                            brainmask_sha256=sha(brain_path), target=str(out)))
        print('Prepared', subject, maper_id, flush=True)
    manifest = REPO / 'logs/slurm/shi100_maper_20261005.tsv'
    with manifest.open('w', newline='') as f:
        csv.writer(f, delimiter='\t', lineterminator='\n').writerows(jobs)
    (ANC / 'shi100_provenance.json').write_text(json.dumps(dict(
        archive_url=URL, archive_sha256=sha(archive), subjects=records,
        labels='Existing geometry-corrected ancillaries/seg/seg95, copied unchanged',
        target_conversion='brainmask.mgz -> float32 NIfTI with unchanged native affine'), indent=2) + '\n')
    print('READY', manifest, 'tasks', len(jobs), flush=True)


if __name__ == '__main__':
    main()
