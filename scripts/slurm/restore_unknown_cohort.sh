#!/bin/bash
#SBATCH --job-name=insula_unknown_cohort
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/unknown_cohort_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/unknown_cohort_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,shutil
root=Path('/cwork/ns458/insula-paper')
archive=root/'docs/evidence/before_unknown_endpoints_2026-10-17'
if not archive.exists():
    archive.mkdir()
    for name in ['sound_only_cohort_2026-10-17','figure1_sound_only_2026-10-17','figure2_sound_only_2026-10-17']:
        shutil.copytree(root/'docs/evidence'/name,archive/name)
    for name in ['fig1','fig2']:
        shutil.copy2(root/f'vizpub/{name}.ipynb',archive/f'{name}.ipynb')
    hashes={str(p.relative_to(archive)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in archive.rglob('*') if p.is_file()}
    (archive/'archive_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
else:
    hashes=json.loads((archive/'archive_manifest.json').read_text())
    assert all(hashlib.sha256((archive/p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
print(f'Preserved previous sound-only cohort and figures: {archive}',flush=True)
PY
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/build_sound_only_cohort.py
