#!/bin/bash
#SBATCH --job-name=insula_anatomy_inventory
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/anatomy_inventory_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/anatomy_inventory_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common
#SBATCH --chdir=/cwork/ns458/insula-paper
set -euo pipefail
mkdir -p results/audit/zero_endpoints_2026-10-17
python - <<'PY'
import json,os,re,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

roots=[Path('/hpc/group/coganlab')/x for x in ['ECoG_Recon','akm87','ht203','ln154','ae166','jz421','jq81']]
roots += [Path('/hpc/group/coganlab/nanlinshi')/x for x in ['cache','pycortex_store','pycortex_cache','tmp','af','insula','insula-functional','seeg-preprocessing']]
roots += [Path('/cwork/ns458')/x for x in ['ecog_recon','ECoG_Recon','maper_run','maper_run_qc_rerun','CoganLab','BIDS-1.0_LexicalDecRepDelay','BIDS-1.3_PictureNaming','BIDS-1.4_Phoneme_sequencing','BIDS-1.4_SentenceRep','insula_rebuttal']]
prune={'.git','node_modules','miniconda3','.conda','.cache','site-packages','anaconda3','venv','.venv','freesurfer','FreeSurfer','software','Software','__pycache__','epoch(band)(zscore)','epoch(band)(sig)','epoch(band)','epoch(bipolar)','epoch(car)','raw','wave','ieeg','mne_cache'}
names={'orig.mgz','aseg.mgz','aparc.a2009s+aseg.mgz','aparc+aseg.mgz','wm.mgz','ribbon.mgz','raw.nii.gz','whitematter.nii.gz'}
def scan(root):
 found=[];n=0;limit=time.monotonic()+240
 if not root.exists():return dict(root=str(root),status='missing',found=[])
 for base,dirs,files in os.walk(root):
  dirs[:]=[d for d in dirs if d not in prune]
  for name in files:
   if name in names or ('tc3crisp' in name or ('seg95' in name and name.endswith(('.nii','.nii.gz'))) or (re.search('([Ee][Cc][Oo][Gg].*[Rr]econ|[Rr]econ.*[Bb]ack|[Rr]econ.*[Aa]rch)',name) and name.endswith(('.tar','.tar.gz','.tgz','.zip')))):
    p=Path(base)/name
    try:st=p.stat();found.append(dict(path=str(p),size=st.st_size,mtime_ns=st.st_mtime_ns))
    except OSError:pass
  n+=1
  if n>150000 or time.monotonic()>limit:return dict(root=str(root),status='bounded_stop',dirs=n,found=found)
 return dict(root=str(root),status='complete',dirs=n,found=found)
with ThreadPoolExecutor(max_workers=2) as pool:
 results=list(pool.map(scan,roots))
out=Path('results/audit/zero_endpoints_2026-10-17/anatomy_inventory.json')
out.write_text(json.dumps(results,indent=2))
for r in results:print(r['root'],r['status'],'files',len(r['found']),flush=True)
print('Total matching anatomy files:',sum(len(r['found']) for r in results),flush=True)
PY
