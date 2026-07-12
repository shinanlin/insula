# HGA Explorer — HPC access

Module: `viewer/hga_explorer`

## Build production bundle

```bash
cd /hpc/group/coganlab/nanlinshi/insula/viewer/hga_explorer
npm install
npm run build
```

## Export data

Validation cohort (3 subjects):

```bash
sbatch scripts/build_data.sh
```

Full cohort (all packaged subjects, union across tasks):

```bash
sbatch scripts/build_data_full.sh
```

Export brain meshes (template + native where surf exists):

```bash
sbatch scripts/export_brain_mesh.sh
```

QA after export:

```bash
conda activate ieeg
python scripts/qa_export.py public/data
```

## Serve on HPC

```bash
cd /hpc/group/coganlab/nanlinshi/insula/viewer/hga_explorer
sbatch scripts/serve.sh
```

Check job status:

```bash
squeue -u $USER
```

Read the Slurm log for the compute node name and port (default **18081**).

## SSH tunnel from laptop

After the serve job starts, set the compute node from the log and run on your laptop:

```bash
export HGA_EXPLORER_NODE=<compute-node-from-log>
bash scripts/connect_tunnel.sh
```

Open: http://localhost:18081/

## Local development

```bash
conda activate ieeg
cd /hpc/group/coganlab/nanlinshi/insula/viewer/hga_explorer
npm run dev
```

Default dev URL: http://localhost:5173/

## Notes

- Native brain meshes require `export/export_native_brain_mesh.py` and FreeSurfer `surf/lh.pial` + `surf/rh.pial` under `/cwork/ns458/ECoG_Recon/D{subject_num}/`.
- Multi-subject selection forces template brain coordinates.
- Bipolar endpoints appear after clicking a midpoint electrode.
