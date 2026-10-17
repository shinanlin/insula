SHI-100 recon-only bipolar anatomy

Run scripts/slurm/shi100_recon_bipolar.sh with a NEW derivatives/parcellation
output directory argument. Current final results are in:
/cwork/ns458/insula_recon_bipolar/derivatives/parcellation
This standalone adapter imports the canonical shaft-pairing and native-label
helpers from seeg-preprocessing. It neither changes that pipeline nor reads
neural signals. All computations run on Slurm.

Subjects: D148, D149, D150, D152, D153, D154, D157, D158. Automatically select
the recon with most contacts (D152B). D152B shares D0152A MAPER segmentation
only after checking orig.mgz and brainmask.mgz SHA256 identity.

Pairing: contacts sorted by numeric suffix within a shaft; use canonical
20 mm maximum spacing and 60 degree turning-angle gates. Adjacent available
contacts can have a numeric gap; these gaps are reported, not silently removed.
No task bad-channel list exists for this anatomical result. Preserve full
anatomical channel names and endpoint identities; keep the upstream EDF alias
only for audit because its 16-character limit can truncate long shaft names.

Labeling: all coordinates are native FreeSurfer tkRAS millimeters. Sample the
nearest voxel using the MRI vox2ras_tkr header, without snapping to cortex.
Use aparc2009s everywhere except its five insular parcels (short and long gyri,
three circular sulci). Within that mask, use same-hemisphere Hammers insular
labels, gated by MAPER gray matter tissue==2. If absent, retain the original
aparc label, use broad INS, and flag insula_subdivision_unresolved. Hammers
insula labels outside the aparc insula mask are audit flags, not replacements.
Thus hybrid coverage may differ from the earlier Hammers-only contact screen.

Each subject has ONE final table using the existing project naming convention:
sub-Dxxxx/bipolar/sub-Dxxxx_aparc2009sHammers.csv
The suffix identifies the hybrid atlas, not either unmixed source atlas.
Its 14 columns are:
subject,name,contact_1,contact_2,contact_1_label,midpoint_label,contact_2_label,
roi,mix,hemi,x,y,z,spacing_mm
The x/y/z coordinates describe the bipolar midpoint.
Other fields are used only in temporary QC tables during validation and are
deleted after publication. Only the eight final CSV files are retained.
Coordinates are native only; there are no invented MNI/CVS fields.

ROI consensus reuses the canonical spatial order endpoint1/midpoint/endpoint2
and ignores Unknown and WM. aic_consensus means ROI=AIC with mix=False; it
does NOT mean all three positions are AIC. aic_all_three is the stricter flag.
aic_any_point includes mixed pairs; aic_midpoint describes the midpoint only.
Fine Hammers subdivisions remain in endpoint/midpoint labels; ROI uses AIC/PIC.

QC is generated in an isolated staging run under insula_recon_screen and
includes independent voxel sampling, pair geometry and coverage checks.
finalize.py verifies final file hashes, then deletes that staging run including
QC, summaries, provenance and completion markers. Slurm logs record success.
Failed validation retains staging for diagnosis. These are anatomical candidate
pairs, not signal-qualified task channels. No signal preprocessing or brainplot runs.
