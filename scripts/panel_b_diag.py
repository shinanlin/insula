"""Lightweight Panel B alignment diagnostic (no Brain rendering)."""
import re
import numpy as np
import pandas as pd
import nibabel as nib
import mne
from mne_bids import BIDSPath
from scipy.spatial import cKDTree

recon_dir = "/cwork/ns458/ECoG_Recon/"
FS_SUBJECT = "cvs_avg35_inMNI152"
HAMMERS_MNI152_PATH = (
    "/cwork/ns458/atlases/Hammersmith_n30r95/derivatives/faillenot_spm_mni152"
    "/probability_maps/Hammers-n30r95-maxprob-full-MNI152.nii.gz"
)
HAMMERS_INSULA_INDEX = {
    20: "posterior long gyrus",
    21: "posterior long gyrus",
    86: "anterior short gyrus",
    87: "anterior short gyrus",
    88: "middle short gyrus",
    89: "middle short gyrus",
    90: "posterior short gyrus",
    91: "posterior short gyrus",
    92: "anterior pole",
    93: "anterior pole",
    94: "anterior long gyrus",
    95: "anterior long gyrus",
}
HAMMERS_INSULA_IDS = frozenset(HAMMERS_INSULA_INDEX)


def insula_mask(df, atlas):
    mask = df["roi"].isin({"AIC", "PIC"})
    if "mix" in df.columns:
        mask = mask & ~df["mix"].fillna(False).astype(bool)
    return mask


def _sample_hammers_labels(surface_coords, atlas_data, atlas_inv):
    ijk = nib.affines.apply_affine(atlas_inv, surface_coords).round().astype(int)
    out = np.zeros(len(surface_coords), dtype=int)
    shape = atlas_data.shape
    valid = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < shape[0])
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < shape[1])
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < shape[2])
    )
    if valid.any():
        v = ijk[valid]
        out[valid] = atlas_data[v[:, 0], v[:, 1], v[:, 2]].astype(int)
    return out


def build_insula_vertex_sets(pial_by_hemi, atlas_data, atlas_inv):
    out = {}
    for hemi, pial in pial_by_hemi.items():
        vtx_ids = _sample_hammers_labels(pial, atlas_data, atlas_inv)
        insula_vtx = set()
        for atlas_idx in HAMMERS_INSULA_IDS:
            expected_hemi = "lh" if atlas_idx % 2 == 0 else "rh"
            if expected_hemi != hemi:
                continue
            insula_vtx |= set(np.where(vtx_ids == atlas_idx)[0])
        out[hemi] = insula_vtx
    return out


def hammers_insula_gyrus(label):
    if pd.isna(label):
        return pd.NA
    text = str(label).strip()
    match = re.search(r"insula\s+(.+?)\s+[LR]$", text, flags=re.I)
    return match.group(1).strip().lower() if match else pd.NA


def main():
    task = ["PhonemeSequence", "LexicalDelay", "LexicalNoDelay", "PictureNaming"]
    ref = "bipolar"
    atlas = "hammers"
    coord_paths = []
    for t in task:
        coord_paths.extend(
            BIDSPath(
                root=f"results/{t}({ref})({atlas})",
                datatype="HGA",
                suffix="coord",
                check=False,
            ).match()
        )
    coords = pd.concat([pd.read_csv(p) for p in coord_paths], ignore_index=True)
    coords = coords.dropna(subset=["x", "y", "z"])
    coords_unique = coords.drop_duplicates(subset=["channel"]).copy()
    coords_unique["is_insula"] = insula_mask(coords_unique, atlas)
    insula = coords_unique.loc[coords_unique["is_insula"]].copy()

    print("=== 1. COORD LOAD ===")
    print(f"Unique electrodes: {len(coords_unique)}, insula: {len(insula)}")

    img = nib.load(HAMMERS_MNI152_PATH)
    atlas_data = np.asarray(img.dataobj).squeeze()
    atlas_inv = np.linalg.inv(img.affine)

    print("\n=== 2. HAMMERS VOLUME / AFFINE ===")
    print("Volume shape:", atlas_data.shape)
    print("Affine:\n", img.affine)
    print("Voxel sizes (from affine columns):", np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0)))

    lh_pial, _ = mne.read_surface(f"{recon_dir}/{FS_SUBJECT}/surf/lh.pial")
    rh_pial, _ = mne.read_surface(f"{recon_dir}/{FS_SUBJECT}/surf/rh.pial")
    pial_by_hemi = {"lh": lh_pial, "rh": rh_pial}
    trees = {h: cKDTree(p) for h, p in pial_by_hemi.items()}
    insula_vtx = build_insula_vertex_sets(pial_by_hemi, atlas_data, atlas_inv)

    print("\n=== 3. HAMMERS INSULA SURFACE VERTICES ===")
    print(f"LH: {len(insula_vtx['lh'])}, RH: {len(insula_vtx['rh'])}")

    cord = insula[["x", "y", "z"]].values
    mask = {"lh": cord[:, 0] < 0, "rh": cord[:, 0] > 0}

    print("\n=== 4. PROJECTION ONTO FULL PIAL vs HAMMERS INSULA SURFACE ===")
    total_on = 0
    for hemi in ("lh", "rh"):
        sub = cord[mask[hemi]]
        if len(sub) == 0:
            continue
        _, idx = trees[hemi].query(sub)
        on = np.array([i in insula_vtx[hemi] for i in idx])
        total_on += on.sum()
        proj = pial_by_hemi[hemi][idx]
        snap = np.linalg.norm(sub - proj, axis=1)
        vol = _sample_hammers_labels(sub, atlas_data, atlas_inv)
        print(
            f"{hemi.upper()} n={len(sub)}: on_insula_surf={on.sum()} ({100*on.mean():.1f}%), "
            f"snap_dist median={np.median(snap):.1f}mm max={snap.max():.1f}mm, "
            f"vol_six_gyrus={np.isin(vol, list(HAMMERS_INSULA_IDS)).sum()}"
        )

    print(
        f"\nOverall on Hammers insula surface after full-pial projection: "
        f"{total_on}/{len(insula)} ({100*total_on/len(insula):.1f}%)"
    )

    print("\n=== 5. VOLUME SAMPLING AT ELECTRODE XYZ ===")
    vol_at_elec = _sample_hammers_labels(cord, atlas_data, atlas_inv)
    for lab, cnt in sorted(zip(*np.unique(vol_at_elec, return_counts=True)), key=lambda x: -x[1])[:12]:
        name = HAMMERS_INSULA_INDEX.get(int(lab), f"non-insula-{lab}")
        print(f"  label {int(lab):3d} ({name}): {cnt}")
    in_six = np.isin(vol_at_elec, list(HAMMERS_INSULA_IDS)).sum()
    print(f"In six-gyrus union at raw coords: {in_six}/{len(insula)} ({100*in_six/len(insula):.1f}%)")

    print("\n=== 6. ROI / MIX BREAKDOWN ===")
    print(insula["roi"].value_counts().to_string())
    if "mix" in insula.columns:
        print("mix=True:", insula["mix"].fillna(False).astype(bool).sum())

    print("\n=== 7. CENTROID OFFSET ===")
    for hemi in ("lh", "rh"):
        vtx = list(insula_vtx[hemi])
        surf_c = pial_by_hemi[hemi][vtx].mean(axis=0)
        elec_c = cord[mask[hemi]].mean(axis=0)
        print(f"{hemi.upper()} offset mm: {np.linalg.norm(elec_c - surf_c):.1f}")

    print("\n=== 8. DISTANCE TO NEAREST INSULA SURFACE VERTEX (raw xyz) ===")
    dists = []
    for hemi in ("lh", "rh"):
        ins_tree = cKDTree(pial_by_hemi[hemi][list(insula_vtx[hemi])])
        d, _ = ins_tree.query(cord[mask[hemi]])
        dists.extend(d.tolist())
    dists = np.array(dists)
    print(
        f"mean={dists.mean():.1f} median={np.median(dists):.1f} max={dists.max():.1f} | "
        f">5mm={(dists>5).sum()} >10mm={(dists>10).sum()} >20mm={(dists>20).sum()}"
    )

    if {"x_native", "y_native", "z_native"}.issubset(insula.columns):
        nat = insula[["x_native", "y_native", "z_native"]].values
        nat_on = 0
        for hemi in ("lh", "rh"):
            sub = nat[mask[hemi]]
            _, idx = trees[hemi].query(sub)
            nat_on += sum(i in insula_vtx[hemi] for i in idx)
        print("\n=== 9. NATIVE vs TEMPLATE ON INSULA SURF ===")
        print(f"template: {total_on}/{len(insula)}, native: {nat_on}/{len(insula)}")

    for hemi, pial in pial_by_hemi.items():
        vtx_ids = _sample_hammers_labels(pial, atlas_data, atlas_inv)
        frac = np.isin(vtx_ids, list(HAMMERS_INSULA_IDS)).mean()
        print(f"Pial {hemi}: {100*frac:.2f}% vertices are Hammers insula gyri")

    insula["hammers_gyrus"] = insula["label"].map(hammers_insula_gyrus)
    has_gyrus = insula["hammers_gyrus"].notna()
    print("\n=== 10. B2 GYRUS LABEL FILTER ===")
    print(f"parseable gyrus: {has_gyrus.sum()}, missing: {(~has_gyrus).sum()}")
    print("missing gyrus by roi:\n", insula.loc[~has_gyrus, "roi"].value_counts().to_string())

    sub = insula.loc[has_gyrus]
    cord2 = sub[["x", "y", "z"]].values
    on2 = 0
    for hemi in ("lh", "rh"):
        m = cord2[:, 0] < 0 if hemi == "lh" else cord2[:, 0] > 0
        subc = cord2[m]
        _, idx = trees[hemi].query(subc)
        on2 += sum(i in insula_vtx[hemi] for i in idx)
    print(f"B2 subset on insula surface: {on2}/{len(sub)} ({100*on2/len(sub):.1f}%)")

    print("\n=== 11. AIC/PIC vs VOLUME (per roi) ===")
    for roi in ("AIC", "PIC"):
        rsub = insula[insula.roi == roi]
        v = _sample_hammers_labels(rsub[["x", "y", "z"]].values, atlas_data, atlas_inv)
        in_g = np.isin(v, list(HAMMERS_INSULA_IDS)).sum()
        print(f"{roi}: n={len(rsub)}, in six-gyrus vol={in_g} ({100*in_g/len(rsub):.1f}%)")

    print("\n=== 12. VIEW GEOMETRY (insula center vs electrode cloud) ===")
    for hemi in ("lh", "rh"):
        vtx = list(insula_vtx[hemi])
        center = pial_by_hemi[hemi][vtx].mean(axis=0)
        elec = cord[mask[hemi]]
        spread = np.linalg.norm(elec - center, axis=1)
        print(
            f"{hemi.upper()} focal spread from insula center: "
            f"median={np.median(spread):.1f}mm max={spread.max():.1f}mm"
        )


def deep_dive():
    paths = []
    for t in ["PhonemeSequence", "LexicalDelay", "LexicalNoDelay", "PictureNaming"]:
        paths += BIDSPath(
            root=f"results/{t}(bipolar)(hammers)",
            datatype="HGA",
            suffix="coord",
            check=False,
        ).match()
    df = pd.concat([pd.read_csv(p) for p in paths]).dropna(subset=["x", "y", "z"]).drop_duplicates("channel")
    df = df[insula_mask(df, "hammers")].copy()

    img = nib.load(HAMMERS_MNI152_PATH)
    data = np.asarray(img.dataobj).squeeze()
    inv = np.linalg.inv(img.affine)
    lh, _ = mne.read_surface(f"{recon_dir}/{FS_SUBJECT}/surf/lh.pial")
    rh, _ = mne.read_surface(f"{recon_dir}/{FS_SUBJECT}/surf/rh.pial")

    def insula_vtx(pial, hemi):
        ids = _sample_hammers_labels(pial, data, inv)
        s = set()
        for lab in HAMMERS_INSULA_IDS:
            if ("lh" if lab % 2 == 0 else "rh") != hemi:
                continue
            s |= set(np.where(ids == lab)[0])
        return s

    lv, rv = insula_vtx(lh, "lh"), insula_vtx(rh, "rh")
    lt, rt = cKDTree(lh), cKDTree(rh)
    lit, rit = cKDTree(lh[list(lv)]), cKDTree(rh[list(rv)])

    cord = df[["x", "y", "z"]].values
    rows = []
    for hemi, mask, tree, pial, iv, itree in [
        ("lh", cord[:, 0] < 0, lt, lh, lv, lit),
        ("rh", cord[:, 0] > 0, rt, rh, rv, rit),
    ]:
        sub = cord[mask]
        d_full, idx = tree.query(sub)
        proj = pial[idx]
        on_ins = [i in iv for i in idx]
        d_ins, _ = itree.query(sub)
        vol_proj = _sample_hammers_labels(proj, data, inv)
        for i in range(len(sub)):
            rows.append(
                dict(
                    hemi=hemi,
                    on_insula=on_ins[i],
                    d_full=d_full[i],
                    d_insula=d_ins[i],
                    vol_proj=int(vol_proj[i]),
                )
            )

    r = pd.DataFrame(rows)
    print("\n=== DEEP: SNAP vs INSULA DISTANCE ===")
    print(r.groupby("on_insula")[["d_full", "d_insula"]].agg(["mean", "median", "count"]))

    print("\n=== DEEP: PROJECTED VERTEX LABELS (off-insula) ===")
    print(r.loc[~r.on_insula, "vol_proj"].value_counts().head(12))

    print("\n=== DEEP: ELECTRODE LABEL STRINGS (off-insula) ===")
    off_idx = r.index[~r.on_insula]
    print(df.iloc[off_idx]["label"].value_counts().head(15))

    print("\n=== DEEP: OFF-INSULA BY ROI ===")
    df2 = df.copy()
    df2["on_insula"] = r.on_insula
    print(df2.groupby(["roi", "on_insula"]).size().unstack(fill_value=0))

    nat = df[["x_native", "y_native", "z_native"]].values
    on_t, on_n = [], []
    for hemi, mask, tree, iv in [("lh", cord[:, 0] < 0, lt, lv), ("rh", cord[:, 0] > 0, rt, rv)]:
        for arr, bucket in [(cord, on_t), (nat, on_n)]:
            sub = arr[mask]
            _, idx = tree.query(sub)
            bucket.extend([i in iv for i in idx])
    print("\n=== DEEP: TEMPLATE vs NATIVE ===")
    print(f"template on insula surf: {sum(on_t)}/{len(cord)}")
    print(f"native on insula surf: {sum(on_n)}/{len(cord)}")
    df2["t_on"] = on_t
    df2["n_on"] = on_n
    improved = df2[~df2.t_on & df2.n_on]
    print(f"native rescues {len(improved)} electrodes")
    print(improved["label"].value_counts().head(8))


if __name__ == "__main__":
    main()
    deep_dive()
