"""Native surface helpers copied from shi100_insula_contacts.ipynb.
Geometry, Hammers display-only filling, boundary extraction and camera are unchanged.
"""
from pathlib import Path
import numpy as np
import nibabel as nib
import pyvista as pv
from scipy.spatial import cKDTree
RECON = Path('/cwork/ns458/ECoG_Recon')
INSULA_NAMES = ('G_insular_short','G_Ins_lg_and_S_cent_ins','S_circular_insula_ant','S_circular_insula_inf','S_circular_insula_sup')
GYRI = {'AP':(92,93),'ASG':(86,87),'MSG':(88,89),'PSG':(90,91),'ALG':(94,95),'PLG':(20,21)}
AIC_IDS = {86,87,88,89,90,91,92,93}
PIC_IDS = {20,21,94,95}
SURFACE_FILL_MAX_MM = 8.0
AIC_FILL, PIC_FILL, UNMAPPED_FILL = '#EBDDB0','#D8DFE5','#EEEEEE'
def prepare_surface(subject, hemi, atlas, orig):
    pial_path = RECON / subject / 'surf' / f'{hemi}.pial'
    annot_path = RECON / subject / 'label' / f'{hemi}.aparc.a2009s.annot'
    xyz, triangles = nib.freesurfer.read_geometry(pial_path)
    annot, _, names = nib.freesurfer.read_annot(annot_path)
    region_ids = [i for i, name in enumerate(names)
                  if name.decode().replace('&', '_and_') in INSULA_NAMES]
    assert len(region_ids) == len(INSULA_NAMES), (subject, hemi, region_ids)
    selected = np.isin(annot, region_ids)
    triangles = triangles[selected[triangles].all(axis=1)]
    verts, inverse = np.unique(triangles, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    points = xyz[verts]
    assert len(faces) > 50, (subject, hemi, 'missing insula mesh')
    tkr = orig.header.get_vox2ras_tkr()
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(tkr), points)).astype(int)
    valid = ((ijk >= 0) & (ijk < np.asarray(atlas.shape))).all(axis=1)
    labels = np.zeros(len(points), dtype=int)
    labels[valid] = atlas[tuple(ijk[valid].T)]
    hemi_ids = [ids[0 if hemi == 'lh' else 1] for ids in GYRI.values()]
    direct = np.isin(labels, hemi_ids)
    gm_ijk = np.column_stack(np.where(np.isin(atlas, hemi_ids)))
    assert len(gm_ijk), (subject, hemi, 'no Hammers insula gray matter')
    gm_xyz = nib.affines.apply_affine(tkr, gm_ijk)
    distance, nearest = cKDTree(gm_xyz).query(points[~direct])
    labels[~direct] = 0
    fill_indices = np.flatnonzero(~direct)[distance <= SURFACE_FILL_MAX_MM]
    near_vox = gm_ijk[nearest[distance <= SURFACE_FILL_MAX_MM]]
    labels[fill_indices] = atlas[tuple(near_vox.T)]
    fill_distance = np.zeros(len(points))
    fill_distance[~direct] = distance
    # Face colors only convey broad AIC/PIC; all six gyrus boundaries are overlaid.
    face_ids = np.array([np.bincount(ids).argmax() for ids in labels[faces]])
    from matplotlib.colors import to_rgb
    colors = np.tile(to_rgb(UNMAPPED_FILL), (len(faces), 1))
    colors[np.isin(face_ids, list(AIC_IDS))] = to_rgb(AIC_FILL)
    colors[np.isin(face_ids, list(PIC_IDS))] = to_rgb(PIC_FILL)
    mesh = pv.PolyData(points, np.column_stack([np.full(len(faces), 3), faces]).ravel())
    mesh.cell_data['region_rgb'] = np.rint(colors * 255).astype(np.uint8)
    return dict(points=points, faces=faces, labels=labels, mesh=mesh,
                hemi_ids=hemi_ids, direct=direct, fill_distance=fill_distance,
                pial_path=pial_path, annot_path=annot_path)


def boundary_segments(surface):
    # Triangle-edge midpoints give continuous, unsmoothed label boundaries.
    points, faces, labels = (surface[k] for k in ('points', 'faces', 'labels'))
    minor, major = [], []
    for face in faces:
        if np.any(labels[face] == 0):
            continue
        crosses, groups = [], []
        for i, j in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            if labels[i] != labels[j]:
                crosses.append((points[i] + points[j]) / 2)
                groups.append((labels[i] in AIC_IDS) != (labels[j] in AIC_IDS))
        if len(crosses) == 2:
            (major if all(groups) else minor).append(crosses)
        elif len(crosses) == 3:
            center = points[face].mean(axis=0)
            for cross, major_edge in zip(crosses, groups):
                (major if major_edge else minor).append([cross, center])
    return np.asarray(minor).reshape(-1, 2, 3), np.asarray(major).reshape(-1, 2, 3)
def render_panel(subject, hemi, surface, df):
    points = surface['points']
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    plotter = pv.Plotter(off_screen=True, notebook=False, window_size=(900, 900))
    try:
        plotter.set_background('white')
        plotter.add_mesh(surface['mesh'], scalars='region_rgb', rgb=True,
                         preference='cell', show_scalar_bar=False,
                         smooth_shading=True, ambient=0.35, diffuse=0.65)
        outward = -1 if hemi == 'lh' else 1
        plotter.camera_position = [center + [outward * 180, 0, 0], center, [0, 0, 1]]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 43.0  # 86 mm field, shared across all subjects
        plotter.show(auto_close=False)
        rgb = plotter.screenshot(return_img=True)
        matrix = plotter.camera.GetCompositeProjectionTransformMatrix(1.0, -1, 1)
        matrix = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
        def project(xyz):
            xyz = np.asarray(xyz).reshape(-1, 3)
            hom = np.column_stack([xyz, np.ones(len(xyz))]) @ matrix.T
            ndc = hom[:, :3] / hom[:, 3:4]
            return np.column_stack([(ndc[:, 0] + 1) * 450, (1 - ndc[:, 1]) * 450])
        minor, major = boundary_segments(surface)
        xy = project(df[['x', 'y', 'z']].to_numpy())
        assert ((xy > 35) & (xy < 865)).all(), (subject, 'contact clipped; enlarge field')
        return dict(rgb=rgb, xy=xy, contacts=df,
                    minor=project(minor).reshape(-1, 2, 2),
                    major=project(major).reshape(-1, 2, 2),
                    subject=subject, hemi=hemi)
    finally:
        plotter.close()


