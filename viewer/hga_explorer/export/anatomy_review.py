"""Native LEPTO contacts, T1 review slices, and manual anatomy labels.

Geometry follows the approved MRI plane in ``src/supplementary/operculum_anatomy.py``
without calling ``load_subject`` (that helper writes an audit JSON report).
Coordinates stay in the original native tkRAS millimeters. CT is not loaded.
"""

from __future__ import annotations

import csv
import os
import re
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

os.environ.setdefault(
    'MPLCONFIGDIR',
    str(Path(__file__).resolve().parents[3] / 'tmp' / 'matplotlib'),
)
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from nibabel.affines import apply_affine
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[3]
RECON_ROOT = Path('/cwork/ns458/ECoG_Recon')
LUT_PATH = Path('/hpc/group/coganlab/nanlinshi/seeg-preprocessing/common/FreeSurferColorLUT.txt')
LABELS_PATH = ROOT / 'results' / 'supplementary' / 'operculum' / 'manual_anatomy_labels.csv'

INSULA_NAMES = {
    'G_insular_short',
    'G_Ins_lg_and_S_cent_ins',
    'S_circular_insula_ant',
    'S_circular_insula_inf',
    'S_circular_insula_sup',
}
ATLAS_COLORS = {
    'Insula': '#6A5ACD',
    'Frontal operculum': '#2369BD',
    'Other': '#777777',
}
OVERLAY_COLORS = {'AIC': '#6A5ACD', 'FO': '#2369BD'}
MANUAL_LABELS = ('Operculum', 'Anterior Insula')
MANUAL_COLORS = {
    'Operculum': '#EA580C',
    'Anterior Insula': '#DB2777',
}
LABEL_COLUMNS = (
    'subject', 'contact', 'shaft', 'x', 'y', 'z', 'native_aparc', 'label', 'timestamp',
)
SUBJECT_RE = re.compile(r'^D\d{4}$')
CONTACT_RE = re.compile(r'^[A-Za-z0-9_]+$')

_LOAD_LOCK = threading.Lock()
_LABEL_LOCK = threading.Lock()
_RENDER_LOCK = threading.Lock()
_SUBJECTS: OrderedDict[str, dict] = OrderedDict()
_LUT = None


def recon_id(subject: str) -> str:
    """Map a frozen subject id onto its reconstruction folder.

    D0057 matches the 2022 D57_2 LEPTO reconstruction, not D57.
    """
    validate_subject(subject)
    if subject == 'D0057':
        return 'D57_2'
    return 'D' + str(int(subject[1:]))


def validate_subject(subject: str) -> str:
    if not subject or not SUBJECT_RE.fullmatch(subject):
        raise ValueError('Subject must look like D00NN')
    return subject


def validate_contact(contact: str) -> str:
    if not contact or not CONTACT_RE.fullmatch(contact):
        raise ValueError('Invalid contact name')
    return contact


def normalize_manual_label(value) -> str | None:
    if value is None or value == '' or value == 'clear':
        return None
    if value not in MANUAL_LABELS:
        raise ValueError('Label must be Operculum, Anterior Insula, or empty to clear')
    return value


def _read_lut() -> dict[int, str]:
    global _LUT
    if _LUT is not None:
        return _LUT
    if not LUT_PATH.is_file():
        raise FileNotFoundError(f'FreeSurfer color lookup is missing: {LUT_PATH}')
    lut = {}
    for line in LUT_PATH.read_text().splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[0].isdigit():
            lut[int(parts[0])] = parts[1]
    _LUT = lut
    return lut


def _native_region(label: str) -> str:
    if any(label.endswith('_' + name) for name in INSULA_NAMES):
        return 'Insula'
    if label.endswith('_G_front_inf-Opercular'):
        return 'Frontal operculum'
    return 'Other'


def _shaft_of(name: str) -> tuple[str, int]:
    match = re.fullmatch(r'(.+?)(\d+)', name)
    if not match:
        return name, 0
    return match.group(1), int(match.group(2))


def load_subject_geometry(subject: str) -> dict:
    """Load MRI, aparc, and LEPTO contacts. Does not write audit files or load CT."""
    validate_subject(subject)
    recon = recon_id(subject)
    folder = RECON_ROOT / recon
    mri_path = folder / 'mri' / 'orig.mgz'
    aparc_path = folder / 'mri' / 'aparc.a2009s+aseg.mgz'
    name_path = folder / 'elec_recon' / f'{recon}.electrodeNames'
    lepto_path = folder / 'elec_recon' / f'{recon}.LEPTO'
    missing = [path for path in (mri_path, aparc_path, name_path, lepto_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f'{subject} reconstruction {recon} is missing {", ".join(str(path) for path in missing)}'
        )

    native = nib.load(str(mri_path))
    inverse = np.linalg.inv(native.header.get_vox2ras_tkr())
    mri = np.asarray(native.dataobj, dtype=np.float32)
    atlas = nib.load(str(aparc_path))
    np.testing.assert_allclose(atlas.affine, native.affine, atol=1e-4)
    aparc = np.asarray(atlas.dataobj)
    lut = _read_lut()
    positive = mri[mri > 0]
    limits = np.percentile(positive, [1, 99.5]) if positive.size else np.array([0.0, 1.0])

    name_lines = [line for line in name_path.read_text().splitlines()[2:] if line.strip()]
    lepto_lines = [line for line in lepto_path.read_text().splitlines()[2:] if line.strip()]
    if len(name_lines) != len(lepto_lines):
        raise ValueError(
            f'{recon} electrodeNames ({len(name_lines)}) and LEPTO ({len(lepto_lines)}) disagree'
        )

    contacts = []
    for name_line, lepto_line in zip(name_lines, lepto_lines):
        name_parts = name_line.split()
        lepto_parts = lepto_line.split()
        if len(lepto_parts) < 3:
            raise ValueError(f'{recon}.LEPTO has a short coordinate row: {lepto_line}')
        contact = name_parts[0]
        shaft, number = _shaft_of(contact)
        xyz = [float(lepto_parts[0]), float(lepto_parts[1]), float(lepto_parts[2])]
        hem = name_parts[2] if len(name_parts) > 2 and name_parts[2] in {'L', 'R'} else None
        if hem is None:
            hem = 'R' if xyz[0] >= 0 else 'L'
        ijk = np.rint(apply_affine(inverse, xyz)).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= np.array(aparc.shape)):
            native_aparc = 'Unknown'
        else:
            code = int(aparc[tuple(ijk)])
            native_aparc = lut.get(code, str(code))
        contacts.append({
            'contact': contact,
            'shaft': shaft,
            'contact_number': number,
            'x': xyz[0],
            'y': xyz[1],
            'z': xyz[2],
            'hemisphere': hem,
            'native_aparc': native_aparc,
            'native_region': _native_region(native_aparc),
        })
    if len({row['contact'] for row in contacts}) != len(contacts):
        raise ValueError(f'{recon} electrode names are not unique')

    note = None
    if subject == 'D0057':
        note = (
            'Contacts and MRI use reconstruction D57_2. '
            'The exported native surface was built from D57, so the 3D mesh can disagree with these coordinates.'
        )
    return {
        'subject': subject,
        'reconstruction': recon,
        'folder': str(folder),
        'inverse': inverse,
        'mri': mri,
        'aparc': aparc,
        'lut': lut,
        'mri_limits': limits,
        'contacts': contacts,
        'mesh_note': note,
        'coordinate_space': 'native tkRAS millimeters',
    }


def get_subject(subject: str) -> dict:
    validate_subject(subject)
    with _LOAD_LOCK:
        cached = _SUBJECTS.get(subject)
        if cached is not None:
            _SUBJECTS.move_to_end(subject)
            return cached
        loaded = load_subject_geometry(subject)
        _SUBJECTS[subject] = loaded
        while len(_SUBJECTS) > 3:
            _SUBJECTS.popitem(last=False)
        return loaded


def contacts_payload(subject: str) -> dict:
    record = get_subject(subject)
    return {
        'subject': record['subject'],
        'reconstruction': record['reconstruction'],
        'coordinate_space': record['coordinate_space'],
        'mesh_note': record['mesh_note'],
        'contacts': record['contacts'],
    }


def _contact_row(record: dict, contact: str) -> dict:
    validate_contact(contact)
    for row in record['contacts']:
        if row['contact'] == contact:
            return row
    raise LookupError(f'{record["subject"]} has no LEPTO contact {contact}')


def _read_label_rows() -> list[dict]:
    if not LABELS_PATH.is_file():
        return []
    with LABELS_PATH.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _write_label_rows(rows: list[dict]) -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = LABELS_PATH.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, '') for column in LABEL_COLUMNS})
    temporary.replace(LABELS_PATH)


def read_labels(subject: str | None = None) -> list[dict]:
    if subject:
        validate_subject(subject)
    with _LABEL_LOCK:
        rows = _read_label_rows()
    if subject:
        rows = [row for row in rows if row.get('subject') == subject]
    parsed = []
    for row in rows:
        item = {column: row.get(column, '') for column in LABEL_COLUMNS}
        for axis in ('x', 'y', 'z'):
            try:
                item[axis] = float(item[axis])
            except (TypeError, ValueError):
                item[axis] = None
        parsed.append(item)
    return parsed


def write_label(subject: str, contact: str, label) -> dict:
    """Insert, replace, or remove one manual decision. Atlas files are not touched."""
    validate_subject(subject)
    validate_contact(contact)
    decision = normalize_manual_label(label)
    record = get_subject(subject)
    source = _contact_row(record, contact)
    with _LABEL_LOCK:
        rows = [row for row in _read_label_rows() if not (
            row.get('subject') == subject and row.get('contact') == contact
        )]
        stored = None
        if decision is not None:
            stored = {
                'subject': subject,
                'contact': contact,
                'shaft': source['shaft'],
                'x': f'{source["x"]:.6f}',
                'y': f'{source["y"]:.6f}',
                'z': f'{source["z"]:.6f}',
                'native_aparc': source['native_aparc'],
                'label': decision,
                'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            }
            rows.append(stored)
        _write_label_rows(rows)
    if stored is None:
        return {'ok': True, 'cleared': True, 'label': None}
    response = dict(stored)
    for axis in ('x', 'y', 'z'):
        response[axis] = float(response[axis])
    return {'ok': True, 'cleared': False, 'label': response}


def _plane_axes(shaft_axis: np.ndarray):
    """Same upright contact plane as ``draw_mri`` (plane 1)."""
    axis1 = np.asarray(shaft_axis, dtype=float)
    norm = np.linalg.norm(axis1)
    if norm < 1e-6:
        axis1 = np.array([1.0, 0.0, 0.0])
    else:
        axis1 = axis1 / norm
    vertical = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(vertical, axis1)) > 0.9:
        vertical = np.array([0.0, 1.0, 0.0])
    axis2 = vertical - np.dot(vertical, axis1) * axis1
    axis2 = axis2 / np.linalg.norm(axis2)
    normal = np.cross(axis1, axis2)
    normal = normal / np.linalg.norm(normal)
    axial_like = int(np.argmax(np.abs(normal))) == 2
    up = np.array([0.0, 1.0, 0.0]) if axial_like else np.array([0.0, 0.0, 1.0])
    upright = up - np.dot(up, normal) * normal
    axis2 = upright / np.linalg.norm(upright)
    axis1 = np.cross(axis2, normal)
    if (abs(axis1[0]) >= abs(axis1[1]) and axis1[0] < 0) or (
        abs(axis1[0]) < abs(axis1[1]) and axis1[1] > 0
    ):
        axis1 = -axis1
    return axis1, axis2, normal, axial_like


def _local_shaft_axis(shaft_rows: list[dict], contact: str) -> np.ndarray:
    ordered = sorted(shaft_rows, key=lambda row: row['contact_number'])
    if len(ordered) < 2:
        return np.array([1.0, 0.0, 0.0])
    index = next(i for i, row in enumerate(ordered) if row['contact'] == contact)
    if index == 0:
        start, stop = ordered[0], ordered[1]
    elif index == len(ordered) - 1:
        start, stop = ordered[-2], ordered[-1]
    else:
        start, stop = ordered[index - 1], ordered[index + 1]
    direction = np.array([stop[axis] - start[axis] for axis in ('x', 'y', 'z')], dtype=float)
    if np.linalg.norm(direction) < 1e-6:
        direction = np.array(
            [ordered[-1][axis] - ordered[0][axis] for axis in ('x', 'y', 'z')],
            dtype=float,
        )
    return direction


def _sample_plane(volume, inverse, center, axis1, axis2, half=90.0, spacing=0.5, order=1):
    scale = np.arange(-half, half + spacing * 0.5, spacing)
    xx, yy = np.meshgrid(scale, scale, indexing='xy')
    world = center + xx[..., None] * axis1 + yy[..., None] * axis2
    ijk = apply_affine(inverse, world).reshape(-1, 3)
    plane = map_coordinates(volume, ijk.T, order=order, mode='constant', cval=0).reshape(xx.shape)
    return scale, plane


def _region_codes(lut: dict[int, str], hemi: str, kind: str) -> list[int]:
    prefix = f'ctx_{hemi}_'
    codes = []
    for code, name in lut.items():
        if not str(name).startswith(prefix):
            continue
        if kind == 'AIC' and any(token in name for token in INSULA_NAMES):
            codes.append(code)
        elif kind == 'FO' and 'G_front_inf-Opercular' in name:
            codes.append(code)
    return codes


def _marker_color(row: dict, manual: str | None) -> str:
    if manual in MANUAL_COLORS:
        return MANUAL_COLORS[manual]
    return ATLAS_COLORS.get(row['native_region'], '#777777')


def render_slice_png(subject: str, contact: str) -> bytes:
    validate_contact(contact)
    record = get_subject(subject)
    selected = _contact_row(record, contact)
    labels = {
        row['contact']: row['label']
        for row in read_labels(subject)
        if row.get('label')
    }
    shaft_rows = [row for row in record['contacts'] if row['shaft'] == selected['shaft']]
    axis1, axis2, normal, axial_like = _plane_axes(_local_shaft_axis(shaft_rows, contact))
    xyz = np.array([selected['x'], selected['y'], selected['z']], dtype=float)
    center = xyz - np.dot(xyz, axis1) * axis1 - np.dot(xyz, axis2) * axis2
    hemi = 'lh' if selected['hemisphere'] == 'L' else 'rh'

    with _RENDER_LOCK:
        scale, t1 = _sample_plane(record['mri'], record['inverse'], center, axis1, axis2, order=1)
        _, atlas = _sample_plane(record['aparc'], record['inverse'], center, axis1, axis2, order=0)
        atlas = np.rint(atlas).astype(int)
        extent = [float(scale[0]), float(scale[-1]), float(scale[0]), float(scale[-1])]
        figure = plt.figure(figsize=(5.6, 5.6), dpi=140)
        try:
            axis = figure.add_axes([0.01, 0.01, 0.98, 0.90])
            axis.imshow(
                t1,
                cmap='gray',
                origin='lower',
                extent=extent,
                vmin=float(record['mri_limits'][0]),
                vmax=float(record['mri_limits'][1]),
                interpolation='none',
            )
            for kind, title, offset in (
                ('AIC', 'Insula', (-36, 16)),
                ('FO', 'Frontal operculum', (16, -24)),
            ):
                codes = _region_codes(record['lut'], hemi, kind)
                mask = np.isin(atlas, codes) if codes else np.zeros(atlas.shape, dtype=bool)
                if not mask.any() or mask.all():
                    continue
                axis.imshow(
                    np.ma.masked_where(~mask, np.ones(mask.shape)),
                    cmap=ListedColormap([OVERLAY_COLORS[kind]]),
                    origin='lower',
                    extent=extent,
                    alpha=0.18,
                    interpolation='none',
                    vmin=0,
                    vmax=1,
                )
                axis.contour(
                    scale, scale, mask.astype(float),
                    levels=[0.5], colors=[OVERLAY_COLORS[kind]], linewidths=0.7, alpha=0.9,
                )
                yy, xx = np.nonzero(mask)
                anchor = (float(np.median(scale[xx])), float(np.median(scale[yy])))
                axis.annotate(
                    title,
                    anchor,
                    xytext=offset,
                    textcoords='offset points',
                    fontsize=8,
                    color=OVERLAY_COLORS[kind],
                    arrowprops={'arrowstyle': '-', 'color': OVERLAY_COLORS[kind], 'lw': 0.6},
                    bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none', 'pad': 0.8},
                )

            path = np.array(
                [[row['x'], row['y'], row['z']] for row in sorted(shaft_rows, key=lambda row: row['contact_number'])],
                dtype=float,
            )
            depth = (path - center) @ normal
            xy = np.column_stack([(path - center) @ axis1, (path - center) @ axis2])
            for index in range(len(path) - 1):
                d0, d1 = depth[index:index + 2]
                lo, hi = 0.0, 1.0
                if abs(d1 - d0) < 1e-10:
                    if abs(d0) > 0.5:
                        continue
                else:
                    bounds = sorted([(-0.5 - d0) / (d1 - d0), (0.5 - d0) / (d1 - d0)])
                    lo = max(lo, bounds[0])
                    hi = min(hi, bounds[1])
                    if lo > hi:
                        continue
                visible = xy[index] + np.array([lo, hi])[:, None] * (xy[index + 1] - xy[index])
                axis.plot(visible[:, 0], visible[:, 1], color='#A0A0A0', lw=1.4, alpha=0.95, zorder=5)
            ordered = sorted(shaft_rows, key=lambda row: row['contact_number'])
            for index, row in enumerate(ordered):
                if abs(depth[index]) > 0.5:
                    continue
                manual = labels.get(row['contact'])
                clicked = row['contact'] == contact
                axis.scatter(
                    xy[index, 0], xy[index, 1],
                    s=46 if clicked else (28 if manual else 16),
                    c=_marker_color(row, manual),
                    edgecolors='white',
                    linewidths=0.8 if clicked else 0.25,
                    zorder=7 if clicked else 6,
                )
                if clicked:
                    axis.scatter(
                        xy[index, 0], xy[index, 1],
                        s=90, facecolors='none', edgecolors='white', linewidths=1.1, zorder=8,
                    )
                    axis.annotate(
                        row['contact'],
                        (xy[index, 0], xy[index, 1]),
                        xytext=(6, 8),
                        textcoords='offset points',
                        color='white',
                        fontsize=8,
                        bbox={'facecolor': 'black', 'alpha': 0.55, 'edgecolor': 'none', 'pad': 0.6},
                    )
            axis.plot([-78, -58], [-76, -76], c='white', lw=1.4)
            axis.text(-68, -84, '20 mm', color='white', ha='center', fontsize=8)
            for image_axis, positive, negative in (
                (axis1, (0.98, 0.5), (0.02, 0.5)),
                (axis2, (0.5, 0.98), (0.5, 0.02)),
            ):
                dominant = int(np.argmax(np.abs(image_axis)))
                pos = ['R', 'A', 'S'][dominant]
                neg = ['L', 'P', 'I'][dominant]
                if image_axis[dominant] < 0:
                    pos, neg = neg, pos
                for name, where in ((pos, positive), (neg, negative)):
                    axis.text(
                        *where, name, transform=axis.transAxes, color='white', fontsize=8,
                        ha='center', va='center',
                        bbox={'facecolor': 'black', 'alpha': 0.5, 'edgecolor': 'none', 'pad': 0.4},
                    )
            orientation = 'anterior up' if axial_like else 'superior up'
            figure.suptitle(
                f'{contact} · native T1 ({orientation})',
                fontsize=11,
                y=0.975,
            )
            axis.set_xlim(-90, 90)
            axis.set_ylim(-90, 90)
            axis.set_aspect('equal')
            axis.axis('off')
            buffer = BytesIO()
            figure.savefig(buffer, format='png')
            return buffer.getvalue()
        finally:
            plt.close(figure)
