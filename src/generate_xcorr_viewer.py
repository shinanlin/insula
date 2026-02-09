"""Generate an interactive 3D HTML viewer for Insula×IFG cross-correlation.

Usage:
    python src/generate_xcorr_viewer.py \
        --bids_root /cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/ \
        --subject D0094 --phase Response --desc Repeat \
        --output viz/3d_xcorr/D0094_Response_Repeat.html
"""

import argparse
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from scipy.spatial import cKDTree

try:
    import pyvista as pv
except ImportError:
    pv = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RECON_DIR = "/cwork/ns458/ECoG_Recon/"

INSULA_PATTERNS = [
    "G_insular_short",
    "G_Ins_lg_and_S_cent_ins",
    "S_circular_insula_ant",
    "S_circular_insula_inf",
    "S_circular_insula_sup",
]
IFG_PATTERNS = [
    "G_front_inf-Opercular",
    "G_front_inf-Orbital",
    "G_front_inf-Triangul",
]

COLORS = {
    "insula": [212, 175, 55],   # gold
    "ifg": [35, 105, 189],      # blue
    "default": [220, 220, 220],  # light gray
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def bids_to_recon_id(bids_subject: str) -> str:
    """D0094 -> D94, D0106 -> D106."""
    return "D" + str(int(bids_subject[1:]))


def get_shank_prefix(ch_name: str) -> str:
    """Strip trailing digit-digit pattern: 'D0040_L1IF2-3' -> 'D0040_L1IF'."""
    return re.sub(r"\d+-\d+$", "", ch_name)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_parcellation(bids_root: str, subject: str) -> pd.DataFrame:
    parc_path = BIDSPath(
        root=os.path.join(bids_root, "derivatives", "parcellation"),
        subject=subject,
        suffix="aparc2009s",
        datatype="bipolar",
        processing="3mm",
        extension=".csv",
        check=False,
    )
    matches = parc_path.match()
    if not matches:
        raise FileNotFoundError(f"No parcellation CSV for {subject} in {bids_root}")
    df = pd.read_csv(matches[0])
    df.rename(columns={"name": "channel"}, inplace=True)
    return df


def load_epochs(bids_root: str, subject: str, phase: str, desc: str,
                band: str = "highgamma") -> mne.Epochs:
    raw_pt = BIDSPath(
        root=os.path.join(bids_root, "derivatives", "epoch(bipolar)"),
        datatype="epoch(band)(zscore)",
        subject=subject,
        suffix=band,
        processing=phase,
        extension=".h5",
        check=False,
    )
    matches = raw_pt.match()
    # Filter by description
    matches = [m for m in matches if m.description == desc]
    if not matches:
        raise FileNotFoundError(
            f"No epoch file for {subject}, phase={phase}, desc={desc} in {bids_root}"
        )
    return mne.read_epochs(matches[0], preload=True, verbose="error")


def classify_channels(parc: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (insula_chns, ifg_chns) from parcellation."""
    if "roi" not in parc.columns:
        return [], []
    ins_mask = parc["roi"].str.contains("INS", case=False, na=False)
    ifg_mask = parc["roi"].str.match(r"^IFG[s]?$", case=False, na=False)
    ins_chns = parc.loc[ins_mask, "channel"].unique().tolist()
    ifg_chns = parc.loc[ifg_mask, "channel"].unique().tolist()
    return ins_chns, ifg_chns


def filter_same_hemisphere(ins_chns: List[str], ifg_chns: List[str],
                           parc: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Keep only channels where Insula and IFG share at least one hemisphere."""
    parc_idx = parc.set_index("channel")
    ins_hemis = set(parc_idx.loc[parc_idx.index.isin(ins_chns), "hemi"].unique())
    ifg_hemis = set(parc_idx.loc[parc_idx.index.isin(ifg_chns), "hemi"].unique())
    shared = ins_hemis & ifg_hemis
    if not shared:
        return [], []
    # Keep channels in shared hemispheres
    ins_chns = [ch for ch in ins_chns
                if parc_idx.loc[ch, "hemi"] in shared] if ins_chns else []
    ifg_chns = [ch for ch in ifg_chns
                if parc_idx.loc[ch, "hemi"] in shared] if ifg_chns else []
    return ins_chns, ifg_chns


def get_shank_channels(roi_chns: List[str], all_ch_names: List[str]) -> List[str]:
    """Get all channels on shanks that contain at least one ROI channel."""
    prefixes = {get_shank_prefix(ch) for ch in roi_chns}
    return [ch for ch in all_ch_names if get_shank_prefix(ch) in prefixes]


# ---------------------------------------------------------------------------
# Cross-correlation (adapted from run_xcorr.py)
# ---------------------------------------------------------------------------
def compute_xcorr_matrix(
    xdata: np.ndarray, sfreq: float, max_lag_s: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute trial×chan×chan×lag cross-correlation (squared = variance explained).

    Args:
        xdata: (n_trials, n_chan, n_time) — already z-scored
        sfreq: sampling rate
        max_lag_s: symmetric window in seconds

    Returns:
        xcorr: (n_trials, n_chan, n_chan, n_lags)
        lag_times: (n_lags,)
    """
    n_trials, n_chan, n_time = xdata.shape
    max_lag = int(max_lag_s * sfreq)
    lags = np.arange(-max_lag, max_lag + 1)
    lag_times = lags / sfreq

    pad_len = 1 << int(np.ceil(np.log2(n_time + 2 * max_lag)))
    freqs = np.fft.rfft(xdata, n=pad_len, axis=2)
    xcorr = np.empty((n_trials, n_chan, n_chan, len(lags)), dtype=np.float32)

    for t in range(n_trials):
        F = freqs[t]
        cs = F[:, None, :] * np.conj(F[None, :, :])
        corr_full = np.fft.irfft(cs, n=pad_len, axis=-1)
        neg = corr_full[..., -max_lag:]
        pos = corr_full[..., : max_lag + 1]
        seg = np.concatenate([neg, pos], axis=-1) / n_time
        idx = np.abs(seg).argmax(axis=-1)[..., None]
        peak_sign = np.sign(np.take_along_axis(seg, idx, axis=-1)[..., 0])
        peak_sign[peak_sign == 0] = 1.0
        xcorr[t] = seg * peak_sign[..., None]

    xcorr = xcorr ** 2
    return xcorr, lag_times


# ---------------------------------------------------------------------------
# Mesh processing
# ---------------------------------------------------------------------------
def load_and_decimate_pial(
    recon_id: str, recon_dir: str, hemi: str,
    labels: list, target_faces: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pial surface, color by labels, then decimate.

    Returns (verts, faces, colors) where colors is uint8 (n_verts, 3).
    """
    surf_path = os.path.join(recon_dir, recon_id, "surf", f"{hemi}.pial")
    coords, faces = mne.read_surface(surf_path)

    # Color on original (full) mesh — label vertices are valid here
    orig_colors = _color_vertices_original(len(coords), labels, hemi)

    n_faces_orig = len(faces)
    if pv is not None and n_faces_orig > target_faces:
        pv_faces = np.column_stack([np.full(n_faces_orig, 3), faces]).flatten()
        mesh = pv.PolyData(coords, pv_faces)
        ratio = 1.0 - target_faces / n_faces_orig
        mesh = mesh.decimate(ratio)
        dec_verts = np.array(mesh.points, dtype=np.float32)
        dec_faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
        # Map colors: find nearest original vertex for each decimated vertex
        tree = cKDTree(coords)
        _, idx = tree.query(dec_verts)
        dec_colors = orig_colors[idx]
        return dec_verts, dec_faces, dec_colors

    return coords.astype(np.float32), faces.astype(np.int32), orig_colors


def _color_vertices_original(n_verts: int, labels: list, hemi: str) -> np.ndarray:
    """Assign RGB color per vertex on the original (non-decimated) mesh."""
    colors = np.tile(COLORS["default"], (n_verts, 1)).astype(np.uint8)
    for lab in labels:
        if lab.hemi != hemi:
            continue
        if any(p in lab.name for p in INSULA_PATTERNS):
            c = COLORS["insula"]
        elif any(p in lab.name for p in IFG_PATTERNS):
            c = COLORS["ifg"]
        else:
            continue
        for v in lab.vertices:
            if v < n_verts:
                colors[v] = c
    return colors


def color_vertices(verts: np.ndarray, labels: list, hemi: str) -> np.ndarray:
    """Assign RGB color per vertex based on parcellation labels (for testing)."""
    return _color_vertices_original(len(verts), labels, hemi)


# ---------------------------------------------------------------------------
# Electrode processing
# ---------------------------------------------------------------------------
def build_electrode_list(
    shank_chns: List[str],
    ins_chns: List[str],
    ifg_chns: List[str],
    montage_pos: Dict[str, np.ndarray],
    parc: pd.DataFrame,
    pial_coords_lh: np.ndarray,
    pial_coords_rh: np.ndarray,
) -> List[dict]:
    """Build electrode metadata list with projected coordinates."""
    ins_set, ifg_set = set(ins_chns), set(ifg_chns)
    parc_idx = parc.set_index("channel")
    lh_tree = cKDTree(pial_coords_lh)
    rh_tree = cKDTree(pial_coords_rh)

    electrodes = []
    for ch in shank_chns:
        if ch not in montage_pos:
            continue
        xyz = montage_pos[ch] * 1000  # meters -> mm
        if not np.all(np.isfinite(xyz)):
            continue

        # Project onto nearest pial vertex
        if xyz[0] < 0:
            _, idx = lh_tree.query(xyz)
            proj = pial_coords_lh[idx]
        else:
            _, idx = rh_tree.query(xyz)
            proj = pial_coords_rh[idx]

        if ch in ins_set:
            etype = "insula"
        elif ch in ifg_set:
            etype = "ifg"
        else:
            etype = "other"

        roi = parc_idx.loc[ch, "roi"] if ch in parc_idx.index else "unknown"
        label = parc_idx.loc[ch, "label"] if ch in parc_idx.index else "unknown"
        # Handle duplicate index (multiple rows per channel)
        if isinstance(roi, pd.Series):
            roi = roi.iloc[0]
        if isinstance(label, pd.Series):
            label = label.iloc[0]

        electrodes.append({
            "name": ch,
            "x": float(proj[0]),
            "y": float(proj[1]),
            "z": float(proj[2]),
            "roi": str(roi),
            "label": str(label),
            "type": etype,
        })
    return electrodes


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
def build_html(
    lh_verts: np.ndarray, lh_faces: np.ndarray, lh_colors: np.ndarray,
    rh_verts: np.ndarray, rh_faces: np.ndarray, rh_colors: np.ndarray,
    electrodes: List[dict],
    shank_chns: List[str],
    xcorr_mean: np.ndarray,
    lag_times: np.ndarray,
    metadata: dict,
) -> str:
    """Build self-contained HTML string."""

    # Prepare xcorr data: shank_chns order matches xcorr_mean axes
    # Replace NaN/Inf with 0 (NaN breaks JSON.parse in browser → blank page)
    xcorr_clean = np.nan_to_num(xcorr_mean, nan=0.0, posinf=0.0, neginf=0.0)
    # Downsample lag axis and round to reduce file size (N×N×lags can be huge)
    step = max(1, len(lag_times) // 128)
    xcorr_ds = xcorr_clean[:, :, ::step]
    lag_ds = lag_times[::step]
    xcorr_list = np.round(xcorr_ds, 4).tolist()
    lag_list = np.round(lag_ds, 4).tolist()

    data_json = json.dumps({
        "lh_verts": np.round(lh_verts, 2).tolist(),
        "lh_faces": lh_faces.tolist(),
        "lh_colors": lh_colors.tolist(),
        "rh_verts": np.round(rh_verts, 2).tolist(),
        "rh_faces": rh_faces.tolist(),
        "rh_colors": rh_colors.tolist(),
        "electrodes": electrodes,
        "shank_chns": shank_chns,
        "xcorr": xcorr_list,
        "lag_times": lag_list,
        "meta": metadata,
    }, separators=(",", ":"))

    html = _HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    return html


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>XCorr Viewer</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background:#f5f5f5; color:#333; overflow:hidden; }
#container { display:flex; width:100vw; height:100vh; }
#brain-panel { flex:2; position:relative; }
#info-panel { flex:1; min-width:320px; max-width:420px; padding:16px;
              display:flex; flex-direction:column; gap:12px;
              background:#fff; border-left:1px solid #ddd; overflow-y:auto; }
canvas { display:block; }
h2 { font-size:14px; color:#666; text-transform:uppercase; letter-spacing:1px; }
.meta { font-size:13px; color:#555; line-height:1.6; }
.meta b { color:#222; }
#sel-a, #sel-b { font-size:13px; padding:6px 10px; background:#e8eef6;
                 border-radius:6px; min-height:32px; color:#333; }
#plot { width:100%; height:260px; }
.legend { display:flex; gap:12px; font-size:12px; align-items:center;
          position:absolute; bottom:12px; left:12px; background:rgba(255,255,255,0.85);
          padding:6px 14px; border-radius:6px; border:1px solid #ccc; }
.legend span { display:flex; align-items:center; gap:4px; }
.dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
</style>
</head>
<body>
<div id="container">
  <div id="brain-panel">
    <div class="legend">
      <span><span class="dot" style="background:#D4AF37"></span> Insula</span>
      <span><span class="dot" style="background:#2369BD"></span> IFG</span>
      <span><span class="dot" style="background:#888"></span> Other</span>
    </div>
  </div>
  <div id="info-panel">
    <div class="meta" id="meta-info"></div>
    <h2>Selection</h2>
    <div id="sel-a">A: click an electrode</div>
    <div id="sel-b">B: click another electrode</div>
    <h2>Cross-correlation</h2>
    <div id="plot"></div>
  </div>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.163.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.163.0/examples/jsm/"
  }
}
</script>
<script src="https://cdn.plot.ly/plotly-basic-2.32.0.min.js"></script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DATA = JSON.parse('__DATA_JSON__');

// --- Metadata ---
document.getElementById('meta-info').innerHTML =
  `<b>Subject:</b> ${DATA.meta.subject} &nbsp; <b>Task:</b> ${DATA.meta.task}<br>` +
  `<b>Phase:</b> ${DATA.meta.phase} &nbsp; <b>Desc:</b> ${DATA.meta.desc}`;

// --- Scene setup ---
const panel = document.getElementById('brain-panel');
const W = panel.clientWidth, H = panel.clientHeight;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf5f5f5);
const camera = new THREE.PerspectiveCamera(45, W / H, 1, 2000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(W, H);
renderer.setPixelRatio(window.devicePixelRatio);
panel.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.15;
controls.rotateSpeed = 0.5;
controls.zoomSpeed = 0.8;
controls.panSpeed = 0.5;

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dLight = new THREE.DirectionalLight(0xffffff, 0.6);
dLight.position.set(100, 200, 150);
scene.add(dLight);
const dLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
dLight2.position.set(-100, -50, -100);
scene.add(dLight2);

// --- Build brain mesh ---
function buildBrainMesh(verts, faces, colors) {
  const geom = new THREE.BufferGeometry();
  const pos = new Float32Array(faces.length * 3 * 3);
  const col = new Float32Array(faces.length * 3 * 3);
  let idx = 0;
  for (const f of faces) {
    for (const vi of f) {
      pos[idx]   = verts[vi][0];
      pos[idx+1] = verts[vi][1];
      pos[idx+2] = verts[vi][2];
      col[idx]   = colors[vi][0] / 255;
      col[idx+1] = colors[vi][1] / 255;
      col[idx+2] = colors[vi][2] / 255;
      idx += 3;
    }
  }
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geom.computeVertexNormals();
  const mat = new THREE.MeshPhongMaterial({
    vertexColors: true, transparent: true, opacity: 0.35,
    side: THREE.DoubleSide, depthWrite: false,
  });
  return new THREE.Mesh(geom, mat);
}

const lhMesh = buildBrainMesh(DATA.lh_verts, DATA.lh_faces, DATA.lh_colors);
const rhMesh = buildBrainMesh(DATA.rh_verts, DATA.rh_faces, DATA.rh_colors);
scene.add(lhMesh);
scene.add(rhMesh);

// --- Electrodes ---
const TYPE_COLORS = {
  insula: new THREE.Color(0xD4AF37),
  ifg: new THREE.Color(0x2369BD),
  other: new THREE.Color(0x888888),
};
const SELECTED_COLOR = new THREE.Color(0xff4444);

const electrodeMeshes = [];
const electrodeMap = new Map(); // mesh.uuid -> electrode data

for (const e of DATA.electrodes) {
  const radius = e.type === 'other' ? 1.0 : 2.0;
  const geom = new THREE.SphereGeometry(radius, 16, 12);
  const mat = new THREE.MeshPhongMaterial({ color: TYPE_COLORS[e.type] || 0x888888, emissive: TYPE_COLORS[e.type] || new THREE.Color(0x888888), emissiveIntensity: 0.15 });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.set(e.x, e.y, e.z);
  scene.add(mesh);
  electrodeMeshes.push(mesh);
  electrodeMap.set(mesh.uuid, e);
}

// --- Camera position ---
// Center on all electrodes
let cx=0, cy=0, cz=0;
for (const e of DATA.electrodes) { cx+=e.x; cy+=e.y; cz+=e.z; }
const n = DATA.electrodes.length || 1;
cx/=n; cy/=n; cz/=n;
camera.position.set(cx - 180, cy, cz);
controls.target.set(cx, cy, cz);
controls.update();

// --- Selection state ---
let selA = null, selB = null;
let selMeshA = null, selMeshB = null;
let connectionLine = null;

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

renderer.domElement.addEventListener('click', (event) => {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(electrodeMeshes);
  if (hits.length === 0) return;
  const hitMesh = hits[0].object;
  const eData = electrodeMap.get(hitMesh.uuid);
  if (!eData) return;

  // Shift selection: B becomes A, new click becomes B
  if (selMeshA) {
    // Restore A color
    selMeshA.material.color.copy(TYPE_COLORS[selA.type] || new THREE.Color(0x888888));
  }
  selA = selB;
  selMeshA = selMeshB;
  selB = eData;
  selMeshB = hitMesh;
  hitMesh.material.color.copy(SELECTED_COLOR);

  updateInfoPanel();
  updatePlot();
  updateConnectionLine();
});

function updateInfoPanel() {
  const fmt = (e) => e ? `<b>${e.name}</b> (${e.roi})` : 'click an electrode';
  document.getElementById('sel-a').innerHTML = 'A: ' + fmt(selA);
  document.getElementById('sel-b').innerHTML = 'B: ' + fmt(selB);
}

// --- Connection line ---
function updateConnectionLine() {
  if (connectionLine) { scene.remove(connectionLine); connectionLine = null; }
  if (!selA || !selB) return;
  const pts = [
    new THREE.Vector3(selA.x, selA.y, selA.z),
    new THREE.Vector3(selB.x, selB.y, selB.z),
  ];
  const geom = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({ color: 0xff4444, linewidth: 2 });
  connectionLine = new THREE.Line(geom, mat);
  scene.add(connectionLine);
}

// --- Plotly chart ---
const plotDiv = document.getElementById('plot');
Plotly.newPlot(plotDiv, [{
  x: DATA.lag_times, y: DATA.lag_times.map(() => 0),
  type: 'scatter', mode: 'lines',
  line: { color: '#D4AF37', width: 2 },
}], {
  margin: { t: 30, b: 40, l: 50, r: 10 },
  xaxis: { title: 'Lag (s)', color: '#555', gridcolor: '#ddd' },
  yaxis: { title: 'xcorr\u00b2', color: '#555', gridcolor: '#ddd', range: [0, 0.25] },
  paper_bgcolor: '#fff',
  plot_bgcolor: '#fff',
  font: { color: '#333', size: 11 },
  title: { text: 'Select two electrodes', font: { size: 13 } },
  shapes: [{
    type: 'line', x0: 0, x1: 0, y0: 0, y1: 1,
    yref: 'paper', line: { color: '#ff4444', dash: 'dash', width: 1 },
  }],
}, { responsive: true, displayModeBar: false });

function updatePlot() {
  if (!selA || !selB) return;
  const roiChns = DATA.shank_chns;
  const iA = roiChns.indexOf(selA.name);
  const iB = roiChns.indexOf(selB.name);
  if (iA < 0 || iB < 0) {
    Plotly.update(plotDiv, {}, { title: { text: 'No xcorr data for this pair' } });
    return;
  }
  const curve = DATA.xcorr[iA][iB];
  Plotly.update(plotDiv,
    { y: [curve] },
    { title: { text: `${selA.name} vs ${selB.name}`, font: { size: 13 } },
      yaxis: { title: 'xcorr\u00b2', color: '#555', gridcolor: '#ddd', range: [0, 0.25] } }
  );
}

// --- Resize ---
window.addEventListener('resize', () => {
  const w = panel.clientWidth, h = panel.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});

// --- Animate ---
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_viewer(
    bids_root: str,
    subject: str,
    phase: str,
    desc: str,
    band: str = "highgamma",
    recon_dir: str = RECON_DIR,
    output: Optional[str] = None,
    max_lag_s: float = 1.0,
) -> str:
    """Generate interactive HTML viewer for one subject/phase/desc."""

    recon_id = bids_to_recon_id(subject)
    logger.info(f"Subject {subject} (recon: {recon_id})")

    # 1. Load parcellation & classify channels
    parc = load_parcellation(bids_root, subject)
    ins_chns, ifg_chns = classify_channels(parc)
    logger.info(f"  Insula channels: {len(ins_chns)}, IFG channels: {len(ifg_chns)}")

    if not ins_chns or not ifg_chns:
        raise ValueError(f"Subject {subject} lacks Insula or IFG channels")

    ins_chns, ifg_chns = filter_same_hemisphere(ins_chns, ifg_chns, parc)
    if not ins_chns or not ifg_chns:
        raise ValueError(f"Subject {subject}: Insula and IFG not in same hemisphere")

    logger.info(f"  After hemisphere filter: Insula={len(ins_chns)}, IFG={len(ifg_chns)}")

    # 2. Load epochs
    epochs = load_epochs(bids_root, subject, phase, desc, band)
    avail = set(epochs.ch_names)
    ins_chns = [ch for ch in ins_chns if ch in avail]
    ifg_chns = [ch for ch in ifg_chns if ch in avail]
    if not ins_chns or not ifg_chns:
        raise ValueError(f"No Insula/IFG channels found in epoch file")

    # ROI channels
    roi_chns = ins_chns + ifg_chns

    # Shank channels (for display AND xcorr — includes ROI + other on same shanks)
    shank_chns = get_shank_channels(roi_chns, epochs.ch_names)
    logger.info(f"  Shank channels: {len(shank_chns)}")

    # 3. Compute cross-correlation on ALL shank channels
    epochs_shank = epochs.copy().pick_channels(shank_chns, verbose="error")
    xdata = epochs_shank.get_data()  # (n_trials, n_shank, n_time)
    sfreq = epochs_shank.info["sfreq"]
    logger.info(f"  Computing xcorr: {xdata.shape[0]} trials, {xdata.shape[1]} channels")
    xcorr, lag_times = compute_xcorr_matrix(xdata, sfreq, max_lag_s)
    xcorr_mean = xcorr.mean(axis=0)  # (n_shank, n_shank, n_lags)

    # 4. Load parcellation labels, then load & decimate pial surfaces
    logger.info("  Loading pial surfaces...")
    labels = mne.read_labels_from_annot(
        subject=recon_id, parc="aparc.a2009s",
        hemi="both", subjects_dir=recon_dir, surf_name="pial",
    )
    lh_verts, lh_faces, lh_colors = load_and_decimate_pial(
        recon_id, recon_dir, "lh", labels,
    )
    rh_verts, rh_faces, rh_colors = load_and_decimate_pial(
        recon_id, recon_dir, "rh", labels,
    )

    # 6. Build electrode list
    montage = epochs.get_montage()
    pos = montage.get_positions()["ch_pos"]
    electrodes = build_electrode_list(
        shank_chns, ins_chns, ifg_chns, pos, parc, lh_verts, rh_verts,
    )
    logger.info(f"  Electrodes in viewer: {len(electrodes)}")

    # 7. Build HTML
    meta = {"subject": subject, "task": "", "phase": phase, "desc": desc}
    # Try to get task name from BIDSPath
    try:
        ep_path = BIDSPath(
            root=os.path.join(bids_root, "derivatives", "epoch(bipolar)"),
            datatype="epoch(band)(zscore)", subject=subject, suffix=band,
            processing=phase, extension=".h5", check=False,
        )
        matches = [m for m in ep_path.match() if m.description == desc]
        if matches:
            meta["task"] = matches[0].task or ""
    except Exception:
        pass

    html = build_html(
        lh_verts, lh_faces, lh_colors,
        rh_verts, rh_faces, rh_colors,
        electrodes, shank_chns, xcorr_mean, lag_times, meta,
    )

    # 8. Write output
    if output is None:
        output = f"viz/3d_xcorr/{subject}_{phase}_{desc}.html"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.write(html)
    logger.info(f"  Written: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate interactive 3D xcorr viewer")
    parser.add_argument("--bids_root", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--phase", default="Response")
    parser.add_argument("--desc", default="Repeat")
    parser.add_argument("--band", default="highgamma")
    parser.add_argument("--recon_dir", default=RECON_DIR)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_lag_s", type=float, default=1.0)
    args = parser.parse_args()
    generate_viewer(**vars(args))


if __name__ == "__main__":
    main()
