export const BRAIN_MESH_URL = '/assets/cvs_avg35_pial.glb';

export const BRAIN_MESH_CENTER = [0.803, -2.16, -3.09];
export const BRAIN_HEMI_SPLIT_X = BRAIN_MESH_CENTER[0];

export const BRAIN_ORBIT_TARGET = [-4, 8, -10];
export const BRAIN_ORBIT_DISTANCE = 218;
export const BRAIN_ORBIT_AZIMUTH_DEG = 118;
export const BRAIN_ORBIT_ELEVATION_DEG = 28;
export const BRAIN_CAMERA_Z_OFFSET = -16;

export const BRAIN_HEMISPHERE_OPTIONS = [
  { id: 'left', label: 'Left' },
  { id: 'right', label: 'Right' },
  { id: 'both', label: 'Both' },
];

export const BRAIN_VIEW_OPTIONS = [
  { id: 'electrodes', label: 'Electrodes' },
  { id: 'kde', label: 'KDE projection' },
];

export const DEFAULT_UNIFORM_MARK_COLOR = '#ef4444';
export const INACTIVE_ELECTRODE_COLOR = '#64748b';
export const BRAIN_UP = [0, 0, 1];

/** MNE Brain(..., background='white') in univarite.ipynb */
export const BRAIN_SCENE_BACKGROUND = '#ffffff';
/** MNE Brain(..., cortex=(0.9, 0.9, 0.9)) */
export const DEFAULT_BRAIN_COLOR = '#e6e6e6';
export const DEFAULT_BRAIN_OPACITY = 0.1;
export const DEFAULT_BRAIN_VIEW_MODE = 'kde';
export const DEFAULT_ELECTRODE_BRAIN_OPACITY = 0.3;
export const DEFAULT_KDE_BRAIN_OPACITY = 0.9;
export const KDE_ELECTRODE_MODE_MAX = 5000;
/** 0 = flat unlit overlay, 1 = full Lambert; higher = stronger sulcal depth */
export const KDE_OVERLAY_LIT_MIX = 0.62;
/** Minimum brightness multiplier for wrap shading (lower = deeper sulci) */
export const KDE_OVERLAY_SHADE_FLOOR = 0.74;
/** Cap channel brightness to reduce gyral blow-out on white/pale colormap */
export const KDE_OVERLAY_HIGHLIGHT_CAP = 0.90;

export const HGA_RADIUS_MIN = 0.8;
export const HGA_RADIUS_MAX = 2.4;
export const ELECTRODE_BASE_RADIUS = 1;

export const WAVEFORM_PLOT_HEIGHT = 128;
export const WAVEFORM_PLOT_MIN_HEIGHT = 128;

export function brainCameraPosition(target, distance, azimuthDeg, elevationDeg, zOffset = 0) {
  const elev = (elevationDeg * Math.PI) / 180;
  const azim = (azimuthDeg * Math.PI) / 180;
  const horizontal = distance * Math.cos(elev);
  const dz = distance * Math.sin(elev);
  return [
    target[0] + horizontal * Math.cos(azim),
    target[1] + horizontal * Math.sin(azim),
    target[2] + dz + zOffset,
  ];
}

export const BRAIN_CAMERA = {
  position: brainCameraPosition(
    BRAIN_ORBIT_TARGET,
    BRAIN_ORBIT_DISTANCE,
    BRAIN_ORBIT_AZIMUTH_DEG,
    BRAIN_ORBIT_ELEVATION_DEG,
    BRAIN_CAMERA_Z_OFFSET,
  ),
  fov: 50,
  up: BRAIN_UP,
};
