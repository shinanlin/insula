export const INSULA_MESH_URL = '/assets/cvs_avg35_insula_pial.glb';
export const INSULA_META_URL = '/assets/cvs_avg35_insula.meta.json';

export const CLUSTER_COLORS = {
  sustain: '#A9373B',
  motor: '#C4A35A',
  sensory: '#2369BD',
};

export const CLUSTER_LABELS = {
  sustain: 'sustain',
  motor: 'motor',
  sensory: 'sensory',
};

export const CLUSTERS = Object.keys(CLUSTER_COLORS);

/** 3D opacity / style by loading-margin purity (see utils/purity.js). */
export const PURITY_OPACITY = {
  pure: 0.95,
  borderline: 0.48,
  mixed: 0.9,
};

export const INSULA_ORBIT_TARGET = [2.009, 19.925, -18.837];
export const INSULA_ORBIT_DISTANCE = 180;
export const INSULA_ORBIT_AZIMUTH_DEG = 118;
export const INSULA_ORBIT_ELEVATION_DEG = 90;

export const ELECTRODE_BASE_RADIUS = 1.1;
export const ELECTRODE_RENDER_ORDER = 10;

export function brainCameraPosition(target, distance, azimuthDeg, elevationDeg) {
  const elev = (azimuthDeg * Math.PI) / 180;
  const azim = (elevationDeg * Math.PI) / 180;
  const horizontal = distance * Math.cos(elev);
  const dz = distance * Math.sin(elev);
  return [
    target[0] + horizontal * Math.cos(azim),
    target[1] + horizontal * Math.sin(azim),
    target[2] + dz,
  ];
}

export const INSULA_CAMERA = {
  position: brainCameraPosition(
    INSULA_ORBIT_TARGET,
    INSULA_ORBIT_DISTANCE,
    INSULA_ORBIT_AZIMUTH_DEG,
    INSULA_ORBIT_ELEVATION_DEG,
  ),
  fov: 50,
};
