export const BRAIN_SPACES = {
  template: 'template',
  native: 'native',
};

export function nativeMeshUrl(subject) {
  if (!subject) return null;
  return `/assets/native/${subject}_pial.glb`;
}

export function resolvePhaseFlags(electrode, task = 'all') {
  if (task !== 'all' && electrode?.phase_flags_by_task?.[task]) {
    return electrode.phase_flags_by_task[task];
  }
  return electrode?.phase_flags || {};
}

export function resolveMidpointCoords(electrode, brainSpace = BRAIN_SPACES.template) {
  if (!electrode) return null;
  if (brainSpace === BRAIN_SPACES.native) {
    const x = electrode.x_native ?? electrode.x;
    const y = electrode.y_native ?? electrode.y;
    const z = electrode.z_native ?? electrode.z;
    if ([x, y, z].every((value) => value != null && Number.isFinite(value))) {
      return { x, y, z };
    }
  }
  return { x: electrode.x, y: electrode.y, z: electrode.z };
}

export function resolveEndpointCoords(electrode, contact, brainSpace = BRAIN_SPACES.template) {
  if (!electrode || !contact) return null;
  const suffix = brainSpace === BRAIN_SPACES.native ? 'native' : 'template';
  const prefix = contact === 'contact_2' ? '2' : '1';
  const x = electrode[`x${prefix}_${suffix}`];
  const y = electrode[`y${prefix}_${suffix}`];
  const z = electrode[`z${prefix}_${suffix}`];
  if ([x, y, z].every((value) => value != null && Number.isFinite(value))) {
    return { x, y, z };
  }
  return null;
}

export function withResolvedCoords(electrode, brainSpace = BRAIN_SPACES.template) {
  const midpoint = resolveMidpointCoords(electrode, brainSpace);
  if (!midpoint) return electrode;
  return {
    ...electrode,
    x: midpoint.x,
    y: midpoint.y,
    z: midpoint.z,
  };
}

export function resolveElectrodesForBrainSpace(electrodes, brainSpace = BRAIN_SPACES.template) {
  return (electrodes || [])
    .map((electrode) => withResolvedCoords(electrode, brainSpace))
    .filter((electrode) => (
      electrode.x != null && electrode.y != null && electrode.z != null
    ));
}
