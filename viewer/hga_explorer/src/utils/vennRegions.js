import { PHASE_LABELS } from '../constants/phases.js';
import { resolvePhaseFlags } from './electrodeCoords.js';

export function regionLabel(regionId) {
  return regionId.split('_').map((phase) => PHASE_LABELS[phase] || phase).join(' ∩ ');
}

export function allExclusiveRegionIds(vennPhases) {
  const ids = [];
  for (let mask = 1; mask < (1 << vennPhases.length); mask += 1) {
    const active = vennPhases.filter((_, index) => mask & (1 << index));
    ids.push(active.join('_'));
  }
  return ids;
}

export function computeVennRegions(electrodes, vennPhases, selectedTask = 'all') {
  const members = Object.fromEntries(allExclusiveRegionIds(vennPhases).map((id) => [id, []]));
  (electrodes || []).forEach((electrode) => {
    const phaseFlags = resolvePhaseFlags(electrode, selectedTask);
    const active = vennPhases.filter((phase) => phaseFlags[phase]);
    if (active.length === 0) return;
    const regionId = active.join('_');
    if (members[regionId]) members[regionId].push(electrode.id);
  });
  return allExclusiveRegionIds(vennPhases).map((id) => ({
    id,
    label: regionLabel(id),
    phases_on: id.split('_'),
    phases_off: vennPhases.filter((phase) => !id.split('_').includes(phase)),
    electrode_ids: members[id],
    count: members[id].length,
  }));
}

export function electrodeExclusiveRegionId(electrode, vennPhases, selectedTask = 'all') {
  const phaseFlags = resolvePhaseFlags(electrode, selectedTask);
  const active = vennPhases.filter((phase) => phaseFlags[phase]);
  return active.length ? active.join('_') : null;
}

/** Default Venn selection: region with the most electrodes (not full intersection). */
export function pickDefaultVennRegionId(vennRegions) {
  if (!vennRegions?.length) return null;
  const best = [...vennRegions].sort((a, b) => (
    b.count - a.count || a.id.localeCompare(b.id)
  ))[0];
  return best?.count > 0 ? best.id : null;
}
