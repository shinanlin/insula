import {
  DEFAULT_UNIFORM_MARK_COLOR,
  INACTIVE_ELECTRODE_COLOR,
} from '../constants/brain.js';
import {
  regionDisplayColor,
  regionPhasesOn,
} from '../constants/colors.js';
import { electrodeExclusiveRegionId } from './vennRegions.js';

export function resolveBrainElectrodeColor({
  electrode,
  vennPhases,
  selected,
  colorByFunctional,
}) {
  if (!colorByFunctional) {
    return DEFAULT_UNIFORM_MARK_COLOR;
  }
  const regionId = electrodeExclusiveRegionId(electrode, vennPhases);
  if (selected && regionId) {
    return regionDisplayColor(regionPhasesOn(regionId));
  }
  return INACTIVE_ELECTRODE_COLOR;
}
