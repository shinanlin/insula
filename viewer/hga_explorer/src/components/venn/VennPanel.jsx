import React, { useMemo } from 'react';
import { PHASES, PHASE_LABELS } from '../../constants/phases.js';
import { VENN_MAX_PHASES, VENN_MIN_PHASES } from '../../constants/venn.js';
import { phaseColor, regionHitStyle } from '../../constants/colors.js';
import { buildVennConfig } from '../../utils/vennLayout.js';
import VennDefs from './VennDefs.jsx';
import SubjectDropdown from './SubjectDropdown.jsx';

export default function VennPanel({
  vennPhases,
  regions,
  availableSubjects,
  selectedSubjects,
  onToggleSubject,
  onSelectAllSubjects,
  onDeselectAllSubjects,
  selectedRegionIds,
  onTogglePhase,
  onSelect,
}) {
  const config = useMemo(() => buildVennConfig(vennPhases, regions), [vennPhases, regions]);
  const regionMap = useMemo(() => {
    const map = new Map();
    regions.forEach((region) => map.set(region.id, region));
    return map;
  }, [regions]);
  const isSelected = (id) => selectedRegionIds.includes(id);
  const dimUnselected = selectedRegionIds.length > 0;

  return (
    <div className="venn-wrap">
      <div className="venn-main" data-tour="venn-selector">
      <div className="venn-phase-picker">
        {PHASES.map((phase) => {
          const active = vennPhases.includes(phase);
          const disabled = active
            ? vennPhases.length <= VENN_MIN_PHASES
            : vennPhases.length >= VENN_MAX_PHASES;
          return (
            <button
              key={phase}
              type="button"
              className={active ? 'venn-phase-chip active' : 'venn-phase-chip'}
              style={{ '--phase-color': phaseColor(phase) }}
              disabled={disabled}
              onClick={() => onTogglePhase(phase)}
            >
              {PHASE_LABELS[phase]}
            </button>
          );
        })}
      </div>
      <div className="venn-phase-hint">Choose 2–4 phases for the Venn diagram.</div>
      <svg
        viewBox={config.viewBox}
        className={`venn-svg${vennPhases.length >= 4 ? ' is-four-phase' : ''}${dimUnselected ? ' has-region-focus' : ''}`}
        role="img"
        aria-label="Phase overlap Venn selector"
      >
        <VennDefs config={config} />
        {config.circles.map((circle, index) => (
          <circle
            key={circle.key}
            cx={circle.cx}
            cy={circle.cy}
            r={circle.r}
            className={`venn-circle ${vennPhases[index]}-circle`}
          />
        ))}
        {config.subregions.map((sub) => {
          const region = regionMap.get(sub.id);
          if (!region) return null;
          const selected = isSelected(sub.id);
          const style = regionHitStyle(region.phases_on, selected, dimUnselected);
          return (
            <rect
              key={sub.id}
              x={config.minX ?? 0}
              y={config.minY ?? 0}
              width={config.width}
              height={config.height}
              clipPath={`url(#${sub.clipId})`}
              mask={sub.maskId ? `url(#${sub.maskId})` : undefined}
              className={`venn-region-hit${selected ? ' selected' : ''}${dimUnselected && !selected ? ' dimmed' : ''}`}
              style={style}
              onClick={() => onSelect(sub.id)}
            />
          );
        })}
        {config.subregions.map((sub) => {
          const region = regionMap.get(sub.id);
          if (!region || sub.hideCount) return null;
          const selected = isSelected(sub.id);
          return (
            <text
              key={`label-${sub.id}`}
              x={sub.countX}
              y={sub.countY}
              textAnchor="middle"
              dominantBaseline="middle"
              className={`venn-count${selected ? ' selected' : ''}${dimUnselected && !selected ? ' dimmed' : ''}`}
            >
              {region.count}
            </text>
          );
        })}
        {config.circles.map((circle, index) => (
          <text
            key={`title-${circle.key}`}
            x={circle.labelX}
            y={circle.labelY}
            textAnchor={circle.labelAnchor || 'middle'}
            className="venn-label"
          >
            {PHASE_LABELS[vennPhases[index]]}
          </text>
        ))}
      </svg>
      <div className="venn-instruction">Click Venn components to toggle; highlighted electrodes are the union of selected components.</div>
      </div>

      {availableSubjects.length > 0 && (
        <div className="venn-subject-section" data-tour="subject-filter">
          <div className="venn-subject-section-title">Subject filter</div>
          <SubjectDropdown
            availableSubjects={availableSubjects}
            selectedSubjects={selectedSubjects}
            onToggleSubject={onToggleSubject}
            onSelectAllSubjects={onSelectAllSubjects}
            onDeselectAllSubjects={onDeselectAllSubjects}
          />
          <div className="venn-subject-hint">
            Filter subjects for Venn counts, brain map, and waveforms.
          </div>
        </div>
      )}
    </div>
  );
}
