import React, { useMemo } from 'react';
import {
  EXPORT_PHASE_TO_KEY,
  normalizePhaseKey,
  PHASES,
} from '../constants/waveform.js';
import { computeTraceYRange, traceFromExport } from '../utils/waveformTraces.js';
import PhaseWaveformPlot from './waveform/PhaseWaveformPlot.jsx';

const DISPLAY_PHASES = ['Stimulus', 'Delay', 'Go', 'Response'];

export default function WaveformPanel({
  title,
  traces,
  empty = false,
  traceKey = 'selected',
  taskUnavailable = false,
  taskUnavailableMessage = null,
  playingPhase = null,
  animationTime = null,
  lineColor,
}) {
  const phaseTraces = useMemo(() => {
    const out = {};
    DISPLAY_PHASES.forEach((exportPhase) => {
      const key = normalizePhaseKey(exportPhase);
      out[key] = traceFromExport(traces?.[exportPhase]);
    });
    return out;
  }, [traces]);

  const yRange = useMemo(() => {
    const ranges = PHASES
      .map((phase) => phaseTraces[phase])
      .filter(Boolean)
      .map((trace) => computeTraceYRange(trace));
    if (!ranges.length) return [-0.5, 1.5];
    return [
      Math.min(...ranges.map((r) => r[0])),
      Math.max(...ranges.map((r) => r[1])),
    ];
  }, [phaseTraces]);

  return (
    <div className="waveform-body">
      <div className="waveform-title" title={title}>{title}</div>
      {taskUnavailable && taskUnavailableMessage && (
        <div className="waveform-hint">{taskUnavailableMessage}</div>
      )}
      <div className="waveform-grid-wrap">
        <div className={`waveform-grid${empty ? ' is-empty' : ''}`}>
          {DISPLAY_PHASES.map((exportPhase, index) => {
            const phaseKey = EXPORT_PHASE_TO_KEY[exportPhase] ?? normalizePhaseKey(exportPhase);
            return (
              <div key={exportPhase} className="plot-card">
                <PhaseWaveformPlot
                  phase={phaseKey}
                  index={index}
                  trace={phaseTraces[phaseKey]}
                  traceKey={traceKey}
                  yRange={yRange}
                  animationTime={playingPhase === phaseKey ? animationTime : null}
                  showPlaybackLine={playingPhase === phaseKey && animationTime != null}
                  lineColor={lineColor}
                />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
