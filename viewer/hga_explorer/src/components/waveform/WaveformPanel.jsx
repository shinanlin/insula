import React, { useMemo } from 'react';
import { PHASES, PHASE_LABELS } from '../../constants/phases.js';
import {
  resolvePanelPhaseTrace,
  clipTraceToPhaseWindow,
} from '../../utils/traces.js';
import { resolveWaveformYRange } from '../../constants/waveform.js';
import { formatWaveformTitle } from '../../utils/selectionSummary.js';
import PhaseAnimationControls from './PhaseAnimationControls.jsx';
import PhaseWaveformPlot from './PhaseWaveformPlot.jsx';
import PanelEmptyState from '../layout/PanelEmptyState.jsx';
import TraceLoadProgress from '../ui/TraceLoadProgress.jsx';
import { formatViewSelectionLabel } from '../../utils/viewSelection.js';

const StaticWaveformBody = React.memo(function StaticWaveformBody({
  phase,
  index,
  trace,
  traceKey,
  yRange,
  hasTrace,
  isSingleElectrode,
  isActivePhase,
  currentTime,
  selectionEmpty,
}) {
  if (!hasTrace) {
    return (
      <div className="plot-empty">
        {selectionEmpty
          ? selectionEmpty.message
          : `No HGA trace for ${PHASE_LABELS[phase]} in this selection`}
      </div>
    );
  }

  return (
    <PhaseWaveformPlot
      key={`${traceKey}-${phase}`}
      phase={phase}
      index={index}
      trace={trace}
      traceKey={traceKey}
      yRange={yRange}
      isSingleElectrode={isSingleElectrode}
      isActivePhase={isActivePhase}
      currentTime={currentTime}
    />
  );
}, (prev, next) => (
  prev.phase === next.phase
  && prev.index === next.index
  && prev.traceKey === next.traceKey
  && prev.hasTrace === next.hasTrace
  && prev.isSingleElectrode === next.isSingleElectrode
  && prev.yRange[0] === next.yRange[0]
  && prev.yRange[1] === next.yRange[1]
  && prev.isActivePhase === next.isActivePhase
  && prev.currentTime === next.currentTime
  && prev.selectionEmpty === next.selectionEmpty
));

const PhasePlotCard = React.memo(function PhasePlotCard({
  phase,
  index,
  staticTrace,
  traceKey,
  playback,
  isSingleElectrode,
  canPlay,
  animationLoadingPhase,
  animationLoadProgress,
  playingPhase,
  isPlaying,
  awaitingKdeRender,
  renderProgress,
  animationFrameIdx,
  selectionEmpty,
  onTogglePlay,
  onSeek,
}) {
  const isActivePhase = playback.isActivePhase;
  const isLoadingAnimation = animationLoadingPhase === phase;
  const showAnimationLoadOverlay = isLoadingAnimation && animationLoadProgress.phase === phase;

  return (
    <div className={`plot-card${isActivePhase ? ' playing' : ''}${showAnimationLoadOverlay ? ' loading-animation' : ''}`}>
      <PhaseAnimationControls
        phase={phase}
        bundle={playback.controlsBundle}
        canPlay={canPlay}
        isLoading={isLoadingAnimation}
        isPreparing={isActivePhase && awaitingKdeRender}
        renderProgress={renderProgress}
        playingPhase={playingPhase}
        isPlaying={isPlaying}
        frameIdx={animationFrameIdx}
        onTogglePlay={onTogglePlay}
        onSeek={onSeek}
      />
      {showAnimationLoadOverlay && (
        <div className="plot-animation-loading">
          <TraceLoadProgress
            compact
            title="Loading animation"
            progress={animationLoadProgress.progress}
            completed={animationLoadProgress.completed}
            total={animationLoadProgress.total}
            subjectLabel="subjects"
          />
        </div>
      )}
      <StaticWaveformBody
        phase={phase}
        index={index}
        trace={staticTrace.trace}
        traceKey={traceKey}
        yRange={staticTrace.yRange}
        hasTrace={staticTrace.hasTrace}
        isSingleElectrode={isSingleElectrode}
        isActivePhase={isActivePhase}
        currentTime={playback.currentTime}
        selectionEmpty={selectionEmpty}
      />
    </div>
  );
}, (prev, next) => (
  prev.phase === next.phase
  && prev.index === next.index
  && prev.traceKey === next.traceKey
  && prev.isSingleElectrode === next.isSingleElectrode
  && prev.canPlay === next.canPlay
  && prev.animationLoadingPhase === next.animationLoadingPhase
  && prev.playingPhase === next.playingPhase
  && prev.isPlaying === next.isPlaying
  && prev.awaitingKdeRender === next.awaitingKdeRender
  && prev.renderProgress === next.renderProgress
  && prev.animationFrameIdx === next.animationFrameIdx
  && prev.selectionEmpty === next.selectionEmpty
  && prev.staticTrace === next.staticTrace
  && prev.playback === next.playback
  && prev.animationLoadProgress === next.animationLoadProgress
));

function WaveformPanel({
  electrode,
  summary,
  electrodes,
  traces,
  selectedLoad,
  layout = 'split',
  tracesLoading = false,
  tracesLoadProgress = { completed: 0, total: 0, progress: 0 },
  initialLoadComplete = true,
  animationCache,
  animationLoadingPhase,
  animationLoadProgress = { completed: 0, total: 0, progress: 0, phase: null },
  canPlay,
  selectionEmpty = null,
  playingPhase,
  isPlaying,
  awaitingKdeRender = false,
  renderProgress = 0,
  animationFrameIdx,
  onTogglePlay,
  onSeek,
}) {
  const loadLabel = formatViewSelectionLabel(selectedLoad);
  const isSingleElectrode = Boolean(electrode);
  const traceKey = electrode?.id ?? 'aggregate';
  const { title, fullTitle } = formatWaveformTitle({
    summary,
    isSingleElectrode,
    electrode,
    loadLabel,
    electrodeCount: electrodes.length,
  });

  const allowMock = layout === 'mock';
  const awaitingTraces = layout === 'split' && tracesLoading && initialLoadComplete;

  const yRange = useMemo(() => resolveWaveformYRange(), []);

  const staticTraces = useMemo(() => (
    Object.fromEntries(PHASES.map((phase, index) => {
      const resolved = awaitingTraces
        ? null
        : resolvePanelPhaseTrace(traces, electrodes, phase, selectedLoad, electrode, allowMock);
      const rawTrace = resolved
        ? { x: resolved.time, y: resolved.value, sem: resolved.sem ?? null }
        : { x: [], y: [], sem: null };
      const trace = clipTraceToPhaseWindow(rawTrace, phase);
      return [phase, {
        phase,
        index,
        trace,
        yRange,
        hasTrace: trace.x.length > 0,
      }];
    }))
  ), [
    traces,
    electrodes,
    selectedLoad,
    electrode,
    allowMock,
    awaitingTraces,
    yRange,
  ]);

  const playbackByPhase = useMemo(() => (
    Object.fromEntries(PHASES.map((phase) => {
      const phaseBundle = animationCache?.[phase];
      const isActivePhase = playingPhase === phase;
      return [phase, {
        isActivePhase,
        currentTime: isActivePhase
          ? phaseBundle?.frames?.[animationFrameIdx]?.time ?? null
          : null,
        controlsBundle: phaseBundle,
      }];
    }))
  ), [animationCache, playingPhase, animationFrameIdx]);

  return (
    <div className="waveform-body">
      <div className="waveform-header">
        <div className="waveform-title" title={fullTitle}>{title}</div>
      </div>
      <div className="waveform-grid-wrap" data-tour="waveform-panel">
        {awaitingTraces && (
          <div className="waveform-loading">
            <div className="waveform-loading-card">
              <TraceLoadProgress
                progress={tracesLoadProgress.progress}
                completed={tracesLoadProgress.completed}
                total={tracesLoadProgress.total}
              />
              <p className="waveform-loading-note">
                Waveforms will appear once subject traces finish loading.
              </p>
            </div>
          </div>
        )}
        <div
          className={`waveform-grid${selectionEmpty ? ' is-empty' : ''}${awaitingTraces ? ' is-loading' : ''}`}
          style={{ gridTemplateColumns: `repeat(${PHASES.length}, minmax(0, 1fr))` }}
        >
          {PHASES.map((phase) => (
            <PhasePlotCard
              key={phase}
              phase={phase}
              index={staticTraces[phase].index}
              staticTrace={staticTraces[phase]}
              traceKey={traceKey}
              playback={playbackByPhase[phase]}
              isSingleElectrode={isSingleElectrode}
              canPlay={canPlay}
              animationLoadingPhase={animationLoadingPhase}
              animationLoadProgress={animationLoadProgress}
              playingPhase={playingPhase}
              isPlaying={isPlaying}
              awaitingKdeRender={awaitingKdeRender}
              renderProgress={renderProgress}
              animationFrameIdx={animationFrameIdx}
              selectionEmpty={selectionEmpty}
              onTogglePlay={onTogglePlay}
              onSeek={onSeek}
            />
          ))}
        </div>
        <PanelEmptyState emptyState={selectionEmpty} className="waveform-empty-state" />
      </div>
    </div>
  );
}

export default React.memo(WaveformPanel);
