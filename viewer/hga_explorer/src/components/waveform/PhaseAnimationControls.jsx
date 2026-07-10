import React from 'react';
import { Loader2, Pause, Play } from 'lucide-react';
import { PHASE_LABELS } from '../../constants/phases.js';
import { PHASE_TIME_RANGES } from '../../constants/loads.js';
import { bundleHasPlayableFrames } from '../../utils/animationBundle.js';
import PieProgress from '../ui/PieProgress.jsx';

export default function PhaseAnimationControls({
  phase,
  bundle,
  canPlay,
  isLoading,
  isPreparing,
  renderProgress = 0,
  playingPhase,
  isPlaying,
  frameIdx,
  onTogglePlay,
  onSeek,
}) {
  const { min } = PHASE_TIME_RANGES[phase];
  const hasFrames = bundleHasPlayableFrames(bundle);
  const frameCount = bundle?.frames?.length ?? 0;
  const isActive = playingPhase === phase;
  const activeFrameIdx = isActive ? frameIdx : 0;
  const currentTime = hasFrames
    ? bundle.frames[activeFrameIdx]?.time ?? min
    : min;
  const playDisabled = !canPlay || isLoading;
  const playLabel = isLoading
    ? 'Loading'
    : isActive && isPreparing
      ? 'Cancel'
      : isActive && isPlaying
        ? 'Pause'
        : 'Play';

  return (
    <div className={`phase-animation-controls${isActive ? ' playing' : ''}${isActive && isPreparing ? ' preparing' : ''}`}>
      <div className="phase-animation-controls-row">
        <button
          type="button"
          className={`play-btn${isLoading ? ' loading' : ''}${isActive && isPreparing ? ' preparing' : ''}`}
          disabled={playDisabled}
          onClick={() => onTogglePlay(phase)}
          aria-label={isLoading
            ? `Loading ${PHASE_LABELS[phase]} animation`
            : isActive && isPreparing
              ? `Cancel ${PHASE_LABELS[phase]} map preparation`
              : isActive && isPlaying
                ? `Pause ${PHASE_LABELS[phase]} animation`
                : `Play ${PHASE_LABELS[phase]} animation`}
        >
          {isLoading
            ? <Loader2 size={14} className="spin-icon" />
            : isActive && isPreparing
              ? (
                <PieProgress
                  progress={renderProgress}
                  size={18}
                  strokeWidth={3}
                  className="play-btn-pie"
                  compact
                />
              )
            : isActive && isPlaying
              ? <Pause size={14} />
              : <Play size={14} />}
          <span>{playLabel}</span>
        </button>
        <span className="time-label">t = {currentTime.toFixed(2)}s</span>
        <span
          className={`scale-label${isActive && bundle?.scale?.vmax != null ? ' visible' : ' reserved'}`}
          aria-hidden={!(isActive && bundle?.scale?.vmax != null)}
        >
          {isActive && bundle?.scale?.vmax != null
            ? `max = ${bundle.scale.vmax.toFixed(2)} z`
            : 'max = — z'}
        </span>
      </div>
      <input
        type="range"
        className="time-scrubber-input"
        min={0}
        max={Math.max(0, frameCount - 1)}
        step={1}
        value={activeFrameIdx}
        disabled={!hasFrames || isLoading || (isActive && isPreparing)}
        onChange={(event) => onSeek(phase, Number(event.target.value))}
        aria-label={`${PHASE_LABELS[phase]} animation time scrubber`}
        aria-valuetext={`${currentTime.toFixed(2)} seconds`}
      />
    </div>
  );
}
