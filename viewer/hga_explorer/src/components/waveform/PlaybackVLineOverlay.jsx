import React from 'react';
import { PHASE_TIME_RANGES } from '../../constants/loads.js';

const MARGIN_TOP = 30;
const MARGIN_BOTTOM = 36;

export default function PlaybackVLineOverlay({
  phase,
  index,
  currentTime,
  visible,
}) {
  const { min, max } = PHASE_TIME_RANGES[phase];
  const marginLeft = index === 0 ? 48 : 28;
  const marginRight = 12;

  if (!visible || currentTime == null || !Number.isFinite(currentTime)) {
    return null;
  }

  const fraction = Math.max(0, Math.min(1, (currentTime - min) / (max - min)));

  return (
    <div
      className="playback-vline"
      style={{
        top: MARGIN_TOP,
        bottom: MARGIN_BOTTOM,
        left: `calc(${marginLeft}px + ${fraction} * (100% - ${marginLeft + marginRight}px))`,
      }}
      aria-hidden="true"
    />
  );
}
