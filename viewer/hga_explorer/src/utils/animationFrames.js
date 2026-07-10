import {
  ANIM_GAUSSIAN_SIGMA_SEC,
  ANIM_STEP_SEC,
  ANIM_WINDOW_SEC,
} from '../constants/animation.js';
import { PHASE_TIME_RANGES } from '../constants/loads.js';
import {
  causalWindowMeanForElectrode,
} from './traces.js';

function percentile95(values) {
  if (!values.length) return 1;
  const sorted = values.map((value) => Math.abs(value)).sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor(0.95 * (sorted.length - 1)));
  return sorted[idx] > 0 ? sorted[idx] : 1;
}

function causalGaussianSmoothSeries(times, values, sigmaSec) {
  if (!values.length) return [];
  if (sigmaSec <= 0) return [...values];
  const sigmaSq2 = 2 * sigmaSec * sigmaSec;
  return values.map((_, index) => {
    const tCurrent = times[index];
    let weightedSum = 0;
    let weightSum = 0;
    for (let j = 0; j <= index; j += 1) {
      const value = values[j];
      if (value == null || !Number.isFinite(value)) continue;
      const dt = tCurrent - times[j];
      const weight = Math.exp(-(dt * dt) / sigmaSq2);
      weightedSum += weight * value;
      weightSum += weight;
    }
    return weightSum > 0 ? weightedSum / weightSum : null;
  });
}

function smoothAnimationFrames(frames, sigmaSec = ANIM_GAUSSIAN_SIGMA_SEC) {
  if (!frames.length || sigmaSec <= 0) return frames;
  const times = frames.map((frame) => frame.time);
  const electrodeIds = new Set();
  frames.forEach((frame) => {
    Object.keys(frame.hgaByElectrodeId).forEach((id) => electrodeIds.add(id));
  });

  const smoothedByElectrode = {};
  electrodeIds.forEach((electrodeId) => {
    const values = frames.map((frame) => frame.hgaByElectrodeId[electrodeId] ?? null);
    smoothedByElectrode[electrodeId] = causalGaussianSmoothSeries(times, values, sigmaSec);
  });

  return frames.map((frame, index) => {
    const hgaByElectrodeId = {};
    electrodeIds.forEach((electrodeId) => {
      const value = smoothedByElectrode[electrodeId][index];
      if (value != null && Number.isFinite(value)) {
        hgaByElectrodeId[electrodeId] = value;
      }
    });
    return { time: frame.time, hgaByElectrodeId };
  });
}

export function buildSlidingWindowFrames(
  electrodes,
  traces,
  phase,
  selectedLoad,
  { windowSec = ANIM_WINDOW_SEC, stepSec = ANIM_STEP_SEC, allowMock = false } = {},
) {
  const { min, max } = PHASE_TIME_RANGES[phase];
  const tEnd = max - windowSec;
  const frames = [];

  for (let t = min; t <= tEnd + 1e-9; t += stepSec) {
    const hgaByElectrodeId = {};
    electrodes.forEach((electrode) => {
      const mean = causalWindowMeanForElectrode(
        traces,
        electrode,
        phase,
        selectedLoad,
        t,
        windowSec,
        allowMock,
      );
      if (mean != null && Number.isFinite(mean)) {
        hgaByElectrodeId[electrode.id] = mean;
      }
    });
    frames.push({ time: Number(t.toFixed(4)), hgaByElectrodeId });
  }

  const smoothedFrames = smoothAnimationFrames(frames);
  const smoothedValues = [];
  smoothedFrames.forEach((frame) => {
    Object.values(frame.hgaByElectrodeId).forEach((value) => smoothedValues.push(value));
  });

  return {
    frames: smoothedFrames,
    scale: {
      vmin: 0,
      vmax: percentile95(smoothedValues),
      method: 'p95_abs_sliding_window_gaussian',
    },
  };
}
