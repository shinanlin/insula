import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ANIM_STEP_MS } from '../constants/animation.js';
import { PHASES } from '../constants/waveform.js';
import {
  buildSlidingWindowFrames,
  bundleHasPlayableFrames,
} from '../utils/animationFrames.js';

function buildCacheKey(phase, selectedTask, electrodeKey) {
  return `${selectedTask ?? 'none'}|${phase}|${electrodeKey}`;
}

export default function usePhasePlayback({
  visibleElectrodes,
  visibleElectrodesKey,
  tracesBySubject,
  selectedTask,
}) {
  const [playingPhase, setPlayingPhase] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [frameIdx, setFrameIdx] = useState(0);
  const [cacheVersion, setCacheVersion] = useState(0);
  const animationCacheRef = useRef(new Map());

  const getCacheKey = useCallback(
    (phase) => buildCacheKey(phase, selectedTask, visibleElectrodesKey),
    [selectedTask, visibleElectrodesKey],
  );

  const getCachedBundle = useCallback((phase) => {
    return animationCacheRef.current.get(getCacheKey(phase)) ?? null;
  }, [getCacheKey, cacheVersion]);

  const setCachedBundle = useCallback((phase, bundle) => {
    animationCacheRef.current.set(getCacheKey(phase), bundle);
    setCacheVersion((version) => version + 1);
  }, [getCacheKey]);

  const loadPhaseAnimation = useCallback((phase) => {
    const cached = getCachedBundle(phase);
    if (bundleHasPlayableFrames(cached)) {
      return cached;
    }
    if (!selectedTask || !visibleElectrodes.length) {
      return { frames: [], scale: null };
    }
    const bundle = buildSlidingWindowFrames(
      visibleElectrodes,
      tracesBySubject,
      phase,
      selectedTask,
    );
    if (bundleHasPlayableFrames(bundle)) {
      setCachedBundle(phase, bundle);
    }
    return bundle;
  }, [
    getCachedBundle,
    selectedTask,
    visibleElectrodes,
    tracesBySubject,
    setCachedBundle,
  ]);

  const activeAnimation = playingPhase ? getCachedBundle(playingPhase) : null;
  const liveHgaByElectrodeId = activeAnimation?.frames?.[frameIdx]?.hgaByElectrodeId ?? null;
  const animationScale = playingPhase ? activeAnimation?.scale ?? null : null;
  const animationTime = activeAnimation?.frames?.[frameIdx]?.time ?? null;

  useEffect(() => {
    setIsPlaying(false);
    setPlayingPhase(null);
    setFrameIdx(0);
  }, [visibleElectrodesKey, selectedTask]);

  useEffect(() => {
    if (!isPlaying || !playingPhase) return undefined;
    const bundle = getCachedBundle(playingPhase);
    if (!bundle?.frames?.length) {
      setIsPlaying(false);
      return undefined;
    }

    let rafId = null;
    let lastTimestamp = null;
    let accumulator = 0;
    let stopped = false;

    const tick = (timestamp) => {
      if (stopped) return;
      if (lastTimestamp == null) {
        lastTimestamp = timestamp;
      }
      accumulator += timestamp - lastTimestamp;
      lastTimestamp = timestamp;

      if (accumulator >= ANIM_STEP_MS) {
        accumulator %= ANIM_STEP_MS;
        setFrameIdx((current) => {
          if (current >= bundle.frames.length - 1) {
            setIsPlaying(false);
            return current;
          }
          return current + 1;
        });
      }

      rafId = window.requestAnimationFrame(tick);
    };

    rafId = window.requestAnimationFrame(tick);
    return () => {
      stopped = true;
      if (rafId != null) window.cancelAnimationFrame(rafId);
    };
  }, [isPlaying, playingPhase, cacheVersion, getCachedBundle]);

  const togglePlay = useCallback((phase) => {
    if (!selectedTask) return;

    if (playingPhase === phase && isPlaying) {
      setIsPlaying(false);
      return;
    }

    if (playingPhase === phase && !isPlaying) {
      setFrameIdx(0);
      setIsPlaying(true);
      return;
    }

    const bundle = loadPhaseAnimation(phase);
    if (!bundleHasPlayableFrames(bundle)) return;

    setPlayingPhase(phase);
    setFrameIdx(0);
    setIsPlaying(true);
  }, [selectedTask, playingPhase, isPlaying, loadPhaseAnimation]);

  const stopPlayback = useCallback(() => {
    setIsPlaying(false);
    setPlayingPhase(null);
    setFrameIdx(0);
  }, []);

  const animationCache = useMemo(() => {
    const byPhase = {};
    PHASES.forEach((phase) => {
      const bundle = animationCacheRef.current.get(getCacheKey(phase));
      if (bundle) byPhase[phase] = bundle;
    });
    return byPhase;
  }, [getCacheKey, cacheVersion]);

  return {
    playingPhase,
    isPlaying,
    frameIdx,
    animationCache,
    liveHgaByElectrodeId,
    animationScale,
    animationTime,
    togglePlay,
    stopPlayback,
  };
}
