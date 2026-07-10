import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { PHASES } from '../constants/phases.js';
import { ANIM_STEP_MS } from '../constants/animation.js';
import { fetchAndMergePhaseAnimation } from '../utils/mergeAnimationClient.js';
import { bundleHasPlayableFrames } from '../utils/animationBundle.js';
import { buildAnimationCacheKey } from '../utils/animationCacheKey.js';

export default function useAnimationPlayback({
  manifest,
  layout,
  tableElectrodes,
  tableElectrodesKey,
  traces,
  selectedLoad,
  selectedRegionIds,
  vennPhases,
  availableSubjectsKey,
  selectedSubjects,
  kdeRenderRequired = false,
  kdeFrameCacheStatus = { ready: true, progress: 1 },
  onKdeRenderStart,
}) {
  const [playingPhase, setPlayingPhase] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [animationFrameIdx, setAnimationFrameIdx] = useState(0);
  const [cacheVersion, setCacheVersion] = useState(0);
  const [animationLoadingPhase, setAnimationLoadingPhase] = useState(null);
  const [animationLoadProgress, setAnimationLoadProgress] = useState({
    completed: 0,
    total: 0,
    progress: 0,
    phase: null,
  });
  const [awaitingKdeRender, setAwaitingKdeRender] = useState(false);
  const awaitingPhaseRef = useRef(null);
  const animationCacheRef = useRef(new Map());

  const subjectsKey = useMemo(
    () => [...selectedSubjects].sort().join('|'),
    [selectedSubjects],
  );

  const electrodeFilterSet = useMemo(
    () => new Set(tableElectrodes.map((electrode) => electrode.id)),
    [tableElectrodesKey],
  );

  const getCacheKey = useCallback(
    (phase) => buildAnimationCacheKey(phase, selectedLoad, subjectsKey, tableElectrodesKey),
    [selectedLoad, subjectsKey, tableElectrodesKey],
  );

  const getCachedBundle = useCallback((phase) => {
    return animationCacheRef.current.get(getCacheKey(phase)) ?? null;
  }, [getCacheKey, cacheVersion]);

  const setCachedBundle = useCallback((phase, bundle) => {
    animationCacheRef.current.set(getCacheKey(phase), bundle);
    setCacheVersion((version) => version + 1);
  }, [getCacheKey]);

  const animationCache = useMemo(() => {
    const byPhase = {};
    PHASES.forEach((phase) => {
      const bundle = animationCacheRef.current.get(getCacheKey(phase));
      if (bundle) byPhase[phase] = bundle;
    });
    return byPhase;
  }, [getCacheKey, cacheVersion]);

  const loadPhaseAnimation = useCallback(async (phase) => {
    const cached = getCachedBundle(phase);
    if (bundleHasPlayableFrames(cached)) {
      return cached;
    }

    if (layout === 'split' && manifest) {
      const subjects = [...selectedSubjects];
      setAnimationLoadingPhase(phase);
      setAnimationLoadProgress({
        completed: 0,
        total: subjects.length,
        progress: 0,
        phase,
      });
      try {
        const merged = await fetchAndMergePhaseAnimation({
          manifest,
          subjects,
          phase,
          selectedLoad,
          electrodeFilterSet,
          onProgress: (status) => {
            setAnimationLoadProgress({
              completed: status.completed,
              total: status.total,
              progress: status.progress,
              phase,
            });
          },
        });
        if (merged && bundleHasPlayableFrames(merged)) {
          setCachedBundle(phase, merged);
        }
        return merged;
      } finally {
        setAnimationLoadingPhase((current) => (current === phase ? null : current));
        setAnimationLoadProgress((current) => (
          current.phase === phase
            ? { completed: 0, total: 0, progress: 0, phase: null }
            : current
        ));
      }
    }

    const { buildSlidingWindowFrames } = await import('../utils/animationFrames.js');
    await new Promise((resolve) => window.setTimeout(resolve, 0));
    const bundle = buildSlidingWindowFrames(tableElectrodes, traces, phase, selectedLoad, {
      allowMock: layout === 'mock',
    });
    if (bundleHasPlayableFrames(bundle)) {
      setCachedBundle(phase, bundle);
    }
    return bundle;
  }, [
    layout,
    manifest,
    selectedSubjects,
    selectedLoad,
    electrodeFilterSet,
    tableElectrodes,
    traces,
    getCachedBundle,
    setCachedBundle,
  ]);

  const activeAnimation = playingPhase ? getCachedBundle(playingPhase) : null;
  const liveHgaByElectrodeId = activeAnimation?.frames?.[animationFrameIdx]?.hgaByElectrodeId ?? null;
  const animationScale = playingPhase ? activeAnimation?.scale ?? null : null;
  const animationTime = activeAnimation?.frames?.[animationFrameIdx]?.time ?? null;

  useEffect(() => {
    setIsPlaying(false);
    setPlayingPhase(null);
    setAnimationFrameIdx(0);
    setAwaitingKdeRender(false);
    awaitingPhaseRef.current = null;
  }, [
    tableElectrodesKey,
    selectedLoad,
    selectedRegionIds.join('|'),
    vennPhases.join('|'),
    availableSubjectsKey,
    subjectsKey,
  ]);

  useEffect(() => {
    if (!awaitingKdeRender || !playingPhase) return undefined;
    if (awaitingPhaseRef.current !== playingPhase) return undefined;

    const canStart = !kdeRenderRequired || kdeFrameCacheStatus.ready;
    if (!canStart) return undefined;

    setAwaitingKdeRender(false);
    awaitingPhaseRef.current = null;
    setIsPlaying(true);
    return undefined;
  }, [
    awaitingKdeRender,
    playingPhase,
    kdeRenderRequired,
    kdeFrameCacheStatus.ready,
  ]);

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
        setAnimationFrameIdx((current) => {
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

  const beginPlayback = useCallback((phase) => {
    setPlayingPhase(phase);
    setAnimationFrameIdx(0);
    setIsPlaying(false);

    if (kdeRenderRequired) {
      onKdeRenderStart?.();
      awaitingPhaseRef.current = phase;
      setAwaitingKdeRender(true);
      return;
    }

    awaitingPhaseRef.current = null;
    setAwaitingKdeRender(false);
    setIsPlaying(true);
  }, [kdeRenderRequired, onKdeRenderStart]);

  const handleTogglePlay = useCallback(async (phase) => {
    if (animationLoadingPhase === phase) return;

    if (playingPhase === phase && awaitingKdeRender) {
      setAwaitingKdeRender(false);
      awaitingPhaseRef.current = null;
      setPlayingPhase(null);
      setAnimationFrameIdx(0);
      return;
    }

    let bundle = getCachedBundle(phase);
    if (!bundleHasPlayableFrames(bundle)) {
      bundle = await loadPhaseAnimation(phase);
    }
    if (!bundleHasPlayableFrames(bundle)) return;

    if (playingPhase === phase && isPlaying) {
      setIsPlaying(false);
      return;
    }

    beginPlayback(phase);
  }, [
    animationLoadingPhase,
    playingPhase,
    isPlaying,
    awaitingKdeRender,
    getCachedBundle,
    loadPhaseAnimation,
    beginPlayback,
  ]);

  const handleSeek = useCallback(async (phase, frameIdx) => {
    if (animationLoadingPhase === phase) return;

    let bundle = getCachedBundle(phase);
    if (!bundleHasPlayableFrames(bundle)) {
      bundle = await loadPhaseAnimation(phase);
    }
    if (!bundleHasPlayableFrames(bundle)) return;

    const clamped = Math.max(0, Math.min(frameIdx, bundle.frames.length - 1));
    setAwaitingKdeRender(false);
    awaitingPhaseRef.current = null;
    setPlayingPhase(phase);
    setIsPlaying(false);
    setAnimationFrameIdx(clamped);
  }, [animationLoadingPhase, getCachedBundle, loadPhaseAnimation]);

  return {
    playingPhase,
    isPlaying,
    animationFrameIdx,
    animationCache,
    animationLoadingPhase,
    animationLoadProgress,
    awaitingKdeRender,
    renderProgress: kdeFrameCacheStatus.progress,
    liveHgaByElectrodeId,
    animationScale,
    animationTime,
    handleTogglePlay,
    handleSeek,
  };
}
