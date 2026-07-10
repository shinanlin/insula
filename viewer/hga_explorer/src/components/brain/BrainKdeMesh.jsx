import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useGLTF } from '@react-three/drei';
import {
  buildFrameVertexColors,
  buildInfluenceMap,
  extractMeshPositions,
  resolveDensityRange,
} from '../../brainKde.js';
import { BRAIN_MESH_URL, BRAIN_HEMI_SPLIT_X } from '../../constants/brain.js';
import { INSULA_GHOST_OPACITY } from '../../constants/insula.js';
import { buildKdeFrameColorsOffThread } from '../../utils/kdeFrameColorClient.js';
import {
  applyBrainMaterial,
  applyHemisphereClipping,
  applyKdeOverlayMaterial,
  applyOverlayVertexColors,
  prepareBrainWithHemispheres,
  prepareKdeOverlayBrain,
  setBrainHemisphereVisibility,
} from '../../lib/brainMesh.js';

const KDE_PRECOMPUTE_DELAY_MS = 0;

function overrideFixedDensityRange(vmaxOverride) {
  if (vmaxOverride != null && Number.isFinite(vmaxOverride) && vmaxOverride > 0) {
    return { vmin: 0, vmax: vmaxOverride, hasData: true };
  }
  return null;
}

export default function BrainKdeMesh({
  meshUrl = BRAIN_MESH_URL,
  opacity,
  hemisphereView,
  insulaMode = false,
  insulaMask = null,
  insulaMaskReady = true,
  influencePoints,
  hgaValues,
  fixedHgaMax = null,
  frameHgaValues = null,
  frameIndex = 0,
  kdePreRenderToken = 0,
  densityVmaxOverride = null,
  onDensityRange,
  onFrameCacheStatus,
}) {
  const { scene } = useGLTF(meshUrl);
  const meshData = useMemo(() => extractMeshPositions(scene), [scene]);
  const pointsKey = useMemo(
    () => influencePoints.map((point) => `${point.x},${point.y},${point.z}`).join('|'),
    [influencePoints],
  );

  const influenceMap = useMemo(
    () => buildInfluenceMap(meshData.positions, influencePoints),
    [meshData.positions, meshData.vertexCount, pointsKey],
  );

  const frameHgaKey = useMemo(
    () => (frameHgaValues?.length
      ? `${frameHgaValues.length}:${frameHgaValues[0]?.length ?? 0}:${frameHgaValues.at(-1)?.length ?? 0}`
      : ''),
    [frameHgaValues],
  );

  const colorCacheRef = useRef(null);
  const fixedRangeRef = useRef(null);
  const frameIndexRef = useRef(frameIndex);
  const overlayBrainRef = useRef(null);
  const lastAppliedColorsRef = useRef(null);
  const densityRangeReportedRef = useRef(false);
  const [cacheVersion, setCacheVersion] = useState(0);

  const fixedDensityRange = useMemo(
    () => overrideFixedDensityRange(densityVmaxOverride),
    [densityVmaxOverride],
  );

  const insulaVertexMask = useMemo(() => {
    if (!insulaMode) return null;
    if (!insulaMaskReady || !insulaMask?.length) {
      return meshData.vertexCount > 0
        ? new Uint8Array(meshData.vertexCount)
        : null;
    }
    if (insulaMask.length !== meshData.vertexCount) {
      console.warn(
        `Insula mask length (${insulaMask.length}) does not match mesh vertices (${meshData.vertexCount})`,
      );
      return new Uint8Array(meshData.vertexCount);
    }
    return insulaMask instanceof Uint8Array
      ? insulaMask
      : Uint8Array.from(insulaMask);
  }, [insulaMode, insulaMask, insulaMaskReady, meshData.vertexCount]);

  const baseOpacity = insulaMode ? INSULA_GHOST_OPACITY : opacity;

  const kdeOptions = useMemo(() => ({
    statsHemisphere: hemisphereView,
    maskColorsToHemisphere: false,
    insulaVertexMask,
  }), [hemisphereView, insulaVertexMask]);

  const { baseBrain, overlayBrain } = useMemo(() => ({
    baseBrain: prepareBrainWithHemispheres(scene.clone(true), BRAIN_HEMI_SPLIT_X),
    overlayBrain: prepareKdeOverlayBrain(scene, BRAIN_HEMI_SPLIT_X),
  }), [scene]);

  overlayBrainRef.current = overlayBrain;

  useEffect(() => {
    frameIndexRef.current = frameIndex;
  }, [frameIndex]);

  useEffect(() => {
    applyBrainMaterial(baseBrain, baseOpacity, { forceSolid: !insulaMode, lit: true });
    applyHemisphereClipping(baseBrain, 'both');
    setBrainHemisphereVisibility(baseBrain, hemisphereView);
  }, [baseBrain, baseOpacity, hemisphereView, insulaMode]);

  useEffect(() => {
    applyKdeOverlayMaterial(overlayBrain, 'both');
    setBrainHemisphereVisibility(overlayBrain, hemisphereView);
  }, [overlayBrain, hemisphereView]);

  useEffect(() => {
    colorCacheRef.current = null;
    fixedRangeRef.current = null;
    densityRangeReportedRef.current = false;
  }, [densityVmaxOverride, insulaVertexMask, frameHgaKey]);

  useEffect(() => {
    if (!frameHgaValues?.length || !fixedHgaMax || !influenceMap.vertexCount) {
      colorCacheRef.current = null;
      fixedRangeRef.current = null;
      densityRangeReportedRef.current = false;
      onFrameCacheStatus?.({ ready: true, progress: 1 });
      return undefined;
    }

    let cancelled = false;
    if (!colorCacheRef.current || colorCacheRef.current.length !== frameHgaValues.length) {
      colorCacheRef.current = new Array(frameHgaValues.length);
      fixedRangeRef.current = null;
      densityRangeReportedRef.current = false;
    }

    const allFramesCached = colorCacheRef.current.every(Boolean);
    if (allFramesCached) {
      onFrameCacheStatus?.({ ready: true, progress: 1 });
      return undefined;
    }

    const hasCurrentFrame = Boolean(colorCacheRef.current[frameIndexRef.current]);
    if (!hasCurrentFrame) {
      onFrameCacheStatus?.({ ready: false, progress: 0 });
    }

    const startHandle = window.setTimeout(() => {
      if (cancelled) return;

      buildKdeFrameColorsOffThread({
        positions: meshData.positions,
        influencePoints,
        frameHgaValues,
        globalHgaMax: fixedHgaMax,
        splitX: BRAIN_HEMI_SPLIT_X,
        statsHemisphere: hemisphereView,
        insulaVertexMask,
        startIndex: frameIndexRef.current,
        fixedDensityRange,
        onFrameReady: (readyIndex, colors) => {
          if (cancelled) return;
          if (!colorCacheRef.current) {
            colorCacheRef.current = new Array(frameHgaValues.length);
          }
          colorCacheRef.current[readyIndex] = colors;
          if (readyIndex === frameIndexRef.current && overlayBrainRef.current) {
            applyOverlayVertexColors(overlayBrainRef.current, colors);
          }
        },
        onProgress: (done, total, autoRange) => {
          if (cancelled) return;
          if (autoRange?.hasData) {
            fixedRangeRef.current = autoRange;
            if (!densityRangeReportedRef.current) {
              onDensityRange?.(autoRange);
              densityRangeReportedRef.current = true;
            }
          }
          const progress = total > 0 ? done / total : 0;
          onFrameCacheStatus?.({ ready: false, progress });
          if (done > 0) {
            setCacheVersion((version) => version + 1);
          }
        },
      }).then(({ fixedRange }) => {
        if (cancelled) return;
        if (fixedRange?.hasData) {
          fixedRangeRef.current = fixedRange;
        }
        if (!densityRangeReportedRef.current && fixedRangeRef.current?.hasData) {
          onDensityRange?.(fixedRangeRef.current);
          densityRangeReportedRef.current = true;
        }
        onFrameCacheStatus?.({ ready: true, progress: 1 });
        setCacheVersion((version) => version + 1);
      }).catch((error) => {
        console.error('Failed to precompute KDE frame colors', error);
        if (!cancelled) {
          onFrameCacheStatus?.({ ready: true, progress: 1 });
        }
      });
    }, KDE_PRECOMPUTE_DELAY_MS);

    return () => {
      cancelled = true;
      window.clearTimeout(startHandle);
    };
  }, [
    influenceMap,
    frameHgaKey,
    fixedHgaMax,
    fixedDensityRange,
    pointsKey,
    hemisphereView,
    insulaVertexMask,
    meshData.positions,
    onDensityRange,
    onFrameCacheStatus,
    kdePreRenderToken,
    densityVmaxOverride,
  ]);

  useEffect(() => {
    if (!meshData.vertexCount || !influencePoints.length) {
      lastAppliedColorsRef.current = null;
      const empty = new Float32Array(meshData.vertexCount * 4);
      applyOverlayVertexColors(overlayBrain, empty);
      onDensityRange?.({ vmin: 0, vmax: 1, hasData: false });
      return;
    }

    const applyColors = (colors) => {
      applyOverlayVertexColors(overlayBrain, colors);
      lastAppliedColorsRef.current = colors;
    };

    const buildWithEffectiveRange = (values, autoFixedRange) => {
      const autoResult = buildFrameVertexColors(
        influenceMap,
        values,
        fixedHgaMax,
        meshData.positions,
        BRAIN_HEMI_SPLIT_X,
        kdeOptions,
        autoFixedRange,
      );
      const autoRange = autoFixedRange?.hasData ? autoFixedRange : autoResult.range;
      onDensityRange?.(autoRange);

      const effectiveRange = resolveDensityRange(autoRange, densityVmaxOverride);
      if (
        densityVmaxOverride != null
        && autoRange?.hasData
        && effectiveRange.vmax !== autoRange.vmax
      ) {
        return buildFrameVertexColors(
          influenceMap,
          values,
          fixedHgaMax,
          meshData.positions,
          BRAIN_HEMI_SPLIT_X,
          kdeOptions,
          effectiveRange,
        ).colors;
      }
      return autoResult.colors;
    };

    if (frameHgaValues?.length && fixedHgaMax) {
      const cachedColors = colorCacheRef.current?.[frameIndex];
      if (cachedColors) {
        applyColors(cachedColors);
        return;
      }

      applyColors(buildWithEffectiveRange(
        frameHgaValues[frameIndex] ?? hgaValues,
        fixedRangeRef.current,
      ));
      return;
    }

    applyColors(buildWithEffectiveRange(hgaValues, null));
  }, [
    overlayBrain,
    influenceMap,
    hgaValues,
    fixedHgaMax,
    frameHgaValues,
    frameIndex,
    cacheVersion,
    meshData.vertexCount,
    meshData.positions,
    hemisphereView,
    influencePoints.length,
    onDensityRange,
    densityVmaxOverride,
    kdeOptions,
  ]);

  return (
    <>
      <primitive object={baseBrain} />
      <primitive object={overlayBrain} />
    </>
  );
}

useGLTF.preload(BRAIN_MESH_URL);
