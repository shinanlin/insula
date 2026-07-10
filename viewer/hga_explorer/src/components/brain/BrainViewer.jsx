import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';
import { Info, RotateCcw } from 'lucide-react';
import { PHASE_LABELS } from '../../constants/phases.js';
import {
  BRAIN_MESH_URL,
  BRAIN_CAMERA,
  BRAIN_HEMISPHERE_OPTIONS,
  BRAIN_VIEW_OPTIONS,
  DEFAULT_BRAIN_VIEW_MODE,
  DEFAULT_ELECTRODE_BRAIN_OPACITY,
  DEFAULT_KDE_BRAIN_OPACITY,
} from '../../constants/brain.js';
import {
  buildKdeSources,
  kdeSourceHgaValues,
  kdeSourcesForInfluenceMap,
} from '../../utils/roiKdeSources.js';
import { resolveElectrodesForBrainSpace } from '../../utils/electrodeCoords.js';
import BrainSceneLighting from './BrainSceneLighting.jsx';
import BrainSceneControls from './BrainSceneControls.jsx';
import KdeColorbar from './KdeColorbar.jsx';
import FallbackBrainSphere from './FallbackBrainSphere.jsx';
import AverageBrainMesh from './AverageBrainMesh.jsx';
import SubjectBrainMesh from './SubjectBrainMesh.jsx';
import BipolarEndpoints from './BipolarEndpoints.jsx';
import BrainKdeMesh from './BrainKdeMesh.jsx';
import ElectrodePoint from './ElectrodePoint.jsx';
import ElectrodeInstances from './ElectrodeInstances.jsx';
import { resolveHgaMean } from '../../utils/hga.js';
import { electrodeMatchesHemisphere } from '../../lib/brainMesh.js';
import PanelEmptyState from '../layout/PanelEmptyState.jsx';
import BrainRenderOverlay from './BrainRenderOverlay.jsx';

export default function BrainViewer({
  electrodes,
  metadata,
  traces = null,
  brainSpace = 'template',
  nativeMeshUrl = null,
  brainSpaceOptions = [],
  onBrainSpaceChange,
  vennPhases,
  selectedTask = 'all',
  selectedLoad,
  selectedIds,
  selectedElectrodeId,
  selectedEndpoint = null,
  hoveredId,
  playingPhase,
  isPlaying,
  animationFrameIdx,
  animationTime,
  animationScale,
  animationFrames,
  liveHgaByElectrodeId,
  selectionEmpty = null,
  awaitingKdeRender = false,
  kdeFrameCacheStatus = { ready: true, progress: 1 },
  kdePreRenderToken = 0,
  onFrameCacheStatus,
  onBrainViewModeChange,
  onHover,
  onSelect,
  onSelectEndpoint,
}) {
  const hgaScale = metadata?.hga_size_scale;
  const significanceWindows = metadata?.significance_windows;
  const activeMeshUrl = nativeMeshUrl || BRAIN_MESH_URL;
  const resolvedElectrodes = useMemo(
    () => resolveElectrodesForBrainSpace(electrodes, brainSpace),
    [electrodes, brainSpace],
  );
  const selectedElectrode = useMemo(
    () => resolvedElectrodes.find((electrode) => electrode.id === selectedElectrodeId) ?? null,
    [resolvedElectrodes, selectedElectrodeId],
  );
  const [brainAssetOk, setBrainAssetOk] = useState(null);
  const [showAllElectrodes, setShowAllElectrodes] = useState(false);
  const [brainOpacity, setBrainOpacity] = useState(DEFAULT_KDE_BRAIN_OPACITY);
  const [brainHemisphere, setBrainHemisphere] = useState('both');
  const [colorByFunctional, setColorByFunctional] = useState(true);
  const [brainViewMode, setBrainViewMode] = useState(DEFAULT_BRAIN_VIEW_MODE);
  const [cameraResetToken, setCameraResetToken] = useState(0);
  const [kdeAutoRange, setKdeAutoRange] = useState({ vmin: 0, vmax: 1, hasData: false });
  const [kdeVmaxOverride, setKdeVmaxOverride] = useState(null);

  const effectiveVmax = kdeVmaxOverride ?? kdeAutoRange.vmax;
  const effectiveRange = kdeAutoRange.hasData
    ? { ...kdeAutoRange, vmax: effectiveVmax }
    : kdeAutoRange;

  const handleDensityRange = useCallback((range) => {
    setKdeAutoRange(range);
  }, []);

  const handleFrameCacheStatus = useCallback((status) => {
    onFrameCacheStatus?.(status);
  }, [onFrameCacheStatus]);

  useEffect(() => {
    onBrainViewModeChange?.(brainViewMode);
  }, [brainViewMode, onBrainViewModeChange]);

  const visibleElectrodes = useMemo(() => {
    const base = showAllElectrodes
      ? resolvedElectrodes
      : resolvedElectrodes.filter((electrode) => selectedIds.has(electrode.id));
    return base.filter((electrode) => electrodeMatchesHemisphere(electrode, brainHemisphere));
  }, [resolvedElectrodes, selectedIds, showAllElectrodes, brainHemisphere]);

  const visibleElectrodesKey = useMemo(
    () => visibleElectrodes.map((electrode) => electrode.id).join('|'),
    [visibleElectrodes],
  );

  useEffect(() => {
    setKdeVmaxOverride(null);
  }, [visibleElectrodesKey, selectedLoad, playingPhase, brainHemisphere, brainViewMode]);

  const kdeHgaValues = useMemo(
    () => visibleElectrodes.map((electrode) => {
      if (playingPhase && liveHgaByElectrodeId?.[electrode.id] != null) {
        return liveHgaByElectrodeId[electrode.id];
      }
      return resolveHgaMean(electrode, selectedLoad, traces, significanceWindows);
    }),
    [visibleElectrodesKey, playingPhase, liveHgaByElectrodeId, selectedLoad, animationTime, traces, significanceWindows],
  );

  const kdeSources = useMemo(
    () => buildKdeSources(visibleElectrodes, kdeHgaValues),
    [visibleElectrodesKey, kdeHgaValues],
  );

  const kdeLayout = useMemo(() => {
    const placeholderHga = visibleElectrodes.map(() => 0);
    const sources = buildKdeSources(visibleElectrodes, placeholderHga);
    return {
      mode: sources.mode,
      sourceIds: sources.sources.map((source) => source.id),
      influencePoints: kdeSourcesForInfluenceMap(sources.sources),
    };
  }, [visibleElectrodesKey, visibleElectrodes]);

  const kdeSourceHga = useMemo(
    () => kdeSourceHgaValues(kdeSources.sources),
    [kdeSources],
  );

  const kdeFrameHgaValues = useMemo(() => {
    if (!playingPhase || !animationFrames?.length) return null;
    if (kdeLayout.mode === 'electrode') {
      return animationFrames.map((frame) => kdeLayout.sourceIds.map(
        (sourceId) => frame.hgaByElectrodeId?.[sourceId] ?? 0,
      ));
    }
    return animationFrames.map((frame) => kdeLayout.sourceIds.map((sourceId) => {
      const roi = sourceId.startsWith('roi:') ? sourceId.slice(4) : null;
      if (!roi) return 0;
      const members = visibleElectrodes.filter((electrode) => electrode.roi === roi);
      const values = members
        .map((electrode) => frame.hgaByElectrodeId?.[electrode.id])
        .filter((value) => value != null && Number.isFinite(value));
      if (!values.length) return 0;
      return values.reduce((sum, value) => sum + value, 0) / values.length;
    }));
  }, [playingPhase, animationFrames, kdeLayout, visibleElectrodesKey, visibleElectrodes]);

  const useInstancedElectrodes = visibleElectrodes.length > 80;
  const highlightedElectrodes = useMemo(
    () => (useInstancedElectrodes
      ? visibleElectrodes.filter(
        (electrode) => electrode.id === selectedElectrodeId || electrode.id === hoveredId,
      )
      : []),
    [useInstancedElectrodes, visibleElectrodes, selectedElectrodeId, hoveredId, visibleElectrodesKey],
  );
  const instancedElectrodes = useMemo(
    () => (useInstancedElectrodes
      ? visibleElectrodes.filter(
        (electrode) => electrode.id !== selectedElectrodeId && electrode.id !== hoveredId,
      )
      : []),
    [useInstancedElectrodes, visibleElectrodes, selectedElectrodeId, hoveredId, visibleElectrodesKey],
  );

  useEffect(() => {
    setBrainOpacity(
      brainViewMode === 'kde' ? DEFAULT_KDE_BRAIN_OPACITY : DEFAULT_ELECTRODE_BRAIN_OPACITY,
    );
  }, [brainViewMode]);

  useEffect(() => {
    if (brainViewMode === 'kde' && brainAssetOk === false) {
      setBrainViewMode('electrodes');
    }
  }, [brainViewMode, brainAssetOk]);

  useEffect(() => {
    let cancelled = false;
    fetch(activeMeshUrl, { method: 'HEAD' })
      .then((response) => {
        if (!cancelled) {
          setBrainAssetOk(response.ok);
          if (!response.ok) {
            console.warn(`Brain mesh not found at ${activeMeshUrl}; using fallback sphere.`);
          }
        }
      })
      .catch(() => {
        if (!cancelled) {
          setBrainAssetOk(false);
          console.warn(`Brain mesh failed to load from ${activeMeshUrl}; using fallback sphere.`);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [activeMeshUrl]);

  return (
    <div className="brain-canvas">
      <Canvas
        camera={BRAIN_CAMERA}
        gl={{ localClippingEnabled: true, alpha: false }}
        onCreated={({ gl }) => {
          gl.toneMapping = THREE.NoToneMapping;
        }}
      >
        <BrainSceneLighting mneStyle />
        {brainAssetOk === true && brainViewMode === 'electrodes' && (
          nativeMeshUrl
            ? <SubjectBrainMesh key={nativeMeshUrl} meshUrl={nativeMeshUrl} opacity={brainOpacity} hemisphereView={brainHemisphere} useLitCortex />
            : <AverageBrainMesh opacity={brainOpacity} hemisphereView={brainHemisphere} useLitCortex />
        )}
        {brainAssetOk === true && brainViewMode === 'kde' && (
          <BrainKdeMesh
            opacity={brainOpacity}
            hemisphereView={brainHemisphere}
            influencePoints={kdeLayout.influencePoints}
            hgaValues={kdeSourceHga}
            fixedHgaMax={playingPhase ? animationScale?.vmax : null}
            frameHgaValues={kdeFrameHgaValues}
            frameIndex={animationFrameIdx}
            kdePreRenderToken={kdePreRenderToken}
            densityVmaxOverride={kdeVmaxOverride}
            onDensityRange={handleDensityRange}
            onFrameCacheStatus={handleFrameCacheStatus}
          />
        )}
        {brainAssetOk === false && (
          <FallbackBrainSphere opacity={brainOpacity} hemisphereView={brainHemisphere} />
        )}
        {brainViewMode === 'electrodes' && useInstancedElectrodes && (
          <ElectrodeInstances
            electrodes={instancedElectrodes}
            vennPhases={vennPhases}
            selectedLoad={selectedLoad}
            traces={traces}
            significanceWindows={significanceWindows}
            hgaScale={hgaScale}
            animationScale={animationScale}
            liveHgaByElectrodeId={liveHgaByElectrodeId}
            isAnimating={Boolean(playingPhase)}
            selectedIds={selectedIds}
            selectedElectrodeId={selectedElectrodeId}
            hoveredId={hoveredId}
            colorByFunctional={colorByFunctional}
            onHover={onHover}
            onSelect={onSelect}
          />
        )}
        {brainViewMode === 'electrodes' && (useInstancedElectrodes ? highlightedElectrodes : visibleElectrodes).map((electrode) => (
          <ElectrodePoint
            key={electrode.id}
            electrode={electrode}
            vennPhases={vennPhases}
            selectedLoad={selectedLoad}
            traces={traces}
            significanceWindows={significanceWindows}
            hgaScale={hgaScale}
            animationScale={animationScale}
            liveHga={liveHgaByElectrodeId?.[electrode.id]}
            isAnimating={Boolean(playingPhase)}
            selected={selectedIds.has(electrode.id)}
            dimmed={showAllElectrodes && !selectedIds.has(electrode.id)}
            active={selectedElectrodeId === electrode.id}
            hovered={hoveredId === electrode.id}
            onHover={onHover}
            onSelect={onSelect}
            colorByFunctional={colorByFunctional}
          />
        ))}
        {selectedElectrode && (
          <BipolarEndpoints
            electrode={selectedElectrode}
            brainSpace={brainSpace}
            selectedEndpoint={selectedEndpoint}
            onSelectEndpoint={onSelectEndpoint}
          />
        )}
        <BrainSceneControls resetToken={cameraResetToken} />
      </Canvas>
      <div className="brain-toolbar">
        <div className="brain-controls-row" data-tour="brain-controls">
          <div className="brain-control-pill" data-tour="brain-space-toggle">
            <span className="brain-control-label">Brain</span>
            {brainSpaceOptions.map((option) => (
              <button
                key={option.id}
                type="button"
                className={`brain-chip${brainSpace === option.id ? ' active' : ''}`}
                disabled={option.disabled}
                title={option.title}
                onClick={() => onBrainSpaceChange?.(option.id)}
              >
                {option.label}
              </button>
            ))}
          </div>
          <div className="brain-control-pill">
            <span className="brain-control-label">Hemisphere</span>
            {BRAIN_HEMISPHERE_OPTIONS.map((option) => (
              <button
                key={option.id}
                type="button"
                className={brainHemisphere === option.id ? 'brain-chip active' : 'brain-chip'}
                onClick={() => setBrainHemisphere(option.id)}
              >
                {option.label}
              </button>
            ))}
          </div>
          <div className="brain-control-pill">
            <span className="brain-control-label">View</span>
            {BRAIN_VIEW_OPTIONS.map((option) => {
              const disabled = option.id === 'kde' && brainAssetOk !== true;
              return (
                <button
                  key={option.id}
                  type="button"
                  className={`brain-chip${brainViewMode === option.id ? ' active' : ''}`}
                  disabled={disabled}
                  title={disabled ? 'KDE projection requires the average pial mesh (cvs_avg35_pial.glb)' : undefined}
                  onClick={() => setBrainViewMode(option.id)}
                >
                  {option.label}
                </button>
              );
            })}
          </div>
          {brainViewMode === 'electrodes' && (
            <div className="brain-control-pill">
              <button
                type="button"
                className={colorByFunctional ? 'brain-chip active' : 'brain-chip'}
                onClick={() => setColorByFunctional((current) => !current)}
                title={colorByFunctional ? 'Color electrodes by functional phase' : 'Use uniform electrode color'}
              >
                Functional
              </button>
            </div>
          )}
          <button
            type="button"
            className="brain-chip brain-reset-btn"
            onClick={() => setCameraResetToken((token) => token + 1)}
            title="Reset camera to default view"
          >
            <RotateCcw size={13} />
            Reset view
          </button>
          <button
            type="button"
            className={`brain-chip${showAllElectrodes ? ' active' : ''}`}
            onClick={() => setShowAllElectrodes((current) => !current)}
            title={showAllElectrodes ? 'Show selected electrodes only' : 'Show all electrodes for context'}
          >
            {showAllElectrodes ? 'Selected only' : 'Show all'}
          </button>
        </div>
        <label className="brain-opacity-control">
          <span className="brain-opacity-label">Opacity</span>
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={Math.round(brainOpacity * 100)}
            onChange={(event) => setBrainOpacity(Number(event.target.value) / 100)}
            aria-valuetext={`${Math.round(brainOpacity * 100)} percent`}
          />
          <span className="brain-opacity-value">{Math.round(brainOpacity * 100)}%</span>
        </label>
      </div>
      {brainViewMode === 'kde' && (
        <KdeColorbar
          range={effectiveRange}
          vmaxValue={effectiveVmax}
          disabled={!kdeAutoRange.hasData}
          onVmaxChange={setKdeVmaxOverride}
        />
      )}
      <button
        type="button"
        className="brain-help-btn"
        title="Drag to rotate · scroll to zoom · click an electrode for details. Animation updates KDE or sphere size for the active phase."
        aria-label="Brain view help"
      >
        <Info size={14} />
      </button>
      {brainViewMode === 'kde' && kdeSources.mode === 'roi' && (
        <div className="brain-status-pill brain-kde-mode-pill">{kdeSources.label}</div>
      )}
      {playingPhase && (animationTime != null || awaitingKdeRender) && (
        <div className="brain-status-pill">
          {PHASE_LABELS[playingPhase]}
          {animationTime != null && ` · t = ${animationTime.toFixed(2)}s`}
          {awaitingKdeRender && brainViewMode === 'kde' && ' · preparing map'}
          {!awaitingKdeRender && isPlaying && ' · playing'}
          {!awaitingKdeRender && !isPlaying && animationTime != null && ' · paused'}
        </div>
      )}
      {awaitingKdeRender && brainViewMode === 'kde' && (
        <BrainRenderOverlay
          active
          progress={kdeFrameCacheStatus.progress}
          phaseLabel={playingPhase ? PHASE_LABELS[playingPhase] : null}
        />
      )}
      <PanelEmptyState emptyState={selectionEmpty} className="brain-empty-state" />
    </div>
  );
}
