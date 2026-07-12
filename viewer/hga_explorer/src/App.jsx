import React, { useCallback, useMemo, useState } from 'react';
import { Activity, Brain, Filter, HelpCircle, Info } from 'lucide-react';
import { PHASES } from './constants/phases.js';
import { VENN_MAX_PHASES, VENN_MIN_PHASES } from './constants/venn.js';
import { TASK_LABELS, TASK_OPTIONS } from './constants/tasks.js';
import { DEFAULT_BRAIN_VIEW_MODE } from './constants/brain.js';
import { electrodeInInsula } from './constants/insula.js';
import useHgaExplorerData from './hooks/useHgaExplorerData.js';
import useViewSelection from './hooks/useViewSelection.js';
import useBrainSpace from './hooks/useBrainSpace.js';
import useSelectionPipeline from './hooks/useSelectionPipeline.js';
import useAnimationPlayback from './hooks/useAnimationPlayback.js';
import useOnboardingTour from './hooks/useOnboardingTour.js';
import PanelTitle from './components/layout/PanelTitle.jsx';
import VennPanel from './components/venn/VennPanel.jsx';
import BrainViewer from './components/brain/BrainViewer.jsx';
import DetailPanel from './components/detail/DetailPanel.jsx';
import WaveformPanel from './components/waveform/WaveformPanel.jsx';
import ViewerInitialLoadScreen from './components/layout/ViewerInitialLoadScreen.jsx';
import { getSelectionEmptyState } from './utils/selectionEmptyState.js';
import { filterElectrodesForTask, filterElectrodesForView } from './utils/taskFilter.js';
import { formatViewSelectionLabel } from './utils/viewSelection.js';
import { resolveElectrodesForBrainSpace } from './utils/electrodeCoords.js';
import AtlasToggle from './components/layout/AtlasToggle.jsx';
import { usesSplitTraces } from './utils/viewerLayout.js';

export default function App() {
  const [brainViewMode, setBrainViewMode] = useState(DEFAULT_BRAIN_VIEW_MODE);
  const [insulaModeActive, setInsulaModeActive] = useState(false);
  const [kdeFrameCacheStatus, setKdeFrameCacheStatus] = useState({ ready: true, progress: 1 });
  const [kdePreRenderToken, setKdePreRenderToken] = useState(0);

  const handleKdeRenderStart = useCallback(() => {
    setKdeFrameCacheStatus({ ready: false, progress: 0 });
    setKdePreRenderToken((token) => token + 1);
  }, []);

  const handleFrameCacheStatus = useCallback((status) => {
    setKdeFrameCacheStatus(status);
  }, []);

  const handleInsulaModeChange = useCallback((active) => {
    setInsulaModeActive(active);
  }, []);

  const {
    data,
    isInitialLoading,
    initialLoadProgress,
    initialLoadStage,
    initialLoadLabel,
    loadError,
    bootstrapProgress,
    tracesLoading,
    tracesLoadProgress,
    initialLoadComplete,
    electrodeById,
    availableSubjects,
    availableSubjectsKey,
    selectedSubjects,
    subjectFilteredElectrodes,
    toggleSubject,
    selectAllSubjects,
    deselectAllSubjects,
    selectedAtlas,
    switchAtlas,
    atlasOptions,
    atlasSwitching,
  } = useHgaExplorerData();

  const {
    selectedTask,
    selectedCondition,
    availableConditions,
    viewSelection,
    selectTask,
    selectCondition,
  } = useViewSelection(data?.metadata);

  const taskScopedElectrodes = useMemo(
    () => filterElectrodesForTask(
      subjectFilteredElectrodes,
      selectedTask,
      data?.traces,
      data?.metadata,
    ),
    [
      subjectFilteredElectrodes,
      selectedTask,
      data?.traces,
      data?.metadata,
    ],
  );

  const brainScopedElectrodes = useMemo(
    () => filterElectrodesForView(
      taskScopedElectrodes,
      selectedTask,
      selectedCondition,
      data?.traces,
    ),
    [
      taskScopedElectrodes,
      selectedTask,
      selectedCondition,
      data?.traces,
    ],
  );

  const {
    vennPhases,
    setVennPhases,
    vennRegions,
    selectedRegionIds,
    selectRegion,
    selectedElectrodeId,
    selectElectrode,
    selectedEndpoint,
    selectEndpoint,
    clearSelectedElectrode,
    hoveredId,
    setHoveredId,
    enabledRois,
    toggleRoi,
    enableAllRois,
    deselectAllRois,
    availableRois,
    roiFilteredIds,
    selectedRegions,
    selectedSummary,
    selectedElectrode,
    tableElectrodes,
    tableElectrodesKey,
    roiBarItems,
  } = useSelectionPipeline({
    subjectFilteredElectrodes: taskScopedElectrodes,
    electrodeById,
    selectedTask,
    resetAtlasKey: selectedAtlas,
  });

  const {
    brainSpace,
    setBrainSpace,
    brainSpaceOptions,
    nativeMeshUrl,
    singleSubject,
    forcedTemplate,
    insulaAvailableForSelection,
  } = useBrainSpace({
    selectedSubjects,
    manifest: data?.manifest,
  });

  const insulaFilter = useCallback(
    (electrode) => electrodeInInsula(electrode, selectedAtlas),
    [selectedAtlas],
  );

  const waveformElectrodes = useMemo(() => {
    if (!insulaModeActive) return tableElectrodes;
    return resolveElectrodesForBrainSpace(taskScopedElectrodes, brainSpace)
      .filter(insulaFilter);
  }, [
    insulaModeActive,
    tableElectrodes,
    tableElectrodesKey,
    taskScopedElectrodes,
    brainSpace,
    insulaFilter,
  ]);

  const animationElectrodes = useMemo(
    () => filterElectrodesForView(
      waveformElectrodes,
      selectedTask,
      selectedCondition,
      data?.traces,
    ),
    [
      waveformElectrodes,
      selectedTask,
      selectedCondition,
      data?.traces,
    ],
  );

  const animationElectrodesKey = useMemo(
    () => animationElectrodes.map((electrode) => electrode.id).join('|'),
    [animationElectrodes],
  );

  const kdeRenderRequired = brainViewMode === 'kde';

  const {
    playingPhase,
    isPlaying,
    animationFrameIdx,
    animationCache,
    animationLoadingPhase,
    animationLoadProgress,
    awaitingKdeRender,
    renderProgress,
    liveHgaByElectrodeId,
    animationScale,
    animationTime,
    handleTogglePlay,
    handleSeek,
  } = useAnimationPlayback({
    manifest: data?.manifest,
    layout: data?.layout,
    tableElectrodes: animationElectrodes,
    tableElectrodesKey: animationElectrodesKey,
    traces: data?.traces,
    selectedLoad: viewSelection,
    selectedRegionIds,
    vennPhases,
    availableSubjectsKey,
    selectedSubjects,
    kdeRenderRequired,
    kdeFrameCacheStatus,
    onKdeRenderStart: handleKdeRenderStart,
  });

  const enabledRoiCount = availableRois.filter((roi) => enabledRois.has(roi)).length;
  const selectionEmpty = useMemo(() => {
    const baseEmpty = getSelectionEmptyState({
      selectedSubjectCount: selectedSubjects.size,
      selectedRegionCount: selectedRegions.length,
      availableRoiCount: availableRois.length,
      enabledRoiCount,
      visibleElectrodeCount: insulaModeActive ? waveformElectrodes.length : tableElectrodes.length,
    });
    if (
      !baseEmpty
      && insulaModeActive
      && waveformElectrodes.length === 0
      && tableElectrodes.length > 0
    ) {
      return {
        code: 'no_insula_electrodes',
        title: 'No insula electrodes in scope',
        message: `Insula mode is on, but no insula electrodes match the current ${selectedAtlas} atlas, task, and subject filters.`,
      };
    }
    return baseEmpty;
  }, [
    selectedSubjects.size,
    selectedRegions.length,
    availableRois.length,
    enabledRoiCount,
    tableElectrodes.length,
    insulaModeActive,
    waveformElectrodes.length,
    selectedAtlas,
  ]);

  const canPlay = animationElectrodes.length > 0
    && !selectionEmpty
    && (usesSplitTraces(data?.layout) || !tracesLoading);

  const usingExport = usesSplitTraces(data?.layout) || data?.dataSource === 'results';

  if (isInitialLoading) {
    return (
      <ViewerInitialLoadScreen
        progress={initialLoadProgress}
        stage={initialLoadStage}
        stageLabel={initialLoadLabel}
        completed={
          initialLoadStage === 'traces'
            ? tracesLoadProgress.completed
            : bootstrapProgress.completed
        }
        total={
          initialLoadStage === 'traces'
            ? tracesLoadProgress.total
            : bootstrapProgress.total
        }
        error={loadError}
      />
    );
  }

  if (!data) {
    return (
      <ViewerInitialLoadScreen
        progress={0}
        stage="manifest"
        stageLabel="Loading viewer metadata…"
        error={loadError}
      />
    );
  }

  return (
    <TourHost enabled>
      {({ startTour, isTourActive }) => (
    <div className="app-shell">
      <div className="tour-welcome-anchor" data-tour="tour-welcome" aria-hidden="true" />
      <header className="topbar">
        <div>
          <div className="eyebrow">Insula iEEG</div>
          <h1><Brain size={24} /> HGA Explorer</h1>
          <p className={`data-source-badge${usingExport ? ' live' : ''}`}>
            {usingExport ? 'Live export' : 'Mock data'}
            {' · '}
            {data.metadata?.n_electrodes ?? data.electrodes.length} electrodes
            {' · '}
            {data.atlasOptions?.find((option) => option.id === selectedAtlas)?.label || selectedAtlas}
            {' · '}
            {formatViewSelectionLabel(viewSelection)}
            {forcedTemplate ? ' · template brain' : ` · ${brainSpace} brain`}
          </p>
        </div>
        <div className="topbar-controls">
          <div className="chip-group" data-tour="task-selector" aria-label="Task selector">
            <span className="chip-group-label"><Filter size={14} /> Task</span>
            {TASK_OPTIONS.map((task) => (
              <button
                key={task}
                type="button"
                className={`chip${selectedTask === task ? ' active' : ''}`}
                onClick={() => {
                  selectTask(task);
                  clearSelectedElectrode();
                }}
              >
                {TASK_LABELS[task] ?? task}
              </button>
            ))}
          </div>
          <div
            className="chip-group"
            data-tour="condition-selector"
            aria-label="Map condition selector"
            title="Controls brain map color and animation only"
          >
            <span className="chip-group-label">Map condition</span>
            {availableConditions.map((condition) => (
              <button
                key={condition}
                type="button"
                className={`chip${selectedCondition === condition ? ' active' : ''}`}
                onClick={() => {
                  selectCondition(condition);
                  clearSelectedElectrode();
                }}
              >
                {condition}
              </button>
            ))}
          </div>
          <AtlasToggle
            options={atlasOptions}
            selectedAtlas={selectedAtlas}
            onSelectAtlas={switchAtlas}
            disabled={atlasSwitching}
          />
          <button
            type="button"
            className="tour-replay-btn"
            onClick={() => startTour({ force: true })}
            title="Replay product tour"
            disabled={isTourActive}
          >
            <HelpCircle size={14} />
            Tour
          </button>
        </div>
      </header>

      <main className="dashboard">
        <aside className="panel venn-panel" data-tour="venn-selector">
          <PanelTitle icon={<Activity size={18} />} title="Phase overlap selector" />
          <VennPanel
            vennPhases={vennPhases}
            regions={vennRegions}
            availableSubjects={availableSubjects}
            selectedSubjects={selectedSubjects}
            onToggleSubject={(subject) => {
              toggleSubject(subject);
              clearSelectedElectrode();
            }}
            onSelectAllSubjects={() => {
              selectAllSubjects();
              clearSelectedElectrode();
            }}
            onDeselectAllSubjects={() => {
              deselectAllSubjects();
              clearSelectedElectrode();
            }}
            selectedRegionIds={selectedRegionIds}
            onTogglePhase={(phase) => {
              setVennPhases((current) => {
                if (current.includes(phase)) {
                  if (current.length <= VENN_MIN_PHASES) return current;
                  return current.filter((item) => item !== phase);
                }
                if (current.length >= VENN_MAX_PHASES) return current;
                return PHASES.filter((item) => current.includes(item) || item === phase);
              });
            }}
            onSelect={(id) => {
              selectRegion(id);
            }}
          />
        </aside>

        <section className="panel brain-panel">
          <PanelTitle icon={<Brain size={18} />} title="Cortical HGA map" />
          <BrainViewer
            electrodes={brainScopedElectrodes}
            metadata={data.metadata}
            traces={data.traces}
            brainSpace={brainSpace}
            nativeMeshUrl={nativeMeshUrl}
            singleSubject={singleSubject}
            insulaAvailableForSelection={insulaAvailableForSelection}
            brainSpaceOptions={brainSpaceOptions}
            onBrainSpaceChange={setBrainSpace}
            vennPhases={vennPhases}
            selectedTask={selectedTask}
            selectedLoad={viewSelection}
            selectedIds={roiFilteredIds}
            selectedElectrodeId={selectedElectrodeId}
            selectedEndpoint={selectedEndpoint}
            hoveredId={hoveredId}
            selectedAtlas={selectedAtlas}
            playingPhase={playingPhase}
            isPlaying={isPlaying}
            animationFrameIdx={animationFrameIdx}
            animationTime={animationTime}
            animationScale={animationScale}
            animationFrames={playingPhase ? animationCache[playingPhase]?.frames : null}
            liveHgaByElectrodeId={liveHgaByElectrodeId}
            selectionEmpty={selectionEmpty}
            awaitingKdeRender={awaitingKdeRender}
            kdeFrameCacheStatus={kdeFrameCacheStatus}
            kdePreRenderToken={kdePreRenderToken}
            onFrameCacheStatus={handleFrameCacheStatus}
            onBrainViewModeChange={setBrainViewMode}
            onInsulaModeChange={handleInsulaModeChange}
            onHover={setHoveredId}
            onSelect={selectElectrode}
            onSelectEndpoint={selectEndpoint}
          />
        </section>

        <aside className="panel detail-panel">
          <PanelTitle icon={<Info size={18} />} title="Selection details" />
          <DetailPanel
            summary={selectedSummary}
            selectedElectrode={selectedElectrode}
            selectedEndpoint={selectedEndpoint}
            brainSpace={brainSpace}
            selectedTask={selectedTask}
            tableElectrodes={tableElectrodes}
            roiBarItems={roiBarItems}
            availableRois={availableRois}
            enabledRois={enabledRois}
            selectionEmpty={selectionEmpty}
            onToggleRoi={toggleRoi}
            onEnableAllRois={enableAllRois}
            onDeselectAllRois={deselectAllRois}
            tracesLoading={tracesLoading}
            tracesLoadProgress={tracesLoadProgress}
          />
        </aside>
      </main>

      <section className="panel waveform-panel" data-tour="waveform-panel">
        <PanelTitle icon={<Activity size={18} />} title="Four-phase HGA time courses" />
        <WaveformPanel
          electrode={selectedElectrode}
          summary={selectedSummary}
          electrodes={waveformElectrodes}
          insulaModeActive={insulaModeActive}
          traces={data.traces || {}}
          layout={data.layout}
          tracesLoading={tracesLoading}
          tracesLoadProgress={tracesLoadProgress}
          initialLoadComplete={initialLoadComplete}
          selectedTask={selectedTask}
          metadata={data.metadata}
          animationCache={animationCache}
          animationLoadingPhase={animationLoadingPhase}
          animationLoadProgress={animationLoadProgress}
          canPlay={canPlay}
          selectionEmpty={selectionEmpty}
          playingPhase={playingPhase}
          isPlaying={isPlaying}
          awaitingKdeRender={awaitingKdeRender}
          renderProgress={renderProgress}
          animationFrameIdx={animationFrameIdx}
          onTogglePlay={handleTogglePlay}
          onSeek={handleSeek}
        />
      </section>
    </div>
      )}
    </TourHost>
  );
}

function TourHost({ children, enabled = true }) {
  const tour = useOnboardingTour({ enabled });
  return children(tour);
}
