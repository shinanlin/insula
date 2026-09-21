import React, { useCallback, useEffect, useMemo, useState } from 'react';
import BrainCanvas from './components/BrainCanvas.jsx';
import WaveformPanel from './components/WaveformPanel.jsx';
import usePhasePlayback from './hooks/usePhasePlayback.js';
import {
  CLUSTER_COLORS,
  CLUSTER_LABELS,
  CLUSTERS,
} from './constants.js';
import { PHASE_LABELS, PHASES } from './constants/waveform.js';
import { resolveInspectorTasks, taskLabel } from './constants/tasks.js';
import {
  electrodePurity,
  PURITY_BORDERLINE_MIN,
  PURITY_LABELS,
  PURITY_PURE_MIN,
  PURITY_TIERS,
} from './utils/purity.js';

async function loadJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  return res.json();
}

function downloadOverridesCsv(overrides, electrodes) {
  const byId = Object.fromEntries(electrodes.map((e) => [e.id, e]));
  const rows = [['channel', 'functional_cluster_original', 'functional_cluster_manual']];
  for (const [channel, manual] of Object.entries(overrides)) {
    const orig = byId[channel]?.functional_cluster ?? '';
    rows.push([channel, orig, manual]);
  }
  const body = rows.map((r) => r.join(',')).join('\n');
  const blob = new Blob([body], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'channel_assignments_manual.csv';
  a.click();
  URL.revokeObjectURL(a.href);
}

export default function App() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [manifest, setManifest] = useState(null);
  const [electrodes, setElectrodes] = useState([]);
  const [tracesBySubject, setTracesBySubject] = useState({});
  const [clusterFilter, setClusterFilter] = useState('all');
  const [purityFilter, setPurityFilter] = useState('all');
  const [selectedId, setSelectedId] = useState(null);
  const [hoveredId, setHoveredId] = useState(null);
  const [selectedTask, setSelectedTask] = useState(null);
  const [overrides, setOverrides] = useState({});

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [manifestData, electrodeData] = await Promise.all([
          loadJson('/data/manifest.json'),
          loadJson('/data/electrodes.json'),
        ]);
        if (cancelled) return;
        setManifest(manifestData);
        setElectrodes(electrodeData);
        const subjects = manifestData.metadata?.subjects ?? [];
        const traceEntries = await Promise.all(
          subjects.map(async (subject) => {
            const tree = await loadJson(`/data/traces/${subject}.json`);
            return [subject, tree];
          }),
        );
        if (cancelled) return;
        setTracesBySubject(Object.fromEntries(traceEntries));
        const tasks = resolveInspectorTasks(manifestData.metadata);
        setSelectedTask(tasks[0] ?? null);
        setLoading(false);
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const clusterForElectrode = useCallback(
    (electrode) => overrides[electrode.id] ?? electrode.functional_cluster,
    [overrides],
  );

  const allTasks = useMemo(
    () => resolveInspectorTasks(manifest?.metadata),
    [manifest],
  );

  const selectElectrode = useCallback((id) => {
    setSelectedId(id);
    const electrode = electrodes.find((item) => item.id === id);
    const available = (electrode?.tasks ?? []).filter((task) => allTasks.includes(task));
    if (!available.length) return;
    setSelectedTask((current) => (
      current && available.includes(current) ? current : available[0]
    ));
  }, [electrodes, allTasks]);

  const purityById = useMemo(() => {
    const map = {};
    electrodes.forEach((electrode) => {
      map[electrode.id] = electrodePurity(electrode);
    });
    return map;
  }, [electrodes]);

  const purityCounts = useMemo(() => {
    const counts = { pure: 0, borderline: 0, mixed: 0 };
    electrodes.forEach((electrode) => {
      const tier = purityById[electrode.id]?.purity;
      if (tier in counts) counts[tier] += 1;
    });
    return counts;
  }, [electrodes, purityById]);

  const visibleElectrodes = useMemo(() => {
    return electrodes.filter((electrode) => {
      if (clusterFilter !== 'all' && clusterForElectrode(electrode) !== clusterFilter) {
        return false;
      }
      if (purityFilter !== 'all' && purityById[electrode.id]?.purity !== purityFilter) {
        return false;
      }
      return true;
    });
  }, [electrodes, clusterFilter, clusterForElectrode, purityFilter, purityById]);

  const visibleElectrodesKey = useMemo(
    () => visibleElectrodes.map((electrode) => electrode.id).sort().join('|'),
    [visibleElectrodes],
  );

  const {
    playingPhase,
    isPlaying,
    liveHgaByElectrodeId,
    animationScale,
    animationTime,
    togglePlay,
    stopPlayback,
  } = usePhasePlayback({
    visibleElectrodes,
    visibleElectrodesKey,
    tracesBySubject,
    selectedTask,
  });

  const selected = useMemo(
    () => electrodes.find((e) => e.id === selectedId) ?? null,
    [electrodes, selectedId],
  );

  const electrodeTasks = useMemo(() => {
    if (!selected) return [];
    return (selected.tasks ?? []).filter((task) => allTasks.includes(task));
  }, [selected, allTasks]);

  const taskHasTraces = Boolean(selected && selectedTask && electrodeTasks.includes(selectedTask));

  const phaseTraces = useMemo(() => {
    if (!selected || !selectedTask || !taskHasTraces) return {};
    const subjectTree = tracesBySubject[selected.subject];
    return subjectTree?.[selected.id]?.[selectedTask] ?? {};
  }, [selected, selectedTask, taskHasTraces, tracesBySubject]);

  const activeCluster = selected ? clusterForElectrode(selected) : null;
  const selectedPurity = selected ? purityById[selected.id] : null;

  if (loading) return <div className="status">Loading NMF inspector data…</div>;
  if (error) return <div className="status status-error">Error: {error}</div>;

  const waveformTitle = selected
    ? `${selected.channel} · ${taskLabel(selectedTask)} · ${manifest.metadata.condition}`
    : 'Select an electrode to view waveforms';

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">Insula functional</div>
          <h1>NMF Insula Inspector</h1>
        </div>
        <div className="topbar-controls">
          <div className="load-selector">
            <span className="load-selector-label">Task</span>
            {allTasks.map((task) => {
              const unavailable = selected && !electrodeTasks.includes(task);
              return (
                <button
                  key={task}
                  type="button"
                  className={`load-chip ${selectedTask === task ? 'active' : ''}${unavailable ? ' unavailable' : ''}`}
                  onClick={() => setSelectedTask(task)}
                  title={unavailable ? 'No HGA traces for this electrode in this task' : undefined}
                >
                  {taskLabel(task)}
                </button>
              );
            })}
          </div>
          <div className="load-selector">
            <span className="load-selector-label">Component</span>
            <button
              type="button"
              className={`load-chip ${clusterFilter === 'all' ? 'active' : ''}`}
              onClick={() => setClusterFilter('all')}
            >
              all
            </button>
            {CLUSTERS.map((cluster) => (
              <button
                key={cluster}
                type="button"
                className={`load-chip ${clusterFilter === cluster ? 'active' : ''}`}
                onClick={() => setClusterFilter(cluster)}
              >
                {CLUSTER_LABELS[cluster]}
              </button>
            ))}
          </div>
          <div className="load-selector">
            <span className="load-selector-label">Purity</span>
            <button
              type="button"
              className={`load-chip ${purityFilter === 'all' ? 'active' : ''}`}
              onClick={() => setPurityFilter('all')}
              title={`margin = top − second loading · pure ≥ ${PURITY_PURE_MIN} · mixed < ${PURITY_BORDERLINE_MIN}`}
            >
              all
            </button>
            {PURITY_TIERS.map((tier) => (
              <button
                key={tier}
                type="button"
                className={`load-chip purity-chip purity-${tier}${purityFilter === tier ? ' active' : ''}`}
                onClick={() => setPurityFilter(tier)}
                title={
                  tier === 'pure'
                    ? `margin ≥ ${PURITY_PURE_MIN} (${purityCounts.pure})`
                    : tier === 'borderline'
                      ? `${PURITY_BORDERLINE_MIN} ≤ margin < ${PURITY_PURE_MIN} (${purityCounts.borderline})`
                      : `margin < ${PURITY_BORDERLINE_MIN} (${purityCounts.mixed})`
                }
              >
                {PURITY_LABELS[tier]}
                <span className="chip-count">{purityCounts[tier]}</span>
              </button>
            ))}
          </div>
          <span className="data-source-pill">
            {visibleElectrodes.length}/{manifest.metadata.n_electrodes} shown
            {manifest.metadata.k != null ? ` · k=${manifest.metadata.k}` : ''}
            {' · '}
            {manifest.metadata.condition}
          </span>
        </div>
      </header>

      <div className="dashboard">
        <aside className="panel panel-list">
          <div className="panel-title">
            Electrodes
            <span className="panel-title-meta">{visibleElectrodes.length}</span>
          </div>
          <ul className="electrode-list">
            {visibleElectrodes.map((electrode) => {
              const cluster = clusterForElectrode(electrode);
              const isOverride = overrides[electrode.id] != null;
              const { purity, margin } = purityById[electrode.id] ?? {};
              return (
                <li key={electrode.id}>
                  <button
                    type="button"
                    className={`electrode-item ${selectedId === electrode.id ? 'selected' : ''}`}
                    onClick={() => selectElectrode(electrode.id)}
                  >
                    <span className="electrode-item-main">
                      <span
                        className={`cluster-dot purity-dot-${purity || 'mixed'}`}
                        style={{ background: CLUSTER_COLORS[cluster] || '#94a3b8' }}
                      />
                      {electrode.channel}
                      <span className={`purity-badge purity-${purity}`}>
                        {PURITY_LABELS[purity] ?? purity}
                      </span>
                      {isOverride && <span className="override-badge">edited</span>}
                    </span>
                    <span className="electrode-item-roi">
                      {electrode.label}
                      {margin != null && Number.isFinite(margin)
                        ? ` · Δ${margin.toFixed(2)}`
                        : ''}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </aside>

        <main className="panel panel-brain">
          <div className="panel-title panel-title-brain">
            <span>Insula</span>
            <div className="phase-playback-controls">
              {PHASES.map((phase) => (
                <button
                  key={phase}
                  type="button"
                  className={`load-chip phase-play-chip${playingPhase === phase ? ' active' : ''}${playingPhase === phase && isPlaying ? ' playing' : ''}`}
                  onClick={() => togglePlay(phase)}
                  disabled={!selectedTask}
                  title={selectedTask ? `Play ${PHASE_LABELS[phase]} phase` : 'Select a task first'}
                >
                  {playingPhase === phase && isPlaying ? '■' : '▶'} {PHASE_LABELS[phase]}
                </button>
              ))}
              {playingPhase && animationTime != null && (
                <span className="playback-time-label">
                  t = {animationTime.toFixed(2)}s
                  {!isPlaying && ' · paused'}
                </span>
              )}
              {playingPhase && (
                <button
                  type="button"
                  className="playback-stop-btn"
                  onClick={stopPlayback}
                >
                  Stop
                </button>
              )}
            </div>
          </div>
          <div className="canvas-wrap">
            <BrainCanvas
              electrodes={visibleElectrodes}
              clusterForElectrode={clusterForElectrode}
              selectedId={selectedId}
              hoveredId={hoveredId}
              onHover={setHoveredId}
              onSelect={selectElectrode}
              isAnimating={Boolean(playingPhase)}
              animationScale={animationScale}
              liveHgaByElectrodeId={liveHgaByElectrodeId}
            />
          </div>
        </main>

        <aside className="panel panel-detail">
          <div className="panel-title">Channel</div>
          {selected ? (
            <div className="panel-body">
              <dl className="meta-grid">
                <dt>channel</dt><dd>{selected.channel}</dd>
                <dt>subject</dt><dd>{selected.subject}</dd>
                <dt>region</dt><dd>{selected.label}</dd>
                <dt>component</dt><dd>{selected.component}</dd>
                <dt>cluster</dt>
                <dd>
                  {CLUSTER_LABELS[activeCluster] ?? activeCluster}
                  {overrides[selected.id] && (
                    <span className="override-badge">
                      (was {CLUSTER_LABELS[selected.functional_cluster]})
                    </span>
                  )}
                </dd>
                <dt>purity</dt>
                <dd>
                  {selectedPurity ? (
                    <span className={`purity-badge purity-${selectedPurity.purity}`}>
                      {PURITY_LABELS[selectedPurity.purity]}
                    </span>
                  ) : '—'}
                </dd>
                <dt>margin</dt>
                <dd>
                  {selectedPurity?.margin != null && Number.isFinite(selectedPurity.margin)
                    ? selectedPurity.margin.toFixed(3)
                    : '—'}
                  <span className="meta-hint"> top − 2nd</span>
                </dd>
                <dt>2nd component</dt>
                <dd>
                  {selectedPurity?.secondCluster
                    ? (CLUSTER_LABELS[selectedPurity.secondCluster]
                      ?? selectedPurity.secondCluster)
                    : '—'}
                </dd>
                <dt>dominance</dt>
                <dd>
                  {selected.dominance != null && Number.isFinite(selected.dominance)
                    ? selected.dominance.toFixed(3)
                    : '—'}
                </dd>
              </dl>
              {selected.loadings && Object.keys(selected.loadings).length > 0 && (
                <div className="detail-section">
                  <div className="detail-label">NMF loadings</div>
                  <ul className="loading-bars">
                    {CLUSTERS.map((cluster) => {
                      const value = selected.loadings[cluster];
                      if (value == null || !Number.isFinite(value)) return null;
                      const max = Math.max(
                        ...CLUSTERS.map((c) => selected.loadings[c] ?? 0),
                        1e-9,
                      );
                      return (
                        <li key={cluster}>
                          <span
                            className="cluster-dot"
                            style={{ background: CLUSTER_COLORS[cluster] }}
                          />
                          <span className="loading-name">
                            {CLUSTER_LABELS[cluster]}
                          </span>
                          <span className="loading-track">
                            <span
                              className="loading-fill"
                              style={{
                                width: `${(100 * value) / max}%`,
                                background: CLUSTER_COLORS[cluster],
                              }}
                            />
                          </span>
                          <span className="loading-value">{value.toFixed(3)}</span>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
              <div className="detail-section">
                <div className="detail-label">Reassign cluster</div>
                <select
                  className="detail-select"
                  value={activeCluster}
                  onChange={(e) => {
                    const next = e.target.value;
                    setOverrides((prev) => {
                      if (next === selected.functional_cluster) {
                        const { [selected.id]: _, ...rest } = prev;
                        return rest;
                      }
                      return { ...prev, [selected.id]: next };
                    });
                  }}
                >
                  {CLUSTERS.map((c) => (
                    <option key={c} value={c}>{CLUSTER_LABELS[c]}</option>
                  ))}
                </select>
                {Object.keys(overrides).length > 0 && (
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={() => downloadOverridesCsv(overrides, electrodes)}
                  >
                    Download overrides CSV ({Object.keys(overrides).length})
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="panel-empty">Select an electrode on the insula mesh.</div>
          )}
        </aside>
      </div>

      <section className="panel waveform-panel">
        <WaveformPanel
          title={waveformTitle}
          traces={phaseTraces}
          empty={!selected}
          taskUnavailable={Boolean(selected && selectedTask && !taskHasTraces)}
          taskUnavailableMessage={
            selected && selectedTask && !taskHasTraces
              ? `No packaged HGA for ${taskLabel(selectedTask)} on this electrode. Try another task chip.`
              : null
          }
          traceKey={selected ? `${selected.id}-${selectedTask}` : 'none'}
          playingPhase={playingPhase}
          animationTime={animationTime}
          lineColor={activeCluster ? CLUSTER_COLORS[activeCluster] : undefined}
        />
      </section>
    </div>
  );
}
