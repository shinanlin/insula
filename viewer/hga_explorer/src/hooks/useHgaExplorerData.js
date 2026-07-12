import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  atlasLabel,
  isMultiAtlasManifest,
  loadAtlasElectrodes,
  loadTracesForSubjects,
  loadViewerBootstrap,
  resolveDefaultAtlas,
} from '../data/hgaExplorerStore.js';

const BOOTSTRAP_LOAD_WEIGHT = 0.15;
const TRACES_LOAD_WEIGHT = 0.85;

const STAGE_LABELS = {
  manifest: 'Loading viewer metadata…',
  electrodes: 'Loading electrode catalog…',
  traces: 'Loading HGA traces…',
};

function usesSplitTraces(layout) {
  return layout === 'split' || layout === 'split-multi-atlas';
}

export default function useHgaExplorerData() {
  const [bootstrap, setBootstrap] = useState(null);
  const [bootstrapLoading, setBootstrapLoading] = useState(true);
  const [bootstrapProgress, setBootstrapProgress] = useState({
    stage: 'manifest',
    completed: 0,
    total: 2,
  });
  const [electrodes, setElectrodes] = useState([]);
  const [regions, setRegions] = useState([]);
  const [selectedAtlas, setSelectedAtlasState] = useState('hammers');
  const [atlasSwitching, setAtlasSwitching] = useState(false);
  const [traces, setTraces] = useState({});
  const [tracesLoading, setTracesLoading] = useState(false);
  const [tracesLoadStarted, setTracesLoadStarted] = useState(false);
  const [tracesLoadProgress, setTracesLoadProgress] = useState({
    completed: 0,
    total: 0,
    progress: 0,
  });
  const [initialLoadComplete, setInitialLoadComplete] = useState(false);
  const [loadError, setLoadError] = useState(null);
  const [selectedSubjects, setSelectedSubjects] = useState(() => new Set());

  useEffect(() => {
    let cancelled = false;
    setBootstrapLoading(true);
    setLoadError(null);
    setBootstrapProgress({ stage: 'manifest', completed: 0, total: 2 });

    loadViewerBootstrap({
      onProgress: (status) => {
        if (!cancelled) setBootstrapProgress(status);
      },
    })
      .then((payload) => {
        if (cancelled) return;
        setBootstrap(payload);
        setElectrodes(payload.electrodes || []);
        setRegions(payload.regions || []);
        setSelectedAtlasState(payload.selectedAtlas || resolveDefaultAtlas(payload.manifest));
      })
      .catch((error) => {
        console.error('Failed to load HGA explorer data', error);
        if (!cancelled) setLoadError(error?.message || 'Failed to load viewer data');
      })
      .finally(() => {
        if (!cancelled) setBootstrapLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const manifest = bootstrap?.manifest ?? null;
  const layout = bootstrap?.layout ?? null;
  const dataSource = bootstrap?.dataSource ?? null;

  const availableAtlases = useMemo(() => {
    if (bootstrap?.availableAtlases?.length) return bootstrap.availableAtlases;
    if (manifest && isMultiAtlasManifest(manifest)) {
      return manifest.atlases || Object.keys(manifest.atlas || {});
    }
    return ['aparc2009s'];
  }, [bootstrap, manifest]);

  const atlasOptions = useMemo(
    () => availableAtlases.map((id) => ({
      id,
      label: atlasLabel(manifest, id),
    })),
    [availableAtlases, manifest],
  );

  const metadata = useMemo(() => {
    if (!bootstrap?.metadata) return null;
    const atlasMeta = isMultiAtlasManifest(manifest)
      ? (manifest?.atlas?.[selectedAtlas]?.metadata || {})
      : {};
    return {
      ...bootstrap.metadata,
      ...atlasMeta,
      atlas: selectedAtlas,
      n_electrodes: electrodes.length,
    };
  }, [bootstrap, manifest, selectedAtlas, electrodes.length]);

  const switchAtlas = useCallback(async (atlasId) => {
    if (!manifest || atlasId === selectedAtlas) return;
    if (!isMultiAtlasManifest(manifest)) return;

    setAtlasSwitching(true);
    setLoadError(null);
    try {
      const payload = await loadAtlasElectrodes(manifest, atlasId);
      setElectrodes(payload.electrodes);
      setRegions(payload.regions);
      setSelectedAtlasState(atlasId);
    } catch (error) {
      console.error('Failed to switch atlas', error);
      setLoadError(error?.message || 'Failed to switch atlas');
    } finally {
      setAtlasSwitching(false);
    }
  }, [manifest, selectedAtlas]);

  const electrodeById = useMemo(() => {
    const map = new Map();
    electrodes.forEach((electrode) => map.set(electrode.id, electrode));
    return map;
  }, [electrodes]);

  const availableSubjects = useMemo(() => {
    const fromMeta = metadata?.subjects ?? bootstrap?.metadata?.subjects;
    if (fromMeta?.length) return [...fromMeta].sort();
    return [...new Set(electrodes.map((electrode) => electrode.subject))].sort();
  }, [metadata, bootstrap, electrodes]);

  const availableSubjectsKey = availableSubjects.join('|');

  useEffect(() => {
    if (!availableSubjects.length) return;
    setSelectedSubjects(new Set(availableSubjects));
  }, [availableSubjectsKey]);

  const selectedSubjectsKey = useMemo(
    () => [...selectedSubjects].sort().join('|'),
    [selectedSubjects, availableSubjectsKey],
  );

  useEffect(() => {
    if (!bootstrap) return undefined;
    let cancelled = false;

    if (!usesSplitTraces(bootstrap.layout)) {
      setTraces(bootstrap.traces || {});
      setTracesLoading(false);
      setTracesLoadProgress({ completed: 0, total: 0, progress: 0 });
      return undefined;
    }

    const subjects = [...selectedSubjects];
    if (!subjects.length) {
      setTraces({});
      setTracesLoading(false);
      setTracesLoadProgress({ completed: 0, total: 0, progress: 0 });
      return undefined;
    }

    setTracesLoadStarted(true);
    setTracesLoading(true);
    setTracesLoadProgress({ completed: 0, total: subjects.length, progress: 0 });
    loadTracesForSubjects(manifest, subjects, {}, (status) => {
      if (!cancelled) setTracesLoadProgress(status);
    })
      .then((merged) => {
        if (!cancelled) setTraces(merged);
      })
      .catch((error) => {
        console.error('Failed to load subject traces', error);
        if (!cancelled) {
          setTraces({});
          setLoadError(error?.message || 'Failed to load subject traces');
        }
      })
      .finally(() => {
        if (!cancelled) {
          setTracesLoading(false);
          setTracesLoadProgress((current) => (
            current.total > 0
              ? { completed: current.total, total: current.total, progress: 1 }
              : { completed: 0, total: 0, progress: 0 }
          ));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [bootstrap, manifest, selectedSubjectsKey]);

  useEffect(() => {
    if (initialLoadComplete) return undefined;
    if (bootstrapLoading || !bootstrap) return undefined;
    if (!usesSplitTraces(bootstrap.layout)) {
      setInitialLoadComplete(true);
      return undefined;
    }
    if (tracesLoadStarted && !tracesLoading) {
      setInitialLoadComplete(true);
    }
    return undefined;
  }, [
    initialLoadComplete,
    bootstrapLoading,
    bootstrap,
    tracesLoadStarted,
    tracesLoading,
  ]);

  const isInitialLoading = !initialLoadComplete;

  const initialLoadStage = useMemo(() => {
    if (initialLoadComplete) return 'ready';
    if (bootstrapLoading || !bootstrap) return bootstrapProgress.stage;
    if (usesSplitTraces(layout)) return 'traces';
    return 'electrodes';
  }, [
    initialLoadComplete,
    bootstrapLoading,
    bootstrap,
    bootstrapProgress.stage,
    layout,
  ]);

  const initialLoadProgress = useMemo(() => {
    if (initialLoadComplete) return 1;
    if (bootstrapLoading || !bootstrap) {
      const bootstrapFraction = bootstrapProgress.total > 0
        ? bootstrapProgress.completed / bootstrapProgress.total
        : 0;
      return bootstrapFraction * BOOTSTRAP_LOAD_WEIGHT;
    }
    if (!usesSplitTraces(layout)) return 1;
    return BOOTSTRAP_LOAD_WEIGHT + tracesLoadProgress.progress * TRACES_LOAD_WEIGHT;
  }, [
    initialLoadComplete,
    bootstrapLoading,
    bootstrap,
    bootstrapProgress,
    layout,
    tracesLoadProgress.progress,
  ]);

  const initialLoadLabel = STAGE_LABELS[initialLoadStage] ?? STAGE_LABELS.traces;

  const subjectFilteredElectrodes = useMemo(
    () => electrodes.filter((electrode) => selectedSubjects.has(electrode.subject)),
    [electrodes, selectedSubjectsKey],
  );

  const toggleSubject = (subject) => {
    setSelectedSubjects((current) => {
      const next = new Set(current);
      if (next.has(subject)) next.delete(subject);
      else next.add(subject);
      return next;
    });
  };

  const selectAllSubjects = () => {
    setSelectedSubjects(new Set(availableSubjects));
  };

  const deselectAllSubjects = () => {
    setSelectedSubjects(new Set());
  };

  const data = useMemo(
    () => (bootstrap
      ? {
        metadata,
        electrodes,
        regions,
        traces,
        manifest,
        layout,
        dataSource,
        selectedAtlas,
        availableAtlases,
        atlasOptions,
      }
      : null),
    [
      bootstrap,
      metadata,
      electrodes,
      regions,
      traces,
      manifest,
      layout,
      dataSource,
      selectedAtlas,
      availableAtlases,
      atlasOptions,
    ],
  );

  return {
    data,
    isInitialLoading,
    initialLoadComplete,
    initialLoadProgress,
    initialLoadStage,
    initialLoadLabel,
    loadError,
    bootstrapProgress,
    tracesLoading,
    tracesLoadProgress,
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
    availableAtlases,
    atlasOptions,
    atlasSwitching,
  };
}
