import { useEffect, useMemo, useState } from 'react';
import { DEFAULT_VENN_PHASES } from '../constants/phases.js';
import { computeVennRegions } from '../utils/vennRegions.js';
import { buildSelectionSummary } from '../utils/selectionSummary.js';

export default function useSelectionPipeline({
  subjectFilteredElectrodes,
  electrodeById,
  selectedTask = 'all',
  resetAtlasKey = null,
}) {
  const [vennPhases, setVennPhases] = useState(DEFAULT_VENN_PHASES);
  const [selectedRegionIds, setSelectedRegionIds] = useState(() => [DEFAULT_VENN_PHASES.join('_')]);
  const [selectedElectrodeId, setSelectedElectrodeId] = useState(null);
  const [selectedEndpoint, setSelectedEndpoint] = useState(null);
  const [hoveredId, setHoveredId] = useState(null);
  const [disabledRois, setDisabledRois] = useState(() => new Set());

  const vennRegions = useMemo(
    () => computeVennRegions(subjectFilteredElectrodes, vennPhases, selectedTask),
    [subjectFilteredElectrodes, vennPhases, selectedTask],
  );

  const regionsById = useMemo(() => {
    const map = new Map();
    vennRegions.forEach((region) => map.set(region.id, region));
    return map;
  }, [vennRegions]);

  useEffect(() => {
    setDisabledRois(new Set());
  }, [resetAtlasKey]);

  useEffect(() => {
    const fullId = vennPhases.join('_');
    setSelectedRegionIds([fullId]);
    setSelectedElectrodeId(null);
    setSelectedEndpoint(null);
  }, [vennPhases, selectedTask]);

  const selectedRegions = useMemo(
    () => selectedRegionIds.map((id) => regionsById.get(id)).filter(Boolean),
    [regionsById, selectedRegionIds],
  );

  const selectedIds = useMemo(() => {
    const ids = new Set();
    selectedRegions.forEach((region) => {
      region.electrode_ids?.forEach((id) => ids.add(id));
    });
    return ids;
  }, [selectedRegions]);

  const vennSelectedElectrodes = useMemo(
    () => subjectFilteredElectrodes.filter((electrode) => selectedIds.has(electrode.id)),
    [subjectFilteredElectrodes, selectedIds],
  );

  const availableRois = useMemo(() => {
    const rois = new Set(vennSelectedElectrodes.map((electrode) => electrode.roi));
    return [...rois].sort();
  }, [vennSelectedElectrodes]);

  const enabledRois = useMemo(
    () => new Set(availableRois.filter((roi) => !disabledRois.has(roi))),
    [availableRois, disabledRois],
  );

  const roiFilteredIds = useMemo(() => {
    if (enabledRois.size === 0) return new Set();
    return new Set(
      vennSelectedElectrodes
        .filter((electrode) => enabledRois.has(electrode.roi))
        .map((electrode) => electrode.id),
    );
  }, [vennSelectedElectrodes, enabledRois]);

  useEffect(() => {
    if (!selectedElectrodeId) return;
    const electrode = electrodeById.get(selectedElectrodeId);
    if (electrode && !enabledRois.has(electrode.roi)) {
      setSelectedElectrodeId(null);
      setSelectedEndpoint(null);
    }
  }, [enabledRois, selectedElectrodeId, electrodeById]);

  const selectedSummary = useMemo(
    () => buildSelectionSummary(selectedRegions, roiFilteredIds.size),
    [selectedRegions, roiFilteredIds],
  );

  const selectedElectrode = selectedElectrodeId ? electrodeById.get(selectedElectrodeId) : null;

  const tableElectrodes = useMemo(
    () => subjectFilteredElectrodes.filter((electrode) => roiFilteredIds.has(electrode.id)),
    [subjectFilteredElectrodes, roiFilteredIds],
  );

  const tableElectrodesKey = useMemo(
    () => tableElectrodes.map((electrode) => electrode.id).join('|'),
    [tableElectrodes],
  );

  const roiCounts = useMemo(() => {
    const counts = {};
    vennSelectedElectrodes.forEach((electrode) => {
      counts[electrode.roi] = (counts[electrode.roi] || 0) + 1;
    });
    return counts;
  }, [vennSelectedElectrodes]);

  const roiBarItems = useMemo(
    () => availableRois
      .map((roi) => ({ roi, count: roiCounts[roi] || 0 }))
      .sort((a, b) => b.count - a.count || a.roi.localeCompare(b.roi)),
    [availableRois, roiCounts],
  );

  const toggleRoi = (roi) => {
    setDisabledRois((current) => {
      const next = new Set(current);
      if (next.has(roi)) next.delete(roi);
      else next.add(roi);
      return next;
    });
  };

  const enableAllRois = () => {
    setDisabledRois((current) => {
      const next = new Set(current);
      availableRois.forEach((roi) => next.delete(roi));
      return next;
    });
  };

  const deselectAllRois = () => {
    setDisabledRois((current) => new Set([...current, ...availableRois]));
  };

  const selectRegion = (id) => {
    setSelectedRegionIds((current) => (
      current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id]
    ));
    setSelectedElectrodeId(null);
    setSelectedEndpoint(null);
  };

  const selectElectrode = (id) => {
    setSelectedElectrodeId(id);
    setSelectedEndpoint(null);
  };

  const selectEndpoint = (endpoint) => {
    setSelectedEndpoint(endpoint);
  };

  const clearSelectedElectrode = () => {
    setSelectedElectrodeId(null);
    setSelectedEndpoint(null);
  };

  return {
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
  };
}
