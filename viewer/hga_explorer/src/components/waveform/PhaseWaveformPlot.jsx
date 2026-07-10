import React, { useEffect, useRef, useState } from 'react';
import { WAVEFORM_PLOT_MIN_HEIGHT } from '../../constants/brain.js';
import StaticPhasePlot from './StaticPhasePlot.jsx';
import PlaybackVLineOverlay from './PlaybackVLineOverlay.jsx';

function PhaseWaveformPlot({
  phase,
  index,
  seriesList,
  traceKey,
  yRange,
  isSingleElectrode,
  isActivePhase,
  currentTime,
}) {
  const shellRef = useRef(null);
  const [plotHeight, setPlotHeight] = useState(WAVEFORM_PLOT_MIN_HEIGHT);
  const [relayoutToken, setRelayoutToken] = useState(0);

  useEffect(() => {
    setRelayoutToken((token) => token + 1);
  }, [isActivePhase]);

  useEffect(() => {
    const node = shellRef.current;
    if (!node) return undefined;

    const updateHeight = () => {
      const measured = Math.floor(node.clientHeight);
      if (measured <= 0) return;
      const nextHeight = Math.max(WAVEFORM_PLOT_MIN_HEIGHT, measured);
      setPlotHeight((current) => (
        current === nextHeight ? current : Math.max(current, nextHeight)
      ));
      setRelayoutToken((token) => token + 1);
    };

    updateHeight();
    const observer = new ResizeObserver(updateHeight);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={shellRef}
      className={`phase-waveform-plot-shell${isActivePhase ? ' playing' : ''}`}
    >
      <StaticPhasePlot
        phase={phase}
        index={index}
        seriesList={seriesList}
        traceKey={traceKey}
        yRange={yRange}
        isSingleElectrode={isSingleElectrode}
        plotHeight={plotHeight}
        relayoutToken={relayoutToken}
      />
      <PlaybackVLineOverlay
        phase={phase}
        index={index}
        currentTime={currentTime}
        visible={isActivePhase}
      />
    </div>
  );
}

export default React.memo(PhaseWaveformPlot, (prev, next) => {
  if (
    prev.phase !== next.phase
    || prev.index !== next.index
    || prev.traceKey !== next.traceKey
    || prev.isSingleElectrode !== next.isSingleElectrode
    || prev.yRange[0] !== next.yRange[0]
    || prev.yRange[1] !== next.yRange[1]
    || prev.seriesList !== next.seriesList
  ) {
    return false;
  }
  if (prev.isActivePhase !== next.isActivePhase) return false;
  if (next.isActivePhase && prev.currentTime !== next.currentTime) return false;
  return true;
});
