import React, { useEffect, useRef, useState } from 'react';
import { WAVEFORM_PLOT_MIN_HEIGHT } from '../../constants/waveform.js';
import StaticPhasePlot from './StaticPhasePlot.jsx';
import PlaybackVLineOverlay from './PlaybackVLineOverlay.jsx';

export default function PhaseWaveformPlot({
  phase,
  index,
  trace,
  traceKey,
  yRange,
  animationTime = null,
  showPlaybackLine = false,
  lineColor,
}) {
  const shellRef = useRef(null);
  const [plotHeight, setPlotHeight] = useState(WAVEFORM_PLOT_MIN_HEIGHT);
  const [relayoutToken, setRelayoutToken] = useState(0);

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
    window.addEventListener('resize', updateHeight);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', updateHeight);
    };
  }, []);

  return (
    <div ref={shellRef} className="phase-waveform-plot-shell">
      <StaticPhasePlot
        phase={phase}
        index={index}
        trace={trace}
        traceKey={traceKey}
        yRange={yRange}
        plotHeight={plotHeight}
        relayoutToken={relayoutToken}
        lineColor={lineColor}
      />
      <PlaybackVLineOverlay
        phase={phase}
        index={index}
        currentTime={animationTime}
        visible={showPlaybackLine}
      />
    </div>
  );
}
