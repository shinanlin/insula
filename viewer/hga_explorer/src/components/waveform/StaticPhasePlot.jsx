import React, { useLayoutEffect, useMemo, useRef } from 'react';
import Plot from 'react-plotly.js';
import Plotly from 'plotly.js-dist-min';
import { PHASE_LABELS } from '../../constants/phases.js';
import { PHASE_TIME_RANGES } from '../../constants/loads.js';
import {
  PLOT_AXIS_TITLE_SIZE,
  PLOT_FONT_FAMILY,
  PLOT_TICK_SIZE,
  PLOT_TITLE_SIZE,
} from '../../constants/typography.js';
import { buildWaveformPlotData } from '../../utils/traces.js';

function xAxisTickConfig(min, max) {
  const span = max - min;
  if (span <= 4.5) {
    return {
      nticks: Math.max(2, Math.round(span) + 1),
      automargin: true,
    };
  }
  return {
    dtick: span > 8 ? 2 : 1,
    tick0: min,
    automargin: true,
  };
}

const StaticPhasePlot = React.memo(function StaticPhasePlot({
  phase,
  index,
  trace,
  traceKey = 'aggregate',
  yRange,
  isSingleElectrode,
  plotHeight,
  relayoutToken = 0,
}) {
  const plotRef = useRef(null);

  const applyPlotSize = (node) => {
    if (!node?._fullLayout || !plotHeight) return;
    const width = node.offsetWidth;
    if (!width) return;
    try {
      Plotly.relayout(node, { width, height: plotHeight });
    } catch {
      // Plot may be mid-init or torn down (e.g. React StrictMode remount).
    }
  };

  const plotData = useMemo(
    () => buildWaveformPlotData(trace, phase, !isSingleElectrode),
    [trace, phase, isSingleElectrode],
  );
  const { min, max } = PHASE_TIME_RANGES[phase];
  const plotFont = { family: PLOT_FONT_FAMILY, color: '#334155', size: PLOT_TICK_SIZE };
  const layout = useMemo(() => ({
    uirevision: `waveform-static-${traceKey}-${phase}`,
    showlegend: false,
    title: {
      text: PHASE_LABELS[phase],
      font: { family: PLOT_FONT_FAMILY, size: PLOT_TITLE_SIZE, color: '#0f172a' },
    },
    margin: {
      l: index === 0 ? 48 : 28,
      r: 12,
      t: 30,
      b: 36,
    },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: plotFont,
    xaxis: {
      title: { text: 'Time (s)', font: { family: PLOT_FONT_FAMILY, size: PLOT_AXIS_TITLE_SIZE, color: '#334155' }, standoff: 6 },
      range: [min, max],
      autorange: false,
      fixedrange: true,
      showticklabels: true,
      showline: true,
      showgrid: false,
      zeroline: true,
      zerolinecolor: '#475569',
      zerolinewidth: 1,
      ...xAxisTickConfig(min, max),
    },
    yaxis: {
      title: index === 0
        ? { text: 'HGA (z)', font: { family: PLOT_FONT_FAMILY, size: PLOT_AXIS_TITLE_SIZE, color: '#334155' }, standoff: 6 }
        : undefined,
      range: yRange,
      autorange: false,
      showgrid: false,
      zeroline: true,
      zerolinecolor: '#475569',
      zerolinewidth: 1,
      showticklabels: index === 0,
    },
    height: plotHeight,
  }), [phase, index, traceKey, yRange, min, max, plotHeight]);

  useLayoutEffect(() => {
    applyPlotSize(plotRef.current);
  }, [plotHeight, relayoutToken]);

  useLayoutEffect(() => () => {
    plotRef.current = null;
  }, []);

  return (
    <Plot
      data={plotData}
      layout={layout}
      config={{ displayModeBar: false, responsive: false, scrollZoom: false }}
      style={{ width: '100%', height: plotHeight }}
      onInitialized={(_figure, graphDiv) => {
        plotRef.current = graphDiv;
        applyPlotSize(graphDiv);
      }}
      onUpdate={(_figure, graphDiv) => {
        plotRef.current = graphDiv;
      }}
    />
  );
}, (prev, next) => (
  prev.phase === next.phase
  && prev.index === next.index
  && prev.traceKey === next.traceKey
  && prev.isSingleElectrode === next.isSingleElectrode
  && prev.plotHeight === next.plotHeight
  && prev.relayoutToken === next.relayoutToken
  && prev.yRange[0] === next.yRange[0]
  && prev.yRange[1] === next.yRange[1]
));

export default StaticPhasePlot;
