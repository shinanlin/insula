import React from 'react';

export default function RoiBarChart({ items, enabledRois, onToggleRoi }) {
  const maxCount = items.reduce((max, item) => Math.max(max, item.count), 0);

  return (
    <div className="roi-bar-chart">
      {items.map(({ roi, count }) => {
        const active = enabledRois.has(roi);
        const widthPct = maxCount > 0 ? (count / maxCount) * 100 : 0;
        return (
          <button
            key={roi}
            type="button"
            className={active ? 'roi-bar-row active' : 'roi-bar-row'}
            onClick={() => onToggleRoi(roi)}
            title={`${roi}: ${count} electrode${count === 1 ? '' : 's'}`}
          >
            <span className="roi-bar-label">{roi}</span>
            <span className="roi-bar-track" aria-hidden="true">
              <span className="roi-bar-fill" style={{ width: `${widthPct}%` }} />
            </span>
            <span className="roi-bar-count">{count}</span>
          </button>
        );
      })}
    </div>
  );
}
