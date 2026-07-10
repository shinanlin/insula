import React from 'react';

export default function LinearProgress({
  progress = 0,
  className = '',
  'aria-label': ariaLabel = 'Loading progress',
}) {
  const clamped = Math.max(0, Math.min(1, progress));
  const pct = Math.round(clamped * 100);

  return (
    <div
      className={`linear-progress ${className}`.trim()}
      role="progressbar"
      aria-label={ariaLabel}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
    >
      <div className="linear-progress-fill" style={{ width: `${pct}%` }} />
    </div>
  );
}
