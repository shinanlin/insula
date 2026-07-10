import React from 'react';
import LinearProgress from './LinearProgress.jsx';

export default function TraceLoadProgress({
  progress = 0,
  completed = 0,
  total = 0,
  compact = false,
  className = '',
  title = 'Loading HGA traces',
  subjectLabel,
}) {
  const pct = Math.round(Math.max(0, Math.min(1, progress)) * 100);
  const resolvedSubjectLabel = subjectLabel ?? (total === 1 ? 'subject' : 'subjects');

  return (
    <div className={`trace-load-progress ${compact ? 'compact' : ''} ${className}`.trim()} role="status" aria-live="polite">
      {!compact && (
        <div className="trace-load-progress-title">{title}</div>
      )}
      {compact && title !== 'Loading HGA traces' && (
        <div className="trace-load-progress-title compact-title">{title}</div>
      )}
      <LinearProgress progress={progress} aria-label={`${title} progress`} />
      <div className="trace-load-progress-meta">
        {total > 0
          ? `${completed} of ${total} ${resolvedSubjectLabel} · ${pct}%`
          : `Preparing… · ${pct}%`}
      </div>
    </div>
  );
}
