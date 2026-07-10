import React from 'react';
import { Brain } from 'lucide-react';
import TraceLoadProgress from '../ui/TraceLoadProgress.jsx';

export default function ViewerInitialLoadScreen({
  progress = 0,
  stage = 'manifest',
  stageLabel = 'Loading viewer data…',
  completed = 0,
  total = 0,
  error = null,
}) {
  const traceTitle = stage === 'traces' ? 'Loading HGA traces' : stageLabel;

  return (
    <div className="viewer-initial-load" role="status" aria-live="polite">
      <div className="viewer-initial-load-card">
        <div className="viewer-initial-load-header">
          <Brain size={28} aria-hidden="true" />
          <h1>Loading HGA Phase Overlap Viewer</h1>
        </div>
        {error ? (
          <p className="viewer-initial-load-error">{error}</p>
        ) : (
          <>
            <TraceLoadProgress
              title={traceTitle}
              progress={progress}
              completed={completed}
              total={total}
            />
            <p className="viewer-initial-load-note">
              {stage === 'traces'
                ? 'Subject traces are loading. The interactive viewer will appear when complete.'
                : 'Electrode metadata and traces are loading. The interactive viewer will appear when complete.'}
            </p>
          </>
        )}
      </div>
    </div>
  );
}
