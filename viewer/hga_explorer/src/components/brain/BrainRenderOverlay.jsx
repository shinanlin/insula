import React, { useEffect, useState } from 'react';
import PieProgress from '../ui/PieProgress.jsx';

const OVERLAY_DELAY_MS = 300;

export default function BrainRenderOverlay({
  active = false,
  progress = 0,
  phaseLabel,
}) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (!active) {
      setVisible(false);
      return undefined;
    }

    const handle = window.setTimeout(() => {
      setVisible(true);
    }, OVERLAY_DELAY_MS);

    return () => window.clearTimeout(handle);
  }, [active]);

  if (!active || !visible) return null;

  return (
    <div className="brain-render-overlay" role="status" aria-live="polite">
      <div className="brain-render-overlay-card">
        <PieProgress
          progress={progress}
          size={104}
          strokeWidth={8}
          label="Preparing"
          sublabel={phaseLabel ? `${phaseLabel} map` : 'KDE map'}
        />
        <p className="brain-render-overlay-note">
          Pre-rendering animation frames. Playback starts when complete.
        </p>
      </div>
    </div>
  );
}
