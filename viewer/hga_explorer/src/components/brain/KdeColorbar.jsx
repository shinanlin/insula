import React, { useEffect, useState } from 'react';
import { vlagPositiveCssGradient } from '../../brainKde.js';

const KDE_COLORBAR_GRADIENT = vlagPositiveCssGradient();

function formatVmax(value) {
  return Number.isFinite(value) ? value.toFixed(2) : '';
}

export default function KdeColorbar({
  range,
  vmaxValue,
  disabled = false,
  onVmaxChange,
}) {
  const [draft, setDraft] = useState(() => formatVmax(vmaxValue));
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    if (!isEditing) {
      setDraft(formatVmax(vmaxValue));
    }
  }, [vmaxValue, isEditing]);

  const commitDraft = () => {
    setIsEditing(false);
    const trimmed = draft.trim();
    if (!trimmed) {
      onVmaxChange?.(null);
      setDraft(formatVmax(vmaxValue));
      return;
    }
    const parsed = parseFloat(trimmed);
    if (Number.isFinite(parsed) && parsed > 0) {
      onVmaxChange?.(parsed);
      setDraft(parsed.toFixed(2));
      return;
    }
    onVmaxChange?.(null);
    setDraft(formatVmax(vmaxValue));
  };

  if (!range?.hasData) {
    return (
      <div className="kde-colorbar kde-colorbar-empty">
        <span className="kde-colorbar-title">HGA density</span>
        <span className="kde-colorbar-empty-text">No KDE data</span>
      </div>
    );
  }

  return (
    <div className="kde-colorbar">
      <span className="kde-colorbar-title">HGA density</span>
      <div className="kde-colorbar-body kde-colorbar-body-minimal">
        <input
          type="text"
          inputMode="decimal"
          className="kde-colorbar-max-input"
          value={draft}
          disabled={disabled}
          aria-label="KDE color scale maximum"
          onFocus={() => setIsEditing(true)}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={commitDraft}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.currentTarget.blur();
            }
          }}
        />
        <div className="kde-colorbar-track" style={{ background: KDE_COLORBAR_GRADIENT }} />
        <span className="kde-colorbar-tick">0</span>
      </div>
    </div>
  );
}
