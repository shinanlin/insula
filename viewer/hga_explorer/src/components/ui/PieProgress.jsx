import React from 'react';

export default function PieProgress({
  progress = 0,
  size = 88,
  strokeWidth = 7,
  label,
  sublabel,
  className = '',
  compact = false,
}) {
  const clamped = Math.max(0, Math.min(1, progress));
  const pct = Math.round(clamped * 100);
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const dash = clamped * circumference;

  return (
    <div className={`pie-progress ${className}`.trim()} style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#dbeafe"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#2563eb"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circumference - dash}`}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <div className="pie-progress-center">
        {!compact && (
          <>
            <span className="pie-progress-value">{pct}%</span>
            {label && <span className="pie-progress-label">{label}</span>}
            {sublabel && <span className="pie-progress-sublabel">{sublabel}</span>}
          </>
        )}
      </div>
    </div>
  );
}
