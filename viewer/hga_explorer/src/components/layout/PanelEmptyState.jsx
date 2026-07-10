import React from 'react';

export default function PanelEmptyState({ emptyState, className = '' }) {
  if (!emptyState) return null;

  return (
    <div className={`panel-empty-state ${className}`.trim()} role="status">
      <strong>{emptyState.title}</strong>
      <span>{emptyState.message}</span>
    </div>
  );
}
