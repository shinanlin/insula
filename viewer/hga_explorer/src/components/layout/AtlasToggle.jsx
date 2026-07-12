import React from 'react';
import { Layers } from 'lucide-react';

export default function AtlasToggle({
  options = [],
  selectedAtlas,
  onSelectAtlas,
  disabled = false,
}) {
  if (options.length <= 1) return null;

  return (
    <div
      className="chip-group"
      data-tour="atlas-selector"
      aria-label="Parcellation atlas selector"
    >
      <span className="chip-group-label"><Layers size={14} /> Atlas</span>
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          className={`chip${selectedAtlas === option.id ? ' active' : ''}`}
          disabled={disabled}
          onClick={() => onSelectAtlas(option.id)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
