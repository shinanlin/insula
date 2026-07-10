import React from 'react';

export default function PanelTitle({ icon, title }) {
  return (
    <div className="panel-title">
      {icon}
      <span>{title}</span>
    </div>
  );
}
