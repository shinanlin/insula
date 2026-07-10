import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, Search } from 'lucide-react';

export default function SubjectDropdown({
  availableSubjects,
  selectedSubjects,
  onToggleSubject,
  onSelectAllSubjects,
  onDeselectAllSubjects,
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const rootRef = useRef(null);

  const selectedCount = selectedSubjects.size;
  const totalCount = availableSubjects.length;
  const allSelected = totalCount > 0 && selectedCount === totalCount;
  const noneSelected = selectedCount === 0;

  const filteredSubjects = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return availableSubjects;
    return availableSubjects.filter((subject) => subject.toLowerCase().includes(normalized));
  }, [availableSubjects, query]);

  const triggerLabel = noneSelected
    ? 'No subjects selected'
    : allSelected
      ? `All subjects (${totalCount})`
      : `${selectedCount} of ${totalCount} subjects`;

  useEffect(() => {
    if (!open) return undefined;

    const handlePointerDown = (event) => {
      if (!rootRef.current?.contains(event.target)) {
        setOpen(false);
      }
    };

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') setOpen(false);
    };

    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [open]);

  useEffect(() => {
    if (!open) setQuery('');
  }, [open]);

  if (!availableSubjects.length) return null;

  return (
    <div className="subject-dropdown" ref={rootRef}>
      <button
        type="button"
        className={`subject-dropdown-trigger${open ? ' open' : ''}`}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        <span className="subject-dropdown-value">{triggerLabel}</span>
        <ChevronDown size={16} className="subject-dropdown-chevron" />
      </button>

      {open && (
        <div className="subject-dropdown-menu" role="listbox" aria-multiselectable="true">
          <div className="subject-dropdown-search">
            <Search size={14} />
            <input
              type="search"
              value={query}
              placeholder="Search subjects"
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>

          <div className="subject-dropdown-actions">
            <button
              type="button"
              className="subject-dropdown-action"
              disabled={allSelected}
              onClick={onSelectAllSubjects}
            >
              Select all
            </button>
            <button
              type="button"
              className="subject-dropdown-action"
              disabled={noneSelected}
              onClick={onDeselectAllSubjects}
            >
              Deselect all
            </button>
          </div>

          <div className="subject-dropdown-list">
            {filteredSubjects.length === 0 && (
              <div className="subject-dropdown-empty">No subjects match your search.</div>
            )}
            {filteredSubjects.map((subject) => {
              const checked = selectedSubjects.has(subject);
              return (
                <label
                  key={subject}
                  className={`subject-dropdown-option${checked ? ' checked' : ''}`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => onToggleSubject(subject)}
                  />
                  <span>{subject}</span>
                </label>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
