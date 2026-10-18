#!/usr/bin/env python3
"""Record the approved baseline and update notebook input wiring only."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/figures3_6_refresh_2026-10-17'


def source(cell):
    return ''.join(cell.get('source', []))


def put(cell, text):
    cell['source'] = text.splitlines(keepends=True)
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    baseline = EVIDENCE / 'baseline.json'
    if not baseline.exists():
        paths = [ROOT / f'vizpub/fig{n}.ipynb' for n in (1, 2)]
        paths += [p for n in (1, 2) for p in (ROOT / f'img/fig{n}').glob('*.svg')]
        paths += [p for p in (ROOT / 'results/nmf').glob('*') if p.is_file()]
        protected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        old_sources = {}
        for n in range(3, 7):
            nb = json.loads((ROOT / f'vizpub/fig{n}.ipynb').read_text())
            old_sources[str(n)] = [dict(id=c.get('id'), cell_type=c['cell_type'], source=c.get('source', [])) for c in nb['cells']]
        baseline.write_text(json.dumps(dict(protected=protected, original_notebook_sources=old_sources), indent=2) + '\n')
    for number in range(3, 7):
        path = ROOT / f'vizpub/fig{number}.ipynb'
        nb = json.loads(path.read_text())
        if nb.get('metadata', {}).get('approved_refresh_wiring'):
            continue
        for cell in nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            text = source(cell)
            if '%matplotlib inline' in text:
                text = text.replace('%matplotlib inline', '%matplotlib inline\nfrom matplotlib_inline.backend_inline import set_matplotlib_formats\nset_matplotlib_formats("svg")')
            if 'from __future__ import annotations' in text:
                text = 'from __future__ import annotations\n' + text.replace('from __future__ import annotations\n', '')
            if number == 3:
                # Keep the old mask and row ordering, but distinguish absent evidence.
                text = text.replace('out["mask"] = True\n        return out',
                    'out["mask_observed"] = False\n        out["mask"] = False\n        return out')
                text = text.replace('showing raw HGA', 'no evidence; heatmap stays blank')
                text = text.replace('out["mask"] = out["mask"].fillna(False).astype(bool)',
                    'out["mask_observed"] = out["mask"].notna()\n    out["mask"] = out["mask"].fillna(False).astype(bool)')
                text = text.replace('df["subject"] = subj', 'df["subject"] = subj')
                if 'def evoked_to_long(' in text:
                    text = text.replace('"value": data.reshape(-1),', '"value": data.reshape(-1),\n            "n_trials": len(epo.events),')
                if 'def load_picture_naming(' in text:
                    text = text.replace('n_t = len(epochs.times)', 'n_trials = len(epochs.events)\n        n_t = len(epochs.times)')
                    text = text.replace('"value": mean.reshape(-1),', '"value": mean.reshape(-1),\n                "n_trials": n_trials,')
            if number == 4:
                text = text.replace('ANATOMICAL_ROOT = Path("/hpc/group/coganlab/nanlinshi/insula/results")',
                    'ANATOMICAL_ROOT = FUNCTIONAL_ROOT  # unchanged controls copied with provenance')
                text = text.replace('RESULTS_ROOTS = [FUNCTIONAL_ROOT, ANATOMICAL_ROOT]', 'RESULTS_ROOTS = [FUNCTIONAL_ROOT]')
                text = text.replace('PREPARED_ROOT = Path(DEFAULT_DATASETS[TASK]) / "derivatives" / f"decoding({REF})"',
                    'PREPARED_ROOT = ROOT / "results" / "decoding_inputs" / TASK')
                # Preserve the original Figure 4 control palette exactly.
            if number == 6:
                text = text.replace('H_PATH = Path("/work/ns458/nmf_nogo_window_test/results/H_by_phase.csv")',
                    'H_PATH = ROOT / "results" / "nmf" / "H_by_phase.csv"')
            put(cell, text)
        if number == 3:
            audit = '''from pathlib import Path
import json
_audit_dir = ROOT / "results" / "fig3"
_audit_dir.mkdir(parents=True, exist_ok=True)
_rows = []
for _name in ("working", "pn_repeat", "rfig3_input_df", "rfig3_delay_df", "rfig3_output_df"):
    if _name not in globals():
        continue
    _frame = globals()[_name]
    _cols = [c for c in ("task", "subject", "channel", "phase", "description", "modality", "functional_cluster", "n_trials") if c in _frame]
    _members = _frame[_cols].drop_duplicates()
    _members.to_csv(_audit_dir / f"{_name}_membership.csv", index=False)
    _frame.to_csv(_audit_dir / f"{_name}_waveforms.csv.gz", index=False)
    _keys = [c for c in ("task", "phase", "description", "modality", "functional_cluster") if c in _members]
    _counts = _members.groupby(_keys, dropna=False).agg(n_subjects=("subject", "nunique"), n_sites=("channel", "nunique")).reset_index()
    _counts.insert(0, "table", _name)
    _rows.append(_counts)
    if "mask_observed" in _frame:
        _missing = _frame.loc[~_frame.mask_observed, _cols].drop_duplicates()
        _missing.to_csv(_audit_dir / f"{_name}_missing_masks.csv", index=False)
pd.concat(_rows, ignore_index=True).to_csv(_audit_dir / "panel_counts.csv", index=False)
_paired_rows = []
for _name, _field, _levels in (("rfig3_input_df", "modality", ("sound", "image")), ("rfig3_delay_df", "description", ("Decision", "Repeat")), ("rfig3_output_df", "description", ("Repeat", "Passive"))):
    _frame = globals()[_name]
    _members = _frame[["task", "subject", "channel", "phase", "functional_cluster", _field]].drop_duplicates()
    for _key, _group in _members.groupby(["task", "subject", "channel", "phase", "functional_cluster"]):
        _present = set(_group[_field])
        _paired_rows.append(dict(table=_name, **dict(zip(("task", "subject", "channel", "phase", "functional_cluster"), _key)), paired=all(x in _present for x in _levels), missing_levels="|".join(x for x in _levels if x not in _present)))
pd.DataFrame(_paired_rows).to_csv(_audit_dir / "condition_pairing.csv", index=False)
(_audit_dir / "original_exclusions.json").write_text(json.dumps(dict(DROP_CHANNELS=sorted(DROP_CHANNELS), scope="Preserved at each original notebook use; not silently propagated to other panels"), indent=2))
'''
            nb['cells'].append(dict(cell_type='code', id='paper-fig3-provenance', metadata={}, source=audit.splitlines(keepends=True), outputs=[], execution_count=None))
        nb.setdefault('metadata', {})['approved_refresh_wiring'] = 'figures3_6_refresh_2026-10-17'
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
    print('Recorded baseline; publication notebook input paths updated.')


if __name__ == '__main__':
    main()
