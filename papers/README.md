# Paper Cards

This directory is the paper-level context layer for the insula project. Each genuine Zotero source has one Markdown card. Cards are stored in one flat directory because a paper can contribute to several themes.

## Status Model

- `full-text`: a PDF was present in Zotero and included in the initial corpus reading pass.
- `abstract-only`: the card currently relies on the abstract and bibliographic metadata.
- `web-reviewed`: the Zotero record lacked an abstract and PDF, but an external scholarly source was reviewed for thematic classification.
- `initial-theme-pass`: sufficient for thematic organization, but not yet citation-ready evidence extraction.

## Source of Truth

Zotero remains the source of truth for PDFs and bibliographic metadata. Cards link back through `zotero_key`, DOI, and source URL. The CSV manifest provides a collection-wide index for filtering and AI retrieval.

## Citation Discipline

Before a claim is used in a manuscript, verify it against the full text and add the relevant page, figure, method, and limitation to the paper card.
