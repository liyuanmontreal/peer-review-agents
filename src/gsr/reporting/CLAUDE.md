# reporting — Pipeline Output Interpretation and Presentation

## Role in the GSR Pipeline

`reporting` is **pipeline stage 5** — the final read-only output layer.

It reads from SQLite (claims, verification results, evidence chunks, paper metadata, reviews) and
renders human-readable artifacts that surface what the pipeline found, why, and how confidently.

```
SQLite (papers + reviews + claims + verification_results + paper_chunks)
    → [ reporting ] → Markdown reports, annotated PDFs, API workspace payloads
                                ↓
                    CLI output  /  frontend UI  /  debugging
```

Reporting is **purely read-only**. It never runs pipeline stages, never modifies database records,
and never redefines what the pipeline concluded. Its role is to faithfully present upstream results
in a form that maximizes trust and interpretability.

If a report looks wrong, the cause is almost always upstream (extraction, retrieval, or verification).
Check pipeline stages before editing rendering logic.

**📍 Code Pointers:**
- Public API: `src/gsr/reporting/__init__.py` (exports `export_verification_report`, `export_extraction_report`, `export_extraction_comparison_report`)
- CLI integration: `src/gsr/cli.py` → `report` command handler

---

## Core Design Philosophy

1. **Faithful, not interpretive.** Reports reflect what the pipeline stored. Reporting must not
   silently filter, reorder, or restate verdicts in ways that obscure the pipeline's actual output.
   If a claim is missing from a report, the claim is likely absent from the DB, not a rendering bug.

2. **Review-centric grouping.** Reports group claims and verdicts by review, then by source field
   (weaknesses, strengths, questions, summary). This mirrors how a human reads a review and maps
   to the extraction schema.

3. **Experiment-aware.** Because extraction can run multiple times (different models, configs,
   thresholds), reports scope to an `experiment_id`. The default is the most recent experiment.
   Changing this default changes which claims appear in the report.

4. **Evidence keys can be IDs or chunk indices.** The `evidence_chunk_ids` stored in
   `verification_results` may reference `paper_chunks.id` (string UUID) or `paper_chunks.chunk_index`
   (integer). `detect_evidence_mode()` probes the DB to resolve which format was used.

5. **Traceability over aesthetics.** Every rendered report entry should trace back to:
   - the exact review sentence the claim came from (`verbatim_quote`)
   - the evidence chunk(s) that informed the verdict
   - the page and section of that evidence in the paper

---

## Module Structure

```
src/gsr/reporting/
├── __init__.py           # public API: export_verification_report, export_extraction_report,
│                         #             export_extraction_comparison_report
├── exporter.py           # top-level export functions; orchestrates queries → render → file write
├── queries.py            # all DB reads; structured queries returning list[dict]
├── render_md.py          # Markdown/HTML rendering; pure functions taking dicts, returning str
├── workspace_data.py     # API/UI data contract; builds workspace payloads for the frontend
├── annotate_pdf.py       # PDF annotation with claim-specific bounding boxes (PyMuPDF)
├── schema.py             # DB introspection helpers (table_exists, get_table_columns, require_tables)
└── utils.py              # string helpers: safe_filename, md_escape, as_quote_block,
                          #                 parse_chunk_keys, file_url, file_url_page
```

---

## Key Concepts

### Report Types

| Report | Entry point | DB tables required |
|---|---|---|
| **Verification report** | `export_verification_report()` | `verification_results`, `claims`, `extraction_runs`, `paper_chunks`, `papers`, `experiments` |
| **Extraction report** | `export_extraction_report()` | `claims`, `extraction_runs`, `papers` |
| **Extraction comparison** | `export_extraction_comparison_report()` | `extraction_runs`, `claims` (needs ≥2 experiments) |
| **Retrieval report** | `export_retrieval_report()` | `papers`, `paper_chunks`, `chunk_embeddings` |

All reports are written to `REPORT_DIR` (from `gsr.config`) unless `out_path` is specified.

**📍 Code Pointers:**
- `src/gsr/reporting/exporter.py` → `export_verification_report()`, `export_extraction_report()`,
  `export_extraction_comparison_report()`, `export_retrieval_report()`

---

### Query Layer (`queries.py`)

All database reads live here. Functions return `list[dict]` or plain dicts — never SQLite Row
objects. Key functions:

| Function | Purpose |
|---|---|
| `load_verification_rows()` | Core join: `verification_results` ⋈ `claims` ⋈ `extraction_runs` ⋈ `reviews`. Supports filtering by `experiment_id`, `verdict`, `min_conf`. Results ordered by verdict priority then confidence desc. |
| `load_extraction_rows_rich()` | Claims + extraction runs + review text fields + paper meta. Primary source for extraction reports. |
| `load_reviews_for_paper()` | All reviews for a paper, with all rubric fields. |
| `load_experiment_overview_for_paper()` | Per-experiment summary: claim counts, avg scores, run config. Used to choose the default experiment. |
| `load_claims_for_paper_grouped_by_review_experiment()` | Claims grouped by `review_id` for a specific experiment. |
| `collect_all_evidence_keys()` | Deduplicates `evidence_chunk_ids` across all verification rows. |
| `detect_evidence_mode()` | Probes `paper_chunks` to determine whether keys are `id` or `chunk_index`. |
| `fetch_chunks_strict()` | Fetches chunks by `paper_id` + keys, using detected mode. Strict paper-ID scoping prevents cross-paper leakage. |

**📍 Code Pointers:**
- `src/gsr/reporting/queries.py`

---

### Rendering Layer (`render_md.py`)

Pure functions: take dicts, return Markdown strings. No DB access. Key functions:

| Function | Purpose |
|---|---|
| `render_verification_markdown()` | Verification report (review-centric): groups by review → source field, shows claim highlights in raw review text, verdict badges, evidence excerpts |
| `render_extraction_markdown_review_first()` | Extraction report: grouped by review → experiment → field → claim |
| `render_extraction_comparison()` | Side-by-side comparison of two or more extraction experiments |
| `render_retrieval_markdown()` | Retrieval report: parsing/chunking/indexing/retrieval preview |
| `_highlight_many_sentences()` | Injects colored `<mark>` spans into review text for each claim verbatim quote |
| `_normalize_verdict()` | Normalizes verdict strings (`"refuted"` / `"contradicted"` → `"refuted"`, etc.) |

Evidence in verification reports is length-limited by `evidence_max_chars` (default 1400 chars).
Page numbers in evidence chunks may be 0-indexed or 1-indexed — controlled by `assume_page_0_indexed`.

**📍 Code Pointers:**
- `src/gsr/reporting/render_md.py`

---

### Workspace Data Contract (`workspace_data.py`)

This is the **backend-to-frontend bridge**. The API layer calls these functions to produce the
JSON payloads consumed by the React UI. This is separate from Markdown reporting.

| Function | Purpose |
|---|---|
| `get_review_workspace()` | Full payload for a single review: paper meta, review fields, enriched claims with verdicts and evidence |
| `get_paper_workspace()` | Paper-level payload: all reviews with per-review claim/verdict summaries |
| `get_claim_boxes()` | Bounding boxes for a specific claim's evidence in the PDF, used by `PdfPanel` for red-box overlays |

Evidence normalization, bbox filtering (margin/header/footer removal), and page selection happen
here — **not** in the React components. If evidence cards or PDF overlays look wrong in the UI,
inspect the output of `get_review_workspace()` and `get_claim_boxes()` before touching frontend code.

**📍 Code Pointers:**
- `src/gsr/reporting/workspace_data.py` → `get_review_workspace()`, `get_paper_workspace()`, `get_claim_boxes()`
- `_normalize_evidence_item()` — normalizes a single evidence dict into the stable UI shape
- `_filter_margin_boxes()` — removes header/footer/page-number bboxes
- `_choose_primary_page_for_evidence()` — selects which page to navigate to for an evidence item

---

### PDF Annotation (`annotate_pdf.py`)

Produces annotated PDF copies where claim evidence regions are highlighted with colored rectangles.
Used by the CLI `annotate` command (not by the frontend — the frontend uses `get_claim_boxes()`
and renders overlays in-browser via `PdfPanel`).

**📍 Code Pointers:**
- `src/gsr/reporting/annotate_pdf.py` → `annotate_pdf_by_claim()`

---

### Schema and Utilities

- `schema.py`: `table_exists()`, `get_table_columns()`, `require_tables()` — used defensively
  throughout queries and exporters to give clear errors when required pipeline steps haven't run.
- `utils.py`: `safe_filename()`, `parse_chunk_keys()`, `file_url()`, `file_url_page()`, `as_quote_block()`

---

## Experiment ID Scoping

Because extraction can run multiple times, reports must be scoped to an `experiment_id`:

- `export_verification_report()` and `export_extraction_report()` both default to the **most recent**
  experiment (by `last_started_at` from `load_experiment_overview_for_paper()`).
- Pass `experiment_id=` explicitly to target a specific run.
- `load_verification_rows()` in `queries.py` accepts `experiment_id` as a filter via the
  `extraction_runs` join.

If a report is missing claims that should be present, first check which `experiment_id` is being
used and whether those claims belong to a different experiment.

**📍 Code Pointers:**
- `src/gsr/reporting/queries.py` → `load_experiment_overview_for_paper()`
- `src/gsr/reporting/exporter.py` → experiment selection logic in `export_verification_report()`

---

## Evidence Key Resolution

`verification_results.evidence_chunk_ids` is stored as a JSON array or CSV of keys. These keys
may be `paper_chunks.id` (string/UUID) or `paper_chunks.chunk_index` (integer string), depending
on which version of the retrieval/verification pipeline wrote them.

The reporting layer handles this transparently:
1. `collect_all_evidence_keys()` parses and deduplicates all keys across rows
2. `detect_evidence_mode()` probes `paper_chunks` to identify the format (`'id'` or `'chunk_index'`)
3. `fetch_chunks_strict()` fetches using the detected mode, always constrained to the correct `paper_id`

Do not assume a fixed key format. If evidence looks empty in reports, check `detect_evidence_mode()`
against the actual values stored in `evidence_chunk_ids`.

**📍 Code Pointers:**
- `src/gsr/reporting/queries.py` → `collect_all_evidence_keys()`, `detect_evidence_mode()`, `fetch_chunks_strict()`

---

## CLI Entry Points

```bash
# Extraction report (latest experiment by default)
gsr report extraction --paper-id "openreview::<forum_id>"

# Verification report
gsr report verification --paper-id "openreview::<forum_id>"

# Retrieval report
gsr report retrieval --paper-id "openreview::<forum_id>"

# Compare two extraction experiments
gsr report compare --paper-id "openreview::<forum_id>"

# Annotate PDF with claim evidence boxes
gsr annotate --paper-id "openreview::<forum_id>" --claim-id "<claim_id>"
```

---

## Debugging Guide

**Report is empty or has no claims:**
- Check whether extraction ran successfully for this paper and experiment
- Confirm `experiment_id` — the default selects the latest experiment, which may not be the one you expect
- Verify required tables exist (`require_tables()` raises a clear error if not)

**Evidence is missing or shows wrong text:**
- Inspect `evidence_chunk_ids` in `verification_results` for a sample claim
- Run `detect_evidence_mode()` manually to confirm ID format
- Check `fetch_chunks_strict()` is receiving the correct `paper_id` — cross-paper leakage is prevented by strict scoping

**Claim highlights missing in verification report:**
- The renderer uses `verbatim_quote` from `claims` to locate sentences in review text
- If `verbatim_quote` is empty or doesn't match the review field text exactly, highlighting fails silently
- This is an extraction-quality issue, not a rendering bug

**UI evidence cards or red boxes look wrong:**
- Inspect `get_review_workspace()` output for the review in question
- Check `_normalize_evidence_item()` for the specific evidence shape being produced
- Check `_filter_margin_boxes()` — it may be filtering boxes you expect to keep
- Do not edit `PdfPanel.tsx` before confirming the backend payload is correct

**Report shows wrong page numbers:**
- Check `assume_page_0_indexed` — if the PDF is 0-indexed and this flag is `False`, all page references will be off by one
- Page numbers come from `paper_chunks.page`, which is set during paper parsing/chunking, not during reporting
