# link_ingest — Module 1: External Link to Normalized Pipeline Input

## Role in the GSR Pipeline

`link_ingest` is **pipeline stage 1** — the entry point for all paper+review data entering the GSR system from external sources.

Its job is to accept an external URL (currently OpenReview forum URLs), resolve the provider, fetch all associated content (paper metadata, PDF, review threads), classify the thread replies by type, and normalize them into the data contract consumed by all downstream stages.

```
external URL → [ link_ingest ] → normalized paper + reviews (SQLite + JSON)
                                          ↓
                             claim_extraction → paper_retrieval → verification
```

`link_ingest` is the **only** point in the pipeline where external I/O happens for paper and review data. Everything downstream reads from SQLite or the JSON snapshot produced here. If paper metadata looks wrong, a review is missing, or the PDF failed to download, this is the first module to inspect.

The module is consumed by:
- `gsr fetch --url "<openreview_url>"` (CLI, via `cli.py`)
- `gsr run --url "<openreview_url>"` (full-pipeline CLI entry point, via `cli.py`)

---

## Core Design Philosophy

1. **Link-driven, not batch-driven.** This module takes a single URL and resolves it. Venue-level batch ingestion lives in `data_collection/`, not here. `link_ingest` is the per-paper entry path.

2. **Provider-specific logic is fully encapsulated in submodules.** The top-level module is intentionally thin. All OpenReview-specific API knowledge, schema quirks, and type detection live in `link_ingest/openreview/`. Adding a new provider means adding a new submodule — not changing existing code.

3. **Output is a normalized paper bundle, not raw API data.** The forum data dict produced by this module is a stable contract. Field names and shapes are intentional: they map directly to the SQLite schema consumed by claim extraction. Do not change field names without updating storage DDL and all downstream readers.

4. **Best-effort PDF, hard-fail on review data.** PDF download failure is logged as a warning and stored in `pdf_error` — it does not abort the fetch. Missing or malformed review content causes the note to be skipped, not an exception. Downstream stages deal with absent PDF gracefully; claim extraction needs reviews.

5. **Type detection by invitation string, not heuristics on content.** OpenReview reply types (review, meta-review, rebuttal, decision) are classified by inspecting the `invitations` field, not by guessing from content keys. This is intentional robustness against venue-specific content schema variation.

---

## Key Concepts

### URL Resolution

Before any API call is made, the external URL is parsed to a provider-specific identifier. For OpenReview this is the `forum_id` (the `id` query parameter).

**📍 Code Pointers:**
- `src/gsr/link_ingest/openreview/resolve.py` → `resolve_openreview_url()`
- Accepts both `https://openreview.net/forum?id=XXX` and `https://openreview.net/pdf?id=XXX`

---

### OpenReview Fetch and Thread Parsing

`fetch_forum.py` is the core of this module. Given a `forum_id` it:

1. Authenticates with the OpenReview API via `build_client()`
2. Fetches the submission note and extracts paper metadata (title, authors, abstract, keywords, venue)
3. Downloads the PDF and records its SHA-256 checksum
4. Retrieves the full reply thread for the forum
5. Classifies each reply by type and extracts structured fields

**Reply classification priority** (highest wins):
1. **Decision** — detected by `"official_decision"` in invitations or `"decision"` in content; exits early
2. **Review** — detected by `"official_review"` in invitations or `"review"` in content
3. **Meta-Review** — detected by `"meta_review"`, `"metareview"`, `"senior_area_chair"`, or `"official_meta_review"` in invitations
4. **Rebuttal** — detected by `"rebuttal"`, `"official_rebuttal"`, `"author_response"`, or `"author"+"comment"` in invitations

Notes that match none of these patterns (official comments, etc.) are silently ignored.

**Schema flexibility:** Reviews may use either a simple single-field schema (`review`) or a rubric schema (`summary`, `strengths`, `weaknesses`, `questions`, `soundness`, `presentation`, `contribution`). Both are handled; a composite text is constructed if no top-level `review` field is present.

**OpenReview v2 value wrapping:** The v2 API returns `{"value": "..."}` instead of direct string values in many fields. `_unwrap()` handles this transparently throughout the fetch logic.

**📍 Code Pointers:**
- `src/gsr/link_ingest/openreview/fetch_forum.py` → `fetch_forum_data()`
- `src/gsr/link_ingest/openreview/fetch_forum.py` → `_unwrap()`, `_invs_text()`, `_first_present()`
- API client: `src/gsr/data_collection/client.py` → `build_client()`
- PDF download: `src/gsr/data_collection/fetcher.py` → `_download_pdf()`

---

### The Paper Bundle — Output Data Contract

The return value of `fetch_forum_data()` is the **foundational data contract** for this module. It is passed directly to `save_forum_bundle()` and ultimately stored in SQLite.

```python
{
    "id": "openreview::{forum_id}",   # canonical paper ID used throughout GSR
    "forum": str,                      # raw forum ID
    "number": None,
    "venue_id": str,
    "title": str | None,
    "authors": str,                    # JSON-serialized list
    "abstract": str | None,
    "keywords": str,                   # JSON-serialized list
    "fetched_at": str,                 # ISO 8601 timestamp
    "pdf_path": str | None,            # filesystem path to downloaded PDF
    "pdf_sha256": str | None,          # integrity check
    "pdf_error": str | None,           # populated if PDF download failed
    "reviews": list[ReviewDict],
    "rebuttals": list[RebuttalDict],
    "meta_reviews": list[MetaReviewDict],
    "decision": DecisionDict | None,
}
```

**Review shape** (fields used by `claim_extraction` and UI):

| Field | Purpose |
|---|---|
| `id` | `"openreview::{note_id}"` — canonical review ID |
| `paper_id` | `"openreview::{forum_id}"` — links review to paper |
| `forum` | Raw forum ID |
| `replyto` | Parent note ID (review thread structure) |
| `signatures` | JSON-serialized list — reviewer identity |
| `rating` | Numeric score string (from `rating`, `score`, or `recommendation` fields) |
| `confidence` | Reviewer confidence string |
| `summary` | Summary field text |
| `strengths` | Strengths field text |
| `weaknesses` | Weaknesses field text |
| `questions` | Questions field text |
| `soundness` / `presentation` / `contribution` | Rubric scores (conference-style reviews) |

The `claim_extraction` module reads reviews from SQLite by `paper_id` and uses `weaknesses`, `strengths`, `questions`, and `summary` as its default extraction fields. **Field names here must match what extraction expects.**

---

### Storage Adapter

`storage_adapter.py` is a thin bridge: it wraps the paper bundle in a venue dict and delegates to the shared `data_collection/storage` layer, which handles both JSON snapshot persistence and SQLite insertion.

**📍 Code Pointers:**
- `src/gsr/link_ingest/openreview/storage_adapter.py` → `save_forum_bundle()`
- Storage layer: `src/gsr/data_collection/storage.py` → `save_json()`, `save_to_db()`

---

## Module Structure

```
src/gsr/link_ingest/
├── __init__.py                        # empty; no public API exported
└── openreview/
    ├── __init__.py                    # empty
    ├── resolve.py                     # URL → forum_id
    ├── fetch_forum.py                 # forum_id → paper bundle dict
    └── storage_adapter.py             # paper bundle → JSON + SQLite
```

The module is integrated at the CLI level only — `cli.py` imports all three functions directly and orchestrates the fetch → save flow for the `gsr fetch --url` command.

**📍 Code Pointers:**
- CLI integration: `src/gsr/cli.py` → `fetch` command handler
  - `resolve_openreview_url()` → `fetch_forum_data()` → `save_forum_bundle()`

---

## Integration with `data_collection`

`link_ingest` depends on `data_collection` for shared infrastructure:

| Dependency | Purpose |
|---|---|
| `gsr.data_collection.client.build_client()` | Authenticates and returns an OpenReview API client |
| `gsr.data_collection.fetcher._download_pdf()` | Downloads PDF to `PDF_DIR`, returns path and SHA-256 |
| `gsr.data_collection.storage.save_json()` | Writes venue-level JSON snapshot |
| `gsr.data_collection.storage.save_to_db()` | Inserts paper, reviews, rebuttals, meta-reviews into SQLite |

`data_collection` handles batch/venue-level ingestion separately. `link_ingest` uses it only as a storage and client infrastructure dependency — it does not re-implement storage or API auth.

---

## Adding a New Provider

To support a new link type (e.g., arXiv, Semantic Scholar, ACL Anthology):

1. Create `src/gsr/link_ingest/<provider>/` with:
   - `resolve.py` — URL → provider-specific ID
   - `fetch_<entity>.py` — ID → paper bundle dict matching the same contract
   - `storage_adapter.py` — bundle → `save_forum_bundle()` (or equivalent)
2. Register the provider in `cli.py` by detecting the URL pattern and routing to the new resolver/fetcher
3. The paper bundle dict shape must be preserved — `id`, `reviews`, `pdf_path`, etc. are load-bearing for all downstream stages

---

## Reasoning Guidelines for Future Agents

**If a paper is missing its PDF:**
- Check `pdf_error` in SQLite or the bundle dict — the error message is stored there
- PDF failure is non-fatal; paper retrieval will fail later if PDF is required
- The PDF download path is controlled by `gsr.config.PDF_DIR`

**If reviews are missing or incomplete:**
- Check whether the OpenReview forum actually has reviews (some venues keep reviews private)
- Check whether the invitation strings match the expected patterns — venues vary significantly
- Inspect `_invs_text(note)` output to see what invitations the API is actually returning
- Do not assume a missing `review` key means no review — check `summary`/`strengths`/`weaknesses` too

**If review field content is empty or `None`:**
- OpenReview content may use wrapped format `{"value": "..."}` — check `_unwrap()` is covering the relevant fields
- The content object may be a non-dict type for some venues — the fetch logic attempts multiple coercion strategies

**If `paper_id` references look wrong downstream:**
- All IDs in this module are prefixed `"openreview::{raw_id}"` — this is the canonical ID format throughout GSR
- If downstream tables show mismatches, check the prefix convention is being preserved

**Before editing `fetch_forum.py`:**
1. Confirm whether the issue is invitation-based type detection or content field extraction
2. Test against a real OpenReview forum with the actual API response — do not assume from field names alone
3. Changes to review field names cascade to `claim_extraction` (which reads those fields by name from SQLite)

---

## CLI Entry Points

```bash
# Fetch a single paper by OpenReview URL
gsr fetch --url "https://openreview.net/forum?id=<forum_id>"

# Fetch a paper and include PDF download
gsr fetch --url "https://openreview.net/forum?id=<forum_id>" --pdf

# Fetch all papers for a venue (batch path via data_collection, not link_ingest)
gsr fetch --venue "<venue_id>" --limit 10 --pdf

# Full pipeline starting from URL (fetch → extract → retrieve → verify)
gsr run --url "https://openreview.net/forum?id=<forum_id>"
```
