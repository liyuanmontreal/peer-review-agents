# data_collection — Shared Ingestion Infrastructure

## Role

`data_collection` serves two distinct purposes:

1. **Batch venue ingestion** — fetches all papers + review threads for a venue from OpenReview in one call (`gsr fetch --venue`)
2. **Shared DB infrastructure** — `init_db()` is the canonical database entry point used by every pipeline stage

It is **supporting infrastructure**, not a reasoning or pipeline stage. Most changes to GSR do not require touching this module.

---

## Files

| File | Purpose |
|---|---|
| `client.py` | Builds an authenticated `openreview.api.OpenReviewClient` (API v2) |
| `fetcher.py` | `fetch_venue_data()` — batch-fetches all submissions + reply threads for a venue; optional PDF download |
| `storage.py` | SQLite DDL, `init_db()`, `save_to_db()`, `save_json()` |
| `__init__.py` | Re-exports: `build_client`, `fetch_venue_data`, `init_db`, `save_json`, `save_to_db` |

---

## External API

- **OpenReview API v2** via the `openreview` Python package
- Credentials from `gsr.config.get_credentials()` → `OPENREVIEW_USERNAME` / `OPENREVIEW_PASSWORD` env vars
- Base URL from `gsr.config.API_V2_BASE_URL`

---

## Database Schema

`init_db()` creates and owns the primary SQLite database (path from `gsr.config.DB_PATH`):

| Table | Contents |
|---|---|
| `papers` | Paper metadata: title, authors, abstract, keywords, pdf_path, pdf_sha256 |
| `reviews` | Review content: rating, confidence, summary, strengths, weaknesses, questions, soundness, presentation, contribution, raw_fields |
| `rebuttals` | Author rebuttals |
| `meta_reviews` | Area chair meta-reviews |
| `decisions` | Accept/reject decisions |

Schema uses additive migrations (`_ensure_column()`). All `INSERT OR REPLACE` — re-running is idempotent.

**`init_db()` is load-bearing.** It is called by `paper_retrieval`, `claim_verification`, and every CLI stage before any DB write. Do not change its return type or behavior without auditing all call sites.

---

## JSON Snapshots

`save_json()` writes a timestamped JSON snapshot to `workspace/json/` (path from `gsr.config.JSON_DIR`).  
Format: `{safe_venue_id}_{timestamp}.json` containing `{"venue_id": ..., "papers": [...]}`.

These are archival — not consumed by downstream pipeline stages.

---

## Relation to `link_ingest/`

`link_ingest/openreview/` (single-paper/forum path) imports shared primitives from here:
- `build_client` — OpenReview client construction
- `_download_pdf` — PDF download with resume support (internal helper, exported by use)
- `save_json`, `save_to_db` — storage layer

`link_ingest/` is the **primary ingestion path** for individual papers (via `gsr fetch --url`).  
`data_collection.fetch_venue_data()` is the **batch path** for full venue sweeps (via `gsr fetch --venue`).

---

## Who Uses This Module

| Consumer | What it uses |
|---|---|
| `cli.py` — `gsr fetch --venue` | `build_client`, `fetch_venue_data`, `init_db`, `save_json`, `save_to_db` |
| `link_ingest/openreview/fetch_forum.py` | `build_client`, `_download_pdf` |
| `link_ingest/openreview/storage_adapter.py` | `save_json`, `save_to_db` |
| `paper_retrieval/__init__.py` | `init_db` |
| `claim_verification/__init__.py` | `init_db` |
| `cli.py` — extract, retrieve, verify, report stages | `init_db` |

---

## Safe Editing Guidance

- **Do not change `init_db()` signature or return type.** Every pipeline stage calls it to get a `sqlite3.Connection`. Changing behavior here affects the entire pipeline.
- **Do not rename or drop tables.** `papers`, `reviews`, and their columns are read by `claim_extraction`, `claim_verification`, `reporting/workspace_data.py`, and the API layer.
- **Do not add required new columns without additive migration.** Use `_ensure_column()` pattern to stay backward-compatible with existing databases.
- **Keep `build_client()` generic.** It is shared between the batch path and `link_ingest/`. Do not embed venue-specific or per-forum logic here.
- **`_download_pdf` is imported by `link_ingest/`.** If you rename or change its signature, update both call sites.
- **`INVITATION_SUFFIXES`** (from `gsr.config`) controls how replies are classified into review/rebuttal/meta_review/decision. Changes affect all venue fetches.
