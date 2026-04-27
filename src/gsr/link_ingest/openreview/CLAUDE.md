# link_ingest/openreview — OpenReview Provider Implementation

## Role in the GSR Pipeline

`link_ingest/openreview` is the **OpenReview-specific provider** within `link_ingest`.
It owns the full path from an OpenReview URL to a normalized paper bundle dict ready for storage.

```
openreview URL
    → resolve_openreview_url()     # URL → forum_id
    → fetch_forum_data()           # forum_id → paper bundle dict
    → save_forum_bundle()          # paper bundle → JSON + SQLite
```

The parent module (`link_ingest`) is intentionally thin and delegates all
OpenReview-specific knowledge to this subpackage. Nothing outside this subpackage
should contain OpenReview API logic.

**📍 Code Pointers:**
- CLI orchestration: `src/gsr/cli.py` → `fetch` command handler (calls all three functions in sequence)
- Parent context: `src/gsr/link_ingest/llms.md`

---

## Module Structure

```
src/gsr/link_ingest/openreview/
├── __init__.py            # empty
├── resolve.py             # URL → forum_id
├── fetch_forum.py         # forum_id → paper bundle dict (core logic)
└── storage_adapter.py     # paper bundle → JSON + SQLite (thin bridge)
```

---

## Key Concepts

### 1. URL Resolution (`resolve.py`)

`resolve_openreview_url()` extracts the `forum_id` from an OpenReview URL.

**Supported URL patterns:**
- `https://openreview.net/forum?id=<forum_id>`
- `https://openreview.net/pdf?id=<forum_id>`

Both map to the same `forum_id` — the `id` query parameter.

**📍 Code Pointers:**
- `src/gsr/link_ingest/openreview/resolve.py` → `resolve_openreview_url()`

---

### 2. Forum Fetch and Thread Parsing (`fetch_forum.py`)

`fetch_forum_data(forum_id)` is the core of this module. It does all of:

1. Authenticate with the OpenReview API via `build_client()`
2. Fetch the submission note → extract paper metadata
3. Download the PDF (best-effort) via `_download_pdf()`
4. Retrieve all reply notes for the forum via `get_all_notes(forum=...)`
5. Classify each reply and extract structured fields
6. Return a single paper bundle dict

#### OpenReview v2 Value Wrapping

OpenReview v2 API returns field values as `{"value": "..."}` instead of bare strings.
`_unwrap(v)` handles this transparently everywhere content fields are read.

If a field returns `None` unexpectedly, check whether `_unwrap()` is being applied.

**📍 Code Pointers:**
- `src/gsr/link_ingest/openreview/fetch_forum.py` → `_unwrap()`

#### Reply Type Detection

Reply classification uses the `invitations` field **only** — not content keys.
This is intentional: OpenReview venue schemas vary, but invitations are stable identifiers.

Helper `_invs_text(note)` joins all invitation strings into a single lowercased text for matching.

**Classification priority (first match wins):**

| Type | Invitation patterns |
|---|---|
| Decision | `"official_decision"` in invitations, **or** `"decision"` in both invitations and content keys |
| Review | `"official_review"` in invitations, **or** `"review"` in content keys |
| Meta-Review | `"meta_review"`, `"metareview"`, `"senior_area_chair"`, `"official_meta_review"` |
| Rebuttal | `"rebuttal"`, `"official_rebuttal"`, `"author_response"`, or `"author"+"comment"` |

Notes matching none of the above are silently skipped (official comments, etc.).

**📍 Code Pointers:**
- `src/gsr/link_ingest/openreview/fetch_forum.py` → `_invs_text()`, `_first_present()`
- Reply loop: `fetch_forum.py` → `for r in replies:` block

#### Review Schema Handling

OpenReview reviews use one of two schemas depending on venue:

**Workshop-style (single field):**
```python
content["review"]  →  stored in the `summary` field
```

**Conference-style (rubric):**
```python
content["summary"], content["strengths"], content["weaknesses"],
content["questions"], content["soundness"], content["presentation"],
content["contribution"]
```

If `content["review"]` is absent, a composite text is constructed by joining all rubric fields
and stored in `review_text` (used for logging/validation). Rubric fields are stored individually.

**Downstream coupling:** `claim_extraction` reads `weaknesses`, `strengths`, `questions`, and `summary`
from SQLite **by name**. Renaming these fields in the review dict here will break claim extraction.

**📍 Code Pointers:**
- `src/gsr/link_ingest/openreview/fetch_forum.py` → `fetch_forum_data()`, review handling block

#### Content Object Coercion

OpenReview `note.content` may not always be a plain dict. The fetch loop coerces it via:
1. `dict(content_obj)` if possible
2. `content_obj.to_json()` as a fallback
3. Empty dict `{}` if both fail

**📍 Code Pointers:**
- `fetch_forum.py` → `content_obj` coercion block near top of reply loop

---

### 3. Paper Bundle — Output Data Contract

`fetch_forum_data()` returns a single dict. This is the **canonical output contract** for this subpackage.
All field names are load-bearing — they map to the SQLite schema via `data_collection/storage.py`.

```python
{
    # Paper identity
    "id":          "openreview::{forum_id}",  # canonical paper ID throughout GSR
    "forum":       str,                        # raw forum_id (no prefix)
    "number":      None,
    "venue_id":    str,

    # Paper metadata
    "title":       str | None,
    "authors":     str,          # JSON-serialized list
    "abstract":    str | None,
    "keywords":    str,          # JSON-serialized list
    "fetched_at":  str,          # ISO 8601 UTC timestamp

    # PDF
    "pdf_path":    str | None,   # filesystem path; None if download failed
    "pdf_sha256":  str | None,
    "pdf_error":   str | None,   # populated on failure; PDF failure is non-fatal

    # Review thread
    "reviews":      list[dict],  # see review shape below
    "rebuttals":    list[dict],
    "meta_reviews": list[dict],
    "decision":     dict | None,
}
```

**Review dict shape** (fields consumed by `claim_extraction`):

| Field | Value |
|---|---|
| `id` | `"openreview::{note_id}"` |
| `paper_id` | `"openreview::{forum_id}"` |
| `forum` | raw forum_id |
| `replyto` | parent note id |
| `signatures` | JSON-serialized list |
| `rating` | string or None |
| `confidence` | string or None |
| `summary` | summary or composite review text |
| `strengths` | rubric strengths text |
| `weaknesses` | rubric weaknesses text |
| `questions` | rubric questions text |
| `soundness` / `presentation` / `contribution` | rubric score strings |

---

### 4. Storage Adapter (`storage_adapter.py`)

`save_forum_bundle(paper_dict, conn)` wraps the paper bundle in a venue dict and delegates to the
shared `data_collection` storage layer. It does not contain any OpenReview-specific logic.

```python
venue_data = {"venue_id": ..., "papers": [paper_dict]}
save_json(venue_data)    # JSON snapshot
save_to_db(venue_data, conn)  # SQLite insertion
```

**📍 Code Pointers:**
- `src/gsr/link_ingest/openreview/storage_adapter.py` → `save_forum_bundle()`
- `src/gsr/data_collection/storage.py` → `save_json()`, `save_to_db()`

---

## External Dependencies

| Dependency | Purpose |
|---|---|
| `gsr.data_collection.client.build_client()` | Returns an authenticated OpenReview API client |
| `gsr.data_collection.fetcher._download_pdf()` | Downloads PDF, returns `(path, sha256, error)` |
| `gsr.data_collection.storage.save_json()` | Writes venue-level JSON snapshot |
| `gsr.data_collection.storage.save_to_db()` | Inserts into SQLite |
| `gsr.config.PDF_DIR` | Root directory for PDF storage |
| `gsr.config.DEFAULT_VENUE_ID` | Fallback when venue cannot be resolved from note |

---

## Debugging Guide

**Reviews are missing or empty:**
- Check whether the OpenReview forum has published reviews (some venues keep them private)
- Add a log or inspect `_invs_text(note)` on raw replies to see what invitations are actually returned
- Check whether the review content uses a wrapped `{"value": ...}` format — `_unwrap()` must be applied
- Verify that the invitation string patterns in the classification block match the actual venue strings

**PDF failed to download:**
- `pdf_error` in the bundle dict holds the exception message
- PDF failure is **non-fatal** — the fetch still completes
- `paper_retrieval` will fail later if the PDF is absent; fix at the ingestion level

**Field values are `None` unexpectedly:**
- Apply `_unwrap()` — v2 API wraps many values as `{"value": "..."}`
- Check whether the content object coercion succeeded (it may have fallen back to `{}`)

**Downstream `paper_id` mismatches:**
- All IDs use the `"openreview::{raw_id}"` prefix — verify this prefix is intact
- `paper_id` in reviews must equal `"openreview::{forum_id}"` exactly

**Before editing reply type detection:**
- Confirm the actual invitation strings returned by the API for the target venue
- Do not change detection order without considering all four types
- Meta-review detection is deliberately conservative to avoid misclassifying reviews

**Before renaming review fields:**
- `weaknesses`, `strengths`, `questions`, `summary` are read by name in `claim_extraction`
- Renaming cascades to the SQLite schema and all downstream SQL queries
