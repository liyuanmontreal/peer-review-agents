# paper_retrieval — Module 3: PDF Parsing, Indexing, and Evidence Retrieval

## Role in the GSR Pipeline

`paper_retrieval` is **pipeline stage 3** — it sits between claim extraction and verification.

Its job is to take a paper PDF and a set of challengeable claims, and produce ranked **evidence
candidates** that verification can consume. Everything downstream (verdicts, evidence cards, PDF
overlays) depends on the quality and shape of what this module returns.

```
claim extraction → [ paper_retrieval ] → claim verification
                        ↑
              PDF parse → chunk → embed → index → retrieve → evidence objects
```

The module is consumed by:
- `gsr retrieve --paper-id ...` (CLI, via `cli.py`)
- `retrieve_all_claims_for_paper()` (batch caching path)
- `claim_verification` (reads cached `retrieval_results` from SQLite)
- `reporting/workspace_data.py` (frontend data contract, loads `evidence_objects`)

---

## Module Layout

The module is organized into subpackages. Only `retrieval.py` and `__init__.py` remain at the
top level.

```
paper_retrieval/
├── __init__.py             # high-level orchestration: index_paper(), index_all_papers()
├── retrieval.py            # hybrid retrieval entry point (public API)
├── parsing/
│   ├── parser.py           # PyMuPDF V1 and V2 parsers
│   ├── chunking.py         # V1 text chunking + V2 span chunking (boundary-aware)
│   ├── caption_extractor.py # caption prefix detection from span stream
│   ├── normalized_document.py # NormalizedDocument / NormalizedTable / NormalizedFigure
│   ├── parse_router.py     # routes PyMuPDF V2 → optional Docling enrichment
│   ├── parser_docling.py   # Docling-based semantic parser (optional dependency)
│   └── reference_parser.py # claim → explicit Table/Figure/Section reference extraction
├── evidence/
│   ├── evidence_objects.py # EvidenceObject dataclass
│   └── evidence_builder.py # builds text_chunk / table / figure evidence objects
├── storage/
│   ├── storage.py          # SQLite DDL, all read/write helpers
│   └── embeddings.py       # SentenceTransformer model loading and embedding
└── vision/                 # optional / experimental enhancement layer
    ├── bbox_refinement.py  # P1.5: GroundingDINO/RTDeTR bbox refinement
    ├── grounding_dino.py   # GroundingDINO detector wrapper
    ├── rtdetr_detector.py  # RT-DeTR detector wrapper
    ├── ocr_lighton.py      # LightOnOCR-2 figure text recovery (experimental)
    ├── paddlex_layout_detector.py  # PaddleX layout detector
    └── figure_semantics_qwen.py    # future figure semantics (not in stable path)
```

---

## Core Design Philosophy

1. **Evidence is a normalized abstraction.** All output — text chunks, tables, figures — is
   expressed as `EvidenceObject`. Consumers should never need to special-case `text_chunk` vs
   `figure` vs `table` at the retrieval layer.

2. **PyMuPDF V2 spans are the grounding layer.** Span-level bboxes are the source of truth for
   PDF overlay red boxes, regardless of which higher-level parser ran. Docling and vision tools
   enrich structure but do not replace span grounding.

3. **Caption-first identity.** Figure and table `EvidenceObject`s are identified by their
   caption prefix (`Figure N`, `Table N`) detected in the PyMuPDF span stream. This identity is
   stable across indexing runs. Docling and bbox refinement enrich the content and bbox
   *after* identity is established — they do not redefine it.

4. **Heuristics are intentional.** Chunking windows, bbox gap thresholds, caption continuation
   rules, section heading detection, and reference boost weights are deliberate tradeoffs tuned
   on real ML papers. Do not replace them without testing on real PDFs.

5. **Optional paths are explicitly opt-in.** The default `index_paper()` call uses PyMuPDF V2
   + caption_extractor only. Docling, figure OCR, and bbox refinement are flags that must be
   explicitly enabled. If something is broken in the stable path, look in `parsing/` and
   `evidence/` before suspecting `vision/`.

---

## Key Concepts

### Parsing — Three Tiers

#### V1 — `parse_paper_pdf()` in `parsing/parser.py`
- Uses PyMuPDF `page.get_text("text")` for plain text extraction
- Heuristic section heading detection via `_KNOWN_HEADINGS` / `_NUMBERED_SECTION` patterns
- Returns: `{paper_id, pdf_path, n_pages, sections: [{heading, page, text}]}`
- No spans, no bboxes — not suitable for evidence objects or PDF overlays
- Still supported; used when `layout_aware=False` (the code default for `index_paper()`)

#### V2 — `parse_paper_pdf_v2()` in `parsing/parser.py`
- Uses PyMuPDF `page.get_text("dict")` for layout-aware extraction
- Produces line-level **spans** with `bbox` coordinates — the foundation for PDF overlay boxes
- Returns: `{paper_id, pdf_path, n_pages, spans: [{id, page_num, span_index, text, bbox, ...}], sections}`
- Spans are persisted to `pdf_spans` for later retrieval by `workspace_data.py`
- **This is the recommended mode** for all evidence-producing runs (`layout_aware=True`)

#### Docling+V2 — via `parsing/parse_router.py` (optional)
- Activated by `use_docling=True` in `index_paper()`
- **At index time, `prefer_docling=False` is always forced.** `index_paper()` calls
  `parse_router.route(prefer_docling=False)` — PyMuPDF V2 runs, Docling does NOT run.
- Full-document Docling at index time was too slow (~180 s per paper). Table `content_text`
  is instead enriched lazily just before verification via `enrich_topk_tables_with_docling()`
  in `retrieval.py` (see "Late Docling Table Enrichment" below).
- `parse_router.route()` still accepts `prefer_docling=True` for standalone use and
  benchmarks — it is only forced off inside `index_paper()`.
- Docling's `do_ocr=False` — OCR is not Docling's job here
- Note: `parse_router.py` defines a Layer 3 (PaddleOCR) as **future / not yet implemented**

**📍 Code Pointers:**
- `src/gsr/paper_retrieval/parsing/parser.py` → `parse_paper_pdf()`, `parse_paper_pdf_v2()`
- `src/gsr/paper_retrieval/parsing/parse_router.py` → `route()`
- `src/gsr/paper_retrieval/parsing/parser_docling.py` → `parse_with_docling()`, `docling_available()`

---

### NormalizedDocument

`NormalizedDocument` (dataclass in `parsing/normalized_document.py`) is the **intermediate
representation** produced by `parse_router.route()`. Downstream code (`chunking.py`,
`evidence_builder.py`) reads from this rather than calling parser functions directly.

Key fields:
| Field | Source | Always present? |
|---|---|---|
| `spans` | PyMuPDF V2 | Yes (when `layout_aware=True`) |
| `sections` | Docling (or V2 heuristic) | Yes |
| `tables` | Docling | Only when Docling ran successfully |
| `figures` | Docling | Only when Docling ran successfully |
| `source_parser` | `"pymupdf"` or `"docling+pymupdf"` | Yes |

`NormalizedTable` and `NormalizedFigure` are also defined here. They hold label, caption, page,
section, content_text, bbox, and span_ids. These are **internal** — downstream consumers work
with `EvidenceObject`, not with these.

**📍 Code Pointers:**
- `src/gsr/paper_retrieval/parsing/normalized_document.py` → `NormalizedDocument`, `NormalizedTable`, `NormalizedFigure`

---

### Chunking

#### V1 — `chunk_paper()` in `parsing/chunking.py`
- Word-based sliding window over section text
- Outputs `char_start` / `char_end` offsets

#### V2 — `chunk_paper_v2_from_spans()` in `parsing/chunking.py`
- Span-based sliding window (each unit is a line span)
- Outputs `span_ids`, `page_start`, `page_end` — enables bbox-aware downstream rendering
- **P1.7 boundary-aware mode** (always active in V2): prevents chunks from crossing page
  boundaries, section changes, heading-like spans, or large vertical gaps
- **P1.7b Docling hints** (optional): when a `NormalizedDocument` is passed via `docling_doc=`,
  Docling-identified section boundaries and caption regions act as additional hard segment breaks.
  If `docling_doc` is None (default), V2 behavior is unchanged.
- `_majority_section()` assigns the dominant section heading to each chunk window

Chunks are persisted to the `paper_chunks` table.

**📍 Code Pointers:**
- `src/gsr/paper_retrieval/parsing/chunking.py` → `chunk_paper()`, `chunk_paper_v2_from_spans()`

---

### Caption Extraction and Non-Text Evidence

`parsing/caption_extractor.py` identifies `Figure N` and `Table N` caption prefixes in the
PyMuPDF span stream. This is the **baseline path** for building figure and table evidence
objects when Docling is not used (or when Docling found no match for a given label+page).

`evidence/evidence_builder.py` uses these captions to construct `figure` and `table` evidence
objects:
- Infers the visual region by scanning spans spatially above (figures) or below (tables) the
  caption line
- Applies gap thresholds relative to caption height to stop expansion at section boundaries
- **Inline-reference rejection:** spans that start with `Figure N` / `Table N` but are
  body-text prose references are hard-rejected before any evidence object is created. Patterns
  rejected: comma immediately after the number, "respectively" within the span, conjunction
  lists (`Figure 1 and Figure 2`), and 3rd-person prose verbs (`shows`, `compares`, `presents`,
  etc.). Example: `Fig. 7 also shows ...` is rejected entirely — no figure object is created.
- P1.9 table region validation: rejects regions that look like prose paragraphs
- Trims caption text to one paragraph via `_keep_single_caption_paragraph()`
- Builds concise `retrieval_text` = `label + Caption: ... + Section: ...`
- **`bbox_confidence`** (`"high"` / `"inferred"` / `"caption_only"`) reflects how reliably the
  figure body was localized. `caption_only` means a valid identity was found (label, caption,
  page) but the spatial expansion above the caption found no figure body spans.

**Table `content_text` is always empty at index time.** All table `EvidenceObject`s are built
with `content_text=""`. Docling markdown content is **not** injected at build time — it is
filled lazily by `enrich_topk_tables_with_docling()` in `retrieval.py` just before verification
runs. If late enrichment is skipped or fails, evidence cards fall back to `caption_text`.
The `docling_tables` argument to `build_evidence_objects_for_paper()` is accepted but intentionally
left `None` at index time.

**Important layout assumptions (PyMuPDF path):**
- Figure captions are typically **below** the figure body → expand **upward** from caption
- Table captions are typically **above** the table body → expand **downward** from caption

**📍 Code Pointers:**
- `src/gsr/paper_retrieval/parsing/caption_extractor.py` → `extract_captions_from_spans()`
- `src/gsr/paper_retrieval/evidence/evidence_builder.py` → `build_evidence_objects_for_paper()`
- Region expansion: `evidence_builder.py` → `_collect_figure_region_with_caption()`, `_collect_object_region()`
- Caption trimming: `evidence_builder.py` → `_keep_single_caption_paragraph()`

---

### The EvidenceObject

`EvidenceObject` (dataclass in `evidence/evidence_objects.py`) is the **canonical output shape**
of this module. It is what gets stored in the `evidence_objects` table and what downstream
verification and UI consume.

Key fields:
| Field | Purpose |
|---|---|
| `id` | `{paper_id}_{kind}_{idx}` or `{paper_id}_chunk_{n}` |
| `object_type` | `text_chunk` / `table` / `figure` |
| `label` | `"Figure 3"`, `"Table 2"`, or `None` for text |
| `page` / `page_start` / `page_end` | Page location |
| `section` / `section_number` | Section context |
| `caption_text` | For figures/tables — concise, single paragraph |
| `retrieval_text` | What BM25 and semantic search operate on |
| `content_text` | What evidence cards display |
| `bbox` | Enclosing bbox for PDF overlay (union of selected spans) |
| `span_ids` | Span IDs for fine-grained page rendering |
| `asset_path` | PNG crop path for figure assets; populated when `has_image=True` in `metadata` |
| `metadata` | Arbitrary per-object metadata (used by bbox refinement) |

**📍 Code Pointers:**
- `src/gsr/paper_retrieval/evidence/evidence_objects.py` → `EvidenceObject`, `TEXT_CHUNK`, `TABLE`, `FIGURE`
- Serialization for SQLite: `EvidenceObject.to_row()`

---

### Figure Asset Trust Policy

Figure image assets (PNG crops) are generated conservatively. The pipeline separates **figure
identity / evidence bbox** (always conservative) from **figure asset generation** (best-effort).

**Inline prose rejection — no object created:**
Spans that open with `Figure N` / `Table N` but are mid-sentence body text references are
hard-rejected before any evidence object is created. This prevents fake figure objects from
prose mentions like `Fig. 7 also shows ...`.

**`caption_only` — explicit intermediate state, conservative asset default:**
When the main region expansion finds no figure body spans above the caption, the object gets
`bbox_confidence = "caption_only"`. This is not a failure — the evidence object is still valid
for retrieval and verification. By default no PNG crop is saved:
`has_image = False`, `crop_skipped_reason = "caption_only"`.
A narrow crop of just the caption line is intentionally avoided because it conveys no useful
visual information and could mislead.

**Caption-expand heuristic — asset-only recovery path:**
Before giving up on a `caption_only` figure, the builder tries a cheap recovery:
1. Expand upward from the caption bbox by up to ~35% of page height (max 280 pt).
2. Expand slightly beyond caption width (±20 pt), clamped to page bounds.
3. Plausibility checks: minimum box size; candidate must be ≥3× taller than caption alone;
   above-caption region must not be prose-dense (< 3 lines longer than 80 chars).
4. If plausible: save the crop, `has_image = True`,
   `asset_bbox_source = "caption_expand_heuristic"`, `asset_recovered_from_caption_only = True`.
5. If not plausible: preserve conservative behavior —
   `has_image = False`, `crop_skipped_reason = "caption_only"`.

**UI bbox vs asset bbox — intentionally separated:**
The caption-expand heuristic is **asset-only**. It does not change `evidence_objects.bbox`,
`bbox_confidence`, `span_ids`, or any retrieval / verification field. The UI overlay always
uses the conservative evidence bbox. This preserves PDF overlay trust while improving figure
image coverage for evidence cards.

**Figure audit logging (`[figure-*]` log lines per paper):**
- `[figure-filter]` — inline prose references rejected before object creation
- `[figure-dedupe]` — duplicate (kind, label) candidates dropped
- `[figure-crop]` — `caption_only_skipped` (heuristic failed or rejected) +
  `caption_only_attempted` + `caption_only_recovered` counts
- `[figure-summary]` — total figure objects, `with_image` count, `without_image` count

**📍 Code Pointers:**
- `src/gsr/paper_retrieval/evidence/evidence_builder.py` → `_is_inline_reference_not_caption()`
- `src/gsr/paper_retrieval/evidence/evidence_builder.py` → `_try_caption_expand_heuristic()`
- Figure crop block: `build_evidence_objects_for_paper()` — Phase A section

---

### Embedding

Model: SPECTER2 (`allenai/specter2_base`) — scientific text, fine-tuned on ML/AI literature.
Fallback: `all-MiniLM-L6-v2` for offline environments.

Override via `GSR_EMBEDDING_MODEL` environment variable.

Two embedding tables exist:
- `chunk_embeddings` — legacy, keyed on `paper_chunks.id`
- `evidence_embeddings` — preferred, keyed on `evidence_objects.id`

Retrieval always prefers `evidence_embeddings` when available, falls back to `chunk_embeddings`.

**Skip optimization:** In the V2 layout-aware path, when `evidence_objects` are non-empty,
the legacy `chunk_embeddings` pass is skipped entirely — only `evidence_embeddings` are
generated. The unconditional log line `[embed] DECISION ... skip_chunk_embed=True/False`
records which path ran and why. `chunk_embeddings` are only generated when `layout_aware=False`
or the evidence object build produced zero objects.

**📍 Code Pointers:**
- `src/gsr/paper_retrieval/storage/embeddings.py` → `load_embedding_model()`, `embed_chunks()`
- `src/gsr/paper_retrieval/__init__.py` → `index_paper()` embedding block (`skip_chunk_embed` logic)

---

### Hybrid Retrieval and Ranking

`retrieve_evidence_for_claim()` in `retrieval.py` performs a **three-signal blend**:

| Signal | Default weight | Description |
|---|---|---|
| BM25 (lexical) | 0.35 | `rank_bm25.BM25Okapi` over `retrieval_text` |
| Semantic cosine | 0.45 | SPECTER2 embeddings vs. query vector |
| Reference boost | 0.20 | Explicit mention of `Table N` / `Figure N` / `Section N` in claim |
| Type prior | 0.05 | Small bonus for table/figure when claim uses result/comparison language |

**Reference boosting** (`parsing/reference_parser.py`):
- Parses claim text for explicit `Table N`, `Figure N`, `Section N` mentions
- Awards `1.0` boost when a matching labeled `evidence_object` exists
- Falls back to partial section number matching (`0.7` / `0.4`)

**Hard include** (`_force_include_explicit_object_matches()`):
- If the claim explicitly names `Figure 9` and a `figure` object labeled `Figure 9` exists,
  it is moved to the front of results regardless of combined score.

**Diversity caps:**
- `text_chunk` capped at 3 per top-k result set
- `table` capped at 2
- `figure` capped at 1

**Fallback path:**
- If `evidence_objects` table is empty or absent, retrieval falls back to `paper_chunks`
  (text-only mode).
- Embeddings fall back from `evidence_embeddings` to `chunk_embeddings` in the same way.
- If neither aligned embeddings exist: lexical-only (BM25 + reference boost, no semantic).

**📍 Code Pointers:**
- `src/gsr/paper_retrieval/retrieval.py` → `retrieve_evidence_for_claim()`, `retrieve_evidence_for_claims()`, `retrieve_all_claims_for_paper()`
- Reference parsing: `src/gsr/paper_retrieval/parsing/reference_parser.py` → `compute_reference_boost()`
- Hard include: `retrieval.py` → `_force_include_explicit_object_matches()`

---

### Retrieval Cache

`retrieve_all_claims_for_paper()` is the **batch caching entry point**:
- Loads all claims for a paper from SQLite
- Calls `retrieve_evidence_for_claim()` per claim
- Writes results to `retrieval_results` table
- Emits `GSR_PROGRESS retrieve_claims {n} {total}` to stdout for the API progress layer
- Respects `--force` to re-retrieve already-cached claims

The verification stage reads directly from this cache — it does not call retrieval live.

**📍 Code Pointers:**
- `retrieval.py` → `retrieve_all_claims_for_paper()`

---

### Late Docling Table Enrichment

Because full-document Docling is too slow to run at index time, table `content_text` is
injected **lazily just before verification**, by `enrich_topk_tables_with_docling()` in
`retrieval.py`. This runs as the `verify_enrich` stage in `verify_all_claims()`.

**How it works:**
1. Queries `retrieval_results JOIN evidence_objects WHERE object_type='table'` for the given
   claim IDs to find the top-k table candidates actually used in this verification run.
2. Skips objects already enriched (`metadata_json.docling_enriched = true`).
3. Groups unique tables by paper, looks up `pdf_path` from the `papers` table.
4. For each paper, calls `extract_tables_for_pages(pdf_path, pages)`:
   - **Temp-PDF optimization:** builds a minimal temp PDF containing only the requested pages
     using PyMuPDF `fitz.insert_pdf()` — Docling runs only on that subset (~3–4× faster).
   - Maps extracted table page numbers back to original page numbers via `temp_to_orig` dict.
5. Matches each Docling-extracted table to an existing `EvidenceObject` via
   `_match_docling_table(cand, page_tables)`:
   - Priority: `exact_label_page` → `label_page` (normalized) → `caption_page` (≥4 word overlap) → `page_only` (only when exactly 1 table on page)
6. On match: calls `update_table_evidence_docling_enrichment()` → writes `content_text` +
   sets `metadata_json.docling_enriched = true`.
7. On failure: calls `mark_table_evidence_docling_failed()` → records reason in `metadata_json`.

**Fallback:** If enrichment is skipped, fails, or Docling is not installed, `content_text`
remains `""` and evidence cards fall back to `caption_text`.

**📍 Code Pointers:**
- `src/gsr/paper_retrieval/retrieval.py` → `enrich_topk_tables_with_docling()`, `_match_docling_table()`
- `src/gsr/paper_retrieval/parsing/parser_docling.py` → `extract_tables_for_pages()`
- `src/gsr/paper_retrieval/storage/storage.py` → `update_table_evidence_docling_enrichment()`, `mark_table_evidence_docling_failed()`
- `src/gsr/claim_verification/verifier.py` → late enrichment call site (pre-ThreadPoolExecutor block)

---

## Storage Schema

Six tables owned by this module (DDL in `storage/storage.py` → `init_retrieval_db()`):

| Table | Contents |
|---|---|
| `paper_chunks` | Text chunks with section, page, char offsets, optional span_ids |
| `pdf_spans` | V2 layout-aware line spans with bbox coordinates |
| `evidence_objects` | Normalized evidence: text_chunk / table / figure |
| `chunk_embeddings` | Legacy chunk-level SPECTER2 embeddings |
| `evidence_embeddings` | Preferred evidence-object-level embeddings |
| `retrieval_results` | Cached ranked evidence per claim, with all score components |

**📍 Code Pointers:**
- `src/gsr/paper_retrieval/storage/storage.py` → `init_retrieval_db()`, `save_evidence_objects()`, `save_retrieval_results()`, `load_evidence_objects_for_paper()`, `load_pdf_spans_for_paper()`

---

## High-Level Orchestration

`index_paper()` in `__init__.py` is the top-level entry point:

```
index_paper(paper_id, pdf_path, conn, layout_aware=False, use_docling=False, ...)

V1 path (layout_aware=False — code default):
    → parse_paper_pdf()              # plain text, sections only
    → chunk_paper()                  # word-based sliding window
    → [embed + save if embed=True]

V2 path (layout_aware=True — recommended):
    → print GSR_PROGRESS index_prepare 1 1
    → if use_docling=True:
        parse_router.route(prefer_docling=False)  # always PyMuPDF V2 only at index time
    → else:
        parse_paper_pdf_v2()         # PyMuPDF V2 only
    → save_pdf_spans()
    → chunk_paper_v2_from_spans()    # span-window, boundary-aware
    → save_chunks()
    → print GSR_PROGRESS index_evidence 1 1
    → build_evidence_objects_for_paper()  # text_chunk + caption_extractor; tables get content_text=""
    → save_evidence_objects()
    → [optional: vision/bbox_refinement  (use_bbox_refine=True)]
    → print GSR_PROGRESS index_embed 1 1
    → if evidence_objects non-empty:     # skip_chunk_embed = layout_aware and bool(evidence_objects)
        # chunk_embeddings pass SKIPPED — evidence_embeddings are sufficient
        save_evidence_embeddings()   # SPECTER2 on evidence objects only
    → else:                              # fallback: no evidence objects
        embed_chunks()               # SPECTER2 on chunks
        save_embeddings()            # chunk_embeddings
        [save_evidence_embeddings()  # if evidence_objects exist despite layout_aware=False]

NOTE: Docling table content_text is NOT filled at index time.
      It is injected lazily pre-verification by enrich_topk_tables_with_docling() in retrieval.py.
```

**📍 Code Pointers:**
- `src/gsr/paper_retrieval/__init__.py` → `index_paper()`, `index_all_papers()`
- `src/gsr/paper_retrieval/evidence/evidence_builder.py` → `build_evidence_objects_for_paper()`

---

## Progress Events

Events emitted to stdout (`flush=True`) for the API progress layer:

| Event | Emitted from | When |
|---|---|---|
| `GSR_PROGRESS index_prepare 1 1` | `__init__.py` | V2 parse/chunk phase starting |
| `GSR_PROGRESS index_evidence 1 1` | `__init__.py` | Evidence object construction starting |
| `GSR_PROGRESS index_embed 1 1` | `__init__.py` | Embedding phase starting |
| `GSR_PROGRESS retrieve_claims {n} {total}` | `retrieval.py` | Per-claim retrieval progress |

The `index_prepare`, `index_evidence`, `index_embed` events are always `1 1` (single-step
signals, not count-based). They exist so the API can surface the slow indexing phases before
per-claim progress begins.

---

## Vision Subpackage (Optional / Experimental)

The `vision/` subpackage provides enhancement features that run **after** the stable evidence
object pipeline has completed. None of these are active in the default `index_paper()` call.

**P1.5 — Bbox refinement** (`vision/bbox_refinement.py`, activated by `use_bbox_refine=True`):
- Matches GroundingDINO or RT-DeTR detections against existing figure/table `EvidenceObject`s
- **Does NOT change object identity** (id, label, page, object_type)
- **Does NOT change** retrieval_text, content_text, caption_text, or embeddings
- Stores refined bbox in `evidence_objects.metadata_json` only — no schema migration
- Falls back gracefully if detectors are unavailable
- `bbox_detector` parameter selects the detector (`"groundingdino"` is the default)

**Figure OCR** (`vision/ocr_lighton.py`, activated by `use_figure_ocr=True`):
- Runs LightOnOCR-2 on figure objects with high-confidence bboxes
- Requires `pip install lightonai` — skipped silently if not installed
- Stores OCR text in `metadata_json` under the `figure_ocr_*` schema
- Experimental A/B evaluation flag — not part of the default pipeline

**Other vision files** (`grounding_dino.py`, `rtdetr_detector.py`, `paddlex_layout_detector.py`,
`figure_semantics_qwen.py`) are detector backends and experimental components. Do not depend on
their APIs being stable.

---

## Reasoning Guidelines for Future Agents

**Before changing retrieval ranking:**
1. Inspect the actual `retrieval_results` rows for a specific claim — not the UI output.
2. Check whether `evidence_objects` or `paper_chunks` are being used (log line:
   `using_evidence_objects=True/False`).
3. Verify embeddings are aligned with the evidence objects (count must match).

**Before changing evidence card content:**
1. Check `evidence_objects.content_text` in SQLite — that is what the UI receives.
2. For table objects: `content_text` is `""` at index time by design. It is populated by
   `enrich_topk_tables_with_docling()` pre-verification. If a table card shows no text,
   check whether late enrichment ran (look for `docling_enriched=True` in `metadata_json`).
3. If content is too long or noisy, fix it in `evidence_builder.py`
   (`_keep_single_caption_paragraph`, `_truncate`).
4. Do not truncate or reshape content in React components.

**Before changing PDF overlay boxes:**
1. Check `evidence_objects.bbox_json` — that is the source of truth for red boxes.
2. Region expansion logic is in `evidence_builder.py` → `_collect_figure_region_with_caption()`.
3. If a bbox covers only the caption line and not the figure body, the expansion heuristic
   stopped too early — diagnose the gap threshold, not the frontend.
4. If `use_bbox_refine` was run, also check `metadata_json` for a `bbox_refined` key — the
   bbox may have been updated after the initial evidence build.
5. A figure with `bbox_confidence = "caption_only"` may still have `has_image = True` if the
   caption-expand heuristic recovered an asset. The PDF overlay bbox intentionally remains the
   caption anchor — the asset bbox is stored in metadata only and does not flow to the overlay.

**Before modifying chunking:**
- V1 chunks use character counts; V2 chunks use span (line) counts.
- Changing `chunk_size` / `chunk_overlap` requires re-indexing — existing cached embeddings
  will be misaligned.

**If retrieval returns no results:**
1. Check `paper_chunks` is non-empty for the paper.
2. Check `evidence_objects` — if absent, retrieval falls back to chunks silently.
3. Check `chunk_embeddings` or `evidence_embeddings` — if absent, retrieval is lexical-only.

**If you need to inspect stale paths from before the refactor:**
- Any flat-file references like `paper_retrieval/parser.py` are now at
  `paper_retrieval/parsing/parser.py`. The same pattern applies across all subpackages.

---

## CLI Entry Points

```bash
# Index a single paper (V2 layout-aware — recommended)
gsr retrieve --paper-id "<paper_id>" --layout-aware

# Force re-index
gsr retrieve --paper-id "<paper_id>" --layout-aware --force

# Index with Docling table enrichment (requires: pip install docling)
gsr retrieve --paper-id "<paper_id>" --layout-aware --use-docling

# V1 plain-text index (no evidence objects, no PDF overlays)
gsr retrieve --paper-id "<paper_id>"
```
