# claim_verification — Module 4: LLM-Based Claim Verification Against Paper Evidence

## Role in the GSR Pipeline

`claim_verification` is **pipeline stage 4** — it sits between paper retrieval and reporting/UI.

Its job is to consume retrieved evidence candidates for each claim and produce a structured verdict: was the reviewer's factual claim supported, refuted, insufficiently evidenced, or simply not verifiable from the paper? Verdicts, reasoning, and supporting quotes produced here are the **final trust signal** delivered to the user.

```
claim_extraction → paper_retrieval → [ claim_verification ] → reporting / UI
                                              ↑
                        claims (SQLite) + retrieval_results (SQLite)
                                → LLM verification → persist results
```

The module is consumed by:
- `gsr verify --paper-id ...` (CLI, via `cli.py`)
- `verify_all_claims()` (batch path for API progress)
- `reporting/workspace_data.py` (reads verdicts + evidence JSON for UI payloads)
- `gsr.api.app` (verdict summaries and per-claim evidence endpoints)
- `gsr report verification --paper-id ...` (CLI reporting)

---

## Core Design Philosophy

1. **Reason only from provided evidence.** The LLM is instructed to make no use of outside knowledge. If the retrieved evidence does not resolve the claim, the correct answer is `insufficient_evidence`. This is a trust guarantee — users can trace every verdict to specific evidence items.

2. **Conservative verdict bias.** The system prompts the LLM to prefer `insufficient_evidence` over `supported` or `refuted` when ambiguous. Low-confidence responses should resolve conservatively. Over-claiming is more damaging to user trust than under-claiming.

3. **Explicit object prioritization is evidence-driven, not heuristic.** If the claim text references "Table 5" or "Figure 3", those specific objects are retrieved and placed first in the evidence block. The LLM is instructed to prioritize them. This is not a heuristic to replace — it is a deliberate grounding mechanism.

4. **Span alignment closes the loop to the PDF.** The LLM returns a `supporting_quote` string. `alignment.py` fuzzy-matches this quote back to raw `pdf_spans` to yield concrete span IDs with bounding boxes. Without this step, the verdict exists but cannot be overlaid on the PDF.

5. **Multimodal evidence is a first-class citizen.** Text chunks, figures, and tables are normalized to a unified schema before the prompt is built. Each type has a type-appropriate character budget in the prompt. The LLM is told to prefer table evidence for quantitative claims and figure evidence for architecture/pipeline claims.

---

## Key Concepts

### Verdict Taxonomy

Four mutually exclusive verdict values, applied to every claim:

| Verdict | Meaning |
|---|---|
| `supported` | Evidence directly and unambiguously confirms the claim |
| `refuted` | Evidence directly contradicts the claim |
| `insufficient_evidence` | Evidence does not contain enough information to decide |
| `not_verifiable` | Claim is opinion, value judgment, or untestable from the paper |

The taxonomy is enforced via the Pydantic response model. Any LLM response outside these four values will fail to parse. `not_verifiable` is appropriate for subjective or procedural claims that slipped past extraction filtering — it is not the same as `insufficient_evidence`.

**📍 Code Pointers:**
- `src/gsr/claim_verification/prompts.py` → `VerificationResponse` (Pydantic response model with `verdict` Literal)
- `src/gsr/claim_verification/storage.py` → `verification_results` table DDL (verdict column)

---

### The Verification Result Object

The canonical output of this module, persisted to `verification_results`:

| Field | Purpose |
|---|---|
| `id` | SHA256 of `claim_id + model_id` |
| `claim_id` | Foreign key to `claims` table |
| `paper_id` | Paper the claim belongs to |
| `review_id` | Review the claim came from |
| `verdict` | One of four verdict values above |
| `reasoning` | 2–4 sentence explanation citing evidence items |
| `confidence` | LLM-reported confidence (0–1) |
| `supporting_quote` | Most relevant passage or caption from the evidence |
| `evidence` | JSON list of normalized evidence objects (see below) |
| `evidence_chunk_ids` | JSON list of raw chunk IDs (legacy / summary use) |
| `model_id` | LLM used for this verification |
| `status` | `success` or `error` |
| `error_message` | Set if `status == "error"` |
| `verified_at` | ISO timestamp |

**📍 Code Pointers:**
- `src/gsr/claim_verification/storage.py` → `init_verification_db()`, `save_verification_results()`, DDL
- `src/gsr/claim_verification/verifier.py` → `normalize_structured_evidence()` (evidence list shape)

---

### Evidence Object Shape

Each item in the `evidence` JSON list (on the result) follows a unified schema spanning text, figures, and tables:

| Field | Meaning |
|---|---|
| `chunk_id` | Text chunk ID (from `paper_chunks`) if text evidence |
| `evidence_object_id` | Figure/table ID (from `evidence_objects`) if non-text |
| `object_type` | `text_chunk` / `figure` / `table` |
| `label` | Human-readable label, e.g. `Figure 3`, `Table 2` |
| `page` | Primary page number |
| `page_start` / `page_end` | For multi-page objects |
| `section` | Section heading from the paper |
| `text` | Extracted or surrounding text |
| `caption_text` | Caption (for figures/tables) |
| `bbox` | JSON bounding box `[x, y, w, h]` |
| `span_ids` | Raw PDF span IDs linked to this evidence |
| `aligned_span_ids` | Span IDs fuzzy-matched to `supporting_quote` (for red boxes) |
| `score` | Retrieval relevance score |
| `reference_matched` | Whether this item matched an explicit reference in the claim |

**📍 Code Pointers:**
- `src/gsr/claim_verification/verifier.py` → `load_cached_evidence_mixed()` (evidence loading and normalization)
- `src/gsr/claim_verification/verifier.py` → `normalize_structured_evidence()` (output normalization)

---

### LLM Prompt Structure

Verification uses **structured output** — LiteLLM is called with `response_format=VerificationResponse`, and the response is parsed directly into the Pydantic model.

**System prompt** establishes:
- Fact-checker persona with no outside knowledge
- The four verdict categories with explicit decision rules
- Conservative-verdict instruction (prefer `insufficient_evidence` when ambiguous)
- Evidence type priorities (table → quantitative; figure → architecture/pipeline)

**User prompt** includes:
- Claim text, verbatim quote from the review, and claim type
- Numbered evidence block with each item formatted by type:
  - Text items: up to ~1400 chars
  - Table items: up to ~1800 chars (more detail for quantitative claims)
  - Figure items: up to ~1400 chars
  - Each item shows: type, section, page, label (if any), reference match status, content

**📍 Code Pointers:**
- `src/gsr/claim_verification/prompts.py` → `SYSTEM_PROMPT`, `build_messages()`
- `src/gsr/claim_verification/prompts.py` → `format_evidence_item()`, `format_evidence_block()`
- `src/gsr/claim_verification/prompts.py` → `VerificationResponse`

---

### Evidence Prioritization Logic

Before building the prompt, evidence items are reordered by `_prepare_evidence_for_prompt()`:

1. **Extract explicit figure/table references** from the claim text (e.g. "Table 5", "Fig. 3") via regex
2. **Identify matching objects** — evidence items whose `label` matches the extracted refs
3. **Add one supplemental text chunk** on the same page as the first explicit object (for context)
4. **Sort the remainder** by: `reference_matched` > type bonus (table/figure preferred by claim type) > retrieval `score`
5. Final order: explicit objects first, supplemental context, then sorted remainder

This ensures that when the claim names a specific paper artifact, that artifact is in the LLM's primary attention window.

**📍 Code Pointers:**
- `src/gsr/claim_verification/verifier.py` → `_prepare_evidence_for_prompt()`
- `src/gsr/claim_verification/verifier.py` → `_extract_explicit_refs()`, `_label_matches_any()`
- `src/gsr/claim_verification/verifier.py` → `_claim_prefers_table()`, `_claim_prefers_figure()`

---

### Span Alignment (Supporting Quote → PDF Red Boxes)

After verification, the `supporting_quote` string returned by the LLM is a human-readable passage. To overlay a red box on the PDF, it must be mapped back to `pdf_spans` with real bounding geometry.

`alignment.py` does this via fuzzy string matching:
1. Load candidate spans from `pdf_spans` table by the span IDs attached to evidence items
2. Score each span against `supporting_quote` using a combined similarity: `0.7 × char_similarity + 0.3 × token_jaccard`
3. Keep spans scoring ≥ `min_score` (default 0.25)
4. Return top `top_n` (default 3) span IDs; optionally expand to adjacent spans on the same page

The `aligned_span_ids` field on each evidence item carries these results to `workspace_data.py` and the UI.

**📍 Code Pointers:**
- `src/gsr/claim_verification/alignment.py` → `select_best_span_ids_for_supporting_quote()`
- `src/gsr/claim_verification/alignment.py` → `load_spans_by_ids()`, `_combined_similarity()`

---

### Evidence Loading Strategy

`load_cached_evidence_mixed()` queries the `retrieval_results` table and joins to both `evidence_objects` (figures/tables) and `paper_chunks` (text). The result is normalized to the unified evidence schema above.

Fallback behavior:
- Prefers `evidence_objects` when available (richer metadata: bboxes, captions, labels)
- Falls back to `paper_chunks` for text-only papers
- Returns at most `top_k` items (default 5), ordered by retrieval rank

If no cached evidence exists and `allow_live_retrieval=True`, the module calls `paper_retrieval.retrieve_evidence_for_claim()` directly and saves results back to cache before proceeding.

**📍 Code Pointers:**
- `src/gsr/claim_verification/verifier.py` → `load_cached_evidence_mixed()`
- `src/gsr/claim_verification/verifier.py` → `_retrieve_live()`

---

### Batch Verification and Parallelism

`verify_all_claims()` is the batch entry point. It emits four progress stages:

1. **`verify_prepare`** — emitted at function entry; loads claims, builds model client
2. Loads all verifiable claims for a paper from the `claims` table
3. Filters already-verified claims (unless `force=True`)
4. **`verify_enrich`** — emitted before late Docling enrichment; calls
   `enrich_topk_tables_with_docling()` to fill `content_text` for top-k table candidates
5. **`verify_model`** — emitted after enrichment, before the per-claim loop
6. **`verify_claims 0 {total}`** — emitted just before the loop to signal count-mode to the API
7. Runs `verify_claim()` per claim in a `ThreadPoolExecutor` (default `max_workers=2`)
8. After each result, runs span alignment and appends `aligned_span_ids`
9. Emits `GSR_PROGRESS verify_claims {completed} {total}` to stdout after each claim
10. Persists all results via `save_verification_results()`

**📍 Code Pointers:**
- `src/gsr/claim_verification/verifier.py` → `verify_all_claims()`
- `src/gsr/claim_verification/__init__.py` → public API exports
- Progress signals: `GSR_PROGRESS verify_prepare/verify_enrich/verify_model/verify_claims` in `verifier.py`
- Late enrichment: `src/gsr/paper_retrieval/retrieval.py` → `enrich_topk_tables_with_docling()`

---

## Storage Schema

One table owned by this module (DDL in `storage.py` → `init_verification_db()`):

| Table | Contents |
|---|---|
| `verification_results` | One row per (claim_id, model_id) pair — verdict, reasoning, supporting_quote, evidence JSON |

Key indexes: `claim_id`, `paper_id`, `verdict`.

Schema uses additive migrations (`_ensure_column()`). The `evidence_json` column was added post-launch; older databases are upgraded on `init_verification_db()`.

**📍 Code Pointers:**
- `src/gsr/claim_verification/storage.py` → `init_verification_db()`, `save_verification_results()`, `load_verification_results()`, `get_verdict_summary()`

---

## High-Level Orchestration

```
verify_all_claims(conn, paper_id=..., model=..., top_k=5, max_workers=2, ...)
    → GSR_PROGRESS verify_prepare 0 1           # loading claims, building model client
    → load claims from claims table
    → filter already-verified (unless force=True)
    → GSR_PROGRESS verify_enrich 0 1            # pre-verify Docling table enrichment
    → enrich_topk_tables_with_docling()         # fills content_text for top-k table candidates
        → extract_tables_for_pages() with temp-PDF (requested pages only)
        → _match_docling_table() → update_table_evidence_docling_enrichment()
    → GSR_PROGRESS verify_model 0 1             # enrichment done; starting per-claim loop
    → GSR_PROGRESS verify_claims 0 {total}      # count-mode signal to API
    → per claim (threaded):
        → load_cached_evidence_mixed()          # retrieval_results → unified evidence schema
        → [_retrieve_live() if allow_live_retrieval and no cache]
        → verify_claim(claim, evidence)
            → _prepare_evidence_for_prompt()    # prioritize explicit refs
            → build_messages()                  # system + user prompt
            → complete(response_format=VerificationResponse)  # LiteLLM structured output
            → normalize_structured_evidence()   # flatten to output schema
        → select_best_span_ids_for_supporting_quote()  # fuzzy align → aligned_span_ids
        → GSR_PROGRESS verify_claims {completed} {total}
    → save_verification_results()
```

**📍 Code Pointers:**
- `src/gsr/claim_verification/__init__.py` → `verify_claim`, `verify_all_claims` (public API)
- `src/gsr/claim_verification/verifier.py` → full orchestration

---

## Integration Points

### Upstream Dependencies

| Module | What it provides | How used |
|---|---|---|
| `gsr.claim_extraction` | `claims` table rows + `complete()` / `get_model_id()` LLM wrapper | Claims are loaded from SQLite; same LiteLLM wrapper is reused |
| `gsr.paper_retrieval` | `retrieval_results` table (cached evidence) + `retrieve_evidence_for_claim()` | Primary evidence source; live fallback available |
| `gsr.data_collection` | DB initialization | `init_db()` must run before this module can write |

### Downstream Consumers

| Consumer | What it reads | Purpose |
|---|---|---|
| `gsr.reporting.workspace_data` | `verification_results` (verdict + evidence JSON + aligned_span_ids) | Builds evidence cards + PDF overlay payloads for UI |
| `gsr.api.app` | `verification_results` | Progress polling, verdict summaries, per-claim inspection endpoints |
| CLI reports | `verification_results` | `gsr report verification --paper-id ...` |

The `evidence` JSON blob and `aligned_span_ids` on each result are the **primary data contract** to `workspace_data.py`. If evidence cards or red boxes look wrong in the UI, inspect these fields in the database before touching React components.

---

## Reasoning Guidelines for Future Agents

**If verdicts feel wrong or inconsistent:**
1. Check the evidence items that were actually sent to the LLM — inspect `retrieval_results` and `evidence_objects` for the claim, not just the final verdict.
2. Check whether explicit figure/table references in the claim are being matched by `_extract_explicit_refs()` and `_label_matches_any()`.
3. Check `top_k` — if only 3–5 items are available and none are relevant, `insufficient_evidence` is the correct conservative response, not a bug.
4. Check the system prompt in `prompts.py` — the conservative-bias instruction directly affects verdict distribution.

**If red boxes are missing or misplaced on the PDF:**
1. Check `aligned_span_ids` in the stored `evidence_json` — if empty, alignment failed.
2. Check `supporting_quote` — if it is empty or very short, there is nothing to align against.
3. Inspect `pdf_spans` for the relevant page — if spans are not present, the issue is upstream in paper parsing (Stage 3), not here.
4. Adjust `min_score` in `select_best_span_ids_for_supporting_quote()` only if spans exist but none score above threshold.

**If evidence cards show wrong content:**
1. Inspect `evidence_json` in `verification_results` directly in SQLite — this is what `workspace_data.py` reads.
2. If `object_type`, `label`, `page`, or `caption_text` are wrong, the issue is in `load_cached_evidence_mixed()` or the upstream `evidence_objects` table.
3. Do not patch evidence formatting in React before confirming the backend payload is correct.

**If verification is skipping claims:**
1. `require_cached_evidence=True` (default) causes claims without cached retrieval to be silently skipped — check retrieval stage first.
2. Claims already in `verification_results` are skipped unless `--force` is passed.
3. `not_verifiable` claims are still verified — only claims with `claim_type` filtering would be excluded, and that filtering is not present by default.

**If you need to change the evidence schema:**
1. Add fields to `normalize_structured_evidence()` in `verifier.py`.
2. Update `load_cached_evidence_mixed()` to populate the new field from the DB query.
3. Confirm `workspace_data.py` is updated to pass the new field to the UI.
4. Do not alter the `evidence_json` blob shape casually — it is a serialized contract read by multiple consumers.

---

## CLI Entry Points

```bash
# Verify all claims for a single paper (uses cached retrieval)
gsr verify --paper-id "<paper_id>"

# Force re-verification (bypass already-verified skip)
gsr verify --paper-id "<paper_id>" --force

# Verify a specific review only
gsr verify --review-id "<review_id>"

# Use a specific model
gsr verify --paper-id "<paper_id>" --model gpt-4o

# Control evidence count per claim
gsr verify --paper-id "<paper_id>" --top-k 8

# Allow live retrieval fallback if no cache exists
gsr verify --paper-id "<paper_id>" --allow-live-retrieval

# Generate a verification report after verifying
gsr report verification --paper-id "<paper_id>"
```
