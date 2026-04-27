# claim_extraction — Module 2: Challengeable Claim Extraction from Peer Reviews

## Role in the GSR Pipeline

`claim_extraction` is **pipeline stage 2** — it sits between ingestion and paper retrieval.

Its job is to read peer review text (loaded from SQLite) and decompose it into discrete, **binary-verifiable factual claims** that can be checked against the paper PDF. Everything downstream (retrieval, verification, evidence cards, verdicts) depends on the quality and coverage of what this module produces.

```
ingestion → [ claim_extraction ] → paper_retrieval → verification
                    ↑
     reviews (SQLite) → LLM extraction → filter → persist claims
```

The module is consumed by:
- `gsr extract --paper-id ...` (CLI, via `cli.py`)
- `extract_all_claims()` (batch path for API progress)
- `paper_retrieval` (reads claims from the `claims` table)
- `claim_verification` (reads claims from the `claims` table)
- `reporting/workspace_data.py` (loads claims + verbatim quotes for UI panel)

---

## Core Design Philosophy

1. **Claims are binary-verifiable, paper-grounded assertions.** The extraction prompt is calibrated to yield claims that can be answered YES/NO by reading the paper. Subjective opinions, vague critiques, and suggestions are explicitly excluded.

2. **High recall over aggressive precision at extraction time.** The system intentionally extracts borderline claims and relies on downstream filtering and verification rather than discarding at the prompt level. If claim counts seem low, suspect the thresholds or the prompt — not an abundance of challenging claims.

3. **Field-by-field source attribution is structural, not optional.** Every claim carries a `source_field` (e.g. `summary`, `strengths`, `weaknesses`, `questions`, or any other Tier A/B canonical field name resolved by the field policy). This is required by the UI (ReviewPanel highlights) and by the data contract with verification. Never lose source attribution.

4. **Extraction modes are a research dial, not a UI concern.** Three modes exist (`field`, `review`, `grouped`) with different LLM call patterns. The `field` mode is the stable default. `review` and `grouped` modes fall back to `field` on error.

5. **Config hashing enables idempotent reruns.** Each extraction configuration (model, fields, thresholds) produces a deterministic `config_hash`. Already-processed review+config pairs are skipped unless `--force` is passed.

---

## Key Concepts

### Extraction Modes

**`field` mode** (default) — one LLM call per review field
- Each of `summary`, `strengths`, `weaknesses`, `questions` is sent independently
- `_extract_claims_from_field()` handles each call
- `on_field_done` callback enables per-field progress reporting

**`review` mode** — one LLM call for the full review (all fields together)
- `_extract_claims_from_review()` sends the complete review in a single prompt
- Model must return `source_field` per claim; invalid values are dropped
- Falls back to `field` mode on error

**`grouped` mode** — two LLM calls per review, by field group
- Group 1: `summary` + `strengths`; Group 2: `weaknesses` + `questions`
- Uses `_extract_claims_from_review()` with fields zeroed outside the group
- Falls back to `field` mode on error

In **field mode**, the set of fields actually extracted is determined by `_resolve_extraction_fields()`. When the review has a `raw_fields` JSON blob (heterogeneous OpenReview schemas), the field policy is consulted instead of using `DEFAULT_FIELDS` directly:
- Each raw field key is canonicalized via `canonicalize_field_name()`
- Eligibility is checked via `should_extract_field()` (Tier A = always include, Tier B = include if ≥ 100 chars, Tier C = always skip)
- Unknown field names get the same substantive-text heuristic as Tier B

When `raw_fields` is absent, the caller-supplied `fields` tuple is used directly (backward-compatible path).

**📍 Code Pointers:**
- `src/gsr/claim_extraction/extractor.py` → `extract_review_claims()` (mode dispatch)
- `src/gsr/claim_extraction/extractor.py` → `_extract_claims_from_field()`, `_extract_claims_from_review()`, `_resolve_extraction_fields()`
- `src/gsr/claim_extraction/field_policy.py` → `canonicalize_field_name()`, `should_extract_field()`, `FIELD_POLICY_VERSION`, `TIER_A`, `TIER_B`, `TIER_C`

---

### Field Policy (`field_policy.py`)

`field_policy.py` is the single source of truth for OpenReview field name normalization and extraction eligibility. It handles the fact that different venues use different field names for the same content.

**Key exports:**
- `FIELD_POLICY_VERSION` — version string embedded in every `_config_hash()`. **Bumping this invalidates the dedup cache** for all reviews (equivalent to `--force`) without requiring a schema change.
- `canonicalize_field_name(raw)` — maps raw OpenReview field keys to canonical names (e.g. `"summary_of_the_paper"` → `"summary"`, `"overall_assessment"` → `"final_justification"`)
- `should_extract_field(canonical, text)` → `(include, reason)` — applies tier policy
- `NORMALIZED_FIELD_LABELS` / `NORMALIZED_FIELD_ORDER` — display metadata consumed by the UI

**Tier policy:**

| Tier | Canonical fields | Eligibility |
|---|---|---|
| A | `summary`, `strengths`, `weaknesses`, `questions`, `final_justification`, `limitations` | Always include if non-empty |
| B | `quality`, `clarity`, `significance`, `soundness`, `presentation`, `contribution`, `ethics`, `reproducibility` | Include only if ≥ 100 chars |
| C | `rating`, `confidence`, `ethical_concerns`, flags, acknowledgements | Always skip |

**Safe editing:** Bump `FIELD_POLICY_VERSION` whenever tier assignments or alias mappings change in a way that would produce materially different extraction results. Do not rename or remove canonical field names without updating the UI's `NORMALIZED_FIELD_LABELS` and downstream `source_field` consumers.

---

### Claim Taxonomy and Filtering

**Categories (strict 6-value taxonomy):**

| Category | Meaning |
|---|---|
| `methodology` | Missing/incorrect experimental design or ablations |
| `results` | Numbers, error bars, statistical tests, quantitative evidence |
| `comparison` | Baselines, fairness of comparison, prior methods |
| `literature` | Citations, originality claims |
| `scope` | Datasets, domains, languages, evaluation conditions |
| `reproducibility` | Code, hyperparameters, training details, artifacts |

The LLM may return aliases (e.g. `methods`, `baseline`, `related_work`). These are normalized by `normalize_category()`. Claims with unmappable categories are silently dropped.

**Threshold filtering:**

| Parameter | Default | Meaning |
|---|---|---|
| `min_challengeability` | `0.6` | Minimum binary-checkability score (0–1) |
| `min_confidence` | `0.5` | Minimum extraction confidence (0–1) |

Both thresholds are applied in `_keep_claim()` after the LLM call. Raising `min_challengeability` reduces yield; lowering it increases noise passed to retrieval.

**📍 Code Pointers:**
- `src/gsr/claim_extraction/extractor.py` → `MIN_CHALLENGEABILITY`, `MIN_CONFIDENCE`, `ALLOWED_CATEGORIES`, `CATEGORY_ALIASES`
- `src/gsr/claim_extraction/extractor.py` → `normalize_category()`, `_keep_claim()`

---

### The Claim Object

The **canonical output shape** of this module is a dict row persisted to the `claims` table. Downstream consumers read directly from SQLite.

Key fields:

| Field | Purpose |
|---|---|
| `id` | SHA256 of `review_id|source_field|claim_index|config_hash` |
| `review_id` | Source review |
| `paper_id` | Paper the review belongs to |
| `source_field` | Canonical field name (e.g. `summary`, `weaknesses`, `final_justification`; any Tier A/B field) |
| `source_field_raw` | Original OpenReview key when it differs from canonical (e.g. `"summary_of_the_paper"`); null if same |
| `claim_text` | Self-contained reformulation of the claim |
| `verbatim_quote` | Exact span from the review (used for UI highlight) |
| `claim_type` | Always stored as `"factual"` (LLM returns 4-value taxonomy but storage hardcodes this; field is vestigial) |
| `confidence` | LLM extraction confidence (0–1) |
| `challengeability` | Binary-checkability score (0–1) |
| `category` | Normalized 6-value taxonomy |
| `binary_question` | YES/NO form of the claim, answerable from the paper |
| `why_challengeable` | One-sentence rationale for verifiability |
| `calibrated_score` | Optional faithfulness score (1–5, from `ClaimScorer`) |

**📍 Code Pointers:**
- `src/gsr/claim_extraction/prompts.py` → `ExtractedClaim`, `ReviewExtractedClaim` (Pydantic response models)
- `src/gsr/claim_extraction/storage.py` → `BASE_DDL` (claims table DDL)

---

### Prompts and Response Models

The system uses **structured output** — LiteLLM is called with a Pydantic `response_format`, and the response is parsed directly into a typed model.

Two response types exist:

- `ExtractionResponse` — used in `field` mode; list of `ExtractedClaim` (no `source_field`)
- `ReviewExtractionResponse` — used in `review`/`grouped` modes; list of `ReviewExtractedClaim` (includes `source_field`)

The system prompt enforces the binary-verifiability definition. The user prompt varies by mode:
- `USER_PROMPT_TEMPLATE` — single-field variant
- `REVIEW_USER_PROMPT_TEMPLATE` — multi-field variant with per-field processing instructions

**📍 Code Pointers:**
- `src/gsr/claim_extraction/prompts.py` → `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE`, `REVIEW_USER_PROMPT_TEMPLATE`
- `src/gsr/claim_extraction/prompts.py` → `ExtractedClaim`, `ExtractionResponse`, `ReviewExtractedClaim`, `ReviewExtractionResponse`

---

### LLM Wrapper

`llm.py` is a thin LiteLLM wrapper. It is intentionally minimal — no retry logic, no streaming, no async. The model is resolved via `gsr.config.get_llm_model()` or an explicit `model` override.

**📍 Code Pointers:**
- `src/gsr/claim_extraction/llm.py` → `complete()`, `get_model_id()`
- Model config: `gsr/config.py` → `get_llm_model()`

---

### Optional Faithfulness Scoring (ClaimScorer)

`scorer.py` provides an optional post-hoc faithfulness check using a secondary LLM (default: Llama-3.1-8B via `featherless-ai`). It scores how faithfully each extracted claim is grounded in the source review text (1–5 scale).

This is a calibration/research tool — it is **disabled by default** and must be explicitly enabled via `--score-claims`. Scores are stored in `claims.calibrated_score` and `calibrated_score_norm`.

**📍 Code Pointers:**
- `src/gsr/claim_extraction/scorer.py` → `ClaimScorer`, `ClaimScorerConfig`, `score_claim()`

---

### Experiment Tracking

Each run of `extract_all_claims()` creates an **experiment** row (UUID) in the `experiments` table. Each per-review run creates an `extraction_runs` row linked to the experiment. This enables:
- Comparing extraction configurations across runs
- Diagnosing per-review failures
- Filtering claims by config via `config_hash`

`_config_hash()` hashes: `model_id`, `fields`, `min_confidence`, `min_challengeability`, and **`FIELD_POLICY_VERSION`** from `field_policy.py`. Bumping `FIELD_POLICY_VERSION` automatically invalidates the dedup cache — re-extraction runs without `--force`.

**📍 Code Pointers:**
- `src/gsr/claim_extraction/storage.py` → `create_experiment()`, `ensure_experiment_schema()`
- `src/gsr/claim_extraction/extractor.py` → `_config_hash()`
- `src/gsr/claim_extraction/field_policy.py` → `FIELD_POLICY_VERSION`

---

### Batch Extraction and Parallelism

`extract_all_claims()` is the batch entry point. It:
1. Creates an experiment record
2. Loads all reviews for a paper (or all papers)
3. Filters already-processed review+config pairs (unless `--force`)
4. Runs `extract_review_claims()` in a `ThreadPoolExecutor` (default `max_workers=2`)
5. Persists results via `save_extraction_results()`
6. Emits `GSR_PROGRESS extract_claims {completed} {total}` per review for the API progress layer

**📍 Code Pointers:**
- `src/gsr/claim_extraction/extractor.py` → `extract_all_claims()`, `_already_processed()`
- Progress signal: look for `GSR_PROGRESS extract_claims` in `extractor.py`

---

## Storage Schema

Three tables owned or extended by this module (DDL in `storage.py` → `init_claims_db()`):

| Table | Contents |
|---|---|
| `claims` | One row per extracted claim, with all fields above |
| `extraction_runs` | One row per review extraction attempt (success or error), linked to experiment |
| `experiments` | One row per `extract_all_claims()` invocation, with params JSON |

Schema is managed via additive migrations (`_ensure_column()`). The tables may exist in older databases with missing columns — `init_claims_db()` handles this safely.

**📍 Code Pointers:**
- `src/gsr/claim_extraction/storage.py` → `init_claims_db()`, `save_extraction_results()`, `BASE_DDL`

---

## High-Level Orchestration

```
extract_all_claims(conn, paper_id=..., model=..., fields=..., extract_mode=...)
    → create_experiment()             # experiment record
    → _load_reviews()                 # load reviews from SQLite
    → _already_processed()            # skip if config_hash already succeeded
    → extract_review_claims()         # per review (threaded)
        → _extract_claims_from_field() or _extract_claims_from_review()
        → complete()                  # LiteLLM call → Pydantic parse
        → normalize_category()        # canonicalize taxonomy
        → _keep_claim()               # threshold filter
        → ClaimScorer.score_claim()   # optional faithfulness score
    → save_extraction_results()       # persist claims + runs
    → GSR_PROGRESS extract_claims     # progress signal
```

**📍 Code Pointers:**
- `src/gsr/claim_extraction/__init__.py` → `extract_all_claims`, `extract_review_claims` (public API)
- `src/gsr/claim_extraction/extractor.py` → full orchestration

---

## Reasoning Guidelines for Future Agents

**If claim counts are too low:**
1. Check `min_challengeability` and `min_confidence` thresholds — the defaults filter aggressively.
2. Check `extraction_runs` for error rows — LLM failures silently produce 0 claims.
3. Inspect `ALLOWED_CATEGORIES` and `CATEGORY_ALIASES` — LLM category drift causes silent drops.
4. Check whether the right `fields` are being passed — `DEFAULT_FIELDS = ("weaknesses", "strengths", "questions", "summary")`.

**If claim source_field attribution looks wrong:**
1. In `field` mode, `source_field` is set by `_resolve_extraction_fields()` — never by the LLM. Check that `field_policy.py`'s `canonicalize_field_name()` is mapping the raw field key correctly; check `should_extract_field()` isn't skipping the field due to tier or length.
2. In `review`/`grouped` mode, `source_field` comes from the LLM. Invalid values are dropped by `normalize_source_field()`.
3. If ReviewPanel highlights are on the wrong review section, check `claims.source_field` in SQLite before touching the UI.
4. If a field from a non-standard venue is missing, check whether its raw key appears in `CANONICAL_ALIASES` in `field_policy.py` and whether it lands in Tier A/B.

**If verbatim_quote is missing or mismatched:**
1. `verbatim_quote` is an LLM output — it is not extracted programmatically from the review text.
2. If quotes are drifting or hallucinated, adjusting the prompt in `prompts.py` is the right lever, not a post-processing patch.

**If you need to change the claim schema:**
1. Add columns via `_ensure_column()` in `storage.py` — do not alter `BASE_DDL` for fields that may exist in production databases.
2. Update `save_extraction_results()` to persist the new field.
3. Confirm downstream consumers in `workspace_data.py` and `claim_verification` still work.

**Before changing thresholds:**
- `min_challengeability` and `min_confidence` affect claim yield nonlinearly. Test on real data, not assumptions.
- The `config_hash` changes when thresholds change — previously skipped reviews will be re-extracted on the next run.

---

## CLI Entry Points

```bash
# Extract claims for a single paper
gsr extract --paper-id "<paper_id>"

# Force re-extraction (bypass config_hash skip)
gsr extract --paper-id "<paper_id>" --force

# Use review-level extraction mode
gsr extract --paper-id "<paper_id>" --extract-mode review

# Override thresholds
gsr extract --paper-id "<paper_id>" --min-challengeability 0.5 --min-confidence 0.4

# Enable faithfulness scoring
gsr extract --paper-id "<paper_id>" --score-claims
```
