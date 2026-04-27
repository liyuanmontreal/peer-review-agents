# paper_retrieval/vision — Enhancement Layer

This subpackage provides **optional** figure/table grounding enrichment for Stage 3.  
It sits **on top of** the stable baseline (PyMuPDF V2 parsing + caption-first identity + hybrid retrieval).  
Nothing in this subpackage is active by default in the `gsr run` pipeline.

---

## Role in Stage 3

The stable baseline establishes `EvidenceObject` identity from caption-prefix detection in the PyMuPDF span stream.  
Vision capabilities enrich objects *after* identity is set — they do not replace grounding.

**Storage rule:** All vision outputs land in `metadata_json` only.  
No vision result changes `object_type`, `retrieval_text`, `embeddings`, or primary `bbox` fields.  
No schema migration is needed to enable or disable any capability here.

**Coordinate convention:** All bboxes are in PDF user-space (points, origin top-left, per-page).

---

## File Index

| File | Status | Purpose |
|---|---|---|
| `bbox_refinement.py` | Optional — not active in `gsr run` index step | Dispatch layer for detector-based bbox refinement |
| `grounding_dino.py` | Optional — default detector when refinement runs | Open-vocab figure/table detector (P1.5) |
| `rtdetr_detector.py` | Experimental — P1.6 spike; not recommended | RT-DeTR COCO/O365; NOT document-layout specialized |
| `paddlex_layout_detector.py` | Experimental — extra install required | PicoDet-L document layout specialist (17 classes) |
| `ocr_lighton.py` | Experimental — two activation paths | LightOnOCR-2 figure text extraction |
| `figure_semantics_qwen.py` | Phase P2 — wired but off by default | Qwen2.5-VL figure type classifier |

---

## `bbox_refinement.py`

**Status:** Optional. Two call sites, one disabled in production.

**Entry point:** `refine_bbox_for_paper(paper_id, detector="groundingdino", ...)`

**Call sites:**
- `paper_retrieval/__init__.py` — `index_paper(..., use_bbox_refine=True)`. Disabled: `cli.py` hardcodes `use_bbox_refine=False` in `gsr run` with explicit comment.
- `paper_retrieval/retrieval.py` — `retrieve_all_claims_for_paper(..., use_bbox_refine=True)`. **This is the preferred path.** Selective, post-retrieval, per-claim.

**Detector dispatch:** `"groundingdino"` (default) → `"rtdetr"` → `"paddlex_layout"`

**Output:** Refined bbox written to `metadata_json["refined_bbox"]` on the `EvidenceObject`. Does not touch `retrieval_text`, embeddings, or object identity.

**Safe edits:** Never change the no-identity-change rule. Do not route a new detector without implementing the same interface as `grounding_dino.py`. Do not change the `metadata_json`-only storage contract.

---

## `grounding_dino.py`

**Status:** Optional — active only when bbox refinement runs.

**Model:** `IDEA-Research/grounding-dino-base` (loaded once at module level; singleton pattern)

**Text prompt:** `"figure . table . chart . graph . diagram . plot."`

**Key thresholds:**
- `_BOX_THRESHOLD = 0.35`
- `_FILTER_MIN_AREA = 300.0` (px²)
- `_FILTER_MAX_COVERAGE = 0.85` (rejects boxes that fill nearly the whole page)

**Graceful degradation:** If the model fails to load or inference errors, the call site logs a warning and continues without refinement.

**Safe edits:** Adjust thresholds carefully — `_BOX_THRESHOLD` controls precision/recall trade-off. Changes to the text prompt affect which classes are detected. Test on real PDFs with figures and tables before changing either.

---

## `rtdetr_detector.py`

**Status:** Experimental. P1.6 spike. **Not recommended for production use.**

**Model:** `PekingU/rtdetr_r50vd_coco_o365` (COCO + Objects365)

**Critical limitation (documented in module docstring):** RT-DeTR is NOT document-layout specialized. COCO/O365 has no explicit figure, chart, or table class. Label mapping to document objects is approximate.

**Override:** `GSR_RTDETR_MODEL` env var

**When to use:** Only for experimental comparison against GroundingDINO. Do not activate by default.

**Safe edits:** Do not change the label mapping without understanding COCO/O365 class taxonomy. The approximate mapping is intentional and its limitations are documented.

---

## `paddlex_layout_detector.py`

**Status:** Experimental. Requires extra install.

**Model:** `PicoDet-L_layout_17cls` — document-layout specialist with 17 classes (figures, tables, titles, headers, etc.)

**Override:** `GSR_PADDLEX_LAYOUT_MODEL` env var

**Install:** `pip install paddlex` (not in base requirements)

**Same interface as `grounding_dino.py`** — can be used as drop-in detector via `bbox_detector="paddlex_layout"`.

**Safe edits:** Verify PaddleX version compatibility if upgrading. The 17-class model is significantly more document-aware than RT-DeTR; prefer this over RT-DeTR for any document layout experiments.

---

## `ocr_lighton.py`

**Status:** Experimental. Two activation paths.

**Model:** `lightonai/LightOnOCR-2-1B`

**Install:** `pip install lightonai` (not in base requirements)

**Activation paths:**
1. Index time: `index_paper(..., use_figure_ocr=True)` → called from `evidence_builder.py`
2. Verify time: `verify_all_claims(..., selective_figure_ocr=True)` → called from `claim_verification/figure_escalation.py`

**Key constants:**
- `_DEFAULT_MAX_NEW_TOKENS = 256`
- `_DEFAULT_TIMEOUT_SECONDS = 120.0`
- `_MAX_CROP_LONGEST_SIDE = 1536`

**Skip reason taxonomy:** 10+ documented skip codes (e.g. `no_asset_path`, `small_figure`, `timeout`, `decode_error`). Always log the reason; do not suppress silently.

**Output:** OCR text stored in `metadata_json["ocr_text"]` on the `EvidenceObject`. Does not alter caption, retrieval_text, or embeddings.

**Safe edits:** Do not reduce timeout without profiling real figures. The skip taxonomy is important for diagnosing why OCR didn't run on a given figure — preserve all codes.

---

## `figure_semantics_qwen.py`

**Status:** Phase P2. Implemented and wired. Off by default.

**Model:** `Qwen/Qwen2.5-VL-3B-Instruct` (default); override via `GSR_FIGSEM_MODEL`

**Activation:** `verify_all_claims(..., selective_figure_semantic=True)` in `claim_verification/verifier.py`

**Purpose:** Classifies figure type to enable downstream reasoning (e.g. bar_chart, line_chart, scatter_plot, heatmap, architecture_diagram, etc.)

**Key functions:**
- `ensure_figure_semantics_for_evidence_object()` — idempotent; reads `metadata_json["figure_type"]` if already cached
- `make_figsem_config_hash()` — cache key based on model config

**Output:** `metadata_json["figure_type"]` — does not change object identity, retrieval_text, or bbox.

**Safe edits:** The figure type taxonomy drives downstream LLM reasoning in `verifier.py`. Adding a new type requires corresponding handling in the verification prompt. Do not rename existing type labels without updating the verifier.

---

## Safe Editing Guidelines

1. **Never remove graceful degradation.** Every vision component must fail cleanly if the model isn't available. The stable baseline must always work without vision.

2. **Never make optional capabilities required.** If you add a new vision step, it must be off-by-default, gated by an explicit flag.

3. **Never write vision results to primary schema fields.** All outputs go to `metadata_json`. This preserves the invariant that vision can be disabled without a schema migration.

4. **Never change detector selection semantics in `bbox_refinement.py`** without updating all three detector implementations to stay interface-compatible.

5. **Test on real PDFs** before changing thresholds (`_BOX_THRESHOLD`, `_FILTER_MIN_AREA`, `_FILTER_MAX_COVERAGE`). These are tuned heuristics, not arbitrary constants.

6. **Prefer selective (per-claim) refinement over full-paper refinement.** The `gsr run` pipeline intentionally disables full-paper bbox refinement at index time. This is a deliberate latency/quality trade-off.
