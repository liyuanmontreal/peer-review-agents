"""Selective verify-time figure OCR escalation — Phase 2.0.

Provides three public helpers used by verify_all_claims() when
``selective_figure_ocr=True``:

- ``make_ocr_config_hash()``  — stable string for cache validity
- ``check_escalation_trigger()`` — rule-based trigger (no extra LLM call)
- ``pick_top_figure_evidence()`` — highest-ranked figure from retrieved evidence
- ``ensure_figure_ocr_for_evidence_object()`` — cache check + OCR + persist

Cache design
------------
OCR results are persisted in ``evidence_objects.metadata_json`` under the
``figure_ocr_*`` schema already used by the retrieve-time full OCR path.
A ``figure_ocr_config_hash`` field is stored alongside every outcome so the
cache can be invalidated when OCR parameters change.

Negative outcomes are cached as well so the same figure is never retried
within the same config (timeout, low_quality, etc.).

The ``ensure_figure_ocr_for_evidence_object()`` function also updates
``evidence_objects.content_text`` with accepted OCR text so the same
evidence object can be re-used by later claims without re-running OCR.

Note: ``evidence_embeddings`` are NOT recomputed in this phase; retrieval
scoring is unchanged.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Explicit figure reference pattern: "Figure 3", "Figures 2", "Fig. 4", "Fig 5"
_FIGURE_REF_PATTERN = re.compile(r"\b(?:Figures?|Fig\.?)\s*\d+\b", re.IGNORECASE)

# Lexical cue patterns for figure-related language (used by heuristic gate)
# Strong cues: direct figure words + chart/plot/visual/diagram words
_STRONG_CUE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bfigures?\b"), "figure"),
    (re.compile(r"\bfig\b"), "fig"),
    (re.compile(r"\bfig\."), "fig."),
    (re.compile(r"\bplots?\b"), "plot"),
    (re.compile(r"\bcharts?\b"), "chart"),
    (re.compile(r"\bgraphs?\b"), "graph"),
    (re.compile(r"\bcurves?\b"), "curve"),
    (re.compile(r"\bscatter\b"), "scatter"),
    (re.compile(r"\bhistogram\b"), "histogram"),
    (re.compile(r"\bheatmap\b"), "heatmap"),
    (re.compile(r"\bbar\s+chart\b"), "bar chart"),
    (re.compile(r"\bline\s+chart\b"), "line chart"),
    (re.compile(r"\bdiagram\b"), "diagram"),
    (re.compile(r"\barchitecture\b"), "architecture"),
    (re.compile(r"\bpipeline\b"), "pipeline"),
    (re.compile(r"\bframework\b"), "framework"),
    (re.compile(r"\boverview\b"), "overview"),
]
# Weak cues: shown/illustrated/visual-context phrases
_WEAK_CUE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bshown\b"), "shown"),
    (re.compile(r"\billustrated\b"), "illustrated"),
    (re.compile(r"\bdepicted\b"), "depicted"),
    (re.compile(r"\bvisualized\b"), "visualized"),
    (re.compile(r"\bsee\b"), "see"),
    (re.compile(r"\bas\s+shown\b"), "as shown"),
    (re.compile(r"\bin\s+the\s+figure\b"), "in the figure"),
    (re.compile(r"\bin\s+the\s+plot\b"), "in the plot"),
    (re.compile(r"\bfrom\s+the\s+plot\b"), "from the plot"),
    (re.compile(r"\bthe\s+curve\b"), "the curve"),
    (re.compile(r"\bthe\s+graph\b"), "the graph"),
]

# Negative OCR outcomes that should not be retried when config hash matches
_TERMINAL_NEGATIVE_OUTCOMES = frozenset({
    "too_short",
    "duplicate_caption",
    "low_quality",
    "timeout",
    "inference_failed",
    "skipped_no_bbox",
    "skipped_small_crop",
})

# Confidence below this threshold triggers escalation (when a figure is present)
_LOW_CONFIDENCE_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Config hash
# ---------------------------------------------------------------------------

def make_ocr_config_hash(
    *,
    max_new_tokens: int,
    timeout_seconds: float,
    max_crop_longest_side: int,
    render_scale: float,
) -> str:
    """Return a stable string used as the OCR cache validity key.

    Changing any of these parameters invalidates the cache, causing a
    re-OCR on the next call to ``ensure_figure_ocr_for_evidence_object()``.
    """
    return (
        f"lightonocr2_tok{max_new_tokens}"
        f"_t{int(timeout_seconds)}"
        f"_side{max_crop_longest_side}"
        f"_scale{render_scale:.1f}"
    )


# ---------------------------------------------------------------------------
# Lexical gate helper
# ---------------------------------------------------------------------------

def detect_figure_lexical_cues(claim: dict[str, Any]) -> dict[str, Any]:
    """Detect figure-related lexical cues in claim text fields.

    Inspects ``claim_text``, ``verbatim_quote``, and ``binary_question``
    (if present), concatenated and lowercased.

    Gate passes if: at least 1 strong cue OR at least 2 weak cues.

    Returns:
        has_strong_cue, has_weak_cue, strong_cues, weak_cues,
        lexical_gate_passed, text_sample
    """
    text = " ".join(
        p for p in [
            claim.get("claim_text") or "",
            claim.get("verbatim_quote") or "",
            claim.get("binary_question") or "",
        ]
        if p
    ).lower()

    strong: list[str] = [label for pat, label in _STRONG_CUE_PATTERNS if pat.search(text)]
    weak: list[str] = [label for pat, label in _WEAK_CUE_PATTERNS if pat.search(text)]

    return {
        "has_strong_cue": bool(strong),
        "has_weak_cue": bool(weak),
        "strong_cues": strong,
        "weak_cues": weak,
        "lexical_gate_passed": len(strong) >= 1 or len(weak) >= 2,
        "text_sample": text[:120],
    }


# ---------------------------------------------------------------------------
# Escalation trigger
# ---------------------------------------------------------------------------

def check_escalation_trigger(
    claim: dict[str, Any],
    evidence_items: list[dict[str, Any]],
    *,
    first_pass_verdict: str | None,
    first_pass_confidence: float | None,
    enable_heuristic: bool = False,
) -> dict[str, Any]:
    """Return a structured trigger result dict for escalation policy (Phase 2.3a).

    Policy (evaluated in priority order):
    1. No figure evidence → no escalation (skip_reason="no_figure").
    2. Explicit figure reference in claim text → escalate regardless of rank/lexical
       (trigger="explicit_figure_ref"). Always evaluated.
    3. [only if enable_heuristic=True] insufficient_evidence + rank <= 3 →
       escalate only if lexical gate passes.
    4. [only if enable_heuristic=True] Low confidence (< 0.55) + rank <= 2 →
       escalate only if lexical gate passes.
    5. No trigger met → skip_reason="trigger_not_met".

    When enable_heuristic=False (default), rules 3 and 4 are skipped entirely.
    The result dict will still include the heuristic audit fields with their
    default (non-escalated) values.

    Returns a dict with keys:
      should_escalate, trigger, figure_present, top_figure_rank,
      top_figure_label, explicit_figure_ref_detected,
      rejected_by_rank_policy, rejected_by_lexical_policy,
      lexical_gate_applied, lexical_gate_passed,
      lexical_strong_cues, lexical_weak_cues, skip_reason
    """
    claim_id = claim.get("id")

    # Find top figure evidence and its 1-based rank in evidence list
    figure_present = False
    top_figure_rank: int | None = None
    top_figure_label: str | None = None
    for idx, ev in enumerate(evidence_items):
        if ev.get("object_type") == "figure":
            figure_present = True
            top_figure_rank = idx + 1
            top_figure_label = ev.get("label")
            break

    # Explicit figure reference detection (claim + verbatim_quote + binary_question)
    ref_text = " ".join(p for p in [
        claim.get("claim_text") or "",
        claim.get("verbatim_quote") or "",
        claim.get("binary_question") or "",
    ] if p)
    explicit_figure_ref_detected = bool(_FIGURE_REF_PATTERN.search(ref_text))

    result: dict[str, Any] = {
        "should_escalate": False,
        "trigger": None,
        "figure_present": figure_present,
        "top_figure_rank": top_figure_rank,
        "top_figure_label": top_figure_label,
        "explicit_figure_ref_detected": explicit_figure_ref_detected,
        "rejected_by_rank_policy": False,
        "rejected_by_lexical_policy": False,
        "lexical_gate_applied": False,
        "lexical_gate_passed": False,
        "lexical_strong_cues": [],
        "lexical_weak_cues": [],
        "skip_reason": None,
    }

    # Rule 1: no figure
    if not figure_present:
        result["skip_reason"] = "no_figure"
        log.info("[verify-figure-ocr] no_figure claim=%s", claim_id)
        return result

    # Rule 2: explicit figure reference — rank-unrestricted, lexical-gate-bypassed
    if explicit_figure_ref_detected:
        result["should_escalate"] = True
        result["trigger"] = "explicit_figure_ref"
        result["lexical_gate_applied"] = False
        result["lexical_gate_passed"] = True  # bypass, not blocked
        log.info(
            "[verify-figure-ocr] trigger=explicit_figure_ref figure=%s rank=%s claim=%s",
            top_figure_label, top_figure_rank, claim_id,
        )
        return result

    # Rules 3–4: heuristic paths — only evaluated when enable_heuristic=True
    if not enable_heuristic:
        result["skip_reason"] = "trigger_not_met"
        return result

    # Rule 3: insufficient_evidence + rank gate (rank <= 3) + lexical gate
    if first_pass_verdict == "insufficient_evidence":
        if top_figure_rank is not None and top_figure_rank <= 3:
            lex = detect_figure_lexical_cues(claim)
            result.update({
                "lexical_gate_applied": True,
                "lexical_gate_passed": lex["lexical_gate_passed"],
                "lexical_strong_cues": lex["strong_cues"],
                "lexical_weak_cues": lex["weak_cues"],
                "trigger": "insufficient_with_figure_rank_le_3",
            })
            if lex["lexical_gate_passed"]:
                result["should_escalate"] = True
                log.info(
                    "[verify-figure-ocr] trigger=insufficient_with_figure_rank_le_3 figure=%s rank=%s claim=%s",
                    top_figure_label, top_figure_rank, claim_id,
                )
            else:
                result["rejected_by_lexical_policy"] = True
                result["skip_reason"] = "lexical_gate_failed"
                log.info(
                    "[sel-fig-ocr] lexical gate blocked claim=%s trigger=insufficient_with_figure_rank_le_3 rank=%s strong=%s weak=%s",
                    claim_id, top_figure_rank, lex["strong_cues"], lex["weak_cues"],
                )
        else:
            result["rejected_by_rank_policy"] = True
            result["skip_reason"] = "rank_policy"
            log.info(
                "[verify-figure-ocr] skip_rank_policy figure=%s rank=%s reason=insufficient_with_figure claim=%s",
                top_figure_label, top_figure_rank, claim_id,
            )
        return result

    # Rule 4: low confidence + rank gate (rank <= 2) + lexical gate
    if (
        first_pass_confidence is not None
        and first_pass_confidence < _LOW_CONFIDENCE_THRESHOLD
    ):
        if top_figure_rank is not None and top_figure_rank <= 2:
            lex = detect_figure_lexical_cues(claim)
            result.update({
                "lexical_gate_applied": True,
                "lexical_gate_passed": lex["lexical_gate_passed"],
                "lexical_strong_cues": lex["strong_cues"],
                "lexical_weak_cues": lex["weak_cues"],
                "trigger": "low_confidence_with_figure_rank_le_2",
            })
            if lex["lexical_gate_passed"]:
                result["should_escalate"] = True
                log.info(
                    "[verify-figure-ocr] trigger=low_confidence_with_figure_rank_le_2 figure=%s rank=%s claim=%s",
                    top_figure_label, top_figure_rank, claim_id,
                )
            else:
                result["rejected_by_lexical_policy"] = True
                result["skip_reason"] = "lexical_gate_failed"
                log.info(
                    "[sel-fig-ocr] lexical gate blocked claim=%s trigger=low_confidence_with_figure_rank_le_2 rank=%s strong=%s weak=%s",
                    claim_id, top_figure_rank, lex["strong_cues"], lex["weak_cues"],
                )
        else:
            result["rejected_by_rank_policy"] = True
            result["skip_reason"] = "rank_policy"
            log.info(
                "[verify-figure-ocr] skip_rank_policy figure=%s rank=%s reason=low_confidence_with_figure claim=%s",
                top_figure_label, top_figure_rank, claim_id,
            )
        return result

    # Rule 5: no trigger met
    result["skip_reason"] = "trigger_not_met"
    return result


# ---------------------------------------------------------------------------
# Evidence selection
# ---------------------------------------------------------------------------

def pick_top_figure_evidence(
    evidence_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the highest-ranked figure evidence object from retrieved evidence.

    Evidence items are assumed to be ordered by retrieval rank (ascending).
    Returns None if no figure evidence is present.
    """
    for ev in evidence_items:
        if ev.get("object_type") == "figure":
            return ev
    return None


# ---------------------------------------------------------------------------
# OCR cache + run
# ---------------------------------------------------------------------------

def ensure_figure_ocr_for_evidence_object(
    conn: sqlite3.Connection,
    evidence_item: dict[str, Any],
    pdf_path: str,
    *,
    config_hash: str,
) -> tuple[dict[str, Any], str]:
    """Ensure OCR is available for a figure evidence object, using a DB cache.

    Steps:
    1. Load current metadata from ``evidence_objects``.
    2. If ``figure_ocr_config_hash`` matches and outcome is ``accepted``:
       return the cached OCR text (cache_hit_positive).
    3. If config hash matches and outcome is a terminal negative: skip
       re-OCR (cache_hit_negative).
    4. Otherwise: run LightOnOCR-2, persist result to ``evidence_objects``,
       and return the enriched evidence item (fresh_ocr).

    Returns:
        (updated_evidence_item, cache_status) where cache_status is one of:
        - ``"cache_hit_positive"``  — cached accepted OCR reused
        - ``"cache_hit_negative"``  — cached negative outcome, OCR skipped
        - ``"fresh_ocr"``           — OCR just ran (outcome may be accepted or failed)
        - ``"error"``               — unexpected problem (missing ID, missing DB row)

    Side effects:
        On fresh_ocr: updates ``evidence_objects.metadata_json`` and
        (when accepted) ``evidence_objects.content_text``.
        Does NOT update ``evidence_embeddings`` — retrieval is unchanged.
    """
    evidence_object_id = evidence_item.get("evidence_object_id")
    if not evidence_object_id:
        log.warning("[verify-figure-ocr] evidence_item missing evidence_object_id")
        return evidence_item, "error"

    # --- Load current DB state ---
    row = conn.execute(
        "SELECT metadata_json, content_text, bbox_json FROM evidence_objects WHERE id = ?",
        (evidence_object_id,),
    ).fetchone()
    if not row:
        log.warning(
            "[verify-figure-ocr] evidence_object id=%s not in DB", evidence_object_id
        )
        return evidence_item, "error"

    try:
        metadata: dict[str, Any] = json.loads(row[0]) if row[0] else {}
    except Exception:
        metadata = {}
    current_content_text: str | None = row[1]
    try:
        bbox = json.loads(row[2]) if row[2] else None
    except Exception:
        bbox = None

    # --- Cache check ---
    cached_config = metadata.get("figure_ocr_config_hash")
    cached_outcome = metadata.get("figure_ocr_outcome")

    if cached_config == config_hash and cached_outcome is not None:
        if cached_outcome == "accepted":
            log.info(
                "[verify-figure-ocr] cache_hit_positive figure=%s id=%s",
                evidence_item.get("label"),
                evidence_object_id,
            )
            updated = dict(evidence_item)
            ocr_text = metadata.get("figure_ocr_text") or ""
            updated["figure_ocr_text"] = ocr_text
            updated["figure_ocr_attempted"] = True
            if ocr_text:
                updated["content_text"] = ocr_text
                updated["text"] = ocr_text
            return updated, "cache_hit_positive"

        if cached_outcome in _TERMINAL_NEGATIVE_OUTCOMES:
            log.info(
                "[verify-figure-ocr] cache_hit_negative figure=%s id=%s outcome=%s",
                evidence_item.get("label"),
                evidence_object_id,
                cached_outcome,
            )
            return evidence_item, "cache_hit_negative"

    # --- Fresh OCR ---
    try:
        from gsr.paper_retrieval.vision.ocr_lighton import (  # type: ignore[import]
            ocr_figure_region,
            lighton_ocr_available,
            _DEFAULT_MAX_NEW_TOKENS,
            _DEFAULT_TIMEOUT_SECONDS,
        )
    except ImportError:
        log.warning("[verify-figure-ocr] ocr_lighton not importable, skipping")
        return evidence_item, "error"

    if not lighton_ocr_available():
        log.warning("[verify-figure-ocr] LightOnOCR-2 unavailable, skipping")
        return evidence_item, "error"

    page = evidence_item.get("page") or evidence_item.get("page_start") or 1
    label = evidence_item.get("label")
    caption_text = evidence_item.get("caption_text") or ""

    log.info(
        "[verify-figure-ocr] fresh_ocr figure=%s page=%s id=%s config=%s",
        label,
        page,
        evidence_object_id,
        config_hash,
    )

    t0 = time.monotonic()
    try:
        ocr_result: dict[str, Any] = ocr_figure_region(
            pdf_path,
            page=page,
            bbox=bbox,
            caption_text=caption_text or None,
            max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        log.warning("[verify-figure-ocr] OCR exception figure=%s: %s", label, exc)
        ocr_result = {
            "figure_ocr_attempted": False,
            "figure_ocr_outcome": "inference_failed",
            "figure_ocr_text": "",
        }

    elapsed = time.monotonic() - t0
    outcome: str = ocr_result.get("figure_ocr_outcome") or (
        "accepted" if ocr_result.get("figure_ocr_text") else "inference_failed"
    )
    ocr_text = ocr_result.get("figure_ocr_text") or ""

    log.info(
        "[verify-figure-ocr] fresh_ocr done figure=%s elapsed=%.1fs outcome=%s chars=%d",
        label,
        elapsed,
        outcome,
        len(ocr_text),
    )

    # --- Persist to evidence_objects ---
    now = datetime.now(timezone.utc).isoformat()
    metadata.update({
        "figure_ocr_attempted": bool(ocr_result.get("figure_ocr_attempted", True)),
        "figure_ocr_outcome": outcome,
        "figure_ocr_text": ocr_text,
        "figure_ocr_model": ocr_result.get("figure_ocr_model") or "",
        "figure_ocr_elapsed_s": round(elapsed, 2),
        "figure_ocr_quality": ocr_result.get("figure_ocr_quality"),
        "figure_ocr_config_hash": config_hash,
        "figure_ocr_cached": True,
        "figure_ocr_last_updated_at": now,
    })

    new_content_text = current_content_text
    if outcome == "accepted" and ocr_text:
        new_content_text = ocr_text

    with conn:
        conn.execute(
            "UPDATE evidence_objects SET metadata_json = ?, content_text = ? WHERE id = ?",
            (json.dumps(metadata, ensure_ascii=False), new_content_text, evidence_object_id),
        )

    updated = dict(evidence_item)
    updated["figure_ocr_text"] = ocr_text
    updated["figure_ocr_attempted"] = bool(ocr_result.get("figure_ocr_attempted", True))
    if outcome == "accepted" and ocr_text:
        updated["content_text"] = ocr_text
        updated["text"] = ocr_text

    if outcome == "accepted":
        return updated, "fresh_ocr"
    elif outcome == "inference_failed":
        return updated, "fresh_ocr_failed"
    else:
        return updated, "fresh_ocr_rejected"
