"""Field normalization policy for heterogeneous OpenReview review schemas.

This module is the single source of truth for:
  - raw OpenReview field name → canonical field name mapping
  - tier-based extraction eligibility (A / B / C / unknown)
  - display labels and ordering for the UI

Bump ``FIELD_POLICY_VERSION`` whenever the policy changes in a way that would
produce materially different extraction results (new aliases, tier changes,
substantive-threshold changes, etc.).  The version is embedded in the
extraction config hash, so a bump automatically invalidates the dedup cache
and triggers fresh extraction on the next run — no ``--force`` needed.
"""
from __future__ import annotations

import re
import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version marker — bump this when the policy produces different extraction results
# ---------------------------------------------------------------------------

#: Included in every extraction config hash.  Changing this value
#: invalidates the existing dedup cache without touching DB rows.
FIELD_POLICY_VERSION = "heterogeneous_v1"

_NORMALIZE_RE = re.compile(r"[\s\-]+")

# ---------------------------------------------------------------------------
# Alias table
# Maps: lowercased + spaces/hyphens-collapsed-to-undersscores → canonical name
# ---------------------------------------------------------------------------
CANONICAL_ALIASES: dict[str, str] = {
    # --- summary ---
    "summary": "summary",
    "review_summary": "summary",
    "paper_summary": "summary",
    "summary_of_the_paper": "summary",
    "summary_of_the_review": "summary",
    "review": "summary",  # workshop-style single-field reviews

    # --- strengths ---
    "strengths": "strengths",
    "strength": "strengths",

    # --- weaknesses ---
    "weaknesses": "weaknesses",
    "weakness": "weaknesses",
    "weaknesses_and_limitations": "weaknesses",
    "limitations_and_weaknesses": "weaknesses",

    # --- questions ---
    "questions": "questions",
    "question": "questions",
    "questions_for_authors": "questions",
    "questions_to_authors": "questions",
    "questions_for_the_authors": "questions",

    # --- final_justification ---
    "final_justification": "final_justification",
    "finaljustification": "final_justification",
    "overall": "final_justification",
    "overall_assessment": "final_justification",
    "overall_recommendation": "final_justification",
    "summary_of_review": "final_justification",
    "justification": "final_justification",
    "justification_for_score": "final_justification",

    # --- limitations ---
    "limitations": "limitations",
    "limitation": "limitations",
    "limitations_and_societal_impact": "limitations",
    "broader_impact": "limitations",

    # --- ethics ---
    "ethics": "ethics",
    "ethics_review": "ethics",
    "ethical_concerns_review": "ethics",
    "ethical_review": "ethics",

    # --- reproducibility ---
    "reproducibility": "reproducibility",
    "reproducibility_and_replicability": "reproducibility",

    # --- Tier B: substantive rubric fields ---
    "soundness": "soundness",
    "presentation": "presentation",
    "contribution": "contribution",
    "contributions": "contribution",
    "quality": "quality",
    "paper_quality": "quality",
    "clarity": "clarity",
    "significance": "significance",
    "originality": "originality",

    # --- Tier C: metadata / yes-no / short scores (always skipped) ---
    "rating": "rating",
    "score": "rating",
    "overall_rating": "rating",
    "confidence": "confidence",
    "reviewer_confidence": "confidence",
    "ethical_concerns": "ethical_concerns",
    "flag_for_ethics_review": "flag_for_ethics_review",
    "paper_formatting_concerns": "paper_formatting_concerns",
    "code_of_conduct_acknowledgement": "code_of_conduct_acknowledgement",
    "responsible_reviewing_acknowledgement": "responsible_reviewing_acknowledgement",
    "resubmission": "resubmission",
    "first_time_reviewer": "first_time_reviewer",
}

# ---------------------------------------------------------------------------
# Tier sets (operate on canonical names)
# ---------------------------------------------------------------------------

#: Always eligible for extraction (if non-empty).
TIER_A: frozenset[str] = frozenset({
    "summary",
    "strengths",
    "weaknesses",
    "questions",
    "final_justification",
    "limitations",
})

#: Eligible only when the field text is substantive (>= _SUBSTANTIVE_MIN_CHARS).
TIER_B: frozenset[str] = frozenset({
    "quality",
    "clarity",
    "significance",
    "originality",
    "soundness",
    "presentation",
    "contribution",
    "ethics",
    "reproducibility",
})

#: Never extracted — metadata-only, yes/no, or short numeric scores.
TIER_C: frozenset[str] = frozenset({
    "rating",
    "confidence",
    "ethical_concerns",
    "flag_for_ethics_review",
    "paper_formatting_concerns",
    "code_of_conduct_acknowledgement",
    "responsible_reviewing_acknowledgement",
    "resubmission",
    "first_time_reviewer",
})

#: Minimum character count for Tier B / unknown fields to be considered substantive.
_SUBSTANTIVE_MIN_CHARS: int = 100

# ---------------------------------------------------------------------------
# Display metadata
# ---------------------------------------------------------------------------

NORMALIZED_FIELD_LABELS: dict[str, str] = {
    "summary": "Summary",
    "strengths": "Strengths",
    "weaknesses": "Weaknesses",
    "questions": "Questions",
    "final_justification": "Final Justification",
    "limitations": "Limitations",
    "ethics": "Ethics Review",
    "reproducibility": "Reproducibility",
    "quality": "Quality",
    "clarity": "Clarity",
    "significance": "Significance",
    "originality": "Originality",
    "soundness": "Soundness",
    "presentation": "Presentation",
    "contribution": "Contribution",
    "other": "Other",
}

NORMALIZED_FIELD_ORDER: list[str] = [
    "summary",
    "strengths",
    "weaknesses",
    "questions",
    "final_justification",
    "limitations",
    "ethics",
    "reproducibility",
    "quality",
    "clarity",
    "significance",
    "originality",
    "soundness",
    "presentation",
    "contribution",
    "other",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def canonicalize_field_name(raw: str | None) -> str | None:
    """Normalize a raw OpenReview field name to a canonical key.

    Steps:
      1. Strip leading/trailing whitespace
      2. Lowercase
      3. Collapse spaces and hyphens to underscores
      4. Look up in CANONICAL_ALIASES
      5. If not found, return the cleaned key (unknown field)

    Returns None for empty/None input.
    """
    if not raw:
        return None
    s = _NORMALIZE_RE.sub("_", raw.strip().lower())
    return CANONICAL_ALIASES.get(s, s)


def should_extract_field(canonical: str, text: str) -> tuple[bool, str]:
    """Return ``(include, reason)`` for a field extraction policy decision.

    Policy:
      - Tier C  → always skip
      - Tier A  → always include if text is non-empty
      - Tier B  → include only when text length >= _SUBSTANTIVE_MIN_CHARS
      - Unknown → include when text length >= _SUBSTANTIVE_MIN_CHARS (heuristic)

    The ``reason`` string is suitable for structured log lines, e.g.::

        log.info("field_policy field=%s include=%s reason=%s", canonical, include, reason)
    """
    stripped = (text or "").strip()

    if canonical in TIER_C:
        return False, "tier_c_metadata_only"

    if canonical in TIER_A:
        if not stripped:
            return False, "tier_a_empty"
        return True, "tier_a"

    if canonical in TIER_B:
        if len(stripped) >= _SUBSTANTIVE_MIN_CHARS:
            return True, "tier_b_substantive_text"
        return False, "tier_b_insufficient_text"

    # Unknown field — apply same substantive heuristic
    if len(stripped) >= _SUBSTANTIVE_MIN_CHARS:
        return True, "unknown_substantive_heuristic"
    return False, "unknown_insufficient_text"
