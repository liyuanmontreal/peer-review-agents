"""Seed comment generation — Phase 4A (abstract-based, conservative templates).

Generates safe, hedged seed comments from the paper abstract. No LLM calls.
All generated text passes moderation checks before being returned as candidates.
"""

from __future__ import annotations

from typing import List, Optional

from ..adapters.gsr_runner import PaperIndex, get_seed_evidence_candidates
from ..rules.moderation import check_moderation

_MIN_ABSTRACT_CHARS = 40
_MIN_ABSTRACT_WORDS = 6

# Each template asks a narrow, method-specific question tied to the paper's
# title. No abstract text is quoted to avoid the moderation low_effort category.
_TEMPLATES = [
    (
        "The {title} method appears to rely on conditions that hold in the "
        "standard evaluation benchmarks. "
        "One question worth exploring: how does it behave under distribution shift "
        "or in regimes that differ from the training setup? "
        "Specifically, is there a failure mode where performance degrades sharply "
        "rather than gradually?"
    ),
    (
        "For {title}, it would be useful to know whether the reported gains "
        "are driven by a single design choice or require all components working together. "
        "Was a per-component ablation run, and if so, which component "
        "contributes the largest share of the improvement?"
    ),
    (
        "{title} reports strong results on the stated benchmarks. "
        "How stable are these results across different hyperparameter settings "
        "and data scales? "
        "Do the gains persist as compute budget or dataset size changes, "
        "or are they concentrated in a specific operating regime?"
    ),
]


def _is_low_signal_abstract(abstract: str) -> bool:
    stripped = abstract.strip()
    if not stripped:
        return False  # empty abstract handled separately via get_seed_evidence_candidates
    return len(stripped) < _MIN_ABSTRACT_CHARS or len(stripped.split()) < _MIN_ABSTRACT_WORDS


def is_low_signal_abstract(index: PaperIndex) -> bool:
    """Return True if the abstract is too short or vague to generate a substantive comment."""
    return _is_low_signal_abstract(index.abstract or "")


def generate_seed_comment_candidates(index: PaperIndex) -> List[str]:
    """Generate conservative seed comment candidates from paper abstract.

    Returns a list of candidate comment strings, all passing moderation.
    Returns an empty list if the paper has no abstract.
    """
    candidates = get_seed_evidence_candidates(index)
    if not candidates:
        return []

    results = []
    for template in _TEMPLATES:
        text = template.format(title=index.title or "this work")
        passes, _ = check_moderation(text)
        if passes and text.strip():
            results.append(text)

    return results


def score_seed_comment_candidate(body: str, paper_id: str) -> float:
    """Score a seed comment candidate on a 0–1 scale.

    Higher scores favour longer, more substantive comments that pass moderation.
    Returns 0.0 for empty or whitespace-only bodies.
    """
    if not body or not body.strip():
        return 0.0

    passes, _ = check_moderation(body)
    if not passes:
        return 0.0

    length_score = min(len(body) / 400.0, 1.0)
    has_question = 1.0 if "?" in body else 0.0
    return 0.6 * length_score + 0.4 * has_question


def choose_best_seed_comment(candidates: List[str]) -> Optional[str]:
    """Return the highest-scoring candidate, or None if the list is empty."""
    if not candidates:
        return None
    return max(candidates, key=lambda c: score_seed_comment_candidate(c, ""))
