"""Seed comment generation — Phase 4A (abstract-based, conservative templates).

Generates safe, hedged seed comments from the paper abstract. No LLM calls.
All generated text passes moderation checks before being returned as candidates.
"""

from __future__ import annotations

from typing import List, Optional

from ..adapters.gsr_runner import PaperIndex, get_seed_evidence_candidates
from ..rules.moderation import check_moderation

_TEMPLATES = [
    (
        "The abstract states: \"{claim_excerpt}\"\n\n"
        "Could the authors clarify what empirical evidence most directly supports "
        "this claim? I'd like to understand the evaluation methodology before "
        "drawing conclusions."
    ),
    (
        "Based on the abstract, the paper addresses {title}. "
        "The abstract mentions: \"{claim_excerpt}\"\n\n"
        "What are the main limitations or scope boundaries that future work "
        "should be aware of?"
    ),
    (
        "Reading the abstract, I note the focus on {title}. "
        "I look forward to reviewing the methodology section to assess "
        "how the authors handle: \"{claim_excerpt}\""
    ),
]

_MAX_CLAIM_EXCERPT = 150


def generate_seed_comment_candidates(index: PaperIndex) -> List[str]:
    """Generate conservative seed comment candidates from paper abstract.

    Returns a list of candidate comment strings, all passing moderation.
    Returns an empty list if the paper has no abstract.
    """
    candidates = get_seed_evidence_candidates(index)
    if not candidates:
        return []

    claim = candidates[0].claim
    excerpt = claim[:_MAX_CLAIM_EXCERPT].rstrip()
    if len(claim) > _MAX_CLAIM_EXCERPT:
        excerpt += "..."

    results = []
    for template in _TEMPLATES:
        text = template.format(
            claim_excerpt=excerpt,
            title=index.title or "this topic",
        )
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
