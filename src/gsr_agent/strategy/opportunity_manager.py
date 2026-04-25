"""Rule-based paper opportunity classifier (Phase 4A — no LLM).

Classifies each paper into one of four opportunities based on timing,
karma, and participation state. All rules are derived from the competition
timeline and karma ledger — no external calls or LLM reasoning.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Callable, List

from ..koala.models import Paper
from ..rules.karma import (
    DEFAULT_RESERVE_FLOOR,
    FOLLOWUP_ACTION_COST,
    FIRST_ACTION_COST,
    can_afford,
    get_action_cost,
    should_block_new_paper_entry,
)
from ..rules.timeline import MicroPhase, PaperPhase, get_micro_phase, get_paper_phase


class PaperOpportunity(Enum):
    SEED = "SEED"                  # Post initial seed comment (0–12h, no prior participation)
    FOLLOWUP = "FOLLOWUP"          # Post follow-up comment (12–48h, participated before)
    VERDICT_READY = "VERDICT_READY"  # Submit verdict (60–72h, participated before)
    SKIP = "SKIP"                  # No actionable opportunity right now


def classify_paper_opportunity(
    paper: Paper,
    has_participated: bool,
    karma_remaining: float,
    now: datetime,
) -> PaperOpportunity:
    """Classify what action, if any, should be taken on this paper right now.

    Args:
        paper:            the paper to evaluate
        has_participated: True if we have posted at least one comment on this paper
        karma_remaining:  current karma budget
        now:              current UTC datetime

    Returns:
        PaperOpportunity enum value
    """
    phase = get_paper_phase(now, paper.open_time)
    if phase in (PaperPhase.NEW, PaperPhase.EXPIRED):
        return PaperOpportunity.SKIP

    micro = get_micro_phase(now, paper.open_time)

    if micro == MicroPhase.SEED_WINDOW:
        if has_participated:
            return PaperOpportunity.SKIP
        if should_block_new_paper_entry(karma_remaining, DEFAULT_RESERVE_FLOOR):
            return PaperOpportunity.SKIP
        if not can_afford(karma_remaining, get_action_cost("comment", has_prior_participation=False)):
            return PaperOpportunity.SKIP
        return PaperOpportunity.SEED

    if micro in (MicroPhase.BUILD_WINDOW, MicroPhase.LOCK_IN_WINDOW):
        if not has_participated:
            return PaperOpportunity.SKIP
        if not can_afford(karma_remaining, get_action_cost("comment", has_prior_participation=True)):
            return PaperOpportunity.SKIP
        return PaperOpportunity.FOLLOWUP

    if micro in (MicroPhase.ELIGIBILITY_WINDOW, MicroPhase.SUBMISSION_WINDOW):
        if not has_participated:
            return PaperOpportunity.SKIP
        return PaperOpportunity.VERDICT_READY

    return PaperOpportunity.SKIP


def should_seed(
    paper: Paper,
    has_participated: bool,
    karma_remaining: float,
    now: datetime,
) -> bool:
    """True when the paper is a seed comment candidate right now."""
    return (
        classify_paper_opportunity(paper, has_participated, karma_remaining, now)
        == PaperOpportunity.SEED
    )


def get_actionable_papers(
    papers: List[Paper],
    has_participated_fn: Callable[[str], bool],
    karma_remaining: float,
    now: datetime,
) -> List[Paper]:
    """Filter papers to those with a non-SKIP opportunity.

    Args:
        papers:              all papers to evaluate
        has_participated_fn: callable mapping paper_id → bool
        karma_remaining:     current karma budget
        now:                 current UTC datetime

    Returns:
        Subset of papers that have an actionable opportunity.
    """
    result = []
    for paper in papers:
        participated = has_participated_fn(paper.paper_id)
        opp = classify_paper_opportunity(paper, participated, karma_remaining, now)
        if opp != PaperOpportunity.SKIP:
            result.append(paper)
    return result
