"""Rule-based paper opportunity classifier (Phase 4A — no LLM).

Classifies each paper into one of four opportunities based on timing,
karma, and participation state. All rules are derived from the competition
timeline and karma ledger — no external calls or LLM reasoning.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

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
from ..rules.verdict_eligibility import MIN_DISTINCT_OTHER_AGENTS
from .heat import crowding_score, paper_heat_band


class PaperOpportunity(Enum):
    SEED = "SEED"                  # Post initial seed comment (0–12h, no prior participation)
    FOLLOWUP = "FOLLOWUP"          # Post follow-up comment (12–48h, participated before)
    VERDICT_READY = "VERDICT_READY"  # Submit verdict (60–72h, participated before)
    SKIP = "SKIP"                  # No actionable opportunity right now


# Processing priority: lower number = higher priority.
# Verdicts (VERDICT_READY) are processed before new seed comments.
OPPORTUNITY_PRIORITY: Dict[PaperOpportunity, int] = {
    PaperOpportunity.VERDICT_READY: 0,
    PaperOpportunity.FOLLOWUP: 1,
    PaperOpportunity.SEED: 2,
    PaperOpportunity.SKIP: 3,
}

# Minimum citeable other-agent comments required for a valid verdict opportunity.
MIN_VERDICT_CITATIONS: int = MIN_DISTINCT_OTHER_AGENTS  # = 3

# SEED comment crowding thresholds (based on citable_other comment count).
PREFERRED_COMMENT_MIN: int = 1   # inclusive lower bound for preferred seeding zone
PREFERRED_COMMENT_MAX: int = 8   # inclusive upper bound for preferred seeding zone
SATURATED_COMMENT_THRESHOLD: int = 12  # > this → SEED candidate is saturated/skipped

# Maximum actionable papers to process per operational-loop run.
CANDIDATE_BUDGET: int = 3


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
    if phase == PaperPhase.EXPIRED:
        return PaperOpportunity.SKIP

    micro = get_micro_phase(now, paper.open_time)

    if phase == PaperPhase.NEW:
        if micro != MicroPhase.SEED_WINDOW:
            return PaperOpportunity.SKIP
        if has_participated:
            return PaperOpportunity.SKIP
        if should_block_new_paper_entry(karma_remaining, DEFAULT_RESERVE_FLOOR):
            return PaperOpportunity.SKIP
        if not can_afford(karma_remaining, get_action_cost("comment", has_prior_participation=False)):
            return PaperOpportunity.SKIP
        return PaperOpportunity.SEED

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


# ---------------------------------------------------------------------------
# Goldilocks crowding helpers (Phase 5A.5 / organizer guidance)
# ---------------------------------------------------------------------------

# Soft policy: cold-paper (0 other agents) seeds should stay at or below this
# fraction of total seed actions.  Not enforced mechanically — use as a budget
# guideline when selecting among multiple seed candidates.
COLD_PAPER_SEED_TARGET_PCT: float = 0.10


def get_seed_crowding_note(
    distinct_citable_other_agents: int,
) -> Tuple[str, Optional[str]]:
    """Return a (tier, reason) soft signal for a seed candidate.

    Tier values: "prefer" | "neutral" | "deprioritize"
    Reason is a loggable/storable string when tier is "deprioritize", else None.

    This is advisory only — the caller decides whether to proceed.
    A "deprioritize" tier is a soft penalty, NOT a hard ban.

    Args:
        distinct_citable_other_agents: other-agent citable comment count for
            the paper at the time of the seeding decision.

    Examples:
        >>> get_seed_crowding_note(0)
        ('deprioritize', 'too_cold_no_social_proof')
        >>> get_seed_crowding_note(2)
        ('prefer', None)
        >>> get_seed_crowding_note(8)
        ('deprioritize', 'too_crowded_low_marginal_value')
    """
    band = paper_heat_band(distinct_citable_other_agents)
    if band == "cold":
        return "deprioritize", "too_cold_no_social_proof"
    if band == "warm":
        return "neutral", None
    if band == "goldilocks":
        return "prefer", None
    # crowded or saturated
    return "deprioritize", "too_crowded_low_marginal_value"


def sort_seed_papers_by_crowding(
    papers: List[Paper],
    distinct_counts: Dict[str, int],
) -> List[Paper]:
    """Sort seed candidate papers by crowding_score descending (best first).

    Papers whose paper_id is absent from distinct_counts are treated as cold
    (score 0.15).  Preserves relative order for equal scores (stable sort).

    Args:
        papers:          candidate papers to sort
        distinct_counts: mapping of paper_id → distinct_citable_other_agents

    Returns:
        New list sorted highest crowding_score first.
    """
    return sorted(
        papers,
        key=lambda p: crowding_score(distinct_counts.get(p.paper_id, 0)),
        reverse=True,
    )
