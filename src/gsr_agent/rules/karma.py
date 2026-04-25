"""Karma budget engine — hard budgeting, not heuristics.

Every agent starts with 100 karma. Actions cost:
  First comment or thread on a paper   1.0
  Subsequent comment/thread same paper 0.1
  Verdict submission                   0.0

The reserve floor (default 15) prevents opening new papers when karma is low.
"""

from __future__ import annotations

from typing import Literal

INITIAL_KARMA: float = 100.0
FIRST_ACTION_COST: float = 1.0
FOLLOWUP_ACTION_COST: float = 0.1
VERDICT_COST: float = 0.0
DEFAULT_RESERVE_FLOOR: float = 15.0

ActionType = Literal["comment", "thread", "verdict"]


def get_action_cost(action_type: ActionType, has_prior_participation: bool) -> float:
    """Return the karma cost for an action on a paper.

    For "comment" and "thread": 1.0 on first participation, 0.1 on follow-up.
    For "verdict": always 0.0.
    """
    if action_type == "verdict":
        return VERDICT_COST
    return FOLLOWUP_ACTION_COST if has_prior_participation else FIRST_ACTION_COST


def can_afford(karma_remaining: float, cost: float) -> bool:
    """Return True if karma_remaining is sufficient to cover the cost."""
    return karma_remaining >= cost


def should_block_new_paper_entry(
    karma_remaining: float,
    reserve_floor: float = DEFAULT_RESERVE_FLOOR,
) -> bool:
    """Return True if entering a new paper would violate the reserve floor.

    When True, the agent must not spend the first 1.0 karma on a new paper
    until karma is replenished or opportunities are re-evaluated.
    """
    return karma_remaining < reserve_floor
