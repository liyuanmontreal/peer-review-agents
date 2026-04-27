"""Goldilocks heat-band model for paper crowding assessment.

Organizer guidance (ICML 2026 Koala competition):
  - Do NOT over-invest in papers already heavily reviewed by many agents
    (marginal value is low; duplicate-risk is high).
  - Do NOT rely on completely untouched papers as a core strategy
    (they may fail to accumulate enough distinct other-agent citations for verdict).
  - BEST TARGET: papers with 1–3 distinct citable other agents, especially 2–3,
    where the paper is active enough to be verdict-reachable but not yet crowded.

With 3 distinct other agents required for a verdict (down from 5) and ~21 active
participants, the goldilocks zone is both achievable and the highest expected-value
region of the participation distribution.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Point scores for distinct_citable_other_agents = 0..6; 7+ is _SCORE_SATURATED.
# Peak at 2 (1.00) to match the goldilocks sweet-spot; non-monotonic by design.
# ---------------------------------------------------------------------------
_CROWDING_SCORES: dict[int, float] = {
    0: 0.15,
    1: 0.75,
    2: 1.00,
    3: 0.95,
    4: 0.75,
    5: 0.55,
    6: 0.35,
}
_SCORE_SATURATED: float = 0.10


def paper_heat_band(distinct_citable_other_agents: int) -> str:
    """Classify a paper's participation level into a named heat band.

    Args:
        distinct_citable_other_agents: count of agents with citable comments
            on the paper, excluding our own.

    Returns:
        One of: "cold" | "warm" | "goldilocks" | "crowded" | "saturated"

    Band definitions:
        cold        0 agents   — no social proof; verdict may stall
        warm        1 agent    — some activity; watch and consider
        goldilocks  2–3 agents — near verdict threshold; best expected value
        crowded     4–6 agents — good coverage already; marginal value decreasing
        saturated   7+ agents  — heavy competition; low marginal value
    """
    n = distinct_citable_other_agents
    if n == 0:
        return "cold"
    if n == 1:
        return "warm"
    if n <= 3:
        return "goldilocks"
    if n <= 6:
        return "crowded"
    return "saturated"


def crowding_score(distinct_citable_other_agents: int) -> float:
    """Return a 0–1 priority score for seeding based on crowding level.

    Non-monotonic: peaks at 2 (goldilocks sweet-spot) and falls on both sides.
    Use this to rank seed candidates — higher score = more attractive target.

    Args:
        distinct_citable_other_agents: count of other agents with citable comments.

    Returns:
        Float in (0, 1]. See module-level mapping for the reference values.
    """
    if distinct_citable_other_agents >= 7:
        return _SCORE_SATURATED
    return _CROWDING_SCORES.get(distinct_citable_other_agents, _SCORE_SATURATED)
