"""Goldilocks heat-band model for paper crowding assessment.

Endgame configuration (ICML 2026 Koala competition final day):
  - BEST TARGETS: papers with 3–10 distinct other-agent comments — active enough
    for verdict-reachability with strong marginal value.
  - EXTENDED TARGETS: 11–14 comments — still eligible, marginal value decreasing
    but not zero; include in endgame to maximise verdict funnel.
  - AVOID: 0-comment papers as primary targets (verdict may stall); 15+ saturated.
  - Cold papers (0) remain allowed only as fallback.

With 3 distinct other agents required for a verdict and an endgame priority on
total verdict count, the goldilocks zone is widened to 3–10 and the saturation
ceiling is raised to 15+ so that threads up to 14 comments remain in scope.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Point scores for distinct_citable_other_agents = 0..14; 15+ is _SCORE_SATURATED.
# Peak at 5–6 (goldilocks centre); warm (1–2) below goldilocks; crowded (11–14)
# above cold but below warm, to preserve the 3-10 > 11-14 > 1-2 > 0 ordering.
# ---------------------------------------------------------------------------
_CROWDING_SCORES: dict[int, float] = {
    0: 0.15,
    1: 0.45,
    2: 0.55,
    3: 0.80,
    4: 0.90,
    5: 1.00,
    6: 1.00,
    7: 0.95,
    8: 0.90,
    9: 0.85,
    10: 0.80,
    11: 0.40,
    12: 0.32,
    13: 0.25,
    14: 0.20,
}
_SCORE_SATURATED: float = 0.10


def paper_heat_band(distinct_citable_other_agents: int) -> str:
    """Classify a paper's participation level into a named heat band.

    Args:
        distinct_citable_other_agents: count of agents with citable comments
            on the paper, excluding our own.

    Returns:
        One of: "cold" | "warm" | "goldilocks" | "crowded" | "saturated"

    Band definitions (endgame thresholds):
        cold        0 agents    — no social proof; verdict may stall
        warm        1–2 agents  — some activity; below preferred zone
        goldilocks  3–10 agents — optimal verdict-funnel zone; highest priority
        crowded     11–14 agents — still eligible; marginal value decreasing
        saturated   15+ agents  — skip; thread is over-covered
    """
    n = distinct_citable_other_agents
    if n == 0:
        return "cold"
    if n <= 2:
        return "warm"
    if n <= 10:
        return "goldilocks"
    if n <= 14:
        return "crowded"
    return "saturated"


def crowding_score(distinct_citable_other_agents: int) -> float:
    """Return a 0–1 priority score for seeding based on crowding level.

    Peaks at 5–6 (goldilocks centre). Use to rank seed candidates — higher
    score = more attractive target.

    Args:
        distinct_citable_other_agents: count of other agents with citable comments.

    Returns:
        Float in (0, 1]. See module-level mapping for the reference values.
    """
    if distinct_citable_other_agents >= 15:
        return _SCORE_SATURATED
    return _CROWDING_SCORES.get(distinct_citable_other_agents, _SCORE_SATURATED)
