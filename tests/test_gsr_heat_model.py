"""Tests for Phase 5A.5 Goldilocks heat-band model and crowding helpers."""

from datetime import datetime, timezone, timedelta

import pytest

from gsr_agent.strategy.heat import paper_heat_band, crowding_score
from gsr_agent.strategy.opportunity_manager import (
    COLD_PAPER_SEED_TARGET_PCT,
    get_seed_crowding_note,
    sort_seed_papers_by_crowding,
)
from gsr_agent.koala.models import Paper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper(paper_id: str) -> Paper:
    now = datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        open_time=now,
        review_end_time=now + timedelta(hours=48),
        verdict_end_time=now + timedelta(hours=72),
        state="REVIEW_ACTIVE",
    )


# ---------------------------------------------------------------------------
# A. Heat band classification
# ---------------------------------------------------------------------------

def test_heat_band_0_is_cold():
    assert paper_heat_band(0) == "cold"


def test_heat_band_1_is_warm():
    assert paper_heat_band(1) == "warm"


def test_heat_band_2_is_goldilocks():
    assert paper_heat_band(2) == "goldilocks"


def test_heat_band_3_is_goldilocks():
    assert paper_heat_band(3) == "goldilocks"


def test_heat_band_4_is_crowded():
    assert paper_heat_band(4) == "crowded"


def test_heat_band_6_is_crowded():
    assert paper_heat_band(6) == "crowded"


def test_heat_band_7_is_saturated():
    assert paper_heat_band(7) == "saturated"


def test_heat_band_large_is_saturated():
    assert paper_heat_band(50) == "saturated"


def test_heat_band_5_is_crowded():
    assert paper_heat_band(5) == "crowded"


# ---------------------------------------------------------------------------
# B. Crowding score shape (non-monotonic, peaks at 2)
# ---------------------------------------------------------------------------

def test_crowding_score_2_beats_1():
    assert crowding_score(2) > crowding_score(1)


def test_crowding_score_2_beats_4():
    assert crowding_score(2) > crowding_score(4)


def test_crowding_score_3_beats_5():
    assert crowding_score(3) > crowding_score(5)


def test_crowding_score_0_below_1():
    assert crowding_score(0) < crowding_score(1)


def test_crowding_score_7_below_4():
    assert crowding_score(7) < crowding_score(4)


def test_crowding_score_2_is_maximum():
    """Score at 2 is the global peak."""
    assert crowding_score(2) == 1.00


def test_crowding_score_0_is_low():
    assert crowding_score(0) == 0.15


def test_crowding_score_7_is_lowest():
    assert crowding_score(7) == 0.10


def test_crowding_score_large_equals_saturated():
    assert crowding_score(100) == crowding_score(7)


def test_crowding_score_returns_float():
    for n in range(10):
        assert isinstance(crowding_score(n), float)


# ---------------------------------------------------------------------------
# C. Opportunity logic behavior
# ---------------------------------------------------------------------------

def test_2_agent_paper_scores_higher_than_0_agent():
    assert crowding_score(2) > crowding_score(0)


def test_2_agent_paper_scores_higher_than_7_agent():
    assert crowding_score(2) > crowding_score(7)


def test_2_agent_paper_is_not_deprioritized():
    tier, _ = get_seed_crowding_note(2)
    assert tier == "prefer"


def test_2_agent_paper_is_not_dead_in_heat_band():
    """2-agent paper must be in goldilocks, never saturated or cold."""
    band = paper_heat_band(2)
    assert band not in ("cold", "saturated", "crowded")
    assert band == "goldilocks"


def test_0_agent_paper_is_soft_penalty_not_hard_ban():
    """Cold papers get deprioritize but no absolute ban — tier is 'deprioritize', not 'banned'."""
    tier, reason = get_seed_crowding_note(0)
    assert tier == "deprioritize"
    assert reason == "too_cold_no_social_proof"
    # Critically, tier is NOT "banned" or "skip" — caller may still proceed.
    assert tier != "banned"
    assert tier != "skip"


def test_sort_favors_2_agent_over_0_agent():
    p0 = _make_paper("p-cold")
    p2 = _make_paper("p-goldilocks")
    counts = {"p-cold": 0, "p-goldilocks": 2}
    sorted_papers = sort_seed_papers_by_crowding([p0, p2], counts)
    assert sorted_papers[0].paper_id == "p-goldilocks"
    assert sorted_papers[1].paper_id == "p-cold"


def test_sort_favors_2_agent_over_7_agent():
    p2 = _make_paper("p-goldilocks")
    p7 = _make_paper("p-saturated")
    counts = {"p-goldilocks": 2, "p-saturated": 7}
    sorted_papers = sort_seed_papers_by_crowding([p7, p2], counts)
    assert sorted_papers[0].paper_id == "p-goldilocks"


def test_sort_full_ordering():
    """Verify full sort: goldilocks > warm > cold > saturated."""
    p_cold = _make_paper("cold")
    p_warm = _make_paper("warm")
    p_gold = _make_paper("gold")
    p_sat = _make_paper("sat")
    counts = {"cold": 0, "warm": 1, "gold": 2, "sat": 8}
    result = sort_seed_papers_by_crowding([p_sat, p_cold, p_warm, p_gold], counts)
    ids = [p.paper_id for p in result]
    assert ids[0] == "gold"
    assert ids[-1] == "sat"


def test_sort_missing_count_treated_as_cold():
    """Papers not in distinct_counts dict default to score(0) = 0.15."""
    p_unknown = _make_paper("unknown")
    p_cold = _make_paper("cold")
    counts = {"cold": 0}
    result = sort_seed_papers_by_crowding([p_unknown, p_cold], counts)
    # Both have score 0.15 — order is stable (both tied, so original order preserved)
    assert set(p.paper_id for p in result) == {"unknown", "cold"}


def test_sort_empty_list():
    assert sort_seed_papers_by_crowding([], {}) == []


# ---------------------------------------------------------------------------
# D. Reason strings
# ---------------------------------------------------------------------------

def test_cold_paper_emits_too_cold_no_social_proof():
    tier, reason = get_seed_crowding_note(0)
    assert tier == "deprioritize"
    assert reason == "too_cold_no_social_proof"


def test_saturated_paper_emits_too_crowded_low_marginal_value():
    tier, reason = get_seed_crowding_note(7)
    assert tier == "deprioritize"
    assert reason == "too_crowded_low_marginal_value"


def test_crowded_paper_emits_too_crowded_low_marginal_value():
    tier, reason = get_seed_crowding_note(5)
    assert tier == "deprioritize"
    assert reason == "too_crowded_low_marginal_value"


def test_goldilocks_paper_no_reason():
    tier, reason = get_seed_crowding_note(3)
    assert tier == "prefer"
    assert reason is None


def test_warm_paper_no_reason():
    tier, reason = get_seed_crowding_note(1)
    assert tier == "neutral"
    assert reason is None


def test_all_counts_have_valid_tier():
    valid_tiers = {"prefer", "neutral", "deprioritize"}
    for n in range(12):
        tier, _ = get_seed_crowding_note(n)
        assert tier in valid_tiers, f"count={n} produced invalid tier {tier!r}"


def test_cold_paper_seed_target_pct_is_small():
    """Policy constant: cold-paper seeds should be a minority (<= 15%)."""
    assert COLD_PAPER_SEED_TARGET_PCT <= 0.15
    assert COLD_PAPER_SEED_TARGET_PCT > 0.0
