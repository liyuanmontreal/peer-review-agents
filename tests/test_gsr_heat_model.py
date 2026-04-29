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


def test_heat_band_2_is_warm():
    assert paper_heat_band(2) == "warm"


def test_heat_band_3_is_goldilocks():
    assert paper_heat_band(3) == "goldilocks"


def test_heat_band_4_is_goldilocks():
    assert paper_heat_band(4) == "goldilocks"


def test_heat_band_6_is_goldilocks():
    assert paper_heat_band(6) == "goldilocks"


def test_heat_band_7_is_goldilocks():
    assert paper_heat_band(7) == "goldilocks"


def test_heat_band_10_is_goldilocks():
    assert paper_heat_band(10) == "goldilocks"


def test_heat_band_11_is_crowded():
    assert paper_heat_band(11) == "crowded"


def test_heat_band_14_is_crowded():
    assert paper_heat_band(14) == "crowded"


def test_heat_band_15_is_saturated():
    assert paper_heat_band(15) == "saturated"


def test_heat_band_large_is_saturated():
    assert paper_heat_band(50) == "saturated"


def test_heat_band_5_is_goldilocks():
    assert paper_heat_band(5) == "goldilocks"


# ---------------------------------------------------------------------------
# B. Crowding score shape (non-monotonic, peaks at 2)
# ---------------------------------------------------------------------------

def test_crowding_score_5_beats_2():
    assert crowding_score(5) > crowding_score(2)


def test_crowding_score_4_beats_1():
    assert crowding_score(4) > crowding_score(1)


def test_crowding_score_3_beats_2():
    assert crowding_score(3) > crowding_score(2)


def test_crowding_score_0_below_1():
    assert crowding_score(0) < crowding_score(1)


def test_crowding_score_11_below_10():
    assert crowding_score(11) < crowding_score(10)


def test_crowding_score_5_is_maximum():
    """Score at 5–6 is the global peak (goldilocks centre)."""
    assert crowding_score(5) == 1.00
    assert crowding_score(6) == 1.00


def test_crowding_score_0_is_low():
    assert crowding_score(0) == 0.15


def test_crowding_score_15_is_saturated():
    assert crowding_score(15) == 0.10


def test_crowding_score_large_equals_saturated():
    assert crowding_score(100) == crowding_score(15)


def test_crowding_score_returns_float():
    for n in range(10):
        assert isinstance(crowding_score(n), float)


# ---------------------------------------------------------------------------
# C. Opportunity logic behavior
# ---------------------------------------------------------------------------

def test_2_agent_paper_scores_higher_than_0_agent():
    assert crowding_score(2) > crowding_score(0)


def test_7_agent_paper_scores_higher_than_2_agent():
    """In endgame model, 7 agents (goldilocks) beats 2 agents (warm)."""
    assert crowding_score(7) > crowding_score(2)


def test_5_agent_paper_is_preferred():
    tier, _ = get_seed_crowding_note(5)
    assert tier == "prefer"


def test_2_agent_paper_is_not_dead_in_heat_band():
    """2-agent paper must not be saturated or cold (it is warm)."""
    band = paper_heat_band(2)
    assert band not in ("cold", "saturated")
    assert band == "warm"


def test_0_agent_paper_is_soft_penalty_not_hard_ban():
    """Cold papers get deprioritize but no absolute ban — tier is 'deprioritize', not 'banned'."""
    tier, reason = get_seed_crowding_note(0)
    assert tier == "deprioritize"
    assert reason == "too_cold_no_social_proof"
    # Critically, tier is NOT "banned" or "skip" — caller may still proceed.
    assert tier != "banned"
    assert tier != "skip"


def test_sort_favors_5_agent_over_0_agent():
    p0 = _make_paper("p-cold")
    p5 = _make_paper("p-goldilocks")
    counts = {"p-cold": 0, "p-goldilocks": 5}
    sorted_papers = sort_seed_papers_by_crowding([p0, p5], counts)
    assert sorted_papers[0].paper_id == "p-goldilocks"
    assert sorted_papers[1].paper_id == "p-cold"


def test_sort_favors_5_agent_over_15_agent():
    p5 = _make_paper("p-goldilocks")
    p15 = _make_paper("p-saturated")
    counts = {"p-goldilocks": 5, "p-saturated": 15}
    sorted_papers = sort_seed_papers_by_crowding([p15, p5], counts)
    assert sorted_papers[0].paper_id == "p-goldilocks"


def test_sort_full_ordering():
    """Verify full sort: goldilocks (5) > warm (1) > cold (0) > saturated (15)."""
    p_cold = _make_paper("cold")
    p_warm = _make_paper("warm")
    p_gold = _make_paper("gold")
    p_sat = _make_paper("sat")
    counts = {"cold": 0, "warm": 1, "gold": 5, "sat": 15}
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
    tier, reason = get_seed_crowding_note(15)
    assert tier == "deprioritize"
    assert reason == "too_crowded_low_marginal_value"


def test_crowded_paper_emits_too_crowded_low_marginal_value():
    tier, reason = get_seed_crowding_note(12)
    assert tier == "deprioritize"
    assert reason == "too_crowded_low_marginal_value"


def test_goldilocks_paper_no_reason():
    tier, reason = get_seed_crowding_note(5)
    assert tier == "prefer"
    assert reason is None


def test_goldilocks_paper_10_no_reason():
    tier, reason = get_seed_crowding_note(10)
    assert tier == "prefer"
    assert reason is None


def test_warm_paper_no_reason():
    tier, reason = get_seed_crowding_note(1)
    assert tier == "neutral"
    assert reason is None


def test_all_counts_have_valid_tier():
    valid_tiers = {"prefer", "neutral", "deprioritize"}
    for n in range(20):
        tier, _ = get_seed_crowding_note(n)
        assert tier in valid_tiers, f"count={n} produced invalid tier {tier!r}"


def test_cold_paper_seed_target_pct_is_small():
    """Policy constant: cold-paper seeds should be a minority (<= 15%)."""
    assert COLD_PAPER_SEED_TARGET_PCT <= 0.15
    assert COLD_PAPER_SEED_TARGET_PCT > 0.0
