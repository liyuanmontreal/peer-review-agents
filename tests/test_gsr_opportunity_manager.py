"""Tests for gsr_agent.strategy.opportunity_manager — rule-based paper classification."""

from datetime import datetime, timedelta, timezone

import pytest

from gsr_agent.koala.models import Paper
from gsr_agent.rules.karma import DEFAULT_RESERVE_FLOOR, FIRST_ACTION_COST, INITIAL_KARMA
from gsr_agent.strategy.opportunity_manager import (
    PaperOpportunity,
    classify_paper_opportunity,
    get_actionable_papers,
    should_seed,
)

UTC = timezone.utc
_OPEN = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)


def _at(hours: float) -> datetime:
    return _OPEN + timedelta(hours=hours)


def _make_paper(paper_id: str = "paper-001", open_time: datetime = _OPEN) -> Paper:
    from gsr_agent.rules.timeline import compute_paper_windows
    w = compute_paper_windows(open_time)
    return Paper(
        paper_id=paper_id,
        title="Test Paper",
        open_time=w.open_time,
        review_end_time=w.review_end_time,
        verdict_end_time=w.verdict_end_time,
        state="REVIEW_ACTIVE",
    )


_NOW_SEED = _at(6)       # inside SEED_WINDOW (0–12h)
_NOW_BUILD = _at(20)     # inside BUILD_WINDOW (12–36h)
_NOW_LOCKIN = _at(40)    # inside LOCK_IN_WINDOW (36–48h)
_NOW_VERDICT = _at(62)   # inside SUBMISSION_WINDOW (60–72h)
_NOW_EXPIRED = _at(75)   # past VERDICT_ACTIVE


# ---------------------------------------------------------------------------
# classify_paper_opportunity — SEED
# ---------------------------------------------------------------------------

def test_classify_seed_window_no_prior():
    opp = classify_paper_opportunity(_make_paper(), False, 50.0, _NOW_SEED)
    assert opp == PaperOpportunity.SEED


def test_classify_seed_window_exact_karma():
    # Minimum karma that passes both the reserve floor gate (karma >= floor)
    # and the can_afford check (karma >= FIRST_ACTION_COST).
    min_karma = DEFAULT_RESERVE_FLOOR  # 15.0 — not blocked (strict <), can afford 1.0
    opp = classify_paper_opportunity(_make_paper(), False, min_karma, _NOW_SEED)
    assert opp == PaperOpportunity.SEED


def test_classify_seed_window_already_participated_is_skip():
    opp = classify_paper_opportunity(_make_paper(), True, 50.0, _NOW_SEED)
    assert opp == PaperOpportunity.SKIP


def test_classify_seed_window_insufficient_karma_is_skip():
    opp = classify_paper_opportunity(_make_paper(), False, 0.5, _NOW_SEED)
    assert opp == PaperOpportunity.SKIP


def test_classify_seed_window_zero_karma_is_skip():
    opp = classify_paper_opportunity(_make_paper(), False, 0.0, _NOW_SEED)
    assert opp == PaperOpportunity.SKIP


def test_classify_seed_blocked_by_reserve_floor():
    # karma_remaining just below reserve floor — should_block_new_paper_entry triggers
    below_floor = DEFAULT_RESERVE_FLOOR - 0.1
    opp = classify_paper_opportunity(_make_paper(), False, below_floor, _NOW_SEED)
    assert opp == PaperOpportunity.SKIP


# ---------------------------------------------------------------------------
# classify_paper_opportunity — FOLLOWUP
# ---------------------------------------------------------------------------

def test_classify_build_window_participated():
    opp = classify_paper_opportunity(_make_paper(), True, 50.0, _NOW_BUILD)
    assert opp == PaperOpportunity.FOLLOWUP


def test_classify_lockin_window_participated():
    opp = classify_paper_opportunity(_make_paper(), True, 50.0, _NOW_LOCKIN)
    assert opp == PaperOpportunity.FOLLOWUP


def test_classify_build_window_no_prior_is_skip():
    opp = classify_paper_opportunity(_make_paper(), False, 50.0, _NOW_BUILD)
    assert opp == PaperOpportunity.SKIP


def test_classify_build_window_insufficient_karma_is_skip():
    opp = classify_paper_opportunity(_make_paper(), True, 0.0, _NOW_BUILD)
    assert opp == PaperOpportunity.SKIP


# ---------------------------------------------------------------------------
# classify_paper_opportunity — VERDICT_READY
# ---------------------------------------------------------------------------

def test_classify_verdict_window_participated():
    opp = classify_paper_opportunity(_make_paper(), True, 50.0, _NOW_VERDICT)
    assert opp == PaperOpportunity.VERDICT_READY


def test_classify_verdict_window_no_prior_is_skip():
    opp = classify_paper_opportunity(_make_paper(), False, 50.0, _NOW_VERDICT)
    assert opp == PaperOpportunity.SKIP


# ---------------------------------------------------------------------------
# classify_paper_opportunity — SKIP (expired / wrong phase)
# ---------------------------------------------------------------------------

def test_classify_expired_is_skip():
    opp = classify_paper_opportunity(_make_paper(), True, 50.0, _NOW_EXPIRED)
    assert opp == PaperOpportunity.SKIP


def test_classify_pre_open_seed_window_no_prior_is_seed():
    opp = classify_paper_opportunity(_make_paper(), False, 50.0, _OPEN - timedelta(hours=1))
    assert opp == PaperOpportunity.SEED


def test_classify_pre_open_seed_window_participated_is_skip():
    opp = classify_paper_opportunity(_make_paper(), True, 50.0, _OPEN - timedelta(hours=1))
    assert opp == PaperOpportunity.SKIP


def test_classify_pre_open_seed_window_no_karma_is_skip():
    opp = classify_paper_opportunity(_make_paper(), False, 0.0, _OPEN - timedelta(hours=1))
    assert opp == PaperOpportunity.SKIP


# ---------------------------------------------------------------------------
# should_seed
# ---------------------------------------------------------------------------

def test_should_seed_true_in_seed_window():
    assert should_seed(_make_paper(), False, 50.0, _NOW_SEED) is True


def test_should_seed_false_with_prior_participation():
    assert should_seed(_make_paper(), True, 50.0, _NOW_SEED) is False


def test_should_seed_false_in_build_window():
    assert should_seed(_make_paper(), False, 50.0, _NOW_BUILD) is False


def test_should_seed_false_with_insufficient_karma():
    assert should_seed(_make_paper(), False, 0.5, _NOW_SEED) is False


# ---------------------------------------------------------------------------
# get_actionable_papers
# ---------------------------------------------------------------------------

def test_get_actionable_filters_expired():
    papers = [
        _make_paper("p1"),                          # SEED_WINDOW → SEED
        _make_paper("p2", _at(-75)),                # expired → SKIP
    ]
    def has_participated(pid: str) -> bool:
        return False

    result = get_actionable_papers(papers, has_participated, 50.0, _NOW_SEED)
    ids = [p.paper_id for p in result]
    assert "p1" in ids
    assert "p2" not in ids


def test_get_actionable_returns_all_actionable():
    papers = [
        _make_paper("p1"),                          # SEED_WINDOW, no prior → SEED
        _make_paper("p2", _at(-15)),                # BUILD_WINDOW now (_NOW_SEED - 15h opens)
    ]
    def has_participated(pid: str) -> bool:
        return pid == "p2"

    # p2 opened 15h before NOW_SEED, so NOW_SEED is 6h into p2 = SEED_WINDOW
    # but p2 has_participated=True → SKIP in SEED_WINDOW
    result = get_actionable_papers(papers, has_participated, 50.0, _NOW_SEED)
    ids = [p.paper_id for p in result]
    assert "p1" in ids


def test_get_actionable_empty_when_no_karma():
    papers = [_make_paper("p1")]
    result = get_actionable_papers(papers, lambda pid: False, 0.0, _NOW_SEED)
    assert result == []


def test_get_actionable_empty_list():
    result = get_actionable_papers([], lambda pid: False, 50.0, _NOW_SEED)
    assert result == []
