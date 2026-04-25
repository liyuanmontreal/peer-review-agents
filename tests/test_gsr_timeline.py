"""Tests for gsr_agent.rules.timeline — paper phases and micro-phases."""

from datetime import datetime, timedelta, timezone

import pytest

from gsr_agent.rules.timeline import (
    MicroPhase,
    PaperPhase,
    PaperWindows,
    _ensure_utc,
    compute_paper_windows,
    get_micro_phase,
    get_paper_phase,
)

UTC = timezone.utc
_OPEN = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)


def _at(hours: float) -> datetime:
    return _OPEN + timedelta(hours=hours)


# ---------------------------------------------------------------------------
# compute_paper_windows
# ---------------------------------------------------------------------------

def test_review_end_time_is_48h_after_open():
    w = compute_paper_windows(_OPEN)
    assert w.review_end_time == _OPEN + timedelta(hours=48)


def test_verdict_end_time_is_72h_after_open():
    w = compute_paper_windows(_OPEN)
    assert w.verdict_end_time == _OPEN + timedelta(hours=72)


def test_compute_paper_windows_naive_datetime_treated_as_utc():
    naive = datetime(2026, 4, 24, 12, 0, 0)  # no tzinfo
    w = compute_paper_windows(naive)
    assert w.open_time.tzinfo is not None


# ---------------------------------------------------------------------------
# get_paper_phase — coarse phases
# ---------------------------------------------------------------------------

def test_phase_new_before_open():
    assert get_paper_phase(_OPEN - timedelta(hours=1), _OPEN) == PaperPhase.NEW


def test_phase_review_active_at_open():
    assert get_paper_phase(_OPEN, _OPEN) == PaperPhase.REVIEW_ACTIVE


def test_phase_review_active_midway():
    assert get_paper_phase(_at(24), _OPEN) == PaperPhase.REVIEW_ACTIVE


def test_phase_review_active_at_boundary():
    # At exactly 48h the review window is still inclusive.
    assert get_paper_phase(_at(48), _OPEN) == PaperPhase.REVIEW_ACTIVE


def test_phase_verdict_active_just_after_review():
    assert get_paper_phase(_at(48) + timedelta(seconds=1), _OPEN) == PaperPhase.VERDICT_ACTIVE


def test_phase_verdict_active_midway():
    assert get_paper_phase(_at(60), _OPEN) == PaperPhase.VERDICT_ACTIVE


def test_phase_verdict_active_at_boundary():
    assert get_paper_phase(_at(72), _OPEN) == PaperPhase.VERDICT_ACTIVE


def test_phase_expired_after_72h():
    assert get_paper_phase(_at(72) + timedelta(seconds=1), _OPEN) == PaperPhase.EXPIRED


def test_phase_expired_well_past():
    assert get_paper_phase(_at(100), _OPEN) == PaperPhase.EXPIRED


# ---------------------------------------------------------------------------
# get_micro_phase — finer-grained phases
# ---------------------------------------------------------------------------

def test_micro_seed_window_at_open():
    assert get_micro_phase(_at(0), _OPEN) == MicroPhase.SEED_WINDOW


def test_micro_seed_window_6h():
    assert get_micro_phase(_at(6), _OPEN) == MicroPhase.SEED_WINDOW


def test_micro_seed_window_just_before_12h():
    assert get_micro_phase(_at(11.9), _OPEN) == MicroPhase.SEED_WINDOW


def test_micro_build_window_at_12h():
    assert get_micro_phase(_at(12), _OPEN) == MicroPhase.BUILD_WINDOW


def test_micro_build_window_24h():
    assert get_micro_phase(_at(24), _OPEN) == MicroPhase.BUILD_WINDOW


def test_micro_build_window_just_before_36h():
    assert get_micro_phase(_at(35.9), _OPEN) == MicroPhase.BUILD_WINDOW


def test_micro_lock_in_window_at_36h():
    assert get_micro_phase(_at(36), _OPEN) == MicroPhase.LOCK_IN_WINDOW


def test_micro_lock_in_window_42h():
    assert get_micro_phase(_at(42), _OPEN) == MicroPhase.LOCK_IN_WINDOW


def test_micro_lock_in_window_just_before_48h():
    assert get_micro_phase(_at(47.9), _OPEN) == MicroPhase.LOCK_IN_WINDOW


def test_micro_eligibility_window_at_48h():
    assert get_micro_phase(_at(48), _OPEN) == MicroPhase.ELIGIBILITY_WINDOW


def test_micro_eligibility_window_54h():
    assert get_micro_phase(_at(54), _OPEN) == MicroPhase.ELIGIBILITY_WINDOW


def test_micro_eligibility_window_just_before_60h():
    assert get_micro_phase(_at(59.9), _OPEN) == MicroPhase.ELIGIBILITY_WINDOW


def test_micro_submission_window_at_60h():
    assert get_micro_phase(_at(60), _OPEN) == MicroPhase.SUBMISSION_WINDOW


def test_micro_submission_window_66h():
    assert get_micro_phase(_at(66), _OPEN) == MicroPhase.SUBMISSION_WINDOW


def test_micro_submission_window_just_before_72h():
    assert get_micro_phase(_at(71.9), _OPEN) == MicroPhase.SUBMISSION_WINDOW


def test_micro_expired_at_72h():
    assert get_micro_phase(_at(72), _OPEN) == MicroPhase.EXPIRED


def test_micro_expired_well_past():
    assert get_micro_phase(_at(100), _OPEN) == MicroPhase.EXPIRED


# ---------------------------------------------------------------------------
# Timezone handling
# ---------------------------------------------------------------------------

def test_naive_datetime_accepted_in_get_paper_phase():
    naive_open = datetime(2026, 4, 24, 12, 0, 0)
    naive_now = datetime(2026, 4, 24, 18, 0, 0)
    assert get_paper_phase(naive_now, naive_open) == PaperPhase.REVIEW_ACTIVE


def test_aware_datetime_in_different_timezone():
    import zoneinfo
    tz_et = zoneinfo.ZoneInfo("America/Montreal")
    open_et = datetime(2026, 4, 24, 8, 0, 0, tzinfo=tz_et)  # 12:00 UTC
    now_et = datetime(2026, 4, 24, 20, 0, 0, tzinfo=tz_et)  # 24:00 UTC = +12h
    assert get_paper_phase(now_et, open_et) == PaperPhase.REVIEW_ACTIVE
