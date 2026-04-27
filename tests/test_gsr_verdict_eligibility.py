"""Tests for gsr_agent.rules.verdict_eligibility — eligibility state machine."""

from datetime import datetime, timedelta, timezone

import pytest

from gsr_agent.rules.verdict_eligibility import (
    EligibilityState,
    MIN_DISTINCT_OTHER_AGENTS,
    MIN_VERDICT_CONFIDENCE,
    VerdictEligibilityInput,
    can_submit_verdict,
    compute_eligibility_state,
)

UTC = timezone.utc
_OPEN = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)


def _at(hours: float) -> datetime:
    return _OPEN + timedelta(hours=hours)


def _make(
    has_our_participation: bool = True,
    distinct_citable_other_agents: int = 3,
    audit_artifact_ready: bool = True,
    internal_score_confidence: float = 0.8,
    submitted: bool = False,
    skipped: bool = False,
) -> VerdictEligibilityInput:
    return VerdictEligibilityInput(
        paper_id="paper-001",
        has_our_participation=has_our_participation,
        distinct_citable_other_agents=distinct_citable_other_agents,
        open_time=_OPEN,
        audit_artifact_ready=audit_artifact_ready,
        internal_score_confidence=internal_score_confidence,
        submitted=submitted,
        skipped=skipped,
    )


_NOW_VERDICT = _at(60)   # inside SUBMISSION_WINDOW
_NOW_REVIEW = _at(24)    # inside REVIEW_ACTIVE


# ---------------------------------------------------------------------------
# can_submit_verdict — hard gate
# ---------------------------------------------------------------------------

def test_can_submit_when_all_conditions_met():
    assert can_submit_verdict(_make(), _NOW_VERDICT) is True


def test_cannot_submit_without_prior_participation():
    assert can_submit_verdict(_make(has_our_participation=False), _NOW_VERDICT) is False


def test_cannot_submit_with_2_agents():
    assert can_submit_verdict(_make(distinct_citable_other_agents=2), _NOW_VERDICT) is False


def test_cannot_submit_with_0_agents():
    assert can_submit_verdict(_make(distinct_citable_other_agents=0), _NOW_VERDICT) is False


def test_can_submit_with_exactly_3_agents():
    assert can_submit_verdict(_make(distinct_citable_other_agents=3), _NOW_VERDICT) is True


def test_can_submit_with_4_agents():
    assert can_submit_verdict(_make(distinct_citable_other_agents=4), _NOW_VERDICT) is True


def test_can_submit_with_more_than_3_agents():
    assert can_submit_verdict(_make(distinct_citable_other_agents=10), _NOW_VERDICT) is True


def test_cannot_submit_outside_verdict_window():
    assert can_submit_verdict(_make(), _NOW_REVIEW) is False


def test_cannot_submit_when_expired():
    assert can_submit_verdict(_make(), _at(75)) is False


def test_cannot_submit_without_audit_artifact():
    assert can_submit_verdict(_make(audit_artifact_ready=False), _NOW_VERDICT) is False


def test_cannot_submit_if_already_submitted():
    assert can_submit_verdict(_make(submitted=True), _NOW_VERDICT) is False


def test_cannot_submit_with_low_confidence():
    assert can_submit_verdict(
        _make(internal_score_confidence=0.3), _NOW_VERDICT, min_confidence=0.6
    ) is False


def test_can_submit_at_confidence_threshold():
    assert can_submit_verdict(
        _make(internal_score_confidence=0.6), _NOW_VERDICT, min_confidence=0.6
    ) is True


def test_custom_min_confidence_respected():
    assert can_submit_verdict(
        _make(internal_score_confidence=0.75), _NOW_VERDICT, min_confidence=0.8
    ) is False
    assert can_submit_verdict(
        _make(internal_score_confidence=0.8), _NOW_VERDICT, min_confidence=0.8
    ) is True


# ---------------------------------------------------------------------------
# compute_eligibility_state
# ---------------------------------------------------------------------------

def test_state_submitted():
    est, _ = compute_eligibility_state(_make(submitted=True), _NOW_VERDICT)
    assert est == EligibilityState.SUBMITTED


def test_state_expired():
    est, reason = compute_eligibility_state(_make(), _at(75))
    assert est == EligibilityState.EXPIRED
    assert "expired" in reason.lower()


def test_state_skipped_by_policy():
    est, _ = compute_eligibility_state(_make(skipped=True), _NOW_VERDICT)
    assert est == EligibilityState.SKIPPED_BY_POLICY


def test_state_not_participated():
    est, reason = compute_eligibility_state(_make(has_our_participation=False), _NOW_VERDICT)
    assert est == EligibilityState.NOT_PARTICIPATED
    assert "participated" in reason.lower()


def test_state_not_enough_others():
    est, reason = compute_eligibility_state(
        _make(distinct_citable_other_agents=2), _NOW_VERDICT
    )
    assert est == EligibilityState.PARTICIPATED_BUT_NOT_ENOUGH_OTHERS
    assert "2" in reason and "3" in reason


def test_state_not_in_verdict_window():
    est, reason = compute_eligibility_state(_make(), _NOW_REVIEW)
    assert est == EligibilityState.ENOUGH_OTHERS_BUT_NOT_IN_VERDICT_WINDOW
    assert "verdict" in reason.lower() or "window" in reason.lower() or "phase" in reason.lower()


def test_state_not_in_verdict_window_no_artifact():
    est, reason = compute_eligibility_state(
        _make(audit_artifact_ready=False), _NOW_VERDICT
    )
    assert est == EligibilityState.ENOUGH_OTHERS_BUT_NOT_IN_VERDICT_WINDOW
    assert "artifact" in reason.lower()


def test_state_eligible_low_confidence():
    est, reason = compute_eligibility_state(
        _make(internal_score_confidence=0.3), _NOW_VERDICT
    )
    assert est == EligibilityState.ELIGIBLE_LOW_CONFIDENCE
    assert "0.30" in reason or "0.3" in reason


def test_state_eligible_ready():
    est, reason = compute_eligibility_state(_make(), _NOW_VERDICT)
    assert est == EligibilityState.ELIGIBLE_READY
    assert reason == ""


def test_submitted_takes_priority_over_expired():
    # submitted=True evaluated before expired check
    est, _ = compute_eligibility_state(_make(submitted=True), _at(75))
    assert est == EligibilityState.SUBMITTED


def test_expired_takes_priority_over_skipped():
    est, _ = compute_eligibility_state(_make(skipped=True), _at(75))
    assert est == EligibilityState.EXPIRED


# ---------------------------------------------------------------------------
# Min constants
# ---------------------------------------------------------------------------

def test_min_distinct_other_agents_is_3():
    assert MIN_DISTINCT_OTHER_AGENTS == 3


def test_min_verdict_confidence_is_0_6():
    assert MIN_VERDICT_CONFIDENCE == 0.6


# ---------------------------------------------------------------------------
# Rule change: threshold lowered from 5 → 3 (21-participant field)
# ---------------------------------------------------------------------------

def test_2_agents_is_not_enough():
    assert can_submit_verdict(_make(distinct_citable_other_agents=2), _NOW_VERDICT) is False


def test_3_agents_is_enough_when_all_other_gates_pass():
    assert can_submit_verdict(_make(distinct_citable_other_agents=3), _NOW_VERDICT) is True


def test_3_agents_produces_eligible_ready_state():
    est, reason = compute_eligibility_state(_make(distinct_citable_other_agents=3), _NOW_VERDICT)
    assert est == EligibilityState.ELIGIBLE_READY
    assert reason == ""


def test_2_agents_produces_not_enough_others_state():
    est, reason = compute_eligibility_state(_make(distinct_citable_other_agents=2), _NOW_VERDICT)
    assert est == EligibilityState.PARTICIPATED_BUT_NOT_ENOUGH_OTHERS


def test_4_agents_is_enough_with_new_threshold():
    assert can_submit_verdict(_make(distinct_citable_other_agents=4), _NOW_VERDICT) is True
