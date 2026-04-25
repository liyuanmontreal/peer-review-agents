"""Tests for gsr_agent.rules.preflight — action preflight checks."""

from datetime import datetime, timedelta, timezone

import pytest

from gsr_agent.koala.errors import KoalaPreflightError
from gsr_agent.rules.preflight import (
    CommentPreflightInput,
    VerdictPreflightInput,
    preflight_comment_action,
    preflight_verdict_action,
)
from gsr_agent.rules.verdict_eligibility import VerdictEligibilityInput

UTC = timezone.utc
_OPEN = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
_REAL_URL = "https://github.com/owner/repo/blob/main/logs/paper-001/artifact.md"


def _at(hours: float) -> datetime:
    return _OPEN + timedelta(hours=hours)


_NOW_REVIEW = _at(6)    # SEED_WINDOW inside REVIEW_ACTIVE
_NOW_VERDICT = _at(60)  # SUBMISSION_WINDOW inside VERDICT_ACTIVE


def _make_comment(**kwargs) -> CommentPreflightInput:
    defaults = dict(
        paper_id="paper-001",
        body="This paper claims X but the support in §3 appears limited to Y.",
        github_file_url=_REAL_URL,
        open_time=_OPEN,
        now=_NOW_REVIEW,
        karma_remaining=50.0,
        has_prior_participation=False,
    )
    defaults.update(kwargs)
    return CommentPreflightInput(**defaults)


def _make_eligibility(**kwargs) -> VerdictEligibilityInput:
    defaults = dict(
        paper_id="paper-001",
        has_our_participation=True,
        distinct_citable_other_agents=5,
        open_time=_OPEN,
        audit_artifact_ready=True,
        internal_score_confidence=0.8,
        submitted=False,
        skipped=False,
    )
    defaults.update(kwargs)
    return VerdictEligibilityInput(**defaults)


def _make_verdict(**kwargs) -> VerdictPreflightInput:
    defaults = dict(
        paper_id="paper-001",
        score=7.0,
        cited_comment_ids=["c1", "c2", "c3", "c4", "c5"],
        github_file_url=_REAL_URL,
        eligibility=_make_eligibility(),
        now=_NOW_VERDICT,
        min_confidence=0.6,
    )
    defaults.update(kwargs)
    return VerdictPreflightInput(**defaults)


# ---------------------------------------------------------------------------
# Comment preflight — happy path
# ---------------------------------------------------------------------------

def test_comment_preflight_passes_on_valid_input():
    preflight_comment_action(_make_comment())  # must not raise


def test_comment_preflight_passes_followup():
    preflight_comment_action(_make_comment(has_prior_participation=True, karma_remaining=5.0))


# ---------------------------------------------------------------------------
# Comment preflight — github_file_url
# ---------------------------------------------------------------------------

def test_comment_preflight_rejects_empty_github_url():
    with pytest.raises(KoalaPreflightError, match="github_file_url"):
        preflight_comment_action(_make_comment(github_file_url=""))


def test_comment_preflight_rejects_todo_placeholder():
    todo_url = "TODO: set KOALA_GITHUB_REPO — local artifact at ./logs/paper-001/x.md"
    with pytest.raises(KoalaPreflightError, match="github_file_url"):
        preflight_comment_action(_make_comment(github_file_url=todo_url))


def test_comment_preflight_rejects_test_artifact_placeholder():
    with pytest.raises(KoalaPreflightError, match="github_file_url"):
        preflight_comment_action(_make_comment(github_file_url="test-artifact://paper-001/xyz"))


# ---------------------------------------------------------------------------
# Comment preflight — karma
# ---------------------------------------------------------------------------

def test_comment_preflight_rejects_zero_karma():
    with pytest.raises(KoalaPreflightError, match="karma"):
        preflight_comment_action(_make_comment(karma_remaining=0.0))


def test_comment_preflight_rejects_insufficient_karma_for_first_action():
    with pytest.raises(KoalaPreflightError, match="karma"):
        preflight_comment_action(_make_comment(karma_remaining=0.5, has_prior_participation=False))


def test_comment_preflight_accepts_exact_karma_for_first_action():
    preflight_comment_action(_make_comment(karma_remaining=1.0, has_prior_participation=False))


def test_comment_preflight_accepts_exact_karma_for_followup():
    preflight_comment_action(_make_comment(karma_remaining=0.1, has_prior_participation=True))


def test_comment_preflight_rejects_insufficient_karma_for_followup():
    with pytest.raises(KoalaPreflightError, match="karma"):
        preflight_comment_action(_make_comment(karma_remaining=0.05, has_prior_participation=True))


# ---------------------------------------------------------------------------
# Comment preflight — phase
# ---------------------------------------------------------------------------

def test_comment_preflight_rejects_verdict_phase():
    with pytest.raises(KoalaPreflightError, match="phase"):
        preflight_comment_action(_make_comment(now=_NOW_VERDICT))


def test_comment_preflight_rejects_expired_phase():
    with pytest.raises(KoalaPreflightError, match="phase"):
        preflight_comment_action(_make_comment(now=_at(75)))


def test_comment_preflight_rejects_new_phase():
    with pytest.raises(KoalaPreflightError, match="phase"):
        preflight_comment_action(_make_comment(now=_OPEN - timedelta(hours=1)))


# ---------------------------------------------------------------------------
# Comment preflight — body / moderation
# ---------------------------------------------------------------------------

def test_comment_preflight_rejects_empty_body():
    with pytest.raises(KoalaPreflightError):
        preflight_comment_action(_make_comment(body=""))


def test_comment_preflight_rejects_whitespace_only_body():
    with pytest.raises(KoalaPreflightError):
        preflight_comment_action(_make_comment(body="   \n  "))


def test_comment_preflight_rejects_moderation_blocked_phrase():
    with pytest.raises(KoalaPreflightError, match="[Mm]oderation"):
        preflight_comment_action(_make_comment(body="The authors committed fraud."))


# ---------------------------------------------------------------------------
# Verdict preflight — happy path
# ---------------------------------------------------------------------------

def test_verdict_preflight_passes_on_valid_input():
    preflight_verdict_action(_make_verdict())  # must not raise


def test_verdict_preflight_passes_score_0():
    preflight_verdict_action(_make_verdict(score=0.0))


def test_verdict_preflight_passes_score_10():
    preflight_verdict_action(_make_verdict(score=10.0))


# ---------------------------------------------------------------------------
# Verdict preflight — github_file_url
# ---------------------------------------------------------------------------

def test_verdict_preflight_rejects_empty_github_url():
    with pytest.raises(KoalaPreflightError, match="github_file_url"):
        preflight_verdict_action(_make_verdict(github_file_url=""))


def test_verdict_preflight_rejects_todo_placeholder():
    with pytest.raises(KoalaPreflightError, match="github_file_url"):
        preflight_verdict_action(
            _make_verdict(
                github_file_url="TODO: set KOALA_GITHUB_REPO — local artifact at ./logs/x.md"
            )
        )


# ---------------------------------------------------------------------------
# Verdict preflight — score range
# ---------------------------------------------------------------------------

def test_verdict_preflight_rejects_negative_score():
    with pytest.raises(KoalaPreflightError, match="[Ss]core"):
        preflight_verdict_action(_make_verdict(score=-0.1))


def test_verdict_preflight_rejects_score_above_10():
    with pytest.raises(KoalaPreflightError, match="[Ss]core"):
        preflight_verdict_action(_make_verdict(score=10.1))


def test_verdict_preflight_rejects_score_100():
    with pytest.raises(KoalaPreflightError, match="[Ss]core"):
        preflight_verdict_action(_make_verdict(score=100.0))


# ---------------------------------------------------------------------------
# Verdict preflight — cited_comment_ids count
# ---------------------------------------------------------------------------

def test_verdict_preflight_rejects_too_few_citations():
    with pytest.raises(KoalaPreflightError, match="[Cc]it"):
        preflight_verdict_action(_make_verdict(cited_comment_ids=["c1", "c2"]))


def test_verdict_preflight_rejects_empty_citations():
    with pytest.raises(KoalaPreflightError, match="[Cc]it"):
        preflight_verdict_action(_make_verdict(cited_comment_ids=[]))


def test_verdict_preflight_rejects_4_citations():
    with pytest.raises(KoalaPreflightError, match="[Cc]it"):
        preflight_verdict_action(_make_verdict(cited_comment_ids=["c1", "c2", "c3", "c4"]))


def test_verdict_preflight_accepts_exactly_5_citations():
    preflight_verdict_action(_make_verdict(cited_comment_ids=["c1", "c2", "c3", "c4", "c5"]))


def test_verdict_preflight_deduplicates_cited_ids():
    # 5 entries but only 4 distinct — should be rejected
    with pytest.raises(KoalaPreflightError, match="[Cc]it"):
        preflight_verdict_action(
            _make_verdict(cited_comment_ids=["c1", "c1", "c2", "c3", "c4"])
        )


# ---------------------------------------------------------------------------
# Verdict preflight — eligibility gate
# ---------------------------------------------------------------------------

def test_verdict_preflight_rejects_without_participation():
    with pytest.raises(KoalaPreflightError):
        preflight_verdict_action(
            _make_verdict(eligibility=_make_eligibility(has_our_participation=False))
        )


def test_verdict_preflight_rejects_insufficient_other_agents():
    with pytest.raises(KoalaPreflightError):
        preflight_verdict_action(
            _make_verdict(eligibility=_make_eligibility(distinct_citable_other_agents=4))
        )


def test_verdict_preflight_rejects_review_phase():
    with pytest.raises(KoalaPreflightError):
        preflight_verdict_action(_make_verdict(now=_NOW_REVIEW))


def test_verdict_preflight_rejects_missing_audit_artifact():
    with pytest.raises(KoalaPreflightError):
        preflight_verdict_action(
            _make_verdict(eligibility=_make_eligibility(audit_artifact_ready=False))
        )


def test_verdict_preflight_rejects_already_submitted():
    with pytest.raises(KoalaPreflightError):
        preflight_verdict_action(
            _make_verdict(eligibility=_make_eligibility(submitted=True))
        )
