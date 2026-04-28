"""Tests for gsr_agent.commenting.orchestrator — plan_and_post_seed_comment."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from gsr_agent.commenting.orchestrator import plan_and_post_seed_comment
from gsr_agent.koala.errors import KoalaPreflightError
from gsr_agent.koala.models import Paper

UTC = timezone.utc
_OPEN = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
_NOW_SEED = _OPEN + timedelta(hours=6)
_NOW_BUILD = _OPEN + timedelta(hours=20)


def _make_paper(abstract: str = "We propose a novel method for faster gradient estimation.") -> Paper:
    from gsr_agent.rules.timeline import compute_paper_windows
    w = compute_paper_windows(_OPEN)
    return Paper(
        paper_id="paper-001",
        title="Gradient Estimation Paper",
        open_time=w.open_time,
        review_end_time=w.review_end_time,
        verdict_end_time=w.verdict_end_time,
        state="REVIEW_ACTIVE",
        abstract=abstract,
    )


def _make_client(test_mode: bool = True) -> MagicMock:
    client = MagicMock()
    client._test_mode = test_mode
    client.post_comment.return_value = "comment-001"
    return client


def _make_db() -> MagicMock:
    db = MagicMock()
    db.has_prior_participation.return_value = False
    db.get_karma_spent.return_value = 0.0
    return db


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_plan_and_post_returns_comment_id():
    paper = _make_paper()
    client = _make_client()
    db = _make_db()
    result = plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=True
    )
    assert result[0] is not None
    assert isinstance(result[0], str)
    assert result[1] is None


def test_plan_and_post_calls_post_comment():
    paper = _make_paper()
    client = _make_client()
    db = _make_db()
    plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=True
    )
    assert client.post_comment.called


def test_plan_and_post_passes_paper_id_to_post_comment():
    paper = _make_paper()
    client = _make_client()
    db = _make_db()
    plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=True
    )
    call_args = client.post_comment.call_args
    assert call_args[0][0] == "paper-001"


def test_plan_and_post_logs_action_in_db():
    paper = _make_paper()
    client = _make_client()
    db = _make_db()
    plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=True
    )
    assert db.log_action.called


def test_plan_and_post_records_karma_in_db():
    paper = _make_paper()
    client = _make_client()
    db = _make_db()
    plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=True
    )
    assert db.record_karma.called


# ---------------------------------------------------------------------------
# Skip paths — returns None without posting
# ---------------------------------------------------------------------------

def test_returns_none_when_not_in_seed_window():
    paper = _make_paper()
    client = _make_client()
    db = _make_db()
    result = plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=_NOW_BUILD, test_mode=True
    )
    assert result[0] is None
    assert result[1] == "seed_plan_not_seed_opportunity"
    assert not client.post_comment.called


def test_returns_none_when_already_participated():
    paper = _make_paper()
    client = _make_client()
    db = _make_db()
    db.has_prior_participation.return_value = True
    result = plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=True
    )
    assert result[0] is None
    assert result[1] == "seed_plan_not_seed_opportunity"


def test_returns_none_when_no_karma():
    paper = _make_paper()
    client = _make_client()
    db = _make_db()
    result = plan_and_post_seed_comment(
        paper, client, db, karma_remaining=0.0, now=_NOW_SEED, test_mode=True
    )
    assert result[0] is None
    assert result[1] == "seed_plan_not_seed_opportunity"


def test_returns_none_when_no_abstract():
    paper = _make_paper(abstract="")
    client = _make_client()
    db = _make_db()
    result = plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=True
    )
    assert result[0] is None
    assert result[1] == "seed_plan_missing_abstract"
    assert not client.post_comment.called


# ---------------------------------------------------------------------------
# Preflight enforcement
# ---------------------------------------------------------------------------

def test_post_comment_receives_valid_github_url():
    paper = _make_paper()
    client = _make_client()
    db = _make_db()
    plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=True
    )
    call_args = client.post_comment.call_args
    github_url = call_args[0][2]
    assert github_url.startswith("https://github.com/")
    assert not github_url.startswith("TODO:")


# ---------------------------------------------------------------------------
# Live preflight — API token acceptance (test_mode=False)
# ---------------------------------------------------------------------------

_FAKE_ARTIFACT_URL = (
    "https://github.com/owner/repo/blob/main/logs/paper-001/comment_draft_paper-001_ts.md"
)


def _patch_live_artifacts(monkeypatch):
    monkeypatch.setattr(
        "gsr_agent.commenting.orchestrator.publish_comment_artifact",
        lambda paper_id, body, test_mode=False: _FAKE_ARTIFACT_URL,
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.orchestrator.validate_artifact_for_live_action",
        lambda url: None,
    )


def test_live_preflight_accepts_koala_api_token(monkeypatch):
    monkeypatch.setenv("KOALA_API_TOKEN", "tok-live-test")
    monkeypatch.delenv("KOALA_API_KEY", raising=False)
    monkeypatch.setenv("KOALA_RUN_MODE", "live")
    _patch_live_artifacts(monkeypatch)
    result = plan_and_post_seed_comment(
        _make_paper(), _make_client(), _make_db(), karma_remaining=50.0,
        now=_NOW_SEED, test_mode=False,
    )
    assert result[0] is not None
    assert result[1] is None


def test_live_preflight_accepts_koala_api_key_only(monkeypatch):
    monkeypatch.delenv("KOALA_API_TOKEN", raising=False)
    monkeypatch.setenv("KOALA_API_KEY", "key-live-test")
    monkeypatch.setenv("KOALA_RUN_MODE", "live")
    _patch_live_artifacts(monkeypatch)
    result = plan_and_post_seed_comment(
        _make_paper(), _make_client(), _make_db(), karma_remaining=50.0,
        now=_NOW_SEED, test_mode=False,
    )
    assert result[0] is not None
    assert result[1] is None


def test_live_preflight_missing_both_returns_skip_reason(monkeypatch):
    monkeypatch.delenv("KOALA_API_TOKEN", raising=False)
    monkeypatch.delenv("KOALA_API_KEY", raising=False)
    monkeypatch.setenv("KOALA_RUN_MODE", "live")
    _patch_live_artifacts(monkeypatch)
    result = plan_and_post_seed_comment(
        _make_paper(), _make_client(), _make_db(), karma_remaining=50.0,
        now=_NOW_SEED, test_mode=False,
    )
    assert result == (None, "seed_plan_preflight_failed")


def test_live_preflight_no_token_does_not_post(monkeypatch):
    monkeypatch.delenv("KOALA_API_TOKEN", raising=False)
    monkeypatch.delenv("KOALA_API_KEY", raising=False)
    monkeypatch.setenv("KOALA_RUN_MODE", "live")
    _patch_live_artifacts(monkeypatch)
    client = _make_client()
    plan_and_post_seed_comment(
        _make_paper(), client, _make_db(), karma_remaining=50.0,
        now=_NOW_SEED, test_mode=False,
    )
    assert not client.post_comment.called


def test_live_preflight_does_not_log_token_values(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("KOALA_API_TOKEN", "super-secret-token-xyz")
    monkeypatch.delenv("KOALA_API_KEY", raising=False)
    monkeypatch.setenv("KOALA_RUN_MODE", "live")
    _patch_live_artifacts(monkeypatch)
    with caplog.at_level(logging.DEBUG, logger="gsr_agent"):
        plan_and_post_seed_comment(
            _make_paper(), _make_client(), _make_db(), karma_remaining=50.0,
            now=_NOW_SEED, test_mode=False,
        )
    for record in caplog.records:
        assert "super-secret-token-xyz" not in record.message

# ---------------------------------------------------------------------------
# Phase:NEW preflight fix — pre-open papers in SEED_WINDOW must not raise
# ---------------------------------------------------------------------------

def _make_pre_open_paper(
    abstract: str = "We propose a novel method for faster gradient estimation.",
) -> Paper:
    """open_time is 2h in the future: Phase=NEW, Micro=SEED_WINDOW."""
    from gsr_agent.rules.timeline import compute_paper_windows
    _future_open = datetime(2026, 4, 24, 14, 0, 0, tzinfo=UTC)  # 2h after _NOW_BUILD baseline
    _now_pre_open = _future_open - timedelta(hours=2)
    w = compute_paper_windows(_future_open)
    return Paper(
        paper_id="paper-new-001",
        title="Pre-Open Seed Paper",
        open_time=w.open_time,
        review_end_time=w.review_end_time,
        verdict_end_time=w.verdict_end_time,
        state="REVIEW_ACTIVE",
        abstract=abstract,
    ), _now_pre_open


def test_pre_open_paper_does_not_raise_preflight():
    """Phase:NEW paper (now < open_time) in SEED_WINDOW must not raise KoalaPreflightError."""
    paper, now_pre_open = _make_pre_open_paper()
    client = _make_client()
    db = _make_db()
    # Should NOT raise; should return a comment id
    result = plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=now_pre_open, test_mode=True
    )
    assert result[0] is not None


def test_pre_open_paper_creates_seed_comment():
    """Phase:NEW paper calls post_comment on the stub client."""
    paper, now_pre_open = _make_pre_open_paper()
    client = _make_client()
    db = _make_db()
    plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=now_pre_open, test_mode=True
    )
    assert client.post_comment.called


def test_pre_open_paper_no_abstract_returns_none():
    """Phase:NEW paper with no abstract returns seed_plan_missing_abstract."""
    paper, now_pre_open = _make_pre_open_paper(abstract="")
    client = _make_client()
    db = _make_db()
    result = plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=now_pre_open, test_mode=True
    )
    assert result[0] is None
    assert result[1] == "seed_plan_missing_abstract"
    assert not client.post_comment.called


# ---------------------------------------------------------------------------
# Low-signal abstract and moderation 422 low_effort skip paths
# ---------------------------------------------------------------------------

def test_low_signal_abstract_returns_low_signal_reason():
    """Abstract present but too short to generate a concrete question."""
    paper = _make_paper(abstract="A study on X.")  # 14 chars / 4 words — below threshold
    client = _make_client()
    db = _make_db()
    result = plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=True
    )
    assert result == (None, "seed_plan_low_signal")
    assert not client.post_comment.called


def test_moderation_422_low_effort_returns_skip_reason():
    """Koala 422 low_effort rejection is caught and returned as a skip reason."""
    from gsr_agent.koala.errors import KoalaAPIError
    paper = _make_paper()
    client = _make_client()
    db = _make_db()
    client.post_comment.side_effect = KoalaAPIError(
        "Koala API error: POST /comments/ → 422: "
        '{"detail": {"message": "Comment rejected by moderation", "category": "low_effort"}}'
    )
    result = plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=True
    )
    assert result == (None, "seed_plan_moderation_low_effort")
    assert not db.record_karma.called


def test_non_422_koala_error_propagates():
    """Non-moderation KoalaAPIError is re-raised, not swallowed."""
    from gsr_agent.koala.errors import KoalaAPIError
    paper = _make_paper()
    client = _make_client()
    db = _make_db()
    client.post_comment.side_effect = KoalaAPIError("Koala API error: POST /comments/ → 500: server error")
    with pytest.raises(KoalaAPIError):
        plan_and_post_seed_comment(
            paper, client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=True
        )
