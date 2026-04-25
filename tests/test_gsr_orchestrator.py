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
    assert result is not None
    assert isinstance(result, str)


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
    assert result is None
    assert not client.post_comment.called


def test_returns_none_when_already_participated():
    paper = _make_paper()
    client = _make_client()
    db = _make_db()
    db.has_prior_participation.return_value = True
    result = plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=True
    )
    assert result is None


def test_returns_none_when_no_karma():
    paper = _make_paper()
    client = _make_client()
    db = _make_db()
    result = plan_and_post_seed_comment(
        paper, client, db, karma_remaining=0.0, now=_NOW_SEED, test_mode=True
    )
    assert result is None


def test_returns_none_when_no_abstract():
    paper = _make_paper(abstract="")
    client = _make_client()
    db = _make_db()
    result = plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=True
    )
    assert result is None
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
