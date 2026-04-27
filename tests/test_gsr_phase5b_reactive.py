"""Tests for Phase 5B reactive candidate selection and execution."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from gsr_agent.commenting.reactive_analysis import (
    ReactiveAnalysisResult,
    _STRONG_CONTRADICTION_OVERRIDE_CONFIDENCE,
    select_best_reactive_candidate,
    select_best_reactive_candidate_for_paper,
)
from gsr_agent.commenting.orchestrator import plan_and_post_reactive_comment, _prepare_reactive_body
from gsr_agent.koala.models import Paper

UTC = timezone.utc
_OPEN = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
_NOW_REVIEW = _OPEN + timedelta(hours=10)

_PAPER_ID = "paper-5b-001"
_COMMENT_ID = "comment-ext-001"


def _make_paper() -> Paper:
    from gsr_agent.rules.timeline import compute_paper_windows
    w = compute_paper_windows(_OPEN)
    return Paper(
        paper_id=_PAPER_ID,
        title="Phase 5B Test Paper",
        open_time=w.open_time,
        review_end_time=w.review_end_time,
        verdict_end_time=w.verdict_end_time,
        state="REVIEW_ACTIVE",
        abstract="A paper about testing Phase 5B.",
    )


def _make_react_result(
    comment_id: str = _COMMENT_ID,
    confidence: float = 0.8,
) -> ReactiveAnalysisResult:
    verif = {
        "id": f"verif-{comment_id}",
        "claim_id": f"claim-{comment_id}",
        "verdict": "refuted",
        "confidence": confidence,
        "reasoning": "Table 3 contradicts the claim.",
    }
    return ReactiveAnalysisResult(
        comment_id=comment_id,
        paper_id=_PAPER_ID,
        recommendation="react",
        verifications=[verif],
        draft_text=(
            f"[DRY-RUN — not posted]\n"
            f"Reactive fact-check draft for comment {comment_id}:\n\n"
            f"Claim: The model achieves 95% accuracy.\n"
            f"Verdict: refuted (confidence {confidence:.2f})\n"
            f"Reasoning: Table 3 shows 87%, not 95%."
        ),
    )


def _make_skip_result(comment_id: str = _COMMENT_ID) -> ReactiveAnalysisResult:
    return ReactiveAnalysisResult(
        comment_id=comment_id,
        paper_id=_PAPER_ID,
        recommendation="skip",
        skip_reason="all claims supported by paper",
    )


def _make_unclear_result(comment_id: str = _COMMENT_ID) -> ReactiveAnalysisResult:
    return ReactiveAnalysisResult(
        comment_id=comment_id,
        paper_id=_PAPER_ID,
        recommendation="unclear",
    )


def _make_client() -> MagicMock:
    client = MagicMock()
    client._test_mode = True
    client.post_comment.return_value = "reactive-comment-001"
    return client


def _make_db(participated: bool = False) -> MagicMock:
    db = MagicMock()
    db.has_prior_participation.return_value = participated
    return db


# ---------------------------------------------------------------------------
# A. Candidate selection — basic ranking
# ---------------------------------------------------------------------------

def test_select_best_returns_react_result():
    result = _make_react_result(confidence=0.8)
    selected = select_best_reactive_candidate([result], distinct_citable_other_agents=2)
    assert selected is result


def test_select_best_empty_list_returns_none():
    assert select_best_reactive_candidate([], distinct_citable_other_agents=2) is None


def test_select_best_all_skip_returns_none():
    results = [_make_skip_result("c1"), _make_skip_result("c2")]
    assert select_best_reactive_candidate(results, distinct_citable_other_agents=2) is None


def test_select_best_all_unclear_returns_none():
    results = [_make_unclear_result("c1"), _make_unclear_result("c2")]
    assert select_best_reactive_candidate(results, distinct_citable_other_agents=2) is None


def test_select_best_mixed_picks_react():
    react = _make_react_result(comment_id="c-react", confidence=0.8)
    skip = _make_skip_result("c-skip")
    unclear = _make_unclear_result("c-unclear")
    selected = select_best_reactive_candidate(
        [skip, react, unclear], distinct_citable_other_agents=2
    )
    assert selected is react


def test_select_best_picks_highest_confidence():
    low = _make_react_result(comment_id="c-low", confidence=0.6)
    high = _make_react_result(comment_id="c-high", confidence=0.9)
    selected = select_best_reactive_candidate([low, high], distinct_citable_other_agents=2)
    assert selected is high


def test_select_best_picks_highest_confidence_reverse_order():
    high = _make_react_result(comment_id="c-high", confidence=0.9)
    low = _make_react_result(comment_id="c-low", confidence=0.6)
    selected = select_best_reactive_candidate([high, low], distinct_citable_other_agents=2)
    assert selected is high


# ---------------------------------------------------------------------------
# B. Heat-band preference affects selection
# ---------------------------------------------------------------------------

def test_heat_goldilocks_2_returns_candidate():
    result = _make_react_result(confidence=0.6)
    assert select_best_reactive_candidate([result], distinct_citable_other_agents=2) is result


def test_heat_goldilocks_3_returns_candidate():
    result = _make_react_result(confidence=0.6)
    assert select_best_reactive_candidate([result], distinct_citable_other_agents=3) is result


def test_heat_warm_1_returns_candidate():
    """Warm band is neutral — candidate returned without override requirement."""
    result = _make_react_result(confidence=0.6)
    assert select_best_reactive_candidate([result], distinct_citable_other_agents=1) is result


def test_heat_cold_low_confidence_returns_none():
    """Cold paper with weak contradiction → soft penalty → no candidate."""
    low_conf = _STRONG_CONTRADICTION_OVERRIDE_CONFIDENCE - 0.1
    result = _make_react_result(confidence=low_conf)
    assert select_best_reactive_candidate([result], distinct_citable_other_agents=0) is None


def test_heat_cold_strong_contradiction_overrides():
    """Cold paper with strong contradiction confidence → override soft penalty."""
    result = _make_react_result(confidence=_STRONG_CONTRADICTION_OVERRIDE_CONFIDENCE)
    assert select_best_reactive_candidate([result], distinct_citable_other_agents=0) is result


def test_heat_crowded_low_confidence_returns_none():
    """Crowded paper (4–6) with low confidence → deprioritized."""
    low_conf = _STRONG_CONTRADICTION_OVERRIDE_CONFIDENCE - 0.1
    result = _make_react_result(confidence=low_conf)
    assert select_best_reactive_candidate([result], distinct_citable_other_agents=5) is None


def test_heat_crowded_strong_contradiction_overrides():
    """Crowded paper with strong contradiction → override."""
    result = _make_react_result(confidence=_STRONG_CONTRADICTION_OVERRIDE_CONFIDENCE)
    assert select_best_reactive_candidate([result], distinct_citable_other_agents=5) is result


def test_heat_saturated_low_confidence_returns_none():
    """Saturated paper (7+) with low confidence → deprioritized."""
    low_conf = _STRONG_CONTRADICTION_OVERRIDE_CONFIDENCE - 0.1
    result = _make_react_result(confidence=low_conf)
    assert select_best_reactive_candidate([result], distinct_citable_other_agents=8) is None


def test_heat_saturated_strong_contradiction_overrides():
    """Saturated paper with strong contradiction → still return candidate."""
    result = _make_react_result(confidence=_STRONG_CONTRADICTION_OVERRIDE_CONFIDENCE)
    assert select_best_reactive_candidate([result], distinct_citable_other_agents=8) is result


# ---------------------------------------------------------------------------
# C. Posting — dry-run is the default
# ---------------------------------------------------------------------------

def test_plan_and_post_reactive_is_dry_run_by_default(monkeypatch):
    """With KOALA_RUN_MODE unset (defaults to dry_run), returns None without posting."""
    monkeypatch.delenv("KOALA_RUN_MODE", raising=False)
    paper = _make_paper()
    candidate = _make_react_result()
    client = _make_client()
    db = _make_db()

    result = plan_and_post_reactive_comment(
        paper, candidate, client, db,
        karma_remaining=50.0, now=_NOW_REVIEW,
        test_mode=False,
    )

    assert result is None
    assert not client.post_comment.called


def test_plan_and_post_reactive_dry_run_logs_action(monkeypatch):
    """Dry-run mode logs the intended action with status='dry_run'."""
    monkeypatch.delenv("KOALA_RUN_MODE", raising=False)
    paper = _make_paper()
    candidate = _make_react_result()
    client = _make_client()
    db = _make_db()

    plan_and_post_reactive_comment(
        paper, candidate, client, db,
        karma_remaining=50.0, now=_NOW_REVIEW,
        test_mode=False,
    )

    assert db.log_action.called
    call_kwargs = db.log_action.call_args[1]
    assert call_kwargs["status"] == "dry_run"
    assert call_kwargs["details"]["source_comment_id"] == _COMMENT_ID


def test_plan_and_post_reactive_test_mode_returns_comment_id():
    """test_mode=True uses stub client and returns the created comment ID."""
    paper = _make_paper()
    candidate = _make_react_result()
    client = _make_client()
    db = _make_db()

    result = plan_and_post_reactive_comment(
        paper, candidate, client, db,
        karma_remaining=50.0, now=_NOW_REVIEW,
        test_mode=True,
    )

    assert result == "reactive-comment-001"
    assert client.post_comment.called


def test_plan_and_post_reactive_test_mode_logs_success():
    """test_mode=True logs action with status='success'."""
    paper = _make_paper()
    candidate = _make_react_result()
    client = _make_client()
    db = _make_db()

    plan_and_post_reactive_comment(
        paper, candidate, client, db,
        karma_remaining=50.0, now=_NOW_REVIEW,
        test_mode=True,
    )

    assert db.log_action.called
    call_kwargs = db.log_action.call_args[1]
    assert call_kwargs["status"] == "success"


def test_plan_and_post_reactive_records_karma():
    """test_mode=True records karma spend in DB."""
    paper = _make_paper()
    candidate = _make_react_result()
    client = _make_client()
    db = _make_db()

    plan_and_post_reactive_comment(
        paper, candidate, client, db,
        karma_remaining=50.0, now=_NOW_REVIEW,
        test_mode=True,
    )

    assert db.record_karma.called


def test_plan_and_post_reactive_none_draft_returns_none():
    """Candidate with no draft text returns None without calling the client."""
    paper = _make_paper()
    candidate = ReactiveAnalysisResult(
        comment_id=_COMMENT_ID,
        paper_id=_PAPER_ID,
        recommendation="react",
        draft_text=None,
    )
    client = _make_client()
    db = _make_db()

    result = plan_and_post_reactive_comment(
        paper, candidate, client, db,
        karma_remaining=50.0, now=_NOW_REVIEW,
        test_mode=True,
    )

    assert result is None
    assert not client.post_comment.called


def test_plan_and_post_reactive_passes_valid_github_url():
    """test_mode=True provides a structurally valid GitHub URL to post_comment."""
    paper = _make_paper()
    candidate = _make_react_result()
    client = _make_client()
    db = _make_db()

    plan_and_post_reactive_comment(
        paper, candidate, client, db,
        karma_remaining=50.0, now=_NOW_REVIEW,
        test_mode=True,
    )

    call_args = client.post_comment.call_args[0]
    github_url = call_args[2]
    assert github_url.startswith("https://github.com/")
    assert not github_url.startswith("TODO:")


# ---------------------------------------------------------------------------
# D. Citation count is already 3 in verdict/preflight flow
# ---------------------------------------------------------------------------

def test_min_distinct_other_agents_is_3():
    """Phase 5B uses the 3-citation rule (down from 5 in earlier plans)."""
    from gsr_agent.rules.verdict_eligibility import MIN_DISTINCT_OTHER_AGENTS
    assert MIN_DISTINCT_OTHER_AGENTS == 3


# ---------------------------------------------------------------------------
# E. Phase 5B.5 — wrapper, thread awareness, body hardening
# ---------------------------------------------------------------------------

# -- E1. select_best_reactive_candidate_for_paper wrapper -------------------

def test_wrapper_calls_get_comment_stats():
    """Wrapper must call db.get_comment_stats to read the crowding count."""
    results = [_make_react_result(confidence=0.8)]
    db = _make_db()
    db.get_comment_stats.return_value = {"total": 5, "ours": 1, "citable_other": 2}

    select_best_reactive_candidate_for_paper(_PAPER_ID, results, db)

    db.get_comment_stats.assert_called_once_with(_PAPER_ID)


def test_wrapper_picks_goldilocks_candidate():
    """Wrapper delegates correctly: goldilocks count (2) returns the react result."""
    result = _make_react_result(confidence=0.8)
    db = _make_db()
    db.get_comment_stats.return_value = {"total": 3, "ours": 1, "citable_other": 2}

    selected = select_best_reactive_candidate_for_paper(_PAPER_ID, [result], db)

    assert selected is result


def test_wrapper_cold_paper_no_override_returns_none():
    """Wrapper returns None when paper is cold and contradiction is weak."""
    low_conf = _STRONG_CONTRADICTION_OVERRIDE_CONFIDENCE - 0.1
    result = _make_react_result(confidence=low_conf)
    db = _make_db()
    db.get_comment_stats.return_value = {"total": 1, "ours": 1, "citable_other": 0}

    selected = select_best_reactive_candidate_for_paper(_PAPER_ID, [result], db)

    assert selected is None


def test_wrapper_cold_paper_strong_contradiction_overrides():
    """Wrapper allows cold paper when contradiction confidence exceeds override threshold."""
    result = _make_react_result(confidence=_STRONG_CONTRADICTION_OVERRIDE_CONFIDENCE)
    db = _make_db()
    db.get_comment_stats.return_value = {"total": 1, "ours": 1, "citable_other": 0}

    selected = select_best_reactive_candidate_for_paper(_PAPER_ID, [result], db)

    assert selected is result


# -- E2. Thread-aware posting -----------------------------------------------

def _make_react_result_with_thread(
    comment_id: str = _COMMENT_ID,
    thread_id: str = "thread-001",
    confidence: float = 0.8,
) -> ReactiveAnalysisResult:
    result = _make_react_result(comment_id=comment_id, confidence=confidence)
    result.thread_id = thread_id
    return result


def test_thread_id_passed_to_post_comment():
    """When candidate has a thread_id, it is forwarded to client.post_comment."""
    paper = _make_paper()
    candidate = _make_react_result_with_thread(thread_id="thread-abc")
    client = _make_client()
    db = _make_db()

    plan_and_post_reactive_comment(
        paper, candidate, client, db,
        karma_remaining=50.0, now=_NOW_REVIEW,
        test_mode=True,
    )

    call_kwargs = client.post_comment.call_args[1]
    assert call_kwargs["thread_id"] == "thread-abc"


def test_parent_id_set_to_source_comment_id():
    """parent_id is always set to the source comment being reacted to."""
    paper = _make_paper()
    candidate = _make_react_result_with_thread(comment_id="comment-source-99")
    client = _make_client()
    db = _make_db()

    plan_and_post_reactive_comment(
        paper, candidate, client, db,
        karma_remaining=50.0, now=_NOW_REVIEW,
        test_mode=True,
    )

    call_kwargs = client.post_comment.call_args[1]
    assert call_kwargs["parent_id"] == "comment-source-99"


def test_fallback_top_level_when_no_thread():
    """When thread_id is None, thread_id kwarg is None (API omits it → top-level)."""
    paper = _make_paper()
    candidate = _make_react_result()  # thread_id defaults to None
    assert candidate.thread_id is None
    client = _make_client()
    db = _make_db()

    plan_and_post_reactive_comment(
        paper, candidate, client, db,
        karma_remaining=50.0, now=_NOW_REVIEW,
        test_mode=True,
    )

    call_kwargs = client.post_comment.call_args[1]
    assert call_kwargs["thread_id"] is None


# -- E3. _prepare_reactive_body hardening -----------------------------------

def test_prepare_reactive_body_strips_dry_run_header():
    draft = "[DRY-RUN — not posted]\nSome real content here."
    assert _prepare_reactive_body(draft) == "Some real content here."


def test_prepare_reactive_body_returns_none_for_header_only():
    """After stripping the header, if nothing remains, return None."""
    draft = "[DRY-RUN — not posted]"
    assert _prepare_reactive_body(draft) is None


def test_prepare_reactive_body_returns_none_for_whitespace_after_strip():
    """Whitespace-only body after stripping → None."""
    draft = "[DRY-RUN — not posted]\n   \n   "
    assert _prepare_reactive_body(draft) is None


def test_prepare_reactive_body_returns_none_for_none_input():
    assert _prepare_reactive_body(None) is None


def test_prepare_reactive_body_leaves_no_header_text_unchanged():
    """Text without the DRY-RUN header passes through unchanged."""
    draft = "Claim: X\nVerdict: refuted"
    assert _prepare_reactive_body(draft) == draft


def test_empty_body_after_strip_does_not_post():
    """Candidate whose draft_text is only the DRY-RUN header → no post."""
    paper = _make_paper()
    candidate = ReactiveAnalysisResult(
        comment_id=_COMMENT_ID,
        paper_id=_PAPER_ID,
        recommendation="react",
        draft_text="[DRY-RUN — not posted]",
    )
    client = _make_client()
    db = _make_db()

    result = plan_and_post_reactive_comment(
        paper, candidate, client, db,
        karma_remaining=50.0, now=_NOW_REVIEW,
        test_mode=True,
    )

    assert result is None
    assert not client.post_comment.called
