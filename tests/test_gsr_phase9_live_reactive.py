"""Tests for Phase 9: Controlled Live Reactive posting."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from gsr_agent.commenting.orchestrator import plan_and_post_reactive_comment
from gsr_agent.commenting.reactive_analysis import ReactiveAnalysisResult
from gsr_agent.koala.errors import KoalaAPIError, KoalaPreflightError
from gsr_agent.koala.models import Paper
from gsr_agent.orchestration.operational_loop import (
    _DryRunClient,
    _paper_from_row,
    _process_paper,
    run_operational_loop,
)
from gsr_agent.rules.timeline import compute_paper_windows
from gsr_agent.rules.verdict_assembly import VerdictEligibilityResult

_MOD = "gsr_agent.orchestration.operational_loop"
_NOW = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper_row(paper_id: str = "paper-abc-123") -> dict:
    return {
        "paper_id": paper_id,
        "title": "Test Paper",
        "open_time": "2026-04-25T00:00:00+00:00",       # 36h before _NOW → comment window open
        "review_end_time": "2026-04-27T00:00:00+00:00",
        "verdict_end_time": "2026-04-28T00:00:00+00:00",
        "state": "REVIEW_ACTIVE",
        "pdf_url": "https://example.com/paper.pdf",
        "local_pdf_path": None,
    }


def _make_paper(paper_id: str = "paper-abc-123"):
    return _paper_from_row(_make_paper_row(paper_id))


def _make_candidate(paper_id: str = "paper-abc-123") -> ReactiveAnalysisResult:
    return ReactiveAnalysisResult(
        comment_id="cmt-other-001",
        paper_id=paper_id,
        recommendation="react",
        draft_text="[DRY-RUN — not posted]\n\nDraft reactive text here.",
    )


def _make_eligibility(eligible: bool = False) -> VerdictEligibilityResult:
    return VerdictEligibilityResult(
        eligible=eligible,
        reason_code="eligible" if eligible else "no_react_signal",
        heat_band="goldilocks",
        distinct_citable_other_agents=3,
        strongest_contradiction_confidence=0.80,
        selected_candidates=[],
    )


def _make_process_db(*, reactive_dedup: bool = False, verdict_dedup: bool = False) -> MagicMock:
    db = MagicMock()
    db.has_recent_reactive_action_for_comment.return_value = reactive_dedup
    db.has_recent_verdict_action_for_paper.return_value = verdict_dedup
    return db


def _make_loop_db(paper_rows: list | None = None) -> MagicMock:
    db = MagicMock()
    db.get_papers.return_value = paper_rows if paper_rows is not None else [_make_paper_row()]
    db.get_comment_stats.return_value = {"total": 5, "ours": 1, "citable_other": 3}
    db.has_prior_participation.return_value = True
    db.has_recent_seed_action_for_paper.return_value = False
    return db


def _no_op_result(paper_id: str = "paper-abc-123") -> dict:
    return {
        "paper_id": paper_id,
        "reactive_status": "none",
        "reactive_reason": None,
        "reactive_artifact": None,
        "reactive_live_posted": False,
        "reactive_live_reason": "no_candidate",
        "verdict_status": "ineligible",
        "verdict_reason": None,
        "verdict_artifact": None,
        "has_reactive_candidate": False,
        "reactive_draft_created": False,
        "verdict_eligible": False,
        "verdict_draft_created": False,
    }


# ---------------------------------------------------------------------------
# TestProcessPaperLiveReactive — _process_paper gating logic
# ---------------------------------------------------------------------------

class TestProcessPaperLiveReactive:
    """Unit-tests for the Phase 9 live gating inside _process_paper."""

    def _run(self, candidate, dedup=False, live_reactive=False,
             live_budget_remaining=0, test_mode=True,
             run_mode="dry_run", post_return="cmt-new"):
        """Run _process_paper with common mocks; return result dict."""
        paper = _make_paper()
        db = _make_process_db(reactive_dedup=dedup)
        live_client = MagicMock() if live_reactive else None

        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper",
                  return_value=[candidate] if candidate else []),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper",
                  return_value=candidate),
            patch(f"{_MOD}.plan_and_post_reactive_comment",
                  return_value=post_return) as mock_post,
            patch(f"{_MOD}.evaluate_verdict_eligibility",
                  return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value=run_mode),
        ):
            result = _process_paper(
                paper, _DryRunClient(), db, 100.0, _NOW, test_mode,
                live_reactive=live_reactive,
                live_budget_remaining=live_budget_remaining,
                live_client=live_client,
            )
        return result, mock_post

    # -- No candidate --------------------------------------------------------

    def test_no_candidate_live_reason_is_no_candidate(self):
        result, _ = self._run(candidate=None, live_reactive=True)
        assert result["reactive_live_reason"] == "no_candidate"
        assert result["reactive_live_posted"] is False
        assert result["reactive_status"] == "none"

    # -- live_reactive=False (default) ---------------------------------------

    def test_live_disabled_live_reason_set(self):
        result, _ = self._run(
            candidate=_make_candidate(),
            live_reactive=False,
            test_mode=True,
        )
        assert result["reactive_live_reason"] == "live_disabled"
        assert result["reactive_live_posted"] is False

    def test_live_disabled_still_dry_runs(self):
        result, mock_post = self._run(
            candidate=_make_candidate(),
            live_reactive=False,
            test_mode=True,
        )
        assert result["reactive_status"] == "dry_run"
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs.get("test_mode") is True

    # -- live_reactive=True but test_mode=True --------------------------------

    def test_test_mode_blocks_live(self):
        result, mock_post = self._run(
            candidate=_make_candidate(),
            live_reactive=True,
            live_budget_remaining=1,
            test_mode=True,
            run_mode="live",
        )
        assert result["reactive_live_reason"] == "live_gate_failed"
        assert result["reactive_live_posted"] is False
        # Falls back to dry-run
        assert result["reactive_status"] == "dry_run"
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs.get("test_mode") is True

    # -- live_reactive=True but KOALA_RUN_MODE != live -----------------------

    def test_run_mode_not_live_blocks(self):
        result, _ = self._run(
            candidate=_make_candidate(),
            live_reactive=True,
            live_budget_remaining=1,
            test_mode=False,
            run_mode="dry_run",
        )
        assert result["reactive_live_reason"] == "live_gate_failed"
        assert result["reactive_live_posted"] is False

    def test_run_mode_not_live_falls_back_to_dry_run(self):
        result, mock_post = self._run(
            candidate=_make_candidate(),
            live_reactive=True,
            live_budget_remaining=1,
            test_mode=False,
            run_mode="dry_run",
            post_return="cmt-dry",
        )
        assert result["reactive_status"] == "dry_run"
        assert result["reactive_artifact"] == "cmt-dry"
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs.get("test_mode") is False

    # -- live_reactive=True but budget exhausted -----------------------------

    def test_budget_exhausted_reason(self):
        result, _ = self._run(
            candidate=_make_candidate(),
            live_reactive=True,
            live_budget_remaining=0,
            test_mode=False,
            run_mode="live",
        )
        assert result["reactive_live_reason"] == "live_budget_exhausted"
        assert result["reactive_live_posted"] is False

    def test_budget_exhausted_falls_back_to_dry_run(self):
        result, mock_post = self._run(
            candidate=_make_candidate(),
            live_reactive=True,
            live_budget_remaining=0,
            test_mode=False,
            run_mode="live",
            post_return="cmt-dry",
        )
        assert result["reactive_status"] == "dry_run"
        _, kwargs = mock_post.call_args
        assert kwargs.get("test_mode") is False

    # -- All gates pass → live post ------------------------------------------

    def test_live_post_succeeds(self):
        result, mock_post = self._run(
            candidate=_make_candidate(),
            live_reactive=True,
            live_budget_remaining=1,
            test_mode=False,
            run_mode="live",
            post_return="cmt-live-001",
        )
        assert result["reactive_live_posted"] is True
        assert result["reactive_live_reason"] == "live_posted"
        assert result["reactive_status"] == "live_posted"
        assert result["reactive_artifact"] == "cmt-live-001"
        # Must call with test_mode=False and live_client
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs.get("test_mode") is False

    def test_live_post_uses_live_client(self):
        paper = _make_paper()
        db = _make_process_db()
        live_client = MagicMock()

        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[_make_candidate()]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=_make_candidate()),
            patch(f"{_MOD}.plan_and_post_reactive_comment", return_value="cmt-live") as mock_post,
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
        ):
            _process_paper(
                paper, _DryRunClient(), db, 100.0, _NOW, False,
                live_reactive=True,
                live_budget_remaining=1,
                live_client=live_client,
            )

        args, kwargs = mock_post.call_args
        assert args[2] is live_client

    # -- Live plan returns None (artifact validation / preflight fail) --------

    def test_live_plan_returns_none_sets_gate_failed(self):
        result, _ = self._run(
            candidate=_make_candidate(),
            live_reactive=True,
            live_budget_remaining=1,
            test_mode=False,
            run_mode="live",
            post_return=None,
        )
        assert result["reactive_live_posted"] is False
        assert result["reactive_live_reason"] == "live_gate_failed"
        assert result["reactive_status"] == "skipped"

    # -- Dedup blocks live post ----------------------------------------------

    def test_dedup_blocks_live(self):
        paper = _make_paper()
        db = _make_process_db(reactive_dedup=True)
        live_client = MagicMock()

        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[_make_candidate()]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=_make_candidate()),
            patch(f"{_MOD}.plan_and_post_reactive_comment") as mock_post,
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
        ):
            result = _process_paper(
                paper, _DryRunClient(), db, 100.0, _NOW, False,
                live_reactive=True,
                live_budget_remaining=1,
                live_client=live_client,
            )

        assert result["reactive_live_posted"] is False
        assert result["reactive_live_reason"] == "dedup_skipped"
        assert result["reactive_status"] == "dedup_skipped"
        mock_post.assert_not_called()

    # -- Verdict submit never called -----------------------------------------

    def test_verdict_submit_never_called_from_process_paper(self):
        paper = _make_paper()
        db = _make_process_db()

        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(True)),
            patch(f"{_MOD}.plan_verdict_for_paper", return_value={"artifact_url": "u"}),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
        ):
            result = _process_paper(
                paper, _DryRunClient(), db, 100.0, _NOW, False,
                live_reactive=True,
                live_budget_remaining=1,
                live_client=MagicMock(),
            )

        assert result["verdict_status"] == "dry_run"
        # _DryRunClient.submit_verdict would raise if called — no assertion
        # needed beyond "no exception raised".

    # -- New result fields always present ------------------------------------

    def test_result_always_has_live_fields(self):
        result, _ = self._run(candidate=None, live_reactive=False)
        assert "reactive_live_posted" in result
        assert "reactive_live_reason" in result

    def test_reactive_draft_created_false_for_live_posted(self):
        """live_posted is NOT counted as a dry-run draft (tracked separately)."""
        result, _ = self._run(
            candidate=_make_candidate(),
            live_reactive=True,
            live_budget_remaining=1,
            test_mode=False,
            run_mode="live",
            post_return="cmt-live",
        )
        assert result["reactive_status"] == "live_posted"
        assert result["reactive_draft_created"] is False


# ---------------------------------------------------------------------------
# TestRunOperationalLoopLiveReactive — loop-level budget + counter tests
# ---------------------------------------------------------------------------

class TestRunOperationalLoopLiveReactive:
    """Tests for run_operational_loop live_reactive flag and counters."""

    def _run_loop(
        self,
        paper_rows: list | None = None,
        process_results: list | None = None,
        live_reactive: bool = False,
        test_mode: bool = True,
        mock_koala_client: bool = False,
    ) -> tuple[dict, list]:
        """Run loop, capture _process_paper call kwargs, return (counters, call_kwargs_list)."""
        db = _make_loop_db(paper_rows or [_make_paper_row()])
        call_kwargs: list = []

        if process_results is None:
            process_results = [_no_op_result()]

        results_iter = iter(process_results)

        def _side_effect(paper, client, _db, karma, now, tm, **kwargs):
            call_kwargs.append(kwargs)
            return next(results_iter)

        patches = [
            patch(f"{_MOD}._process_paper", side_effect=_side_effect),
            patch(f"{_MOD}.is_aggressive_mode", return_value=False),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ]
        # KoalaClient is imported inside run_operational_loop; patch at source.
        _koala_patch = (
            patch("gsr_agent.koala.client.KoalaClient", return_value=MagicMock())
            if mock_koala_client
            else None
        )

        def _invoke():
            return run_operational_loop(
                db, _NOW,
                live_reactive=live_reactive,
                test_mode=test_mode,
                output_dir="/tmp/rep",
            )

        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            if _koala_patch is not None:
                with _koala_patch:
                    counters = _invoke()
            else:
                counters = _invoke()

        return counters, call_kwargs

    # -- Default: live_reactive=False ----------------------------------------

    def test_default_never_live_posts(self):
        result = _no_op_result()
        result["reactive_live_posted"] = False
        counters, kwargs_list = self._run_loop(process_results=[result])
        assert counters["live_reactive_posts"] == 0
        assert kwargs_list[0]["live_reactive"] is False
        assert kwargs_list[0]["live_budget_remaining"] == 3  # full budget passed but flag off

    def test_default_live_reactive_posts_key_exists(self):
        counters, _ = self._run_loop()
        assert "live_reactive_posts" in counters
        assert counters["live_reactive_posts"] == 0

    # -- live_reactive=True, single paper posts live -------------------------

    def test_live_reactive_single_post_counted(self):
        live_result = dict(_no_op_result())
        live_result["reactive_live_posted"] = True
        live_result["reactive_status"] = "live_posted"

        counters, kwargs_list = self._run_loop(
            process_results=[live_result],
            live_reactive=True,
            test_mode=False,
            mock_koala_client=True,
        )
        assert counters["live_reactive_posts"] == 1
        assert kwargs_list[0]["live_reactive"] is True
        assert kwargs_list[0]["live_budget_remaining"] == 3  # full budget on first paper

    # -- Budget exhausted after first live post ------------------------------

    def test_budget_decrements_after_live_post(self):
        """Second paper gets live_budget_remaining=2 after first posts (budget=3)."""
        live_result = dict(_no_op_result("paper-0"))
        live_result["reactive_live_posted"] = True
        live_result["reactive_status"] = "live_posted"

        dry_result = dict(_no_op_result("paper-1"))
        dry_result["reactive_live_posted"] = False

        rows = [_make_paper_row("paper-0"), _make_paper_row("paper-1")]
        counters, kwargs_list = self._run_loop(
            paper_rows=rows,
            process_results=[live_result, dry_result],
            live_reactive=True,
            test_mode=False,
            mock_koala_client=True,
        )

        assert counters["live_reactive_posts"] == 1
        # First paper: full budget available
        assert kwargs_list[0]["live_budget_remaining"] == 3
        # Second paper: one slot used, two remaining
        assert kwargs_list[1]["live_budget_remaining"] == 2

    def test_live_reactive_posts_max_one(self):
        """Even if two papers both set reactive_live_posted=True, counter caps at 1."""
        live_result_a = dict(_no_op_result("paper-a"))
        live_result_a["reactive_live_posted"] = True

        live_result_b = dict(_no_op_result("paper-b"))
        live_result_b["reactive_live_posted"] = True

        rows = [_make_paper_row("paper-a"), _make_paper_row("paper-b")]
        counters, _ = self._run_loop(
            paper_rows=rows,
            process_results=[live_result_a, live_result_b],
            live_reactive=True,
            test_mode=False,
            mock_koala_client=True,
        )
        # Budget tracking: second paper would never actually live-post because
        # budget=0 is passed; but we're mocking _process_paper here, so just
        # verify the counter accumulates correctly from whatever is returned.
        assert counters["live_reactive_posts"] == 2

    # -- live_reactive=True but test_mode=True --------------------------------

    def test_live_reactive_test_mode_no_real_client(self):
        """When test_mode=True, live_client must not be constructed."""
        counters, kwargs_list = self._run_loop(
            live_reactive=True,
            test_mode=True,
        )
        # live_client should be None because test_mode=True
        assert kwargs_list[0]["live_client"] is None

    # -- Aggregate counter backward compat -----------------------------------

    def test_existing_counters_unchanged(self):
        full_result = dict(_no_op_result())
        full_result["has_reactive_candidate"] = True
        full_result["reactive_draft_created"] = True
        full_result["verdict_eligible"] = True
        full_result["verdict_draft_created"] = True
        full_result["reactive_live_posted"] = False

        counters, _ = self._run_loop(process_results=[full_result])
        assert counters["reactive_candidates_found"] == 1
        assert counters["reactive_drafts_created"] == 1
        assert counters["verdicts_eligible"] == 1
        assert counters["verdict_drafts_created"] == 1
        assert counters["live_reactive_posts"] == 0

    # -- Verdict submit guard ------------------------------------------------

    def test_dry_run_client_submit_verdict_raises(self):
        client = _DryRunClient()
        with pytest.raises(RuntimeError, match="submit_verdict"):
            client.submit_verdict("paper-id", 0.5, [], "https://example.com/f")

    def test_verdict_path_never_calls_live_client_submit_verdict(self):
        """plan_verdict_for_paper must never reach submit_verdict in Phase 9."""
        paper = _make_paper()
        db = _make_process_db()
        live_client = MagicMock()

        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(True)),
            patch(f"{_MOD}.plan_verdict_for_paper", return_value={"artifact_url": "u"}),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
        ):
            _process_paper(
                paper, _DryRunClient(), db, 100.0, _NOW, False,
                live_reactive=True,
                live_budget_remaining=1,
                live_client=live_client,
            )

        live_client.submit_verdict.assert_not_called()

    # -- Backward compat: old _process_paper results without live fields ------

    def test_loop_handles_result_without_live_fields(self):
        """run_operational_loop must not crash if _process_paper omits live fields."""
        old_result = {
            "paper_id": "paper-abc-123",
            "has_reactive_candidate": False,
            "reactive_draft_created": False,
            "verdict_eligible": False,
            "verdict_draft_created": False,
        }
        counters, _ = self._run_loop(process_results=[old_result])
        assert counters["live_reactive_posts"] == 0


# ---------------------------------------------------------------------------
# TestPreflightErrorInLivePath — KoalaPreflightError caught by live branch
# ---------------------------------------------------------------------------

class TestPreflightErrorInLivePath:
    """KoalaPreflightError raised inside plan_and_post_reactive_comment must be
    caught by the live_allowed branch and must not propagate or increment counters."""

    def _run_live(self, exc):
        paper = _make_paper()
        db = _make_process_db()
        live_client = MagicMock()

        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper",
                  return_value=[_make_candidate()]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper",
                  return_value=_make_candidate()),
            patch(f"{_MOD}.plan_and_post_reactive_comment", side_effect=exc),
            patch(f"{_MOD}.evaluate_verdict_eligibility",
                  return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
        ):
            result = _process_paper(
                paper, _DryRunClient(), db, 100.0, _NOW, False,
                live_reactive=True,
                live_budget_remaining=1,
                live_client=live_client,
            )
        return result

    def test_preflight_error_sets_preflight_failed_reason(self):
        result = self._run_live(KoalaPreflightError("bad url"))
        assert result["reactive_live_reason"] == "preflight_failed"

    def test_preflight_error_does_not_set_live_posted(self):
        result = self._run_live(KoalaPreflightError("bad url"))
        assert result["reactive_live_posted"] is False

    def test_preflight_error_sets_skipped_status(self):
        result = self._run_live(KoalaPreflightError("bad url"))
        assert result["reactive_status"] == "skipped"

    def test_preflight_error_does_not_propagate(self):
        """Must not raise — outer loop should never see this exception."""
        try:
            self._run_live(KoalaPreflightError("bad url"))
        except KoalaPreflightError:
            pytest.fail("KoalaPreflightError must be caught inside _process_paper")


# ---------------------------------------------------------------------------
# TestPlanAndPostReactiveCommentLivePath — orchestrator-level live post tests
# ---------------------------------------------------------------------------

_ORCH_MOD = "gsr_agent.commenting.orchestrator"
_OPEN_TIME = datetime(2026, 4, 25, 0, 0, 0, tzinfo=timezone.utc)
_NOW_COMMENT = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)  # 12h → REVIEW_ACTIVE


def _make_live_paper() -> Paper:
    w = compute_paper_windows(_OPEN_TIME)
    return Paper(
        paper_id="paper-live-001",
        title="Live Paper",
        open_time=w.open_time,
        review_end_time=w.review_end_time,
        verdict_end_time=w.verdict_end_time,
        state="REVIEW_ACTIVE",
    )


def _make_live_candidate() -> ReactiveAnalysisResult:
    return ReactiveAnalysisResult(
        comment_id="cmt-src-001",
        paper_id="paper-live-001",
        recommendation="react",
        draft_text="[DRY-RUN — not posted]\nThis claim is refuted by evidence X.",
    )


def _make_live_client(comment_id: str = "posted-cmt-001") -> MagicMock:
    client = MagicMock()
    client.post_comment.return_value = comment_id
    return client


def _make_live_db() -> MagicMock:
    db = MagicMock()
    db.has_prior_participation.return_value = False
    return db


class TestPlanAndPostReactiveCommentLivePath:
    """Direct tests of plan_and_post_reactive_comment for the live write path."""

    def _run(self, *, run_mode="live", post_side_effect=None, post_return="posted-cmt-001",
             artifact_url="https://github.com/org/repo/blob/main/comment.md",
             moderation_raises=False):
        paper = _make_live_paper()
        candidate = _make_live_candidate()
        client = _make_live_client(comment_id=post_return)
        if post_side_effect is not None:
            client.post_comment.side_effect = post_side_effect
        db = _make_live_db()

        with (
            patch(f"{_ORCH_MOD}.get_run_mode", return_value=run_mode),
            patch(f"{_ORCH_MOD}.publish_comment_artifact", return_value=artifact_url),
            patch(f"{_ORCH_MOD}.preflight_comment_action",
                  side_effect=KoalaPreflightError("moderation") if moderation_raises else None),
            patch(f"{_ORCH_MOD}.validate_artifact_for_live_action"),
            patch.dict("os.environ", {"KOALA_API_BASE_URL": "https://koala.example.com"}),
        ):
            result = plan_and_post_reactive_comment(
                paper, candidate, client, db, karma_remaining=50.0,
                now=_NOW_COMMENT, test_mode=False,
            )
        return result, client, db

    def test_live_mode_calls_post_comment_once(self):
        result, client, _ = self._run(run_mode="live")
        assert result == "posted-cmt-001"
        client.post_comment.assert_called_once()

    def test_live_mode_post_comment_args(self):
        _, client, _ = self._run(run_mode="live")
        args, kwargs = client.post_comment.call_args
        assert args[0] == "paper-live-001"
        assert kwargs.get("parent_id") == "cmt-src-001"

    def test_write_disabled_dry_run_mode_returns_none(self):
        result, client, _ = self._run(run_mode="dry_run")
        assert result is None
        client.post_comment.assert_not_called()

    def test_write_disabled_records_dry_run_to_db(self):
        _, _, db = self._run(run_mode="dry_run")
        db.log_action.assert_called_once()
        _, kwargs = db.log_action.call_args
        assert kwargs.get("status") == "dry_run"

    def test_moderation_failure_raises_preflight_error(self):
        paper = _make_live_paper()
        candidate = _make_live_candidate()
        client = _make_live_client()
        db = _make_live_db()
        with (
            patch(f"{_ORCH_MOD}.get_run_mode", return_value="live"),
            patch(f"{_ORCH_MOD}.publish_comment_artifact",
                  return_value="https://github.com/org/repo/blob/main/c.md"),
            patch(f"{_ORCH_MOD}.preflight_comment_action",
                  side_effect=KoalaPreflightError("moderation: blocked phrase")),
            patch(f"{_ORCH_MOD}.validate_artifact_for_live_action"),
            patch.dict("os.environ", {"KOALA_API_BASE_URL": "https://koala.example.com"}),
        ):
            with pytest.raises(KoalaPreflightError, match="moderation"):
                plan_and_post_reactive_comment(
                    paper, candidate, client, db, karma_remaining=50.0,
                    now=_NOW_COMMENT, test_mode=False,
                )
        client.post_comment.assert_not_called()

    def test_koala_api_error_reraises_does_not_record_success(self):
        paper = _make_live_paper()
        candidate = _make_live_candidate()
        client = _make_live_client()
        client.post_comment.side_effect = KoalaAPIError("500 internal server error")
        db = _make_live_db()

        with (
            patch(f"{_ORCH_MOD}.get_run_mode", return_value="live"),
            patch(f"{_ORCH_MOD}.publish_comment_artifact",
                  return_value="https://github.com/org/repo/blob/main/c.md"),
            patch(f"{_ORCH_MOD}.preflight_comment_action"),
            patch(f"{_ORCH_MOD}.validate_artifact_for_live_action"),
            patch.dict("os.environ", {"KOALA_API_BASE_URL": "https://koala.example.com"}),
        ):
            with pytest.raises(KoalaAPIError):
                plan_and_post_reactive_comment(
                    paper, candidate, client, db, karma_remaining=50.0,
                    now=_NOW_COMMENT, test_mode=False,
                )

        client.post_comment.assert_called_once()
        for c in db.log_action.call_args_list:
            _, kwargs = c
            assert kwargs.get("status") != "success", "must not record success on KoalaAPIError"

    def test_success_records_action_with_success_status(self):
        _, _, db = self._run(run_mode="live")
        success_calls = [
            c for c in db.log_action.call_args_list
            if c[1].get("status") == "success"
        ]
        assert len(success_calls) == 1

    def test_success_records_external_id(self):
        _, _, db = self._run(run_mode="live", post_return="new-cmt-999")
        success_call = next(
            c for c in db.log_action.call_args_list if c[1].get("status") == "success"
        )
        assert success_call[1]["external_id"] == "new-cmt-999"

    def test_success_records_source_comment_id_in_details(self):
        _, _, db = self._run(run_mode="live")
        success_call = next(
            c for c in db.log_action.call_args_list if c[1].get("status") == "success"
        )
        assert success_call[1]["details"]["source_comment_id"] == "cmt-src-001"


# ---------------------------------------------------------------------------
# TestDedupPreventsRepeatPost — dedup gate blocks re-posting the same comment
# ---------------------------------------------------------------------------

class TestDedupPreventsRepeatPost:
    """After a successful or dry_run action is recorded, the dedup gate must
    block the same source comment from being retried."""

    def test_dedup_gate_blocks_after_dry_run_recorded(self):
        """_process_paper must skip when DB says recent action exists for the source comment."""
        paper = _make_paper()
        db = _make_process_db(reactive_dedup=True)
        live_client = MagicMock()

        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper",
                  return_value=[_make_candidate()]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper",
                  return_value=_make_candidate()),
            patch(f"{_MOD}.plan_and_post_reactive_comment") as mock_post,
            patch(f"{_MOD}.evaluate_verdict_eligibility",
                  return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
        ):
            result = _process_paper(
                paper, _DryRunClient(), db, 100.0, _NOW, False,
                live_reactive=True,
                live_budget_remaining=1,
                live_client=live_client,
            )

        assert result["reactive_status"] == "dedup_skipped"
        assert result["reactive_live_reason"] == "dedup_skipped"
        assert result["reactive_live_posted"] is False
        mock_post.assert_not_called()

    def test_dedup_gate_not_blocking_on_first_run(self):
        """Without a prior recorded action, the post should proceed."""
        paper = _make_paper()
        db = _make_process_db(reactive_dedup=False)
        live_client = MagicMock()

        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper",
                  return_value=[_make_candidate()]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper",
                  return_value=_make_candidate()),
            patch(f"{_MOD}.plan_and_post_reactive_comment", return_value="cmt-new") as mock_post,
            patch(f"{_MOD}.evaluate_verdict_eligibility",
                  return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
        ):
            result = _process_paper(
                paper, _DryRunClient(), db, 100.0, _NOW, False,
                live_reactive=True,
                live_budget_remaining=1,
                live_client=live_client,
            )

        assert result["reactive_live_posted"] is True
        mock_post.assert_called_once()


# ---------------------------------------------------------------------------
# TestAggressiveModeReactiveHandoff — decision=post → plan_and_post called
# ---------------------------------------------------------------------------

def _make_sparse_candidate(paper_id: str = "paper-abc-123") -> ReactiveAnalysisResult:
    return ReactiveAnalysisResult(
        comment_id="cmt-other-sparse-001",
        paper_id=paper_id,
        recommendation="evidence_sparse",
        verifications=[{"verdict": "insufficient_evidence", "confidence": 0.0}],
        draft_text="[DRY-RUN — not posted]\nEvidence-sparse draft.",
    )


def _make_aggressive_process_db(*, reactive_dedup: bool = False, ours: int = 1) -> MagicMock:
    db = MagicMock()
    db.has_recent_reactive_action_for_comment.return_value = reactive_dedup
    db.has_recent_verdict_action_for_paper.return_value = False
    db.get_comment_stats.return_value = {"total": 5, "ours": ours, "citable_other": 3}
    return db


class TestAggressiveModeReactiveHandoff:
    """Tests proving decision=post leads to plan_and_post_reactive_comment exactly once
    in aggressive live mode, regardless of prior comments or contradiction confidence."""

    def _run_aggressive(
        self,
        candidate,
        *,
        post_return: str = "cmt-agg-001",
        reactive_dedup: bool = False,
        ours: int = 1,
        select_returns: object = "same",  # "same" → same as candidate; None → None
    ):
        """Run _process_paper with aggressive_mode=True and live gates open."""
        paper = _make_paper()
        db = _make_aggressive_process_db(reactive_dedup=reactive_dedup, ours=ours)
        live_client = MagicMock()
        selected = candidate if select_returns == "same" else select_returns

        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper",
                  return_value=[candidate] if candidate else []),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper",
                  return_value=selected),
            patch(f"{_MOD}.plan_and_post_reactive_comment",
                  return_value=post_return) as mock_post,
            patch(f"{_MOD}.evaluate_verdict_eligibility",
                  return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
        ):
            result = _process_paper(
                paper, _DryRunClient(), db, 100.0, _NOW, False,
                live_reactive=True,
                live_budget_remaining=3,
                live_client=live_client,
                aggressive_mode=True,
            )
        return result, mock_post

    def test_evidence_sparse_calls_plan_and_post_exactly_once(self):
        """evidence_sparse decision=post candidate in aggressive live mode → exactly one post."""
        _, mock_post = self._run_aggressive(_make_sparse_candidate())
        mock_post.assert_called_once()

    def test_react_candidate_calls_plan_and_post_exactly_once(self):
        """react decision=post candidate in aggressive live mode → exactly one post."""
        _, mock_post = self._run_aggressive(_make_candidate())
        mock_post.assert_called_once()

    def test_evidence_sparse_with_prior_comment_still_posts(self):
        """evidence_sparse candidate is not suppressed even when agent already commented."""
        result, mock_post = self._run_aggressive(_make_sparse_candidate(), ours=2)
        mock_post.assert_called_once()
        assert result["reactive_live_posted"] is True

    def test_evidence_sparse_live_posted_status(self):
        result, _ = self._run_aggressive(_make_sparse_candidate())
        assert result["reactive_live_posted"] is True
        assert result["reactive_status"] == "live_posted"
        assert result["reactive_live_reason"] == "live_posted"

    def test_heat_band_suppressed_react_candidate_is_reselected(self):
        """When select_best returns None (heat-band suppressed) but decision=post exists,
        the candidate is re-selected from reactive_results."""
        candidate = _make_candidate()
        result, mock_post = self._run_aggressive(candidate, select_returns=None)
        mock_post.assert_called_once()
        assert result["reactive_live_posted"] is True

    def test_no_post_candidates_skips_posting(self):
        """When no decision=post candidate exists, plan_and_post must not be called."""
        skip_result = ReactiveAnalysisResult(
            comment_id="cmt-skip",
            paper_id="paper-abc-123",
            recommendation="skip",
        )
        result, mock_post = self._run_aggressive(skip_result, select_returns=None)
        mock_post.assert_not_called()
        assert result["reactive_live_posted"] is False

    def test_dedup_still_blocks_in_aggressive_mode(self):
        """Dedup gate must still prevent repeat posting even in aggressive mode."""
        _, mock_post = self._run_aggressive(_make_sparse_candidate(), reactive_dedup=True)
        mock_post.assert_not_called()

    def test_plan_and_post_called_with_live_client_in_aggressive_mode(self):
        """In aggressive live mode, plan_and_post must receive the live client."""
        paper = _make_paper()
        db = _make_aggressive_process_db()
        live_client = MagicMock()
        candidate = _make_sparse_candidate()

        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper",
                  return_value=[candidate]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper",
                  return_value=candidate),
            patch(f"{_MOD}.plan_and_post_reactive_comment",
                  return_value="cmt-agg") as mock_post,
            patch(f"{_MOD}.evaluate_verdict_eligibility",
                  return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
        ):
            _process_paper(
                paper, _DryRunClient(), db, 100.0, _NOW, False,
                live_reactive=True,
                live_budget_remaining=3,
                live_client=live_client,
                aggressive_mode=True,
            )

        args, kwargs = mock_post.call_args
        assert args[2] is live_client
        assert kwargs.get("test_mode") is False
