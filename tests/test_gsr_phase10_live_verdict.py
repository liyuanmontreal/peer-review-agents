"""Tests for Phase 10: Controlled Live Verdict submission."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from gsr_agent.commenting.reactive_analysis import ReactiveAnalysisResult
from gsr_agent.orchestration.operational_loop import (
    _DryRunClient,
    _paper_from_row,
    _process_paper,
    _submit_live_verdict,
    run_operational_loop,
)
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
        "open_time": "2026-04-24T00:00:00+00:00",       # 60h before _NOW → verdict window open
        "review_end_time": "2026-04-26T00:00:00+00:00",
        "verdict_end_time": "2026-04-27T00:00:00+00:00",
        "state": "VERDICT_ACTIVE",
        "pdf_url": "https://example.com/paper.pdf",
        "local_pdf_path": None,
    }


def _make_paper(paper_id: str = "paper-abc-123"):
    return _paper_from_row(_make_paper_row(paper_id))


def _make_eligibility(eligible: bool = True) -> VerdictEligibilityResult:
    return VerdictEligibilityResult(
        eligible=eligible,
        reason_code="eligible" if eligible else "no_react_signal",
        heat_band="goldilocks",
        distinct_citable_other_agents=4,
        strongest_contradiction_confidence=0.85,
        selected_candidates=[],
    )


def _make_process_db(
    *, reactive_dedup: bool = False, verdict_dedup: bool = False
) -> MagicMock:
    db = MagicMock()
    db.has_recent_reactive_action_for_comment.return_value = reactive_dedup
    db.has_recent_verdict_action_for_paper.return_value = verdict_dedup
    return db


def _make_loop_db(paper_rows: list | None = None) -> MagicMock:
    db = MagicMock()
    db.get_papers.return_value = paper_rows if paper_rows is not None else [_make_paper_row()]
    return db


def _base_result(paper_id: str = "paper-abc-123") -> dict:
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
        "verdict_live_submitted": False,
        "verdict_live_reason": "no_eligible_verdict",
        "has_reactive_candidate": False,
        "reactive_draft_created": False,
        "verdict_eligible": False,
        "verdict_draft_created": False,
    }


def _run_loop(
    paper_rows, process_results, *,
    live_verdict=False, live_reactive=False,
    test_mode=True, paper_ids=None,
    mock_koala_client=False,
) -> dict:
    db = _make_loop_db(paper_rows)
    results_iter = iter(process_results)

    def _side(paper, *a, **kw):
        return next(results_iter)

    _koala_patch = (
        patch("gsr_agent.koala.client.KoalaClient", return_value=MagicMock())
        if mock_koala_client else None
    )

    def _invoke():
        return run_operational_loop(
            db, _NOW,
            paper_ids=paper_ids,
            live_reactive=live_reactive,
            live_verdict=live_verdict,
            test_mode=test_mode,
            output_dir="/tmp/rep",
        )

    with (
        patch(f"{_MOD}._process_paper", side_effect=_side),
        patch(f"{_MOD}.build_run_summary", return_value=[]),
        patch(f"{_MOD}.write_run_summary_markdown"),
        patch(f"{_MOD}.write_run_summary_jsonl"),
    ):
        if _koala_patch:
            with _koala_patch:
                return _invoke()
        return _invoke()


# ---------------------------------------------------------------------------
# TestSubmitLiveVerdict — unit-tests for the _submit_live_verdict helper
# ---------------------------------------------------------------------------

class TestSubmitLiveVerdictHelper:
    """_submit_live_verdict is the thin wrapper that calls the real API."""

    def _make_db_with_citations(self, n: int = 3) -> MagicMock:
        db = MagicMock()
        db.get_citable_other_comments_for_paper.return_value = [
            {"comment_id": f"cmt-{i}", "author_agent_id": f"agent-{i}", "created_at": f"2026-0{i+1}"}
            for i in range(n)
        ]
        return db

    def test_returns_true_on_success(self):
        paper = _make_paper()
        db = self._make_db_with_citations(3)
        live_client = MagicMock()

        with (
            patch(f"{_MOD}.select_distinct_other_agent_citations",
                  return_value=[{"comment_id": "cmt-x", "author_agent_id": "a"}] * 3),
            patch(f"{_MOD}.build_verdict_draft_for_paper", return_value="draft body"),
            patch(f"{_MOD}.publish_verdict_artifact",
                  return_value="https://github.com/real-repo/blob/main/v.md"),
            patch(f"{_MOD}.validate_artifact_for_live_action"),
        ):
            submitted, reason = _submit_live_verdict(
                paper, db, [], _NOW, live_client, _make_eligibility(), score=7.5
            )

        assert submitted is True
        assert reason == "live_submitted"
        live_client.submit_verdict.assert_called_once()

    def test_calls_submit_verdict_with_correct_args(self):
        paper = _make_paper()
        db = MagicMock()
        live_client = MagicMock()
        citations = [{"comment_id": f"cmt-{i}", "author_agent_id": f"ag-{i}"} for i in range(3)]
        url = "https://github.com/real-repo/blob/main/v.md"

        with (
            patch(f"{_MOD}.select_distinct_other_agent_citations", return_value=citations),
            patch(f"{_MOD}.build_verdict_draft_for_paper", return_value="draft"),
            patch(f"{_MOD}.publish_verdict_artifact", return_value=url),
            patch(f"{_MOD}.validate_artifact_for_live_action"),
        ):
            _submit_live_verdict(paper, db, [], _NOW, live_client, _make_eligibility(), score=7.5)

        live_client.submit_verdict.assert_called_once_with(
            paper.paper_id,
            score=7.5,
            cited_comment_ids=["cmt-0", "cmt-1", "cmt-2"],
            github_file_url=url,
        )

    def test_returns_false_when_no_citations(self):
        paper = _make_paper()
        db = MagicMock()
        live_client = MagicMock()

        with patch(f"{_MOD}.select_distinct_other_agent_citations", return_value=[]):
            submitted, reason = _submit_live_verdict(
                paper, db, [], _NOW, live_client, _make_eligibility(), score=7.5
            )

        assert submitted is False
        live_client.submit_verdict.assert_not_called()

    def test_returns_false_when_draft_is_none(self):
        paper = _make_paper()
        db = MagicMock()
        live_client = MagicMock()

        with (
            patch(f"{_MOD}.select_distinct_other_agent_citations",
                  return_value=[{"comment_id": "c", "author_agent_id": "a"}] * 3),
            patch(f"{_MOD}.build_verdict_draft_for_paper", return_value=None),
        ):
            submitted, reason = _submit_live_verdict(
                paper, db, [], _NOW, live_client, _make_eligibility(), score=7.5
            )

        assert submitted is False
        live_client.submit_verdict.assert_not_called()

    def test_publishes_with_test_mode_false(self):
        paper = _make_paper()
        db = MagicMock()
        live_client = MagicMock()

        with (
            patch(f"{_MOD}.select_distinct_other_agent_citations",
                  return_value=[{"comment_id": "c", "author_agent_id": "a"}] * 3),
            patch(f"{_MOD}.build_verdict_draft_for_paper", return_value="draft"),
            patch(f"{_MOD}.publish_verdict_artifact",
                  return_value="https://real.github.com/v.md") as mock_pub,
            patch(f"{_MOD}.validate_artifact_for_live_action"),
        ):
            _submit_live_verdict(paper, db, [], _NOW, live_client, _make_eligibility(), score=7.5)

        _, kwargs = mock_pub.call_args
        assert kwargs.get("test_mode") is False

    def test_logs_action_to_db_on_success(self):
        paper = _make_paper()
        db = MagicMock()
        live_client = MagicMock()

        with (
            patch(f"{_MOD}.select_distinct_other_agent_citations",
                  return_value=[{"comment_id": "c", "author_agent_id": "a"}] * 3),
            patch(f"{_MOD}.build_verdict_draft_for_paper", return_value="draft"),
            patch(f"{_MOD}.publish_verdict_artifact", return_value="https://real/v.md"),
            patch(f"{_MOD}.validate_artifact_for_live_action"),
        ):
            _submit_live_verdict(paper, db, [], _NOW, live_client, _make_eligibility(), score=7.5)

        db.log_action.assert_called_once()
        _, kwargs = db.log_action.call_args
        assert kwargs["status"] == "success"
        assert kwargs["action_type"] == "verdict_submission"

    def test_validation_called_before_submit(self):
        paper = _make_paper()
        db = MagicMock()
        live_client = MagicMock()
        call_order = []

        def _validate(url):
            call_order.append("validate")

        def _submit(*args, **kwargs):
            call_order.append("submit")

        live_client.submit_verdict.side_effect = _submit

        with (
            patch(f"{_MOD}.select_distinct_other_agent_citations",
                  return_value=[{"comment_id": "c", "author_agent_id": "a"}] * 3),
            patch(f"{_MOD}.build_verdict_draft_for_paper", return_value="draft"),
            patch(f"{_MOD}.publish_verdict_artifact", return_value="https://real/v.md"),
            patch(f"{_MOD}.validate_artifact_for_live_action", side_effect=_validate),
        ):
            _submit_live_verdict(paper, db, [], _NOW, live_client, _make_eligibility(), score=7.5)

        assert call_order == ["validate", "submit"]


# ---------------------------------------------------------------------------
# TestProcessPaperLiveVerdict — _process_paper verdict live gating
# ---------------------------------------------------------------------------

class TestProcessPaperLiveVerdict:
    """Unit-tests for the Phase 10 live gating inside _process_paper."""

    def _run(
        self,
        *,
        eligible=True,
        verdict_dedup=False,
        verdict_artifact_url="https://test.github.com/v.md",
        live_verdict=False,
        verdict_live_budget_remaining=0,
        test_mode=True,
        run_mode="dry_run",
        allowlisted=False,
        submit_return=(True, "live_submitted"),
    ) -> tuple[dict, MagicMock]:
        paper = _make_paper()
        db = _make_process_db(verdict_dedup=verdict_dedup)
        live_client = MagicMock() if live_verdict else None

        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility",
                  return_value=_make_eligibility(eligible)),
            patch(f"{_MOD}.plan_verdict_for_paper",
                  return_value={"artifact_url": verdict_artifact_url,
                                "status": "dry_run" if verdict_artifact_url else "skipped"}),
            patch(f"{_MOD}._submit_live_verdict", return_value=submit_return) as mock_submit,
            patch(f"{_MOD}.get_run_mode", return_value=run_mode),
        ):
            result = _process_paper(
                paper, _DryRunClient(), db, 100.0, _NOW, test_mode,
                live_verdict=live_verdict,
                verdict_live_budget_remaining=verdict_live_budget_remaining,
                allowlisted=allowlisted,
                live_client=live_client,
            )
        return result, mock_submit

    # -- Default: live_verdict=False -----------------------------------------

    def test_default_verdict_status_dry_run(self):
        result, mock_submit = self._run(eligible=True)
        assert result["verdict_status"] == "dry_run"
        mock_submit.assert_not_called()

    def test_default_verdict_live_reason_is_live_disabled(self):
        result, _ = self._run(eligible=True, live_verdict=False)
        assert result["verdict_live_reason"] == "live_disabled"
        assert result["verdict_live_submitted"] is False

    # -- test_mode=True blocks live ------------------------------------------

    def test_test_mode_blocks_live_verdict(self):
        result, mock_submit = self._run(
            eligible=True, live_verdict=True, verdict_live_budget_remaining=1,
            test_mode=True, run_mode="live", allowlisted=True,
        )
        assert result["verdict_live_submitted"] is False
        assert result["verdict_live_reason"] == "live_gate_failed"
        mock_submit.assert_not_called()

    # -- KOALA_RUN_MODE != live blocks ---------------------------------------

    def test_run_mode_not_live_blocks_verdict(self):
        result, mock_submit = self._run(
            eligible=True, live_verdict=True, verdict_live_budget_remaining=1,
            test_mode=False, run_mode="dry_run", allowlisted=True,
        )
        assert result["verdict_live_submitted"] is False
        assert result["verdict_live_reason"] == "live_gate_failed"
        mock_submit.assert_not_called()

    # -- Allowlist gate ------------------------------------------------------

    def test_no_allowlist_blocks_verdict(self):
        result, mock_submit = self._run(
            eligible=True, live_verdict=True, verdict_live_budget_remaining=1,
            test_mode=False, run_mode="live", allowlisted=False,
        )
        assert result["verdict_live_submitted"] is False
        assert result["verdict_live_reason"] == "allowlist_required"
        mock_submit.assert_not_called()

    # -- Budget exhausted ----------------------------------------------------

    def test_budget_exhausted_blocks_verdict(self):
        result, mock_submit = self._run(
            eligible=True, live_verdict=True, verdict_live_budget_remaining=0,
            test_mode=False, run_mode="live", allowlisted=True,
        )
        assert result["verdict_live_submitted"] is False
        assert result["verdict_live_reason"] == "live_budget_exhausted"
        mock_submit.assert_not_called()

    def test_budget_exhausted_keeps_dry_run_artifact(self):
        result, _ = self._run(
            eligible=True, live_verdict=True, verdict_live_budget_remaining=0,
            test_mode=False, run_mode="live", allowlisted=True,
        )
        assert result["verdict_status"] == "dry_run"
        assert result["verdict_artifact"] is not None

    # -- Ineligible verdict --------------------------------------------------

    def test_ineligible_verdict_never_submits(self):
        result, mock_submit = self._run(
            eligible=False, live_verdict=True, verdict_live_budget_remaining=1,
            test_mode=False, run_mode="live", allowlisted=True,
        )
        assert result["verdict_status"] == "ineligible"
        assert result["verdict_live_submitted"] is False
        assert result["verdict_live_reason"] == "no_eligible_verdict"
        mock_submit.assert_not_called()

    # -- Dedup blocks --------------------------------------------------------

    def test_dedup_blocks_live_verdict(self):
        result, mock_submit = self._run(
            eligible=True, verdict_dedup=True, live_verdict=True,
            verdict_live_budget_remaining=1, test_mode=False, run_mode="live",
            allowlisted=True,
        )
        assert result["verdict_live_submitted"] is False
        assert result["verdict_live_reason"] == "dedup_skipped"
        assert result["verdict_status"] == "dedup_skipped"
        mock_submit.assert_not_called()

    # -- Citation gate failure (no dry-run artifact) -------------------------

    def test_no_artifact_blocks_live_verdict(self):
        result, mock_submit = self._run(
            eligible=True, verdict_artifact_url=None, live_verdict=True,
            verdict_live_budget_remaining=1, test_mode=False, run_mode="live",
            allowlisted=True,
        )
        assert result["verdict_live_submitted"] is False
        assert result["verdict_live_reason"] == "no_eligible_verdict"
        mock_submit.assert_not_called()

    # -- All gates pass: live submit -----------------------------------------

    def test_all_gates_pass_submits_live(self):
        result, mock_submit = self._run(
            eligible=True, live_verdict=True, verdict_live_budget_remaining=1,
            test_mode=False, run_mode="live", allowlisted=True,
            submit_return=(True, "live_submitted"),
        )
        assert result["verdict_live_submitted"] is True
        assert result["verdict_live_reason"] == "live_submitted"
        assert result["verdict_status"] == "live_submitted"
        mock_submit.assert_called_once()

    def test_live_submit_passes_correct_args_to_helper(self):
        paper = _make_paper()
        db = _make_process_db()
        live_client = MagicMock()

        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility",
                  return_value=_make_eligibility(True)) as mock_elig,
            patch(f"{_MOD}.plan_verdict_for_paper",
                  return_value={"artifact_url": "https://t/v.md", "status": "dry_run"}),
            patch(f"{_MOD}._submit_live_verdict", return_value=(True, "live_submitted")) as mock_submit,
            patch(f"{_MOD}.get_run_mode", return_value="live"),
        ):
            _process_paper(
                paper, _DryRunClient(), db, 100.0, _NOW, False,
                live_verdict=True, verdict_live_budget_remaining=1,
                allowlisted=True, live_client=live_client,
            )

        args, kwargs = mock_submit.call_args
        assert args[0] is paper
        assert args[4] is live_client

    def test_submit_helper_returns_false_sets_gate_failed(self):
        result, mock_submit = self._run(
            eligible=True, live_verdict=True, verdict_live_budget_remaining=1,
            test_mode=False, run_mode="live", allowlisted=True,
            submit_return=(False, "no_draft"),
        )
        assert result["verdict_live_submitted"] is False
        assert result["verdict_live_reason"] == "no_eligible_verdict"
        # Dry-run artifact stays (from plan_verdict_for_paper which ran before)
        assert result["verdict_status"] == "dry_run"

    # -- New result fields always present ------------------------------------

    def test_result_always_has_verdict_live_fields(self):
        result, _ = self._run(eligible=False)
        assert "verdict_live_submitted" in result
        assert "verdict_live_reason" in result

    # -- verdict_draft_created is False for live_submitted -------------------

    def test_verdict_draft_created_false_for_live_submitted(self):
        result, _ = self._run(
            eligible=True, live_verdict=True, verdict_live_budget_remaining=1,
            test_mode=False, run_mode="live", allowlisted=True,
        )
        assert result["verdict_status"] == "live_submitted"
        assert result["verdict_draft_created"] is False

    # -- Reactive live behavior unchanged ------------------------------------

    def test_reactive_live_unaffected_by_live_verdict_flag(self):
        paper = _make_paper()
        db = _make_process_db()
        candidate = ReactiveAnalysisResult(
            comment_id="cmt-other", paper_id="paper-abc-123",
            recommendation="react", draft_text="[DRY-RUN — not posted]\n\nbody",
        )

        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[candidate]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=candidate),
            patch(f"{_MOD}.plan_and_post_reactive_comment", return_value="cmt-dry") as mock_post,
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value="dry_run"),
        ):
            result = _process_paper(
                paper, _DryRunClient(), db, 100.0, _NOW, True,
                live_verdict=True,
                verdict_live_budget_remaining=1,
                allowlisted=True,
            )

        assert result["reactive_status"] == "dry_run"
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs.get("test_mode") is True


# ---------------------------------------------------------------------------
# TestRunOperationalLoopLiveVerdict — loop-level budget + counter tests
# ---------------------------------------------------------------------------

class TestRunOperationalLoopLiveVerdict:

    def test_live_verdict_submissions_counter_present(self):
        counters = _run_loop([_make_paper_row()], [_base_result()])
        assert "live_verdict_submissions" in counters
        assert counters["live_verdict_submissions"] == 0

    def test_default_never_live_submits(self):
        counters = _run_loop([_make_paper_row()], [_base_result()])
        assert counters["live_verdict_submissions"] == 0

    def test_live_verdict_post_counted(self):
        result = dict(_base_result())
        result["verdict_live_submitted"] = True
        result["verdict_status"] = "live_submitted"
        counters = _run_loop(
            [_make_paper_row()], [result],
            live_verdict=True, test_mode=False,
            paper_ids=["paper-abc-123"], mock_koala_client=True,
        )
        assert counters["live_verdict_submissions"] == 1

    def test_budget_exhausted_after_first_verdict(self):
        """Second paper gets verdict_live_budget_remaining=0."""
        live_a = dict(_base_result("p-a"))
        live_a["verdict_live_submitted"] = True
        live_a["verdict_status"] = "live_submitted"

        dry_b = dict(_base_result("p-b"))
        dry_b["verdict_live_submitted"] = False

        rows = [_make_paper_row("p-a"), _make_paper_row("p-b")]
        call_kwargs = []

        def _side(paper, *a, **kw):
            call_kwargs.append(kw)
            return live_a if paper.paper_id == "p-a" else dry_b

        db = _make_loop_db(rows)
        with (
            patch(f"{_MOD}._process_paper", side_effect=_side),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
            patch("gsr_agent.koala.client.KoalaClient", return_value=MagicMock()),
        ):
            run_operational_loop(
                db, _NOW,
                paper_ids=["p-a", "p-b"],
                live_verdict=True, test_mode=False, output_dir="/tmp/rep",
            )

        assert call_kwargs[0]["verdict_live_budget_remaining"] == 1
        assert call_kwargs[1]["verdict_live_budget_remaining"] == 0

    def test_max_one_live_verdict_per_run(self):
        """Both papers return submitted=True; counter still caps budget after first."""
        result_a = dict(_base_result("p-a"))
        result_a["verdict_live_submitted"] = True

        result_b = dict(_base_result("p-b"))
        result_b["verdict_live_submitted"] = True

        rows = [_make_paper_row("p-a"), _make_paper_row("p-b")]
        # Mocked _process_paper returns what we give; actual budget enforcement
        # is verified by checking verdict_live_budget_remaining on second call.
        call_kwargs = []

        def _side(paper, *a, **kw):
            call_kwargs.append(kw)
            return result_a if paper.paper_id == "p-a" else result_b

        db = _make_loop_db(rows)
        with (
            patch(f"{_MOD}._process_paper", side_effect=_side),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
            patch("gsr_agent.koala.client.KoalaClient", return_value=MagicMock()),
        ):
            counters = run_operational_loop(
                db, _NOW,
                paper_ids=["p-a", "p-b"],
                live_verdict=True, test_mode=False, output_dir="/tmp/rep",
            )

        # Budget passed to second paper is 0 after first was submitted
        assert call_kwargs[1]["verdict_live_budget_remaining"] == 0

    def test_allowlisted_true_when_paper_ids_provided(self):
        call_kwargs = []

        def _side(paper, *a, **kw):
            call_kwargs.append(kw)
            return _base_result()

        db = _make_loop_db([_make_paper_row()])
        with (
            patch(f"{_MOD}._process_paper", side_effect=_side),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            run_operational_loop(
                db, _NOW,
                paper_ids=["paper-abc-123"],
                live_verdict=True, test_mode=True, output_dir="/tmp/rep",
            )

        assert call_kwargs[0]["allowlisted"] is True

    def test_allowlisted_false_when_no_paper_ids(self):
        call_kwargs = []

        def _side(paper, *a, **kw):
            call_kwargs.append(kw)
            return _base_result()

        db = _make_loop_db([_make_paper_row()])
        with (
            patch(f"{_MOD}._process_paper", side_effect=_side),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            run_operational_loop(
                db, _NOW,
                paper_ids=None,
                live_verdict=True, test_mode=True, output_dir="/tmp/rep",
            )

        assert call_kwargs[0]["allowlisted"] is False

    def test_live_client_constructed_when_live_verdict_and_not_test_mode(self):
        """live_client is constructed when live_verdict=True and test_mode=False."""
        call_kwargs = []

        def _side(paper, *a, **kw):
            call_kwargs.append(kw)
            return _base_result()

        db = _make_loop_db([_make_paper_row()])
        mock_client = MagicMock()

        with (
            patch(f"{_MOD}._process_paper", side_effect=_side),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
            patch("gsr_agent.koala.client.KoalaClient", return_value=mock_client),
        ):
            run_operational_loop(
                db, _NOW,
                paper_ids=["paper-abc-123"],
                live_verdict=True, test_mode=False, output_dir="/tmp/rep",
            )

        assert call_kwargs[0]["live_client"] is mock_client

    def test_live_client_none_when_test_mode_true(self):
        call_kwargs = []

        def _side(paper, *a, **kw):
            call_kwargs.append(kw)
            return _base_result()

        db = _make_loop_db([_make_paper_row()])
        with (
            patch(f"{_MOD}._process_paper", side_effect=_side),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            run_operational_loop(
                db, _NOW,
                paper_ids=["paper-abc-123"],
                live_verdict=True, test_mode=True, output_dir="/tmp/rep",
            )

        assert call_kwargs[0]["live_client"] is None

    # -- Dry-run client still guards submit_verdict --------------------------

    def test_dryrun_client_submit_verdict_still_raises(self):
        client = _DryRunClient()
        with pytest.raises(RuntimeError, match="submit_verdict"):
            client.submit_verdict("paper-id", 0.5, [], "https://example.com/f")

    # -- Backward compat: old results without verdict live fields -------------

    def test_loop_handles_result_without_verdict_live_fields(self):
        old_result = {
            "paper_id": "paper-abc-123",
            "has_reactive_candidate": False,
            "reactive_draft_created": False,
            "verdict_eligible": False,
            "verdict_draft_created": False,
        }
        counters = _run_loop([_make_paper_row()], [old_result])
        assert counters["live_verdict_submissions"] == 0

    # -- Reactive live unaffected --------------------------------------------

    def test_reactive_live_posts_counter_unchanged_by_verdict_flag(self):
        reactive_result = dict(_base_result())
        reactive_result["reactive_live_posted"] = True
        reactive_result["reactive_status"] = "live_posted"
        reactive_result["reactive_live_reason"] = "live_posted"

        counters = _run_loop(
            [_make_paper_row()], [reactive_result],
            live_reactive=True, live_verdict=True,
            test_mode=False, paper_ids=["paper-abc-123"],
            mock_koala_client=True,
        )
        assert counters["live_reactive_posts"] == 1
        assert counters["live_verdict_submissions"] == 0
