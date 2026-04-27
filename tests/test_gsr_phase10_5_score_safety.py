"""Tests for Phase 10.5: Verdict Score Safety Patch."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from gsr_agent.orchestration.operational_loop import (
    _validate_verdict_score,
    _submit_live_verdict,
    _process_paper,
    _DryRunClient,
    _paper_from_row,
    run_operational_loop,
)
from gsr_agent.rules.verdict_assembly import VerdictEligibilityResult

_MOD = "gsr_agent.orchestration.operational_loop"
_NOW = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
_OPEN = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)


def _make_paper_row(paper_id: str = "paper-score-safety") -> dict:
    return {
        "paper_id": paper_id,
        "title": "Score Safety Test",
        "open_time": "2026-04-01T00:00:00+00:00",
        "review_end_time": "2026-04-15T00:00:00+00:00",
        "verdict_end_time": "2026-04-22T00:00:00+00:00",
        "state": "REVIEW_ACTIVE",
        "pdf_url": "https://example.com/paper.pdf",
        "local_pdf_path": None,
    }


def _make_paper(paper_id: str = "paper-score-safety"):
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


# ---------------------------------------------------------------------------
# TestValidateVerdictScore — unit tests for the score validation helper
# ---------------------------------------------------------------------------

class TestValidateVerdictScore:

    def test_none_returns_missing(self):
        assert _validate_verdict_score(None) == "missing_verdict_score"

    def test_zero_is_valid(self):
        assert _validate_verdict_score(0.0) is None

    def test_ten_is_valid(self):
        assert _validate_verdict_score(10.0) is None

    def test_midrange_float_is_valid(self):
        assert _validate_verdict_score(5.5) is None

    def test_int_is_valid(self):
        assert _validate_verdict_score(7) is None

    def test_negative_returns_invalid(self):
        assert _validate_verdict_score(-0.1) == "invalid_verdict_score"

    def test_above_ten_returns_invalid(self):
        assert _validate_verdict_score(10.1) == "invalid_verdict_score"

    def test_bool_true_returns_invalid(self):
        assert _validate_verdict_score(True) == "invalid_verdict_score"

    def test_bool_false_returns_invalid(self):
        assert _validate_verdict_score(False) == "invalid_verdict_score"

    def test_string_returns_invalid(self):
        assert _validate_verdict_score("7.5") == "invalid_verdict_score"

    def test_nan_returns_invalid(self):
        assert _validate_verdict_score(float("nan")) == "invalid_verdict_score"

    def test_inf_returns_invalid(self):
        assert _validate_verdict_score(float("inf")) == "invalid_verdict_score"


# ---------------------------------------------------------------------------
# TestSubmitLiveVerdictScoreSafety — score gate blocks before any live calls
# ---------------------------------------------------------------------------

class TestSubmitLiveVerdictScoreSafety:

    @patch(f"{_MOD}.select_distinct_other_agent_citations")
    def test_missing_score_skips_all_live_calls(self, mock_cit):
        paper = _make_paper()
        db = MagicMock()
        live_client = MagicMock()

        ok, reason = _submit_live_verdict(
            paper, db, [], _NOW, live_client, _make_eligibility(), score=None
        )

        assert ok is False
        assert reason == "missing_verdict_score"
        mock_cit.assert_not_called()
        live_client.submit_verdict.assert_not_called()

    @patch(f"{_MOD}.select_distinct_other_agent_citations")
    def test_invalid_score_skips_all_live_calls(self, mock_cit):
        paper = _make_paper()
        db = MagicMock()
        live_client = MagicMock()

        ok, reason = _submit_live_verdict(
            paper, db, [], _NOW, live_client, _make_eligibility(), score=15.0
        )

        assert ok is False
        assert reason == "invalid_verdict_score"
        mock_cit.assert_not_called()
        live_client.submit_verdict.assert_not_called()

    @patch(f"{_MOD}.select_distinct_other_agent_citations")
    def test_bool_score_skips_all_live_calls(self, mock_cit):
        paper = _make_paper()
        db = MagicMock()
        live_client = MagicMock()

        ok, reason = _submit_live_verdict(
            paper, db, [], _NOW, live_client, _make_eligibility(), score=True
        )

        assert ok is False
        assert reason == "invalid_verdict_score"
        mock_cit.assert_not_called()
        live_client.submit_verdict.assert_not_called()

    @patch(f"{_MOD}.validate_artifact_for_live_action")
    @patch(f"{_MOD}.publish_verdict_artifact", return_value="https://github.com/repo/v.md")
    @patch(f"{_MOD}.build_verdict_draft_for_paper", return_value="verdict body")
    @patch(
        f"{_MOD}.select_distinct_other_agent_citations",
        return_value=[{"comment_id": "c1"}, {"comment_id": "c2"}],
    )
    def test_valid_score_calls_submit_verdict(self, mock_cit, mock_draft, mock_pub, mock_val):
        paper = _make_paper()
        db = MagicMock()
        live_client = MagicMock()

        ok, reason = _submit_live_verdict(
            paper, db, [], _NOW, live_client, _make_eligibility(), score=7.5
        )

        assert ok is True
        assert reason == "live_submitted"
        live_client.submit_verdict.assert_called_once()

    @patch(f"{_MOD}.validate_artifact_for_live_action")
    @patch(f"{_MOD}.publish_verdict_artifact", return_value="https://github.com/repo/v.md")
    @patch(f"{_MOD}.build_verdict_draft_for_paper", return_value="verdict body")
    @patch(
        f"{_MOD}.select_distinct_other_agent_citations",
        return_value=[{"comment_id": "c1"}],
    )
    def test_valid_score_passed_to_publish_and_submit(
        self, mock_cit, mock_draft, mock_pub, mock_val
    ):
        paper = _make_paper()
        db = MagicMock()
        live_client = MagicMock()
        score = 8.0

        _submit_live_verdict(paper, db, [], _NOW, live_client, _make_eligibility(), score=score)

        _, pub_kwargs = mock_pub.call_args
        assert pub_kwargs["score"] == score
        _, sub_kwargs = live_client.submit_verdict.call_args
        assert sub_kwargs["score"] == score


# ---------------------------------------------------------------------------
# TestProcessPaperScoreSafety — score flows through _process_paper
# ---------------------------------------------------------------------------

class TestProcessPaperScoreSafety:
    """Score kwarg is forwarded to _submit_live_verdict and the reason propagates back."""

    def _run_eligible(self, submit_return, score=None):
        paper = _make_paper()
        db = MagicMock()
        db.has_recent_reactive_action_for_comment.return_value = False
        db.has_recent_verdict_action_for_paper.return_value = False
        live_client = MagicMock()

        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(True)),
            patch(f"{_MOD}.plan_verdict_for_paper",
                  return_value={"artifact_url": "https://gh/dry.md", "status": "dry_run"}),
            patch(f"{_MOD}._submit_live_verdict", return_value=submit_return) as mock_submit,
            patch(f"{_MOD}.get_run_mode", return_value="live"),
        ):
            result = _process_paper(
                paper, _DryRunClient(), db, 100.0, _NOW, False,
                live_verdict=True,
                verdict_live_budget_remaining=1,
                live_client=live_client,
                allowlisted=True,
                score=score,
            )
        return result, mock_submit

    def test_missing_score_reason_propagates(self):
        result, _ = self._run_eligible((False, "missing_verdict_score"), score=None)
        assert result["verdict_live_reason"] == "missing_verdict_score"
        assert result["verdict_live_submitted"] is False

    def test_invalid_score_reason_propagates(self):
        result, _ = self._run_eligible((False, "invalid_verdict_score"), score=15.0)
        assert result["verdict_live_reason"] == "invalid_verdict_score"
        assert result["verdict_live_submitted"] is False

    def test_valid_score_submits_and_sets_reason(self):
        result, mock_submit = self._run_eligible((True, "live_submitted"), score=7.5)
        assert result["verdict_live_submitted"] is True
        assert result["verdict_live_reason"] == "live_submitted"
        _, call_kwargs = mock_submit.call_args
        assert call_kwargs.get("score") == 7.5

    def test_non_score_submit_failure_maps_to_no_eligible_verdict(self):
        result, _ = self._run_eligible((False, "no_draft"), score=7.5)
        assert result["verdict_live_reason"] == "no_eligible_verdict"
        assert result["verdict_live_submitted"] is False


# ---------------------------------------------------------------------------
# TestRunLoopScoreCounters — counters track score validation outcomes
# ---------------------------------------------------------------------------

class TestRunLoopScoreCounters:

    def _make_loop_db(self, paper_id: str = "paper-score-safety") -> MagicMock:
        db = MagicMock()
        db.get_papers.return_value = [_make_paper_row(paper_id)]
        return db

    def _base_result(self, paper_id: str = "paper-score-safety", verdict_live_reason=None) -> dict:
        return {
            "paper_id": paper_id,
            "reactive_status": "none",
            "reactive_reason": None,
            "reactive_artifact": None,
            "reactive_live_posted": False,
            "reactive_live_reason": "no_candidate",
            "verdict_status": "dry_run",
            "verdict_reason": None,
            "verdict_artifact": "https://gh/dry.md",
            "verdict_live_submitted": False,
            "verdict_live_reason": verdict_live_reason,
            "has_reactive_candidate": False,
            "reactive_draft_created": False,
            "verdict_eligible": True,
            "verdict_draft_created": True,
        }

    def _run(self, process_result: dict, *, paper_ids=None, live_verdict=False,
             test_mode=True) -> dict:
        db = self._make_loop_db()

        def _side(paper, *a, **kw):
            return process_result

        with (
            patch(f"{_MOD}._process_paper", side_effect=_side),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            return run_operational_loop(
                db, _NOW,
                paper_ids=paper_ids,
                live_verdict=live_verdict,
                test_mode=test_mode,
                output_dir="/tmp/rep",
            )

    def test_missing_score_counter_increments(self):
        result = self._base_result(verdict_live_reason="missing_verdict_score")
        counters = self._run(result)
        assert counters["verdict_live_missing_score"] == 1
        assert counters["verdict_live_invalid_score"] == 0
        assert counters["live_verdict_submissions"] == 0

    def test_invalid_score_counter_increments(self):
        result = self._base_result(verdict_live_reason="invalid_verdict_score")
        counters = self._run(result)
        assert counters["verdict_live_invalid_score"] == 1
        assert counters["verdict_live_missing_score"] == 0

    def test_both_counters_zero_when_no_score_error(self):
        result = self._base_result(verdict_live_reason="live_disabled")
        counters = self._run(result)
        assert counters["verdict_live_missing_score"] == 0
        assert counters["verdict_live_invalid_score"] == 0

    def test_counters_present_in_return_dict(self):
        db = self._make_loop_db()
        db.get_papers.return_value = []
        with (
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            counters = run_operational_loop(db, _NOW)
        assert "verdict_live_missing_score" in counters
        assert "verdict_live_invalid_score" in counters

    def test_old_result_without_verdict_live_reason_leaves_counters_zero(self):
        old_result = {
            "paper_id": "paper-score-safety",
            "has_reactive_candidate": False,
            "reactive_draft_created": False,
            "verdict_eligible": False,
            "verdict_draft_created": False,
        }
        counters = self._run(old_result)
        assert counters["verdict_live_missing_score"] == 0
        assert counters["verdict_live_invalid_score"] == 0


# ---------------------------------------------------------------------------
# TestCLIPrint — CLI stdout includes verdict score observability line
# ---------------------------------------------------------------------------

class TestCLIPrint:

    def test_cli_print_includes_score_counters(self, capsys, tmp_path):
        from gsr_agent.orchestration.operational_loop import main

        mock_counters = {
            "papers_seen": 2,
            "papers_processed": 2,
            "reactive_drafts_created": 0,
            "live_reactive_posts": 0,
            "verdict_drafts_created": 1,
            "live_verdict_submissions": 0,
            "verdict_live_missing_score": 1,
            "verdict_live_invalid_score": 0,
            "reactive_live_eligible": 0,
            "reactive_dedup_skipped": 0,
            "live_budget_exhausted": 0,
            "reactive_live_gate_failed": 0,
            "errors_count": 0,
            "summary_path": str(tmp_path / "run_summary.md"),
        }
        mock_db = MagicMock()
        mock_db.close = MagicMock()

        with (
            patch("gsr_agent.storage.db.KoalaDB", return_value=mock_db),
            patch(f"{_MOD}.run_operational_loop", return_value=mock_counters),
            patch("sys.argv", ["loop"]),
        ):
            main()

        out = capsys.readouterr().out
        assert "missing_score=1" in out
        assert "invalid_score=0" in out
        assert "live_verdict_submissions=0" in out
