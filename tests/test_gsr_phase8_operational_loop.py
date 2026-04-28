"""Tests for Phase 8: Operational Loop (dry-run orchestrator)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from gsr_agent.commenting.reactive_analysis import ReactiveAnalysisResult
from gsr_agent.orchestration.operational_loop import (
    _DryRunClient,
    _paper_from_row,
    _process_paper,
    run_operational_loop,
)
from gsr_agent.rules.verdict_assembly import VerdictEligibilityResult
from gsr_agent.strategy.opportunity_manager import CANDIDATE_BUDGET, PaperOpportunity

_MOD = "gsr_agent.orchestration.operational_loop"
_NOW = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper_row(paper_id: str = "paper-abc-123") -> dict:
    return {
        "paper_id": paper_id,
        "title": "Test Paper",
        "open_time": "2026-04-25T12:00:00+00:00",   # 24h before _NOW → BUILD_WINDOW → FOLLOWUP
        "review_end_time": "2026-04-28T00:00:00+00:00",
        "verdict_end_time": "2026-04-29T00:00:00+00:00",
        "state": "REVIEW_ACTIVE",
        "pdf_url": "https://example.com/paper.pdf",
        "local_pdf_path": None,
    }


def _make_eligibility(eligible: bool = True) -> VerdictEligibilityResult:
    return VerdictEligibilityResult(
        eligible=eligible,
        reason_code="eligible" if eligible else "no_react_signal",
        heat_band="goldilocks",
        distinct_citable_other_agents=3,
        strongest_contradiction_confidence=0.80,
        selected_candidates=[],
    )


def _make_reactive_result(recommendation: str = "react") -> ReactiveAnalysisResult:
    return ReactiveAnalysisResult(
        comment_id="cmt-other-001",
        paper_id="paper-abc-123",
        recommendation=recommendation,
        draft_text="[DRY-RUN — not posted]\n\nDraft reactive text here.",
    )


def _make_db(paper_rows: list | None = None) -> MagicMock:
    db = MagicMock()
    db.get_papers.return_value = paper_rows if paper_rows is not None else [_make_paper_row()]
    return db


def _no_op_result() -> dict:
    return {
        "paper_id": "paper-abc-123",
        "reactive_status": "none",
        "reactive_reason": None,
        "reactive_artifact": None,
        "verdict_status": "ineligible",
        "verdict_reason": None,
        "verdict_artifact": None,
        "has_reactive_candidate": False,
        "reactive_draft_created": False,
        "verdict_eligible": False,
        "verdict_draft_created": False,
    }


def _make_process_db(
    *,
    reactive_dedup: bool = False,
    verdict_dedup: bool = False,
) -> MagicMock:
    """MagicMock db with dedup methods returning False by default."""
    db = MagicMock()
    db.has_recent_reactive_action_for_comment.return_value = reactive_dedup
    db.has_recent_verdict_action_for_paper.return_value = verdict_dedup
    return db


# ---------------------------------------------------------------------------
# TestDryRunClient
# ---------------------------------------------------------------------------

class TestDryRunClient:
    def test_post_comment_returns_fake_id(self):
        client = _DryRunClient()
        result = client.post_comment("paper-abc-123", "body", "https://example.com/f")
        assert result == "dry-run-paper-ab"

    def test_post_comment_truncates_to_8_chars(self):
        client = _DryRunClient()
        result = client.post_comment("x" * 20, "body", "url")
        assert result == f"dry-run-{'x' * 8}"

    def test_post_comment_accepts_thread_and_parent(self):
        client = _DryRunClient()
        result = client.post_comment(
            "paper-abc", "body", "url", thread_id="t1", parent_id="p1"
        )
        assert result.startswith("dry-run-")

    def test_post_comment_short_paper_id(self):
        client = _DryRunClient()
        result = client.post_comment("ab", "body", "url")
        assert result == "dry-run-ab"

    def test_submit_verdict_raises(self):
        client = _DryRunClient()
        with pytest.raises(RuntimeError, match="submit_verdict"):
            client.submit_verdict("paper-id", 0.5)


# ---------------------------------------------------------------------------
# TestPaperFromRow
# ---------------------------------------------------------------------------

class TestPaperFromRow:
    def test_all_fields_parsed(self):
        row = _make_paper_row()
        paper = _paper_from_row(row)
        assert paper.paper_id == "paper-abc-123"
        assert paper.title == "Test Paper"
        assert paper.open_time == datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
        assert paper.review_end_time == datetime(2026, 4, 28, tzinfo=timezone.utc)
        assert paper.verdict_end_time == datetime(2026, 4, 29, tzinfo=timezone.utc)
        assert paper.state == "REVIEW_ACTIVE"
        assert paper.pdf_url == "https://example.com/paper.pdf"
        assert paper.local_pdf_path is None

    def test_empty_title_preserved(self):
        row = {**_make_paper_row(), "title": ""}
        paper = _paper_from_row(row)
        assert paper.title == ""

    def test_missing_title_defaults_empty(self):
        row = _make_paper_row()
        del row["title"]
        paper = _paper_from_row(row)
        assert paper.title == ""

    def test_missing_state_defaults_to_review_active(self):
        row = _make_paper_row()
        del row["state"]
        paper = _paper_from_row(row)
        assert paper.state == "REVIEW_ACTIVE"

    def test_invalid_open_time_falls_back_gracefully(self):
        row = {**_make_paper_row(), "open_time": "not-a-date"}
        paper = _paper_from_row(row)
        assert paper.open_time is not None

    def test_local_pdf_path_preserved_when_set(self):
        row = {**_make_paper_row(), "local_pdf_path": "/tmp/paper.pdf"}
        paper = _paper_from_row(row)
        assert paper.local_pdf_path == "/tmp/paper.pdf"

    def test_different_paper_ids(self):
        for pid in ["p1", "paper-xyz-999", "a" * 40]:
            paper = _paper_from_row({**_make_paper_row(), "paper_id": pid})
            assert paper.paper_id == pid


# ---------------------------------------------------------------------------
# TestProcessPaper
# ---------------------------------------------------------------------------

class TestProcessPaper:
    def _make_paper(self):
        return _paper_from_row({
            **_make_paper_row(),
            "open_time": "2026-04-25T00:00:00+00:00",       # 36h before _NOW → comment window open
            "review_end_time": "2026-04-27T00:00:00+00:00",
            "verdict_end_time": "2026-04-28T00:00:00+00:00",
            "state": "REVIEW_ACTIVE",
        })

    def test_no_candidate_reactive_not_created(self):
        paper = self._make_paper()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.plan_and_post_reactive_comment") as mock_post,
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_process_db(), 100.0, _NOW, test_mode=True
            )

        assert result["has_reactive_candidate"] is False
        assert result["reactive_draft_created"] is False
        mock_post.assert_not_called()

    def test_candidate_found_comment_posted(self):
        paper = self._make_paper()
        candidate = _make_reactive_result()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[candidate]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=candidate),
            patch(f"{_MOD}.plan_and_post_reactive_comment", return_value="cmt-new-001"),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_process_db(), 100.0, _NOW, test_mode=True
            )

        assert result["has_reactive_candidate"] is True
        assert result["reactive_draft_created"] is True

    def test_candidate_found_but_post_returns_none(self):
        paper = self._make_paper()
        candidate = _make_reactive_result()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[candidate]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=candidate),
            patch(f"{_MOD}.plan_and_post_reactive_comment", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_process_db(), 100.0, _NOW, test_mode=True
            )

        assert result["has_reactive_candidate"] is True
        assert result["reactive_draft_created"] is False

    def test_verdict_eligible_draft_created(self):
        paper = self._make_paper()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(True)),
            patch(
                f"{_MOD}.plan_verdict_for_paper",
                return_value={"artifact_url": "url/v.md", "status": "dry_run"},
            ),
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_process_db(), 100.0, _NOW, test_mode=True
            )

        assert result["verdict_eligible"] is True
        assert result["verdict_draft_created"] is True

    def test_verdict_ineligible_plan_not_called(self):
        paper = self._make_paper()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.plan_verdict_for_paper") as mock_verdict,
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_process_db(), 100.0, _NOW, test_mode=True
            )

        assert result["verdict_eligible"] is False
        assert result["verdict_draft_created"] is False
        mock_verdict.assert_not_called()

    def test_verdict_eligible_but_no_artifact_url(self):
        paper = self._make_paper()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(True)),
            patch(f"{_MOD}.plan_verdict_for_paper", return_value={"status": "skipped"}),
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_process_db(), 100.0, _NOW, test_mode=True
            )

        assert result["verdict_eligible"] is True
        assert result["verdict_draft_created"] is False

    def test_test_mode_forwarded_to_reactive_comment(self):
        paper = self._make_paper()
        candidate = _make_reactive_result()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[candidate]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=candidate),
            patch(f"{_MOD}.plan_and_post_reactive_comment", return_value="id") as mock_post,
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
        ):
            _process_paper(
                paper, _DryRunClient(), _make_process_db(), 50.0, _NOW, test_mode=True
            )

        _, kwargs = mock_post.call_args
        assert kwargs.get("test_mode") is True

    def test_test_mode_forwarded_to_verdict(self):
        paper = self._make_paper()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(True)),
            patch(f"{_MOD}.plan_verdict_for_paper", return_value={"artifact_url": "u"}) as mock_v,
        ):
            _process_paper(
                paper, _DryRunClient(), _make_process_db(), 50.0, _NOW, test_mode=False
            )

        _, kwargs = mock_v.call_args
        assert kwargs.get("test_mode") is False

    def test_reactive_results_passed_to_eligibility(self):
        paper = self._make_paper()
        candidate = _make_reactive_result()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[candidate]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)) as mock_elig,
        ):
            _process_paper(
                paper, _DryRunClient(), _make_process_db(), 100.0, _NOW, test_mode=True
            )

        mock_elig.assert_called_once()
        args, _ = mock_elig.call_args_list[0]
        assert args[2] == [candidate]


# ---------------------------------------------------------------------------
# TestRunOperationalLoop
# ---------------------------------------------------------------------------

class TestRunOperationalLoop:
    """Integration-level tests for run_operational_loop (mocks _process_paper)."""

    def _run_loop(
        self,
        paper_rows: list | None = None,
        max_papers: int | None = None,
        paper_ids: list | None = None,
        process_side_effect=None,
        output_dir: str = "/tmp/test_reports",
    ) -> tuple[dict, MagicMock]:
        db = _make_db(paper_rows)
        with (
            patch(f"{_MOD}._process_paper", return_value=_no_op_result()) as mock_proc,
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            if process_side_effect is not None:
                mock_proc.side_effect = process_side_effect
            counters = run_operational_loop(
                db,
                _NOW,
                paper_ids=paper_ids,
                max_papers=max_papers,
                output_dir=output_dir,
            )
        return counters, db

    def test_empty_db_returns_zero_counters(self):
        counters, _ = self._run_loop(paper_rows=[])
        assert counters["papers_seen"] == 0
        assert counters["papers_processed"] == 0
        assert counters["errors"] == []
        assert counters["errors_count"] == 0
        assert counters["skipped"] == 0

    def test_single_paper_no_activity_counted_skipped(self):
        counters, _ = self._run_loop(paper_rows=[_make_paper_row()])
        assert counters["papers_seen"] == 1
        assert counters["papers_processed"] == 1
        assert counters["skipped"] == 1

    def test_single_paper_fully_processed(self):
        full_result = {
            "has_reactive_candidate": True,
            "reactive_draft_created": True,
            "verdict_eligible": True,
            "verdict_draft_created": True,
        }
        with (
            patch(f"{_MOD}._process_paper", return_value=full_result),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            counters = run_operational_loop(
                _make_db([_make_paper_row()]), _NOW, output_dir="/tmp/rep"
            )

        assert counters["papers_seen"] == 1
        assert counters["papers_processed"] == 1
        assert counters["reactive_candidates_found"] == 1
        assert counters["reactive_drafts_created"] == 1
        assert counters["verdicts_eligible"] == 1
        assert counters["verdict_drafts_created"] == 1
        assert counters["skipped"] == 0
        assert counters["errors"] == []
        assert counters["errors_count"] == 0

    def test_max_papers_caps_processing(self):
        rows = [_make_paper_row(f"paper-{i}") for i in range(5)]
        counters, _ = self._run_loop(paper_rows=rows, max_papers=3)
        assert counters["papers_seen"] == 5
        assert counters["papers_processed"] == 3

    def test_paper_ids_passed_to_get_papers(self):
        ids = ["paper-x", "paper-y"]
        _, db = self._run_loop(paper_rows=[], paper_ids=ids)
        db.get_papers.assert_called_once_with(ids)

    def test_none_paper_ids_passed_to_get_papers(self):
        _, db = self._run_loop(paper_rows=[])
        db.get_papers.assert_called_once_with(None)

    def test_error_in_process_paper_is_isolated(self):
        rows = [_make_paper_row("paper-ok"), _make_paper_row("paper-bad")]

        def side_effect(paper, *args, **kwargs):
            if paper.paper_id == "paper-bad":
                raise RuntimeError("unexpected failure")
            return _no_op_result()

        counters, _ = self._run_loop(paper_rows=rows, process_side_effect=side_effect)
        assert counters["papers_processed"] == 1
        assert counters["errors_count"] == 1
        assert len(counters["errors"]) == 1

    def test_summary_path_set_in_counters(self):
        counters, _ = self._run_loop(paper_rows=[], output_dir="/tmp/my_reports")
        assert counters["summary_path"].endswith("run_summary.md")
        assert "my_reports" in counters["summary_path"]

    def test_build_run_summary_called_with_paper_ids(self):
        ids = ["paper-1"]
        db = _make_db([])
        with (
            patch(f"{_MOD}.build_run_summary", return_value=[]) as mock_brs,
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            run_operational_loop(db, _NOW, paper_ids=ids, output_dir="/tmp/rep")
        mock_brs.assert_called_once_with(db, _NOW, ids)

    def test_write_markdown_called(self):
        db = _make_db([])
        with (
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown") as mock_md,
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            run_operational_loop(db, _NOW, output_dir="/tmp/rep")
        mock_md.assert_called_once()

    def test_write_jsonl_called(self):
        db = _make_db([])
        with (
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl") as mock_jl,
        ):
            run_operational_loop(db, _NOW, output_dir="/tmp/rep")
        mock_jl.assert_called_once()

    def test_verdict_eligible_but_not_drafted_counted_correctly(self):
        process_result = {
            "has_reactive_candidate": False,
            "reactive_draft_created": False,
            "verdict_eligible": True,
            "verdict_draft_created": False,
        }
        with (
            patch(f"{_MOD}._process_paper", return_value=process_result),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            counters = run_operational_loop(
                _make_db([_make_paper_row()]), _NOW, output_dir="/tmp/rep"
            )

        assert counters["verdicts_eligible"] == 1
        assert counters["verdict_drafts_created"] == 0
        assert counters["skipped"] == 1

    def test_reactive_only_paper_not_counted_skipped(self):
        process_result = {
            "has_reactive_candidate": True,
            "reactive_draft_created": True,
            "verdict_eligible": False,
            "verdict_draft_created": False,
        }
        with (
            patch(f"{_MOD}._process_paper", return_value=process_result),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            counters = run_operational_loop(
                _make_db([_make_paper_row()]), _NOW, output_dir="/tmp/rep"
            )

        assert counters["reactive_drafts_created"] == 1
        assert counters["skipped"] == 0

    def test_multiple_papers_aggregate_counters(self):
        rows = [_make_paper_row(f"paper-{i}") for i in range(3)]
        results = [
            {"has_reactive_candidate": True, "reactive_draft_created": True, "verdict_eligible": True, "verdict_draft_created": True},
            {"has_reactive_candidate": True, "reactive_draft_created": True, "verdict_eligible": False, "verdict_draft_created": False},
            {"has_reactive_candidate": False, "reactive_draft_created": False, "verdict_eligible": False, "verdict_draft_created": False},
        ]
        call_idx = [0]

        def side_effect(*args, **kwargs):
            r = results[call_idx[0]]
            call_idx[0] += 1
            return r

        counters, _ = self._run_loop(paper_rows=rows, process_side_effect=side_effect)

        assert counters["papers_seen"] == 3
        assert counters["papers_processed"] == 3
        assert counters["reactive_candidates_found"] == 2
        assert counters["reactive_drafts_created"] == 2
        assert counters["verdicts_eligible"] == 1
        assert counters["verdict_drafts_created"] == 1
        assert counters["skipped"] == 1
        assert counters["errors"] == []
        assert counters["errors_count"] == 0

    def test_max_papers_none_processes_all(self):
        rows = [_make_paper_row(f"paper-{i}") for i in range(3)]
        counters, _ = self._run_loop(paper_rows=rows, max_papers=None)
        assert counters["papers_processed"] == 3

    def test_papers_seen_always_reflects_full_list(self):
        rows = [_make_paper_row(f"paper-{i}") for i in range(10)]
        counters, _ = self._run_loop(paper_rows=rows, max_papers=2)
        assert counters["papers_seen"] == 10
        assert counters["papers_processed"] == 2


# ---------------------------------------------------------------------------
# TestNewSeedWindowCandidateSelection
# ---------------------------------------------------------------------------

def _make_new_seed_row(paper_id: str = "paper-new-001", *, abstract: str = "") -> dict:
    """Paper where open_time is 2h in the future: Phase=NEW, Micro=SEED_WINDOW."""
    return {
        "paper_id": paper_id,
        "title": "New Seed Paper",
        "abstract": abstract,
        "open_time": (_NOW + timedelta(hours=2)).isoformat(),
        "review_end_time": (_NOW + timedelta(hours=50)).isoformat(),
        "verdict_end_time": (_NOW + timedelta(hours=74)).isoformat(),
        "state": "REVIEW_ACTIVE",
        "pdf_url": "https://example.com/paper.pdf",
        "local_pdf_path": None,
    }


def _make_review_active_seed_row(paper_id: str = "paper-ra-001") -> dict:
    """Paper opened 6h ago: Phase=REVIEW_ACTIVE, Micro=SEED_WINDOW."""
    return {
        "paper_id": paper_id,
        "title": "Review Active Seed Paper",
        "open_time": (_NOW - timedelta(hours=6)).isoformat(),
        "review_end_time": (_NOW + timedelta(hours=42)).isoformat(),
        "verdict_end_time": (_NOW + timedelta(hours=66)).isoformat(),
        "state": "REVIEW_ACTIVE",
        "pdf_url": "https://example.com/paper.pdf",
        "local_pdf_path": None,
    }


def _run_seed_loop(rows: list, total_comments: int) -> dict:
    db = MagicMock()
    db.get_papers.return_value = rows
    db.get_comment_stats.return_value = {"total": total_comments, "ours": 0, "citable_other": 0}
    db.has_prior_participation.return_value = False
    db.has_recent_reactive_action_for_comment.return_value = False
    db.has_recent_verdict_action_for_paper.return_value = False

    with (
        patch(f"{_MOD}._process_paper", return_value=_no_op_result()),
        patch(f"{_MOD}.build_run_summary", return_value=[]),
        patch(f"{_MOD}.write_run_summary_markdown"),
        patch(f"{_MOD}.write_run_summary_jsonl"),
    ):
        return run_operational_loop(db, _NOW, output_dir="/tmp/test_new_seed")


class TestNewSeedWindowCandidateSelection:
    def test_new_seed_window_5_comments_selected(self):
        """NEW+SEED_WINDOW + 5 total comments → selected seed candidate (processed > 0)."""
        counters = _run_seed_loop([_make_new_seed_row()], total_comments=5)
        assert counters["papers_processed"] == 1

    def test_new_seed_window_0_comments_skip_too_cold(self):
        """NEW+SEED_WINDOW + 0 total comments → skip_too_cold (processed == 0)."""
        counters = _run_seed_loop([_make_new_seed_row()], total_comments=0)
        assert counters["papers_processed"] == 0

    def test_new_seed_window_13_comments_saturated(self):
        """NEW+SEED_WINDOW + 13 total comments → saturated_comments skip (processed == 0)."""
        counters = _run_seed_loop([_make_new_seed_row()], total_comments=13)
        assert counters["papers_processed"] == 0

    def test_review_active_seed_window_selected(self):
        """REVIEW_ACTIVE+SEED_WINDOW + 5 total comments → selected (REVIEW_ACTIVE behavior intact)."""
        counters = _run_seed_loop([_make_review_active_seed_row()], total_comments=5)
        assert counters["papers_processed"] == 1

    def test_review_active_seed_window_0_comments_skip_too_cold(self):
        """REVIEW_ACTIVE+SEED_WINDOW + 0 total comments → skip_too_cold (processed == 0)."""
        counters = _run_seed_loop([_make_review_active_seed_row()], total_comments=0)
        assert counters["papers_processed"] == 0

    def test_review_active_seed_window_13_comments_saturated(self):
        """REVIEW_ACTIVE+SEED_WINDOW + 13 total comments → saturated_comments skip (processed == 0)."""
        counters = _run_seed_loop([_make_review_active_seed_row()], total_comments=13)
        assert counters["papers_processed"] == 0

    def test_candidate_budget_limits_but_processed_gt_zero(self):
        """Budget limits to CANDIDATE_BUDGET but processed > 0 for valid NEW+SEED_WINDOW papers."""
        rows = [_make_new_seed_row(f"paper-new-{i:03d}") for i in range(5)]
        counters = _run_seed_loop(rows, total_comments=5)
        assert 0 < counters["papers_processed"] <= CANDIDATE_BUDGET

    def test_candidate_budget_reaches_eight(self):
        """Candidate budget is 8: 10 valid SEED papers → exactly 8 processed."""
        rows = [_make_review_active_seed_row(f"paper-ra-{i:03d}") for i in range(10)]
        counters = _run_seed_loop(rows, total_comments=5)
        assert CANDIDATE_BUDGET == 8
        assert counters["papers_processed"] == CANDIDATE_BUDGET


# ---------------------------------------------------------------------------
# TestSeedCommentPath
# ---------------------------------------------------------------------------

def _make_seed_process_db(*, seed_dedup: bool = False) -> MagicMock:
    db = MagicMock()
    db.has_recent_reactive_action_for_comment.return_value = False
    db.has_recent_verdict_action_for_paper.return_value = False
    db.has_recent_seed_action_for_paper.return_value = seed_dedup
    db.has_prior_participation.return_value = False
    return db


class TestSeedCommentPath:
    def _make_seed_paper(self) -> "Paper":
        return _paper_from_row(_make_review_active_seed_row())

    def test_seed_paper_creates_draft_when_no_reactive_candidate(self):
        """SEED paper with no reactive candidate creates a seed draft."""
        paper = self._make_seed_paper()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.plan_and_post_seed_comment", return_value=("dry-run-seed-001", None)) as mock_seed,
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_seed_process_db(), 100.0, _NOW, test_mode=True
            )

        mock_seed.assert_called_once()
        assert result["seed_draft_created"] is True
        assert result["seed_live_posted"] is False
        assert result["reactive_draft_created"] is False

    def test_seed_paper_live_posts_when_live_reactive_enabled(self):
        """Live-reactive SEED paper posts at most one live seed comment."""
        paper = self._make_seed_paper()
        live_client = MagicMock()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
            patch(f"{_MOD}.plan_and_post_seed_comment", return_value=("live-seed-cmt-001", None)) as mock_seed,
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_seed_process_db(), 100.0, _NOW,
                test_mode=False,
                live_reactive=True,
                live_budget_remaining=1,
                live_client=live_client,
            )

        mock_seed.assert_called_once()
        assert result["seed_live_posted"] is True
        assert result["seed_draft_created"] is False

    def test_seed_dedup_blocks_post(self):
        """Recent seed action prevents seed draft/post."""
        paper = self._make_seed_paper()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.plan_and_post_seed_comment") as mock_seed,
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_seed_process_db(seed_dedup=True), 100.0, _NOW,
                test_mode=True,
            )

        mock_seed.assert_not_called()
        assert result["seed_draft_created"] is False
        assert result["seed_live_posted"] is False

    def test_reactive_candidate_takes_precedence_over_seed(self):
        """When a reactive candidate exists, seed path is not entered."""
        paper = self._make_seed_paper()
        candidate = ReactiveAnalysisResult(
            comment_id="cmt-other-001",
            paper_id=paper.paper_id,
            recommendation="react",
            draft_text="[DRY-RUN — not posted]\n\nReactive draft.",
        )
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[candidate]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=candidate),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.plan_and_post_reactive_comment", return_value="dry-run-react-001"),
            patch(f"{_MOD}.plan_and_post_seed_comment") as mock_seed,
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_seed_process_db(), 100.0, _NOW, test_mode=True
            )

        mock_seed.assert_not_called()
        assert result["reactive_draft_created"] is True
        assert result["seed_draft_created"] is False

    def test_counters_reflect_seed_draft(self):
        """seed_drafts_created increments for seed draft; live_reactive_posts for live seed post."""
        row = _make_review_active_seed_row()
        db = MagicMock()
        db.get_papers.return_value = [row]
        db.has_recent_reactive_action_for_comment.return_value = False
        db.has_recent_verdict_action_for_paper.return_value = False
        db.has_recent_seed_action_for_paper.return_value = False
        db.has_prior_participation.return_value = False
        db.get_comment_stats.return_value = {"total": 5, "ours": 0, "citable_other": 0}

        seed_result = {
            "paper_id": row["paper_id"],
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
            "seed_draft_created": True,
            "seed_live_posted": False,
        }
        with (
            patch(f"{_MOD}._process_paper", return_value=seed_result),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            counters = run_operational_loop(db, _NOW, output_dir="/tmp/test_seed_counters")

        assert counters["seed_drafts_created"] == 1
        assert counters["reactive_drafts_created"] == 0
        assert counters["live_reactive_posts"] == 0
        assert counters["skipped"] == 0


# ---------------------------------------------------------------------------
# TestNewSeedWindowSeedPath  (fix verification)
# ---------------------------------------------------------------------------

class TestNewSeedWindowSeedPath:
    """Verify that Phase=NEW + SEED_WINDOW papers enter and complete the seed path."""

    def _make_new_seed_paper(self) -> "Paper":
        """open_time 2h in future → Phase=NEW, Micro=SEED_WINDOW, state=REVIEW_ACTIVE."""
        return _paper_from_row(_make_new_seed_row())

    def _make_new_seed_paper_state_new(self) -> "Paper":
        """Same timing but state='NEW' (schema default) — comment_phase_ok would be False."""
        row = {**_make_new_seed_row(), "state": "NEW"}
        return _paper_from_row(row)

    def test_new_phase_enters_seed_path(self):
        """NEW+SEED_WINDOW paper (state=REVIEW_ACTIVE) creates a seed draft."""
        paper = self._make_new_seed_paper()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.plan_and_post_seed_comment", return_value=("dry-run-seed-new", None)) as mock_seed,
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_seed_process_db(), 100.0, _NOW, test_mode=True
            )
        mock_seed.assert_called_once()
        assert result["seed_draft_created"] is True
        assert result["seed_live_reason"] == "draft_created"

    def test_new_phase_state_new_enters_seed_path(self):
        """NEW+SEED_WINDOW paper with state='NEW' (seed_window_ok) still enters seed path."""
        paper = self._make_new_seed_paper_state_new()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.plan_and_post_seed_comment", return_value=("dry-run-seed-state-new", None)) as mock_seed,
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_seed_process_db(), 100.0, _NOW, test_mode=True
            )
        mock_seed.assert_called_once()
        assert result["seed_draft_created"] is True

    def test_seed_live_reason_populated_in_result(self):
        """seed_live_reason field is returned from _process_paper."""
        paper = self._make_new_seed_paper()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.plan_and_post_seed_comment", return_value=(None, "seed_plan_missing_abstract")),
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_seed_process_db(), 100.0, _NOW, test_mode=True
            )
        assert "seed_live_reason" in result
        assert result["seed_draft_created"] is False

    def test_live_posts_increments_for_seed(self):
        """live_reactive_posts counter increments when seed_live_posted is True."""
        row = _make_new_seed_row()
        db = MagicMock()
        db.get_papers.return_value = [row]
        db.has_recent_reactive_action_for_comment.return_value = False
        db.has_recent_verdict_action_for_paper.return_value = False
        db.has_recent_seed_action_for_paper.return_value = False
        db.has_prior_participation.return_value = False
        db.get_comment_stats.return_value = {"total": 5, "ours": 0, "citable_other": 0}

        seed_live_result = {
            "paper_id": row["paper_id"],
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
            "seed_draft_created": False,
            "seed_live_posted": True,
            "seed_live_reason": "live_posted",
        }
        with (
            patch(f"{_MOD}._process_paper", return_value=seed_live_result),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            counters = run_operational_loop(db, _NOW, output_dir="/tmp/test_seed_live_ctr")

        assert counters["live_reactive_posts"] == 1
        assert counters["seed_drafts_created"] == 0


# ---------------------------------------------------------------------------
# TestSeedLiveGate — granular gate reason codes
# ---------------------------------------------------------------------------

class TestSeedLiveGate:
    """seed_gate_* reason codes and live-post path for NEW+SEED_WINDOW papers."""

    def _make_new_seed_paper_state_new(self) -> "Paper":
        row = {**_make_new_seed_row(), "state": "NEW"}
        return _paper_from_row(row)

    def test_new_seed_window_live_posts_when_all_gates_pass(self):
        """NEW+SEED_WINDOW live seed candidate posts when all gates pass."""
        paper = self._make_new_seed_paper_state_new()
        live_client = MagicMock()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
            patch(f"{_MOD}.plan_and_post_seed_comment", return_value=("live-seed-new-001", None)),
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_seed_process_db(), 100.0, _NOW,
                test_mode=False, live_reactive=True, live_budget_remaining=1,
                live_client=live_client,
            )
        assert result["seed_live_posted"] is True
        assert result["seed_live_reason"] == "live_posted"

    def test_seed_gate_window_when_not_in_seed_window(self):
        """seed_gate_window when paper is in comment phase but not SEED_WINDOW."""
        # Paper opened 15h ago: comment_phase_ok=True, micro=BUILD_WINDOW, seed_window_ok=False.
        row = {**_make_review_active_seed_row(), "open_time": (_NOW - timedelta(hours=15)).isoformat()}
        paper = _paper_from_row(row)
        live_client = MagicMock()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
            patch(f"{_MOD}.classify_paper_opportunity", return_value=PaperOpportunity.SEED),
            patch(f"{_MOD}.plan_and_post_seed_comment", return_value=(None, "seed_plan_missing_abstract")),
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_seed_process_db(), 100.0, _NOW,
                test_mode=False, live_reactive=True, live_budget_remaining=1,
                live_client=live_client,
            )
        assert result["seed_live_reason"] == "seed_gate_window"
        assert result["seed_live_posted"] is False

    def test_seed_gate_no_client_when_live_client_absent(self):
        """seed_gate_no_client when live_client is None."""
        paper = self._make_new_seed_paper_state_new()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_seed_process_db(), 100.0, _NOW,
                test_mode=False, live_reactive=True, live_budget_remaining=1,
                live_client=None,
            )
        assert result["seed_live_reason"] == "seed_gate_no_client"
        assert result["seed_live_posted"] is False

    def test_seed_gate_budget_exhausted_blocks_live_post(self):
        """live_budget_remaining=0 blocks live seed post; dry-run path still runs."""
        paper = self._make_new_seed_paper_state_new()
        live_client = MagicMock()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
            patch(f"{_MOD}.plan_and_post_seed_comment", return_value=(None, "seed_plan_missing_abstract")),
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_seed_process_db(), 100.0, _NOW,
                test_mode=False, live_reactive=True, live_budget_remaining=0,
                live_client=live_client,
            )
        assert result["seed_live_posted"] is False
        assert result["seed_draft_created"] is False
        # seed_gate_budget is set before dry-run path; not overwritten when no draft created
        assert result["seed_live_reason"] == "seed_gate_budget"


# ---------------------------------------------------------------------------
# TestSeedPlannerIntegration — end-to-end seed planner with real function
# ---------------------------------------------------------------------------

class TestSeedPlannerIntegration:
    """Verify plan_and_post_seed_comment returns non-None for valid candidates
    and specific seed_plan_* reasons for skip paths (no mocking of the planner)."""

    def _make_db(self):
        db = MagicMock()
        db.has_prior_participation.return_value = False
        return db

    def _make_client(self):
        client = MagicMock()
        client.post_comment.return_value = "comment-integration-001"
        return client

    def test_new_seed_window_with_abstract_returns_non_none(self):
        """NEW+SEED_WINDOW paper with abstract returns non-None from seed planner."""
        from gsr_agent.commenting.orchestrator import plan_and_post_seed_comment
        row = _make_new_seed_row(abstract="We propose a novel gradient estimation method.")
        paper = _paper_from_row(row)
        comment_id, skip_reason = plan_and_post_seed_comment(
            paper, self._make_client(), self._make_db(), karma_remaining=50.0,
            now=_NOW, test_mode=True,
        )
        assert comment_id is not None
        assert skip_reason is None

    def test_missing_abstract_returns_seed_plan_missing_abstract(self):
        """Paper with empty abstract returns (None, 'seed_plan_missing_abstract')."""
        from gsr_agent.commenting.orchestrator import plan_and_post_seed_comment
        row = _make_new_seed_row(abstract="")
        paper = _paper_from_row(row)
        comment_id, skip_reason = plan_and_post_seed_comment(
            paper, self._make_client(), self._make_db(), karma_remaining=50.0,
            now=_NOW, test_mode=True,
        )
        assert comment_id is None
        assert skip_reason == "seed_plan_missing_abstract"

    def test_seed_planner_does_not_reject_new_seed_window_candidates(self):
        """seed planner classifies NEW+SEED_WINDOW as SEED, not SKIP."""
        from gsr_agent.commenting.orchestrator import plan_and_post_seed_comment
        row = _make_new_seed_row(abstract="A study of large language models for code generation.")
        paper = _paper_from_row(row)
        comment_id, skip_reason = plan_and_post_seed_comment(
            paper, self._make_client(), self._make_db(), karma_remaining=50.0,
            now=_NOW, test_mode=True,
        )
        assert skip_reason != "seed_plan_not_seed_opportunity", (
            "seed planner wrongly rejected a NEW+SEED_WINDOW candidate as not a seed opportunity"
        )
        assert comment_id is not None

    def test_live_gate_failed_replaced_by_specific_reason_in_operational_loop(self):
        """operational_loop seed_live_reason shows seed_plan_* not generic live_gate_failed."""
        row = _make_new_seed_row(abstract="")  # no abstract → seed_plan_missing_abstract
        paper = _paper_from_row(row)
        live_client = MagicMock()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
            patch(f"{_MOD}.get_run_mode", return_value="live"),
        ):
            result = _process_paper(
                paper, _DryRunClient(), _make_seed_process_db(), 100.0, _NOW,
                test_mode=False, live_reactive=True, live_budget_remaining=1,
                live_client=live_client,
            )
        assert result["seed_live_reason"] == "seed_plan_missing_abstract"
        assert result["seed_live_reason"] != "live_gate_failed"
        assert result["seed_live_posted"] is False
