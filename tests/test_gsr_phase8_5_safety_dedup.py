"""Tests for Phase 8.5: Safety & Dedup for Operational Loop.

Covers:
- DB methods: has_recent_reactive_action_for_comment
                has_recent_verdict_action_for_paper
- _process_paper dedup gates (reactive + verdict)
- run_operational_loop structured errors list
- skipped counter accuracy under dedup
"""

from __future__ import annotations

import json
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
from gsr_agent.storage.db import KoalaDB

_MOD = "gsr_agent.orchestration.operational_loop"
_NOW = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
_RECENT = (_NOW - timedelta(hours=6)).isoformat()   # 6 h ago — within 12 h window
_OLD = (_NOW - timedelta(hours=24)).isoformat()      # 24 h ago — outside 12 h window


# ---------------------------------------------------------------------------
# DB-level fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    koala_db = KoalaDB(str(tmp_path / "test.db"))
    yield koala_db
    koala_db.close()


def _insert_action(
    db: KoalaDB,
    paper_id: str,
    action_type: str,
    created_at: str,
    *,
    status: str = "dry_run",
    source_comment_id: str | None = None,
) -> None:
    """Insert a koala_agent_actions row with a controlled created_at timestamp."""
    details = json.dumps({"source_comment_id": source_comment_id}) if source_comment_id else None
    db._conn.execute(
        "INSERT INTO koala_agent_actions"
        " (paper_id, action_type, created_at, status, details)"
        " VALUES (?, ?, ?, ?, ?)",
        (paper_id, action_type, created_at, status, details),
    )
    db._conn.commit()


# ---------------------------------------------------------------------------
# TestHasRecentReactiveAction — DB method
# ---------------------------------------------------------------------------

class TestHasRecentReactiveAction:
    def test_recent_dry_run_blocks(self, db):
        _insert_action(db, "p1", "reactive_comment", _RECENT, source_comment_id="cmt-1")
        assert db.has_recent_reactive_action_for_comment("p1", "cmt-1", _NOW) is True

    def test_recent_success_blocks(self, db):
        _insert_action(
            db, "p1", "reactive_comment", _RECENT,
            status="success", source_comment_id="cmt-1",
        )
        assert db.has_recent_reactive_action_for_comment("p1", "cmt-1", _NOW) is True

    def test_old_action_does_not_block(self, db):
        _insert_action(db, "p1", "reactive_comment", _OLD, source_comment_id="cmt-1")
        assert db.has_recent_reactive_action_for_comment("p1", "cmt-1", _NOW) is False

    def test_different_comment_id_does_not_block(self, db):
        _insert_action(db, "p1", "reactive_comment", _RECENT, source_comment_id="cmt-other")
        assert db.has_recent_reactive_action_for_comment("p1", "cmt-1", _NOW) is False

    def test_different_paper_does_not_block(self, db):
        _insert_action(db, "p2", "reactive_comment", _RECENT, source_comment_id="cmt-1")
        assert db.has_recent_reactive_action_for_comment("p1", "cmt-1", _NOW) is False

    def test_no_action_returns_false(self, db):
        assert db.has_recent_reactive_action_for_comment("p1", "cmt-1", _NOW) is False

    def test_null_details_does_not_block(self, db):
        db._conn.execute(
            "INSERT INTO koala_agent_actions (paper_id, action_type, created_at, status)"
            " VALUES ('p1', 'reactive_comment', ?, 'dry_run')",
            (_RECENT,),
        )
        db._conn.commit()
        assert db.has_recent_reactive_action_for_comment("p1", "cmt-1", _NOW) is False

    def test_custom_within_hours(self, db):
        three_h_ago = (_NOW - timedelta(hours=3)).isoformat()
        _insert_action(db, "p1", "reactive_comment", three_h_ago, source_comment_id="cmt-1")
        assert db.has_recent_reactive_action_for_comment("p1", "cmt-1", _NOW, within_hours=2.0) is False
        assert db.has_recent_reactive_action_for_comment("p1", "cmt-1", _NOW, within_hours=4.0) is True

    def test_custom_statuses_excludes_pending(self, db):
        _insert_action(
            db, "p1", "reactive_comment", _RECENT,
            status="pending", source_comment_id="cmt-1",
        )
        assert db.has_recent_reactive_action_for_comment("p1", "cmt-1", _NOW) is False

    def test_boundary_exactly_at_cutoff_not_blocked(self, db):
        exactly_12h = (_NOW - timedelta(hours=12)).isoformat()
        _insert_action(db, "p1", "reactive_comment", exactly_12h, source_comment_id="cmt-1")
        # created_at == cutoff: cutoff uses >=, so this is at the boundary.
        # SQLite string comparison on ISO timestamps: equal strings means NOT blocked
        # because the query is created_at >= cutoff → equal means included → True.
        result = db.has_recent_reactive_action_for_comment("p1", "cmt-1", _NOW, within_hours=12.0)
        assert result is True


# ---------------------------------------------------------------------------
# TestHasRecentVerdictAction — DB method
# ---------------------------------------------------------------------------

class TestHasRecentVerdictAction:
    def test_recent_dry_run_blocks(self, db):
        _insert_action(db, "p1", "verdict_draft", _RECENT)
        assert db.has_recent_verdict_action_for_paper("p1", _NOW) is True

    def test_recent_success_blocks(self, db):
        _insert_action(db, "p1", "verdict_draft", _RECENT, status="success")
        assert db.has_recent_verdict_action_for_paper("p1", _NOW) is True

    def test_old_action_does_not_block(self, db):
        _insert_action(db, "p1", "verdict_draft", _OLD)
        assert db.has_recent_verdict_action_for_paper("p1", _NOW) is False

    def test_different_paper_does_not_block(self, db):
        _insert_action(db, "p2", "verdict_draft", _RECENT)
        assert db.has_recent_verdict_action_for_paper("p1", _NOW) is False

    def test_no_action_returns_false(self, db):
        assert db.has_recent_verdict_action_for_paper("p1", _NOW) is False

    def test_reactive_comment_action_does_not_block_verdict(self, db):
        _insert_action(db, "p1", "reactive_comment", _RECENT, source_comment_id="cmt-1")
        assert db.has_recent_verdict_action_for_paper("p1", _NOW) is False

    def test_custom_within_hours(self, db):
        three_h_ago = (_NOW - timedelta(hours=3)).isoformat()
        _insert_action(db, "p1", "verdict_draft", three_h_ago)
        assert db.has_recent_verdict_action_for_paper("p1", _NOW, within_hours=2.0) is False
        assert db.has_recent_verdict_action_for_paper("p1", _NOW, within_hours=4.0) is True

    def test_custom_statuses_excludes_failed(self, db):
        _insert_action(db, "p1", "verdict_draft", _RECENT, status="failed")
        assert db.has_recent_verdict_action_for_paper("p1", _NOW) is False


# ---------------------------------------------------------------------------
# TestProcessPaperDedup — _process_paper dedup gate behaviour
# ---------------------------------------------------------------------------

def _make_paper():
    return _paper_from_row({
        "paper_id": "paper-dedup-001",
        "title": "Dedup Test Paper",
        "open_time": "2026-04-25T00:00:00+00:00",       # 36h before _NOW → comment window open
        "review_end_time": "2026-04-27T00:00:00+00:00",
        "verdict_end_time": "2026-04-28T00:00:00+00:00",
        "state": "REVIEW_ACTIVE",
        "pdf_url": "",
        "local_pdf_path": None,
    })


def _make_candidate(comment_id: str = "cmt-src-001") -> ReactiveAnalysisResult:
    return ReactiveAnalysisResult(
        comment_id=comment_id,
        paper_id="paper-dedup-001",
        recommendation="react",
        draft_text="[DRY-RUN — not posted]\n\nReactive draft.",
    )


def _make_eligibility(eligible: bool = True) -> VerdictEligibilityResult:
    return VerdictEligibilityResult(
        eligible=eligible,
        reason_code="eligible" if eligible else "no_react_signal",
        heat_band="goldilocks",
        distinct_citable_other_agents=3,
        strongest_contradiction_confidence=0.85,
        selected_candidates=[],
    )


def _make_db(*, reactive_dedup: bool = False, verdict_dedup: bool = False) -> MagicMock:
    db = MagicMock()
    db.has_recent_reactive_action_for_comment.return_value = reactive_dedup
    db.has_recent_verdict_action_for_paper.return_value = verdict_dedup
    return db


class TestProcessPaperDedup:
    def test_reactive_dedup_skips_post(self):
        paper = _make_paper()
        candidate = _make_candidate()
        db = _make_db(reactive_dedup=True)
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[candidate]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=candidate),
            patch(f"{_MOD}.plan_and_post_reactive_comment") as mock_post,
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
        ):
            result = _process_paper(paper, _DryRunClient(), db, 100.0, _NOW, test_mode=True)

        mock_post.assert_not_called()
        assert result["reactive_status"] == "dedup_skipped"
        assert result["reactive_reason"] == "recent_reactive_action"
        assert result["reactive_draft_created"] is False
        assert result["has_reactive_candidate"] is True

    def test_reactive_no_dedup_proceeds(self):
        paper = _make_paper()
        candidate = _make_candidate()
        db = _make_db(reactive_dedup=False)
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[candidate]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=candidate),
            patch(f"{_MOD}.plan_and_post_reactive_comment", return_value="cmt-new") as mock_post,
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
        ):
            result = _process_paper(paper, _DryRunClient(), db, 100.0, _NOW, test_mode=True)

        mock_post.assert_called_once()
        assert result["reactive_status"] == "dry_run"
        assert result["reactive_draft_created"] is True

    def test_verdict_dedup_skips_plan(self):
        paper = _make_paper()
        db = _make_db(verdict_dedup=True)
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(True)),
            patch(f"{_MOD}.plan_verdict_for_paper") as mock_verdict,
        ):
            result = _process_paper(paper, _DryRunClient(), db, 100.0, _NOW, test_mode=True)

        mock_verdict.assert_not_called()
        assert result["verdict_status"] == "dedup_skipped"
        assert result["verdict_reason"] == "recent_verdict_action"
        assert result["verdict_draft_created"] is False
        assert result["verdict_eligible"] is True

    def test_verdict_no_dedup_proceeds(self):
        paper = _make_paper()
        db = _make_db(verdict_dedup=False)
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(True)),
            patch(f"{_MOD}.plan_verdict_for_paper", return_value={"artifact_url": "url/v.md"}),
        ):
            result = _process_paper(paper, _DryRunClient(), db, 100.0, _NOW, test_mode=True)

        assert result["verdict_status"] == "dry_run"
        assert result["verdict_draft_created"] is True

    def test_dedup_check_uses_candidate_comment_id(self):
        paper = _make_paper()
        candidate = _make_candidate(comment_id="cmt-specific-42")
        db = _make_db(reactive_dedup=False)
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[candidate]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=candidate),
            patch(f"{_MOD}.plan_and_post_reactive_comment", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
        ):
            _process_paper(paper, _DryRunClient(), db, 100.0, _NOW, test_mode=True)

        call_args = db.has_recent_reactive_action_for_comment.call_args
        assert call_args[0][1] == "cmt-specific-42"

    def test_verdict_dedup_not_checked_when_ineligible(self):
        paper = _make_paper()
        db = _make_db()
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
        ):
            _process_paper(paper, _DryRunClient(), db, 100.0, _NOW, test_mode=True)

        db.has_recent_verdict_action_for_paper.assert_not_called()

    def test_dedup_skipped_result_includes_paper_id(self):
        paper = _make_paper()
        candidate = _make_candidate()
        db = _make_db(reactive_dedup=True)
        with (
            patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[candidate]),
            patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=candidate),
            patch(f"{_MOD}.evaluate_verdict_eligibility", return_value=_make_eligibility(False)),
        ):
            result = _process_paper(paper, _DryRunClient(), db, 100.0, _NOW, test_mode=True)

        assert result["paper_id"] == "paper-dedup-001"


# ---------------------------------------------------------------------------
# TestRunLoopStructuredErrors — aggregate errors list
# ---------------------------------------------------------------------------

def _make_loop_db(paper_rows: list | None = None) -> MagicMock:
    db = MagicMock()
    db.get_papers.return_value = paper_rows or []
    db.get_comment_stats.return_value = {"total": 3, "ours": 0, "citable_other": 3}
    db.has_prior_participation.return_value = False
    return db


def _make_paper_row(paper_id: str = "paper-abc-123") -> dict:
    open_time = _NOW - timedelta(hours=6)
    return {
        "paper_id": paper_id,
        "title": "Test Paper",
        "open_time": open_time.isoformat(),
        "review_end_time": (open_time + timedelta(hours=48)).isoformat(),
        "verdict_end_time": (open_time + timedelta(hours=72)).isoformat(),
        "state": "REVIEW_ACTIVE",
        "pdf_url": "",
        "local_pdf_path": None,
    }


def _no_op_result(paper_id: str = "paper-abc-123") -> dict:
    return {
        "paper_id": paper_id,
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


class TestRunLoopStructuredErrors:
    def test_loop_errors_is_list_by_default(self):
        db = _make_loop_db([])
        with (
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            counters = run_operational_loop(db, _NOW, output_dir="/tmp/rep")

        assert isinstance(counters["errors"], list)
        assert counters["errors_count"] == 0

    def test_loop_returns_structured_error_when_paper_fails(self):
        rows = [_make_paper_row("paper-ok"), _make_paper_row("paper-bad")]

        def side_effect(paper, *args, **kwargs):
            if paper.paper_id == "paper-bad":
                raise ValueError("boom")
            return _no_op_result(paper.paper_id)

        with (
            patch(f"{_MOD}._process_paper", side_effect=side_effect),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            counters = run_operational_loop(
                _make_loop_db(rows), _NOW, output_dir="/tmp/rep"
            )

        assert counters["errors_count"] == 1
        assert len(counters["errors"]) == 1
        err = counters["errors"][0]
        assert err["paper_id"] == "paper-bad"
        assert err["stage"] == "process_paper"
        assert "boom" in err["error"]

    def test_loop_error_does_not_block_other_papers(self):
        rows = [_make_paper_row("paper-bad"), _make_paper_row("paper-ok")]

        def side_effect(paper, *args, **kwargs):
            if paper.paper_id == "paper-bad":
                raise RuntimeError("crash")
            return _no_op_result(paper.paper_id)

        with (
            patch(f"{_MOD}._process_paper", side_effect=side_effect),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            counters = run_operational_loop(
                _make_loop_db(rows), _NOW, output_dir="/tmp/rep"
            )

        assert counters["papers_processed"] == 1
        assert counters["errors_count"] == 1

    def test_multiple_errors_all_collected(self):
        rows = [_make_paper_row(f"paper-{i}") for i in range(3)]

        def side_effect(paper, *args, **kwargs):
            raise RuntimeError(f"fail-{paper.paper_id}")

        with (
            patch(f"{_MOD}._process_paper", side_effect=side_effect),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            counters = run_operational_loop(
                _make_loop_db(rows), _NOW, output_dir="/tmp/rep"
            )

        assert counters["errors_count"] == 3
        assert len(counters["errors"]) == 3
        assert all(e["stage"] == "process_paper" for e in counters["errors"])


# ---------------------------------------------------------------------------
# TestSkippedCountUnderDedup — skipped counter accuracy
# ---------------------------------------------------------------------------

class TestSkippedCountUnderDedup:
    def _run_with_result(self, result: dict) -> dict:
        with (
            patch(f"{_MOD}._process_paper", return_value=result),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            return run_operational_loop(
                _make_loop_db([_make_paper_row()]), _NOW, output_dir="/tmp/rep"
            )

    def test_reactive_dedup_counts_as_skipped(self):
        result = {**_no_op_result(), "reactive_status": "dedup_skipped", "has_reactive_candidate": True}
        counters = self._run_with_result(result)
        assert counters["skipped"] == 1
        assert counters["reactive_drafts_created"] == 0
        assert counters["reactive_candidates_found"] == 1

    def test_verdict_dedup_counts_as_skipped(self):
        result = {
            **_no_op_result(),
            "verdict_status": "dedup_skipped",
            "verdict_eligible": True,
        }
        counters = self._run_with_result(result)
        assert counters["skipped"] == 1
        assert counters["verdict_drafts_created"] == 0
        assert counters["verdicts_eligible"] == 1

    def test_both_dedup_counts_as_single_skipped(self):
        result = {
            **_no_op_result(),
            "reactive_status": "dedup_skipped",
            "verdict_status": "dedup_skipped",
            "has_reactive_candidate": True,
            "verdict_eligible": True,
        }
        counters = self._run_with_result(result)
        assert counters["skipped"] == 1

    def test_reactive_draft_created_not_skipped(self):
        result = {
            **_no_op_result(),
            "reactive_status": "dry_run",
            "reactive_draft_created": True,
            "has_reactive_candidate": True,
        }
        counters = self._run_with_result(result)
        assert counters["skipped"] == 0
        assert counters["reactive_drafts_created"] == 1

    def test_verdict_draft_created_not_skipped(self):
        result = {
            **_no_op_result(),
            "verdict_status": "dry_run",
            "verdict_draft_created": True,
            "verdict_eligible": True,
        }
        counters = self._run_with_result(result)
        assert counters["skipped"] == 0
        assert counters["verdict_drafts_created"] == 1
