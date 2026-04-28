"""Tests for competition strategy: seed/root and verdict opportunity behavior."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from gsr_agent.orchestration.operational_loop import (
    run_operational_loop,
    run_preflight_checks,
)

_MOD = "gsr_agent.orchestration.operational_loop"
_NOW = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)


def _seed_row(paper_id: str = "seed-001", hours_ago: float = 6.0) -> dict:
    open_time = _NOW - timedelta(hours=hours_ago)
    return {
        "paper_id": paper_id,
        "title": "Seed Test Paper",
        "open_time": open_time.isoformat(),
        "review_end_time": (open_time + timedelta(hours=48)).isoformat(),
        "verdict_end_time": (open_time + timedelta(hours=72)).isoformat(),
        "state": "REVIEW_ACTIVE",
        "pdf_url": "",
        "local_pdf_path": None,
    }


def _deliberating_row(paper_id: str = "delib-001") -> dict:
    open_time = _NOW - timedelta(hours=55)
    return {
        "paper_id": paper_id,
        "title": "Deliberating Paper",
        "open_time": open_time.isoformat(),
        "review_end_time": (open_time + timedelta(hours=48)).isoformat(),
        "verdict_end_time": (open_time + timedelta(hours=72)).isoformat(),
        "state": "deliberating",
        "pdf_url": "",
        "local_pdf_path": None,
        "deliberating_at": (open_time + timedelta(hours=48)).isoformat(),
    }


def _make_db(rows: list, *, citable_other: int = 0, has_participated: bool = False) -> MagicMock:
    db = MagicMock()
    db.get_papers.return_value = rows
    db.get_comment_stats.return_value = {"citable_other": citable_other}
    db.has_prior_participation.return_value = has_participated
    db.has_recent_reactive_action_for_comment.return_value = False
    db.has_recent_verdict_action_for_paper.return_value = False
    return db


_NO_OP_RESULT = {
    "paper_id": "x",
    "reactive_status": "none",
    "reactive_reason": None,
    "reactive_artifact": None,
    "reactive_live_posted": False,
    "reactive_live_reason": None,
    "verdict_status": "ineligible",
    "verdict_reason": None,
    "verdict_artifact": None,
    "verdict_live_submitted": False,
    "verdict_live_reason": None,
    "has_reactive_candidate": False,
    "reactive_draft_created": False,
    "verdict_eligible": False,
    "verdict_draft_created": False,
    "window_skipped": False,
}


def _run(db: MagicMock, output_dir: str = "/tmp/test_competition_strategy") -> dict:
    with (
        patch(f"{_MOD}._process_paper", return_value=_NO_OP_RESULT),
        patch(f"{_MOD}.build_run_summary", return_value=[]),
        patch(f"{_MOD}.write_run_summary_markdown"),
        patch(f"{_MOD}.write_run_summary_jsonl"),
    ):
        return run_operational_loop(db, _NOW, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Part A: Seed/root comment participation
# ---------------------------------------------------------------------------

def test_seed_5_comments_preferred_band_not_skip_too_cold(caplog):
    """REVIEW_ACTIVE paper with 5 comments → seed_candidate_preferred_band, not skip_too_cold."""
    db = _make_db([_seed_row()], citable_other=5, has_participated=False)
    with caplog.at_level("INFO", logger=_MOD):
        _run(db)
    assert "seed_candidate_preferred_band" in caplog.text
    assert "selected_seed_candidate" in caplog.text
    assert "skip_too_cold" not in caplog.text


def test_seed_0_comments_skip_too_cold(caplog):
    """0-comment paper → skip_too_cold, not selected as seed candidate."""
    db = _make_db([_seed_row()], citable_other=0, has_participated=False)
    with caplog.at_level("INFO", logger=_MOD):
        _run(db)
    assert "skip_too_cold" in caplog.text
    assert "selected_seed_candidate" not in caplog.text


def test_seed_13_comments_saturated(caplog):
    """>12 comments → saturated_comments, not selected as seed candidate."""
    db = _make_db([_seed_row()], citable_other=13, has_participated=False)
    with caplog.at_level("INFO", logger=_MOD):
        _run(db)
    assert "saturated_comments" in caplog.text
    assert "selected_seed_candidate" not in caplog.text


# ---------------------------------------------------------------------------
# Part B: Verdict opportunity surfacing
# ---------------------------------------------------------------------------

def test_deliberating_3_citable_verdict_ready(caplog):
    """Deliberating paper with >=3 citable comments → verdict_ready log."""
    db = _make_db([_deliberating_row()], citable_other=3, has_participated=True)
    with caplog.at_level("INFO", logger=_MOD):
        _run(db)
    assert "verdict_ready" in caplog.text
    assert "ours=Y" in caplog.text


def test_deliberating_2_citable_verdict_blocked_insufficient(caplog):
    """Deliberating paper with only 2 citable → verdict_blocked insufficient_citations."""
    db = _make_db([_deliberating_row()], citable_other=2, has_participated=True)
    with caplog.at_level("INFO", logger=_MOD):
        _run(db)
    assert "verdict_blocked" in caplog.text
    assert "insufficient_citations" in caplog.text


def test_verdict_opportunity_report_written(tmp_path):
    """Verdict opportunity report is written to verdict_opportunities.md."""
    db = _make_db([_deliberating_row()], citable_other=4, has_participated=True)
    with (
        patch(f"{_MOD}._process_paper", return_value=_NO_OP_RESULT),
        patch(f"{_MOD}.build_run_summary", return_value=[]),
        patch(f"{_MOD}.write_run_summary_markdown"),
        patch(f"{_MOD}.write_run_summary_jsonl"),
    ):
        run_operational_loop(db, _NOW, output_dir=str(tmp_path))
    report = tmp_path / "verdict_opportunities.md"
    assert report.exists()
    content = report.read_text()
    assert "delib-001" in content
    assert "submit_verdict" in content


def test_live_verdict_requires_paper_id_allowlist(tmp_path):
    """live-verdict still requires explicit --paper-id allowlist."""
    (tmp_path / "workspace").mkdir()
    with (
        patch(f"{_MOD}.get_run_mode", return_value="live"),
        patch(f"{_MOD}.is_github_publish_configured", return_value=True),
        patch.dict(os.environ, {"KOALA_API_TOKEN": "tok"}),
    ):
        ok, failures = run_preflight_checks(
            str(tmp_path / "workspace" / "agent.db"),
            str(tmp_path),
            live_verdict=True,
            paper_ids=None,
        )
    assert not ok
    assert any("paper-id" in f for f in failures)
