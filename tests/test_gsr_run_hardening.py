"""Tests for Competition MVP Final Run Hardening.

Covers:
  - run_preflight_checks: dry-run, live-mode env gates, verdict allowlist gate
  - per-paper failure isolation
  - aggregate counters include failures
  - CLI start line and preflight-fail early exit
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from gsr_agent.orchestration.operational_loop import (
    _DryRunClient,
    _paper_from_row,
    run_operational_loop,
    run_preflight_checks,
)

_MOD = "gsr_agent.orchestration.operational_loop"
_NOW = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

_FULL_LIVE_ENV = {
    "KOALA_RUN_MODE": "live",
    "KOALA_API_TOKEN": "test-token-abc",
    "KOALA_ARTIFACT_MODE": "github",
    "KOALA_GITHUB_REPO": "https://github.com/koala-science/gsr-artifacts",
}


def _make_paper_row(paper_id: str = "paper-harden-001") -> dict:
    return {
        "paper_id": paper_id,
        "title": "Hardening Test Paper",
        "open_time": "2026-04-01T00:00:00+00:00",
        "review_end_time": "2026-04-15T00:00:00+00:00",
        "verdict_end_time": "2026-04-22T00:00:00+00:00",
        "state": "REVIEW_ACTIVE",
        "pdf_url": "https://example.com/paper.pdf",
        "local_pdf_path": None,
    }


# ---------------------------------------------------------------------------
# TestRunPreflightChecks — unit tests for the preflight helper
# ---------------------------------------------------------------------------

class TestRunPreflightChecksDryRun:

    def test_dry_run_minimal_config_passes(self, tmp_path):
        db_path = str(tmp_path / "agent.db")
        out_dir = str(tmp_path / "reports")
        ok, failures = run_preflight_checks(db_path, out_dir)
        assert ok is True
        assert failures == []

    def test_creates_missing_output_dir(self, tmp_path):
        db_path = str(tmp_path / "agent.db")
        out_dir = str(tmp_path / "new" / "reports")
        ok, failures = run_preflight_checks(db_path, out_dir)
        assert ok is True
        from pathlib import Path
        assert Path(out_dir).exists()

    def test_missing_db_parent_fails(self, tmp_path):
        db_path = str(tmp_path / "nonexistent_dir" / "agent.db")
        out_dir = str(tmp_path / "reports")
        ok, failures = run_preflight_checks(db_path, out_dir)
        assert ok is False
        assert any("DB parent" in f for f in failures)

    def test_no_live_flags_no_env_required(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KOALA_RUN_MODE", raising=False)
        monkeypatch.delenv("KOALA_API_TOKEN", raising=False)
        db_path = str(tmp_path / "agent.db")
        out_dir = str(tmp_path / "reports")
        ok, failures = run_preflight_checks(db_path, out_dir, live_reactive=False, live_verdict=False)
        assert ok is True

    def test_live_verdict_without_paper_ids_fails(self, tmp_path, monkeypatch):
        for k, v in _FULL_LIVE_ENV.items():
            monkeypatch.setenv(k, v)
        db_path = str(tmp_path / "agent.db")
        out_dir = str(tmp_path / "reports")
        ok, failures = run_preflight_checks(
            db_path, out_dir, live_verdict=True, paper_ids=None
        )
        assert ok is False
        assert any("allowlist" in f or "paper-id" in f or "paper_ids" in f for f in failures)

    def test_live_verdict_with_paper_ids_passes_env(self, tmp_path, monkeypatch):
        for k, v in _FULL_LIVE_ENV.items():
            monkeypatch.setenv(k, v)
        db_path = str(tmp_path / "agent.db")
        out_dir = str(tmp_path / "reports")
        ok, failures = run_preflight_checks(
            db_path, out_dir, live_verdict=True, paper_ids=["paper-001"]
        )
        assert ok is True
        assert failures == []


class TestRunPreflightChecksLiveEnvGates:

    def test_live_reactive_without_run_mode_fails(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KOALA_RUN_MODE", "dry_run")
        monkeypatch.setenv("KOALA_API_TOKEN", "tok")
        monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
        monkeypatch.setenv("KOALA_GITHUB_REPO", "owner/repo")
        db_path = str(tmp_path / "agent.db")
        out_dir = str(tmp_path / "reports")
        ok, failures = run_preflight_checks(db_path, out_dir, live_reactive=True)
        assert ok is False
        assert any("KOALA_RUN_MODE" in f for f in failures)

    def test_live_reactive_without_api_token_fails(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KOALA_RUN_MODE", "live")
        monkeypatch.delenv("KOALA_API_TOKEN", raising=False)
        monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
        monkeypatch.setenv("KOALA_GITHUB_REPO", "owner/repo")
        db_path = str(tmp_path / "agent.db")
        out_dir = str(tmp_path / "reports")
        ok, failures = run_preflight_checks(db_path, out_dir, live_reactive=True)
        assert ok is False
        assert any("KOALA_API_TOKEN" in f for f in failures)

    def test_live_reactive_without_github_config_fails(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KOALA_RUN_MODE", "live")
        monkeypatch.setenv("KOALA_API_TOKEN", "tok")
        monkeypatch.setenv("KOALA_ARTIFACT_MODE", "local")
        monkeypatch.delenv("KOALA_GITHUB_REPO", raising=False)
        db_path = str(tmp_path / "agent.db")
        out_dir = str(tmp_path / "reports")
        ok, failures = run_preflight_checks(db_path, out_dir, live_reactive=True)
        assert ok is False
        assert any("KOALA_ARTIFACT_MODE" in f or "KOALA_GITHUB_REPO" in f for f in failures)

    def test_live_verdict_all_env_set_with_paper_ids_passes(self, tmp_path, monkeypatch):
        for k, v in _FULL_LIVE_ENV.items():
            monkeypatch.setenv(k, v)
        db_path = str(tmp_path / "agent.db")
        out_dir = str(tmp_path / "reports")
        ok, failures = run_preflight_checks(
            db_path, out_dir, live_verdict=True, paper_ids=["p1"]
        )
        assert ok is True

    def test_multiple_failures_accumulate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KOALA_RUN_MODE", "dry_run")
        monkeypatch.delenv("KOALA_API_TOKEN", raising=False)
        monkeypatch.setenv("KOALA_ARTIFACT_MODE", "local")
        monkeypatch.delenv("KOALA_GITHUB_REPO", raising=False)
        db_path = str(tmp_path / "nonexistent_dir" / "agent.db")
        out_dir = str(tmp_path / "reports")
        ok, failures = run_preflight_checks(
            db_path, out_dir, live_verdict=True, paper_ids=None
        )
        assert ok is False
        assert len(failures) >= 3


# ---------------------------------------------------------------------------
# TestPerPaperFailureIsolation — one failure does not abort remaining papers
# ---------------------------------------------------------------------------

class TestPerPaperFailureIsolation:

    def _run_with_mixed_results(self, tmp_path):
        """Simulate 3 papers where paper-002 raises; others succeed."""
        rows = [
            _make_paper_row("paper-001"),
            _make_paper_row("paper-002"),
            _make_paper_row("paper-003"),
        ]
        db = MagicMock()
        db.get_papers.return_value = rows

        call_count = 0

        def _side(paper, *a, **kw):
            nonlocal call_count
            call_count += 1
            if paper.paper_id == "paper-002":
                raise RuntimeError("Simulated DB timeout")
            return {
                "paper_id": paper.paper_id,
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

        out_dir = str(tmp_path / "reports")
        with (
            patch(f"{_MOD}._process_paper", side_effect=_side),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            counters = run_operational_loop(
                db, _NOW, output_dir=out_dir, test_mode=True
            )
        return counters, call_count

    def test_all_three_papers_attempted(self, tmp_path):
        _, call_count = self._run_with_mixed_results(tmp_path)
        assert call_count == 3

    def test_two_papers_processed_successfully(self, tmp_path):
        counters, _ = self._run_with_mixed_results(tmp_path)
        assert counters["papers_processed"] == 2

    def test_one_paper_recorded_as_error(self, tmp_path):
        counters, _ = self._run_with_mixed_results(tmp_path)
        assert counters["errors_count"] == 1

    def test_error_entry_contains_paper_id(self, tmp_path):
        counters, _ = self._run_with_mixed_results(tmp_path)
        assert counters["errors"][0]["paper_id"] == "paper-002"

    def test_error_entry_contains_error_message(self, tmp_path):
        counters, _ = self._run_with_mixed_results(tmp_path)
        assert "Simulated DB timeout" in counters["errors"][0]["error"]


# ---------------------------------------------------------------------------
# TestAggregateCounters — all required counters present and accurate
# ---------------------------------------------------------------------------

class TestAggregateCounters:

    def _run_empty(self, tmp_path):
        db = MagicMock()
        db.get_papers.return_value = []
        out_dir = str(tmp_path / "reports")
        with (
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            return run_operational_loop(db, _NOW, output_dir=out_dir)

    def test_papers_seen_present(self, tmp_path):
        counters = self._run_empty(tmp_path)
        assert "papers_seen" in counters

    def test_papers_processed_present(self, tmp_path):
        counters = self._run_empty(tmp_path)
        assert "papers_processed" in counters

    def test_errors_count_present(self, tmp_path):
        counters = self._run_empty(tmp_path)
        assert "errors_count" in counters

    def test_live_reactive_posts_present(self, tmp_path):
        counters = self._run_empty(tmp_path)
        assert "live_reactive_posts" in counters

    def test_live_verdict_submissions_present(self, tmp_path):
        counters = self._run_empty(tmp_path)
        assert "live_verdict_submissions" in counters

    def test_verdict_missing_score_counter_present(self, tmp_path):
        counters = self._run_empty(tmp_path)
        assert "verdict_live_missing_score" in counters

    def test_verdict_invalid_score_counter_present(self, tmp_path):
        counters = self._run_empty(tmp_path)
        assert "verdict_live_invalid_score" in counters

    def test_empty_run_all_action_counters_zero(self, tmp_path):
        counters = self._run_empty(tmp_path)
        assert counters["papers_seen"] == 0
        assert counters["papers_processed"] == 0
        assert counters["errors_count"] == 0
        assert counters["live_reactive_posts"] == 0
        assert counters["live_verdict_submissions"] == 0


# ---------------------------------------------------------------------------
# TestCLIHardening — preflight fail exit, startup line
# ---------------------------------------------------------------------------

class TestCLIHardening:

    def test_preflight_fail_prints_fail_lines(self, tmp_path, capsys, monkeypatch):
        """CLI exits without starting loop when preflight fails."""
        from gsr_agent.orchestration.operational_loop import main

        monkeypatch.setenv("KOALA_RUN_MODE", "dry_run")
        monkeypatch.delenv("KOALA_API_TOKEN", raising=False)
        nonexistent_db = str(tmp_path / "no_dir" / "agent.db")

        with patch("sys.argv", ["loop", "--db", nonexistent_db, "--live-reactive"]):
            main()

        out = capsys.readouterr().out
        assert "[preflight] FAIL:" in out

    def test_preflight_fail_does_not_start_loop(self, tmp_path, capsys, monkeypatch):
        from gsr_agent.orchestration.operational_loop import main

        monkeypatch.setenv("KOALA_RUN_MODE", "dry_run")
        monkeypatch.delenv("KOALA_API_TOKEN", raising=False)
        nonexistent_db = str(tmp_path / "no_dir" / "agent.db")

        with (
            patch("sys.argv", ["loop", "--db", nonexistent_db, "--live-reactive"]),
            patch(f"{_MOD}.run_operational_loop") as mock_loop,
        ):
            main()

        mock_loop.assert_not_called()

    def test_startup_line_printed_on_dry_run(self, tmp_path, capsys, monkeypatch):
        from gsr_agent.orchestration.operational_loop import main

        db_path = str(tmp_path / "agent.db")
        out_dir = str(tmp_path / "reports")
        mock_counters = {
            "papers_seen": 0,
            "papers_processed": 0,
            "reactive_drafts_created": 0,
            "live_reactive_posts": 0,
            "verdict_drafts_created": 0,
            "live_verdict_submissions": 0,
            "errors_count": 0,
            "reactive_live_eligible": 0,
            "reactive_dedup_skipped": 0,
            "live_budget_exhausted": 0,
            "reactive_live_gate_failed": 0,
            "verdict_live_missing_score": 0,
            "verdict_live_invalid_score": 0,
            "summary_path": str(tmp_path / "run_summary.md"),
        }

        with (
            patch("sys.argv", ["loop", "--db", db_path, "--out", out_dir]),
            patch(f"{_MOD}.run_operational_loop", return_value=mock_counters),
            patch("gsr_agent.storage.db.KoalaDB"),
        ):
            main()

        out = capsys.readouterr().out
        assert "[loop] START" in out
        assert "mode=dry_run" in out

    def test_startup_line_shows_live_mode(self, tmp_path, capsys, monkeypatch):
        from gsr_agent.orchestration.operational_loop import main

        for k, v in _FULL_LIVE_ENV.items():
            monkeypatch.setenv(k, v)

        db_path = str(tmp_path / "agent.db")
        out_dir = str(tmp_path / "reports")
        mock_counters = {
            "papers_seen": 0,
            "papers_processed": 0,
            "reactive_drafts_created": 0,
            "live_reactive_posts": 0,
            "verdict_drafts_created": 0,
            "live_verdict_submissions": 0,
            "errors_count": 0,
            "reactive_live_eligible": 0,
            "reactive_dedup_skipped": 0,
            "live_budget_exhausted": 0,
            "reactive_live_gate_failed": 0,
            "verdict_live_missing_score": 0,
            "verdict_live_invalid_score": 0,
            "summary_path": str(tmp_path / "run_summary.md"),
        }

        with (
            patch(
                "sys.argv",
                ["loop", "--db", db_path, "--out", out_dir,
                 "--live-reactive", "--paper-id", "paper-001"],
            ),
            patch(f"{_MOD}.run_operational_loop", return_value=mock_counters),
            patch("gsr_agent.storage.db.KoalaDB"),
        ):
            main()

        out = capsys.readouterr().out
        assert "mode=live" in out
        assert "reactive=True" in out


# ---------------------------------------------------------------------------
# TestBackwardCompatibility — dry-run path unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_dry_run_loop_requires_no_env_vars(self, tmp_path, monkeypatch):
        """run_operational_loop works in dry-run with no KOALA_* env vars set."""
        monkeypatch.delenv("KOALA_RUN_MODE", raising=False)
        monkeypatch.delenv("KOALA_API_TOKEN", raising=False)
        monkeypatch.delenv("KOALA_ARTIFACT_MODE", raising=False)
        monkeypatch.delenv("KOALA_GITHUB_REPO", raising=False)

        db = MagicMock()
        db.get_papers.return_value = []
        out_dir = str(tmp_path / "reports")

        with (
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            counters = run_operational_loop(
                db, _NOW, output_dir=out_dir, test_mode=True
            )

        assert counters["papers_seen"] == 0
        assert counters["errors_count"] == 0

    def test_live_gates_remain_when_test_mode_true(self, tmp_path):
        """test_mode=True suppresses all live actions even with live flags set."""
        rows = [_make_paper_row()]
        db = MagicMock()
        db.get_papers.return_value = rows

        result_template = {
            "paper_id": "paper-harden-001",
            "reactive_status": "none",
            "reactive_reason": None,
            "reactive_artifact": None,
            "reactive_live_posted": False,
            "reactive_live_reason": "live_gate_failed",
            "verdict_status": "ineligible",
            "verdict_reason": None,
            "verdict_artifact": None,
            "verdict_live_submitted": False,
            "verdict_live_reason": "live_gate_failed",
            "has_reactive_candidate": False,
            "reactive_draft_created": False,
            "verdict_eligible": False,
            "verdict_draft_created": False,
        }

        out_dir = str(tmp_path / "reports")
        with (
            patch(f"{_MOD}._process_paper", return_value=result_template),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            counters = run_operational_loop(
                db, _NOW, output_dir=out_dir,
                test_mode=True,
                live_reactive=True,
                live_verdict=True,
            )

        assert counters["live_reactive_posts"] == 0
        assert counters["live_verdict_submissions"] == 0
