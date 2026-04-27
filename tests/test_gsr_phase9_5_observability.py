"""Tests for Phase 9.5: Live Reactive Observability."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gsr_agent.orchestration.operational_loop import run_operational_loop
from gsr_agent.reporting.run_summary import (
    write_run_summary_jsonl,
    write_run_summary_markdown,
)

_MOD = "gsr_agent.orchestration.operational_loop"
_NOW = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper_row(paper_id: str = "paper-abc-123") -> dict:
    return {
        "paper_id": paper_id,
        "title": "Test Paper",
        "open_time": "2026-04-01T00:00:00+00:00",
        "review_end_time": "2026-04-15T00:00:00+00:00",
        "verdict_end_time": "2026-04-22T00:00:00+00:00",
        "state": "REVIEW_ACTIVE",
        "pdf_url": "https://example.com/paper.pdf",
        "local_pdf_path": None,
    }


def _make_loop_db(paper_rows: list | None = None) -> MagicMock:
    db = MagicMock()
    db.get_papers.return_value = paper_rows if paper_rows is not None else [_make_paper_row()]
    return db


def _base_result(paper_id: str = "paper-abc-123") -> dict:
    """Minimal _process_paper result with all Phase 9 fields."""
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


def _run_loop(paper_rows, process_results, *, live_reactive=False, test_mode=True) -> dict:
    """Run run_operational_loop with fully mocked _process_paper and writers."""
    db = _make_loop_db(paper_rows)
    results_iter = iter(process_results)

    def _side_effect(paper, *args, **kwargs):
        return next(results_iter)

    with (
        patch(f"{_MOD}._process_paper", side_effect=_side_effect),
        patch(f"{_MOD}.build_run_summary", return_value=[]),
        patch(f"{_MOD}.write_run_summary_markdown"),
        patch(f"{_MOD}.write_run_summary_jsonl"),
    ):
        return run_operational_loop(
            db, _NOW,
            live_reactive=live_reactive,
            test_mode=test_mode,
            output_dir="/tmp/rep",
        )


# ---------------------------------------------------------------------------
# TestNewCounters — counter increment correctness
# ---------------------------------------------------------------------------

class TestNewCounters:
    """run_operational_loop counter increments for Phase 9.5 fields."""

    def test_all_new_counters_present_in_empty_run(self):
        counters = _run_loop([], [])
        for key in (
            "reactive_dedup_skipped",
            "reactive_live_eligible",
            "reactive_live_posted",
            "live_budget_exhausted",
            "reactive_live_gate_failed",
        ):
            assert key in counters, f"missing counter: {key}"

    def test_all_new_counters_zero_for_no_candidate(self):
        counters = _run_loop([_make_paper_row()], [_base_result()])
        assert counters["reactive_dedup_skipped"] == 0
        assert counters["reactive_live_eligible"] == 0
        assert counters["reactive_live_posted"] == 0
        assert counters["live_budget_exhausted"] == 0
        assert counters["reactive_live_gate_failed"] == 0

    def test_dedup_skipped_increments(self):
        r = dict(_base_result())
        r["reactive_status"] = "dedup_skipped"
        r["reactive_live_reason"] = "dedup_skipped"
        counters = _run_loop([_make_paper_row()], [r])
        assert counters["reactive_dedup_skipped"] == 1

    def test_dedup_skipped_does_not_count_as_eligible(self):
        r = dict(_base_result())
        r["reactive_status"] = "dedup_skipped"
        r["reactive_live_reason"] = "dedup_skipped"
        counters = _run_loop([_make_paper_row()], [r])
        assert counters["reactive_live_eligible"] == 0

    def test_live_posted_increments_both_counters(self):
        r = dict(_base_result())
        r["reactive_live_posted"] = True
        r["reactive_live_reason"] = "live_posted"
        r["reactive_status"] = "live_posted"
        counters = _run_loop([_make_paper_row()], [r], live_reactive=True)
        assert counters["reactive_live_posted"] == 1
        assert counters["live_reactive_posts"] == 1
        assert counters["reactive_live_eligible"] == 1

    def test_budget_exhausted_increments(self):
        r = dict(_base_result())
        r["reactive_live_reason"] = "live_budget_exhausted"
        r["has_reactive_candidate"] = True
        counters = _run_loop([_make_paper_row()], [r], live_reactive=True)
        assert counters["live_budget_exhausted"] == 1
        assert counters["reactive_live_eligible"] == 1

    def test_gate_failed_increments(self):
        r = dict(_base_result())
        r["reactive_live_reason"] = "live_gate_failed"
        r["has_reactive_candidate"] = True
        counters = _run_loop([_make_paper_row()], [r], live_reactive=True)
        assert counters["reactive_live_gate_failed"] == 1
        assert counters["reactive_live_eligible"] == 1

    def test_live_disabled_not_eligible(self):
        r = dict(_base_result())
        r["reactive_live_reason"] = "live_disabled"
        r["has_reactive_candidate"] = True
        counters = _run_loop([_make_paper_row()], [r])
        assert counters["reactive_live_eligible"] == 0

    def test_multiple_papers_aggregate_correctly(self):
        rows = [_make_paper_row(f"p-{i}") for i in range(4)]
        results = [
            {**_base_result(f"p-{i}"),
             "reactive_status": s,
             "reactive_live_reason": lr,
             "reactive_live_posted": lp,
             "has_reactive_candidate": hc,
             "reactive_draft_created": False,
             "verdict_eligible": False,
             "verdict_draft_created": False}
            for i, (s, lr, lp, hc) in enumerate([
                ("dedup_skipped", "dedup_skipped",        False, True),
                ("live_posted",   "live_posted",          True,  True),
                ("dry_run",       "live_budget_exhausted",False, True),
                ("dry_run",       "live_gate_failed",     False, True),
            ])
        ]
        counters = _run_loop(rows, results, live_reactive=True)
        assert counters["reactive_dedup_skipped"] == 1
        assert counters["reactive_live_posted"] == 1
        assert counters["live_budget_exhausted"] == 1
        assert counters["reactive_live_gate_failed"] == 1
        assert counters["reactive_live_eligible"] == 3  # live_posted + budget_exhausted + gate_failed

    def test_backward_compat_old_result_without_live_fields(self):
        """Counters must not crash when _process_paper omits Phase 9 live fields."""
        old_result = {
            "paper_id": "paper-abc-123",
            "has_reactive_candidate": False,
            "reactive_draft_created": False,
            "verdict_eligible": False,
            "verdict_draft_created": False,
        }
        counters = _run_loop([_make_paper_row()], [old_result])
        assert counters["reactive_dedup_skipped"] == 0
        assert counters["reactive_live_posted"] == 0


# ---------------------------------------------------------------------------
# TestPaperLiveResultsMerge — live fields merged into run summary entries
# ---------------------------------------------------------------------------

class TestPaperLiveResultsMerge:
    """Live fields from _process_paper are merged into build_run_summary entries."""

    def _run_with_summary(self, process_result: dict, summary_entry: dict) -> dict:
        """Run loop and return the summary entry after merge."""
        db = _make_loop_db([_make_paper_row()])
        captured = []

        def _capture_summary(s, path):
            captured.append([dict(e) for e in s])

        with (
            patch(f"{_MOD}._process_paper", return_value=process_result),
            patch(f"{_MOD}.build_run_summary", return_value=[summary_entry]),
            patch(f"{_MOD}.write_run_summary_markdown", side_effect=_capture_summary),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            run_operational_loop(db, _NOW, output_dir="/tmp/rep")

        return captured[0][0] if captured else {}

    def test_live_fields_merged_into_summary(self):
        process_result = {
            **_base_result(),
            "reactive_live_posted": True,
            "reactive_live_reason": "live_posted",
            "reactive_status": "live_posted",
            "verdict_status": "ineligible",
            "reactive_draft_created": False,
            "verdict_eligible": False,
            "verdict_draft_created": False,
        }
        summary_entry = {"paper_id": "paper-abc-123", "title": "T"}

        merged = self._run_with_summary(process_result, summary_entry)
        assert merged["reactive_live_posted"] is True
        assert merged["reactive_live_reason"] == "live_posted"
        assert merged["reactive_status"] == "live_posted"
        assert merged["verdict_status"] == "ineligible"

    def test_no_live_result_summary_entry_unchanged(self):
        """If paper errored (not in paper_live_results), summary entry stays clean."""
        db = _make_loop_db([_make_paper_row()])
        captured = []

        def _side(_paper, *a, **kw):
            raise RuntimeError("boom")

        def _capture(s, path):
            captured.append([dict(e) for e in s])

        with (
            patch(f"{_MOD}._process_paper", side_effect=_side),
            patch(f"{_MOD}.build_run_summary",
                  return_value=[{"paper_id": "paper-abc-123", "title": "T"}]),
            patch(f"{_MOD}.write_run_summary_markdown", side_effect=_capture),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            run_operational_loop(db, _NOW, output_dir="/tmp/rep")

        entry = captured[0][0]
        assert "reactive_live_posted" not in entry

    def test_unprocessed_paper_in_summary_has_no_live_fields(self):
        """Papers in summary but not in processed set (e.g., max_papers cap) get no live fields."""
        rows = [_make_paper_row("p-1"), _make_paper_row("p-2")]
        db = _make_loop_db(rows)
        captured = []

        def _side(paper, *a, **kw):
            return _base_result(paper.paper_id)

        def _capture(s, path):
            captured.append([dict(e) for e in s])

        with (
            patch(f"{_MOD}._process_paper", side_effect=_side),
            patch(f"{_MOD}.build_run_summary",
                  return_value=[
                      {"paper_id": "p-1", "title": "T"},
                      {"paper_id": "p-3", "title": "Other"},
                  ]),
            patch(f"{_MOD}.write_run_summary_markdown", side_effect=_capture),
            patch(f"{_MOD}.write_run_summary_jsonl"),
        ):
            run_operational_loop(db, _NOW, output_dir="/tmp/rep", max_papers=2)

        entries = {e["paper_id"]: e for e in captured[0]}
        assert "reactive_live_posted" in entries["p-1"]
        assert "reactive_live_posted" not in entries["p-3"]


# ---------------------------------------------------------------------------
# TestMarkdownWriter — write_run_summary_markdown live reactive line
# ---------------------------------------------------------------------------

def _minimal_summary_entry(**extra) -> dict:
    return {
        "paper_id": "paper-test-001",
        "title": "A Test Paper",
        "phase": "REVIEW",
        "micro_phase": "BUILD_WINDOW",
        "heat_band": "warm",
        "distinct_citable_other_agents": 2,
        "comment_counts": {"total": 5, "ours": 1, "citable_other": 4},
        "reactive_stats": {"react": 1, "skip": 0, "unclear": 0},
        "strongest_contradiction_confidence": 0.80,
        "has_best_reactive_candidate": True,
        "verdict_eligibility": {"eligible": False, "reason_code": "no_react_signal"},
        "latest_artifact_url": None,
        "recommended_next_action": "consider_reactive_comment",
        **extra,
    }


class TestMarkdownWriterLiveReactiveLine:
    def _write_and_read(self, summary: list, tmp_path: Path) -> str:
        path = tmp_path / "run_summary.md"
        write_run_summary_markdown(summary, path)
        return path.read_text(encoding="utf-8")

    def test_no_live_fields_no_reactive_live_line(self, tmp_path):
        content = self._write_and_read([_minimal_summary_entry()], tmp_path)
        assert "Reactive live:" not in content

    def test_live_posted_true_renders_line(self, tmp_path):
        entry = _minimal_summary_entry(
            reactive_live_posted=True,
            reactive_live_reason="live_posted",
        )
        content = self._write_and_read([entry], tmp_path)
        assert "Reactive live: posted=True, reason=live_posted" in content

    def test_live_posted_false_renders_line(self, tmp_path):
        entry = _minimal_summary_entry(
            reactive_live_posted=False,
            reactive_live_reason="live_disabled",
        )
        content = self._write_and_read([entry], tmp_path)
        assert "Reactive live: posted=False, reason=live_disabled" in content

    def test_live_reason_none_shows_na(self, tmp_path):
        entry = _minimal_summary_entry(
            reactive_live_posted=False,
            reactive_live_reason=None,
        )
        content = self._write_and_read([entry], tmp_path)
        assert "Reactive live: posted=False, reason=n/a" in content

    def test_live_line_appears_between_candidate_and_artifact(self, tmp_path):
        entry = _minimal_summary_entry(
            reactive_live_posted=True,
            reactive_live_reason="live_posted",
        )
        content = self._write_and_read([entry], tmp_path)
        lines = content.splitlines()
        candidate_idx = next(i for i, l in enumerate(lines) if "Best reactive candidate" in l)
        live_idx = next(i for i, l in enumerate(lines) if "Reactive live:" in l)
        artifact_idx = next(i for i, l in enumerate(lines) if "Latest artifact" in l)
        assert candidate_idx < live_idx < artifact_idx

    def test_multiple_papers_only_live_paper_has_line(self, tmp_path):
        entries = [
            _minimal_summary_entry(**{"paper_id": "p-no-live", "title": "NoLive"}),
            _minimal_summary_entry(
                **{"paper_id": "p-live", "title": "Live"},
                reactive_live_posted=True,
                reactive_live_reason="live_posted",
            ),
        ]
        content = self._write_and_read(entries, tmp_path)
        occurrences = content.count("Reactive live:")
        assert occurrences == 1


# ---------------------------------------------------------------------------
# TestJsonlWriterLiveFields — JSONL preserves live fields transparently
# ---------------------------------------------------------------------------

class TestJsonlWriterLiveFields:
    def test_jsonl_includes_live_fields_when_present(self, tmp_path):
        entry = _minimal_summary_entry(
            reactive_live_posted=True,
            reactive_live_reason="live_posted",
            reactive_status="live_posted",
            verdict_status="ineligible",
        )
        path = tmp_path / "out.jsonl"
        write_run_summary_jsonl([entry], path)
        line = json.loads(path.read_text(encoding="utf-8").strip())
        assert line["reactive_live_posted"] is True
        assert line["reactive_live_reason"] == "live_posted"
        assert line["reactive_status"] == "live_posted"
        assert line["verdict_status"] == "ineligible"

    def test_jsonl_backward_compat_no_live_fields(self, tmp_path):
        entry = _minimal_summary_entry()
        path = tmp_path / "out.jsonl"
        write_run_summary_jsonl([entry], path)
        line = json.loads(path.read_text(encoding="utf-8").strip())
        assert "reactive_live_posted" not in line

    def test_jsonl_one_line_per_paper(self, tmp_path):
        entries = [_minimal_summary_entry(**{"paper_id": f"p-{i}", "title": f"P{i}"})
                   for i in range(3)]
        path = tmp_path / "out.jsonl"
        write_run_summary_jsonl(entries, path)
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# TestCLISummaryLine — run_operational_loop CLI print format
# ---------------------------------------------------------------------------

class TestCLISummaryLine:
    """Verify the live observability line is in the counters dict (indirectly tests CLI)."""

    def test_counters_include_all_observability_fields(self):
        counters = _run_loop([_make_paper_row()], [_base_result()])
        required = {
            "reactive_dedup_skipped",
            "reactive_live_eligible",
            "reactive_live_posted",
            "live_budget_exhausted",
            "reactive_live_gate_failed",
        }
        assert required.issubset(counters.keys())

    def test_live_observability_line_format(self, capsys, tmp_path):
        """main() prints the live observability line with correct keys."""
        from gsr_agent.orchestration.operational_loop import main

        mock_counters = {
            "papers_seen": 2,
            "papers_processed": 2,
            "reactive_drafts_created": 1,
            "live_reactive_posts": 1,
            "verdict_drafts_created": 0,
            "live_verdict_submissions": 0,
            "verdict_live_missing_score": 0,
            "verdict_live_invalid_score": 0,
            "errors_count": 0,
            "reactive_live_eligible": 2,
            "reactive_dedup_skipped": 0,
            "live_budget_exhausted": 1,
            "reactive_live_gate_failed": 0,
            "summary_path": str(tmp_path / "run_summary.md"),
        }
        mock_db = MagicMock()
        mock_db.close = MagicMock()

        # KoalaDB is imported locally inside main(); patch at source.
        with (
            patch("gsr_agent.storage.db.KoalaDB", return_value=mock_db),
            patch(f"{_MOD}.run_operational_loop", return_value=mock_counters),
            patch("sys.argv", ["loop"]),
        ):
            main()

        out = capsys.readouterr().out
        assert "live_reactive_posts=1" in out
        assert "eligible=2" in out
        assert "budget_exhausted=1" in out
        assert "gate_failed=0" in out
