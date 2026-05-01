"""Phase 7: Run Summary — build_paper_summary, build_run_summary, writers."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from gsr_agent.reporting.run_summary import (
    build_paper_summary,
    build_run_summary,
    write_run_summary_jsonl,
    write_run_summary_markdown,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 26, 14, 0, 0, tzinfo=timezone.utc)  # 14 h after open → BUILD_WINDOW
_OPEN = _NOW - timedelta(hours=14)
_PAPER_ID = "paper-p7-001"


def _make_paper_row(
    paper_id: str = _PAPER_ID,
    title: str = "Test Paper Title",
    open_time: Optional[datetime] = None,
) -> dict:
    ot = open_time or _OPEN
    return {
        "paper_id": paper_id,
        "title": title,
        "open_time": ot.isoformat(),
        "review_end_time": (ot + timedelta(hours=48)).isoformat(),
        "verdict_end_time": (ot + timedelta(hours=72)).isoformat(),
        "state": "REVIEW_ACTIVE",
        "pdf_url": "",
    }


def _make_db(
    *,
    citable_other: int = 3,
    distinct_agents: int = 3,
    react_count: int = 1,
    skip_count: int = 0,
    unclear_count: int = 0,
    comments_analyzed: int = 1,
    strongest_conf: Optional[float] = 0.80,
    latest_action: Optional[dict] = None,
    papers: Optional[list] = None,
) -> MagicMock:
    db = MagicMock()
    db.get_papers.return_value = papers if papers is not None else [_make_paper_row()]
    db.get_comment_stats.return_value = {
        "total": citable_other + 2,
        "ours": 2,
        "citable_other": citable_other,
    }
    db.get_distinct_other_agent_count.return_value = distinct_agents
    db.get_phase5a_stats.return_value = {
        "comments_analyzed": comments_analyzed,
        "claims_extracted": react_count * 2,
        "claims_verified": react_count * 2,
        "react_count": react_count,
        "skip_count": skip_count,
        "unclear_count": unclear_count,
        "contradicted_count": react_count,
        "supported_count": skip_count,
        "insufficient_count": 0,
        "verification_error_count": 0,
    }
    db.get_strongest_contradiction_confidence.return_value = strongest_conf
    db.get_latest_action_for_paper.return_value = latest_action
    return db


def _default_summary(db: Optional[MagicMock] = None) -> dict:
    return build_paper_summary(_make_paper_row(), db or _make_db(), _NOW)


# ---------------------------------------------------------------------------
# A. build_paper_summary — required fields and values
# ---------------------------------------------------------------------------

class TestBuildPaperSummary:
    _REQUIRED_KEYS = {
        "paper_id", "title", "phase", "micro_phase", "heat_band",
        "distinct_citable_other_agents", "comment_counts", "reactive_stats",
        "strongest_contradiction_confidence", "has_best_reactive_candidate",
        "verdict_eligibility", "latest_artifact_url", "recommended_next_action",
    }

    def test_has_all_required_keys(self):
        s = _default_summary()
        assert self._REQUIRED_KEYS <= s.keys()

    def test_paper_id_matches(self):
        assert _default_summary()["paper_id"] == _PAPER_ID

    def test_title_matches(self):
        assert _default_summary()["title"] == "Test Paper Title"

    def test_phase_review_active(self):
        assert _default_summary()["phase"] == "REVIEW_ACTIVE"

    def test_micro_phase_build_window(self):
        assert _default_summary()["micro_phase"] == "BUILD_WINDOW"

    def test_heat_band_from_distinct_agents(self):
        db = _make_db(distinct_agents=5)
        s = build_paper_summary(_make_paper_row(), db, _NOW)
        assert s["heat_band"] == "goldilocks"

    def test_distinct_citable_other_agents(self):
        db = _make_db(distinct_agents=4)
        s = build_paper_summary(_make_paper_row(), db, _NOW)
        assert s["distinct_citable_other_agents"] == 4

    def test_comment_counts_has_required_subkeys(self):
        c = _default_summary()["comment_counts"]
        assert {"total", "ours", "citable_other"} <= c.keys()

    def test_comment_counts_values(self):
        db = _make_db(citable_other=5)
        c = build_paper_summary(_make_paper_row(), db, _NOW)["comment_counts"]
        assert c["citable_other"] == 5
        assert c["ours"] == 2

    def test_reactive_stats_has_required_subkeys(self):
        r = _default_summary()["reactive_stats"]
        assert {"react", "skip", "unclear"} <= r.keys()

    def test_reactive_stats_react_count(self):
        db = _make_db(react_count=3, skip_count=1, unclear_count=2)
        r = build_paper_summary(_make_paper_row(), db, _NOW)["reactive_stats"]
        assert r["react"] == 3
        assert r["skip"] == 1
        assert r["unclear"] == 2

    def test_strongest_contradiction_confidence_present(self):
        db = _make_db(strongest_conf=0.88)
        s = build_paper_summary(_make_paper_row(), db, _NOW)
        assert s["strongest_contradiction_confidence"] == pytest.approx(0.88)

    def test_strongest_contradiction_confidence_none(self):
        db = _make_db(strongest_conf=None)
        s = build_paper_summary(_make_paper_row(), db, _NOW)
        assert s["strongest_contradiction_confidence"] is None

    def test_has_react_candidate_true(self):
        db = _make_db(react_count=1)
        assert build_paper_summary(_make_paper_row(), db, _NOW)["has_best_reactive_candidate"] is True

    def test_has_react_candidate_false_when_zero(self):
        db = _make_db(react_count=0, strongest_conf=None)
        assert build_paper_summary(_make_paper_row(), db, _NOW)["has_best_reactive_candidate"] is False

    def test_verdict_eligibility_has_eligible_and_reason_code(self):
        v = _default_summary()["verdict_eligibility"]
        assert "eligible" in v
        assert "reason_code" in v

    def test_verdict_eligible_true(self):
        db = _make_db(distinct_agents=3, strongest_conf=0.80, react_count=1)
        v = build_paper_summary(_make_paper_row(), db, _NOW)["verdict_eligibility"]
        assert v["eligible"] is True
        assert v["reason_code"] == "eligible"

    def test_verdict_not_eligible_no_signal(self):
        db = _make_db(distinct_agents=3, strongest_conf=None, react_count=0)
        v = build_paper_summary(_make_paper_row(), db, _NOW)["verdict_eligibility"]
        assert v["eligible"] is False
        assert v["reason_code"] == "no_react_signal"

    def test_latest_artifact_url_from_action(self):
        action = {"github_file_url": "https://github.com/test/url.md", "status": "dry_run"}
        db = _make_db(latest_action=action)
        s = build_paper_summary(_make_paper_row(), db, _NOW)
        assert s["latest_artifact_url"] == "https://github.com/test/url.md"

    def test_latest_artifact_url_none_when_no_action(self):
        db = _make_db(latest_action=None)
        s = build_paper_summary(_make_paper_row(), db, _NOW)
        assert s["latest_artifact_url"] is None

    def test_recommended_next_action_is_valid(self):
        from gsr_agent.reporting.run_summary import _VALID_ACTIONS
        assert _default_summary()["recommended_next_action"] in _VALID_ACTIONS

    def test_missing_open_time_graceful(self):
        row = {"paper_id": "p-x", "title": "", "open_time": "", "state": "NEW"}
        db = _make_db()
        s = build_paper_summary(row, db, _NOW)
        assert s["phase"] == "UNKNOWN"
        assert s["micro_phase"] == "UNKNOWN"

    def test_missing_title_graceful(self):
        row = _make_paper_row()
        row["title"] = None
        db = _make_db()
        s = build_paper_summary(row, db, _NOW)
        assert s["title"] == ""


# ---------------------------------------------------------------------------
# B. recommended_next_action — heat-band × signal combinations
# ---------------------------------------------------------------------------

class TestRecommendedNextAction:
    def _summary(self, **kwargs) -> str:
        from unittest.mock import patch
        db = _make_db(**kwargs)
        with patch("gsr_agent.reporting.run_summary.is_aggressive_mode", return_value=False):
            return build_paper_summary(_make_paper_row(), db, _NOW)["recommended_next_action"]

    def test_goldilocks_react_candidate_consider_reactive(self):
        action = self._summary(distinct_agents=2, react_count=1, strongest_conf=0.70, comments_analyzed=1)
        assert action == "consider_reactive_comment"

    def test_verdict_eligible_consider_verdict_draft(self):
        action = self._summary(distinct_agents=3, react_count=1, strongest_conf=0.80, comments_analyzed=1)
        assert action == "consider_verdict_draft"

    def test_cold_no_signal_skip_too_cold(self):
        action = self._summary(
            distinct_agents=0, citable_other=0, react_count=0,
            strongest_conf=None, comments_analyzed=0,
        )
        assert action == "skip_too_cold"

    def test_saturated_skip_too_crowded(self):
        action = self._summary(
            distinct_agents=15, citable_other=15, react_count=0,
            strongest_conf=None, comments_analyzed=1,
        )
        assert action == "skip_too_crowded"

    def test_goldilocks_no_signal_analysis_done_skip_no_signal(self):
        action = self._summary(
            distinct_agents=2, citable_other=3, react_count=0,
            strongest_conf=None, comments_analyzed=2,
        )
        assert action == "skip_no_signal"

    def test_warm_no_analysis_run_reactive_analysis(self):
        action = self._summary(
            distinct_agents=1, citable_other=2, react_count=0,
            strongest_conf=None, comments_analyzed=0,
        )
        assert action == "run_reactive_analysis"

    def test_crowded_strong_override_eligible_verdict_draft(self):
        action = self._summary(
            distinct_agents=4, citable_other=4, react_count=1,
            strongest_conf=0.80, comments_analyzed=1,
        )
        assert action == "consider_verdict_draft"

    def test_crowded_no_override_no_signal_skip_no_signal(self):
        action = self._summary(
            distinct_agents=5, citable_other=5, react_count=0,
            strongest_conf=None, comments_analyzed=1,
        )
        assert action == "skip_no_signal"

    def test_react_candidate_beats_run_reactive_analysis(self):
        action = self._summary(
            distinct_agents=1, citable_other=2, react_count=1,
            strongest_conf=0.60, comments_analyzed=0,
        )
        assert action == "consider_reactive_comment"

    def test_verdict_draft_beats_reactive_comment(self):
        action = self._summary(
            distinct_agents=3, react_count=1, strongest_conf=0.80, comments_analyzed=1
        )
        assert action == "consider_verdict_draft"

    def test_review_active_seed_window_5_comments_consider_seed(self):
        """REVIEW_ACTIVE+SEED_WINDOW + 5 total comments → consider_seed_comment."""
        row = _make_paper_row(open_time=_NOW - timedelta(hours=6))
        db = MagicMock()
        db.get_comment_stats.return_value = {"total": 5, "ours": 0, "citable_other": 0}
        db.get_distinct_other_agent_count.return_value = 0
        db.get_phase5a_stats.return_value = {
            "comments_analyzed": 0, "claims_extracted": 0, "claims_verified": 0,
            "react_count": 0, "skip_count": 0, "unclear_count": 0,
            "contradicted_count": 0, "supported_count": 0,
            "insufficient_count": 0, "verification_error_count": 0,
        }
        db.get_strongest_contradiction_confidence.return_value = None
        db.get_latest_action_for_paper.return_value = None
        s = build_paper_summary(row, db, _NOW)
        assert s["micro_phase"] == "SEED_WINDOW"
        assert s["phase"] == "REVIEW_ACTIVE"
        assert s["recommended_next_action"] == "consider_seed_comment"


# ---------------------------------------------------------------------------
# C. build_run_summary — iteration and filtering
# ---------------------------------------------------------------------------

class TestBuildRunSummary:
    def test_returns_list(self):
        db = _make_db()
        result = build_run_summary(db, _NOW)
        assert isinstance(result, list)

    def test_one_entry_per_paper(self):
        papers = [_make_paper_row("p1"), _make_paper_row("p2"), _make_paper_row("p3")]
        db = _make_db(papers=papers)
        result = build_run_summary(db, _NOW)
        assert len(result) == 3

    def test_paper_ids_passed_to_db(self):
        db = _make_db(papers=[_make_paper_row()])
        build_run_summary(db, _NOW, paper_ids=["p1", "p2"])
        db.get_papers.assert_called_once_with(["p1", "p2"])

    def test_none_paper_ids_returns_all(self):
        db = _make_db(papers=[_make_paper_row()])
        build_run_summary(db, _NOW, paper_ids=None)
        db.get_papers.assert_called_once_with(None)

    def test_defaults_now_to_utcnow(self):
        db = _make_db()
        before = datetime.now(timezone.utc)
        result = build_run_summary(db)
        after = datetime.now(timezone.utc)
        assert result[0]["phase"] in {
            "REVIEW_ACTIVE", "VERDICT_ACTIVE", "EXPIRED", "NEW", "UNKNOWN"
        }
        _ = before, after  # consumed

    def test_empty_db_returns_empty_list(self):
        db = _make_db(papers=[])
        assert build_run_summary(db, _NOW) == []

    def test_summary_entries_have_paper_ids(self):
        papers = [_make_paper_row("pa"), _make_paper_row("pb")]
        db = _make_db(papers=papers)
        result = build_run_summary(db, _NOW)
        ids = {s["paper_id"] for s in result}
        assert ids == {"pa", "pb"}


# ---------------------------------------------------------------------------
# D. write_run_summary_markdown
# ---------------------------------------------------------------------------

class TestWriteRunSummaryMarkdown:
    def _run(self, tmp_path: Path, **db_kwargs) -> tuple[Path, str]:
        db = _make_db(**db_kwargs)
        summary = build_run_summary(db, _NOW)
        out = tmp_path / "reports" / "run_summary.md"
        write_run_summary_markdown(summary, out)
        return out, out.read_text(encoding="utf-8")

    def test_file_created(self, tmp_path: Path):
        path, _ = self._run(tmp_path)
        assert path.exists()

    def test_parent_dir_created_automatically(self, tmp_path: Path):
        db = _make_db()
        summary = build_run_summary(db, _NOW)
        nested = tmp_path / "a" / "b" / "c" / "report.md"
        write_run_summary_markdown(summary, nested)
        assert nested.exists()

    def test_contains_header(self, tmp_path: Path):
        _, text = self._run(tmp_path)
        assert "# GSR Agent Run Summary" in text

    def test_contains_paper_id(self, tmp_path: Path):
        _, text = self._run(tmp_path)
        assert _PAPER_ID in text

    def test_contains_heat_band(self, tmp_path: Path):
        _, text = self._run(tmp_path, distinct_agents=5)
        assert "goldilocks" in text

    def test_contains_recommended_action(self, tmp_path: Path):
        _, text = self._run(tmp_path, react_count=1, distinct_agents=3, strongest_conf=0.80)
        assert "consider_verdict_draft" in text

    def test_contains_reactive_stats(self, tmp_path: Path):
        _, text = self._run(tmp_path, react_count=2, skip_count=1)
        assert "react=2" in text
        assert "skip=1" in text

    def test_confidence_formatted(self, tmp_path: Path):
        _, text = self._run(tmp_path, strongest_conf=0.83)
        assert "0.83" in text

    def test_no_confidence_shows_na(self, tmp_path: Path):
        _, text = self._run(tmp_path, strongest_conf=None, react_count=0)
        assert "n/a" in text

    def test_multiple_papers_each_section_present(self, tmp_path: Path):
        papers = [_make_paper_row("px"), _make_paper_row("py")]
        db = _make_db(papers=papers)
        summary = build_run_summary(db, _NOW)
        out = tmp_path / "r.md"
        write_run_summary_markdown(summary, out)
        text = out.read_text()
        assert "px" in text
        assert "py" in text

    def test_artifact_none_shows_none_label(self, tmp_path: Path):
        _, text = self._run(tmp_path, latest_action=None)
        assert "(none)" in text

    def test_artifact_url_shown_when_present(self, tmp_path: Path):
        action = {"github_file_url": "https://github.com/test/url.md"}
        _, text = self._run(tmp_path, latest_action=action)
        assert "https://github.com/test/url.md" in text


# ---------------------------------------------------------------------------
# E. write_run_summary_jsonl
# ---------------------------------------------------------------------------

class TestWriteRunSummaryJsonl:
    def _run(self, tmp_path: Path, **db_kwargs) -> tuple[Path, list]:
        db = _make_db(**db_kwargs)
        summary = build_run_summary(db, _NOW)
        out = tmp_path / "run_summary.jsonl"
        write_run_summary_jsonl(summary, out)
        lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        return out, lines

    def test_file_created(self, tmp_path: Path):
        path, _ = self._run(tmp_path)
        assert path.exists()

    def test_one_line_per_paper(self, tmp_path: Path):
        papers = [_make_paper_row("q1"), _make_paper_row("q2")]
        db = _make_db(papers=papers)
        summary = build_run_summary(db, _NOW)
        out = tmp_path / "out.jsonl"
        write_run_summary_jsonl(summary, out)
        lines = out.read_text().splitlines()
        assert len([l for l in lines if l.strip()]) == 2

    def test_each_line_is_valid_json(self, tmp_path: Path):
        _, lines = self._run(tmp_path)
        assert len(lines) >= 1

    def test_json_contains_paper_id(self, tmp_path: Path):
        _, lines = self._run(tmp_path)
        assert lines[0]["paper_id"] == _PAPER_ID

    def test_json_contains_recommended_action(self, tmp_path: Path):
        _, lines = self._run(tmp_path)
        assert "recommended_next_action" in lines[0]

    def test_empty_summary_creates_empty_file(self, tmp_path: Path):
        db = _make_db(papers=[])
        write_run_summary_jsonl([], tmp_path / "empty.jsonl")
        assert (tmp_path / "empty.jsonl").read_text() == ""

    def test_parent_dir_created_automatically(self, tmp_path: Path):
        db = _make_db()
        summary = build_run_summary(db, _NOW)
        nested = tmp_path / "deep" / "path" / "out.jsonl"
        write_run_summary_jsonl(summary, nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# F. heat-band correctness in summary
# ---------------------------------------------------------------------------

class TestHeatBandInSummary:
    def _band(self, distinct: int) -> str:
        db = _make_db(distinct_agents=distinct)
        return build_paper_summary(_make_paper_row(), db, _NOW)["heat_band"]

    def test_cold_0(self):
        assert self._band(0) == "cold"

    def test_warm_1(self):
        assert self._band(1) == "warm"

    def test_warm_2(self):
        assert self._band(2) == "warm"

    def test_goldilocks_3(self):
        assert self._band(3) == "goldilocks"

    def test_goldilocks_4(self):
        assert self._band(4) == "goldilocks"

    def test_goldilocks_6(self):
        assert self._band(6) == "goldilocks"

    def test_goldilocks_7(self):
        assert self._band(7) == "goldilocks"

    def test_crowded_11(self):
        assert self._band(11) == "crowded"

    def test_crowded_14(self):
        assert self._band(14) == "crowded"

    def test_saturated_15(self):
        assert self._band(15) == "saturated"

    def test_saturated_large(self):
        assert self._band(100) == "saturated"


    def test_seed_live_status_shown_when_present(self, tmp_path: Path):
        """Seed live status line is displayed when seed fields are present in summary entry."""
        db = _make_db()
        summary = build_run_summary(db, _NOW)
        # Inject seed fields (as done by run_operational_loop via paper_live_results)
        for entry in summary:
            entry["seed_draft_created"] = True
            entry["seed_live_posted"] = False
            entry["seed_live_reason"] = "draft_created"
        out = tmp_path / "seed_summary.md"
        write_run_summary_markdown(summary, out)
        text = out.read_text(encoding="utf-8")
        assert "Seed live:" in text
        assert "draft=True" in text

    def test_seed_live_posted_reason_shown(self, tmp_path: Path):
        """Seed live line shows reason=live_posted when seed was live-posted."""
        db = _make_db()
        summary = build_run_summary(db, _NOW)
        for entry in summary:
            entry["seed_draft_created"] = False
            entry["seed_live_posted"] = True
            entry["seed_live_reason"] = "live_posted"
        out = tmp_path / "seed_live_summary.md"
        write_run_summary_markdown(summary, out)
        text = out.read_text(encoding="utf-8")
        assert "Seed live:" in text
        assert "posted=True" in text
        assert "reason=live_posted" in text


# ---------------------------------------------------------------------------
# G. Aggressive-mode overrides (KOALA_AGGRESSIVE_FINAL_24H=1)
# ---------------------------------------------------------------------------

from gsr_agent.reporting.run_summary import _compute_verdict_eligibility, _recommended_action  # noqa: E402


class TestAggressiveModeRunSummary:
    """In aggressive mode, saturated/crowded papers must not be labelled skip_too_crowded
    or show saturated_low_value_v0 / crowded_no_override as verdict reason."""

    # --- _compute_verdict_eligibility ---

    def test_saturated_normal_returns_saturated_low_value(self):
        """Normal mode baseline: saturated → saturated_low_value_v0."""
        eligible, code = _compute_verdict_eligibility("saturated", 0.90, 15, aggressive=False)
        assert eligible is False
        assert code == "saturated_low_value_v0"

    def test_saturated_aggressive_sufficient_citations_eligible(self):
        """Aggressive mode: saturated with >=3 distinct agents → eligible."""
        eligible, code = _compute_verdict_eligibility("saturated", None, 15, aggressive=True)
        assert eligible is True
        assert code == "eligible"

    def test_saturated_aggressive_insufficient_citations_not_eligible(self):
        """Aggressive mode: saturated with <3 distinct agents → citation gate fails."""
        eligible, code = _compute_verdict_eligibility("saturated", None, 2, aggressive=True)
        assert eligible is False
        assert code == "insufficient_distinct_other_agent_citations"

    def test_crowded_no_signal_normal_returns_crowded_no_override(self):
        """Normal mode baseline: crowded + no strong signal → crowded_no_override."""
        eligible, code = _compute_verdict_eligibility("crowded", None, 12, aggressive=False)
        assert eligible is False
        assert code == "crowded_no_override"

    def test_crowded_no_signal_aggressive_sufficient_citations_eligible(self):
        """Aggressive mode: crowded + no strong signal + >=3 agents → eligible."""
        eligible, code = _compute_verdict_eligibility("crowded", None, 12, aggressive=True)
        assert eligible is True
        assert code == "eligible"

    def test_crowded_no_signal_aggressive_insufficient_citations_not_eligible(self):
        """Aggressive mode: crowded + no strong signal + <3 agents → citation gate fails."""
        eligible, code = _compute_verdict_eligibility("crowded", None, 1, aggressive=True)
        assert eligible is False
        assert code == "insufficient_distinct_other_agent_citations"

    # --- _recommended_action for saturated papers ---

    def test_saturated_normal_returns_skip_too_crowded(self):
        """Normal mode baseline: saturated paper → skip_too_crowded."""
        action = _recommended_action(
            verdict_eligible=False,
            has_react_candidate=False,
            heat_band="saturated",
            citable_other=15,
            comments_analyzed=1,
            aggressive=False,
        )
        assert action == "skip_too_crowded"

    def test_saturated_aggressive_with_own_comment_and_citations_seed_or_verdict(self):
        """Aggressive mode: saturated + own comment + citable>=3 → seed_or_verdict_candidate."""
        action = _recommended_action(
            verdict_eligible=False,
            has_react_candidate=False,
            heat_band="saturated",
            citable_other=15,
            comments_analyzed=1,
            aggressive=True,
            has_own_comment=True,
        )
        assert action == "seed_or_verdict_candidate"

    def test_saturated_aggressive_no_own_comment_not_eligible(self):
        """Aggressive mode: saturated + no own comment → not_eligible_no_own_comment."""
        action = _recommended_action(
            verdict_eligible=False,
            has_react_candidate=False,
            heat_band="saturated",
            citable_other=15,
            comments_analyzed=1,
            aggressive=True,
            has_own_comment=False,
        )
        assert action == "not_eligible_no_own_comment"

    def test_saturated_aggressive_not_enough_citations_not_eligible_window(self):
        """Aggressive mode: saturated + own comment but <3 citable → not_eligible_window."""
        action = _recommended_action(
            verdict_eligible=False,
            has_react_candidate=False,
            heat_band="saturated",
            citable_other=2,
            comments_analyzed=1,
            aggressive=True,
            has_own_comment=True,
        )
        assert action == "not_eligible_window"

    # --- build_paper_summary end-to-end with env override ---

    def test_saturated_paper_normal_mode_skip_too_crowded(self):
        """Normal mode: high-comment paper → recommended_action=skip_too_crowded."""
        from unittest.mock import patch
        row = _make_paper_row()
        db = _make_db(distinct_agents=15, citable_other=15, react_count=0, strongest_conf=None)
        with patch("gsr_agent.reporting.run_summary.is_aggressive_mode", return_value=False):
            s = build_paper_summary(row, db, _NOW)
        assert s["recommended_next_action"] == "skip_too_crowded"
        assert s["verdict_eligibility"]["reason_code"] == "saturated_low_value_v0"

    def test_saturated_paper_aggressive_mode_not_skip_too_crowded(self):
        """Aggressive mode: saturated paper with 15 distinct agents → verdict eligible,
        recommended action is consider_verdict_draft (not skip_too_crowded)."""
        from unittest.mock import patch
        row = _make_paper_row()
        db = _make_db(distinct_agents=15, citable_other=15, react_count=0, strongest_conf=None)
        with patch("gsr_agent.reporting.run_summary.is_aggressive_mode", return_value=True):
            s = build_paper_summary(row, db, _NOW)
        assert s["recommended_next_action"] != "skip_too_crowded"
        assert s["recommended_next_action"] == "consider_verdict_draft"
        assert s["verdict_eligibility"]["eligible"] is True

    def test_saturated_paper_aggressive_mode_verdict_eligible_shows_consider_verdict(self):
        """Aggressive mode: saturated + strong signal + >=3 agents → consider_verdict_draft."""
        from unittest.mock import patch
        row = _make_paper_row()
        db = _make_db(distinct_agents=15, citable_other=15, react_count=1, strongest_conf=0.90)
        with patch("gsr_agent.reporting.run_summary.is_aggressive_mode", return_value=True):
            s = build_paper_summary(row, db, _NOW)
        assert s["recommended_next_action"] == "consider_verdict_draft"
        assert s["verdict_eligibility"]["eligible"] is True
