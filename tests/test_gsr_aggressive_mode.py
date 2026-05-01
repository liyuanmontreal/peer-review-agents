"""Tests for KOALA_AGGRESSIVE_FINAL_24H mode."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

_NOW = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
_MOD = "gsr_agent.orchestration.operational_loop"


def _row(paper_id: str, hours_ago: float = 20.0, state: str = "REVIEW_ACTIVE") -> dict:
    open_time = _NOW - timedelta(hours=hours_ago)
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "open_time": open_time.isoformat(),
        "review_end_time": (open_time + timedelta(hours=48)).isoformat(),
        "verdict_end_time": (open_time + timedelta(hours=72)).isoformat(),
        "state": state,
        "pdf_url": "",
        "local_pdf_path": None,
    }


def _make_db(rows, *, citable_other: int = 0, ours: int = 0, has_participated: bool = False):
    db = MagicMock()
    db.get_papers.return_value = rows
    db.get_comment_stats.return_value = {
        "total": citable_other + ours,
        "ours": ours,
        "citable_other": citable_other,
    }
    db.get_distinct_other_agent_count.return_value = citable_other
    db.has_prior_participation.return_value = has_participated
    db.has_recent_reactive_action_for_comment.return_value = False
    db.has_recent_verdict_action_for_paper.return_value = False
    db.has_recent_seed_action_for_paper.return_value = False
    db.get_reactive_analysis_results.return_value = []
    db.get_citable_other_comments_for_paper.return_value = []
    db.get_latest_action_for_paper.return_value = None
    db.get_phase5a_stats.return_value = {
        "react_count": 0, "skip_count": 0, "unclear_count": 0,
        "comments_analyzed": 0, "claims_extracted": 0, "claims_verified": 0,
        "contradicted_count": 0, "supported_count": 0, "unclear_v_count": 0,
    }
    db.get_strongest_contradiction_confidence.return_value = None
    return db


# ---------------------------------------------------------------------------
# aggressive_mode module
# ---------------------------------------------------------------------------

def test_is_aggressive_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("KOALA_AGGRESSIVE_FINAL_24H", raising=False)
    from gsr_agent.strategy.aggressive_mode import is_aggressive_mode
    assert is_aggressive_mode() is False


def test_is_aggressive_mode_on_when_set(monkeypatch):
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    from gsr_agent.strategy.aggressive_mode import is_aggressive_mode
    assert is_aggressive_mode() is True


def test_aggressive_budgets_are_larger():
    from gsr_agent.strategy.aggressive_mode import (
        AGGRESSIVE_CANDIDATE_BUDGET,
        AGGRESSIVE_LIVE_COMMENT_BUDGET,
        AGGRESSIVE_LIVE_VERDICT_BUDGET,
    )
    from gsr_agent.strategy.opportunity_manager import CANDIDATE_BUDGET
    assert AGGRESSIVE_LIVE_COMMENT_BUDGET > 3
    assert AGGRESSIVE_LIVE_VERDICT_BUDGET > 2
    assert AGGRESSIVE_CANDIDATE_BUDGET > CANDIDATE_BUDGET


# ---------------------------------------------------------------------------
# SKIP → SEED promotion for non-participated papers in aggressive mode
# ---------------------------------------------------------------------------

def test_aggressive_mode_promotes_skip_to_seed_for_active_thread(monkeypatch, caplog):
    """Non-participated paper past SEED_WINDOW with citable_other >= 2 → promoted to SEED."""
    import logging
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    # Paper is 20h old (past SEED_WINDOW at 12h) with 3 citable others
    db = _make_db([_row("p-001", hours_ago=20)], citable_other=3, has_participated=False)

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        run_operational_loop(db, _NOW, test_mode=True)

    assert any("promoted_seed" in r.message and "p-001" in r.message for r in caplog.records)


def test_normal_mode_does_not_promote_skip_past_seed_window(monkeypatch, caplog):
    """Without aggressive mode, non-participated papers past SEED_WINDOW stay SKIP."""
    import logging
    monkeypatch.delenv("KOALA_AGGRESSIVE_FINAL_24H", raising=False)
    db = _make_db([_row("p-001", hours_ago=20)], citable_other=3, has_participated=False)

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        run_operational_loop(db, _NOW, test_mode=True)

    assert not any("promoted_seed" in r.message for r in caplog.records)


def test_aggressive_mode_does_not_promote_low_citable(monkeypatch, caplog):
    """Papers with citable_other < 2 should NOT be promoted even in aggressive mode."""
    import logging
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    db = _make_db([_row("p-001", hours_ago=20)], citable_other=1, has_participated=False)

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        run_operational_loop(db, _NOW, test_mode=True)

    assert not any("promoted_seed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Funnel logging
# ---------------------------------------------------------------------------

def test_aggressive_funnel_log_emitted(monkeypatch, caplog):
    """Funnel summary lines are logged in aggressive mode."""
    import logging
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    db = _make_db([_row("p-001"), _row("p-002")], citable_other=2)

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        run_operational_loop(db, _NOW, test_mode=True)

    assert any("aggressive_funnel" in r.message for r in caplog.records)


def test_funnel_log_absent_in_normal_mode(monkeypatch, caplog):
    """No aggressive_funnel log in normal mode."""
    import logging
    monkeypatch.delenv("KOALA_AGGRESSIVE_FINAL_24H", raising=False)
    db = _make_db([_row("p-001")], citable_other=2)

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        run_operational_loop(db, _NOW, test_mode=True)

    assert not any("aggressive_funnel" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Candidate budget is raised in aggressive mode
# ---------------------------------------------------------------------------

def test_aggressive_mode_processes_more_candidates(monkeypatch, caplog):
    """In aggressive mode, candidate budget is raised above default 8."""
    import logging
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    # 15 seed papers (all in SEED_WINDOW, hours_ago=6)
    rows = [_row(f"p-{i:03d}", hours_ago=6) for i in range(15)]
    db = _make_db(rows, citable_other=3, has_participated=False)

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        counters = run_operational_loop(db, _NOW, test_mode=True)

    # Should process more than 8 papers (the default CANDIDATE_BUDGET)
    assert counters["papers_processed"] > 8


# ---------------------------------------------------------------------------
# Reactive-first path logging
# ---------------------------------------------------------------------------

def _make_candidate(conf: float = 0.5):
    from gsr_agent.commenting.reactive_analysis import ReactiveAnalysisResult
    return ReactiveAnalysisResult(
        comment_id="c-001",
        paper_id="p-001",
        recommendation="react",
        verifications=[{"verdict": "refuted", "confidence": conf, "claim_id": "cl-1"}],
        draft_text="[DRY-RUN — not posted]\nClaim: X is true.\nVerdict: refuted (confidence 0.50)\n",
    )


def test_aggressive_weak_signal_not_suppressed(monkeypatch, caplog):
    """Already-participated paper, weak reactive signal → NOT suppressed in aggressive mode.

    The repeat-depth suppression gate is bypassed: any decision=post candidate is
    selected immediately and proceeds to plan_and_post_reactive_comment.
    """
    import logging
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    weak_candidate = _make_candidate(conf=0.4)
    db = _make_db([_row("p-001", hours_ago=20)], citable_other=3, ours=1, has_participated=True)
    monkeypatch.setattr(
        "gsr_agent.orchestration.operational_loop.analyze_reactive_candidates_for_paper",
        lambda *a, **kw: [weak_candidate],
    )
    monkeypatch.setattr(
        "gsr_agent.orchestration.operational_loop.select_best_reactive_candidate_for_paper",
        lambda *a, **kw: weak_candidate,
    )

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        run_operational_loop(db, _NOW, test_mode=True)

    assert not any("repeat_depth_suppressed" in r.message for r in caplog.records)
    assert any("reactive_selection" in r.message and "c-001" in r.message for r in caplog.records)


def test_aggressive_repeat_depth_strong_signal_allows_comment(monkeypatch, caplog):
    """Already-participated paper, strong contradiction signal → NOT suppressed, logs reactive_short."""
    import logging
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    strong_candidate = _make_candidate(conf=0.8)
    db = _make_db([_row("p-001", hours_ago=20)], citable_other=3, ours=1, has_participated=True)
    monkeypatch.setattr(
        "gsr_agent.orchestration.operational_loop.analyze_reactive_candidates_for_paper",
        lambda *a, **kw: [strong_candidate],
    )
    monkeypatch.setattr(
        "gsr_agent.orchestration.operational_loop.select_best_reactive_candidate_for_paper",
        lambda *a, **kw: strong_candidate,
    )

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        run_operational_loop(db, _NOW, test_mode=True)

    assert not any("repeat_depth_suppressed" in r.message for r in caplog.records)
    assert any("comment_path=reactive_short" in r.message for r in caplog.records)


def test_aggressive_comment_path_reactive_short_logged_no_prior(monkeypatch, caplog):
    """No prior participation + reactive candidate → logs comment_path=reactive_short."""
    import logging
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    candidate = _make_candidate(conf=0.6)
    db = _make_db([_row("p-001", hours_ago=10)], citable_other=2, ours=0, has_participated=False)
    monkeypatch.setattr(
        "gsr_agent.orchestration.operational_loop.analyze_reactive_candidates_for_paper",
        lambda *a, **kw: [candidate],
    )
    monkeypatch.setattr(
        "gsr_agent.orchestration.operational_loop.select_best_reactive_candidate_for_paper",
        lambda *a, **kw: candidate,
    )

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        run_operational_loop(db, _NOW, test_mode=True)

    assert any("comment_path=reactive_short" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Verdict path logging
# ---------------------------------------------------------------------------

def test_aggressive_verdict_path_lightweight_logged(monkeypatch, caplog):
    """In aggressive mode, verdict eligibility check logs verdict_path=lightweight."""
    import logging
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    db = _make_db(
        [_row("p-001", hours_ago=49, state="VERDICT_ACTIVE")],
        citable_other=3, ours=1, has_participated=True,
    )

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        run_operational_loop(db, _NOW, test_mode=True)

    assert any("verdict_path=lightweight" in r.message for r in caplog.records)


def test_normal_mode_no_aggressive_path_logs(monkeypatch, caplog):
    """In normal mode, no [aggressive_mode] comment_path or verdict_path logs are emitted."""
    import logging
    monkeypatch.delenv("KOALA_AGGRESSIVE_FINAL_24H", raising=False)
    db = _make_db([_row("p-001", hours_ago=20)], citable_other=3, ours=1, has_participated=True)

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        run_operational_loop(db, _NOW, test_mode=True)

    assert not any(
        "comment_path=" in r.message or "verdict_path=" in r.message
        for r in caplog.records
        if "[aggressive_mode]" in r.message
    )


# ---------------------------------------------------------------------------
# Saturation bypass in aggressive mode
# ---------------------------------------------------------------------------

def test_aggressive_mode_bypasses_saturation_filter(monkeypatch, caplog):
    """In aggressive mode, a SEED paper with >14 comments is NOT dropped; saturation_disabled logged."""
    import logging
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    # Paper in SEED_WINDOW (hours_ago=6) with 20 total comments (above threshold of 14)
    db = _make_db([_row("p-sat", hours_ago=6)], citable_other=3, has_participated=False)
    db.get_comment_stats.return_value = {
        "total": 20,
        "ours": 0,
        "citable_other": 3,
    }

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        counters = run_operational_loop(db, _NOW, test_mode=True)

    assert any(
        "saturation_disabled" in r.message and "p-sat" in r.message
        for r in caplog.records
    )
    assert not any("saturated_comments" in r.message for r in caplog.records)
    assert not any("saturation_bypass" in r.message for r in caplog.records)
    assert counters["papers_processed"] >= 1


def test_normal_mode_drops_saturated_papers(monkeypatch, caplog):
    """In normal mode, a SEED paper with >14 comments IS dropped with saturated_comments log."""
    import logging
    monkeypatch.delenv("KOALA_AGGRESSIVE_FINAL_24H", raising=False)
    db = _make_db([_row("p-sat", hours_ago=6)], citable_other=3, has_participated=False)
    db.get_comment_stats.return_value = {
        "total": 20,
        "ours": 0,
        "citable_other": 3,
    }

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        counters = run_operational_loop(db, _NOW, test_mode=True)

    assert any("saturated_comments" in r.message and "p-sat" in r.message for r in caplog.records)
    assert not any("saturation_disabled" in r.message for r in caplog.records)
    assert not any("saturation_bypass" in r.message for r in caplog.records)
    assert counters["papers_processed"] == 0


def test_aggressive_saturation_disabled_does_not_affect_verdict_path(monkeypatch, caplog):
    """In aggressive mode, a verdict-eligible paper with >14 comments remains eligible."""
    import logging
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    db = _make_db(
        [_row("p-verd", hours_ago=49, state="VERDICT_ACTIVE")],
        citable_other=3, ours=1, has_participated=True,
    )
    db.get_comment_stats.return_value = {
        "total": 18,
        "ours": 1,
        "citable_other": 3,
    }

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        counters = run_operational_loop(db, _NOW, test_mode=True)

    assert not any("saturated_comments" in r.message for r in caplog.records)
    assert counters["papers_processed"] >= 1


def test_aggressive_mode_high_comment_seed_remains_eligible(monkeypatch, caplog):
    """In aggressive mode, SEED paper with 30 total comments and >=3 citable_other is processed."""
    import logging
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    db = _make_db([_row("p-hi-seed", hours_ago=6)], citable_other=5, has_participated=False)
    db.get_comment_stats.return_value = {
        "total": 30,
        "ours": 0,
        "citable_other": 5,
    }

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        counters = run_operational_loop(db, _NOW, test_mode=True)

    saturation_log = [
        r for r in caplog.records
        if "saturation_disabled" in r.message and "p-hi-seed" in r.message
    ]
    assert saturation_log, "Expected saturation_disabled log for high-comment SEED paper"
    assert "comment_count=30" in saturation_log[0].message
    assert "citable_other=5" in saturation_log[0].message
    assert not any("saturated_comments" in r.message for r in caplog.records)
    assert counters["papers_processed"] >= 1


def test_aggressive_mode_high_comment_verdict_paper_eligible(monkeypatch, caplog):
    """In aggressive mode, a verdict paper with 25 total comments and 4 citable_other is processed."""
    import logging
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    db = _make_db(
        [_row("p-hi-verd", hours_ago=49, state="VERDICT_ACTIVE")],
        citable_other=4, ours=1, has_participated=True,
    )
    db.get_comment_stats.return_value = {
        "total": 25,
        "ours": 1,
        "citable_other": 4,
    }

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        counters = run_operational_loop(db, _NOW, test_mode=True)

    assert not any("saturated_comments" in r.message for r in caplog.records)
    assert not any("saturation_disabled" in r.message for r in caplog.records)
    assert counters["papers_processed"] >= 1


def test_aggressive_mode_saturation_disabled_log_format(monkeypatch, caplog):
    """saturation_disabled log must include paper_id, comment_count, and citable_other."""
    import logging
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    db = _make_db([_row("p-fmt", hours_ago=6)], citable_other=4, has_participated=False)
    db.get_comment_stats.return_value = {
        "total": 22,
        "ours": 0,
        "citable_other": 4,
    }

    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        run_operational_loop(db, _NOW, test_mode=True)

    matches = [r for r in caplog.records if "saturation_disabled" in r.message and "p-fmt" in r.message]
    assert matches, "Expected saturation_disabled log"
    msg = matches[0].message
    assert "comment_count=22" in msg
    assert "citable_other=4" in msg


# ---------------------------------------------------------------------------
# Runtime allowlist bypass in aggressive + live_verdict_auto mode
# ---------------------------------------------------------------------------

_VERDICT_NOOP_RESULT = {
    "paper_id": "p-verd",
    "reactive_status": "none",
    "reactive_reason": None,
    "reactive_artifact": None,
    "reactive_live_posted": False,
    "reactive_live_reason": None,
    "verdict_status": "dry_run",
    "verdict_reason": None,
    "verdict_artifact": "https://example.com/v.md",
    "verdict_live_submitted": False,
    "verdict_live_reason": "live_submitted",
    "has_reactive_candidate": False,
    "reactive_draft_created": False,
    "verdict_eligible": True,
    "verdict_draft_created": False,
    "window_skipped": False,
    "seed_draft_created": False,
    "seed_live_posted": False,
}


def _verdict_row(paper_id: str, *, citable_other: int = 4) -> dict:
    open_time = _NOW - timedelta(hours=55)
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "open_time": open_time.isoformat(),
        "review_end_time": (open_time + timedelta(hours=48)).isoformat(),
        "verdict_end_time": (open_time + timedelta(hours=72)).isoformat(),
        "state": "VERDICT_ACTIVE",
        "pdf_url": "",
        "local_pdf_path": None,
        "deliberating_at": (open_time + timedelta(hours=48)).isoformat(),
    }


def _make_live_loop_db(rows: list, *, citable_other: int = 4) -> MagicMock:
    """DB mock for verdict-ready papers (participated, enough citable comments)."""
    db = MagicMock()
    db.get_papers.return_value = rows
    db.get_comment_stats.return_value = {
        "total": citable_other + 1,
        "ours": 1,
        "citable_other": citable_other,
    }
    db.get_distinct_other_agent_count.return_value = citable_other
    db.has_prior_participation.return_value = True
    db.has_recent_reactive_action_for_comment.return_value = False
    db.has_recent_verdict_action_for_paper.return_value = False
    db.has_recent_seed_action_for_paper.return_value = False
    db.get_reactive_analysis_results.return_value = []
    db.get_citable_other_comments_for_paper.return_value = []
    db.get_latest_action_for_paper.return_value = None
    db.get_phase5a_stats.return_value = {
        "react_count": 0, "skip_count": 0, "unclear_count": 0,
        "comments_analyzed": 0, "claims_extracted": 0, "claims_verified": 0,
        "contradicted_count": 0, "supported_count": 0, "unclear_v_count": 0,
    }
    db.get_strongest_contradiction_confidence.return_value = None
    return db


def _run_loop_live(db: MagicMock, *, aggressive: bool, live_verdict_auto: bool,
                   process_result: dict, tmp_path) -> tuple[dict, MagicMock]:
    """Run the operational loop in live mode with _process_paper mocked."""
    process_mock = MagicMock(return_value=process_result)
    with (
        patch(f"{_MOD}.is_aggressive_mode", return_value=aggressive),
        patch(f"{_MOD}._process_paper", process_mock),
        patch("gsr_agent.koala.client.KoalaClient", return_value=MagicMock()),
        patch("gsr_agent.koala.sync.sync_all_active_state"),
        patch(f"{_MOD}.build_run_summary", return_value=[]),
        patch(f"{_MOD}.write_run_summary_markdown"),
        patch(f"{_MOD}.write_run_summary_jsonl"),
        patch(f"{_MOD}._write_verdict_opportunities_report"),
    ):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        counters = run_operational_loop(
            db, _NOW,
            live_verdict=True,
            live_verdict_auto=live_verdict_auto,
            paper_ids=None,
            output_dir=str(tmp_path),
        )
    return counters, process_mock


def test_aggressive_live_verdict_auto_bypasses_allowlist_runtime(
    monkeypatch, tmp_path, caplog
):
    """Non-auto-candidate papers get allowlisted=True and emit the bypass log in aggressive+auto mode.

    With two verdict-ready papers, the loop selects one as the auto-candidate. The second paper
    is NOT the auto-candidate, so pre-fix it would get allowlisted=False. Post-fix it must get
    allowlisted=True via the aggressive bypass, and the bypass must be logged.
    """
    import logging
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    rows = [_verdict_row("p-auto", citable_other=5), _verdict_row("p-other", citable_other=4)]
    db = _make_live_loop_db(rows)

    call_kwargs: list[dict] = []

    def _capture(paper, *a, **kw):
        call_kwargs.append({"paper_id": paper.paper_id, **kw})
        return dict(_VERDICT_NOOP_RESULT, paper_id=paper.paper_id)

    process_mock = MagicMock(side_effect=_capture)
    with caplog.at_level(logging.INFO, logger=_MOD):
        with (
            patch(f"{_MOD}.is_aggressive_mode", return_value=True),
            patch(f"{_MOD}._process_paper", process_mock),
            patch("gsr_agent.koala.client.KoalaClient", return_value=MagicMock()),
            patch("gsr_agent.koala.sync.sync_all_active_state"),
            patch(f"{_MOD}.build_run_summary", return_value=[]),
            patch(f"{_MOD}.write_run_summary_markdown"),
            patch(f"{_MOD}.write_run_summary_jsonl"),
            patch(f"{_MOD}._write_verdict_opportunities_report"),
        ):
            from gsr_agent.orchestration.operational_loop import run_operational_loop
            run_operational_loop(
                db, _NOW,
                live_verdict=True, live_verdict_auto=True, paper_ids=None,
                output_dir=str(tmp_path),
            )

    other_kwargs = next(kw for kw in call_kwargs if kw["paper_id"] == "p-other")
    assert other_kwargs["allowlisted"] is True, "non-auto-candidate must be allowlisted=True via bypass"
    bypass_logs = [r for r in caplog.records if "verdict_allowlist_bypassed_runtime" in r.message
                   and "p-other" in r.message]
    assert bypass_logs, "Expected verdict_allowlist_bypassed_runtime log for non-auto-candidate"


def test_aggressive_live_verdict_auto_reaches_live_submission(monkeypatch, tmp_path):
    """In aggressive+live_verdict_auto mode, submitted verdicts are counted."""
    monkeypatch.setenv("KOALA_AGGRESSIVE_FINAL_24H", "1")
    db = _make_live_loop_db([_verdict_row("p-verd")])
    result = dict(_VERDICT_NOOP_RESULT, verdict_live_submitted=True, verdict_status="live_submitted")

    counters, _ = _run_loop_live(db, aggressive=True, live_verdict_auto=True,
                                  process_result=result, tmp_path=tmp_path)

    assert counters["live_verdict_submissions"] == 1


def test_normal_mode_live_verdict_requires_allowlist(monkeypatch, tmp_path):
    """In normal (non-aggressive) mode, non-auto-candidate papers get allowlisted=False.

    With two verdict-ready papers, only the auto-candidate gets allowlisted=True (from
    _is_auto_candidate). The second paper must NOT be bypassed — it gets allowlisted=False.
    """
    monkeypatch.delenv("KOALA_AGGRESSIVE_FINAL_24H", raising=False)
    rows = [_verdict_row("p-auto", citable_other=5), _verdict_row("p-other", citable_other=4)]
    db = _make_live_loop_db(rows)

    call_kwargs: list[dict] = []

    def _capture(paper, *a, **kw):
        call_kwargs.append({"paper_id": paper.paper_id, **kw})
        return dict(_VERDICT_NOOP_RESULT, paper_id=paper.paper_id)

    with (
        patch(f"{_MOD}.is_aggressive_mode", return_value=False),
        patch(f"{_MOD}._process_paper", side_effect=_capture),
        patch("gsr_agent.koala.client.KoalaClient", return_value=MagicMock()),
        patch("gsr_agent.koala.sync.sync_all_active_state"),
        patch(f"{_MOD}.build_run_summary", return_value=[]),
        patch(f"{_MOD}.write_run_summary_markdown"),
        patch(f"{_MOD}.write_run_summary_jsonl"),
        patch(f"{_MOD}._write_verdict_opportunities_report"),
    ):
        from gsr_agent.orchestration.operational_loop import run_operational_loop
        run_operational_loop(
            db, _NOW,
            live_verdict=True, live_verdict_auto=True, paper_ids=None,
            output_dir=str(tmp_path),
        )

    other_kwargs = next(kw for kw in call_kwargs if kw["paper_id"] == "p-other")
    assert other_kwargs["allowlisted"] is False, "non-auto-candidate must remain allowlisted=False in normal mode"
