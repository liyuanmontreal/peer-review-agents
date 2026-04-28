"""Tests for Phase 10.5: automatic live verdict mode (gsr_agent only)."""

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

_FULL_LIVE_ENV = {
    "KOALA_RUN_MODE": "live",
    "KOALA_API_TOKEN": "test-token",
    "KOALA_ARTIFACT_MODE": "github",
    "KOALA_GITHUB_REPO": "owner/repo",
}


def _deliberating_row(paper_id: str = "delib-001") -> dict:
    open_time = _NOW - timedelta(hours=55)
    return {
        "paper_id": paper_id,
        "title": f"Deliberating Paper {paper_id}",
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


_NOOP_RESULT = {
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


def _run_auto(db: MagicMock, output_dir: str) -> dict:
    with (
        patch(f"{_MOD}._process_paper", return_value=_NOOP_RESULT),
        patch(f"{_MOD}.build_run_summary", return_value=[]),
        patch(f"{_MOD}.write_run_summary_markdown"),
        patch(f"{_MOD}.write_run_summary_jsonl"),
    ):
        return run_operational_loop(db, _NOW, output_dir=output_dir, live_verdict_auto=True)


# ---------------------------------------------------------------------------
# Auto-verdict candidate rejection gates
# ---------------------------------------------------------------------------

def test_auto_verdict_skipped_no_own_comment(tmp_path, caplog):
    """No prior own comment → no_verdict_ready_candidates before any processing."""
    db = _make_db([_deliberating_row()], citable_other=4, has_participated=False)
    with caplog.at_level("INFO", logger=_MOD):
        _run_auto(db, str(tmp_path))
    assert "auto_verdict_skipped" in caplog.text
    assert "no_verdict_ready_candidates" in caplog.text
    assert "selected_auto_verdict" not in caplog.text


def test_auto_verdict_skipped_insufficient_citable(tmp_path, caplog):
    """Fewer than 3 citeable comments → no_verdict_ready_candidates."""
    db = _make_db([_deliberating_row()], citable_other=2, has_participated=True)
    with caplog.at_level("INFO", logger=_MOD):
        _run_auto(db, str(tmp_path))
    assert "auto_verdict_skipped" in caplog.text
    assert "no_verdict_ready_candidates" in caplog.text
    assert "selected_auto_verdict" not in caplog.text


# ---------------------------------------------------------------------------
# At-most-one selection
# ---------------------------------------------------------------------------

def test_auto_verdict_selects_at_most_one(tmp_path, caplog):
    """Two valid deliberating papers → exactly one selected_auto_verdict log entry."""
    rows = [_deliberating_row("delib-001"), _deliberating_row("delib-002")]
    db = _make_db(rows, citable_other=4, has_participated=True)
    with caplog.at_level("INFO", logger=_MOD):
        _run_auto(db, str(tmp_path))
    selected = [m for m in caplog.messages if "selected_auto_verdict" in m]
    assert len(selected) == 1


# ---------------------------------------------------------------------------
# Preflight: --live-verdict-auto does not require --paper-id
# ---------------------------------------------------------------------------

def test_live_verdict_auto_does_not_require_paper_id(tmp_path):
    with (
        patch(f"{_MOD}.get_run_mode", return_value="live"),
        patch(f"{_MOD}.is_github_publish_configured", return_value=True),
        patch.dict(os.environ, {"KOALA_API_TOKEN": "tok"}),
    ):
        ok, failures = run_preflight_checks(
            str(tmp_path / "agent.db"),
            str(tmp_path),
            live_verdict_auto=True,
            paper_ids=None,
        )
    assert ok
    assert failures == []


# ---------------------------------------------------------------------------
# Regression: --live-verdict still requires --paper-id
# ---------------------------------------------------------------------------

def test_old_live_verdict_still_requires_paper_id(tmp_path):
    """--live-verdict without --paper-id must still fail preflight."""
    with (
        patch(f"{_MOD}.get_run_mode", return_value="live"),
        patch(f"{_MOD}.is_github_publish_configured", return_value=True),
        patch.dict(os.environ, {"KOALA_API_TOKEN": "tok"}),
    ):
        ok, failures = run_preflight_checks(
            str(tmp_path / "agent.db"),
            str(tmp_path),
            live_verdict=True,
            paper_ids=None,
        )
    assert not ok
    assert any("paper-id" in f or "allowlist" in f for f in failures)


# ---------------------------------------------------------------------------
# Launch command tests
# ---------------------------------------------------------------------------

def test_gsr_agent_launch_includes_live_verdict_auto(tmp_path):
    """gsr_agent launch command must include --live-verdict-auto."""
    import json
    from unittest.mock import MagicMock, patch as _patch
    from click.testing import CliRunner
    from reva.cli import main

    agents_dir = tmp_path / "agents"
    agent_dir = agents_dir / "gsr_agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "config.json").write_text(
        json.dumps({"name": "gsr_agent", "backend": "claude-code"}), encoding="utf-8"
    )
    (agent_dir / "system_prompt.md").write_text("hi", encoding="utf-8")
    (agent_dir / ".api_key").write_text("KEY", encoding="utf-8")
    global_rules = tmp_path / "GLOBAL_RULES.md"
    global_rules.write_text("R\n", encoding="utf-8")
    platform_skills = tmp_path / "platform_skills.md"
    platform_skills.write_text("S\n", encoding="utf-8")

    mock_cfg = MagicMock()
    mock_cfg.agents_dir = agents_dir
    mock_cfg.global_rules_path = global_rules
    mock_cfg.platform_skills_path = platform_skills
    mock_cfg.github_repo = ""
    mock_cfg.koala_base_url = "https://koala.science"

    captured_cmd: list[str] = []

    def _capture_build_launch_script(cmd, **kwargs):
        captured_cmd.append(cmd)
        return "#!/bin/bash\necho hi\n"

    with (
        _patch("reva.cli._get_config", return_value=mock_cfg),
        _patch("reva.cli.create_session"),
        _patch("reva.cli.build_launch_script", side_effect=_capture_build_launch_script),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["launch", "--name", "gsr_agent"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

    assert captured_cmd, "build_launch_script was not called"
    assert "--live-verdict-auto" in captured_cmd[0]


def test_non_gsr_agent_launch_excludes_live_verdict_auto(tmp_path):
    """Non-gsr agent launch commands must not include --live-verdict-auto."""
    import json
    from unittest.mock import MagicMock, patch as _patch
    from click.testing import CliRunner
    from reva.cli import main

    agents_dir = tmp_path / "agents"
    agent_dir = agents_dir / "other_agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "config.json").write_text(
        json.dumps({"name": "other_agent", "backend": "claude-code"}), encoding="utf-8"
    )
    (agent_dir / "system_prompt.md").write_text("hi", encoding="utf-8")
    (agent_dir / ".api_key").write_text("KEY", encoding="utf-8")
    global_rules = tmp_path / "GLOBAL_RULES.md"
    global_rules.write_text("R\n", encoding="utf-8")
    platform_skills = tmp_path / "platform_skills.md"
    platform_skills.write_text("S\n", encoding="utf-8")

    mock_cfg = MagicMock()
    mock_cfg.agents_dir = agents_dir
    mock_cfg.global_rules_path = global_rules
    mock_cfg.platform_skills_path = platform_skills
    mock_cfg.github_repo = ""
    mock_cfg.koala_base_url = "https://koala.science"

    captured_cmd: list[str] = []

    def _capture_build_launch_script(cmd, **kwargs):
        captured_cmd.append(cmd)
        return "#!/bin/bash\necho hi\n"

    with (
        _patch("reva.cli._get_config", return_value=mock_cfg),
        _patch("reva.cli.create_session"),
        _patch("reva.cli.build_launch_script", side_effect=_capture_build_launch_script),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["launch", "--name", "other_agent"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

    assert captured_cmd, "build_launch_script was not called"
    assert "--live-verdict-auto" not in captured_cmd[0]
