"""
Phase 1 competition-safety patch — minimal harness window tests.

Tests 1-4 cover koala_window_state() from window.py (no reva dependency).
Tests 5-6 cover KoalaClient guards via the _load_koala_module() pattern.
"""
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from agent_definition.harness.window import _UUID_RE, koala_window_state

_NOW = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)


def _dt(hours_ago: float) -> str:
    return (_NOW - timedelta(hours=hours_ago)).isoformat()


def _ensure_httpx_module() -> None:
    try:
        import httpx  # noqa: F401
    except ModuleNotFoundError:
        sys.modules.setdefault("httpx", MagicMock())


def _load_koala_module():
    _ensure_httpx_module()
    harness_dir = Path(__file__).parent.parent / "agent_definition" / "harness"

    if "agent_definition.harness.window" not in sys.modules:
        win_spec = importlib.util.spec_from_file_location(
            "agent_definition.harness.window",
            harness_dir / "window.py",
        )
        win_mod = importlib.util.module_from_spec(win_spec)
        sys.modules["agent_definition.harness.window"] = win_mod
        win_spec.loader.exec_module(win_mod)

    spec = importlib.util.spec_from_file_location("_harness_koala_win", harness_dir / "koala.py")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "agent_definition.harness"
    spec.loader.exec_module(module)
    return module


def test_in_review_within_48h_comment_phase_open():
    paper = {"status": "in_review", "created_at": _dt(10)}
    ws = koala_window_state(paper, now=_NOW)
    assert ws["phase"] == "comment"
    assert ws["open"] is True
    assert ws["seconds_left"] > 0


def test_in_review_expired_closed():
    paper = {"status": "in_review", "created_at": _dt(50)}
    ws = koala_window_state(paper, now=_NOW)
    assert ws["phase"] == "comment"
    assert ws["open"] is False


def test_deliberating_within_24h_verdict_phase_open():
    paper = {
        "status": "deliberating",
        "created_at": _dt(60),
        "deliberating_at": _dt(10),
    }
    ws = koala_window_state(paper, now=_NOW)
    assert ws["phase"] == "verdict"
    assert ws["open"] is True
    assert ws["seconds_left"] > 0


def test_deliberating_at_none_fallback_to_created_plus_48h():
    # deliberating_at=None → delib_start = created+48h → ends = created+72h
    # created 60h ago → ends 12h from now → open
    paper = {
        "status": "deliberating",
        "created_at": _dt(60),
        "deliberating_at": None,
    }
    ws = koala_window_state(paper, now=_NOW)
    assert ws["phase"] == "verdict"
    assert ws["open"] is True
    assert ws["seconds_left"] > 0


def test_short_id_not_converted_to_fake_uuid(monkeypatch, tmp_path):
    monkeypatch.setenv("KOALA_WRITE_ENABLED", "true")
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    module = _load_koala_module()
    client = module.KoalaClient(api_key="test-key")

    result = client.call_tool("post_comment", {
        "paper_id": "abc123",
        "content_markdown": "Test",
        "github_file_url": "https://example.com/file.md",
    })
    data = json.loads(result)
    assert data["error"] == "paper_id must be a full UUID"
    assert data["paper_id"] == "abc123"


def test_409_handled_without_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("KOALA_WRITE_ENABLED", "true")
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    module = _load_koala_module()

    mock_resp = MagicMock()
    mock_resp.status_code = 409
    exc_409 = module.httpx.HTTPStatusError(
        "409 Conflict", request=MagicMock(), response=mock_resp
    )
    monkeypatch.setattr(module.httpx, "post", MagicMock(side_effect=exc_409))

    client = module.KoalaClient(api_key="test-key")
    result = client.call_tool("post_comment", {
        "paper_id": "550e8400-e29b-41d4-a716-446655440000",
        "content_markdown": "Test",
        "github_file_url": "https://github.com/test/repo/blob/main/file.md",
    })
    data = json.loads(result)
    assert data["error"] == "window_closed"
    assert "550e8400" in data["paper_id"]
