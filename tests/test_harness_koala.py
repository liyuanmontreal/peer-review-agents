"""Tests for agent_definition/harness/koala.py — KoalaClient MCP URL and safe mode."""
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock


def _ensure_httpx_module() -> None:
    """Install a MagicMock for httpx only when real httpx is not available.

    Avoids polluting sys.modules with a MagicMock when httpx is installed,
    which would break later imports that expect httpx to be a real package
    (e.g. litellm importing httpx._utils).
    """
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

    spec = importlib.util.spec_from_file_location("_harness_koala", harness_dir / "koala.py")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "agent_definition.harness"
    spec.loader.exec_module(module)
    return module


def _make_httpx_response(text: str = "ok"):
    resp = MagicMock()
    resp.json.return_value = {"result": {"content": [{"type": "text", "text": text}]}}
    resp.raise_for_status.return_value = None
    return resp


# ── URL configuration ──────────────────────────────────────────────────────────

def test_koala_client_mcp_url_uses_default(monkeypatch):
    monkeypatch.delenv("KOALA_BASE_URL", raising=False)
    module = _load_koala_module()
    client = module.KoalaClient(api_key="test-key")
    assert client.mcp_url == "https://koala.science/mcp"


def test_koala_client_mcp_url_honors_env(monkeypatch):
    monkeypatch.setenv("KOALA_BASE_URL", "https://staging.koala.science")
    module = _load_koala_module()
    client = module.KoalaClient(api_key="test-key")
    assert client.mcp_url == "https://staging.koala.science/mcp"


# ── Safe mode: write tools blocked ────────────────────────────────────────────

def test_post_comment_intercepted_when_write_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("KOALA_WRITE_ENABLED", raising=False)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    module = _load_koala_module()
    client = module.KoalaClient(api_key="test-key")

    raw = client.call_tool("post_comment", {
        "paper_id": "p1",
        "content_markdown": "Hello",
        "github_file_url": "https://github.com/test/repo/blob/main/logs/p1/c1.md",
    })
    result = json.loads(raw)
    assert result["safe_mode"] is True
    assert result["would_post"] is True
    assert result["tool"] == "post_comment"
    assert result["paper_id"] == "p1"


def test_post_verdict_intercepted_when_write_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("KOALA_WRITE_ENABLED", raising=False)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    module = _load_koala_module()
    client = module.KoalaClient(api_key="test-key")

    raw = client.call_tool("post_verdict", {
        "paper_id": "p2",
        "score": 7.5,
        "content_markdown": "My verdict",
        "github_file_url": "https://github.com/test/repo/blob/main/logs/p2/v1.md",
    })
    result = json.loads(raw)
    assert result["safe_mode"] is True
    assert result["would_post"] is True
    assert result["tool"] == "post_verdict"


def test_write_false_also_blocks(monkeypatch, tmp_path):
    monkeypatch.setenv("KOALA_WRITE_ENABLED", "false")
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    module = _load_koala_module()
    client = module.KoalaClient(api_key="test-key")

    raw = client.call_tool("post_comment", {
        "paper_id": "p1",
        "content_markdown": "Hello",
        "github_file_url": "https://example.com/log.md",
    })
    result = json.loads(raw)
    assert result["safe_mode"] is True


def test_write_tools_proceed_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("KOALA_WRITE_ENABLED", "true")
    module = _load_koala_module()
    monkeypatch.setattr(module.httpx, "post", MagicMock(return_value=_make_httpx_response("posted")))

    client = module.KoalaClient(api_key="test-key")
    result = client.call_tool("post_comment", {
        "paper_id": "00000000-0000-0000-0000-000000000003",
        "content_markdown": "Real post",
        "github_file_url": "https://github.com/test/repo/blob/main/logs/p3/c1.md",
    })
    assert result == "posted"
    module.httpx.post.assert_called_once()


# ── Safe mode: read tools always pass through ─────────────────────────────────

def test_read_tools_not_intercepted_in_safe_mode(monkeypatch):
    monkeypatch.delenv("KOALA_WRITE_ENABLED", raising=False)
    module = _load_koala_module()
    monkeypatch.setattr(module.httpx, "post", MagicMock(return_value=_make_httpx_response("papers")))

    client = module.KoalaClient(api_key="test-key")
    for tool in ("get_papers", "get_paper", "get_comments", "get_actor_profile", "get_notifications"):
        args = {} if tool == "get_papers" else {"paper_id": "p1"}
        result = client.call_tool(tool, args)
        assert "safe_mode" not in result, f"{tool} should not be intercepted"


# ── Safe mode: transparency log is written ────────────────────────────────────

def test_safe_mode_creates_transparency_log(monkeypatch, tmp_path):
    monkeypatch.delenv("KOALA_WRITE_ENABLED", raising=False)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    module = _load_koala_module()
    client = module.KoalaClient(api_key="test-key")

    client.call_tool("post_comment", {
        "paper_id": "paper-xyz",
        "content_markdown": "Test comment",
        "github_file_url": "https://github.com/test/repo/blob/main/logs/paper-xyz/c1.md",
    })

    log_files = list((tmp_path / "paper-xyz").glob("*.md"))
    assert len(log_files) == 1


def test_safe_mode_log_contains_payload(monkeypatch, tmp_path):
    monkeypatch.delenv("KOALA_WRITE_ENABLED", raising=False)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    module = _load_koala_module()
    client = module.KoalaClient(api_key="test-key")

    client.call_tool("post_verdict", {
        "paper_id": "p-log",
        "score": 6.0,
        "content_markdown": "verdict content here",
        "github_file_url": "https://github.com/test/repo/blob/main/logs/p-log/v1.md",
    })

    log_file = next((tmp_path / "p-log").glob("*.md"))
    text = log_file.read_text()
    assert "verdict content here" in text
    assert "post_verdict" in text


# ── _WRITE_TOOLS constant is correctly defined ────────────────────────────────

def test_write_tools_set_contains_exactly_write_ops(monkeypatch):
    monkeypatch.delenv("KOALA_WRITE_ENABLED", raising=False)
    module = _load_koala_module()
    assert module._WRITE_TOOLS == frozenset({"post_comment", "post_verdict"})
