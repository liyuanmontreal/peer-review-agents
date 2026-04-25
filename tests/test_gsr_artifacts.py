"""Tests for gsr_artifacts hook and KoalaClient on_write integration."""
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock


# ── Module loading helpers ────────────────────────────────────────────────────


def _load_koala_module():
    sys.modules.setdefault("httpx", MagicMock())
    path = Path(__file__).parent.parent / "agent_definition" / "harness" / "koala.py"
    spec = importlib.util.spec_from_file_location("_koala_for_gsr_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _httpx_response(text="ok"):
    resp = MagicMock()
    resp.json.return_value = {"result": {"content": [{"type": "text", "text": text}]}}
    resp.raise_for_status.return_value = None
    return resp


# ── emit_gsr_artifacts: comment artifacts ────────────────────────────────────


def test_safe_mode_post_comment_creates_comment_draft(tmp_path):
    from agent_definition.harness.gsr_artifacts import emit_gsr_artifacts
    emit_gsr_artifacts(
        "post_comment",
        {"paper_id": "p1", "content_markdown": "Missing ablation."},
        safe_mode=True,
        artifact_dir=str(tmp_path),
        logs_base_path="reviews",
    )
    drafts = list((tmp_path / "p1").glob("comment_draft_p1_*.md"))
    assert len(drafts) == 1


def test_safe_mode_post_comment_creates_comment_trace(tmp_path):
    from agent_definition.harness.gsr_artifacts import emit_gsr_artifacts
    emit_gsr_artifacts(
        "post_comment",
        {"paper_id": "p1", "content_markdown": "Missing ablation."},
        safe_mode=True,
        artifact_dir=str(tmp_path),
        logs_base_path="reviews",
    )
    traces = list((tmp_path / "p1").glob("comment_trace_p1_*.json"))
    assert len(traces) == 1
    data = json.loads(traces[0].read_text())
    assert data["artifact_kind"] == "comment_trace"
    assert data["source_phase"] == "review"
    assert data["payload"]["safe_mode"] is True
    assert data["payload"]["action_type"] == "post_comment"


def test_post_comment_trace_captures_parent_id(tmp_path):
    from agent_definition.harness.gsr_artifacts import emit_gsr_artifacts
    emit_gsr_artifacts(
        "post_comment",
        {"paper_id": "p1", "content_markdown": "Reply.", "parent_id": "abc-123"},
        safe_mode=False,
        artifact_dir=str(tmp_path),
        logs_base_path="reviews",
    )
    trace = next((tmp_path / "p1").glob("comment_trace_p1_*.json"))
    data = json.loads(trace.read_text())
    assert data["payload"]["parent_id"] == "abc-123"


def test_post_comment_draft_contains_content(tmp_path):
    from agent_definition.harness.gsr_artifacts import emit_gsr_artifacts
    emit_gsr_artifacts(
        "post_comment",
        {"paper_id": "p1", "content_markdown": "Claim X is unsupported."},
        safe_mode=True,
        artifact_dir=str(tmp_path),
        logs_base_path="reviews",
    )
    draft = next((tmp_path / "p1").glob("comment_draft_p1_*.md"))
    assert "Claim X is unsupported." in draft.read_text()


# ── emit_gsr_artifacts: verdict artifacts ────────────────────────────────────


def test_safe_mode_post_verdict_creates_verdict_draft(tmp_path):
    from agent_definition.harness.gsr_artifacts import emit_gsr_artifacts
    emit_gsr_artifacts(
        "post_verdict",
        {
            "paper_id": "p2",
            "score": 6.5,
            "content_markdown": "Verdict body.",
            "github_file_url": "http://example.com/v1.md",
        },
        safe_mode=True,
        artifact_dir=str(tmp_path),
        logs_base_path="reviews",
    )
    drafts = list((tmp_path / "p2").glob("verdict_draft_p2_*.md"))
    assert len(drafts) == 1


def test_safe_mode_post_verdict_creates_verdict_trace(tmp_path):
    from agent_definition.harness.gsr_artifacts import emit_gsr_artifacts
    emit_gsr_artifacts(
        "post_verdict",
        {"paper_id": "p2", "score": 6.5, "content_markdown": "Verdict."},
        safe_mode=True,
        artifact_dir=str(tmp_path),
        logs_base_path="reviews",
    )
    traces = list((tmp_path / "p2").glob("verdict_trace_p2_*.json"))
    assert len(traces) == 1
    data = json.loads(traces[0].read_text())
    assert data["artifact_kind"] == "verdict_trace"
    assert data["source_phase"] == "verdict"
    assert data["payload"]["safe_mode"] is True
    assert data["payload"]["score"] == 6.5


def test_post_verdict_trace_extracts_cited_comments(tmp_path):
    from agent_definition.harness.gsr_artifacts import emit_gsr_artifacts
    content = "Body [[comment:abc-111]] and [[comment:def-222]]."
    emit_gsr_artifacts(
        "post_verdict",
        {"paper_id": "p2", "score": 7.0, "content_markdown": content},
        safe_mode=False,
        artifact_dir=str(tmp_path),
        logs_base_path="reviews",
    )
    trace = next((tmp_path / "p2").glob("verdict_trace_p2_*.json"))
    data = json.loads(trace.read_text())
    assert "[[comment:abc-111]]" in data["payload"]["cited_comments"]
    assert "[[comment:def-222]]" in data["payload"]["cited_comments"]


def test_post_verdict_draft_contains_score_and_content(tmp_path):
    from agent_definition.harness.gsr_artifacts import emit_gsr_artifacts
    emit_gsr_artifacts(
        "post_verdict",
        {"paper_id": "p2", "score": 8.0, "content_markdown": "Strong contribution."},
        safe_mode=False,
        artifact_dir=str(tmp_path),
        logs_base_path="reviews",
    )
    draft = next((tmp_path / "p2").glob("verdict_draft_p2_*.md"))
    text = draft.read_text()
    assert "8.0" in text
    assert "Strong contribution." in text


# ── paper summary: created once, not overwritten ─────────────────────────────


def test_paper_summary_created_on_first_comment(tmp_path):
    from agent_definition.harness.gsr_artifacts import emit_gsr_artifacts
    emit_gsr_artifacts(
        "post_comment",
        {"paper_id": "p3", "content_markdown": "Comment."},
        safe_mode=True,
        artifact_dir=str(tmp_path),
        logs_base_path="reviews",
    )
    assert (tmp_path / "p3" / "paper_p3_summary.md").exists()


def test_paper_summary_not_overwritten_on_second_comment(tmp_path):
    from agent_definition.harness.gsr_artifacts import emit_gsr_artifacts
    summary_dir = tmp_path / "p3"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / "paper_p3_summary.md"
    summary_path.write_text("# Custom summary\nDo not overwrite.", encoding="utf-8")

    emit_gsr_artifacts(
        "post_comment",
        {"paper_id": "p3", "content_markdown": "Second comment."},
        safe_mode=True,
        artifact_dir=str(tmp_path),
        logs_base_path="reviews",
    )
    assert summary_path.read_text(encoding="utf-8") == "# Custom summary\nDo not overwrite."


# ── KoalaClient on_write hook ─────────────────────────────────────────────────


def test_koala_on_write_fires_on_post_comment_safe_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("KOALA_WRITE_ENABLED", raising=False)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    mod = _load_koala_module()

    calls = []
    client = mod.KoalaClient(api_key="k", on_write=lambda n, a, *, safe_mode: calls.append((n, safe_mode)))
    client.call_tool("post_comment", {"paper_id": "p1", "content_markdown": "x", "github_file_url": "http://x"})
    assert calls == [("post_comment", True)]


def test_koala_on_write_fires_on_post_verdict_safe_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("KOALA_WRITE_ENABLED", raising=False)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    mod = _load_koala_module()

    calls = []
    client = mod.KoalaClient(api_key="k", on_write=lambda n, a, *, safe_mode: calls.append((n, safe_mode)))
    client.call_tool("post_verdict", {"paper_id": "p1", "score": 5.0, "content_markdown": "x", "github_file_url": "http://x"})
    assert calls == [("post_verdict", True)]


def test_koala_on_write_safe_mode_false_in_live_mode(monkeypatch):
    monkeypatch.setenv("KOALA_WRITE_ENABLED", "true")
    mod = _load_koala_module()
    monkeypatch.setattr(mod.httpx, "post", MagicMock(return_value=_httpx_response("ok")))

    calls = []
    client = mod.KoalaClient(api_key="k", on_write=lambda n, a, *, safe_mode: calls.append((n, safe_mode)))
    client.call_tool("post_comment", {"paper_id": "p1", "content_markdown": "x", "github_file_url": "http://x"})
    assert calls == [("post_comment", False)]


def test_koala_on_write_not_fired_for_read_tools(monkeypatch):
    monkeypatch.delenv("KOALA_WRITE_ENABLED", raising=False)
    mod = _load_koala_module()
    monkeypatch.setattr(mod.httpx, "post", MagicMock(return_value=_httpx_response("papers")))

    calls = []
    client = mod.KoalaClient(api_key="k", on_write=lambda n, a, *, safe_mode: calls.append(n))
    client.call_tool("get_papers", {})
    client.call_tool("get_paper", {"paper_id": "p1"})
    assert calls == []


def test_koala_no_on_write_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("KOALA_WRITE_ENABLED", raising=False)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    mod = _load_koala_module()
    client = mod.KoalaClient(api_key="k")
    assert client._on_write is None
    # Should not raise
    client.call_tool("post_comment", {"paper_id": "p1", "content_markdown": "x", "github_file_url": "http://x"})


def test_koala_hook_error_does_not_block_safe_mode_intercept(monkeypatch, tmp_path):
    monkeypatch.delenv("KOALA_WRITE_ENABLED", raising=False)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    mod = _load_koala_module()

    def bad_hook(n, a, *, safe_mode):
        raise RuntimeError("hook exploded")

    client = mod.KoalaClient(api_key="k", on_write=bad_hook)
    # Should not raise; safe-mode intercept should still return normally
    raw = client.call_tool("post_comment", {"paper_id": "p1", "content_markdown": "x", "github_file_url": "http://x"})
    result = json.loads(raw)
    assert result["safe_mode"] is True


# ── Non-gsr agents unaffected ─────────────────────────────────────────────────


def test_agent_without_agent_name_has_no_hook(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("COALESCENCE_API_KEY", "test-key")
    sys.modules.setdefault("httpx", MagicMock())
    sys.modules.setdefault("anthropic", MagicMock())

    from agent_definition.harness.harness import Agent
    agent = Agent(system_prompt="test")
    assert agent.koala._on_write is None


def test_agent_with_gsr_agent_name_wires_hook(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("COALESCENCE_API_KEY", "test-key")
    sys.modules.setdefault("httpx", MagicMock())
    sys.modules.setdefault("anthropic", MagicMock())

    from agent_definition.harness.harness import Agent
    agent = Agent(system_prompt="test", agent_name="gsr_agent")
    assert agent.koala._on_write is not None


def test_agent_with_other_name_has_no_hook(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("COALESCENCE_API_KEY", "test-key")
    sys.modules.setdefault("httpx", MagicMock())
    sys.modules.setdefault("anthropic", MagicMock())

    from agent_definition.harness.harness import Agent
    agent = Agent(system_prompt="test", agent_name="some_other_agent")
    assert agent.koala._on_write is None


# ── write_smoke_artifact ──────────────────────────────────────────────────────


def test_smoke_artifact_file_created(tmp_path):
    from agent_definition.harness.gsr_artifacts import write_smoke_artifact
    write_smoke_artifact(artifact_dir=str(tmp_path))
    assert (tmp_path / "smoke_test" / "paper_smoke_test_summary.md").exists()


def test_smoke_artifact_contains_required_fields(tmp_path):
    from agent_definition.harness.gsr_artifacts import write_smoke_artifact
    write_smoke_artifact(agent_name="gsr_agent", artifact_dir=str(tmp_path))
    text = (tmp_path / "smoke_test" / "paper_smoke_test_summary.md").read_text()
    assert "gsr_agent" in text
    assert "smoke" in text.lower()
    assert "no Koala post" in text


def test_smoke_artifact_contains_timestamp(tmp_path):
    from agent_definition.harness.gsr_artifacts import write_smoke_artifact
    write_smoke_artifact(artifact_dir=str(tmp_path))
    text = (tmp_path / "smoke_test" / "paper_smoke_test_summary.md").read_text()
    assert "timestamp" in text


def test_smoke_artifact_states_no_koala_post(tmp_path):
    from agent_definition.harness.gsr_artifacts import write_smoke_artifact
    write_smoke_artifact(artifact_dir=str(tmp_path))
    text = (tmp_path / "smoke_test" / "paper_smoke_test_summary.md").read_text()
    assert "no Koala post" in text


def test_agent_run_creates_smoke_artifact_for_gsr_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("COALESCENCE_API_KEY", "test-key")

    anthropic_mock = MagicMock()
    end_turn = MagicMock()
    end_turn.stop_reason = "end_turn"
    end_turn.content = []
    anthropic_mock.Anthropic.return_value.messages.create.return_value = end_turn
    sys.modules["anthropic"] = anthropic_mock
    sys.modules.setdefault("httpx", MagicMock())

    import importlib
    import agent_definition.harness.harness as harness_mod
    importlib.reload(harness_mod)

    agent = harness_mod.Agent(
        system_prompt="test",
        agent_name="gsr_agent",
        smoke_artifact_dir=str(tmp_path),
    )
    agent.run()
    assert (tmp_path / "smoke_test" / "paper_smoke_test_summary.md").exists()


def test_local_artifact_smoke_creates_paper_summary(tmp_path):
    from agent_definition.harness.gsr_artifacts import write_local_artifact_smoke
    write_local_artifact_smoke(artifact_dir=str(tmp_path))
    assert (tmp_path / "local_artifact_smoke" / "paper_local_artifact_smoke_summary.md").exists()


def test_local_artifact_smoke_creates_comment_draft(tmp_path):
    from agent_definition.harness.gsr_artifacts import write_local_artifact_smoke
    write_local_artifact_smoke(artifact_dir=str(tmp_path))
    drafts = list((tmp_path / "local_artifact_smoke").glob("comment_draft_local_artifact_smoke_*.md"))
    assert len(drafts) == 1


def test_local_artifact_smoke_draft_contains_placeholder(tmp_path):
    from agent_definition.harness.gsr_artifacts import write_local_artifact_smoke
    write_local_artifact_smoke(artifact_dir=str(tmp_path))
    draft = next((tmp_path / "local_artifact_smoke").glob("comment_draft_local_artifact_smoke_*.md"))
    assert "Local artifact smoke test only; no Koala comment was attempted." in draft.read_text()


def test_local_artifact_smoke_creates_comment_trace(tmp_path):
    from agent_definition.harness.gsr_artifacts import write_local_artifact_smoke
    write_local_artifact_smoke(artifact_dir=str(tmp_path))
    traces = list((tmp_path / "local_artifact_smoke").glob("comment_trace_local_artifact_smoke_*.json"))
    assert len(traces) == 1


def test_local_artifact_smoke_trace_is_valid_json(tmp_path):
    from agent_definition.harness.gsr_artifacts import write_local_artifact_smoke
    write_local_artifact_smoke(artifact_dir=str(tmp_path))
    trace = next((tmp_path / "local_artifact_smoke").glob("comment_trace_local_artifact_smoke_*.json"))
    data = json.loads(trace.read_text())
    assert data["artifact_kind"] == "comment_trace"
    assert data["paper_id"] == "local_artifact_smoke"
    assert data["source_phase"] == "review"
    assert "summary" in data
    assert "payload" in data
    assert data["payload"]["action_type"] == "local_smoke"
    assert data["payload"]["safe_mode"] is True


def test_local_artifact_smoke_summary_says_local_only(tmp_path):
    from agent_definition.harness.gsr_artifacts import write_local_artifact_smoke
    write_local_artifact_smoke(artifact_dir=str(tmp_path))
    text = (tmp_path / "local_artifact_smoke" / "paper_local_artifact_smoke_summary.md").read_text()
    assert "local" in text.lower()
    assert "no Koala" in text


def test_agent_run_creates_local_artifact_smoke_for_gsr_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("COALESCENCE_API_KEY", "test-key")

    anthropic_mock = MagicMock()
    end_turn = MagicMock()
    end_turn.stop_reason = "end_turn"
    end_turn.content = []
    anthropic_mock.Anthropic.return_value.messages.create.return_value = end_turn
    sys.modules["anthropic"] = anthropic_mock
    sys.modules.setdefault("httpx", MagicMock())

    import importlib
    import agent_definition.harness.harness as harness_mod
    importlib.reload(harness_mod)

    agent = harness_mod.Agent(
        system_prompt="test",
        agent_name="gsr_agent",
        smoke_artifact_dir=str(tmp_path),
    )
    agent.run()
    assert (tmp_path / "local_artifact_smoke" / "paper_local_artifact_smoke_summary.md").exists()
    traces = list((tmp_path / "local_artifact_smoke").glob("comment_trace_local_artifact_smoke_*.json"))
    assert len(traces) == 1


def test_agent_run_does_not_create_smoke_artifact_for_other_agents(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("COALESCENCE_API_KEY", "test-key")

    anthropic_mock = MagicMock()
    end_turn = MagicMock()
    end_turn.stop_reason = "end_turn"
    end_turn.content = []
    anthropic_mock.Anthropic.return_value.messages.create.return_value = end_turn
    sys.modules["anthropic"] = anthropic_mock
    sys.modules.setdefault("httpx", MagicMock())

    import importlib
    import agent_definition.harness.harness as harness_mod
    importlib.reload(harness_mod)

    agent = harness_mod.Agent(
        system_prompt="test",
        agent_name="other_agent",
        smoke_artifact_dir=str(tmp_path),
    )
    agent.run()
    assert not (tmp_path / "smoke_test" / "paper_smoke_test_summary.md").exists()
