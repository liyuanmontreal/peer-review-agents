"""Tests for agent_definition/harness/transparency.py."""
import importlib.util
import json
import pytest
from pathlib import Path


def _load_transparency_module():
    path = Path(__file__).parent.parent / "agent_definition" / "harness" / "transparency.py"
    spec = importlib.util.spec_from_file_location("_harness_transparency", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_creates_log_file(tmp_path):
    mod = _load_transparency_module()
    mod.write_transparency_log(
        paper_id="paper-123",
        action_id="comment-001",
        action_type="comment",
        content="This paper looks good.",
        log_dir=str(tmp_path),
        github_repo="https://github.com/test/repo",
    )
    assert (tmp_path / "paper-123" / "comment-001.md").exists()


def test_write_returns_github_url(tmp_path):
    mod = _load_transparency_module()
    url = mod.write_transparency_log(
        paper_id="paper-123",
        action_id="verdict-001",
        action_type="verdict",
        content="Verdict body.",
        score=7.5,
        log_dir=str(tmp_path),
        github_repo="https://github.com/owner/repo",
    )
    assert url == "https://github.com/owner/repo/blob/main/logs/paper-123/verdict-001.md"


def test_write_uses_env_github_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("KOALA_GITHUB_REPO", "https://github.com/env/repo")
    mod = _load_transparency_module()
    url = mod.write_transparency_log(
        paper_id="p1",
        action_id="c1",
        action_type="comment",
        content="test",
        log_dir=str(tmp_path),
    )
    assert url == "https://github.com/env/repo/blob/main/logs/p1/c1.md"


def test_write_placeholder_when_no_github_repo(monkeypatch, tmp_path):
    monkeypatch.delenv("KOALA_GITHUB_REPO", raising=False)
    mod = _load_transparency_module()
    url = mod.write_transparency_log(
        paper_id="p1",
        action_id="c1",
        action_type="comment",
        content="test",
        log_dir=str(tmp_path),
    )
    assert url.startswith("TODO:")


def test_write_creates_parent_directory(tmp_path):
    mod = _load_transparency_module()
    deep_dir = tmp_path / "nested" / "logs"
    mod.write_transparency_log(
        paper_id="p1",
        action_id="c1",
        action_type="comment",
        content="test",
        log_dir=str(deep_dir),
    )
    assert (deep_dir / "p1" / "c1.md").exists()


def test_write_includes_score_for_verdicts(tmp_path):
    mod = _load_transparency_module()
    mod.write_transparency_log(
        paper_id="p1",
        action_id="v1",
        action_type="verdict",
        content="Verdict text.",
        score=8.5,
        log_dir=str(tmp_path),
    )
    content = (tmp_path / "p1" / "v1.md").read_text()
    assert "8.5" in content


def test_write_includes_cited_comments(tmp_path):
    mod = _load_transparency_module()
    mod.write_transparency_log(
        paper_id="p1",
        action_id="v1",
        action_type="verdict",
        content="Verdict with citations.",
        cited_comments=["[[comment:abc-123]]", "[[comment:def-456]]"],
        log_dir=str(tmp_path),
    )
    content = (tmp_path / "p1" / "v1.md").read_text()
    assert "[[comment:abc-123]]" in content
    assert "[[comment:def-456]]" in content


def test_write_includes_metadata_fields(tmp_path):
    mod = _load_transparency_module()
    mod.write_transparency_log(
        paper_id="p1",
        action_id="c1",
        action_type="comment",
        content="Content here.",
        model="claude-sonnet-4-6",
        reasoning="The paper lacks ablation studies.",
        log_dir=str(tmp_path),
    )
    content = (tmp_path / "p1" / "c1.md").read_text()
    assert "claude-sonnet-4-6" in content
    assert "The paper lacks ablation studies." in content


def test_write_custom_logs_base_path(tmp_path):
    mod = _load_transparency_module()
    url = mod.write_transparency_log(
        paper_id="p1",
        action_id="c1",
        action_type="comment",
        content="test",
        log_dir=str(tmp_path),
        github_repo="https://github.com/owner/repo",
        logs_base_path="transparency",
    )
    assert url == "https://github.com/owner/repo/blob/main/transparency/p1/c1.md"


def test_write_strips_trailing_slash_from_repo(tmp_path):
    mod = _load_transparency_module()
    url = mod.write_transparency_log(
        paper_id="p1",
        action_id="c1",
        action_type="comment",
        content="test",
        log_dir=str(tmp_path),
        github_repo="https://github.com/owner/repo/",
    )
    assert not url.count("//") > 1
    assert url.startswith("https://github.com/owner/repo/blob/main/")


# ---------------------------------------------------------------------------
# write_artifact tests
# ---------------------------------------------------------------------------


def test_write_artifact_paper_summary_filename(tmp_path):
    mod = _load_transparency_module()
    mod.write_artifact(
        paper_id="abc123",
        artifact_kind="paper_summary",
        content="## Strengths\nSolid empirical work.",
        log_dir=str(tmp_path),
    )
    assert (tmp_path / "abc123" / "paper_abc123_summary.md").exists()


def test_write_artifact_comment_draft_filename(tmp_path):
    mod = _load_transparency_module()
    mod.write_artifact(
        paper_id="abc123",
        artifact_kind="comment_draft",
        short_id="sid001",
        content="The claim in §3 lacks support.",
        log_dir=str(tmp_path),
    )
    assert (tmp_path / "abc123" / "comment_draft_abc123_sid001.md").exists()


def test_write_artifact_comment_trace_filename_and_schema(tmp_path):
    mod = _load_transparency_module()
    mod.write_artifact(
        paper_id="abc123",
        artifact_kind="comment_trace",
        short_id="sid001",
        source_phase="review",
        summary="Missing baseline",
        payload={"evidence": "Table 3", "decision_impact": "major concern"},
        log_dir=str(tmp_path),
    )
    path = tmp_path / "abc123" / "comment_trace_abc123_sid001.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["artifact_kind"] == "comment_trace"
    assert data["paper_id"] == "abc123"
    assert data["source_phase"] == "review"
    assert data["summary"] == "Missing baseline"
    assert data["payload"] == {"evidence": "Table 3", "decision_impact": "major concern"}
    assert "created_at" in data


def test_write_artifact_verdict_draft_filename(tmp_path):
    mod = _load_transparency_module()
    mod.write_artifact(
        paper_id="p1",
        artifact_kind="verdict_draft",
        short_id="v001",
        content="## Verdict\nScore: 6.5",
        log_dir=str(tmp_path),
    )
    assert (tmp_path / "p1" / "verdict_draft_p1_v001.md").exists()


def test_write_artifact_verdict_trace_filename_and_schema(tmp_path):
    mod = _load_transparency_module()
    mod.write_artifact(
        paper_id="p1",
        artifact_kind="verdict_trace",
        short_id="v001",
        summary="Weak accept",
        payload={"score_band": "weak_accept", "key_issues": ["missing ablation"]},
        log_dir=str(tmp_path),
    )
    path = tmp_path / "p1" / "verdict_trace_p1_v001.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["artifact_kind"] == "verdict_trace"
    assert data["paper_id"] == "p1"
    assert data["source_phase"] == "verdict"
    assert data["summary"] == "Weak accept"
    assert data["payload"]["score_band"] == "weak_accept"


def test_write_artifact_returns_github_url(tmp_path):
    mod = _load_transparency_module()
    url = mod.write_artifact(
        paper_id="p1",
        artifact_kind="comment_draft",
        short_id="sid1",
        content="draft",
        log_dir=str(tmp_path),
        github_repo="https://github.com/owner/repo",
    )
    assert url == "https://github.com/owner/repo/blob/main/logs/p1/comment_draft_p1_sid1.md"


def test_write_artifact_returns_placeholder_without_github(monkeypatch, tmp_path):
    monkeypatch.delenv("KOALA_GITHUB_REPO", raising=False)
    mod = _load_transparency_module()
    url = mod.write_artifact(
        paper_id="p1",
        artifact_kind="paper_summary",
        content="summary",
        log_dir=str(tmp_path),
    )
    assert url.startswith("TODO:")


def test_write_artifact_auto_short_id(tmp_path):
    mod = _load_transparency_module()
    mod.write_artifact(
        paper_id="p1",
        artifact_kind="comment_draft",
        content="draft content",
        log_dir=str(tmp_path),
    )
    files = list((tmp_path / "p1").iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("comment_draft_p1_")
    assert files[0].suffix == ".md"


def test_write_artifact_default_phase_review(tmp_path):
    mod = _load_transparency_module()
    mod.write_artifact(
        paper_id="p1",
        artifact_kind="comment_trace",
        short_id="s1",
        log_dir=str(tmp_path),
    )
    data = json.loads((tmp_path / "p1" / "comment_trace_p1_s1.json").read_text())
    assert data["source_phase"] == "review"


def test_write_artifact_default_phase_verdict(tmp_path):
    mod = _load_transparency_module()
    mod.write_artifact(
        paper_id="p1",
        artifact_kind="verdict_trace",
        short_id="v1",
        log_dir=str(tmp_path),
    )
    data = json.loads((tmp_path / "p1" / "verdict_trace_p1_v1.json").read_text())
    assert data["source_phase"] == "verdict"


def test_write_artifact_invalid_kind(tmp_path):
    mod = _load_transparency_module()
    with pytest.raises(ValueError, match="Unknown artifact_kind"):
        mod.write_artifact(
            paper_id="p1",
            artifact_kind="nonexistent_kind",
            content="test",
            log_dir=str(tmp_path),
        )
