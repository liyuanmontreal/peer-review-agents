"""Tests for Phase 4A artifact publish path (artifacts/github.py)."""

import pytest

from gsr_agent.artifacts.github import (
    get_github_file_url_for_artifact,
    is_github_publish_configured,
    publish_comment_artifact,
    publish_verdict_artifact,
    validate_artifact_for_external_action,
)
from gsr_agent.koala.errors import KoalaPreflightError

_REAL_REPO = "https://github.com/owner/repo"


# ---------------------------------------------------------------------------
# is_github_publish_configured
# ---------------------------------------------------------------------------

def test_not_configured_by_default(monkeypatch):
    monkeypatch.delenv("KOALA_ARTIFACT_MODE", raising=False)
    monkeypatch.delenv("KOALA_GITHUB_REPO", raising=False)
    assert is_github_publish_configured() is False


def test_not_configured_mode_only(monkeypatch):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.delenv("KOALA_GITHUB_REPO", raising=False)
    assert is_github_publish_configured() is False


def test_not_configured_repo_only(monkeypatch):
    monkeypatch.delenv("KOALA_ARTIFACT_MODE", raising=False)
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    assert is_github_publish_configured() is False


def test_configured_when_both_set(monkeypatch):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    assert is_github_publish_configured() is True


def test_not_configured_when_mode_is_local(monkeypatch):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "local")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    assert is_github_publish_configured() is False


# ---------------------------------------------------------------------------
# get_github_file_url_for_artifact
# ---------------------------------------------------------------------------

def test_get_github_file_url_uses_repo_env(monkeypatch):
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    monkeypatch.delenv("KOALA_GITHUB_LOGS_PATH", raising=False)
    url = get_github_file_url_for_artifact("paper-001", "artifact.md")
    assert url.startswith(_REAL_REPO)
    assert "paper-001" in url
    assert "artifact.md" in url


def test_get_github_file_url_default_logs_path(monkeypatch):
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    monkeypatch.delenv("KOALA_GITHUB_LOGS_PATH", raising=False)
    url = get_github_file_url_for_artifact("p1", "f.md")
    assert "/logs/" in url


def test_get_github_file_url_custom_logs_path(monkeypatch):
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    monkeypatch.setenv("KOALA_GITHUB_LOGS_PATH", "artifacts/reviews")
    url = get_github_file_url_for_artifact("p1", "f.md")
    assert "artifacts/reviews" in url


# ---------------------------------------------------------------------------
# publish_comment_artifact — test_mode=True
# ---------------------------------------------------------------------------

def test_publish_comment_test_mode_returns_github_url():
    url = publish_comment_artifact("paper-001", "Test comment.", test_mode=True)
    assert url.startswith("https://github.com/")


def test_publish_comment_test_mode_url_contains_paper_id():
    url = publish_comment_artifact("paper-abc", "Body.", test_mode=True)
    assert "paper-abc" in url


def test_publish_comment_test_mode_url_is_not_todo():
    url = publish_comment_artifact("paper-001", "Test.", test_mode=True)
    assert not url.startswith("TODO:")


def test_publish_comment_test_mode_passes_validation():
    url = publish_comment_artifact("paper-001", "Test.", test_mode=True)
    validate_artifact_for_external_action(url)  # must not raise


def test_publish_comment_test_mode_works_without_env_vars(monkeypatch):
    monkeypatch.delenv("KOALA_ARTIFACT_MODE", raising=False)
    monkeypatch.delenv("KOALA_GITHUB_REPO", raising=False)
    url = publish_comment_artifact("paper-001", "Test.", test_mode=True)
    assert url.startswith("https://github.com/")


def test_publish_comment_test_mode_with_parent_id():
    url = publish_comment_artifact("paper-001", "Reply.", parent_id="thread-x", test_mode=True)
    assert url.startswith("https://github.com/")


# ---------------------------------------------------------------------------
# publish_comment_artifact — unconfigured production raises
# ---------------------------------------------------------------------------

def test_publish_comment_raises_when_not_configured(monkeypatch):
    monkeypatch.delenv("KOALA_ARTIFACT_MODE", raising=False)
    monkeypatch.delenv("KOALA_GITHUB_REPO", raising=False)
    with pytest.raises(KoalaPreflightError, match="KOALA"):
        publish_comment_artifact("paper-001", "Test.")


def test_publish_comment_raises_with_local_mode(monkeypatch):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "local")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    with pytest.raises(KoalaPreflightError):
        publish_comment_artifact("paper-001", "Test.")


# ---------------------------------------------------------------------------
# publish_comment_artifact — production path
# ---------------------------------------------------------------------------

def test_publish_comment_production_returns_github_url(monkeypatch, tmp_path):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    url = publish_comment_artifact("paper-001", "A real comment.", test_mode=False)
    assert url.startswith(_REAL_REPO)
    assert "paper-001" in url


def test_publish_comment_production_writes_file(monkeypatch, tmp_path):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    publish_comment_artifact("paper-001", "My comment.", test_mode=False)
    files = list((tmp_path / "paper-001").glob("comment_draft_*.md"))
    assert len(files) == 1
    assert "My comment." in files[0].read_text()


def test_publish_comment_production_url_passes_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    url = publish_comment_artifact("paper-001", "Test.", test_mode=False)
    validate_artifact_for_external_action(url)  # must not raise


# ---------------------------------------------------------------------------
# publish_verdict_artifact — test_mode
# ---------------------------------------------------------------------------

def test_publish_verdict_test_mode_returns_github_url():
    url = publish_verdict_artifact("paper-001", 7.5, "Good.", ["c1", "c2", "c3"], test_mode=True)
    assert url.startswith("https://github.com/")


def test_publish_verdict_test_mode_url_contains_paper_id():
    url = publish_verdict_artifact("paper-xyz", 6.0, "OK.", [], test_mode=True)
    assert "paper-xyz" in url


def test_publish_verdict_test_mode_url_is_not_todo():
    url = publish_verdict_artifact("paper-001", 7.0, "Verdict.", ["c1"], test_mode=True)
    assert not url.startswith("TODO:")


def test_publish_verdict_test_mode_passes_validation():
    url = publish_verdict_artifact("paper-001", 7.0, "Verdict.", ["c1"], test_mode=True)
    validate_artifact_for_external_action(url)  # must not raise


def test_publish_verdict_test_mode_works_without_env_vars(monkeypatch):
    monkeypatch.delenv("KOALA_ARTIFACT_MODE", raising=False)
    monkeypatch.delenv("KOALA_GITHUB_REPO", raising=False)
    url = publish_verdict_artifact("paper-001", 7.0, "Verdict.", [], test_mode=True)
    assert url.startswith("https://github.com/")


# ---------------------------------------------------------------------------
# publish_verdict_artifact — unconfigured production raises
# ---------------------------------------------------------------------------

def test_publish_verdict_raises_when_not_configured(monkeypatch):
    monkeypatch.delenv("KOALA_ARTIFACT_MODE", raising=False)
    monkeypatch.delenv("KOALA_GITHUB_REPO", raising=False)
    with pytest.raises(KoalaPreflightError, match="KOALA"):
        publish_verdict_artifact("paper-001", 7.0, "Verdict.", ["c1"])


# ---------------------------------------------------------------------------
# publish_verdict_artifact — production path
# ---------------------------------------------------------------------------

def test_publish_verdict_production_writes_file(monkeypatch, tmp_path):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    publish_verdict_artifact("paper-001", 8.0, "Strong.", ["c1", "c2"], test_mode=False)
    files = list((tmp_path / "paper-001").glob("verdict_draft_*.md"))
    assert len(files) == 1
    assert "8.0" in files[0].read_text()


def test_publish_verdict_production_url_starts_with_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    url = publish_verdict_artifact("paper-001", 7.0, "Good.", ["c1"], test_mode=False)
    assert url.startswith(_REAL_REPO)
