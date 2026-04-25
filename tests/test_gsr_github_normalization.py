"""Tests for Phase 4B GitHub repo normalization.

validate_artifact_for_live_action must accept KOALA_GITHUB_REPO in both
  owner/repo  and  https://github.com/owner/repo  forms.
"""

import pytest

from gsr_agent.artifacts.github import (
    normalize_github_repo,
    validate_artifact_for_live_action,
    validate_github_like_url,
    validate_artifact_for_external_action,
)
from gsr_agent.koala.errors import KoalaPreflightError

_REAL_REPO_FULL = "https://github.com/owner/repo"
_REAL_REPO_SHORT = "owner/repo"
_REAL_URL = f"{_REAL_REPO_FULL}/blob/main/logs/paper-001/artifact.md"


# ---------------------------------------------------------------------------
# normalize_github_repo
# ---------------------------------------------------------------------------

def test_normalize_short_form():
    assert normalize_github_repo("owner/repo") == "owner/repo"


def test_normalize_full_url():
    assert normalize_github_repo("https://github.com/owner/repo") == "owner/repo"


def test_normalize_full_url_trailing_slash():
    assert normalize_github_repo("https://github.com/owner/repo/") == "owner/repo"


def test_normalize_full_url_with_branch_or_path_is_None():
    # Only owner/repo should be accepted; deeper paths are not a repo reference
    result = normalize_github_repo("https://github.com/owner/repo/blob/main/file.md")
    assert result is None


def test_normalize_empty_returns_none():
    assert normalize_github_repo("") is None


def test_normalize_none_input():
    assert normalize_github_repo(None) is None  # type: ignore[arg-type]


def test_normalize_single_segment_returns_none():
    assert normalize_github_repo("just-owner") is None


def test_normalize_too_many_segments_without_scheme():
    # "a/b/c" has 3 parts — not a valid repo
    assert normalize_github_repo("a/b/c") is None


def test_normalize_with_whitespace_stripped():
    assert normalize_github_repo("  owner/repo  ") == "owner/repo"


def test_normalize_hyphenated_names():
    assert normalize_github_repo("koala-science/gsr-agent-artifacts") == "koala-science/gsr-agent-artifacts"


def test_normalize_full_url_hyphenated():
    assert normalize_github_repo("https://github.com/koala-science/gsr-agent") == "koala-science/gsr-agent"


# ---------------------------------------------------------------------------
# validate_artifact_for_live_action — both KOALA_GITHUB_REPO formats accepted
# ---------------------------------------------------------------------------

def test_validate_live_accepts_short_repo_env(monkeypatch):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO_SHORT)  # "owner/repo"
    validate_artifact_for_live_action(_REAL_URL)  # must not raise


def test_validate_live_accepts_full_url_repo_env(monkeypatch):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO_FULL)  # "https://github.com/owner/repo"
    validate_artifact_for_live_action(_REAL_URL)  # must not raise


def test_validate_live_short_and_full_are_equivalent(monkeypatch):
    """Both env formats match the same artifact URL."""
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")

    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO_SHORT)
    validate_artifact_for_live_action(_REAL_URL)  # no raise

    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO_FULL)
    validate_artifact_for_live_action(_REAL_URL)  # no raise


def test_validate_live_rejects_mismatched_short_repo(monkeypatch):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", "different/repo")
    with pytest.raises(KoalaPreflightError, match="does not match"):
        validate_artifact_for_live_action(_REAL_URL)


def test_validate_live_rejects_mismatched_full_url_repo(monkeypatch):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", "https://github.com/different/repo")
    with pytest.raises(KoalaPreflightError, match="does not match"):
        validate_artifact_for_live_action(_REAL_URL)


def test_validate_live_invalid_repo_env_rejects(monkeypatch):
    """An invalid/unparseable KOALA_GITHUB_REPO is treated as missing."""
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", "not-a-repo")
    with pytest.raises(KoalaPreflightError):
        validate_artifact_for_live_action(_REAL_URL)


# ---------------------------------------------------------------------------
# validate_github_like_url — backward-compatible alias
# ---------------------------------------------------------------------------

def test_alias_exists_and_works():
    """validate_github_like_url is the loose-tier alias."""
    validate_github_like_url(_REAL_URL)  # must not raise (non-placeholder URL)


def test_alias_rejects_empty():
    with pytest.raises(ValueError):
        validate_github_like_url("")


def test_alias_rejects_todo():
    with pytest.raises(ValueError):
        validate_github_like_url("TODO: set repo")


def test_alias_same_as_original_function():
    """Both names accept the same URL."""
    validate_artifact_for_external_action(_REAL_URL)
    validate_github_like_url(_REAL_URL)  # no exception from either
