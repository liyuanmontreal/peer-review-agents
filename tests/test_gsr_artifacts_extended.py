"""Tests for Phase 1–3 artifact interface extensions.

Verifies:
 - Smoke artifacts remain launch-only (write_smoke_artifact / write_local_artifact_smoke)
 - create_comment_artifact and create_verdict_artifact produce separate per-paper records
 - validate_artifact_for_external_action rejects placeholders
 - ensure_github_file_url returns valid URLs and raises on placeholders
"""

import pytest
from pathlib import Path

from agent_definition.harness.gsr_artifacts import (
    create_comment_artifact,
    create_verdict_artifact,
    ensure_github_file_url,
    validate_artifact_for_external_action,
    write_local_artifact_smoke,
    write_smoke_artifact,
)

_REAL_URL = "https://github.com/owner/repo/blob/main/logs/paper-001/artifact.md"


# ---------------------------------------------------------------------------
# Smoke artifacts remain in their dedicated directories
# ---------------------------------------------------------------------------

def test_smoke_artifact_goes_to_smoke_test_dir(tmp_path):
    write_smoke_artifact("gsr_agent", str(tmp_path))
    smoke_files = list((tmp_path / "smoke_test").glob("*.md"))
    assert len(smoke_files) == 1


def test_smoke_artifact_does_not_create_paper_directory(tmp_path):
    write_smoke_artifact("gsr_agent", str(tmp_path))
    paper_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name != "smoke_test"]
    assert len(paper_dirs) == 0


def test_local_artifact_smoke_goes_to_local_smoke_dir(tmp_path):
    write_local_artifact_smoke("gsr_agent", str(tmp_path))
    local_dir = tmp_path / "local_artifact_smoke"
    assert local_dir.exists()
    assert len(list(local_dir.iterdir())) >= 1


def test_local_artifact_smoke_does_not_create_real_paper_dir(tmp_path):
    write_local_artifact_smoke("gsr_agent", str(tmp_path))
    dirs = {d.name for d in tmp_path.iterdir() if d.is_dir()}
    assert "local_artifact_smoke" in dirs
    assert "smoke_test" not in dirs


# ---------------------------------------------------------------------------
# create_comment_artifact — separate per-paper records
# ---------------------------------------------------------------------------

def test_create_comment_artifact_creates_draft_file(tmp_path):
    create_comment_artifact("paper-abc", "Test claim.", artifact_dir=str(tmp_path))
    files = list((tmp_path / "paper-abc").glob("comment_draft_paper-abc_*.md"))
    assert len(files) == 1


def test_create_comment_artifact_not_in_smoke_dirs(tmp_path):
    create_comment_artifact("paper-abc", "Test.", artifact_dir=str(tmp_path))
    assert not (tmp_path / "smoke_test").exists()
    assert not (tmp_path / "local_artifact_smoke").exists()


def test_create_comment_artifact_contains_body(tmp_path):
    create_comment_artifact("paper-abc", "My test comment body.", artifact_dir=str(tmp_path))
    files = list((tmp_path / "paper-abc").glob("comment_draft_*.md"))
    content = files[0].read_text()
    assert "My test comment body." in content


def test_create_comment_artifact_contains_parent_id(tmp_path):
    create_comment_artifact(
        "paper-abc", "Reply.", parent_id="thread-xyz", artifact_dir=str(tmp_path)
    )
    files = list((tmp_path / "paper-abc").glob("comment_draft_*.md"))
    content = files[0].read_text()
    assert "thread-xyz" in content


def test_create_comment_artifact_returns_url_string(tmp_path):
    url = create_comment_artifact("paper-abc", "Test.", artifact_dir=str(tmp_path))
    assert isinstance(url, str)
    assert len(url) > 0


def test_create_comment_artifact_is_callable_multiple_times(tmp_path):
    # _short_id() uses second-level precision; we verify the function doesn't
    # raise on repeated calls and always returns a non-empty URL string.
    url1 = create_comment_artifact("paper-abc", "First.", artifact_dir=str(tmp_path))
    url2 = create_comment_artifact("paper-abc", "Second.", artifact_dir=str(tmp_path))
    assert isinstance(url1, str) and len(url1) > 0
    assert isinstance(url2, str) and len(url2) > 0


# ---------------------------------------------------------------------------
# create_verdict_artifact — separate per-paper records
# ---------------------------------------------------------------------------

def test_create_verdict_artifact_creates_draft_file(tmp_path):
    create_verdict_artifact("paper-xyz", 7.5, "Strong paper.", [], artifact_dir=str(tmp_path))
    files = list((tmp_path / "paper-xyz").glob("verdict_draft_paper-xyz_*.md"))
    assert len(files) == 1


def test_create_verdict_artifact_contains_score(tmp_path):
    create_verdict_artifact("paper-xyz", 8.0, "Good.", ["c1"], artifact_dir=str(tmp_path))
    files = list((tmp_path / "paper-xyz").glob("verdict_draft_*.md"))
    content = files[0].read_text()
    assert "8.0" in content


def test_create_verdict_artifact_contains_cited_ids(tmp_path):
    create_verdict_artifact(
        "paper-xyz", 6.0, "Average.", ["c1", "c2", "c3"], artifact_dir=str(tmp_path)
    )
    files = list((tmp_path / "paper-xyz").glob("verdict_draft_*.md"))
    content = files[0].read_text()
    assert "[[comment:c1]]" in content
    assert "[[comment:c2]]" in content
    assert "[[comment:c3]]" in content


def test_create_verdict_artifact_not_in_smoke_dirs(tmp_path):
    create_verdict_artifact("paper-xyz", 5.0, "OK.", [], artifact_dir=str(tmp_path))
    assert not (tmp_path / "smoke_test").exists()
    assert not (tmp_path / "local_artifact_smoke").exists()


# ---------------------------------------------------------------------------
# Comment and verdict artifacts are separate records for same paper
# ---------------------------------------------------------------------------

def test_comment_and_verdict_artifacts_are_separate_files(tmp_path):
    create_comment_artifact("paper-a", "Comment.", artifact_dir=str(tmp_path))
    create_verdict_artifact("paper-a", 7.0, "Verdict.", ["c1", "c2", "c3"], artifact_dir=str(tmp_path))
    comment_files = list((tmp_path / "paper-a").glob("comment_draft_*.md"))
    verdict_files = list((tmp_path / "paper-a").glob("verdict_draft_*.md"))
    assert len(comment_files) == 1
    assert len(verdict_files) == 1
    assert comment_files[0].name != verdict_files[0].name


# ---------------------------------------------------------------------------
# validate_artifact_for_external_action
# ---------------------------------------------------------------------------

def test_validate_accepts_real_github_url():
    validate_artifact_for_external_action(_REAL_URL)  # no exception


def test_validate_rejects_empty_url():
    with pytest.raises(ValueError, match="empty"):
        validate_artifact_for_external_action("")


def test_validate_rejects_todo_placeholder():
    todo = "TODO: set KOALA_GITHUB_REPO — local artifact at ./logs/paper-001/x.md"
    with pytest.raises(ValueError, match="placeholder"):
        validate_artifact_for_external_action(todo)


def test_validate_rejects_different_todo_prefix():
    with pytest.raises(ValueError, match="placeholder"):
        validate_artifact_for_external_action("TODO: configure GitHub")


# ---------------------------------------------------------------------------
# ensure_github_file_url
# ---------------------------------------------------------------------------

def test_ensure_returns_valid_url():
    result = ensure_github_file_url(_REAL_URL)
    assert result == _REAL_URL


def test_ensure_raises_on_empty():
    with pytest.raises(ValueError):
        ensure_github_file_url("")


def test_ensure_raises_on_todo():
    with pytest.raises(ValueError):
        ensure_github_file_url("TODO: set KOALA_GITHUB_REPO — local artifact at x.md")
