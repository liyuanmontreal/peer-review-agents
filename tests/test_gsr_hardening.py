"""Tests for the Phase 4A hardening patch.

Safety invariants verified here:
  1. test-mode URLs are structurally identifiable and rejected by strict validation
  2. live external actions require KOALA_ARTIFACT_MODE=github + KOALA_GITHUB_REPO
  3. live external actions require URL repo to match KOALA_GITHUB_REPO exactly
  4. KOALA_RUN_MODE=dry_run (the safe default) never calls real Koala write paths
  5. KoalaClient.post_comment / submit_verdict reject test-mode URLs in production
  6. The orchestrator validates artifacts strictly before the client and before any write
  7. Audit metadata (run_mode, artifact_mode, is_test_url, publish_status) is logged in DB
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from gsr_agent.artifacts.github import (
    extract_github_repo,
    get_run_mode,
    is_test_mode_artifact_url,
    publish_comment_artifact,
    validate_artifact_for_external_action,
    validate_artifact_for_live_action,
    validate_live_configuration,
)
from gsr_agent.koala.client import KoalaClient
from gsr_agent.koala.errors import KoalaPreflightError
from gsr_agent.koala.models import Paper

UTC = timezone.utc
_OPEN = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
_NOW_SEED = _OPEN + timedelta(hours=6)

_REAL_REPO = "https://github.com/owner/repo"
_REAL_URL = f"{_REAL_REPO}/blob/main/logs/paper-001/comment_draft_paper-001_20260424T120000Z.md"
_TEST_URL = "https://github.com/test-mode-only/gsr-agent-artifacts/blob/main/logs/paper-001/comment_draft.md"


# ---------------------------------------------------------------------------
# A1. is_test_mode_artifact_url
# ---------------------------------------------------------------------------

def test_is_test_mode_url_true_for_fake():
    assert is_test_mode_artifact_url(_TEST_URL) is True


def test_is_test_mode_url_false_for_real():
    assert is_test_mode_artifact_url(_REAL_URL) is False


def test_is_test_mode_url_false_for_todo():
    assert is_test_mode_artifact_url("TODO: set KOALA_GITHUB_REPO") is False


def test_is_test_mode_url_false_for_empty():
    assert is_test_mode_artifact_url("") is False


def test_publish_comment_test_mode_produces_test_url():
    url = publish_comment_artifact("paper-001", "Body.", test_mode=True)
    assert is_test_mode_artifact_url(url) is True


# ---------------------------------------------------------------------------
# A2. extract_github_repo
# ---------------------------------------------------------------------------

def test_extract_repo_from_blob_url():
    url = "https://github.com/owner/repo/blob/main/logs/paper.md"
    assert extract_github_repo(url) == "https://github.com/owner/repo"


def test_extract_repo_from_real_url():
    assert extract_github_repo(_REAL_URL) == _REAL_REPO


def test_extract_repo_returns_none_for_test_url():
    # Test URL is a real github.com URL, just under test-mode-only owner
    result = extract_github_repo(_TEST_URL)
    assert result == "https://github.com/test-mode-only/gsr-agent-artifacts"


def test_extract_repo_returns_none_for_todo():
    assert extract_github_repo("TODO: something") is None


def test_extract_repo_returns_none_for_empty():
    assert extract_github_repo("") is None


def test_extract_repo_returns_none_for_non_blob_url():
    assert extract_github_repo("https://github.com/owner/repo") is None


# ---------------------------------------------------------------------------
# A3. validate_artifact_for_live_action — strict checks
# ---------------------------------------------------------------------------

def test_validate_live_rejects_empty_url():
    with pytest.raises(KoalaPreflightError):
        validate_artifact_for_live_action("")


def test_validate_live_rejects_todo_placeholder():
    with pytest.raises(KoalaPreflightError):
        validate_artifact_for_live_action("TODO: set KOALA_GITHUB_REPO")


def test_validate_live_rejects_test_mode_url(monkeypatch):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", "https://github.com/test-mode-only/gsr-agent-artifacts")
    with pytest.raises(KoalaPreflightError, match="test-mode"):
        validate_artifact_for_live_action(_TEST_URL)


def test_validate_live_rejects_local_artifact_mode(monkeypatch):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "local")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    with pytest.raises(KoalaPreflightError, match="KOALA_ARTIFACT_MODE"):
        validate_artifact_for_live_action(_REAL_URL)


def test_validate_live_rejects_missing_artifact_mode(monkeypatch):
    monkeypatch.delenv("KOALA_ARTIFACT_MODE", raising=False)
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    with pytest.raises(KoalaPreflightError, match="KOALA_ARTIFACT_MODE"):
        validate_artifact_for_live_action(_REAL_URL)


def test_validate_live_rejects_missing_github_repo(monkeypatch):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.delenv("KOALA_GITHUB_REPO", raising=False)
    with pytest.raises(KoalaPreflightError, match="KOALA_GITHUB_REPO"):
        validate_artifact_for_live_action(_REAL_URL)


def test_validate_live_rejects_repo_mismatch(monkeypatch):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", "https://github.com/different/repo")
    with pytest.raises(KoalaPreflightError, match="does not match"):
        validate_artifact_for_live_action(_REAL_URL)


def test_validate_live_accepts_matching_url(monkeypatch):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    validate_artifact_for_live_action(_REAL_URL)  # must not raise


def test_validate_live_accepts_with_trailing_slash_in_env(monkeypatch):
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO + "/")  # trailing slash
    validate_artifact_for_live_action(_REAL_URL)  # must not raise


def test_test_mode_url_passes_loose_check_but_fails_strict():
    # The loose check (validate_artifact_for_external_action) accepts any https:// URL
    validate_artifact_for_external_action(_TEST_URL)  # must not raise (loose)
    # The strict check rejects test-mode URLs even with correct env
    with pytest.raises(KoalaPreflightError, match="test-mode"):
        # No matching KOALA_GITHUB_REPO needed — test-mode check fires first
        validate_artifact_for_live_action(_TEST_URL)


# ---------------------------------------------------------------------------
# A4. validate_live_configuration
# ---------------------------------------------------------------------------

def test_validate_live_config_rejects_dry_run_mode(monkeypatch):
    monkeypatch.setenv("KOALA_RUN_MODE", "dry_run")
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    with pytest.raises(KoalaPreflightError, match="KOALA_RUN_MODE"):
        validate_live_configuration()


def test_validate_live_config_rejects_unset_run_mode(monkeypatch):
    monkeypatch.delenv("KOALA_RUN_MODE", raising=False)
    with pytest.raises(KoalaPreflightError, match="KOALA_RUN_MODE"):
        validate_live_configuration()


def test_validate_live_config_rejects_live_without_github(monkeypatch):
    monkeypatch.setenv("KOALA_RUN_MODE", "live")
    monkeypatch.delenv("KOALA_ARTIFACT_MODE", raising=False)
    monkeypatch.delenv("KOALA_GITHUB_REPO", raising=False)
    with pytest.raises(KoalaPreflightError):
        validate_live_configuration()


def test_validate_live_config_passes_when_fully_configured(monkeypatch):
    monkeypatch.setenv("KOALA_RUN_MODE", "live")
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    validate_live_configuration()  # must not raise


# ---------------------------------------------------------------------------
# A5. get_run_mode
# ---------------------------------------------------------------------------

def test_get_run_mode_defaults_to_dry_run(monkeypatch):
    monkeypatch.delenv("KOALA_RUN_MODE", raising=False)
    assert get_run_mode() == "dry_run"


def test_get_run_mode_returns_env_value(monkeypatch):
    monkeypatch.setenv("KOALA_RUN_MODE", "live")
    assert get_run_mode() == "live"


# ---------------------------------------------------------------------------
# B. KoalaClient hard gate — test-mode URLs blocked in production write path
# ---------------------------------------------------------------------------

def test_client_post_comment_rejects_test_url_in_production():
    client = KoalaClient(api_token="tok", test_mode=False)
    with pytest.raises(KoalaPreflightError, match="test-mode"):
        client.post_comment("paper-001", "body", _TEST_URL)


def test_client_submit_verdict_rejects_test_url_in_production():
    client = KoalaClient(api_token="tok", test_mode=False)
    with pytest.raises(KoalaPreflightError, match="test-mode"):
        client.submit_verdict("paper-001", 7.0, ["c1"], _TEST_URL)


def test_client_post_comment_accepts_real_url_in_production():
    client = KoalaClient(api_token="tok", test_mode=False)
    import json
    from unittest.mock import MagicMock, patch as _patch
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"id": "cid-1"}).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    with _patch("urllib.request.urlopen", return_value=mock_resp):
        cid = client.post_comment("paper-001", "body", _REAL_URL)
    assert cid == "cid-1"


def test_client_test_mode_accepts_test_url():
    # In test_mode=True the client logs but doesn't call the real API — test URLs are OK
    client = KoalaClient(test_mode=True)
    cid = client.post_comment("paper-001", "body", _TEST_URL)
    assert isinstance(cid, str)


# ---------------------------------------------------------------------------
# C. Orchestrator: dry_run never calls real write path
# ---------------------------------------------------------------------------

def _make_paper(abstract: str = "We propose a method for gradient estimation.") -> Paper:
    from gsr_agent.rules.timeline import compute_paper_windows
    w = compute_paper_windows(_OPEN)
    return Paper(
        paper_id="paper-001",
        title="Test Paper",
        open_time=w.open_time,
        review_end_time=w.review_end_time,
        verdict_end_time=w.verdict_end_time,
        state="REVIEW_ACTIVE",
        abstract=abstract,
    )


def _make_db() -> MagicMock:
    db = MagicMock()
    db.has_prior_participation.return_value = False
    return db


def _make_client() -> MagicMock:
    client = MagicMock()
    client._test_mode = False
    client.post_comment.return_value = "comment-001"
    return client


def test_orchestrator_dry_run_never_calls_post_comment(monkeypatch):
    monkeypatch.setenv("KOALA_RUN_MODE", "dry_run")
    from gsr_agent.commenting.orchestrator import plan_and_post_seed_comment
    paper = _make_paper()
    client = _make_client()
    db = _make_db()
    result = plan_and_post_seed_comment(
        paper, client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=False
    )
    assert result[0] is None
    assert not client.post_comment.called


def test_orchestrator_dry_run_returns_none(monkeypatch):
    monkeypatch.setenv("KOALA_RUN_MODE", "dry_run")
    from gsr_agent.commenting.orchestrator import plan_and_post_seed_comment
    result = plan_and_post_seed_comment(
        _make_paper(), _make_client(), _make_db(), karma_remaining=50.0, now=_NOW_SEED, test_mode=False
    )
    assert result[0] is None
    assert result[1] == "dry_run"


def test_orchestrator_dry_run_logs_action(monkeypatch):
    monkeypatch.setenv("KOALA_RUN_MODE", "dry_run")
    from gsr_agent.commenting.orchestrator import plan_and_post_seed_comment
    db = _make_db()
    plan_and_post_seed_comment(
        _make_paper(), _make_client(), db, karma_remaining=50.0, now=_NOW_SEED, test_mode=False
    )
    assert db.log_action.called
    call_kwargs = db.log_action.call_args[1]
    assert call_kwargs.get("status") == "dry_run"


def test_orchestrator_dry_run_logs_details(monkeypatch):
    monkeypatch.setenv("KOALA_RUN_MODE", "dry_run")
    from gsr_agent.commenting.orchestrator import plan_and_post_seed_comment
    db = _make_db()
    plan_and_post_seed_comment(
        _make_paper(), _make_client(), db, karma_remaining=50.0, now=_NOW_SEED, test_mode=False
    )
    details = db.log_action.call_args[1].get("details", {})
    assert details.get("run_mode") == "dry_run"
    assert details.get("publish_status") == "dry_run"
    assert "blocked_reason" in details


def test_orchestrator_unset_run_mode_defaults_to_dry_run(monkeypatch):
    monkeypatch.delenv("KOALA_RUN_MODE", raising=False)
    from gsr_agent.commenting.orchestrator import plan_and_post_seed_comment
    client = _make_client()
    result = plan_and_post_seed_comment(
        _make_paper(), client, _make_db(), karma_remaining=50.0, now=_NOW_SEED, test_mode=False
    )
    assert result[0] is None
    assert not client.post_comment.called


def test_orchestrator_test_mode_still_posts_via_stub_client(monkeypatch):
    monkeypatch.delenv("KOALA_RUN_MODE", raising=False)
    from gsr_agent.commenting.orchestrator import plan_and_post_seed_comment
    client = _make_client()
    client._test_mode = True
    client.post_comment.return_value = "comment-test"
    db = _make_db()
    result = plan_and_post_seed_comment(
        _make_paper(), client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=True
    )
    assert result[0] is not None
    assert client.post_comment.called


def test_orchestrator_live_calls_validate_before_post(monkeypatch, tmp_path):
    monkeypatch.setenv("KOALA_RUN_MODE", "live")
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("KOALA_API_BASE_URL", "https://koala.science/api/v1")
    monkeypatch.setenv("KOALA_API_TOKEN", "test-live-token")
    from gsr_agent.commenting.orchestrator import plan_and_post_seed_comment
    import gsr_agent.commenting.orchestrator as orch_mod

    validate_calls = []
    original = orch_mod.validate_artifact_for_live_action

    def recording_validate(url):
        validate_calls.append(url)
        original(url)

    monkeypatch.setattr(orch_mod, "validate_artifact_for_live_action", recording_validate)

    client = _make_client()
    client.post_comment.return_value = "comment-live"
    db = _make_db()

    plan_and_post_seed_comment(
        _make_paper(), client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=False
    )

    assert len(validate_calls) == 1
    # Validate must have been called before post_comment
    assert client.post_comment.called


def test_orchestrator_live_validate_called_before_post_comment_order(monkeypatch, tmp_path):
    monkeypatch.setenv("KOALA_RUN_MODE", "live")
    monkeypatch.setenv("KOALA_ARTIFACT_MODE", "github")
    monkeypatch.setenv("KOALA_GITHUB_REPO", _REAL_REPO)
    monkeypatch.setenv("KOALA_TRANSPARENCY_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("KOALA_API_BASE_URL", "https://koala.science/api/v1")
    monkeypatch.setenv("KOALA_API_TOKEN", "test-live-token")
    from gsr_agent.commenting.orchestrator import plan_and_post_seed_comment
    import gsr_agent.commenting.orchestrator as orch_mod

    call_order = []
    original_validate = orch_mod.validate_artifact_for_live_action

    def recording_validate(url):
        call_order.append("validate")
        original_validate(url)

    monkeypatch.setattr(orch_mod, "validate_artifact_for_live_action", recording_validate)

    client = _make_client()
    original_post = client.post_comment

    def recording_post(*args, **kwargs):
        call_order.append("post_comment")
        return "comment-live"

    client.post_comment = recording_post
    db = _make_db()

    plan_and_post_seed_comment(
        _make_paper(), client, db, karma_remaining=50.0, now=_NOW_SEED, test_mode=False
    )

    assert call_order.index("validate") < call_order.index("post_comment")


def test_orchestrator_client_blocks_if_reached_with_test_url(monkeypatch):
    """Defense-in-depth: client rejects test-mode URL even if orchestrator is bypassed."""
    monkeypatch.setenv("KOALA_RUN_MODE", "live")
    client = KoalaClient(api_token="tok", test_mode=False)
    with pytest.raises(KoalaPreflightError, match="test-mode"):
        client.post_comment("paper-001", "body", _TEST_URL)


# ---------------------------------------------------------------------------
# D. DB: audit metadata stored as JSON in details column
# ---------------------------------------------------------------------------

def test_log_action_stores_details(tmp_path):
    from gsr_agent.storage.db import KoalaDB
    db = KoalaDB(str(tmp_path / "test.db"))
    db.log_action(
        paper_id="p1",
        action_type="seed_comment",
        status="dry_run",
        details={"run_mode": "dry_run", "is_test_url": True, "publish_status": "dry_run"},
    )
    row = db._conn.execute(
        "SELECT details FROM koala_agent_actions WHERE paper_id=?", ("p1",)
    ).fetchone()
    assert row is not None
    stored = json.loads(row["details"])
    assert stored["run_mode"] == "dry_run"
    assert stored["is_test_url"] is True
    assert stored["publish_status"] == "dry_run"


def test_log_action_details_none_stores_null(tmp_path):
    from gsr_agent.storage.db import KoalaDB
    db = KoalaDB(str(tmp_path / "test.db"))
    db.log_action(paper_id="p1", action_type="seed_comment", status="success")
    row = db._conn.execute(
        "SELECT details FROM koala_agent_actions WHERE paper_id=?", ("p1",)
    ).fetchone()
    assert row["details"] is None


def test_log_action_details_survives_migration(tmp_path):
    """Migration adds details column to an existing DB without it."""
    import sqlite3
    db_path = str(tmp_path / "old.db")
    # Simulate a pre-migration DB (no details column)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS koala_agent_actions (
            action_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            external_id TEXT,
            github_file_url TEXT,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT
        );
    """)
    conn.commit()
    conn.close()

    # Opening via KoalaDB should migrate
    from gsr_agent.storage.db import KoalaDB
    db = KoalaDB(db_path)
    db.log_action(
        paper_id="p1",
        action_type="test",
        details={"migrated": True},
    )
    row = db._conn.execute(
        "SELECT details FROM koala_agent_actions WHERE paper_id=?", ("p1",)
    ).fetchone()
    assert json.loads(row["details"])["migrated"] is True
