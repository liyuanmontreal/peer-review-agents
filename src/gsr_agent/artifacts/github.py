"""Artifact creation and publish interface for external Koala actions.

Re-exports canonical implementations from agent_definition.harness.gsr_artifacts
and adds two tiers of validation + run-mode helpers.

Validation tiers
----------------
Loose (validate_artifact_for_external_action / ensure_github_file_url):
  Used by unit tests and internal staging — rejects only empty strings and
  TODO placeholders. Accepts test-mode URLs because tests need a URL that
  passes basic format checks.

Strict (validate_artifact_for_live_action):
  Used immediately before any real Koala write. Additionally rejects:
    - test-mode artifact URLs (github.com/test-mode-only/...)
    - any URL whose repo doesn't match KOALA_GITHUB_REPO exactly
  Also requires KOALA_ARTIFACT_MODE=github and KOALA_GITHUB_REPO to be set.

Run mode
--------
KOALA_RUN_MODE = dry_run (default) | live

  dry_run  — the safe default; orchestration prepares everything but never
             calls post_comment / submit_verdict. Use this for local testing
             with real Koala credentials.

  live     — full production mode; all strict validators are active and
             real Koala API writes are performed.

Environment variables
---------------------
KOALA_RUN_MODE             dry_run | live              default: dry_run
KOALA_ARTIFACT_MODE        local | github              default: local
KOALA_GITHUB_REPO          https://github.com/…        required for live
KOALA_GITHUB_LOGS_PATH     sub-path inside repo        default: logs
KOALA_TRANSPARENCY_LOG_DIR local dir for artifacts     default: ./logs
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import List, Optional

from agent_definition.harness.gsr_artifacts import (  # noqa: F401
    create_comment_artifact,
    create_verdict_artifact,
    ensure_github_file_url,
    validate_artifact_for_external_action,
)

# Backward-compatible alias for validate_artifact_for_external_action.
validate_github_like_url = validate_artifact_for_external_action

from ..koala.errors import KoalaPreflightError

__all__ = [
    # Re-exports (loose tier)
    "create_comment_artifact",
    "create_verdict_artifact",
    "validate_artifact_for_external_action",
    "validate_github_like_url",
    "ensure_github_file_url",
    # URL introspection
    "is_test_mode_artifact_url",
    "extract_github_repo",
    "normalize_github_repo",
    # Strict tier
    "validate_artifact_for_live_action",
    "validate_live_configuration",
    # Publish helpers
    "is_github_publish_configured",
    "get_github_file_url_for_artifact",
    "publish_comment_artifact",
    "publish_verdict_artifact",
    # Run-mode helper
    "get_run_mode",
]

# Unique owner/path that marks all test-mode-generated artifact URLs.
_TEST_GITHUB_OWNER = "test-mode-only"
_TEST_GITHUB_BASE = f"https://github.com/{_TEST_GITHUB_OWNER}/gsr-agent-artifacts/blob/main/logs"
_TEST_MODE_URL_MARKER = f"github.com/{_TEST_GITHUB_OWNER}/"

# Pattern to extract https://github.com/<owner>/<repo> from a blob URL.
_GITHUB_BLOB_RE = re.compile(r"(https://github\.com/[^/]+/[^/]+)/blob/")
# Pattern to extract owner/repo from a full GitHub repo URL (no deeper path).
_GITHUB_REPO_URL_RE = re.compile(r"^https://github\.com/([^/]+/[^/]+?)/?$")


def _short_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# URL introspection
# ---------------------------------------------------------------------------

def normalize_github_repo(value: str) -> Optional[str]:
    """Normalise a GitHub repo reference to *owner/repo* form.

    Accepts either ``owner/repo`` or ``https://github.com/owner/repo`` (with
    optional trailing slash).  Returns None for empty input, None input, or
    any value that cannot be parsed into exactly two path segments.
    """
    if not value:
        return None
    value = value.strip()
    if not value:
        return None

    if value.startswith("https://"):
        m = _GITHUB_REPO_URL_RE.match(value)
        return m.group(1) if m else None

    parts = value.split("/")
    if len(parts) == 2 and parts[0] and parts[1]:
        return value
    return None


def is_test_mode_artifact_url(url: str) -> bool:
    """True if *url* was produced by publish_*_artifact(test_mode=True).

    These URLs are structurally valid GitHub URLs but point to a repo
    owned by 'test-mode-only' and must never reach a real Koala write.
    """
    return _TEST_MODE_URL_MARKER in url


def extract_github_repo(url: str) -> Optional[str]:
    """Extract https://github.com/<owner>/<repo> from a GitHub blob URL.

    Returns None when the URL is not a recognisable GitHub blob URL.
    """
    m = _GITHUB_BLOB_RE.search(url)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Publish configuration
# ---------------------------------------------------------------------------

def is_github_publish_configured() -> bool:
    """True when both KOALA_ARTIFACT_MODE=github and KOALA_GITHUB_REPO are set."""
    return (
        os.environ.get("KOALA_ARTIFACT_MODE", "") == "github"
        and bool(os.environ.get("KOALA_GITHUB_REPO", ""))
    )


def get_github_file_url_for_artifact(paper_id: str, filename: str) -> str:
    """Build a deterministic GitHub blob URL from env vars."""
    repo = os.environ.get("KOALA_GITHUB_REPO", "").rstrip("/")
    logs_path = os.environ.get("KOALA_GITHUB_LOGS_PATH", "logs")
    return f"{repo}/blob/main/{logs_path}/{paper_id}/{filename}"


def get_run_mode() -> str:
    """Return KOALA_RUN_MODE, defaulting to 'dry_run' when unset."""
    return os.environ.get("KOALA_RUN_MODE", "dry_run")


# ---------------------------------------------------------------------------
# Strict validation
# ---------------------------------------------------------------------------

def validate_artifact_for_live_action(url: str) -> None:
    """Strict validator for real external Koala writes.

    Enforces all four conditions in order:
      1. URL is non-empty and not a TODO placeholder (loose check).
      2. URL is not a test-mode artifact URL.
      3. KOALA_ARTIFACT_MODE == "github".
      4. KOALA_GITHUB_REPO is set.
      5. URL's GitHub repo matches KOALA_GITHUB_REPO exactly.

    Raises:
        KoalaPreflightError: on any violation (wraps ValueError from the loose check).
    """
    try:
        validate_artifact_for_external_action(url)
    except ValueError as exc:
        raise KoalaPreflightError(str(exc)) from exc

    if is_test_mode_artifact_url(url):
        raise KoalaPreflightError(
            f"validate_artifact_for_live_action: test-mode artifact URL must not "
            f"be used for real external actions: {url!r}"
        )

    artifact_mode = os.environ.get("KOALA_ARTIFACT_MODE", "")
    if artifact_mode != "github":
        raise KoalaPreflightError(
            f"validate_artifact_for_live_action: KOALA_ARTIFACT_MODE must be 'github' "
            f"for live external actions (current: {artifact_mode!r})"
        )

    configured_repo_raw = os.environ.get("KOALA_GITHUB_REPO", "")
    configured_repo = normalize_github_repo(configured_repo_raw)
    if not configured_repo:
        raise KoalaPreflightError(
            "validate_artifact_for_live_action: KOALA_GITHUB_REPO must be set "
            "for live external actions."
        )

    url_repo_full = extract_github_repo(url)
    url_repo = normalize_github_repo(url_repo_full) if url_repo_full else None
    if url_repo is None or url_repo != configured_repo:
        raise KoalaPreflightError(
            f"validate_artifact_for_live_action: URL repo {url_repo_full!r} does not match "
            f"configured KOALA_GITHUB_REPO {configured_repo_raw!r}. "
            "Artifact must originate from the configured repository."
        )


def validate_live_configuration() -> None:
    """Assert the environment is fully configured for live Koala writes.

    Raises:
        KoalaPreflightError: if KOALA_RUN_MODE != 'live' or GitHub publish is
                             not configured.
    """
    run_mode = get_run_mode()
    if run_mode != "live":
        raise KoalaPreflightError(
            f"validate_live_configuration: KOALA_RUN_MODE must be 'live' for "
            f"external writes (current: {run_mode!r}). "
            "Set KOALA_RUN_MODE=live to enable."
        )
    if not is_github_publish_configured():
        raise KoalaPreflightError(
            "validate_live_configuration: KOALA_ARTIFACT_MODE=github and "
            "KOALA_GITHUB_REPO must both be set for live external writes."
        )


# ---------------------------------------------------------------------------
# Publish helpers
# ---------------------------------------------------------------------------

def publish_comment_artifact(
    paper_id: str,
    body: str,
    *,
    parent_id: Optional[str] = None,
    test_mode: bool = False,
) -> str:
    """Write a comment artifact and return a valid github_file_url.

    test_mode=True → fake-but-valid test URL; no file written, no env required.
    test_mode=False → requires KOALA_ARTIFACT_MODE=github + KOALA_GITHUB_REPO.

    Raises:
        KoalaPreflightError: in production when GitHub publish is not configured.
    """
    if test_mode:
        filename = f"comment_draft_{paper_id}_{_short_id()}.md"
        return f"{_TEST_GITHUB_BASE}/{paper_id}/{filename}"

    if not is_github_publish_configured():
        raise KoalaPreflightError(
            "publish_comment_artifact: KOALA_ARTIFACT_MODE=github and "
            "KOALA_GITHUB_REPO must both be set for production artifact publishing. "
            "Pass test_mode=True for offline testing."
        )

    return create_comment_artifact(
        paper_id,
        body,
        parent_id=parent_id,
        artifact_dir=os.environ.get("KOALA_TRANSPARENCY_LOG_DIR", "./logs"),
        github_repo=os.environ.get("KOALA_GITHUB_REPO"),
    )


def publish_verdict_artifact(
    paper_id: str,
    score: float,
    body: str,
    cited_ids: List[str],
    *,
    test_mode: bool = False,
) -> str:
    """Write a verdict artifact and return a valid github_file_url.

    test_mode=True → fake-but-valid test URL; no file written, no env required.
    test_mode=False → requires KOALA_ARTIFACT_MODE=github + KOALA_GITHUB_REPO.

    Raises:
        KoalaPreflightError: in production when GitHub publish is not configured.
    """
    if test_mode:
        filename = f"verdict_draft_{paper_id}_{_short_id()}.md"
        return f"{_TEST_GITHUB_BASE}/{paper_id}/{filename}"

    if not is_github_publish_configured():
        raise KoalaPreflightError(
            "publish_verdict_artifact: KOALA_ARTIFACT_MODE=github and "
            "KOALA_GITHUB_REPO must both be set for production artifact publishing. "
            "Pass test_mode=True for offline testing."
        )

    return create_verdict_artifact(
        paper_id,
        score,
        body,
        cited_ids,
        artifact_dir=os.environ.get("KOALA_TRANSPARENCY_LOG_DIR", "./logs"),
        github_repo=os.environ.get("KOALA_GITHUB_REPO"),
    )
