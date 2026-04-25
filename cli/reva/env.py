"""Environment accessors for reva."""

import os

DEFAULT_KOALA_BASE_URL = "https://koala.science"


def koala_base_url() -> str:
    """Koala base URL. Honors $KOALA_BASE_URL (empty → default), strips trailing slashes."""
    value = os.environ.get("KOALA_BASE_URL") or DEFAULT_KOALA_BASE_URL
    return value.rstrip("/")


def koala_write_enabled() -> bool:
    """Whether real Koala write operations (post_comment, post_verdict) are enabled.

    Returns True only when KOALA_WRITE_ENABLED=true (case-insensitive).
    Defaults to False when unset — safe mode is the default.
    """
    return os.environ.get("KOALA_WRITE_ENABLED", "").lower() == "true"


def koala_github_repo() -> str:
    """GitHub repo URL used to build transparency log URLs. Empty string if unset."""
    return os.environ.get("KOALA_GITHUB_REPO", "")
