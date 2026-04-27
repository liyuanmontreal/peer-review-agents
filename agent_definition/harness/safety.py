"""Competition safety guards for the harness.

Blocks unsafe bash/Python code patterns that could leak API tokens or probe
unverified Koala API endpoints.
"""
from __future__ import annotations

import re

# Regex patterns for unsafe script content
_UNSAFE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bcurl\b[^\n]*-[a-zA-Z]*v"), "curl_verbose"),
    (re.compile(r"\bcurl\b[^\n]*-[a-zA-Z]*i(?:[^a-zA-Z]|$)"), "curl_print_headers"),
    (re.compile(r"\bAuthorization\b[^\n]*\bBearer\b", re.IGNORECASE), "auth_header_in_command"),
    (re.compile(r"\bcs_[A-Za-z0-9_-]{10,}"), "koala_token_literal"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}"), "anthropic_key_literal"),
]

# Verified comment endpoints
_ALLOWED_COMMENT_READ_RE = re.compile(
    r"^/api/v1/comments/paper/[0-9a-f-]+(\?.*)?$"
)
_ALLOWED_COMMENT_POST_PATHS = {"/api/v1/comments/", "/api/v1/comments"}

# Blocked comment endpoint patterns — probing/guessing
_BLOCKED_ENDPOINT_PATTERNS: list[re.Pattern] = [
    re.compile(r"/comments/\?"),
    re.compile(r"/actors/[^/]+/comments"),
    re.compile(r"/papers/[^/?]+\?[^?]*include_comments"),
]


def check_bash_safety(code: str) -> tuple[bool, str]:
    """Return (is_safe, reason). If not safe, reason names the violation."""
    for pattern, reason in _UNSAFE_PATTERNS:
        if pattern.search(code):
            return False, reason
    return True, ""


def is_allowed_comments_endpoint(path: str, method: str = "GET") -> tuple[bool, str]:
    """Validate a Koala comments endpoint against the approved set.

    Allowed paths:
      GET  /api/v1/comments/paper/{paper_id}   (with optional query string)
      POST /api/v1/comments/

    Returns:
        (is_allowed, reason) — if not allowed, reason describes the violation.
    """
    method = method.upper()

    if method == "GET":
        if _ALLOWED_COMMENT_READ_RE.match(path):
            return True, ""
        for blocked in _BLOCKED_ENDPOINT_PATTERNS:
            if blocked.search(path):
                return False, f"blocked_endpoint_probe: {path!r}"
        if re.search(r"/comments", path, re.IGNORECASE):
            return False, f"unknown_comments_endpoint: {path!r}"
        return True, ""

    if method == "POST":
        if path in _ALLOWED_COMMENT_POST_PATHS:
            return True, ""
        for blocked in _BLOCKED_ENDPOINT_PATTERNS:
            if blocked.search(path):
                return False, f"blocked_endpoint_probe: {path!r}"
        if re.search(r"/comments", path, re.IGNORECASE):
            return False, f"unknown_comments_endpoint: {path!r}"
        return True, ""

    return True, ""
