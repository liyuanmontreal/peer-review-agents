"""Rule-based moderation placeholder.

Rejects obvious personal attacks, explicit misconduct accusations, and
profanity-style language. The list is conservative and easy to extend.
Do not use for nuanced semantic analysis — that belongs in a future LLM
moderation step.

Returns (passes: bool, reason: str). When passes is False the comment must
not be posted.
"""

from __future__ import annotations

_BLOCKED_PHRASES: tuple[str, ...] = (
    # Explicit misconduct accusations (require strong evidence before using)
    "fraud",
    "fabricated",
    "fabrication",
    "plagiarism",
    "plagiarized",
    "misconduct",
    # Personal attacks
    "stupid",
    "idiotic",
    "incompetent",
    "worthless",
    "garbage paper",
    "trash paper",
    # Unsupported rhetorical absolutes
    "fatal flaw",
    "completely wrong",
    "obviously wrong",
    "blatantly false",
)


def check_moderation(text: str) -> tuple[bool, str]:
    """Check text for disallowed language.

    Returns:
        (True, "") if the text passes moderation.
        (False, reason) if a blocked phrase is detected.
    """
    lower = text.lower()
    for phrase in _BLOCKED_PHRASES:
        if phrase in lower:
            return False, f"Blocked phrase detected: {phrase!r}"
    return True, ""
