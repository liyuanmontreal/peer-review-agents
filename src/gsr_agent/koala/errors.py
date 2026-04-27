"""Exceptions for the Koala client and preflight layer."""


class KoalaPreflightError(ValueError):
    """Raised when an action fails the preflight checklist.

    This is a hard error — the action must not proceed when this is raised.
    """


class KoalaAPIError(RuntimeError):
    """Raised when a Koala API call fails with an unexpected response."""


class KoalaRateLimitError(KoalaAPIError):
    """Raised when the Koala API returns a 429 rate-limit response."""


class KoalaWindowClosedError(KoalaAPIError):
    """Raised when the Koala API returns 409 (competition window already closed)."""
