"""Action preflight checks — the last gate before any external write.

Every comment and verdict must pass a complete preflight before the
KoalaClient is called. Any failure raises KoalaPreflightError, which
must be treated as a hard block — the action must not proceed.

Comment preflight checks:
  1. Paper is in REVIEW_ACTIVE phase (commenting is only allowed 0–48h).
  2. Karma is sufficient to cover the action cost.
  3. github_file_url is present and not a placeholder.
  4. Moderation check passes.
  5. Comment body is non-empty.

Verdict preflight checks:
  1. github_file_url is present and not a placeholder.
  2. Score is in [0.0, 10.0].
  3. cited_comment_ids contains at least MIN_DISTINCT_OTHER_AGENTS entries.
  4. can_submit_verdict(...) returns True (hard eligibility gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from .karma import ActionType, can_afford, get_action_cost
from .moderation import check_moderation
from .timeline import PaperPhase, get_paper_phase
from .verdict_eligibility import (
    MIN_DISTINCT_OTHER_AGENTS,
    VerdictEligibilityInput,
    can_submit_verdict,
    compute_eligibility_state,
)
from ..koala.errors import KoalaPreflightError

_PLACEHOLDER_PREFIXES = ("TODO:", "test-artifact://")


def _require_github_url(github_file_url: str, context: str) -> None:
    if not github_file_url:
        raise KoalaPreflightError(f"{context}: github_file_url is required but was empty.")
    for prefix in _PLACEHOLDER_PREFIXES:
        if github_file_url.startswith(prefix):
            raise KoalaPreflightError(
                f"{context}: github_file_url is a placeholder ({github_file_url!r}). "
                "Publish the artifact to GitHub before posting."
            )


@dataclass
class CommentPreflightInput:
    paper_id: str
    body: str
    github_file_url: str
    open_time: datetime
    now: datetime
    karma_remaining: float
    has_prior_participation: bool
    action_type: ActionType = "comment"


@dataclass
class VerdictPreflightInput:
    paper_id: str
    score: float
    cited_comment_ids: List[str]
    github_file_url: str
    eligibility: VerdictEligibilityInput
    now: datetime
    min_confidence: float = 0.6


def preflight_comment_action(inp: CommentPreflightInput) -> None:
    """Run all preflight checks for a comment action.

    Raises:
        KoalaPreflightError: on the first check that fails.
    """
    phase = get_paper_phase(inp.now, inp.open_time)
    if phase != PaperPhase.REVIEW_ACTIVE:
        raise KoalaPreflightError(
            f"Comment not allowed: paper {inp.paper_id!r} is in phase "
            f"{phase.value} (comments are only allowed during REVIEW_ACTIVE, 0–48h)."
        )

    cost = get_action_cost(inp.action_type, inp.has_prior_participation)
    if not can_afford(inp.karma_remaining, cost):
        raise KoalaPreflightError(
            f"Insufficient karma for paper {inp.paper_id!r}: "
            f"need {cost}, have {inp.karma_remaining:.2f}."
        )

    _require_github_url(inp.github_file_url, f"post_comment(paper_id={inp.paper_id!r})")

    passes, reason = check_moderation(inp.body)
    if not passes:
        raise KoalaPreflightError(
            f"Moderation check failed for paper {inp.paper_id!r}: {reason}"
        )

    if not inp.body.strip():
        raise KoalaPreflightError(
            f"Comment body is empty for paper {inp.paper_id!r}."
        )


def preflight_verdict_action(inp: VerdictPreflightInput) -> None:
    """Run all preflight checks for a verdict action.

    Raises:
        KoalaPreflightError: on the first check that fails.
    """
    _require_github_url(inp.github_file_url, f"submit_verdict(paper_id={inp.paper_id!r})")

    if not (0.0 <= inp.score <= 10.0):
        raise KoalaPreflightError(
            f"Score {inp.score} is out of range [0, 10] for paper {inp.paper_id!r}."
        )

    distinct_count = len(set(inp.cited_comment_ids))
    if distinct_count < MIN_DISTINCT_OTHER_AGENTS:
        raise KoalaPreflightError(
            f"Verdict for paper {inp.paper_id!r} requires at least "
            f"{MIN_DISTINCT_OTHER_AGENTS} distinct cited comments, got {distinct_count}."
        )

    if not can_submit_verdict(inp.eligibility, inp.now, inp.min_confidence):
        _, reason = compute_eligibility_state(inp.eligibility, inp.now, inp.min_confidence)
        raise KoalaPreflightError(
            f"Verdict eligibility gate failed for paper {inp.paper_id!r}: {reason}"
        )
