"""Verdict eligibility state machine.

A verdict may only be submitted when ALL of these hard gates pass:
  1. The agent has participated in discussion on the paper.
  2. The paper is currently in its verdict window (48–72h).
  3. At least 5 distinct other agents have citable comments.
  4. An audit artifact is created and its github_file_url is available.
  5. Internal score confidence >= min_confidence threshold.
  6. The verdict has not already been submitted.

The state machine tracks eight states per paper, from NOT_PARTICIPATED
through SUBMITTED (or EXPIRED / SKIPPED_BY_POLICY).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .timeline import PaperPhase, _ensure_utc, get_paper_phase

MIN_DISTINCT_OTHER_AGENTS: int = 5
MIN_VERDICT_CONFIDENCE: float = 0.6


class EligibilityState(str, Enum):
    NOT_PARTICIPATED = "NOT_PARTICIPATED"
    PARTICIPATED_BUT_NOT_ENOUGH_OTHERS = "PARTICIPATED_BUT_NOT_ENOUGH_OTHERS"
    ENOUGH_OTHERS_BUT_NOT_IN_VERDICT_WINDOW = "ENOUGH_OTHERS_BUT_NOT_IN_VERDICT_WINDOW"
    ELIGIBLE_LOW_CONFIDENCE = "ELIGIBLE_LOW_CONFIDENCE"
    ELIGIBLE_READY = "ELIGIBLE_READY"
    SUBMITTED = "SUBMITTED"
    SKIPPED_BY_POLICY = "SKIPPED_BY_POLICY"
    EXPIRED = "EXPIRED"


@dataclass
class VerdictEligibilityInput:
    paper_id: str
    has_our_participation: bool
    distinct_citable_other_agents: int
    open_time: datetime
    audit_artifact_ready: bool
    internal_score_confidence: float
    submitted: bool = False
    skipped: bool = False


def can_submit_verdict(
    state: VerdictEligibilityInput,
    now: datetime,
    min_confidence: float = MIN_VERDICT_CONFIDENCE,
) -> bool:
    """Hard gate: True only when ALL verdict submission conditions are met.

    Does not check the skipped flag — SKIPPED_BY_POLICY is a soft policy
    marker tracked separately in compute_eligibility_state.
    """
    now = _ensure_utc(now)
    phase = get_paper_phase(now, state.open_time)
    return (
        state.has_our_participation
        and phase == PaperPhase.VERDICT_ACTIVE
        and state.distinct_citable_other_agents >= MIN_DISTINCT_OTHER_AGENTS
        and state.audit_artifact_ready
        and state.internal_score_confidence >= min_confidence
        and not state.submitted
    )


def compute_eligibility_state(
    state: VerdictEligibilityInput,
    now: datetime,
    min_confidence: float = MIN_VERDICT_CONFIDENCE,
) -> tuple[EligibilityState, str]:
    """Return (eligibility_state, human_readable_reason).

    Reasons are empty strings for terminal success states.
    States are evaluated in priority order: terminal states first, then
    blocking conditions in the order they must be resolved.
    """
    now = _ensure_utc(now)
    phase = get_paper_phase(now, state.open_time)

    if state.submitted:
        return EligibilityState.SUBMITTED, "Verdict already submitted."

    if phase == PaperPhase.EXPIRED:
        return EligibilityState.EXPIRED, "Paper lifecycle has expired (>72h from open_time)."

    if state.skipped:
        return EligibilityState.SKIPPED_BY_POLICY, "Skipped by agent policy."

    if not state.has_our_participation:
        return (
            EligibilityState.NOT_PARTICIPATED,
            "Agent has not participated in discussion on this paper.",
        )

    if state.distinct_citable_other_agents < MIN_DISTINCT_OTHER_AGENTS:
        return (
            EligibilityState.PARTICIPATED_BUT_NOT_ENOUGH_OTHERS,
            f"Only {state.distinct_citable_other_agents}/{MIN_DISTINCT_OTHER_AGENTS} "
            "distinct citable other agents so far.",
        )

    if phase != PaperPhase.VERDICT_ACTIVE or not state.audit_artifact_ready:
        reason = (
            "Audit artifact not yet created."
            if phase == PaperPhase.VERDICT_ACTIVE and not state.audit_artifact_ready
            else f"Paper is in phase {phase.value}, not in verdict window (48–72h) yet."
        )
        return EligibilityState.ENOUGH_OTHERS_BUT_NOT_IN_VERDICT_WINDOW, reason

    if state.internal_score_confidence < min_confidence:
        return (
            EligibilityState.ELIGIBLE_LOW_CONFIDENCE,
            f"Score confidence {state.internal_score_confidence:.2f} "
            f"is below threshold {min_confidence:.2f}.",
        )

    return EligibilityState.ELIGIBLE_READY, ""
