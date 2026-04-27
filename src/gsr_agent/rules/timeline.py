"""Competition timeline utilities.

Every paper has a 72-hour rolling lifecycle:
  0–48h  review / discussion phase   (PaperPhase.REVIEW_ACTIVE)
  48–72h verdict phase                (PaperPhase.VERDICT_ACTIVE)

Within those 72 hours, five micro-phases guide agent behaviour:
  0–12h   SEED_WINDOW        — cold-start first comments
  12–36h  BUILD_WINDOW       — selective new seeds + reactive follow-ups
  36–48h  LOCK_IN_WINDOW     — higher threshold; prefer follow-ups
  48–60h  ELIGIBILITY_WINDOW — recompute distinct-agent coverage
  60–72h  SUBMISSION_WINDOW  — submit only high-confidence verdicts

All computations use UTC internally. Callers may pass aware or naive
datetimes; naive datetimes are assumed UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

REVIEW_DURATION_H: int = 48
DELIBERATION_DURATION_H: int = 24   # verdict / deliberation phase
VERDICT_DURATION_H: int = 72        # total competition window (review + deliberation)
SAFETY_BUFFER_S: int = 600          # 10-minute buffer before acting near window close


class PaperPhase(str, Enum):
    """Coarse per-paper lifecycle phase."""
    NEW = "NEW"                       # paper not yet open (pre-release)
    REVIEW_ACTIVE = "REVIEW_ACTIVE"   # 0–48h from open_time
    VERDICT_ACTIVE = "VERDICT_ACTIVE" # 48–72h from open_time
    EXPIRED = "EXPIRED"               # after 72h


class MicroPhase(str, Enum):
    """Fine-grained per-paper micro-phase for action policy."""
    SEED_WINDOW = "SEED_WINDOW"               # 0–12h
    BUILD_WINDOW = "BUILD_WINDOW"             # 12–36h
    LOCK_IN_WINDOW = "LOCK_IN_WINDOW"         # 36–48h
    ELIGIBILITY_WINDOW = "ELIGIBILITY_WINDOW" # 48–60h
    SUBMISSION_WINDOW = "SUBMISSION_WINDOW"   # 60–72h
    EXPIRED = "EXPIRED"                       # after 72h


@dataclass
class PhaseWindow:
    """Phase and remaining time for a paper at a specific instant.

    phase:        "comment" | "verdict" | "expired"
    ends_at:      when the current phase window closes (UTC)
    seconds_left: seconds until window close; negative means already closed
    """
    phase: str
    ends_at: datetime
    seconds_left: float


@dataclass
class PaperWindows:
    open_time: datetime
    review_end_time: datetime
    verdict_end_time: datetime


def compute_paper_windows(open_time: datetime) -> PaperWindows:
    """Return review_end_time and verdict_end_time for a paper."""
    open_time = _ensure_utc(open_time)
    return PaperWindows(
        open_time=open_time,
        review_end_time=open_time + timedelta(hours=REVIEW_DURATION_H),
        verdict_end_time=open_time + timedelta(hours=VERDICT_DURATION_H),
    )


def get_paper_phase(now: datetime, open_time: datetime) -> PaperPhase:
    """Return the coarse lifecycle phase for a paper at the given time."""
    now = _ensure_utc(now)
    windows = compute_paper_windows(open_time)
    if now < windows.open_time:
        return PaperPhase.NEW
    if now <= windows.review_end_time:
        return PaperPhase.REVIEW_ACTIVE
    if now <= windows.verdict_end_time:
        return PaperPhase.VERDICT_ACTIVE
    return PaperPhase.EXPIRED


def get_micro_phase(now: datetime, open_time: datetime) -> MicroPhase:
    """Return the fine-grained micro-phase for a paper at the given time."""
    now = _ensure_utc(now)
    open_time = _ensure_utc(open_time)
    elapsed_h = (now - open_time).total_seconds() / 3600
    if elapsed_h < 12:
        return MicroPhase.SEED_WINDOW
    if elapsed_h < 36:
        return MicroPhase.BUILD_WINDOW
    if elapsed_h < 48:
        return MicroPhase.LOCK_IN_WINDOW
    if elapsed_h < 60:
        return MicroPhase.ELIGIBILITY_WINDOW
    if elapsed_h < 72:
        return MicroPhase.SUBMISSION_WINDOW
    return MicroPhase.EXPIRED


def compute_phase_window(
    now: datetime,
    paper_state: str,
    open_time: datetime,
    deliberating_at: Optional[datetime] = None,
) -> PhaseWindow:
    """Compute remaining window for a paper, using timestamps not just status.

    Args:
        now:             current UTC time
        paper_state:     "REVIEW_ACTIVE" | "VERDICT_ACTIVE" | "EXPIRED" | "NEW"
        open_time:       when the comment window opened (created_at / open_time)
        deliberating_at: when deliberation started; None falls back to open_time + 48h

    Returns:
        PhaseWindow with phase, ends_at, and seconds_left.
        seconds_left <= 0 means the window has already closed.
    """
    now = _ensure_utc(now)
    open_time = _ensure_utc(open_time)

    if paper_state == "REVIEW_ACTIVE":
        ends_at = open_time + timedelta(hours=REVIEW_DURATION_H)
        return PhaseWindow(
            phase="comment",
            ends_at=ends_at,
            seconds_left=(ends_at - now).total_seconds(),
        )

    if paper_state == "VERDICT_ACTIVE":
        if deliberating_at is not None:
            delib_start = _ensure_utc(deliberating_at)
        else:
            delib_start = open_time + timedelta(hours=REVIEW_DURATION_H)
        ends_at = delib_start + timedelta(hours=DELIBERATION_DURATION_H)
        return PhaseWindow(
            phase="verdict",
            ends_at=ends_at,
            seconds_left=(ends_at - now).total_seconds(),
        )

    ends_at = open_time + timedelta(hours=VERDICT_DURATION_H)
    return PhaseWindow(
        phase="expired",
        ends_at=ends_at,
        seconds_left=(ends_at - now).total_seconds(),
    )


def _ensure_utc(dt: datetime) -> datetime:
    """Return dt as a UTC-aware datetime. Naive datetimes are assumed UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
