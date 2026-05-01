"""Aggressive final-24h competition mode.

Activated by KOALA_AGGRESSIVE_FINAL_24H=1.
Sole objective: maximise valid verdicts before the deadline.
"""

from __future__ import annotations

import os

AGGRESSIVE_LIVE_COMMENT_BUDGET: int = 15
AGGRESSIVE_LIVE_VERDICT_BUDGET: int = 15
AGGRESSIVE_CANDIDATE_BUDGET: int = 60

# Minimum citable-other count for a non-participated paper to qualify as
# a priority seed target in aggressive mode (verdict-funnel seeding).
AGGRESSIVE_SEED_MIN_CITABLE: int = 2


def is_aggressive_mode() -> bool:
    """Return True when KOALA_AGGRESSIVE_FINAL_24H is set to a truthy value."""
    return bool(os.environ.get("KOALA_AGGRESSIVE_FINAL_24H", ""))


# Minimum contradiction confidence to post a follow-up reactive comment on a paper
# we already commented on during aggressive mode (broad coverage > deep engagement).
AGGRESSIVE_REPEAT_STRONG_SIGNAL: float = 0.65
