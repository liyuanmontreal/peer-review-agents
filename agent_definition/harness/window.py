"""
window.py

Phase-window helpers for the competition harness.
No external dependencies — importable without reva or httpx.
"""
import re
from datetime import datetime, timedelta, timezone

REVIEW_WINDOW_H = 48
VERDICT_WINDOW_H = 24
SAFETY_BUFFER_S = 600

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def koala_window_state(paper: dict, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    status = paper.get("status", "")
    created = _parse_dt(paper.get("created_at", ""))
    deliberating_at = _parse_dt(
        paper.get("deliberating_at") or paper.get("deliberation_started_at")
    )

    if created is None:
        return {"phase": "expired", "open": False, "seconds_left": 0.0, "ends_at": None}

    if status == "in_review":
        ends_at = created + timedelta(hours=REVIEW_WINDOW_H)
        phase = "comment"
    elif status == "deliberating":
        delib_start = deliberating_at or (created + timedelta(hours=REVIEW_WINDOW_H))
        ends_at = delib_start + timedelta(hours=VERDICT_WINDOW_H)
        phase = "verdict"
    else:
        return {"phase": "expired", "open": False, "seconds_left": 0.0, "ends_at": None}

    seconds_left = (ends_at - now).total_seconds()
    return {
        "phase": phase,
        "open": seconds_left > SAFETY_BUFFER_S,
        "seconds_left": seconds_left,
        "ends_at": ends_at.isoformat(),
    }


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
