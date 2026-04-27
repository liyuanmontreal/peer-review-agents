"""Typed dataclasses for all Koala platform objects used by the v4 agent.

These are separate from the koala-gsr-agent platform models and reflect the
v4 design: papers carry explicit timeline windows, comments carry citability
flags, and payloads are typed structs rather than raw dicts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Koala API status → timeline state
_STATUS_MAP: Dict[str, str] = {
    "in_review": "REVIEW_ACTIVE",
    "deliberating": "VERDICT_ACTIVE",
    "reviewed": "EXPIRED",
    "closed": "EXPIRED",
}

# Candidate field names the Koala API might use for paper release time
_OPEN_TIME_FIELDS = (
    "open_time", "opened_at", "created_at",
    "release_time", "published_at", "submission_date",
)


def _parse_datetime(raw: Any) -> Optional[datetime]:
    if not raw or not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    paper_id: str
    title: str
    open_time: datetime
    review_end_time: datetime
    verdict_end_time: datetime
    state: str                          # NEW | REVIEW_ACTIVE | VERDICT_ACTIVE | EXPIRED
    pdf_url: str = ""
    local_pdf_path: Optional[str] = None
    last_synced_at: Optional[datetime] = None
    deliberating_at: Optional[datetime] = None  # when deliberation phase started; None = use fallback
    # Content fields populated by the Koala API or PDF indexer
    abstract: str = ""
    full_text: str = ""
    domains: List[str] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Paper":
        """Build a Paper from a Koala API response dict."""
        from ..rules.timeline import compute_paper_windows  # lazy import

        paper_id = data.get("id") or data.get("paper_id", "")
        title = data.get("title", "")
        status = data.get("status", "in_review")
        state = _STATUS_MAP.get(status, "REVIEW_ACTIVE")

        open_time: Optional[datetime] = None
        for field_name in _OPEN_TIME_FIELDS:
            open_time = _parse_datetime(data.get(field_name))
            if open_time:
                break
        if open_time is None:
            log.warning(
                "[Paper.from_api] paper %r: could not find open_time from fields %s; "
                "using current UTC time as estimate",
                paper_id, _OPEN_TIME_FIELDS,
            )
            open_time = datetime.now(timezone.utc)

        deliberating_at = (
            _parse_datetime(data.get("deliberating_at"))
            or _parse_datetime(data.get("deliberation_started_at"))
        )

        windows = compute_paper_windows(open_time)
        return cls(
            paper_id=paper_id,
            title=title,
            open_time=windows.open_time,
            review_end_time=windows.review_end_time,
            verdict_end_time=windows.verdict_end_time,
            state=state,
            pdf_url=data.get("pdf_url", ""),
            abstract=data.get("abstract", ""),
            full_text=data.get("full_text", ""),
            domains=data.get("domains", []),
            deliberating_at=deliberating_at,
        )


# ---------------------------------------------------------------------------
# Comment
# ---------------------------------------------------------------------------

@dataclass
class Comment:
    comment_id: str
    paper_id: str
    author_agent_id: str
    text: str
    created_at: datetime
    thread_id: Optional[str] = None
    parent_id: Optional[str] = None
    is_ours: bool = False
    is_citable: bool = False

    @classmethod
    def from_api(cls, data: Dict[str, Any], paper_id: str = "") -> "Comment":
        """Build a Comment from a Koala API response dict."""
        created_at = _parse_datetime(data.get("created_at")) or datetime.now(timezone.utc)
        return cls(
            comment_id=data.get("id") or data.get("comment_id", ""),
            paper_id=data.get("paper_id", paper_id),
            author_agent_id=data.get("author_id") or data.get("author_agent_id", "unknown"),
            text=data.get("content_markdown") or data.get("text", ""),
            created_at=created_at,
            thread_id=data.get("thread_id"),
            parent_id=data.get("parent_id"),
            is_ours=False,
            is_citable=bool(data.get("is_citable", False)),
        )


# ---------------------------------------------------------------------------
# Action payloads
# ---------------------------------------------------------------------------

@dataclass
class PostCommentPayload:
    paper_id: str
    body: str
    github_file_url: str
    thread_id: Optional[str] = None
    parent_id: Optional[str] = None


@dataclass
class SubmitVerdictPayload:
    paper_id: str
    score: float
    cited_comment_ids: List[str] = field(default_factory=list)
    github_file_url: str = ""
    bad_contribution_agent_id: Optional[str] = None
