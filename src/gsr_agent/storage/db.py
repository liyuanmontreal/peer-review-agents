"""SQLite persistence for gsr_agent v4 competition state.

Follows the same patterns as koala-gsr-agent/storage/db.py:
- WAL journaling for concurrency
- ISO-8601 UTC timestamps
- schema initialised from schema.sql on first connect
- upsert semantics for idempotent sync
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..koala.models import Comment, Paper

log = logging.getLogger(__name__)

_SCHEMA_FILE = Path(__file__).parent / "schema.sql"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dt_to_str(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


class KoalaDB:
    def __init__(self, db_path: str = "./workspace/koala_agent.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA_FILE.read_text())
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Apply safe, additive schema migrations for existing databases."""
        try:
            self._conn.execute(
                "ALTER TABLE koala_agent_actions ADD COLUMN details TEXT"
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists — normal for new or already-migrated DBs

    # ------------------------------------------------------------------
    # Papers
    # ------------------------------------------------------------------

    def upsert_paper(self, paper: Paper) -> None:
        self._conn.execute(
            """INSERT INTO koala_papers
               (paper_id, title, open_time, review_end_time, verdict_end_time,
                state, pdf_url, local_pdf_path, last_synced_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(paper_id) DO UPDATE SET
                   title=excluded.title,
                   open_time=excluded.open_time,
                   review_end_time=excluded.review_end_time,
                   verdict_end_time=excluded.verdict_end_time,
                   state=excluded.state,
                   pdf_url=excluded.pdf_url,
                   local_pdf_path=excluded.local_pdf_path,
                   last_synced_at=excluded.last_synced_at""",
            (
                paper.paper_id,
                paper.title,
                _dt_to_str(paper.open_time),
                _dt_to_str(paper.review_end_time),
                _dt_to_str(paper.verdict_end_time),
                paper.state,
                paper.pdf_url,
                paper.local_pdf_path,
                _utcnow(),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Comments
    # ------------------------------------------------------------------

    def upsert_comment(self, comment: Comment) -> None:
        created_at = (
            _dt_to_str(comment.created_at)
            if isinstance(comment.created_at, datetime)
            else str(comment.created_at)
        )
        self._conn.execute(
            """INSERT INTO koala_comments
               (comment_id, paper_id, thread_id, parent_id, author_agent_id,
                text, created_at, is_ours, is_citable)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(comment_id) DO UPDATE SET
                   is_ours=excluded.is_ours,
                   is_citable=excluded.is_citable""",
            (
                comment.comment_id,
                comment.paper_id,
                comment.thread_id,
                comment.parent_id,
                comment.author_agent_id,
                comment.text,
                created_at,
                int(comment.is_ours),
                int(comment.is_citable),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Agent actions (append-only audit log)
    # ------------------------------------------------------------------

    def log_action(
        self,
        paper_id: str,
        action_type: str,
        *,
        github_file_url: Optional[str] = None,
        external_id: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        cursor = self._conn.execute(
            """INSERT INTO koala_agent_actions
               (paper_id, action_type, external_id, github_file_url,
                created_at, status, error_message, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                paper_id, action_type, external_id, github_file_url,
                _utcnow(), status, error_message,
                json.dumps(details) if details is not None else None,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Karma ledger (immutable)
    # ------------------------------------------------------------------

    def record_karma(
        self,
        paper_id: str,
        action_type: str,
        cost: float,
        karma_before: float,
        karma_after: float,
    ) -> None:
        self._conn.execute(
            """INSERT INTO koala_karma_ledger
               (paper_id, action_type, cost, karma_before, karma_after, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (paper_id, action_type, cost, karma_before, karma_after, _utcnow()),
        )
        self._conn.commit()

    def get_karma_spent(self, paper_id: Optional[str] = None) -> float:
        """Return total karma spent. Scoped to paper_id if provided."""
        if paper_id:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost), 0) AS total FROM koala_karma_ledger WHERE paper_id=?",
                (paper_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost), 0) AS total FROM koala_karma_ledger"
            ).fetchone()
        return float(row["total"]) if row else 0.0

    # ------------------------------------------------------------------
    # Verdict state
    # ------------------------------------------------------------------

    def upsert_verdict_state(
        self,
        paper_id: str,
        *,
        has_our_participation: bool = False,
        distinct_citable_other_agents: int = 0,
        eligibility_state: str = "NOT_PARTICIPATED",
        reachability_score: Optional[float] = None,
        internal_confidence: Optional[float] = None,
        submitted: bool = False,
        skip_reason: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO koala_verdict_state
               (paper_id, has_our_participation, distinct_citable_other_agents,
                eligibility_state, reachability_score, internal_confidence,
                submitted, skip_reason, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(paper_id) DO UPDATE SET
                   has_our_participation=excluded.has_our_participation,
                   distinct_citable_other_agents=excluded.distinct_citable_other_agents,
                   eligibility_state=excluded.eligibility_state,
                   reachability_score=excluded.reachability_score,
                   internal_confidence=excluded.internal_confidence,
                   submitted=excluded.submitted,
                   skip_reason=excluded.skip_reason,
                   updated_at=excluded.updated_at""",
            (
                paper_id,
                int(has_our_participation),
                distinct_citable_other_agents,
                eligibility_state,
                reachability_score,
                internal_confidence,
                int(submitted),
                skip_reason,
                _utcnow(),
            ),
        )
        self._conn.commit()

    def has_prior_participation(self, paper_id: str) -> bool:
        row = self._conn.execute(
            "SELECT has_our_participation FROM koala_verdict_state WHERE paper_id=?",
            (paper_id,),
        ).fetchone()
        return bool(row and row["has_our_participation"])

    def get_comment_stats(self, paper_id: str) -> Dict[str, int]:
        """Return comment counts for *paper_id*.

        Returns a dict with keys:
          total        — all stored comments for this paper
          ours         — comments where is_ours=1
          citable_other — citable comments where is_ours=0
        """
        row = self._conn.execute(
            """SELECT
                 COUNT(*) AS total,
                 SUM(CASE WHEN is_ours=1 THEN 1 ELSE 0 END) AS ours,
                 SUM(CASE WHEN is_ours=0 AND is_citable=1 THEN 1 ELSE 0 END) AS citable_other
               FROM koala_comments WHERE paper_id=?""",
            (paper_id,),
        ).fetchone()
        if row is None:
            return {"total": 0, "ours": 0, "citable_other": 0}
        return {
            "total": int(row["total"] or 0),
            "ours": int(row["ours"] or 0),
            "citable_other": int(row["citable_other"] or 0),
        }

    def close(self) -> None:
        self._conn.close()
