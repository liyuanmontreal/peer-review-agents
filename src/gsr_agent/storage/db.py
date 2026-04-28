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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        for stmt in (
            "ALTER TABLE koala_agent_actions ADD COLUMN details TEXT",
            "ALTER TABLE koala_comments ADD COLUMN synced_at TEXT",
            "ALTER TABLE koala_papers ADD COLUMN abstract TEXT NOT NULL DEFAULT ''",
        ):
            try:
                self._conn.execute(stmt)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

    # ------------------------------------------------------------------
    # Papers
    # ------------------------------------------------------------------

    def upsert_paper(self, paper: Paper) -> None:
        self._conn.execute(
            """INSERT INTO koala_papers
               (paper_id, title, abstract, open_time, review_end_time, verdict_end_time,
                state, pdf_url, local_pdf_path, last_synced_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(paper_id) DO UPDATE SET
                   title=excluded.title,
                   abstract=excluded.abstract,
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
                paper.abstract,
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
                text, created_at, is_ours, is_citable, synced_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(comment_id) DO UPDATE SET
                   thread_id=excluded.thread_id,
                   parent_id=excluded.parent_id,
                   text=excluded.text,
                   is_ours=excluded.is_ours,
                   is_citable=excluded.is_citable,
                   synced_at=excluded.synced_at""",
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
                _utcnow(),
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

    # ------------------------------------------------------------------
    # Phase 5A: reactive fact-check tables
    # ------------------------------------------------------------------

    def clear_phase5a_for_comment(self, comment_id: str) -> None:
        """Delete all Phase 5A rows for comment_id to enable idempotent reruns."""
        for table in (
            "koala_extracted_claims",
            "koala_claim_verifications",
            "koala_reactive_drafts",
        ):
            self._conn.execute(f"DELETE FROM {table} WHERE comment_id=?", (comment_id,))
        self._conn.commit()

    def insert_extracted_claim(self, claim: Dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO koala_extracted_claims
               (claim_id, comment_id, paper_id, claim_text, category,
                confidence, challengeability, binary_question, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                claim["claim_id"],
                claim["comment_id"],
                claim["paper_id"],
                claim["claim_text"],
                claim.get("category"),
                claim.get("confidence"),
                claim.get("challengeability"),
                claim.get("binary_question"),
                _utcnow(),
            ),
        )
        self._conn.commit()

    def insert_claim_verification(self, verification: Dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO koala_claim_verifications
               (verification_id, claim_id, comment_id, paper_id, verdict,
                confidence, reasoning, supporting_quote, model_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                verification["verification_id"],
                verification["claim_id"],
                verification["comment_id"],
                verification["paper_id"],
                verification["verdict"],
                verification.get("confidence"),
                verification.get("reasoning"),
                verification.get("supporting_quote"),
                verification.get("model_id"),
                _utcnow(),
            ),
        )
        self._conn.commit()

    def insert_reactive_draft(self, draft: Dict[str, Any]) -> None:
        analysis = draft.get("analysis_json")
        if isinstance(analysis, dict):
            analysis = json.dumps(analysis)
        self._conn.execute(
            """INSERT OR REPLACE INTO koala_reactive_drafts
               (draft_id, comment_id, paper_id, recommendation, draft_text,
                analysis_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                draft["draft_id"],
                draft["comment_id"],
                draft["paper_id"],
                draft["recommendation"],
                draft.get("draft_text"),
                analysis,
                _utcnow(),
            ),
        )
        self._conn.commit()

    def get_citable_other_comments_for_paper(self, paper_id: str) -> list:
        """Return citable other-agent comments for paper_id as a list of dicts."""
        rows = self._conn.execute(
            """SELECT comment_id, paper_id, thread_id, parent_id,
                      author_agent_id, text, created_at, is_ours, is_citable
               FROM koala_comments
               WHERE paper_id=? AND is_ours=0 AND is_citable=1
               ORDER BY created_at""",
            (paper_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_phase5a_stats(self, paper_id: Optional[str] = None) -> Dict[str, Any]:
        """Return a summary of Phase 5A dry-run activity.

        Scoped to paper_id when provided; global otherwise.
        """
        w = "WHERE paper_id=?" if paper_id else ""
        p = (paper_id,) if paper_id else ()

        def _scalar(sql: str) -> int:
            return int(self._conn.execute(sql, p).fetchone()[0] or 0)

        conjunction = "AND" if w else "WHERE"

        comments_analyzed = _scalar(
            f"SELECT COUNT(DISTINCT comment_id) FROM koala_reactive_drafts {w}"
        )
        claims_extracted = _scalar(
            f"SELECT COUNT(*) FROM koala_extracted_claims {w}"
        )
        claims_verified = _scalar(
            f"SELECT COUNT(*) FROM koala_claim_verifications {w}"
        )
        react_count = _scalar(
            f"SELECT COUNT(*) FROM koala_reactive_drafts {w} {conjunction} recommendation='react'"
        )
        skip_count = _scalar(
            f"SELECT COUNT(*) FROM koala_reactive_drafts {w} {conjunction} recommendation='skip'"
        )
        unclear_count = _scalar(
            f"SELECT COUNT(*) FROM koala_reactive_drafts {w} {conjunction} recommendation='unclear'"
        )

        verdict_rows = self._conn.execute(
            f"SELECT verdict, COUNT(*) FROM koala_claim_verifications {w} GROUP BY verdict",
            p,
        ).fetchall()
        verdict_map = {row[0]: int(row[1] or 0) for row in verdict_rows}

        contradicted_count = sum(
            verdict_map.get(v, 0) for v in ("refuted", "contradicted", "contradiction")
        )

        return {
            "comments_analyzed": comments_analyzed,
            "claims_extracted": claims_extracted,
            "claims_verified": claims_verified,
            "react_count": react_count,
            "skip_count": skip_count,
            "unclear_count": unclear_count,
            "contradicted_count": contradicted_count,
            "supported_count": verdict_map.get("supported", 0),
            "insufficient_count": verdict_map.get("insufficient_evidence", 0),
            "verification_error_count": 0,
        }

    def get_reactive_analysis_for_comment(self, comment_id: str) -> Optional[Dict[str, Any]]:
        """Return the persisted Phase 5A analysis for comment_id, or None."""
        claims = [
            dict(row)
            for row in self._conn.execute(
                "SELECT * FROM koala_extracted_claims WHERE comment_id=?",
                (comment_id,),
            ).fetchall()
        ]
        verifications = [
            dict(row)
            for row in self._conn.execute(
                "SELECT * FROM koala_claim_verifications WHERE comment_id=?",
                (comment_id,),
            ).fetchall()
        ]
        draft_row = self._conn.execute(
            "SELECT * FROM koala_reactive_drafts WHERE comment_id=?",
            (comment_id,),
        ).fetchone()
        if draft_row is None and not claims and not verifications:
            return None
        return {
            "claims": claims,
            "verifications": verifications,
            "draft": dict(draft_row) if draft_row else None,
            "recommendation": draft_row["recommendation"] if draft_row else None,
        }

    # ------------------------------------------------------------------
    # Phase 7: run-summary queries
    # ------------------------------------------------------------------

    def get_papers(self, paper_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return tracked papers as dicts, optionally filtered by paper_ids.

        Ordered by open_time descending (most recently opened first).
        """
        if paper_ids:
            placeholders = ",".join("?" * len(paper_ids))
            rows = self._conn.execute(
                f"SELECT * FROM koala_papers WHERE paper_id IN ({placeholders})"
                " ORDER BY open_time DESC",
                paper_ids,
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM koala_papers ORDER BY open_time DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_latest_action_for_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Return the most recent koala_agent_actions row for paper_id, or None."""
        row = self._conn.execute(
            "SELECT * FROM koala_agent_actions"
            " WHERE paper_id=? ORDER BY created_at DESC LIMIT 1",
            (paper_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_distinct_other_agent_count(self, paper_id: str) -> int:
        """Return the count of distinct non-self citable agents for paper_id."""
        row = self._conn.execute(
            """SELECT COUNT(DISTINCT author_agent_id)
               FROM koala_comments
               WHERE paper_id=? AND is_ours=0 AND is_citable=1 AND author_agent_id!=''""",
            (paper_id,),
        ).fetchone()
        return int(row[0] or 0) if row else 0

    def get_strongest_contradiction_confidence(self, paper_id: str) -> Optional[float]:
        """Return the highest contradiction confidence across claim verifications, or None."""
        row = self._conn.execute(
            """SELECT MAX(confidence)
               FROM koala_claim_verifications
               WHERE paper_id=?
                 AND verdict IN ('refuted', 'contradicted', 'contradiction')""",
            (paper_id,),
        ).fetchone()
        val = row[0] if row else None
        return float(val) if val is not None else None

    # ------------------------------------------------------------------
    # Phase 8.5: dedup / idempotency queries
    # ------------------------------------------------------------------

    def has_recent_reactive_action_for_comment(
        self,
        paper_id: str,
        comment_id: str,
        now: datetime,
        *,
        within_hours: float = 12.0,
        statuses: tuple = ("dry_run", "success"),
    ) -> bool:
        """Return True if a reactive_comment action already exists for this source comment.

        Matches rows where details->source_comment_id equals comment_id, action was
        logged within within_hours of now, and status is in statuses.
        Both dry_run and success are treated as "already acted" by default.
        """
        cutoff = (now - timedelta(hours=within_hours)).astimezone(timezone.utc).isoformat()
        placeholders = ",".join("?" * len(statuses))
        row = self._conn.execute(
            f"""SELECT 1 FROM koala_agent_actions
                WHERE paper_id=?
                  AND action_type='reactive_comment'
                  AND status IN ({placeholders})
                  AND created_at >= ?
                  AND json_extract(details, '$.source_comment_id') = ?
                LIMIT 1""",
            (paper_id, *statuses, cutoff, comment_id),
        ).fetchone()
        return row is not None

    def has_recent_verdict_action_for_paper(
        self,
        paper_id: str,
        now: datetime,
        *,
        within_hours: float = 12.0,
        statuses: tuple = ("dry_run", "success"),
    ) -> bool:
        """Return True if a verdict_draft action already exists for this paper recently.

        Matches rows where action was logged within within_hours of now and status
        is in statuses. Both dry_run and success are treated as "already acted".
        """
        cutoff = (now - timedelta(hours=within_hours)).astimezone(timezone.utc).isoformat()
        placeholders = ",".join("?" * len(statuses))
        row = self._conn.execute(
            f"""SELECT 1 FROM koala_agent_actions
                WHERE paper_id=?
                  AND action_type='verdict_draft'
                  AND status IN ({placeholders})
                  AND created_at >= ?
                LIMIT 1""",
            (paper_id, *statuses, cutoff),
        ).fetchone()
        return row is not None

    def has_recent_seed_action_for_paper(
        self,
        paper_id: str,
        now: datetime,
        *,
        within_hours: float = 12.0,
        statuses: tuple = ("dry_run", "success"),
    ) -> bool:
        """Return True if a seed_comment action already exists for this paper recently."""
        cutoff = (now - timedelta(hours=within_hours)).astimezone(timezone.utc).isoformat()
        placeholders = ",".join("?" * len(statuses))
        row = self._conn.execute(
            f"""SELECT 1 FROM koala_agent_actions
                WHERE paper_id=?
                  AND action_type='seed_comment'
                  AND status IN ({placeholders})
                  AND created_at >= ?
                LIMIT 1""",
            (paper_id, *statuses, cutoff),
        ).fetchone()
        return row is not None

    def close(self) -> None:
        self._conn.close()
