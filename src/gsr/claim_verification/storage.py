"""SQLite persistence for claim verification results – Module 4.

Adds one table to the shared ``gsr.db``:

* ``verification_results`` – one row per (claim, model) verification attempt,
  storing the LLM verdict, reasoning, confidence, and evidence chunk IDs.
"""
from __future__ import annotations

import json
import logging
import sqlite3

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_VERIFICATION_DDL = """
CREATE TABLE IF NOT EXISTS verification_results (
    id                 TEXT    PRIMARY KEY,
    claim_id           TEXT    NOT NULL,
    paper_id           TEXT    NOT NULL,
    review_id          TEXT    NOT NULL,
    verdict            TEXT,                -- 'supported' | 'refuted' |
                                            -- 'insufficient_evidence' |
                                            -- 'not_verifiable' | NULL on error
    reasoning          TEXT,
    confidence         REAL,
    supporting_quote   TEXT,
    evidence_chunk_ids TEXT    NOT NULL DEFAULT '[]',  -- JSON list of chunk IDs
    evidence_json      TEXT    NOT NULL DEFAULT '[]',  -- JSON list of structured evidence
    model_id           TEXT    NOT NULL,
    status             TEXT    NOT NULL,    -- 'success' | 'error'
    error_message      TEXT,
    verified_at        TEXT    NOT NULL,
    FOREIGN KEY (claim_id) REFERENCES claims(id),
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE INDEX IF NOT EXISTS idx_verification_claim_id
    ON verification_results(claim_id);

CREATE INDEX IF NOT EXISTS idx_verification_paper_id
    ON verification_results(paper_id);

CREATE INDEX IF NOT EXISTS idx_verification_verdict
    ON verification_results(verdict);
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_verification_db(conn: sqlite3.Connection) -> None:
    """Create / migrate the ``verification_results`` table if needed.

    Safe to call repeatedly.
    """
    with conn:
        conn.executescript(_VERIFICATION_DDL)

        # ---- Lightweight migrations for older DBs ----
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(verification_results)").fetchall()
        }

        if "evidence_json" not in cols:
            conn.execute(
                """
                ALTER TABLE verification_results
                ADD COLUMN evidence_json TEXT NOT NULL DEFAULT '[]'
                """
            )
            log.debug("[schema] Migrated verification_results: added evidence_json.")

        if "verify_meta_json" not in cols:
            conn.execute(
                """
                ALTER TABLE verification_results
                ADD COLUMN verify_meta_json TEXT
                """
            )
            log.debug("[schema] Migrated verification_results: added verify_meta_json.")

    log.debug("Verification DB table ensured.")


def save_verification_results(
    results: list[dict],
    conn: sqlite3.Connection,
) -> None:
    """Persist a batch of verification results (``INSERT OR REPLACE``).

    Each item in *results* is a dict produced by
    :func:`~gsr.claim_verification.verifier.verify_claim`:: 

        {
            "id", "claim_id", "paper_id", "review_id",
            "verdict", "reasoning", "confidence", "supporting_quote",
            "evidence", "evidence_chunk_ids",
            "model_id", "status", "error", "verified_at"
        }

    Args:
        results: List of verification result dicts.
        conn: Open SQLite connection.
    """
    if not results:
        return

    rows = [
        (
            r["id"],
            r["claim_id"],
            r["paper_id"],
            r["review_id"],
            r.get("verdict"),
            r.get("reasoning"),
            r.get("confidence"),
            r.get("supporting_quote", ""),
            json.dumps(r.get("evidence_chunk_ids", []), ensure_ascii=False),
            json.dumps(r.get("evidence", []), ensure_ascii=False),
            r["model_id"],
            r["status"],
            r.get("error"),
            r["verified_at"],
            json.dumps(r["verify_meta"], ensure_ascii=False) if r.get("verify_meta") else None,
        )
        for r in results
    ]

    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO verification_results
                (id, claim_id, paper_id, review_id, verdict, reasoning,
                 confidence, supporting_quote, evidence_chunk_ids, evidence_json,
                 model_id, status, error_message, verified_at, verify_meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    successes = sum(1 for r in results if r["status"] == "success")
    log.info(
        "Saved %d verification results (%d success, %d error).",
        len(results), successes, len(results) - successes,
    )


def load_verification_results(
    conn: sqlite3.Connection,
    *,
    paper_id: str | None = None,
    review_id: str | None = None,
    verdict: str | None = None,
) -> list[dict]:
    """Query stored verification results with optional filters."""
    conditions: list[str] = []
    params: list = []

    if paper_id:
        conditions.append("paper_id = ?")
        params.append(paper_id)
    if review_id:
        conditions.append("review_id = ?")
        params.append(review_id)
    if verdict:
        conditions.append("verdict = ?")
        params.append(verdict)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    cur = conn.execute(
        f"SELECT * FROM verification_results {where} ORDER BY verified_at",
        params,
    )
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()

    results = []
    for row in rows:
        d = dict(zip(cols, row))
        d["evidence_chunk_ids"] = json.loads(d.get("evidence_chunk_ids") or "[]")
        d["evidence"] = json.loads(d.get("evidence_json") or "[]")
        results.append(d)
    return results


def get_verdict_summary(
    conn: sqlite3.Connection,
    *,
    paper_id: str | None = None,
) -> dict[str, int]:
    """Return verdict counts, optionally filtered by paper."""
    conditions: list[str] = ["status = 'success'"]
    params: list = []

    if paper_id:
        conditions.append("paper_id = ?")
        params.append(paper_id)

    where = "WHERE " + " AND ".join(conditions)

    cur = conn.execute(
        f"""
        SELECT verdict, COUNT(*) AS cnt
        FROM verification_results
        {where}
        GROUP BY verdict
        """,
        params,
    )
    return {row[0]: row[1] for row in cur.fetchall() if row[0]}