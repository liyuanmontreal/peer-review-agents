"""Persistence layer for extracted claims (Module 2)."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base DDL (NO indexes here, to avoid failing on older schemas)
# ---------------------------------------------------------------------------

BASE_DDL = """\
CREATE TABLE IF NOT EXISTS claims (
    id              TEXT PRIMARY KEY,
    review_id       TEXT NOT NULL,
    paper_id        TEXT,              -- may be missing in older DBs; migrated additively
    source_field    TEXT NOT NULL,
    source_field_raw TEXT,             -- original OpenReview field key (null = same as source_field)
    claim_index     INTEGER NOT NULL,
    claim_text      TEXT NOT NULL,
    verbatim_quote  TEXT,
    claim_type      TEXT,
    confidence      REAL,
    category        TEXT,
    challengeability REAL,
    binary_question TEXT,
    why_challengeable TEXT,
    model_id        TEXT,
    extracted_at    TEXT,
    extraction_run_id INTEGER,         -- lineage to extraction_runs.id
    calibrated_score REAL,
    calibrated_score_norm REAL,
    calibrated_score_raw_text TEXT,
    calibration_error TEXT
);

CREATE TABLE IF NOT EXISTS extraction_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id            TEXT NOT NULL,
    paper_id             TEXT,          -- migrated additively for old DBs
    model_id             TEXT,
    status               TEXT NOT NULL, -- 'success' | 'error'
    error_message        TEXT,
    claims_count         INTEGER DEFAULT 0,
    min_challengeability REAL,          -- optional run metadata
    min_confidence       REAL,          -- optional run metadata
    started_at           TEXT,
    finished_at          TEXT,
    config_hash          TEXT,          -- identify parameter variant
    fields               TEXT           -- e.g., "summary,strengths,weaknesses"
);
"""


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, decl: str) -> None:
    """Add column if missing (SQLite additive migration)."""
    cols = _table_columns(conn, table)
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

def _ensure_indexes(conn: sqlite3.Connection) -> None:
    """Create helpful indexes, only after columns exist."""
    # claims indexes
    cols = _table_columns(conn, "claims")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_review_id ON claims(review_id)")
    if "paper_id" in cols:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_paper_id ON claims(paper_id)")

    # extraction_runs indexes
    rcols = _table_columns(conn, "extraction_runs")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_review_id ON extraction_runs(review_id)")
    if "paper_id" in rcols:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_paper_id ON extraction_runs(paper_id)")
   
    if "extraction_run_id" in cols:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_run_id ON claims(extraction_run_id)")

    rcols = _table_columns(conn, "extraction_runs")
    if "config_hash" in rcols:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_config_hash ON extraction_runs(config_hash)")


def ensure_experiment_schema(conn: sqlite3.Connection) -> None:
    """
    Minimal schema for experiment-level runs:
      - experiments table
      - extraction_runs.experiment_id column
    Safe to call repeatedly.
    """
    # 1) experiments table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id          TEXT PRIMARY KEY,
            paper_id    TEXT,
            created_at  TEXT NOT NULL,
            command     TEXT,
            params_json TEXT
        );
        """
    )

    # 2) add column extraction_runs.experiment_id if missing
    cols = [r[1] for r in conn.execute("PRAGMA table_info(extraction_runs)").fetchall()]
    if "experiment_id" not in cols:
        conn.execute("ALTER TABLE extraction_runs ADD COLUMN experiment_id TEXT;")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_paper ON experiments(paper_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_extruns_expid ON extraction_runs(experiment_id);")
    conn.commit()


def init_claims_db(conn: sqlite3.Connection) -> None:
    """Create tables + run additive migrations (idempotent)."""
    # 1) create base tables (safe even if they exist)
    conn.executescript(BASE_DDL)

    # 2) additive migrations for claims (old DBs may miss these)
    _ensure_column(conn, "claims", "paper_id", "TEXT")
    _ensure_column(conn, "claims", "source_field_raw", "TEXT")
    _ensure_column(conn, "claims", "category", "TEXT")
    _ensure_column(conn, "claims", "challengeability", "REAL")
    _ensure_column(conn, "claims", "binary_question", "TEXT")
    _ensure_column(conn, "claims", "why_challengeable", "TEXT")
    _ensure_column(conn, "claims", "extraction_run_id", "INTEGER")
    _ensure_column(conn, "claims", "calibrated_score", "REAL")
    _ensure_column(conn, "claims", "calibrated_score_norm", "REAL")
    _ensure_column(conn, "claims", "calibrated_score_raw_text", "TEXT")
    _ensure_column(conn, "claims", "calibration_error", "TEXT")
    _ensure_column(conn, "claims", "calibration_error_detail", "TEXT")
  

    # 3) additive migrations for extraction_runs
    _ensure_column(conn, "extraction_runs", "paper_id", "TEXT")
    _ensure_column(conn, "extraction_runs", "min_challengeability", "REAL")
    _ensure_column(conn, "extraction_runs", "min_confidence", "REAL")    
    _ensure_column(conn, "extraction_runs", "config_hash", "TEXT")
    _ensure_column(conn, "extraction_runs", "fields", "TEXT")

    # 4) now it's safe to create indexes
    _ensure_indexes(conn)

    log.debug("Claims tables initialised / migrated.")

import json
from uuid import uuid4

def create_experiment(
    conn: sqlite3.Connection,
    *,
    paper_id: str | None,
    command: str | None,
    params: dict | None = None,
) -> str:
    experiment_id = uuid4().hex
    conn.execute(
        """
        INSERT INTO experiments (id, paper_id, created_at, command, params_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            experiment_id,
            paper_id,
            datetime.now().isoformat(timespec="seconds"),
            command,
            json.dumps(params or {}, ensure_ascii=False),
        ),
    )
    conn.commit()
    return experiment_id

def save_extraction_results(
    results: list[dict],
    conn: sqlite3.Connection,
    *,
    min_challengeability: float | None = None,
    min_confidence: float | None = None,
) -> None:
    """Persist extraction results (claims + run metadata)."""
    now = datetime.now(timezone.utc).isoformat()

    with conn:
        for result in results:
            # Prefer per-result metadata; fall back to function args; then None
            mch = result.get("min_challengeability", min_challengeability)
            mco = result.get("min_confidence", min_confidence)

            cur = conn.execute(
                """
                INSERT INTO extraction_runs
                (review_id, paper_id, model_id, status, error_message, claims_count,
                min_challengeability, min_confidence, started_at, finished_at,
                config_hash, fields, experiment_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result["review_id"],
                    result.get("paper_id"),
                    result.get("model_id"),
                    result.get("status"),
                    result.get("error"),
                    len(result.get("claims", [])),
                    mch,
                    mco,
                    result.get("started_at", now),
                    result.get("finished_at", now),
                    result.get("config_hash"),
                    result.get("fields"),
                    result.get("experiment_id")
  
                ),
            )
            run_id = cur.lastrowid

            # write claims INSIDE the result loop
            for claim in result.get("claims", []):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO claims
                    (id, review_id, paper_id, source_field, source_field_raw, claim_index,
                     claim_text, verbatim_quote, claim_type, confidence,
                     category, challengeability, binary_question, why_challengeable,
                     model_id, extracted_at, extraction_run_id,calibrated_score,
                    calibrated_score_norm,calibrated_score_raw_text,calibration_error,calibration_error_detail)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claim["id"],
                        result["review_id"],
                        result.get("paper_id"),
                        claim["source_field"],
                        claim.get("source_field_raw"),
                        claim["claim_index"],
                        claim["claim_text"],
                        claim.get("verbatim_quote"),
                        "factual",
                        claim.get("confidence"),
                        claim.get("category"),
                        claim.get("challengeability"),
                        claim.get("binary_question"),
                        claim.get("why_challengeable"),
                        result.get("model_id"),
                        result.get("finished_at", now),
                        run_id,
                        claim.get("calibrated_score"),
                        claim.get("calibrated_score_norm"),
                        claim.get("calibrated_score_raw_text"),
                        claim.get("calibration_error"),
                        claim.get("calibration_error_detail")
                    ),
                )

    log.info(
        "Saved %d extraction results (%d total claims).",
        len(results),
        sum(len(r.get("claims", [])) for r in results),
    )