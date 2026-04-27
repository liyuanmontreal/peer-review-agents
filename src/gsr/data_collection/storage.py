from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone

from gsr.config import DB_PATH, JSON_DIR, ensure_workspace_dirs

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON snapshots
# ---------------------------------------------------------------------------

def save_json(venue_data: dict) -> str:
    """Write *venue_data* to a timestamped JSON file under ``data/json/``.

    Returns the path of the written file.
    """
    ensure_workspace_dirs()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_venue = venue_data["venue_id"].replace("/", "_").replace(".", "_")
    filename = f"{safe_venue}_{ts}.json"
    path = JSON_DIR / filename

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(venue_data, fh, indent=2, ensure_ascii=False)

    log.info("JSON snapshot saved to %s", path)
    return str(path)


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

DDL = """\
CREATE TABLE IF NOT EXISTS papers (
    id          TEXT PRIMARY KEY,
    forum       TEXT,
    number      INTEGER,
    venue_id    TEXT,
    title       TEXT,
    authors     TEXT,   -- JSON array
    abstract    TEXT,
    keywords    TEXT,   -- JSON array
    fetched_at  TEXT,
    pdf_path    TEXT,
    pdf_sha256  TEXT,
    pdf_error   TEXT
);

CREATE TABLE IF NOT EXISTS reviews (
    id            TEXT PRIMARY KEY,
    paper_id      TEXT REFERENCES papers(id),
    forum         TEXT,
    replyto       TEXT,
    signatures    TEXT,   -- JSON array
    rating        TEXT,
    confidence    TEXT,
    summary       TEXT,
    strengths     TEXT,
    weaknesses    TEXT,
    questions     TEXT,
    soundness     TEXT,
    presentation  TEXT,
    contribution  TEXT,
    raw_fields    TEXT    -- JSON: all original OpenReview content fields
);

CREATE TABLE IF NOT EXISTS rebuttals (
    id          TEXT PRIMARY KEY,
    paper_id    TEXT REFERENCES papers(id),
    forum       TEXT,
    replyto     TEXT,
    signatures  TEXT,   -- JSON array
    comment     TEXT
);

CREATE TABLE IF NOT EXISTS meta_reviews (
    id              TEXT PRIMARY KEY,
    paper_id        TEXT REFERENCES papers(id),
    forum           TEXT,
    replyto         TEXT,
    signatures      TEXT,   -- JSON array
    recommendation  TEXT,
    metareview      TEXT,
    confidence      TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    id        TEXT PRIMARY KEY,
    paper_id  TEXT REFERENCES papers(id),
    forum     TEXT,
    decision  TEXT
);
"""

def _ensure_column(conn: sqlite3.Connection, table: str, col: str, decl: str) -> None:
    """Add a column to *table* if it does not already exist (additive migration)."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if col not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def init_db() -> sqlite3.Connection:
    """Open (or create) the SQLite database and run DDL."""
    # IMPORTANT: import DB_PATH at runtime so workspace overrides take effect
    from gsr.config import DB_PATH, ensure_workspace_dirs

    ensure_workspace_dirs()  # make sure workspace dir exists
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL)
    # Additive migrations for existing databases
    _ensure_column(conn, "reviews", "raw_fields", "TEXT")
    return conn

def _json_list(value) -> str | None:
    """Serialize a list to JSON text, or return None."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def save_to_db(venue_data: dict, conn: sqlite3.Connection) -> None:
    """Insert or replace all data from *venue_data* into the database."""
    now = datetime.now(timezone.utc).isoformat()
    venue_id = venue_data["venue_id"]

    with conn:
        for paper in venue_data["papers"]:
            conn.execute(
                "INSERT OR REPLACE INTO papers "
                "(id, forum, number, venue_id, title, authors, abstract, keywords, fetched_at, pdf_path, pdf_sha256, pdf_error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    paper["id"],
                    paper["forum"],
                    paper["number"],
                    venue_id,
                    paper["title"],
                    _json_list(paper["authors"]),
                    paper["abstract"],
                    _json_list(paper["keywords"]),
                    now,
                    paper["pdf_path"],
                    paper["pdf_sha256"],
                    paper["pdf_error"],
                ),
            )

            for review in paper["reviews"]:
                conn.execute(
                    "INSERT OR REPLACE INTO reviews "
                    "(id, paper_id, forum, replyto, signatures, rating, confidence, "
                    "summary, strengths, weaknesses, questions, soundness, presentation, contribution, "
                    "raw_fields) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        review["id"],
                        paper["id"],
                        review["forum"],
                        review["replyto"],
                        _json_list(review["signatures"]),
                        review["rating"],
                        review["confidence"],
                        review["summary"],
                        review["strengths"],
                        review["weaknesses"],
                        review["questions"],
                        review["soundness"],
                        review["presentation"],
                        review["contribution"],
                        review.get("raw_fields"),
                    ),
                )

            for rebuttal in paper["rebuttals"]:
                conn.execute(
                    "INSERT OR REPLACE INTO rebuttals "
                    "(id, paper_id, forum, replyto, signatures, comment) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        rebuttal["id"],
                        paper["id"],
                        rebuttal["forum"],
                        rebuttal["replyto"],
                        _json_list(rebuttal["signatures"]),
                        rebuttal["comment"],
                    ),
                )

            for meta in paper["meta_reviews"]:
                conn.execute(
                    "INSERT OR REPLACE INTO meta_reviews "
                    "(id, paper_id, forum, replyto, signatures, recommendation, metareview, confidence) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        meta["id"],
                        paper["id"],
                        meta["forum"],
                        meta["replyto"],
                        _json_list(meta["signatures"]),
                        meta["recommendation"],
                        meta["metareview"],
                        meta["confidence"],
                    ),
                )

            decision = paper.get("decision")
            if decision is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO decisions "
                    "(id, paper_id, forum, decision) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        decision["id"],
                        paper["id"],
                        decision["forum"],
                        decision["decision"],
                    ),
                )

    log.info(
        "Saved %d papers to %s",
        len(venue_data["papers"]),
        DB_PATH,
    )
