from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_CHUNKS_DDL = """
CREATE TABLE IF NOT EXISTS paper_chunks (
    id            TEXT    PRIMARY KEY,
    paper_id      TEXT    NOT NULL,
    chunk_index   INTEGER NOT NULL,
    section       TEXT    NOT NULL,
    page          INTEGER NOT NULL,
    page_start    INTEGER,
    page_end      INTEGER,
    text          TEXT    NOT NULL,
    char_start    INTEGER NOT NULL,
    char_end      INTEGER NOT NULL,
    span_ids_json TEXT,
    chunk_size    INTEGER NOT NULL,
    chunked_at    TEXT    NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);
CREATE INDEX IF NOT EXISTS idx_paper_chunks_paper_id ON paper_chunks(paper_id);
"""

_PDF_SPANS_DDL = """
CREATE TABLE IF NOT EXISTS pdf_spans (
    id          TEXT    PRIMARY KEY,
    paper_id    TEXT    NOT NULL,
    page_num    INTEGER NOT NULL,
    span_index  INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    bbox_json   TEXT    NOT NULL,
    block_index INTEGER,
    line_index  INTEGER,
    section     TEXT,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);
CREATE INDEX IF NOT EXISTS idx_pdf_spans_paper_id ON pdf_spans(paper_id);
CREATE INDEX IF NOT EXISTS idx_pdf_spans_paper_page ON pdf_spans(paper_id, page_num);
"""

_EVIDENCE_OBJECTS_DDL = """
CREATE TABLE IF NOT EXISTS evidence_objects (
    id              TEXT PRIMARY KEY,
    paper_id        TEXT NOT NULL,
    object_type     TEXT NOT NULL,
    label           TEXT,
    page            INTEGER,
    page_start      INTEGER,
    page_end        INTEGER,
    section         TEXT,
    section_number  TEXT,
    caption_text    TEXT,
    retrieval_text  TEXT NOT NULL,
    content_text    TEXT,
    bbox_json       TEXT,
    span_ids_json   TEXT,
    asset_path      TEXT,
    metadata_json   TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);
CREATE INDEX IF NOT EXISTS idx_evidence_objects_paper_id ON evidence_objects(paper_id);
CREATE INDEX IF NOT EXISTS idx_evidence_objects_type ON evidence_objects(paper_id, object_type);
CREATE INDEX IF NOT EXISTS idx_evidence_objects_label ON evidence_objects(paper_id, label);
"""

_EMBEDDINGS_DDL = """
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id    TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    embedding   TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (chunk_id, model_id),
    FOREIGN KEY (chunk_id) REFERENCES paper_chunks(id)
);
CREATE TABLE IF NOT EXISTS evidence_embeddings (
    evidence_object_id TEXT NOT NULL,
    model_id           TEXT NOT NULL,
    embedding          TEXT NOT NULL,
    embedded_at        TEXT NOT NULL,
    PRIMARY KEY (evidence_object_id, model_id),
    FOREIGN KEY (evidence_object_id) REFERENCES evidence_objects(id)
);
"""

_RETRIEVAL_RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS retrieval_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id           TEXT    NOT NULL,
    chunk_id           TEXT,
    evidence_object_id TEXT,
    object_type        TEXT,
    label              TEXT,
    rank               INTEGER NOT NULL,
    bm25_score         REAL    NOT NULL,
    semantic_score     REAL    NOT NULL,
    reference_boost    REAL    NOT NULL DEFAULT 0.0,
    combined_score     REAL    NOT NULL,
    model_id           TEXT    NOT NULL,
    retrieved_at       TEXT    NOT NULL,
    FOREIGN KEY (claim_id) REFERENCES claims(id),
    FOREIGN KEY (chunk_id) REFERENCES paper_chunks(id),
    FOREIGN KEY (evidence_object_id) REFERENCES evidence_objects(id)
);
CREATE INDEX IF NOT EXISTS idx_retrieval_claim_id ON retrieval_results(claim_id);
"""


def init_retrieval_db(conn: sqlite3.Connection) -> None:
    with conn:
        for ddl in (_CHUNKS_DDL, _PDF_SPANS_DDL, _EVIDENCE_OBJECTS_DDL, _EMBEDDINGS_DDL, _RETRIEVAL_RESULTS_DDL):
            conn.executescript(ddl)
        for col, typ in [("page_start", "INTEGER"), ("page_end", "INTEGER"), ("span_ids_json", "TEXT")]:
            _ensure_column(conn, "paper_chunks", col, typ)
        for col, typ in [("section", "TEXT")]:
            _ensure_column(conn, "pdf_spans", col, typ)
        for col, typ in [
            ("evidence_object_id", "TEXT"),
            ("object_type", "TEXT"),
            ("label", "TEXT"),
            ("reference_boost", "REAL DEFAULT 0.0"),
        ]:
            _ensure_column(conn, "retrieval_results", col, typ)


def save_chunks(chunks: list[dict], conn: sqlite3.Connection, *, chunk_size: int = 512) -> None:
    if not chunks:
        return
    now = _now()
    rows = []
    for c in chunks:
        span_ids_json = json.dumps(c.get("span_ids"), ensure_ascii=False) if c.get("span_ids") is not None else None
        rows.append((
            c["id"], c["paper_id"], c["chunk_index"], c.get("section") or "unknown", c.get("page") or c.get("page_start") or 1,
            c.get("page_start"), c.get("page_end"), c["text"], c.get("char_start", 0), c.get("char_end", len(c["text"])),
            span_ids_json, chunk_size, now,
        ))
    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO paper_chunks
            (id, paper_id, chunk_index, section, page, page_start, page_end,
             text, char_start, char_end, span_ids_json, chunk_size, chunked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def save_pdf_spans(spans: list[dict], conn: sqlite3.Connection) -> None:
    if not spans:
        return
    rows = [(
        s["id"], s["paper_id"], s["page_num"], s["span_index"], s["text"], json.dumps(s["bbox"], ensure_ascii=False),
        s.get("block_index"), s.get("line_index"), s.get("section"),
    ) for s in spans]
    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO pdf_spans
            (id, paper_id, page_num, span_index, text, bbox_json, block_index, line_index, section)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def save_evidence_objects(objects: list, conn: sqlite3.Connection) -> None:
    if not objects:
        return
    now = _now()
    rows = []
    for obj in objects:
        row = obj.to_row() if hasattr(obj, "to_row") else dict(obj)
        rows.append((
            row["id"], row["paper_id"], row["object_type"], row.get("label"), row.get("page"), row.get("page_start"), row.get("page_end"),
            row.get("section"), row.get("section_number"), row.get("caption_text"), row["retrieval_text"], row.get("content_text"),
            row.get("bbox_json"), row.get("span_ids_json"), row.get("asset_path"), row.get("metadata_json"), now,
        ))
    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO evidence_objects
            (id, paper_id, object_type, label, page, page_start, page_end, section, section_number,
             caption_text, retrieval_text, content_text, bbox_json, span_ids_json, asset_path,
             metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def load_evidence_objects_for_paper(paper_id: str, conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        """
        SELECT id, paper_id, object_type, label, page, page_start, page_end, section, section_number,
               caption_text, retrieval_text, content_text, bbox_json, span_ids_json, asset_path, metadata_json
        FROM evidence_objects
        WHERE paper_id = ?
        ORDER BY CASE object_type WHEN 'table' THEN 0 WHEN 'figure' THEN 1 ELSE 2 END, COALESCE(page, page_start, 99999), id
        """,
        (paper_id,),
    )
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r[0], "paper_id": r[1], "object_type": r[2], "label": r[3], "page": r[4], "page_start": r[5], "page_end": r[6],
            "section": r[7], "section_number": r[8], "caption_text": r[9], "retrieval_text": r[10], "content_text": r[11],
            "bbox": json.loads(r[12]) if r[12] else None,
            "span_ids": json.loads(r[13]) if r[13] else None,
            "asset_path": r[14],
            "metadata": json.loads(r[15]) if r[15] else {},
        })
    return out


def save_embeddings(chunks: list[dict], embeddings: list[list[float]], model_id: str, conn: sqlite3.Connection) -> None:
    if not chunks or not embeddings:
        return
    now = _now()
    rows = [(c["id"], model_id, json.dumps(emb), now) for c, emb in zip(chunks, embeddings)]
    with conn:
        conn.executemany("INSERT OR REPLACE INTO chunk_embeddings (chunk_id, model_id, embedding, embedded_at) VALUES (?, ?, ?, ?)", rows)


def save_evidence_embeddings(objects: list[dict], embeddings: list[list[float]], model_id: str, conn: sqlite3.Connection) -> None:
    if not objects or not embeddings:
        return
    now = _now()
    rows = [(o["id"], model_id, json.dumps(emb), now) for o, emb in zip(objects, embeddings)]
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO evidence_embeddings (evidence_object_id, model_id, embedding, embedded_at) VALUES (?, ?, ?, ?)",
            rows,
        )


def save_retrieval_results(claim_id: str, results: list[dict], model_id: str, conn: sqlite3.Connection) -> None:
    if not results:
        return

    now = _now()
    rows = []

    for rank, r in enumerate(results, start=1):
        obj_type = r.get("object_type", "text_chunk")
        chunk_id = r.get("chunk_id")
        evidence_object_id = r.get("evidence_object_id")

        # text-only fallback: keep chunk_id, null out evidence_object_id
        if obj_type == "text_chunk":
            if evidence_object_id == chunk_id:
                evidence_object_id = None

        rows.append((
            claim_id,
            chunk_id,
            evidence_object_id,
            obj_type,
            r.get("label"),
            rank,
            r.get("bm25_score", 0.0),
            r.get("semantic_score", 0.0),
            r.get("reference_boost", 0.0),
            r.get("combined_score", 0.0),
            model_id,
            now,
        ))

    with conn:
        conn.executemany(
            """
            INSERT INTO retrieval_results
            (claim_id, chunk_id, evidence_object_id, object_type, label, rank,
             bm25_score, semantic_score, reference_boost, combined_score, model_id, retrieved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def load_pdf_spans_for_paper(paper_id: str, conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        "SELECT id, paper_id, page_num, span_index, text, bbox_json, block_index, line_index, section FROM pdf_spans WHERE paper_id = ? ORDER BY page_num, span_index",
        (paper_id,),
    )
    rows = cur.fetchall()
    return [{"id": r[0], "paper_id": r[1], "page_num": r[2], "span_index": r[3], "text": r[4], "bbox": json.loads(r[5]), "block_index": r[6], "line_index": r[7], "section": r[8]} for r in rows]


def get_indexed_paper_ids(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("SELECT DISTINCT paper_id FROM paper_chunks")
    return {r[0] for r in cur.fetchall()}


def get_embedded_paper_ids(conn: sqlite3.Connection, model_id: str) -> set[str]:
    cur = conn.execute("SELECT DISTINCT paper_id FROM evidence_objects eo JOIN evidence_embeddings ee ON eo.id = ee.evidence_object_id WHERE ee.model_id = ?", (model_id,))
    out = {r[0] for r in cur.fetchall()}
    if out:
        return out
    cur = conn.execute("SELECT DISTINCT pc.paper_id FROM chunk_embeddings ce JOIN paper_chunks pc ON ce.chunk_id = pc.id WHERE ce.model_id = ?", (model_id,))
    return {r[0] for r in cur.fetchall()}


def update_table_evidence_docling_enrichment(
    conn: sqlite3.Connection,
    evidence_object_id: str,
    markdown_text: str,
    metadata_updates: dict | None = None,
) -> None:
    """Write Docling late-enrichment result into evidence_objects.content_text.

    Reads existing metadata_json, merges enrichment fields, then updates both
    content_text and metadata_json in a single write.
    """
    row = conn.execute(
        "SELECT metadata_json FROM evidence_objects WHERE id = ?",
        (evidence_object_id,),
    ).fetchone()
    if not row:
        return

    existing_meta: dict = {}
    if row[0]:
        try:
            existing_meta = json.loads(row[0])
        except Exception:
            pass

    merged = {
        **existing_meta,
        "docling_enriched": True,
        "docling_markdown_chars": len(markdown_text or ""),
        "docling_error": None,
        "docling_extracted_at": _now(),
        **(metadata_updates or {}),
    }

    with conn:
        conn.execute(
            "UPDATE evidence_objects SET content_text = ?, metadata_json = ? WHERE id = ?",
            (markdown_text, json.dumps(merged, ensure_ascii=False), evidence_object_id),
        )


def mark_table_evidence_docling_failed(
    conn: sqlite3.Connection,
    evidence_object_id: str,
    error: str,
) -> None:
    """Record a Docling late-enrichment failure in evidence_objects metadata."""
    row = conn.execute(
        "SELECT metadata_json FROM evidence_objects WHERE id = ?",
        (evidence_object_id,),
    ).fetchone()
    if not row:
        return

    existing_meta: dict = {}
    if row[0]:
        try:
            existing_meta = json.loads(row[0])
        except Exception:
            pass

    merged = {
        **existing_meta,
        "docling_enriched": False,
        "docling_error": error,
    }

    with conn:
        conn.execute(
            "UPDATE evidence_objects SET metadata_json = ? WHERE id = ?",
            (json.dumps(merged, ensure_ascii=False), evidence_object_id),
        )


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_type_sql: str) -> None:
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cur.fetchall()}
    if column_name in existing:
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type_sql}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_chunks_for_paper(paper_id: str, conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        """
        SELECT id, paper_id, chunk_index, section, page, page_start, page_end,
               text, char_start, char_end, span_ids_json
        FROM paper_chunks
        WHERE paper_id = ?
        ORDER BY chunk_index
        """,
        (paper_id,),
    )
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "paper_id": r[1],
            "chunk_index": r[2],
            "section": r[3],
            "page": r[4],
            "page_start": r[5],
            "page_end": r[6],
            "text": r[7],
            "char_start": r[8],
            "char_end": r[9],
            "span_ids": json.loads(r[10]) if r[10] else None,
        })
    return out


def delete_evidence_objects_for_paper(
    paper_id: str,
    conn: sqlite3.Connection,
    *,
    model_id: str | None = None,
) -> None:
    """
    Delete evidence_objects for one paper and their evidence_embeddings.
    If model_id is provided, delete only embeddings for that model first, then objects.
    """
    cur = conn.execute(
        "SELECT id FROM evidence_objects WHERE paper_id = ?",
        (paper_id,),
    )
    evidence_ids = [r[0] for r in cur.fetchall()]
    if not evidence_ids:
        return

    placeholders = ",".join("?" for _ in evidence_ids)

    with conn:
        # retrieval_results.evidence_object_id and evidence_embeddings.evidence_object_id
        # both reference evidence_objects.id with FK constraints.  Delete dependent rows
        # first so the subsequent evidence_objects delete does not violate those constraints.
        conn.execute(
            f"DELETE FROM retrieval_results WHERE evidence_object_id IN ({placeholders})",
            evidence_ids,
        )

        if model_id:
            conn.execute(
                f"""
                DELETE FROM evidence_embeddings
                WHERE model_id = ?
                  AND evidence_object_id IN ({placeholders})
                """,
                [model_id, *evidence_ids],
            )
        else:
            conn.execute(
                f"""
                DELETE FROM evidence_embeddings
                WHERE evidence_object_id IN ({placeholders})
                """,
                evidence_ids,
            )

        conn.execute(
            "DELETE FROM evidence_objects WHERE paper_id = ?",
            (paper_id,),
        )