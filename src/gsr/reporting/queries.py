from __future__ import annotations

import sqlite3
from typing import Any, Iterable

from .schema import table_exists, get_table_columns
from .utils import parse_chunk_keys


def load_reviews_for_paper(conn: sqlite3.Connection, paper_id: str) -> list[dict[str, Any]]:
    if not table_exists(conn, "reviews"):
        return []

    sql = """
    SELECT
      id, paper_id, forum, replyto, signatures,
      rating, confidence,
      summary, strengths, weaknesses, questions,
      soundness, presentation, contribution
    FROM reviews
    WHERE paper_id = ?
    ORDER BY id
    """
    cur = conn.execute(sql, (paper_id,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def load_experiment_overview_for_paper(conn, paper_id: str) -> list[dict[str, Any]]:
    sql = """
    SELECT
      er.experiment_id AS experiment_id,
      MIN(er.model_id) AS model_id,
      MIN(er.fields) AS fields,
      MIN(er.min_confidence) AS min_confidence,
      MIN(er.min_challengeability) AS min_challengeability,
      COUNT(DISTINCT er.review_id) AS reviews_processed,
      COUNT(c.id) AS claims_written,
      AVG(c.confidence) AS avg_conf,
      AVG(c.challengeability) AS avg_chal,
      MAX(er.started_at) AS last_started_at
    FROM extraction_runs er
    LEFT JOIN claims c ON c.extraction_run_id = er.id
    WHERE er.paper_id = ?
      AND er.experiment_id IS NOT NULL
    GROUP BY er.experiment_id
    ORDER BY last_started_at DESC
    """
    cur = conn.execute(sql, (paper_id,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def load_claims_for_paper_grouped_by_review_experiment(
    conn,
    paper_id: str,
    experiment_id: str,
    *,
    limit: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    sql = """
    SELECT
      c.*,
      er.experiment_id AS experiment_id
    FROM claims c
    JOIN extraction_runs er ON er.id = c.extraction_run_id
    WHERE er.paper_id = ?
      AND er.experiment_id = ?
    ORDER BY c.review_id, c.source_field, c.claim_index
    """
    params = [paper_id, experiment_id]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["review_id"], []).append(r)
    return grouped


def load_paper_meta_basic(conn: sqlite3.Connection, paper_id: str) -> dict[str, Any]:
    """
    Return basic paper meta: title, forum, pdf_path, pdf_error.
    Works with your current papers schema.
    """
    if not table_exists(conn, "papers"):
        return {}

    cols = get_table_columns(conn, "papers")
    wanted = ["title", "forum", "pdf_path", "pdf_error"]
    picked = [c for c in wanted if c in cols]
    if not picked:
        return {}

    sql = f"""
        SELECT {", ".join(picked)}
        FROM papers
        WHERE id = ?
        LIMIT 1
    """
    row = conn.execute(sql, (paper_id,)).fetchone()
    if not row:
        return {}

    return dict(zip([c for c in picked], row))


def load_review_count(conn: sqlite3.Connection, paper_id: str) -> int:
    if not table_exists(conn, "reviews"):
        return 0

    row = conn.execute(
        "SELECT COUNT(*) FROM reviews WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    return int(row[0]) if row else 0

def load_paper_meta(conn: sqlite3.Connection, paper_id: str) -> dict[str, Any]:
    """
    Returns: {"title": ..., "pdf_path": ..., "pdf_sha256": ...}
    """
    meta: dict[str, Any] = {"title": None, "pdf_path": None, "pdf_sha256": None}
    if not table_exists(conn, "papers"):
        return meta

    cols = get_table_columns(conn, "papers")
    if "id" not in cols:
        return meta

    select_cols = [c for c in ("title", "pdf_path", "pdf_sha256") if c in cols]
    if not select_cols:
        return meta

    row = conn.execute(
        f"SELECT {', '.join(select_cols)} FROM papers WHERE id=? LIMIT 1",
        (paper_id,),
    ).fetchone()
    if not row:
        return meta

    for i, c in enumerate(select_cols):
        meta[c] = row[i]
    return meta


def load_verification_rows(
    conn,
    paper_id: str,
    *,
    experiment_id: str | None = None,
    include_errors: bool = False,
    only_verdict: str | None = None,
    min_conf: float | None = None,
    limit: int | None = None,
):
    """
    Joins verification_results with claims + extraction_runs (+ reviews).
    Returns a list of row dicts.
    If experiment_id is provided, filters by er.experiment_id.
    """
    where = ["vr.paper_id = ?"]
    params: list[Any] = [paper_id]

    if not include_errors:
        where.append("vr.status = 'success'")

    if only_verdict:
        where.append("vr.verdict = ?")
        params.append(only_verdict)

    if min_conf is not None:
        where.append("vr.confidence >= ?")
        params.append(min_conf)

    if experiment_id is not None:
        where.append("er.experiment_id = ?")
        params.append(experiment_id)

    sql = f"""
    SELECT
      vr.id as verification_id,
      vr.claim_id,
      vr.paper_id,
      vr.review_id as verification_review_id,
      vr.verdict,
      vr.reasoning,
      vr.confidence as verification_confidence,
      vr.supporting_quote,
      vr.evidence_chunk_ids,
      vr.model_id as verification_model_id,
      vr.status,
      vr.error_message,
      vr.verified_at,

      c.review_id as claim_review_id,
      c.source_field,
      c.claim_index,
      c.claim_text,
      c.verbatim_quote,
      c.claim_type,
      c.confidence as extraction_confidence,
      c.model_id as extraction_model_id,
      c.extracted_at,

      er.id as extraction_run_id,
      er.experiment_id as extraction_experiment_id,
      er.config_hash as extraction_config_hash,
      er.fields as extraction_fields,
      er.min_confidence as extraction_min_confidence,
      er.min_challengeability as extraction_min_challengeability,
      er.started_at as extraction_started_at,
      er.finished_at as extraction_finished_at,

      -- ✅ Review original text fields (for report v2)
      r.signatures as review_signatures,
      r.rating as review_rating,
      r.confidence as review_confidence,
      r.summary as review_summary,
      r.strengths as review_strengths,
      r.weaknesses as review_weaknesses,
      r.questions as review_questions,
      r.soundness as review_soundness,
      r.presentation as review_presentation,
      r.contribution as review_contribution

    FROM verification_results vr
    LEFT JOIN claims c ON c.id = vr.claim_id
    LEFT JOIN extraction_runs er ON er.id = c.extraction_run_id
    LEFT JOIN reviews r ON r.id = c.review_id
    WHERE {" AND ".join(where)}
    ORDER BY
      CASE vr.verdict
        WHEN 'insufficient_evidence' THEN 0
        WHEN 'not_verifiable' THEN 1
        WHEN 'contradicted' THEN 2
        WHEN 'supported' THEN 3
        ELSE 9
      END,
      vr.confidence DESC
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]

def collect_all_evidence_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for r in rows:
        keys.extend(parse_chunk_keys(r.get("evidence_chunk_ids")))
    # dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def detect_evidence_mode(conn: sqlite3.Connection, paper_id: str, keys: list[str]) -> str:
    """
    Determine whether evidence keys refer to paper_chunks.id or paper_chunks.chunk_index.
    Always STRICTLY constrained by paper_id (avoid cross-paper leakage).
    Returns: 'id' | 'chunk_index' | 'unknown'
    """
    if not keys or not table_exists(conn, "paper_chunks"):
        return "unknown"

    cols = get_table_columns(conn, "paper_chunks")
    sample = keys[:50]
    placeholders = ",".join("?" for _ in sample)

    if "id" in cols and "paper_id" in cols:
        row = conn.execute(
            f"SELECT 1 FROM paper_chunks WHERE paper_id=? AND id IN ({placeholders}) LIMIT 1",
            [paper_id, *sample],
        ).fetchone()
        if row:
            return "id"

    if "chunk_index" in cols and "paper_id" in cols:
        row = conn.execute(
            f"SELECT 1 FROM paper_chunks WHERE paper_id=? AND chunk_index IN ({placeholders}) LIMIT 1",
            [paper_id, *sample],
        ).fetchone()
        if row:
            return "chunk_index"

    return "unknown"


def fetch_chunks_strict(
    conn: sqlite3.Connection,
    paper_id: str,
    keys: Iterable[str],
    mode: str,
) -> dict[str, dict[str, Any]]:
    """
    Fetch paper_chunks STRICTLY by paper_id.
    Returns mapping:
      - mode=='id'         => key is chunk id
      - mode=='chunk_index'=> key is chunk_index as str
    """
    keys = [str(k) for k in keys if str(k).strip()]
    # dedupe
    seen: set[str] = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]

    if not keys:
        return {}

    placeholders = ",".join("?" for _ in keys)

    if mode == "id":
        sql = f"""
        SELECT id, paper_id, section, page, chunk_index, text
        FROM paper_chunks
        WHERE paper_id=? AND id IN ({placeholders})
        """
        cur = conn.execute(sql, [paper_id, *keys])
        cols = [d[0] for d in cur.description]
        out: dict[str, dict[str, Any]] = {}
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            out[str(d["id"])] = d
        return out

    if mode == "chunk_index":
        sql = f"""
        SELECT id, paper_id, section, page, chunk_index, text
        FROM paper_chunks
        WHERE paper_id=? AND chunk_index IN ({placeholders})
        """
        cur = conn.execute(sql, [paper_id, *keys])
        cols = [d[0] for d in cur.description]
        out: dict[str, dict[str, Any]] = {}
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            out[str(d["chunk_index"])] = d
        return out

    return {}


def load_extraction_variants(conn: sqlite3.Connection, paper_id: str) -> list[dict[str, Any]]:
    """
    Experiment-level variants (one row per experiment_id).
    """
    sql = """
    SELECT
        er.experiment_id AS experiment_id,      
        MIN(er.config_hash) AS config_hash,
        MIN(er.fields) AS fields,
        MIN(er.min_confidence) AS min_confidence,
        MIN(er.min_challengeability) AS min_challengeability,
        COUNT(c.id) AS claim_count,
        AVG(c.confidence) AS avg_conf,
        AVG(c.challengeability) AS avg_chal
    FROM extraction_runs er
    LEFT JOIN claims c ON c.extraction_run_id = er.id
    WHERE er.paper_id = ?
      AND er.experiment_id IS NOT NULL
    GROUP BY er.experiment_id
    ORDER BY MAX(er.started_at) DESC
    """
    cur = conn.execute(sql, (paper_id,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]

def load_claims_for_experiment(
    conn: sqlite3.Connection,
    paper_id: str,
    experiment_id: str,
) -> list[dict[str, Any]]:
    """
    Load all claims produced under a given experiment_id for one paper.
    """
    sql = """
    SELECT
      c.*
    FROM claims c
    JOIN extraction_runs er ON er.id = c.extraction_run_id
    WHERE er.paper_id = ?
      AND er.experiment_id = ?
    ORDER BY c.review_id, c.source_field, c.claim_index
    """
    cur = conn.execute(sql, (paper_id, experiment_id))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def load_extraction_rows(
    conn: sqlite3.Connection,
    paper_id: str,
    *,
    limit: int | None = None,
):
    sql = """
    SELECT
      c.id as claim_id,
      c.review_id,
      c.source_field,
      c.claim_index,
      c.claim_text,
      c.confidence,
      c.challengeability,    
      c.claim_type,
      c.extracted_at,

      er.id as extraction_run_id,
      er.config_hash,
      er.fields,
      er.min_confidence,
      er.min_challengeability,
      er.started_at,
      er.finished_at

    FROM claims c
    LEFT JOIN extraction_runs er ON er.id = c.extraction_run_id
    WHERE c.paper_id = ?
    ORDER BY c.review_id, er.id, c.source_field, c.claim_index
    """

    params = [paper_id]

    if limit:
        sql += " LIMIT ?"
        params.append(limit)

    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def load_extraction_rows_rich(
    conn: sqlite3.Connection,
    paper_id: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Rich extraction rows: claims + extraction_runs + reviews raw text/meta + papers meta (if exists).
    - reviews columns are based on your schema:
      summary/strengths/weaknesses/questions + rating/confidence/signatures + rubric fields.
    """
    has_reviews = table_exists(conn, "reviews")
    has_papers = table_exists(conn, "papers")

    review_cols = get_table_columns(conn, "reviews") if has_reviews else set()
    paper_cols = get_table_columns(conn, "papers") if has_papers else set()

    # review raw text fields
    wanted_review_text_cols = [
        "summary", "strengths", "weaknesses", "questions",
        "soundness", "presentation", "contribution",
    ]
    picked_review_text_cols = [c for c in wanted_review_text_cols if c in review_cols]

    # review metadata fields
    wanted_review_meta_cols = ["rating", "confidence", "signatures", "replyto", "forum"]
    picked_review_meta_cols = [c for c in wanted_review_meta_cols if c in review_cols]

    select_review_sql = ""
    join_review_sql = ""
    if has_reviews and (picked_review_text_cols or picked_review_meta_cols):
        cols = picked_review_text_cols + picked_review_meta_cols
        select_review_sql = ",\n      " + ",\n      ".join([f"r.{c} AS review_{c}" for c in cols])
        join_review_sql = "\n    LEFT JOIN reviews r ON r.id = c.review_id"

    # papers metadata (best-effort; depends on your papers schema)
    wanted_paper_cols = ["title", "forum", "pdf_path", "pdf_error"]
    picked_paper_cols = [c for c in wanted_paper_cols if c in paper_cols]

    select_paper_sql = ""
    join_paper_sql = ""
    if has_papers and picked_paper_cols:
        select_paper_sql = ",\n      " + ",\n      ".join([f"p.{c} AS paper_{c}" for c in picked_paper_cols])
        join_paper_sql = "\n    LEFT JOIN papers p ON p.id = c.paper_id"


    select_paper_sql = ""
    join_paper_sql = ""
    if has_papers and picked_paper_cols:
        select_paper_sql = ",\n      " + ",\n      ".join([f"p.{c} AS paper_{c}" for c in picked_paper_cols])
        join_paper_sql = "\n    LEFT JOIN papers p ON p.id = c.paper_id"

    sql = f"""
    SELECT
      c.id AS claim_id,
      c.review_id,
      c.paper_id,
      c.source_field,
      c.claim_index,
      c.claim_text,
      c.verbatim_quote,
      c.claim_type,
      c.category, 
      c.confidence AS claim_confidence,
      c.challengeability AS claim_challengeability,
      c.binary_question,
      c.why_challengeable,
      c.model_id AS claim_model_id,
      c.extracted_at,
      c.extraction_run_id,

      er.id AS run_id,
      er.model_id AS run_model_id,
      er.config_hash,
      er.fields AS run_fields,
      er.min_confidence AS run_min_confidence,
      er.min_challengeability AS run_min_challengeability,
      er.started_at AS run_started_at,
      er.finished_at AS run_finished_at
      {select_review_sql}
      {select_paper_sql}

    FROM claims c
    LEFT JOIN extraction_runs er ON er.id = c.extraction_run_id
    {join_review_sql}
    {join_paper_sql}
    WHERE c.paper_id = ?
    ORDER BY c.review_id, er.id, c.source_field, c.claim_index
    """

    params: list[Any] = [paper_id]
    if limit is not None:
        sql += "\nLIMIT ?"
        params.append(limit)

    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]