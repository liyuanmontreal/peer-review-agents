from __future__ import annotations



import sqlite3
from pathlib import Path
from typing import Any
from gsr.config import REPORT_DIR
from .schema import require_tables
from .utils import safe_filename
from .queries import (
    load_paper_meta,
    load_verification_rows,
    collect_all_evidence_keys,
    detect_evidence_mode,
    fetch_chunks_strict,
)
from .render_md import render_verification_markdown
from .render_md import render_extraction_comparison
from .render_md import render_retrieval_markdown

from .queries import load_extraction_variants, load_claims_for_experiment,table_exists

from .queries import load_extraction_rows_rich
from .render_md import render_extraction_markdown,render_extraction_markdown_review_first

from .queries import (
    load_paper_meta_basic,
    load_review_count,
    load_reviews_for_paper,
    load_experiment_overview_for_paper,
    load_claims_for_paper_grouped_by_review_experiment,
)



from collections import defaultdict

def load_claims_for_paper_grouped_by_review(
    conn: sqlite3.Connection,
    paper_id: str,
    *,
    limit: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    rows = load_extraction_rows_rich(conn, paper_id, limit=limit)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["review_id"]].append(r)
    return dict(grouped)


def export_extraction_comparison_report(
    *,
    db_path: str | Path,
    paper_id: str,
    out_path: str | Path | None = None,
) -> str:
    db_path = Path(db_path).resolve()

    if out_path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORT_DIR / f"{safe_filename(paper_id)}_extraction_compare.md"
    else:
        out_path = Path(out_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        require_tables(conn, ["extraction_runs", "claims"])

        variants = load_extraction_variants(conn, paper_id)
        if len(variants) < 2:
            raise ValueError("Need at least two extraction runs to compare.")

        experiment_claims = {
            v["experiment_id"]: load_claims_for_experiment(conn, paper_id, v["experiment_id"])
            for v in variants
        }
        md = render_extraction_comparison(paper_id, variants, experiment_claims)

        out_path.write_text(md, encoding="utf-8")
        return str(out_path)

    finally:
        conn.close()


from .queries import load_extraction_rows
from .render_md import render_extraction_markdown



def export_extraction_report(
    *,
    db_path: str | Path,
    paper_id: str,
    out_path: str | Path | None = None,
    limit: int | None = None,
) -> str:
    

    from .queries import (
        load_paper_meta_basic,
        load_review_count,
        load_reviews_for_paper,
        load_experiment_overview_for_paper,
        load_claims_for_paper_grouped_by_review_experiment,
    )

    db_path = Path(db_path).resolve()

    if out_path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORT_DIR / f"{safe_filename(paper_id)}_extraction.md"
    else:
        out_path = Path(out_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        require_tables(conn, ["claims", "extraction_runs", "papers"])

        rows = load_extraction_rows_rich(conn, paper_id, limit=limit)

        paper_meta = load_paper_meta_basic(conn, paper_id)
        review_count = load_review_count(conn, paper_id)

        # 1) overview of all experiments
        exp_overview = load_experiment_overview_for_paper(conn, paper_id)

        # choose latest experiment by default
        selected_experiment_id = exp_overview[0]["experiment_id"] if exp_overview else None

        reviews = load_reviews_for_paper(conn, paper_id)

        # 2) claims filtered by selected experiment (IMPORTANT)
        claims_by_review = (
            load_claims_for_paper_grouped_by_review_experiment(conn, paper_id, selected_experiment_id, limit=limit)
            if selected_experiment_id
            else {}
        )

        md = render_extraction_markdown_review_first(
            paper_id=paper_id,
            paper_meta=paper_meta,
            review_count=review_count,
            reviews=reviews,
            claims_by_review=claims_by_review,          
            experiment_overview=exp_overview,
            selected_experiment_id=selected_experiment_id,
)


        out_path.write_text(md, encoding="utf-8")
        return str(out_path)

    finally:
        conn.close()

def export_verification_report(
    *,
    db_path: str | Path,
    paper_id: str,
    out_path: str | Path | None = None,
    limit: int | None = None,
    include_errors: bool = False,
    only_verdict: str | None = None,
    min_conf: float | None = None,
    experiment_id: str | None = None,   # NEW
    evidence_max_chars: int = 1400,
    assume_page_0_indexed: bool = False,
) -> str:
    """
    Export verification results for a paper into a Markdown file.

    Default behavior (experiment_id is None):
      - Select the latest extraction experiment for this paper (by last_started_at)
      - Filter verification rows to that experiment (via extraction_runs.experiment_id)

    Pure read-only export: does NOT run any pipeline steps.

    Returns:
        Output path as string.
    """
    db_path = Path(db_path).resolve()

    # Default report path under workspace
    if out_path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORT_DIR / f"{safe_filename(paper_id)}_verification.md"
    else:
        out_path = Path(out_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Strictly require tables needed for export
        require_tables(
            conn,
            ["verification_results", "claims", "extraction_runs", "paper_chunks", "papers", "experiments"],
        )

        paper_meta = load_paper_meta(conn, paper_id)

        # ✅ Default: choose latest experiment if not specified
        resolved_experiment_id = experiment_id
        if resolved_experiment_id is None:
            overview = load_experiment_overview_for_paper(conn, paper_id)
            resolved_experiment_id = overview[0]["experiment_id"] if overview else None

        rows = load_verification_rows(
            conn,
            paper_id,
            experiment_id=resolved_experiment_id,   # NEW (requires queries.py support)
            include_errors=include_errors,
            only_verdict=only_verdict,
            min_conf=min_conf,
            limit=limit,
        )

        all_keys = collect_all_evidence_keys(rows)
        evidence_mode = detect_evidence_mode(conn, paper_id, all_keys)
        chunk_map = fetch_chunks_strict(conn, paper_id, all_keys, evidence_mode)

        md = render_verification_markdown(
            paper_id=paper_id,
            paper_meta=paper_meta,
            rows=rows,
            chunk_map=chunk_map,
            evidence_mode=evidence_mode,
            db_path_str=db_path.as_posix(),
            evidence_max_chars=evidence_max_chars,
            assume_page_0_indexed=assume_page_0_indexed,
        )

        out_path.write_text(md, encoding="utf-8")
        return str(out_path)

    finally:
        conn.close()





def export_retrieval_report(
    *,
    db_path: str | Path,
    paper_id: str,
    out_path: str | Path | None = None,
    preview_limit: int = 3,
    retrieval_top_k: int = 5,
) -> str:
    """
    Export a retrieval-focused report for one paper.

    The report summarizes:
      - parsing result (sections)
      - chunking result
      - indexing / embedding status
      - retrieval preview (using cached retrieval_results if available)

    Returns:
        Output path as string.
    """
    db_path = Path(db_path).resolve()

    if out_path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORT_DIR / f"{safe_filename(paper_id)}_retrieval.md"
    else:
        out_path = Path(out_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        require_tables(conn, ["papers", "paper_chunks", "chunk_embeddings"])

        paper_meta = _load_paper_meta_basic(conn, paper_id)
        parsing = _load_parsing_summary(conn, paper_id)
        chunking = _load_chunking_summary(conn, paper_id)
        indexing = _load_indexing_summary(conn, paper_id)
        retrieval_preview = _load_retrieval_preview(
            conn,
            paper_id,
            preview_limit=preview_limit,
            retrieval_top_k=retrieval_top_k,
        )

        md = render_retrieval_markdown(
            paper_id=paper_id,
            paper_meta=paper_meta,
            parsing=parsing,
            chunking=chunking,
            indexing=indexing,
            retrieval_preview=retrieval_preview,
        )

        out_path.write_text(md, encoding="utf-8")
        return str(out_path)

    finally:
        conn.close()


def _load_paper_meta_basic(conn: sqlite3.Connection, paper_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT id, title, forum, pdf_path, pdf_error
        FROM papers
        WHERE id = ?
        """,
        (paper_id,),
    ).fetchone()
    return dict(row) if row else {"id": paper_id}


def _load_parsing_summary(conn: sqlite3.Connection, paper_id: str) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT section, MIN(page) AS page, MIN(chunk_index) AS first_chunk_index
        FROM paper_chunks
        WHERE paper_id = ?
        GROUP BY section
        ORDER BY first_chunk_index
        """,
        (paper_id,),
    ).fetchall()

    section_rows = []
    for r in rows:
        text_row = conn.execute(
            """
            SELECT text
            FROM paper_chunks
            WHERE paper_id = ? AND section = ?
            ORDER BY chunk_index
            LIMIT 1
            """,
            (paper_id, r["section"]),
        ).fetchone()

        section_rows.append({
            "heading": r["section"],
            "page": r["page"],
            "text": (text_row["text"] if text_row else ""),
        })

    num_pages_row = conn.execute(
        """
        SELECT MAX(page) AS max_page
        FROM paper_chunks
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()

    return {
        "num_pages": int(num_pages_row["max_page"] or 0) if num_pages_row else 0,
        "num_sections": len(section_rows),
        "sections": section_rows,
    }


def _load_chunking_summary(conn: sqlite3.Connection, paper_id: str) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT id, chunk_index, section, page, text, char_start, char_end, chunk_size
        FROM paper_chunks
        WHERE paper_id = ?
        ORDER BY chunk_index
        """,
        (paper_id,),
    ).fetchall()

    section_chunk_counts: dict[str, int] = {}
    for r in rows:
        sec = r["section"] or "unknown"
        section_chunk_counts[sec] = section_chunk_counts.get(sec, 0) + 1

    example_chunks = [dict(r) for r in rows[:3]]

    chunk_size = rows[0]["chunk_size"] if rows else None

    # overlap is not stored in DB; infer best-effort from first adjacent pair
    chunk_overlap = None
    if len(rows) >= 2:
        a = rows[0]
        b = rows[1]
        try:
            overlap = int(a["char_end"]) - int(b["char_start"])
            if overlap >= 0:
                chunk_overlap = overlap
        except Exception:
            chunk_overlap = None

    return {
        "num_chunks": len(rows),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "section_chunk_counts": section_chunk_counts,
        "example_chunks": example_chunks,
    }


def _load_indexing_summary(conn: sqlite3.Connection, paper_id: str) -> dict[str, Any]:
    indexed_row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM paper_chunks
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()

    model_row = conn.execute(
        """
        SELECT ce.model_id, COUNT(*) AS n
        FROM chunk_embeddings ce
        JOIN paper_chunks pc ON ce.chunk_id = pc.id
        WHERE pc.paper_id = ?
        GROUP BY ce.model_id
        ORDER BY MAX(ce.embedded_at) DESC
        LIMIT 1
        """,
        (paper_id,),
    ).fetchone()

    embedded_chunks = 0
    embedding_model = None
    if model_row:
        embedded_chunks = int(model_row["n"] or 0)
        embedding_model = model_row["model_id"]

    example_rows = conn.execute(
        """
        SELECT
            pc.id,
            pc.section,
            pc.page,
            pc.text,
            CASE WHEN ce.chunk_id IS NOT NULL THEN 1 ELSE 0 END AS embedded
        FROM paper_chunks pc
        LEFT JOIN chunk_embeddings ce
          ON ce.chunk_id = pc.id
        WHERE pc.paper_id = ?
        ORDER BY pc.chunk_index
        LIMIT 5
        """,
        (paper_id,),
    ).fetchall()

    return {
        "indexed_chunks": int(indexed_row["n"] or 0) if indexed_row else 0,
        "embedded_chunks": embedded_chunks,
        "embedding_model": embedding_model,
        "indexed_examples": [dict(r) for r in example_rows],
    }


def _load_retrieval_preview(
    conn: sqlite3.Connection,
    paper_id: str,
    *,
    preview_limit: int = 3,
    retrieval_top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Load a small retrieval preview using cached retrieval_results if available.

    Strategy:
      - find claims for this paper that already have retrieval_results
      - take a few distinct claims
      - load top-k ranked chunks for each
    """
    claim_rows = conn.execute(
        """
        SELECT DISTINCT
            c.id AS claim_id,
            c.claim_text
        FROM claims c
        JOIN retrieval_results rr ON rr.claim_id = c.id
        WHERE c.paper_id = ?
        ORDER BY c.id
        LIMIT ?
        """,
        (paper_id, preview_limit),
    ).fetchall()

    previews: list[dict[str, Any]] = []

    for crow in claim_rows:
        claim_id = crow["claim_id"]
        results = conn.execute(
            """
            SELECT
                rr.rank,
                rr.bm25_score,
                rr.semantic_score,
                rr.combined_score,
                pc.id AS chunk_id,
                pc.section,
                pc.page,
                pc.text
            FROM retrieval_results rr
            JOIN paper_chunks pc ON rr.chunk_id = pc.id
            WHERE rr.claim_id = ?
            ORDER BY rr.rank
            LIMIT ?
            """,
            (claim_id, retrieval_top_k),
        ).fetchall()

        previews.append({
            "claim_id": claim_id,
            "claim_text": crow["claim_text"],
            "results": [dict(r) for r in results],
        })

    return previews

def export_retrieval_report(
    *,
    db_path: str | Path,
    paper_id: str,
    out_path: str | Path | None = None,
    preview_limit: int = 3,
    retrieval_top_k: int = 5,
) -> str:
    """
    Export retrieval results for a paper into a Markdown file.

    Pure read-only export: does NOT run parsing/chunking/retrieval.
    It summarizes what is already stored in DB.
    """
    db_path = Path(db_path).resolve()

    if out_path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORT_DIR / f"{safe_filename(paper_id)}_retrieval.md"
    else:
        out_path = Path(out_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        require_tables(conn, ["papers", "paper_chunks", "chunk_embeddings"])

        paper_meta = _load_retrieval_paper_meta(conn, paper_id)
        parsing = _load_retrieval_parsing_summary(conn, paper_id)
        chunking = _load_retrieval_chunking_summary(conn, paper_id)
        indexing = _load_retrieval_indexing_summary(conn, paper_id)
        retrieval_preview = _load_retrieval_preview(
            conn,
            paper_id,
            preview_limit=preview_limit,
            retrieval_top_k=retrieval_top_k,
        )

        md = render_retrieval_markdown(
            paper_id=paper_id,
            paper_meta=paper_meta,
            parsing=parsing,
            chunking=chunking,
            indexing=indexing,
            retrieval_preview=retrieval_preview,
        )

        out_path.write_text(md, encoding="utf-8")
        return str(out_path)

    finally:
        conn.close()

def _load_retrieval_paper_meta(
    conn: sqlite3.Connection,
    paper_id: str,
) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT id, title, forum, pdf_path, pdf_error
        FROM papers
        WHERE id = ?
        """,
        (paper_id,),
    ).fetchone()
    return dict(row) if row else {"id": paper_id}


def _load_retrieval_parsing_summary(
    conn: sqlite3.Connection,
    paper_id: str,
) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT section, MIN(page) AS page, MIN(chunk_index) AS first_chunk_index
        FROM paper_chunks
        WHERE paper_id = ?
        GROUP BY section
        ORDER BY first_chunk_index
        """,
        (paper_id,),
    ).fetchall()

    sections = []
    for r in rows:
        text_row = conn.execute(
            """
            SELECT text
            FROM paper_chunks
            WHERE paper_id = ? AND section = ?
            ORDER BY chunk_index
            LIMIT 1
            """,
            (paper_id, r["section"]),
        ).fetchone()

        sections.append({
            "heading": r["section"],
            "page": r["page"],
            "text": (text_row["text"] if text_row else ""),
        })

    num_pages_row = conn.execute(
        """
        SELECT MAX(page) AS max_page
        FROM paper_chunks
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()

    return {
        "num_pages": int(num_pages_row["max_page"] or 0) if num_pages_row else 0,
        "num_sections": len(sections),
        "sections": sections,
    }


def _load_retrieval_chunking_summary(
    conn: sqlite3.Connection,
    paper_id: str,
) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT id, chunk_index, section, page, text, char_start, char_end, chunk_size
        FROM paper_chunks
        WHERE paper_id = ?
        ORDER BY chunk_index
        """,
        (paper_id,),
    ).fetchall()

    section_chunk_counts: dict[str, int] = {}
    for r in rows:
        sec = r["section"] or "unknown"
        section_chunk_counts[sec] = section_chunk_counts.get(sec, 0) + 1

    example_chunks = [dict(r) for r in rows[:3]]

    chunk_size = rows[0]["chunk_size"] if rows else None

    # best-effort overlap inference
    chunk_overlap = None
    if len(rows) >= 2:
        a = rows[0]
        b = rows[1]
        try:
            overlap = int(a["char_end"]) - int(b["char_start"])
            if overlap >= 0:
                chunk_overlap = overlap
        except Exception:
            chunk_overlap = None

    return {
        "num_chunks": len(rows),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "section_chunk_counts": section_chunk_counts,
        "example_chunks": example_chunks,
    }


def _load_retrieval_indexing_summary(
    conn: sqlite3.Connection,
    paper_id: str,
) -> dict[str, Any]:
    indexed_row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM paper_chunks
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()

    model_row = conn.execute(
        """
        SELECT ce.model_id, COUNT(*) AS n
        FROM chunk_embeddings ce
        JOIN paper_chunks pc ON ce.chunk_id = pc.id
        WHERE pc.paper_id = ?
        GROUP BY ce.model_id
        ORDER BY MAX(ce.embedded_at) DESC
        LIMIT 1
        """,
        (paper_id,),
    ).fetchone()

    embedded_chunks = 0
    embedding_model = None
    if model_row:
        embedded_chunks = int(model_row["n"] or 0)
        embedding_model = model_row["model_id"]

    example_rows = conn.execute(
        """
        SELECT
            pc.id,
            pc.section,
            pc.page,
            pc.text,
            CASE WHEN ce.chunk_id IS NOT NULL THEN 1 ELSE 0 END AS embedded
        FROM paper_chunks pc
        LEFT JOIN chunk_embeddings ce
          ON ce.chunk_id = pc.id
        WHERE pc.paper_id = ?
        ORDER BY pc.chunk_index
        LIMIT 5
        """,
        (paper_id,),
    ).fetchall()

    return {
        "indexed_chunks": int(indexed_row["n"] or 0) if indexed_row else 0,
        "embedded_chunks": embedded_chunks,
        "embedding_model": embedding_model,
        "indexed_examples": [dict(r) for r in example_rows],
    }


def _load_retrieval_preview(
    conn: sqlite3.Connection,
    paper_id: str,
    *,
    preview_limit: int = 3,
    retrieval_top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Load a small retrieval preview using cached retrieval_results.
    """
    if not table_exists(conn, "claims") or not table_exists(conn, "retrieval_results"):
        return []

    claim_rows = conn.execute(
        """
        SELECT DISTINCT
            c.id AS claim_id,
            c.claim_text
        FROM claims c
        JOIN retrieval_results rr ON rr.claim_id = c.id
        WHERE c.paper_id = ?
        ORDER BY c.id
        LIMIT ?
        """,
        (paper_id, preview_limit),
    ).fetchall()

    previews: list[dict[str, Any]] = []

    for crow in claim_rows:
        claim_id = crow["claim_id"]
        results = conn.execute(
            """
            SELECT
                rr.rank,
                rr.bm25_score,
                rr.semantic_score,
                rr.combined_score,
                pc.id AS chunk_id,
                pc.section,
                pc.page,
                pc.text
            FROM retrieval_results rr
            JOIN paper_chunks pc ON rr.chunk_id = pc.id
            WHERE rr.claim_id = ?
            ORDER BY rr.rank
            LIMIT ?
            """,
            (claim_id, retrieval_top_k),
        ).fetchall()

        previews.append({
            "claim_id": claim_id,
            "claim_text": crow["claim_text"],
            "results": [dict(r) for r in results],
        })

    return previews