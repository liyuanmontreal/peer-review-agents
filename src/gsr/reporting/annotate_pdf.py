from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

log = logging.getLogger(__name__)


def _safe_filename(text: str) -> str:
    text = re.sub(r'[<>:"/\\\\|?*]+', "_", text)
    text = text.strip().replace(" ", "_")
    return text[:120] if text else "unknown"


def _load_latest_verification_for_claim(
    conn: sqlite3.Connection,
    claim_id: str,
) -> dict[str, Any] | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT *
        FROM verification_results
        WHERE claim_id = ? AND status = 'success'
        ORDER BY verified_at DESC
        LIMIT 1
        """,
        (claim_id,),
    ).fetchone()
    conn.row_factory = None

    if not row:
        return None

    d = dict(row)
    try:
        d["evidence"] = json.loads(d.get("evidence_json") or "[]")
    except Exception:
        d["evidence"] = []
    return d


def _load_pdf_path_for_paper(
    conn: sqlite3.Connection,
    paper_id: str,
) -> str | None:
    row = conn.execute(
        """
        SELECT pdf_path
        FROM papers
        WHERE id = ?
        LIMIT 1
        """,
        (paper_id,),
    ).fetchone()
    return row[0] if row and row[0] else None


def _load_spans_by_ids(
    conn: sqlite3.Connection,
    span_ids: list[str],
) -> list[dict[str, Any]]:
    """Load bbox rows for span IDs from pdf_spans(page_num, bbox_json)."""
    if not span_ids:
        return []

    placeholders = ",".join("?" for _ in span_ids)

    query = f"""
        SELECT
            id,
            paper_id,
            page_num,
            bbox_json
        FROM pdf_spans
        WHERE id IN ({placeholders})
    """

    conn.row_factory = sqlite3.Row
    rows = conn.execute(query, span_ids).fetchall()
    conn.row_factory = None

    out = []
    for r in rows:
        d = dict(r)
        try:
            bbox = json.loads(d["bbox_json"])
        except Exception:
            bbox = None

        if not bbox or len(bbox) != 4:
            log.warning("Invalid bbox_json for span %s: %r", d["id"], d.get("bbox_json"))
            continue

        out.append(
            {
                "id": d["id"],
                "paper_id": d["paper_id"],
                "page": d["page_num"],
                "x0": float(bbox[0]),
                "y0": float(bbox[1]),
                "x1": float(bbox[2]),
                "y1": float(bbox[3]),
            }
        )

    return out

def _collect_unique_span_ids(evidence: list[dict[str, Any]]) -> list[str]:
    seen = set()
    out = []

    for ev in evidence:
        sids = ev.get("aligned_span_ids")
        if not sids:
            sids = ev.get("span_ids", []) or []

        for sid in sids:
            if sid and sid not in seen:
                seen.add(sid)
                out.append(sid)

    return out


def annotate_pdf_by_claim(
    conn: sqlite3.Connection,
    *,
    claim_id: str,
    output_dir: str | Path | None = None,
    color: tuple[float, float, float] = (1, 0, 0),
    border_width: float = 1.5,
) -> Path:
    """Annotate the paper PDF for a claim using verification evidence spans."""
    vr = _load_latest_verification_for_claim(conn, claim_id)
    if not vr:
        raise ValueError(f"No successful verification result found for claim: {claim_id}")

    paper_id = vr["paper_id"]
    evidence = vr.get("evidence", [])

    if not evidence:
        raise ValueError(f"No structured evidence found in verification result for claim: {claim_id}")

    span_ids = _collect_unique_span_ids(evidence)
    if not span_ids:
        raise ValueError(f"No span_ids found in evidence_json for claim: {claim_id}")

    spans = _load_spans_by_ids(conn, span_ids)
    if not spans:
        raise ValueError(f"No pdf_spans rows found for span_ids of claim: {claim_id}")

    pdf_path = _load_pdf_path_for_paper(conn, paper_id)
    if not pdf_path:
        raise ValueError(f"No pdf_path found in papers table for paper: {paper_id}")

    src_pdf = Path(pdf_path)
    if not src_pdf.exists():
        raise ValueError(f"PDF file does not exist on disk: {src_pdf}")

    if output_dir is None:
        workspace_guess = src_pdf.parent.parent if src_pdf.parent.parent.exists() else src_pdf.parent
        output_dir = workspace_guess / "annotated" / _safe_filename(paper_id)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{_safe_filename(claim_id)}_annotated.pdf"

    doc = fitz.open(src_pdf)
    try:
        for sp in spans:
            page_num = int(sp["page"])
            if page_num < 1 or page_num > len(doc):
                log.warning("Skip span %s: invalid page=%s", sp["id"], page_num)
                continue

            page = doc[page_num - 1]
            rect = fitz.Rect(
                sp["x0"],
                sp["y0"],
                sp["x1"],
                sp["y1"],
            )

            shape = page.new_shape()
            shape.draw_rect(rect)
            shape.finish(color=color, width=border_width)
            shape.commit()

        doc.save(out_path)
    finally:
        doc.close()

    log.info("Annotated PDF saved: %s", out_path)
    return out_path