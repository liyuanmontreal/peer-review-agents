from __future__ import annotations

import json
import re
import sqlite3
from typing import Any


_REVIEW_FIELDS = [
    "summary",
    "strengths",
    "weaknesses",
    "questions",
    "soundness",
    "presentation",
    "contribution",
]

_FIELD_ORDER = {
    "summary": 0,
    "strengths": 1,
    "weaknesses": 2,
    "questions": 3,
    "soundness": 4,
    "presentation": 5,
    "contribution": 6,
}

# ---------------------------------------------------------------------------
# Field normalization helpers (heterogeneous schema support)
# ---------------------------------------------------------------------------

def _build_review_fields_payload(review: dict[str, Any]) -> dict[str, Any]:
    """Build the full field payload for a review row.

    Returns a dict with:
      ``fields``            – legacy dict (backward-compat, only _REVIEW_FIELDS)
      ``normalized_fields`` – canonical-key dict, extended from raw_fields
      ``raw_fields``        – original OpenReview field dict (may be empty)
      ``field_labels``      – display labels keyed by canonical name
      ``field_order``       – ordered list of canonical names present in this review
    """
    from gsr.claim_extraction.field_policy import (
        canonicalize_field_name,
        NORMALIZED_FIELD_LABELS,
        NORMALIZED_FIELD_ORDER,
        TIER_C,
    )

    # 1) Legacy fields (backward compat)
    legacy = {f: review.get(f) or "" for f in _REVIEW_FIELDS}

    # 2) Raw fields from JSON blob
    raw_fields: dict[str, str] = {}
    raw_fields_json = review.get("raw_fields")
    if raw_fields_json:
        try:
            parsed = json.loads(raw_fields_json)
            if isinstance(parsed, dict):
                raw_fields = {k: str(v) for k, v in parsed.items() if v and str(v).strip()}
        except Exception:
            pass

    # 3) Normalized fields: start from legacy, then add new canonical fields from raw_fields
    normalized: dict[str, str] = {k: v for k, v in legacy.items() if v}
    for raw_key, raw_text in raw_fields.items():
        canonical = canonicalize_field_name(raw_key)
        if not canonical or canonical in TIER_C:
            continue
        # Add if absent or if the canonical slot is currently empty
        if not normalized.get(canonical):
            normalized[canonical] = raw_text

    # 4) Field labels: use policy table, fall back to title-cased canonical
    field_labels = {
        canonical: NORMALIZED_FIELD_LABELS.get(
            canonical, canonical.replace("_", " ").title()
        )
        for canonical in normalized
    }

    # 5) Field order: policy order first, then any extras
    ordered = [f for f in NORMALIZED_FIELD_ORDER if f in normalized]
    extras = [f for f in normalized if f not in NORMALIZED_FIELD_ORDER]
    field_order = ordered + sorted(extras)

    return {
        "fields": legacy,               # legacy (backward compat)
        "normalized_fields": normalized,
        "raw_fields": raw_fields,
        "field_labels": field_labels,
        "field_order": field_order,
    }

_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!。！？])\s+|\n+")

# Header/footer filtering
_HEADER_FOOTER_PAGE_FRACTION = 0.08
_SHORT_MARGIN_TEXT_MAX_LEN = 80


def _json_load_maybe(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _tokenize_for_sentence_match(text: str) -> set[str]:
    return {
        tok
        for tok in re.split(r"[^a-zA-Z0-9]+", (text or "").lower())
        if len(tok) >= 3
    }


def _best_sentence_like_report(
    raw_text: str,
    claim_text: str,
) -> tuple[str, float, int | None, int | None]:
    raw = raw_text or ""
    claim = (claim_text or "").strip()
    if not raw.strip() or not claim:
        return "", 0.0, None, None

    claim_toks = _tokenize_for_sentence_match(claim)
    if not claim_toks:
        return "", 0.0, None, None

    best_sent = ""
    best_score = 0.0
    best_start: int | None = None
    best_end: int | None = None

    spans: list[tuple[int, int]] = []
    last = 0

    for m in _SENT_SPLIT_RE.finditer(raw):
        start = last
        end = m.start()
        if end > start:
            spans.append((start, end))
        last = m.end()

    if last < len(raw):
        spans.append((last, len(raw)))

    for start, end in spans:
        raw_slice = raw[start:end]
        sent = raw_slice.strip()
        if not sent:
            continue

        sent_toks = _tokenize_for_sentence_match(sent)
        if not sent_toks:
            continue

        inter = len(claim_toks & sent_toks)
        score = inter / (len(claim_toks) ** 0.5)

        if score > best_score:
            best_score = score
            best_sent = sent

            left_trim = len(raw_slice) - len(raw_slice.lstrip())
            right_trimmed_len = len(raw_slice.rstrip())

            best_start = start + left_trim
            best_end = start + right_trimmed_len

    return best_sent, best_score, best_start, best_end


def _claim_sort_key(claim: dict[str, Any]) -> tuple[int, int, int]:
    source_field = claim.get("source_field")
    field_rank = _FIELD_ORDER.get(source_field, 999)

    display_start = claim.get("display_start")
    if not isinstance(display_start, int):
        display_start = 10**9

    claim_index = claim.get("claim_index")
    if not isinstance(claim_index, int):
        claim_index = 10**9

    return (field_rank, display_start, claim_index)


def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        cur = conn.execute(f"PRAGMA table_info({table_name})")
        return {row[1] for row in cur.fetchall()}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Basic loaders
# ---------------------------------------------------------------------------

def _load_review(conn: sqlite3.Connection, review_id: str) -> dict[str, Any] | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT *
        FROM reviews
        WHERE id = ?
        LIMIT 1
        """,
        (review_id,),
    ).fetchone()
    conn.row_factory = None
    return dict(row) if row else None


def _load_reviews_for_paper(conn: sqlite3.Connection, paper_id: str) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT *
        FROM reviews
        WHERE paper_id = ?
        ORDER BY id
        """,
        (paper_id,),
    ).fetchall()
    conn.row_factory = None
    return [dict(r) for r in rows]


def _load_paper(conn: sqlite3.Connection, paper_id: str) -> dict[str, Any] | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT id, title, authors, abstract, keywords, pdf_path
        FROM papers
        WHERE id = ?
        LIMIT 1
        """,
        (paper_id,),
    ).fetchone()
    conn.row_factory = None

    if not row:
        return None

    d = dict(row)
    d["authors"] = _json_load_maybe(d.get("authors"), [])
    d["keywords"] = _json_load_maybe(d.get("keywords"), [])
    return d


def _load_claim_basic(conn: sqlite3.Connection, claim_id: str) -> dict[str, Any] | None:
    claim_cols = _get_table_columns(conn, "claims")
    select_cols = [
        "id",
        "review_id",
        "paper_id",
        "source_field",
        "claim_index",
        "claim_text",
        "verbatim_quote",
        "claim_type",
        "confidence",
        "category",
        "challengeability",
    ]
    if "calibrated_score" in claim_cols:
        select_cols.append("calibrated_score")

    conn.row_factory = sqlite3.Row
    row = conn.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM claims
        WHERE id = ?
        LIMIT 1
        """,
        (claim_id,),
    ).fetchone()
    conn.row_factory = None
    d = dict(row) if row else None
    if d is not None and "calibrated_score" not in d:
        d["calibrated_score"] = None
    return d


def _load_claims_for_review(conn: sqlite3.Connection, review_id: str) -> list[dict[str, Any]]:
    claim_cols = _get_table_columns(conn, "claims")
    select_cols = [
        "id",
        "review_id",
        "paper_id",
        "source_field",
        "claim_index",
        "claim_text",
        "verbatim_quote",
        "claim_type",
        "confidence",
        "category",
        "challengeability",
    ]
    if "source_field_raw" in claim_cols:
        select_cols.append("source_field_raw")
    if "calibrated_score" in claim_cols:
        select_cols.append("calibrated_score")

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM claims
        WHERE review_id = ?
        ORDER BY source_field, claim_index
        """,
        (review_id,),
    ).fetchall()
    conn.row_factory = None

    out = [dict(r) for r in rows]
    if "calibrated_score" not in select_cols:
        for d in out:
            d["calibrated_score"] = None
    return out


def _load_claims_for_paper(conn: sqlite3.Connection, paper_id: str) -> list[dict[str, Any]]:
    claim_cols = _get_table_columns(conn, "claims")
    select_cols = [
        "id",
        "review_id",
        "paper_id",
        "source_field",
        "claim_index",
        "claim_text",
        "verbatim_quote",
        "claim_type",
        "confidence",
        "category",
        "challengeability",
    ]
    if "source_field_raw" in claim_cols:
        select_cols.append("source_field_raw")
    if "calibrated_score" in claim_cols:
        select_cols.append("calibrated_score")

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM claims
        WHERE paper_id = ?
        ORDER BY review_id, source_field, claim_index
        """,
        (paper_id,),
    ).fetchall()
    conn.row_factory = None

    out = [dict(r) for r in rows]
    if "calibrated_score" not in select_cols:
        for d in out:
            d["calibrated_score"] = None
    return out


# ---------------------------------------------------------------------------
# Evidence normalization / loaders
# ---------------------------------------------------------------------------

def _normalize_evidence_item(ev: dict[str, Any] | None) -> dict[str, Any]:
    ev = dict(ev or {})

    object_type = ev.get("object_type") or ev.get("evidence_type") or "text_chunk"
    evidence_object_id = ev.get("evidence_object_id") or ev.get("chunk_id")

    label = (
        ev.get("label")
        or ev.get("evidence_label")
        or ev.get("object_label")
        or ev.get("caption_label")
        or (f"E{ev.get('rank')}" if ev.get("rank") is not None else None)
    )

    if object_type in {"figure", "table"}:
        # Keep caption for figure/table evidence cards; suppress raw retrieval text.
        caption = ev.get("caption") or ev.get("caption_text")
        text = None
    else:
        caption = None
        text = ev.get("text") or ev.get("content_text") or ev.get("caption_text")

    span_ids = ev.get("span_ids")
    if isinstance(span_ids, str):
        span_ids = _json_load_maybe(span_ids, [])
    if span_ids is None:
        span_ids = []

    aligned_span_ids = ev.get("aligned_span_ids")
    if isinstance(aligned_span_ids, str):
        aligned_span_ids = _json_load_maybe(aligned_span_ids, [])
    if aligned_span_ids is None:
        aligned_span_ids = []

    bbox = ev.get("bbox") or ev.get("object_bbox")
    if isinstance(bbox, str):
        bbox = _json_load_maybe(bbox, None)

    page = ev.get("page")
    page_num = ev.get("page_num")
    if page is None and page_num is not None:
        page = page_num
    if page_num is None and page is not None:
        page_num = page

    reference_boost = ev.get("reference_boost", 0.0) or 0.0
    reference_matched = bool(
        ev.get("reference_matched")
        or (isinstance(reference_boost, (int, float)) and reference_boost > 0)
    )

    # figure_ocr_* debug fields — only meaningful for figure evidence objects.
    # Included in the payload so callers can inspect OCR status without a
    # separate DB query (useful for evaluation and future UI debug panels).
    figure_ocr_debug = {
        "figure_ocr_attempted": ev.get("figure_ocr_attempted", False),
        "figure_ocr_skip_reason": ev.get("figure_ocr_skip_reason"),
        "figure_ocr_quality": ev.get("figure_ocr_quality"),
        # Truncated preview for API payloads; full text is in evidence_objects.metadata_json
        "figure_ocr_text_preview": (ev.get("figure_ocr_text") or "")[:200] or None,
        "object_bbox_confidence": ev.get("object_bbox_confidence"),
    } if object_type == "figure" else {}

    return {
        **ev,
        "evidence_object_id": evidence_object_id,
        "object_type": object_type,
        "evidence_type": object_type,
        "label": label,
        "evidence_label": ev.get("evidence_label") or label,
        "caption": caption,
        "caption_text": caption,
        "text": text,
        "page": page,
        "page_num": page_num,
        "bbox": bbox,
        "object_bbox": bbox,
        "span_ids": span_ids,
        "aligned_span_ids": aligned_span_ids,
        "reference_boost": reference_boost,
        "reference_matched": reference_matched,
        **figure_ocr_debug,
    }

def _normalize_evidence_list(evidence: list[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in evidence or []:
        if isinstance(ev, dict):
            out.append(_normalize_evidence_item(ev))
    
    return out


def _load_evidence_object_map(
    conn: sqlite3.Connection,
    evidence_object_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not evidence_object_ids:
        return {}

    cols = _get_table_columns(conn, "evidence_objects")
    if not cols:
        return {}

    placeholders = ",".join("?" for _ in evidence_object_ids)

    select_cols = [
        "id",
        "paper_id",
        "object_type",
        "label",
        "page",
        "page_start",
        "page_end",
        "section",
        "section_number",
        "caption_text",
        "retrieval_text",
        "content_text",
    ]
    optional_cols = ["bbox_json", "span_ids_json", "asset_path", "metadata_json"]
    for c in optional_cols:
        if c in cols:
            select_cols.append(c)

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM evidence_objects
        WHERE id IN ({placeholders})
        """,
        evidence_object_ids,
    ).fetchall()
    conn.row_factory = None

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        d = dict(row)
        d["bbox"] = _json_load_maybe(d.get("bbox_json"), None)
        d["span_ids"] = _json_load_maybe(d.get("span_ids_json"), [])
        d["metadata"] = _json_load_maybe(d.get("metadata_json"), {})
        # Prefer detector-refined bbox over caption-inferred bbox when available
        _detected = d["metadata"].get("detected_bbox") if d["metadata"] else None
        if _detected and len(_detected) == 4:
            d["bbox"] = [float(v) for v in _detected]
        out[d["id"]] = d
    return out


# ---------------------------------------------------------------------------
# Verification loaders
# ---------------------------------------------------------------------------

def _extract_raw_evidence_json(d: dict[str, Any]) -> str:
    return (
        d.get("evidence_json")
        or d.get("structured_evidence_json")
        or d.get("evidence")
        or "[]"
    )


def _load_latest_success_verification_for_claim(
    conn: sqlite3.Connection,
    claim_id: str,
) -> dict[str, Any] | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT *
        FROM verification_results
        WHERE claim_id = ?
          AND (status = 'success' OR status IS NULL OR status = '')
        ORDER BY verified_at DESC
        LIMIT 1
        """,
        (claim_id,),
    ).fetchone()
    conn.row_factory = None

    if not row:
        return None

    d = dict(row)
    d["evidence_chunk_ids"] = _json_load_maybe(d.get("evidence_chunk_ids"), [])
    d["evidence"] = _normalize_evidence_list(
        _json_load_maybe(_extract_raw_evidence_json(d), [])
    )
    return d


def _load_latest_verification_map(
    conn: sqlite3.Connection,
    claim_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not claim_ids:
        return {}

    placeholders = ",".join("?" for _ in claim_ids)

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT *
        FROM verification_results
        WHERE claim_id IN ({placeholders})
          AND (status = 'success' OR status IS NULL OR status = '')
        ORDER BY verified_at DESC
        """,
        claim_ids,
    ).fetchall()
    conn.row_factory = None

    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        d = dict(row)
        cid = d["claim_id"]
        if cid in latest:
            continue

        d["evidence_chunk_ids"] = _json_load_maybe(d.get("evidence_chunk_ids"), [])
        d["evidence"] = _normalize_evidence_list(
            _json_load_maybe(_extract_raw_evidence_json(d), [])
        )
        latest[cid] = d

    return latest


# ---------------------------------------------------------------------------
# Retrieval / chunk / span loaders
# ---------------------------------------------------------------------------

def _load_retrieval_rows_for_claim(
    conn: sqlite3.Connection,
    claim_id: str,
) -> list[dict[str, Any]]:
    rr_cols = _get_table_columns(conn, "retrieval_results")
    if not rr_cols:
        return []

    select_cols = [
        "claim_id",
        "rank",
        "combined_score",
    ]

    optional_cols = [
        "chunk_id",
        "evidence_object_id",
        "object_type",
        "label",
        "reference_boost",
    ]
    for c in optional_cols:
        if c in rr_cols:
            select_cols.append(c)

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM retrieval_results
        WHERE claim_id = ?
        ORDER BY rank ASC
        """,
        (claim_id,),
    ).fetchall()
    conn.row_factory = None

    return [dict(r) for r in rows]


def _load_chunk_map(
    conn: sqlite3.Connection,
    chunk_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not chunk_ids:
        return {}

    placeholders = ",".join("?" for _ in chunk_ids)

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT
            id,
            paper_id,
            chunk_index,
            section,
            page,
            page_start,
            page_end,
            text,
            span_ids_json
        FROM paper_chunks
        WHERE id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    conn.row_factory = None

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        d = dict(row)
        d["span_ids"] = _json_load_maybe(d.get("span_ids_json"), [])
        out[d["id"]] = d
    return out


def _load_pdf_spans_by_ids(
    conn: sqlite3.Connection,
    span_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not span_ids:
        return {}

    placeholders = ",".join("?" for _ in span_ids)

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT
            id,
            paper_id,
            page_num,
            span_index,
            text,
            bbox_json,
            block_index,
            line_index
        FROM pdf_spans
        WHERE id IN ({placeholders})
        """,
        span_ids,
    ).fetchall()
    conn.row_factory = None

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        d = dict(row)
        d["bbox"] = _json_load_maybe(d.get("bbox_json"), None)
        out[d["id"]] = d
    return out


def _looks_like_page_number(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    if re.fullmatch(r"\d{1,3}", t):
        return True
    if re.fullmatch(r"[-–—]?\s*\d{1,3}\s*[-–—]?", t):
        return True
    if re.fullmatch(r"\d{1,3}\s*/\s*\d{1,4}", t):
        return True
    if re.fullmatch(r"(?i)page\s+\d{1,3}", t):
        return True

    return False


def _is_probable_margin_span(
    span: dict[str, Any],
    page_height: float | None,
) -> bool:
    """
    Conservative filter:
    only remove very obvious page numbers for now.
    Do NOT filter top/bottom short text yet, because inferred page heights
    from partial evidence spans can be inaccurate and over-filter正文.
    """
    text = (span.get("text") or "").strip()
    if not text:
        return False

    return _looks_like_page_number(text)


def _infer_page_heights_from_spans(
    span_map: dict[str, dict[str, Any]],
) -> dict[int, float]:
    page_heights: dict[int, float] = {}

    for span in span_map.values():
        page_num = span.get("page_num")
        bbox = span.get("bbox")
        if page_num is None or not bbox or len(bbox) != 4:
            continue

        y1 = bbox[3]
        prev = page_heights.get(page_num, 0.0)
        if y1 > prev:
            page_heights[page_num] = float(y1)

    return page_heights


def _filter_margin_boxes(
    boxes: list[dict[str, Any]],
    span_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove obvious header/footer/page-number span boxes.
    Only applies to span-based text boxes; object-level figure/table boxes are kept.
    """
    if not boxes:
        return boxes

    page_heights = _infer_page_heights_from_spans(span_map)
    kept: list[dict[str, Any]] = []

    for box in boxes:
        span_id = box.get("span_id")
        object_type = box.get("object_type") or box.get("evidence_type") or "text_chunk"

        if not span_id or object_type != "text_chunk":
            kept.append(box)
            continue

        span = span_map.get(span_id)
        if not span:
            kept.append(box)
            continue

        page_num = span.get("page_num")
        page_height = page_heights.get(page_num) if page_num is not None else None

        if _is_probable_margin_span(span, page_height):
            continue

        kept.append(box)

    return kept


def _bbox_area(bbox: list[float] | tuple[float, float, float, float] | None) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    x0, y0, x1, y1 = bbox
    return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))


# ---------------------------------------------------------------------------
# P1.8 — Evidence-type-specific bbox sanitization helpers
# ---------------------------------------------------------------------------

def _union_bbox(bboxes: list[list[float] | None]) -> list[float] | None:
    """Union of a list of [x0, y0, x1, y1] bboxes.  Returns None if no valid entry."""
    valid = [b for b in bboxes if b and len(b) == 4]
    if not valid:
        return None
    return [
        min(float(b[0]) for b in valid),
        min(float(b[1]) for b in valid),
        max(float(b[2]) for b in valid),
        max(float(b[3]) for b in valid),
    ]


def _infer_page_widths_from_spans(
    span_map: dict[str, dict[str, Any]],
) -> dict[int, float]:
    """Infer per-page width from the maximum x1 seen across spans."""
    page_widths: dict[int, float] = {}
    for span in span_map.values():
        page_num = span.get("page_num")
        bbox = span.get("bbox")
        if page_num is None or not bbox or len(bbox) != 4:
            continue
        x1 = float(bbox[2])
        if x1 > page_widths.get(page_num, 0.0):
            page_widths[page_num] = x1
    return page_widths


def _cluster_boxes_by_column(
    boxes: list[dict[str, Any]],
    page_width: float,
) -> list[list[dict[str, Any]]]:
    """Group span boxes into column clusters by horizontal x0 position.

    A new cluster starts when x0 of the next box (sorted left-to-right) lies
    beyond the current cluster's right edge (max x1) by more than
    ``gap_tolerance``.  A small tolerance (~1.5% of page width, min 5 pt)
    absorbs paragraph indentation and minor x0 variation within a column
    while correctly splitting the typical two-column gutter (12–25 pt).

    Why not 10% of page width?  Academic two-column PDFs have gutters of
    12–25 pt on a ~595–612 pt page (2–4%), so a 10% threshold would never
    split columns.  A ~1.5% / 5 pt tolerance is tight enough to split on
    the gutter while loose enough to tolerate intra-column indentation.
    """
    if not boxes:
        return []

    # Small tolerance: absorbs intra-column x variation; splits on column gutters
    gap_tolerance = max(5.0, page_width * 0.015) if page_width > 0 else 8.0

    sorted_boxes = sorted(
        boxes,
        key=lambda b: float((b.get("bbox") or [0, 0, 0, 0])[0]),
    )

    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_x1 = -1.0

    for box in sorted_boxes:
        bbox = box.get("bbox")
        if not bbox or len(bbox) < 4:
            # No spatial info: attach to current cluster (safe fallback)
            if current:
                current.append(box)
            else:
                current = [box]
            continue

        x0 = float(bbox[0])
        x1 = float(bbox[2])

        if not current:
            current = [box]
            current_x1 = x1
        elif x0 > current_x1 + gap_tolerance:
            # x0 of new box is beyond the cluster's right edge — different column
            clusters.append(current)
            current = [box]
            current_x1 = x1
        else:
            current.append(box)
            current_x1 = max(current_x1, x1)

    if current:
        clusters.append(current)

    return clusters


def _select_dominant_cluster(
    clusters: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return the cluster with the most boxes; tie-break by total bbox area."""
    if not clusters:
        return []
    return max(
        clusters,
        key=lambda c: (len(c), sum(_bbox_area(b.get("bbox")) for b in c)),
    )


def _consolidate_text_chunk_boxes(
    raw_boxes: list[dict[str, Any]],
    page_heights: dict[int, float],
    page_widths: dict[int, float],
) -> list[dict[str, Any]]:
    """Sanitize text_chunk span boxes into one merged box per (evidence, page).

    For each group of text_chunk boxes sharing the same evidence_index and
    page_num:

    1. Header/footer geometric filter — remove spans whose vertical centre
       falls in the top 6% or bottom 6% of page height (defensive pass,
       complements the text-based ``_filter_margin_boxes`` already applied).
    2. Column clustering — sort remaining boxes by x0 and split into column
       clusters whenever a horizontal gap > 10% of page width appears.
    3. Dominant cluster selection — keep the cluster with the most boxes
       (tie-break: largest total area) to avoid swallowing a figure/table
       column that leaked into the span list.
    4. Union — merge the dominant cluster into a single bbox.

    Non-text-chunk boxes (figure, table) are passed through unchanged.
    Fails open at every step: if filtering/clustering yields nothing, the
    original boxes for that group are kept.
    """
    non_text: list[dict[str, Any]] = []
    text_boxes: list[dict[str, Any]] = []

    for box in raw_boxes:
        if box.get("object_type", "text_chunk") == "text_chunk":
            text_boxes.append(box)
        else:
            non_text.append(box)

    if not text_boxes:
        return raw_boxes

    # Group by (evidence_index, page_num)
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    ungroupable: list[dict[str, Any]] = []

    for box in text_boxes:
        ev_idx = box.get("evidence_index")
        pg = box.get("page_num")
        if ev_idx is None or pg is None:
            ungroupable.append(box)
            continue
        groups.setdefault((int(ev_idx), int(pg)), []).append(box)

    consolidated: list[dict[str, Any]] = []

    for (ev_idx, page_num), group in sorted(groups.items()):
        ph = page_heights.get(page_num, 0.0)
        pw = page_widths.get(page_num, 0.0)

        # Step 1: header/footer geometric filter
        if ph > 0:
            top_band = 0.06 * ph
            bottom_band = ph - 0.06 * ph
            filtered = [
                b for b in group
                if not (
                    b.get("bbox") and len(b["bbox"]) == 4
                    and (
                        (float(b["bbox"][1]) + float(b["bbox"][3])) / 2.0 < top_band
                        or (float(b["bbox"][1]) + float(b["bbox"][3])) / 2.0 > bottom_band
                    )
                )
            ]
            if not filtered:
                filtered = group  # fail-open: all were in band — keep originals
        else:
            filtered = group

        # Steps 2+3: column clustering (only useful when page width is known)
        if pw > 0 and len(filtered) > 1:
            clusters = _cluster_boxes_by_column(filtered, pw)
            dominant = _select_dominant_cluster(clusters)
            if not dominant:
                dominant = filtered  # fail-open
        else:
            dominant = filtered

        # Step 4: union into one merged box
        union = _union_bbox([b.get("bbox") for b in dominant])

        template = dict(dominant[0])
        if union is not None:
            template["bbox"] = union
        template["is_primary"] = True
        template["rank_within_evidence"] = 0
        consolidated.append(template)

    return non_text + consolidated + ungroupable


def _choose_primary_page_for_evidence(
    boxes_for_evidence: list[dict[str, Any]],
    fallback_page: int | None = None,
) -> int | None:
    """
    Pick a single primary page for one evidence item.

    Priority:
    1) page of first reference_matched box
    2) page with largest total bbox area
    3) first page seen in boxes
    4) fallback_page
    """
    if not boxes_for_evidence:
        return fallback_page

    ref_box = next((b for b in boxes_for_evidence if b.get("reference_matched")), None)
    if ref_box and ref_box.get("page_num") is not None:
        return ref_box.get("page_num")

    area_by_page: dict[int, float] = {}
    seen_pages: list[int] = []

    for b in boxes_for_evidence:
        p = b.get("page_num")
        if p is None:
            continue
        if p not in area_by_page:
            area_by_page[p] = 0.0
            seen_pages.append(p)
        area_by_page[p] += _bbox_area(b.get("bbox"))

    if area_by_page:
        return max(area_by_page.items(), key=lambda kv: kv[1])[0]

    first_page = next((b.get("page_num") for b in boxes_for_evidence if b.get("page_num") is not None), None)
    if first_page is not None:
        return first_page

    return fallback_page


def _build_evidence_list_from_retrieval_rows(
    conn: sqlite3.Connection,
    claim_id: str,
) -> list[dict[str, Any]]:
    rows = _load_retrieval_rows_for_claim(conn, claim_id)
    if not rows:
        return []

    chunk_ids: list[str] = []
    evidence_object_ids: list[str] = []

    for r in rows:
        cid = r.get("chunk_id")
        if cid and cid not in chunk_ids:
            chunk_ids.append(cid)

        eoid = r.get("evidence_object_id")
        if eoid and eoid not in evidence_object_ids:
            evidence_object_ids.append(eoid)

    chunk_map = _load_chunk_map(conn, chunk_ids)
    eo_map = _load_evidence_object_map(conn, evidence_object_ids)

    out: list[dict[str, Any]] = []

    for i, r in enumerate(rows):
        chunk_id = r.get("chunk_id")
        evidence_object_id = r.get("evidence_object_id")

        chunk = chunk_map.get(chunk_id) if chunk_id else None
        eo = eo_map.get(evidence_object_id) if evidence_object_id else None

        object_type = (
            r.get("object_type")
            or (eo.get("object_type") if eo else None)
            or "text_chunk"
        )

        page = (
            (eo.get("page") if eo else None)
            or (eo.get("page_start") if eo else None)
            or (chunk.get("page") if chunk else None)
            or (chunk.get("page_start") if chunk else None)
        )

        ev = {
            "rank": r.get("rank"),
            "chunk_id": chunk_id,
            "evidence_object_id": evidence_object_id,
            "object_type": object_type,
            "evidence_type": object_type,
            "label": r.get("label") or (eo.get("label") if eo else None) or f"E{i + 1}",
            "caption": eo.get("caption_text") if eo else None,
            "caption_text": eo.get("caption_text") if eo else None,
            "text": (
                (eo.get("content_text") if eo else None)
                or (eo.get("retrieval_text") if eo else None)
                or (chunk.get("text") if chunk else None)
            ),
            "page": page,
            "page_num": page,
            "page_start": (eo.get("page_start") if eo else None) or (chunk.get("page_start") if chunk else None),
            "page_end": (eo.get("page_end") if eo else None) or (chunk.get("page_end") if chunk else None),
            "section": (eo.get("section") if eo else None) or (chunk.get("section") if chunk else None),
            "score": r.get("combined_score"),
            "span_ids": (eo.get("span_ids") if eo else None) or (chunk.get("span_ids") if chunk else []),
            "bbox": eo.get("bbox") if eo else None,
            "reference_boost": r.get("reference_boost", 0.0) or 0.0,
            "reference_matched": bool((r.get("reference_boost", 0.0) or 0.0) > 0),
        }

        out.append(_normalize_evidence_item(ev))

    return out


# ---------------------------------------------------------------------------
# Claim enrichment helpers
# ---------------------------------------------------------------------------

def _enrich_claims_for_review(
    *,
    claims: list[dict[str, Any]],
    review_fields: dict[str, str],
    verification_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched_claims: list[dict[str, Any]] = []

    for claim in claims:
        vr = verification_map.get(claim["id"])

        field_text = review_fields.get(claim["source_field"], "")
        display_sentence, display_sentence_score, display_start, display_end = _best_sentence_like_report(
            field_text,
            claim.get("claim_text") or "",
        )

        enriched_claims.append(
            {
                "id": claim["id"],
                "review_id": claim["review_id"],
                "paper_id": claim["paper_id"],
                "source_field": claim["source_field"],
                "source_field_raw": claim.get("source_field_raw"),
                "claim_index": claim["claim_index"],
                "claim_text": claim["claim_text"],
                "verbatim_quote": claim.get("verbatim_quote"),
                "display_sentence": display_sentence,
                "display_sentence_score": display_sentence_score,
                "display_start": display_start,
                "display_end": display_end,
                "claim_type": claim.get("claim_type"),
                "category": claim.get("category"),
                "extraction_confidence": claim.get("confidence"),
                "challengeability": claim.get("challengeability"),
                "calibrated_score": claim.get("calibrated_score"),
                "verification": None
                if not vr
                else {
                    "id": vr.get("id"),
                    "verdict": vr.get("verdict"),
                    "reasoning": vr.get("reasoning"),
                    "confidence": vr.get("confidence"),
                    "supporting_quote": vr.get("supporting_quote"),
                    "evidence_chunk_ids": vr.get("evidence_chunk_ids", []),
                    "evidence": vr.get("evidence", []),
                    "model_id": vr.get("model_id"),
                    "verified_at": vr.get("verified_at"),
                },
            }
        )

    enriched_claims.sort(key=_claim_sort_key)
    return enriched_claims


def _count_verified_claims(claims: list[dict[str, Any]]) -> int:
    return sum(1 for c in claims if c.get("verification"))


# ---------------------------------------------------------------------------
# Review workspace
# ---------------------------------------------------------------------------

def get_review_workspace(
    conn: sqlite3.Connection,
    review_id: str,
) -> dict[str, Any]:
    review = _load_review(conn, review_id)
    if not review:
        raise ValueError(f"Review not found: {review_id}")

    paper_id = review["paper_id"]
    paper = _load_paper(conn, paper_id)
    if not paper:
        raise ValueError(f"Paper not found: {paper_id}")

    fields_payload = _build_review_fields_payload(review)

    review_claims_raw = _load_claims_for_review(conn, review_id)
    review_claim_ids = [c["id"] for c in review_claims_raw]
    review_verification_map = _load_latest_verification_map(conn, review_claim_ids)
    review_claims = _enrich_claims_for_review(
        claims=review_claims_raw,
        review_fields=fields_payload["normalized_fields"],
        verification_map=review_verification_map,
    )

    paper_claims_raw = _load_claims_for_paper(conn, paper_id)
    paper_claim_ids = [c["id"] for c in paper_claims_raw]
    paper_verification_map = _load_latest_verification_map(conn, paper_claim_ids)

    return {
        "mode": "review",
        "paper": {
            "id": paper["id"],
            "title": paper.get("title"),
            "authors": paper.get("authors", []),
            "abstract": paper.get("abstract"),
            "keywords": paper.get("keywords", []),
            "pdf_path": paper.get("pdf_path"),
        },
        "review": {
            "id": review["id"],
            "paper_id": review["paper_id"],
            "forum": review.get("forum"),
            "rating": review.get("rating"),
            "confidence": review.get("confidence"),
            "fields": fields_payload["fields"],
            "normalized_fields": fields_payload["normalized_fields"],
            "raw_fields": fields_payload["raw_fields"],
            "field_labels": fields_payload["field_labels"],
            "field_order": fields_payload["field_order"],
        },
        "claims": review_claims,
        "stats": {
            "paper_total_reviews": len(_load_reviews_for_paper(conn, paper_id)),
            "paper_total_claims": len(paper_claims_raw),
            "paper_verified_claims": len(paper_verification_map),
            "current_review_claims": len(review_claims_raw),
            "current_review_verified_claims": len(review_verification_map),
        },
    }


# ---------------------------------------------------------------------------
# Paper workspace
# ---------------------------------------------------------------------------

def get_paper_workspace(
    conn: sqlite3.Connection,
    paper_id: str,
) -> dict[str, Any]:
    paper = _load_paper(conn, paper_id)
    if not paper:
        raise ValueError(f"Paper not found: {paper_id}")

    reviews = _load_reviews_for_paper(conn, paper_id)
    all_claims_raw = _load_claims_for_paper(conn, paper_id)
    all_claim_ids = [c["id"] for c in all_claims_raw]
    verification_map = _load_latest_verification_map(conn, all_claim_ids)

    claims_by_review: dict[str, list[dict[str, Any]]] = {}
    for claim in all_claims_raw:
        claims_by_review.setdefault(claim["review_id"], []).append(claim)

    review_items: list[dict[str, Any]] = []
    total_verified = 0

    for review in reviews:
        review_id = review["id"]
        fields_payload = _build_review_fields_payload(review)
        raw_claims = claims_by_review.get(review_id, [])

        enriched_claims = _enrich_claims_for_review(
            claims=raw_claims,
            review_fields=fields_payload["normalized_fields"],
            verification_map=verification_map,
        )
        verified_count = _count_verified_claims(enriched_claims)
        total_verified += verified_count

        review_items.append(
            {
                "review": {
                    "id": review["id"],
                    "paper_id": review["paper_id"],
                    "forum": review.get("forum"),
                    "rating": review.get("rating"),
                    "confidence": review.get("confidence"),
                    "fields": fields_payload["fields"],
                    "normalized_fields": fields_payload["normalized_fields"],
                    "raw_fields": fields_payload["raw_fields"],
                    "field_labels": fields_payload["field_labels"],
                    "field_order": fields_payload["field_order"],
                },
                "claims": enriched_claims,
                "stats": {
                    "review_total_claims": len(raw_claims),
                    "review_verified_claims": verified_count,
                },
            }
        )

    return {
        "mode": "paper",
        "paper": {
            "id": paper["id"],
            "title": paper.get("title"),
            "authors": paper.get("authors", []),
            "abstract": paper.get("abstract"),
            "keywords": paper.get("keywords", []),
            "pdf_path": paper.get("pdf_path"),
        },
        "reviews": review_items,
        "stats": {
            "paper_total_reviews": len(reviews),
            "paper_total_claims": len(all_claims_raw),
            "paper_verified_claims": total_verified,
        },
    }


# ---------------------------------------------------------------------------
# Claim boxes for right-panel PDF highlighting
# ---------------------------------------------------------------------------

def get_claim_boxes(
    conn: sqlite3.Connection,
    claim_id: str,
) -> dict[str, Any]:
    claim = _load_claim_basic(conn, claim_id)
    if not claim:
        raise ValueError(f"Claim not found: {claim_id}")

    vr = _load_latest_success_verification_for_claim(conn, claim_id)
    if not vr:
        return {
            "claim_id": claim_id,
            "claim": claim,
            "verification": None,
            "preferred_page": None,
            "boxes": [],
            "grouped_by_page": {},
            "evidences": [],
        }

    evidence_list = _normalize_evidence_list(vr.get("evidence", []) or [])

    # Fallback: rebuild from retrieval cache if verification stored no structured evidence
    if not evidence_list:
        evidence_list = _build_evidence_list_from_retrieval_rows(conn, claim_id)

    fallback_chunk_ids: list[str] = []
    evidence_object_ids: list[str] = []

    for ev in evidence_list:
        if ev.get("chunk_id") and not ev.get("aligned_span_ids") and not ev.get("span_ids"):
            cid = ev["chunk_id"]
            if cid not in fallback_chunk_ids:
                fallback_chunk_ids.append(cid)

        eo_id = ev.get("evidence_object_id")
        if eo_id and eo_id not in evidence_object_ids:
            evidence_object_ids.append(eo_id)

    chunk_map = _load_chunk_map(conn, fallback_chunk_ids)
    evidence_object_map = _load_evidence_object_map(conn, evidence_object_ids)

    all_span_ids: list[str] = []
    evidence_span_plan: list[dict[str, Any]] = []

    for i, ev in enumerate(evidence_list):
        evidence_object_id = ev.get("evidence_object_id")
        eo = evidence_object_map.get(evidence_object_id) if evidence_object_id else None

        object_type = ev.get("object_type") or (eo.get("object_type") if eo else None) or "text_chunk"
        label = ev.get("label") or (eo.get("label") if eo else None) or f"E{i + 1}"

        aligned_span_ids = ev.get("aligned_span_ids") or []
        span_ids = ev.get("span_ids") or []
        chunk_id = ev.get("chunk_id")

        eo_span_ids = (eo or {}).get("span_ids") or []
        eo_bbox = (eo or {}).get("bbox")
        eo_page = (eo or {}).get("page") or (eo or {}).get("page_start")

        # text_chunk: prefer chunk-level coverage
        # figure/table: prefer object-level span_ids/bbox, but no long text
        if object_type == "text_chunk":
            chunk = chunk_map.get(chunk_id) if chunk_id else None
            chunk_span_ids = (chunk or {}).get("span_ids") or []

            if chunk_span_ids:
                chosen_span_ids = chunk_span_ids
                span_source = "chunk_span_ids"
            elif span_ids:
                chosen_span_ids = span_ids
                span_source = "span_ids"
            elif aligned_span_ids:
                chosen_span_ids = aligned_span_ids
                span_source = "aligned_span_ids"
            elif eo_span_ids:
                chosen_span_ids = eo_span_ids
                span_source = "evidence_object_span_ids"
            else:
                chosen_span_ids = []
                span_source = None

            current_text = (
                ev.get("text")
                or (chunk.get("text") if chunk else None)
                or (eo.get("content_text") if eo else None)
            )
        else:
            if eo_span_ids:
                chosen_span_ids = eo_span_ids
                span_source = "evidence_object_span_ids"
            elif span_ids:
                chosen_span_ids = span_ids
                span_source = "span_ids"
            elif aligned_span_ids:
                chosen_span_ids = aligned_span_ids
                span_source = "aligned_span_ids"
            else:
                chunk = chunk_map.get(chunk_id) if chunk_id else None
                chunk_span_ids = (chunk or {}).get("span_ids") or []

                if chunk_span_ids:
                    chosen_span_ids = chunk_span_ids
                    span_source = "chunk_span_ids"
                else:
                    chosen_span_ids = []
                    span_source = None

            current_text = None

        evidence_span_plan.append(
            {
                "evidence_index": i,
                "evidence_label": f"E{i + 1}",
                "chunk_id": chunk_id,
                "evidence_object_id": evidence_object_id,
                "object_type": object_type,
                "label": label,
                "page": ev.get("page") or eo_page,
                "section": ev.get("section") or (eo.get("section") if eo else None),
                "text": current_text,
                "caption": ev.get("caption") or (eo.get("caption_text") if eo else None),
                "score": ev.get("score"),
                "reference_matched": ev.get("reference_matched", False),
                "bbox": ev.get("bbox") or eo_bbox,
                "span_source": span_source,
                "chosen_span_ids": chosen_span_ids,
            }
        )

        for sid in chosen_span_ids:
            if sid not in all_span_ids:
                all_span_ids.append(sid)

    span_map = _load_pdf_spans_by_ids(conn, all_span_ids)

    raw_boxes: list[dict[str, Any]] = []

    for ev_plan in evidence_span_plan:
        chosen_span_ids = ev_plan["chosen_span_ids"]
        object_type = ev_plan["object_type"]

        # figure/table: prefer one object-level bbox
        if object_type != "text_chunk" and ev_plan.get("bbox"):
            # Determine highlight confidence from evidence object metadata.
            eo = evidence_object_map.get(ev_plan.get("evidence_object_id") or "")
            eo_meta = (eo or {}).get("metadata") or {}
            bbox_confidence = eo_meta.get("object_bbox_confidence") or (
                "caption_only" if eo_meta.get("object_bbox_inferred") else "high"
            )
            # "object_bbox" = reliable region; "caption_anchor" = only caption known
            highlight_mode = "caption_anchor" if bbox_confidence == "caption_only" else "object_bbox"

            raw_boxes.append(
                {
                    "evidence_index": ev_plan["evidence_index"],
                    "evidence_label": ev_plan["evidence_label"],
                    "chunk_id": ev_plan.get("chunk_id"),
                    "evidence_object_id": ev_plan.get("evidence_object_id"),
                    "object_type": object_type,
                    "label": ev_plan.get("label"),
                    "page": ev_plan.get("page"),
                    "section": ev_plan.get("section"),
                    "score": ev_plan.get("score"),
                    "reference_matched": ev_plan.get("reference_matched", False),
                    "span_source": "evidence_object_bbox",
                    "span_id": None,
                    "page_num": ev_plan.get("page"),
                    "bbox": ev_plan.get("bbox"),
                    "text": ev_plan.get("text"),
                    "caption": ev_plan.get("caption"),
                    "rank_within_evidence": 0,
                    "is_primary": True,
                    "highlight_mode": highlight_mode,
                    "bbox_confidence": bbox_confidence,
                }
            )
            continue

        # text chunks: keep span-based boxes
        for rank, sid in enumerate(chosen_span_ids):
            span = span_map.get(sid)
            if not span:
                continue

            raw_boxes.append(
                {
                    "evidence_index": ev_plan["evidence_index"],
                    "evidence_label": ev_plan["evidence_label"],
                    "chunk_id": ev_plan.get("chunk_id"),
                    "evidence_object_id": ev_plan.get("evidence_object_id"),
                    "object_type": object_type,
                    "label": ev_plan.get("label"),
                    "page": ev_plan.get("page"),
                    "section": ev_plan.get("section"),
                    "score": ev_plan.get("score"),
                    "reference_matched": ev_plan.get("reference_matched", False),
                    "span_source": ev_plan.get("span_source"),
                    "span_id": sid,
                    "page_num": span.get("page_num"),
                    "bbox": span.get("bbox"),
                    "text": span.get("text"),
                    "caption": ev_plan.get("caption"),
                    "rank_within_evidence": rank,
                    "is_primary": rank == 0,
                }
            )

        if not chosen_span_ids and ev_plan.get("bbox"):
            raw_boxes.append(
                {
                    "evidence_index": ev_plan["evidence_index"],
                    "evidence_label": ev_plan["evidence_label"],
                    "chunk_id": ev_plan.get("chunk_id"),
                    "evidence_object_id": ev_plan.get("evidence_object_id"),
                    "object_type": object_type,
                    "label": ev_plan.get("label"),
                    "page": ev_plan.get("page"),
                    "section": ev_plan.get("section"),
                    "score": ev_plan.get("score"),
                    "reference_matched": ev_plan.get("reference_matched", False),
                    "span_source": "evidence_object_bbox",
                    "span_id": None,
                    "page_num": ev_plan.get("page"),
                    "bbox": ev_plan.get("bbox"),
                    "text": ev_plan.get("text"),
                    "caption": ev_plan.get("caption"),
                    "rank_within_evidence": 0,
                    "is_primary": True,
                }
            )

    raw_boxes = _filter_margin_boxes(raw_boxes, span_map)

    # P1.8 — consolidate text_chunk span boxes: header/footer filter + column clustering
    _page_heights_for_consolidation = _infer_page_heights_from_spans(span_map)
    _page_widths_for_consolidation = _infer_page_widths_from_spans(span_map)
    raw_boxes = _consolidate_text_chunk_boxes(
        raw_boxes,
        _page_heights_for_consolidation,
        _page_widths_for_consolidation,
    )

    raw_boxes.sort(
        key=lambda x: (
            x.get("page_num") if x.get("page_num") is not None else 10**9,
            x.get("evidence_index", 10**9),
            x.get("rank_within_evidence", 10**9),
        )
    )

    boxes_by_evidence_index: dict[int, list[dict[str, Any]]] = {}
    for box in raw_boxes:
        idx = box.get("evidence_index")
        if isinstance(idx, int):
            boxes_by_evidence_index.setdefault(idx, []).append(box)

    evidence_cards: list[dict[str, Any]] = []
    selected_boxes: list[dict[str, Any]] = []

    for ev_plan in evidence_span_plan:
        idx = ev_plan["evidence_index"]
        ev_boxes = boxes_by_evidence_index.get(idx, [])

        all_pages = sorted(
            {
                b.get("page_num")
                for b in ev_boxes
                if b.get("page_num") is not None
            }
        )

        primary_page = _choose_primary_page_for_evidence(
            ev_boxes,
            fallback_page=ev_plan.get("page"),
        )

        primary_page_boxes = [
            b for b in ev_boxes if b.get("page_num") == primary_page
        ] if primary_page is not None else []

        first_box = primary_page_boxes[0] if primary_page_boxes else (ev_boxes[0] if ev_boxes else None)

        evidence_cards.append(
            {
                "evidence_index": idx,
                "evidence_label": ev_plan["evidence_label"],
                "label": ev_plan.get("label"),
                "evidence_object_id": ev_plan.get("evidence_object_id"),
                "chunk_id": ev_plan.get("chunk_id"),
                "type": ev_plan.get("object_type"),
                "object_type": ev_plan.get("object_type"),
                "evidence_type": ev_plan.get("object_type"),
                "page": primary_page,
                "page_num": primary_page,
                "pages": all_pages,
                "page_start": all_pages[0] if all_pages else primary_page,
                "page_end": all_pages[-1] if all_pages else primary_page,
                "section": ev_plan.get("section"),
                "text": None if ev_plan.get("object_type") != "text_chunk" else ev_plan.get("text"),  
                "caption": None if ev_plan.get("object_type") in {"figure", "table"} else ev_plan.get("caption"),
                "caption_text": None if ev_plan.get("object_type") in {"figure", "table"} else ev_plan.get("caption"),
                "score": ev_plan.get("score"),
                "reference_matched": ev_plan.get("reference_matched", False),
                "reference_boost": 1.0 if ev_plan.get("reference_matched", False) else 0.0,
                "bbox": first_box.get("bbox") if first_box else ev_plan.get("bbox"),
                "object_bbox": first_box.get("bbox") if first_box else ev_plan.get("bbox"),
                "span_ids": ev_plan.get("chosen_span_ids") or [],
                "aligned_span_ids": [],
                "box": None if not first_box else {
                    "page": first_box.get("page_num"),
                    "page_num": first_box.get("page_num"),
                    "bbox": first_box.get("bbox"),
                    "label": first_box.get("label"),
                    "text": first_box.get("text"),                    
                    "caption": None if first_box.get("object_type") in {"figure", "table"} else first_box.get("caption"),
                    "score": first_box.get("score"),
                    "evidence_label": first_box.get("evidence_label"),
                    "evidence_index": first_box.get("evidence_index"),
                    "section": first_box.get("section"),
                    "object_type": first_box.get("object_type"),
                    "reference_matched": first_box.get("reference_matched", False),
                },
            }
        )

        selected_boxes.extend(primary_page_boxes)

    selected_boxes.sort(
        key=lambda x: (
            x.get("page_num") if x.get("page_num") is not None else 10**9,
            x.get("evidence_index", 10**9),
            x.get("rank_within_evidence", 10**9),
        )
    )

    grouped_by_page: dict[str, list[dict[str, Any]]] = {}
    for box in selected_boxes:
        key = str(box["page_num"])
        grouped_by_page.setdefault(key, []).append(box)

    preferred_page = None
    ref_box = next((b for b in selected_boxes if b.get("reference_matched")), None)
    if ref_box:
        preferred_page = ref_box.get("page_num")
    elif selected_boxes:
        preferred_page = selected_boxes[0].get("page_num")
    elif evidence_cards:
        preferred_page = evidence_cards[0].get("page_num")

    verification_summary = {
        "id": vr.get("id"),
        "verdict": vr.get("verdict"),
        "reasoning": vr.get("reasoning"),
        "confidence": vr.get("confidence"),
        "supporting_quote": vr.get("supporting_quote"),
        "model_id": vr.get("model_id"),
        "verified_at": vr.get("verified_at"),
    }

    return {
        "claim_id": claim_id,
        "claim": claim,
        "verification": verification_summary,
        "preferred_page": preferred_page,
        "boxes": selected_boxes,
        "grouped_by_page": grouped_by_page,
        "evidences": evidence_cards,
    }