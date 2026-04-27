"""Object-level audit/debug report for figure/table evidence objects.

Reads existing evidence_objects and pdf_spans rows from SQLite and produces:
  object_audit_<paper_id_safe>.csv  — per-object detail table
  object_audit_<paper_id_safe>.md   — human-readable summary

Purpose: diagnose two concrete UI failures:
  A) False table identity (plain text blocks labelled TBL)
  B) Oversized / mixed figure-or-table bounding boxes

This module is strictly read-only.  It never writes to the database.
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from gsr.config import REPORT_DIR

# ---------------------------------------------------------------------------
# Thresholds (mirrors evidence_builder constants — kept local for clarity)
# ---------------------------------------------------------------------------

_HEIGHT_FRACTION_WARN = 0.25    # flag if object_bbox height > 25% of page
_HEIGHT_FRACTION_REJECT = 0.30  # mirrors _TABLE_MAX_HEIGHT_PAGE_FRACTION
_PROSE_LONG_LINE_CHARS = 80     # mirrors _TABLE_PROSE_LONG_LINE_CHARS
_PROSE_LONG_LINE_THRESHOLD = 2  # mirrors _TABLE_PROSE_LONG_LINE_THRESHOLD
_HIGH_SPAN_COUNT = 15           # flag if object has > N spans (likely mixed region)

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _truncate_preview(text: str | None, limit: int) -> str:
    if not text:
        return ""
    t = " ".join(str(text).split())  # collapse whitespace
    if len(t) <= limit:
        return t
    return t[: max(0, limit - 3)].rstrip() + "..."


def _looks_like_table_caption(text: str | None) -> bool:
    """Return True if text starts with a canonical table caption prefix."""
    if not text:
        return False
    return bool(re.match(r"(table|tab\.?)\s*\d+", str(text).strip(), re.IGNORECASE))


def _looks_like_valid_table_label(label: str | None) -> bool:
    """Return True if label looks like a real table label ("Table 4", "Tab. 2")."""
    if not label or not label.strip():
        return False
    return bool(re.match(r"(table|tab\.?)\s*\d+", label.strip(), re.IGNORECASE))


def _looks_like_prose_preview(text: str | None) -> bool:
    """Return True if text looks like a prose paragraph rather than table rows.

    Mirrors the _looks_like_prose_block() heuristic in evidence_builder.py.
    """
    if not text:
        return False
    lines = [ln.strip() for ln in str(text).split("\n") if ln.strip()]
    if not lines:
        return False
    long_count = sum(1 for ln in lines if len(ln) > _PROSE_LONG_LINE_CHARS)
    if long_count >= _PROSE_LONG_LINE_THRESHOLD:
        return True
    short_count = sum(1 for ln in lines if len(ln) <= _PROSE_LONG_LINE_CHARS)
    if len(lines) > 3 and short_count / len(lines) < 0.6:
        return True
    return False


# ---------------------------------------------------------------------------
# BBox helpers (inlined to avoid coupling to evidence_builder internals)
# ---------------------------------------------------------------------------

def _parse_bbox(s: str | None) -> list[float] | None:
    if not s:
        return None
    try:
        v = json.loads(s)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            return [float(x) for x in v]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def _compute_bbox_stats(
    object_bbox: list[float] | None,
    caption_bbox: list[float] | None,
    page_height: float,
) -> dict[str, Any]:
    """Return geometry diagnostics for one evidence object."""
    has_obj = object_bbox is not None
    has_cap = caption_bbox is not None

    stats: dict[str, Any] = {
        "has_object_bbox": has_obj,
        "has_caption_bbox": has_cap,
        "object_bbox_x0": "",
        "object_bbox_y0": "",
        "object_bbox_x1": "",
        "object_bbox_y1": "",
        "object_bbox_width": "",
        "object_bbox_height": "",
        "page_height": round(page_height, 1) if page_height > 0 else "",
        "object_bbox_height_page_fraction": "",
    }

    if has_obj:
        x0, y0, x1, y1 = object_bbox
        w = max(0.0, x1 - x0)
        h = max(0.0, y1 - y0)
        stats["object_bbox_x0"] = round(x0, 1)
        stats["object_bbox_y0"] = round(y0, 1)
        stats["object_bbox_x1"] = round(x1, 1)
        stats["object_bbox_y1"] = round(y1, 1)
        stats["object_bbox_width"] = round(w, 1)
        stats["object_bbox_height"] = round(h, 1)
        if page_height > 0:
            stats["object_bbox_height_page_fraction"] = round(h / page_height, 4)

    return stats


# ---------------------------------------------------------------------------
# Heuristic flags
# ---------------------------------------------------------------------------

def _flag_suspicious_false_table_identity(
    *,
    object_type: str,
    label: str | None,
    caption_text: str | None,
    content_text: str | None,
    height_fraction: float | None,
    docling_match_mode: str | None,
    table_region_validation: str | None,
) -> bool:
    """Diagnostic flag: plain text block potentially mislabelled as TABLE.

    Returns True when object_type == 'table' AND one or more suspicious signals.
    Does NOT affect pipeline behavior.
    """
    if object_type != "table":
        return False

    # Signal 1: missing or malformed label
    if not _looks_like_valid_table_label(label):
        return True

    # Signal 2: content looks like prose instead of table rows
    if _looks_like_prose_preview(content_text):
        return True

    # Signal 3: bbox height fraction unusually large
    if height_fraction is not None and height_fraction > _HEIGHT_FRACTION_REJECT:
        return True

    # Signal 4: caption doesn't look like a table caption
    if not _looks_like_table_caption(caption_text):
        return True

    # Signal 5: no docling match AND table region validation failed AND prose content
    if (
        (not docling_match_mode or docling_match_mode == "none")
        and table_region_validation not in ("passed",)
        and _looks_like_prose_preview(content_text)
    ):
        return True

    return False


def _flag_suspicious_mixed_visual_bbox(
    *,
    object_type: str,
    height_fraction: float | None,
    span_count: int,
    content_text: str | None,
    caption_text: str | None,
) -> bool:
    """Diagnostic flag: object bbox likely covers mixed visual + prose region.

    Returns True when object_type in {'figure','table'} AND one or more signals.
    Does NOT affect pipeline behavior.
    """
    if object_type not in ("figure", "table"):
        return False

    # Signal 1: height fraction larger than warn threshold
    if height_fraction is not None and height_fraction > _HEIGHT_FRACTION_WARN:
        return True

    # Signal 2: unusually high span count (many text lines swept into bbox)
    if span_count > _HIGH_SPAN_COUNT:
        return True

    # Signal 3: content looks like prose leakage (especially for figures)
    if _looks_like_prose_preview(content_text):
        return True

    return False


def _flag_likely_caption_only(
    *,
    object_bbox_confidence: str | None,
    bbox_source: str | None,
) -> bool:
    """Diagnostic flag: bbox covers only the caption anchor, not the visual object."""
    if object_bbox_confidence and "caption_only" in str(object_bbox_confidence):
        return True
    if bbox_source and "caption_only" in str(bbox_source):
        return True
    return False


# ---------------------------------------------------------------------------
# Normalise bbox_source from metadata (mirrors bbox_audit helper)
# ---------------------------------------------------------------------------

def _normalize_bbox_source(meta: dict[str, Any]) -> str:
    src = meta.get("bbox_source")
    if src:
        return str(src)
    detected = meta.get("detected_bbox")
    if detected:
        for detector in ("paddlex_layout", "dino", "rtdetr"):
            if any(k.startswith(detector) for k in meta):
                return "groundingdino" if detector == "dino" else detector
        return "detected_unknown"
    conf = str(meta.get("object_bbox_confidence") or "")
    if "caption_only" in conf or conf == "caption":
        return "caption_only"
    if conf == "inferred":
        return "caption_anchor"
    hm = meta.get("highlight_mode") or ""
    if hm:
        return f"highlight_{hm}"
    return "unknown"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_page_heights(conn: sqlite3.Connection, paper_id: str) -> dict[int, float]:
    """Return {page_num: max_y1} from pdf_spans for this paper.

    Used to compute object_bbox_height_page_fraction.
    Silently returns {} if pdf_spans table is absent.
    """
    try:
        rows = conn.execute(
            "SELECT page_num, bbox_json FROM pdf_spans WHERE paper_id = ?",
            (paper_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    heights: dict[int, float] = {}
    for page_num, bbox_json_str in rows:
        if page_num is None or not bbox_json_str:
            continue
        bbox = _parse_bbox(bbox_json_str)
        if bbox:
            y1 = bbox[3]
            if y1 > heights.get(page_num, 0.0):
                heights[page_num] = y1
    return heights


def _load_objects(
    conn: sqlite3.Connection,
    paper_id: str,
) -> list[dict[str, Any]]:
    """Load figure/table evidence objects and expand metadata inline."""
    rows = conn.execute(
        """
        SELECT id, paper_id, object_type, label,
               page, page_start, page_end, section, section_number,
               caption_text, content_text, retrieval_text,
               bbox_json, span_ids_json, metadata_json
        FROM evidence_objects
        WHERE paper_id = ?
          AND object_type IN ('figure', 'table')
        ORDER BY object_type, page, label
        """,
        (paper_id,),
    ).fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        (
            obj_id, pid, otype, label,
            page, page_start, page_end, section, section_number,
            caption_text, content_text, retrieval_text,
            bbox_json, span_ids_json, metadata_json_str,
        ) = row
        try:
            meta: dict[str, Any] = json.loads(metadata_json_str) if metadata_json_str else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        try:
            span_ids: list[str] = json.loads(span_ids_json) if span_ids_json else []
        except (json.JSONDecodeError, TypeError):
            span_ids = []
        result.append({
            "object_id": obj_id or "",
            "paper_id": pid or "",
            "object_type": otype or "",
            "label": label or "",
            "page": page or page_start or 0,
            "page_start": page_start or 0,
            "page_end": page_end or 0,
            "section": section or "",
            "section_number": section_number or "",
            "caption_text": caption_text or "",
            "content_text": content_text or "",
            "retrieval_text": retrieval_text or "",
            "bbox_json": bbox_json or "",
            "span_ids": span_ids,
            "meta": meta,
        })
    return result


# ---------------------------------------------------------------------------
# Per-object audit record
# ---------------------------------------------------------------------------

def _build_object_audit_record(
    obj: dict[str, Any],
    page_heights: dict[int, float],
) -> dict[str, Any]:
    """Build one flattened audit record for a figure/table evidence object."""
    meta = obj["meta"]
    span_ids: list[str] = obj["span_ids"]
    caption_span_ids: list[str] = meta.get("caption_span_ids") or []

    # --- bbox ---
    object_bbox = _parse_bbox(obj["bbox_json"]) or (
        meta.get("object_bbox") if isinstance(meta.get("object_bbox"), list) else None
    )
    caption_bbox_raw = meta.get("caption_bbox")
    caption_bbox = (
        caption_bbox_raw
        if isinstance(caption_bbox_raw, list) and len(caption_bbox_raw) == 4
        else None
    )

    page_num = obj["page"] or obj["page_start"] or 0
    page_height = page_heights.get(page_num, 0.0)

    geo = _compute_bbox_stats(object_bbox, caption_bbox, page_height)

    height_fraction_raw = geo["object_bbox_height_page_fraction"]
    height_fraction: float | None = (
        float(height_fraction_raw) if height_fraction_raw != "" else None
    )

    # --- source fields ---
    bbox_source = _normalize_bbox_source(meta)
    object_bbox_confidence = str(meta.get("object_bbox_confidence") or "")
    docling_match_mode = str(meta.get("docling_match_mode") or "")
    table_region_validation = str(meta.get("table_region_validation") or "")

    # --- heuristic flags ---
    flag_false_table = _flag_suspicious_false_table_identity(
        object_type=obj["object_type"],
        label=obj["label"],
        caption_text=obj["caption_text"],
        content_text=obj["content_text"],
        height_fraction=height_fraction,
        docling_match_mode=docling_match_mode,
        table_region_validation=table_region_validation,
    )
    flag_mixed_bbox = _flag_suspicious_mixed_visual_bbox(
        object_type=obj["object_type"],
        height_fraction=height_fraction,
        span_count=len(span_ids),
        content_text=obj["content_text"],
        caption_text=obj["caption_text"],
    )
    flag_caption_only = _flag_likely_caption_only(
        object_bbox_confidence=object_bbox_confidence,
        bbox_source=bbox_source,
    )

    return {
        # Identity
        "paper_id": obj["paper_id"],
        "object_id": obj["object_id"],
        "object_type": obj["object_type"],
        "label": obj["label"],
        "page": page_num,
        "page_start": obj["page_start"],
        "page_end": obj["page_end"],
        "section": obj["section"],
        "section_number": obj["section_number"],
        # Source / matching path
        "source_used": meta.get("source_used") or "",
        "bbox_source": bbox_source,
        "object_bbox_confidence": object_bbox_confidence,
        "highlight_mode": meta.get("highlight_mode") or "",
        "docling_match_mode": docling_match_mode,
        "docling_quality_pass": str(meta.get("docling_quality_pass") or ""),
        "docling_rejected_reason": meta.get("docling_rejected_reason") or "",
        # Table-specific validation
        "table_region_validation": table_region_validation,
        "table_fallback_rejected_reason": meta.get("table_fallback_rejected_reason") or "",
        # Geometry
        "has_object_bbox": 1 if geo["has_object_bbox"] else 0,
        "has_caption_bbox": 1 if geo["has_caption_bbox"] else 0,
        "object_bbox_x0": geo["object_bbox_x0"],
        "object_bbox_y0": geo["object_bbox_y0"],
        "object_bbox_x1": geo["object_bbox_x1"],
        "object_bbox_y1": geo["object_bbox_y1"],
        "object_bbox_width": geo["object_bbox_width"],
        "object_bbox_height": geo["object_bbox_height"],
        "page_width": "",  # not currently stored
        "page_height": geo["page_height"],
        "object_bbox_height_page_fraction": geo["object_bbox_height_page_fraction"],
        "span_count": len(span_ids),
        "caption_span_count": len(caption_span_ids),
        # Content previews
        "caption_text_preview": _truncate_preview(obj["caption_text"], 160),
        "content_text_preview": _truncate_preview(obj["content_text"], 220),
        "retrieval_text_preview": _truncate_preview(obj["retrieval_text"], 220),
        # Heuristic flags (diagnostic only — do not affect pipeline)
        "suspicious_false_table_identity": 1 if flag_false_table else 0,
        "suspicious_mixed_visual_bbox": 1 if flag_mixed_bbox else 0,
        "likely_caption_only": 1 if flag_caption_only else 0,
    }


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    # Identity
    "paper_id", "object_id", "object_type", "label",
    "page", "page_start", "page_end", "section", "section_number",
    # Source
    "source_used", "bbox_source", "object_bbox_confidence", "highlight_mode",
    "docling_match_mode", "docling_quality_pass", "docling_rejected_reason",
    # Table validation
    "table_region_validation", "table_fallback_rejected_reason",
    # Geometry
    "has_object_bbox", "has_caption_bbox",
    "object_bbox_x0", "object_bbox_y0", "object_bbox_x1", "object_bbox_y1",
    "object_bbox_width", "object_bbox_height",
    "page_width", "page_height", "object_bbox_height_page_fraction",
    "span_count", "caption_span_count",
    # Content
    "caption_text_preview", "content_text_preview", "retrieval_text_preview",
    # Flags
    "suspicious_false_table_identity", "suspicious_mixed_visual_bbox", "likely_caption_only",
]


def _write_object_audit_csv(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _write_object_audit_md(
    records: list[dict[str, Any]],
    paper_id: str,
    path: Path,
    *,
    generated_at: str | None = None,
    csv_path: Path | None = None,
) -> None:
    ts = generated_at or datetime.now().isoformat(timespec="seconds")
    lines: list[str] = []

    figs = [r for r in records if r["object_type"] == "figure"]
    tbls = [r for r in records if r["object_type"] == "table"]
    n_false_tbl = sum(r["suspicious_false_table_identity"] for r in records)
    n_mixed_bbox = sum(r["suspicious_mixed_visual_bbox"] for r in records)
    n_cap_only = sum(r["likely_caption_only"] for r in records)

    lines.append(f"# Object Audit — {paper_id}")
    lines.append("")
    lines.append(f"**Generated:** {ts}")
    if csv_path:
        lines.append(f"**CSV:** `{csv_path}`")
    lines.append("")

    if not records:
        lines.append("_No figure/table evidence objects found for this paper._")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    # ── Overall counts ──────────────────────────────────────────────────────
    lines.append("## Overall Counts")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| figure objects | {len(figs)} |")
    lines.append(f"| table objects | {len(tbls)} |")
    lines.append(f"| suspicious_false_table_identity | {n_false_tbl} |")
    lines.append(f"| suspicious_mixed_visual_bbox | {n_mixed_bbox} |")
    lines.append(f"| likely_caption_only | {n_cap_only} |")
    lines.append("")

    # ── bbox_source distribution ────────────────────────────────────────────
    src_counts = Counter(r["bbox_source"] for r in records)
    lines.append("## BBox Source Distribution")
    lines.append("")
    lines.append("| bbox_source | count |")
    lines.append("|---|---:|")
    for src, cnt in src_counts.most_common():
        lines.append(f"| `{src}` | {cnt} |")
    lines.append("")

    # ── object_bbox_confidence distribution ────────────────────────────────
    conf_counts = Counter(r["object_bbox_confidence"] for r in records)
    lines.append("## BBox Confidence Distribution")
    lines.append("")
    lines.append("| object_bbox_confidence | count |")
    lines.append("|---|---:|")
    for conf, cnt in conf_counts.most_common():
        lines.append(f"| `{conf or '(empty)'}` | {cnt} |")
    lines.append("")

    # ── Top suspicious tables ───────────────────────────────────────────────
    false_tbls = [r for r in records if r["suspicious_false_table_identity"]]
    lines.append("## Suspicious False Table Identity")
    lines.append("")
    if not false_tbls:
        lines.append("_No suspicious false-table objects detected._")
    else:
        lines.append(
            f"{len(false_tbls)} table object(s) flagged as potentially misidentified:"
        )
        lines.append("")
        top = false_tbls[:20]
        lines.append("| object_id | page | label | bbox_source | bbox_conf | tbl_validation | rejection_reason | caption_preview | content_preview |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in top:
            lines.append(
                f"| {r['object_id']} "
                f"| {r['page']} "
                f"| {r['label'] or '—'} "
                f"| `{r['bbox_source']}` "
                f"| `{r['object_bbox_confidence'] or '—'}` "
                f"| {r['table_region_validation'] or '—'} "
                f"| {r['table_fallback_rejected_reason'] or '—'} "
                f"| {r['caption_text_preview'][:80]} "
                f"| {r['content_text_preview'][:100]} |"
            )
    lines.append("")

    # ── Top suspicious mixed bboxes ─────────────────────────────────────────
    mixed = [r for r in records if r["suspicious_mixed_visual_bbox"]]
    lines.append("## Suspicious Mixed Visual BBox")
    lines.append("")
    if not mixed:
        lines.append("_No suspicious mixed-bbox objects detected._")
    else:
        lines.append(
            f"{len(mixed)} object(s) flagged as potentially having oversized/mixed bboxes:"
        )
        lines.append("")
        top = mixed[:20]
        lines.append("| object_id | object_type | page | label | bbox_source | height_fraction | span_count | caption_preview | content_preview |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in top:
            hf = r["object_bbox_height_page_fraction"]
            hf_str = f"{float(hf):.3f}" if hf != "" else "—"
            lines.append(
                f"| {r['object_id']} "
                f"| {r['object_type']} "
                f"| {r['page']} "
                f"| {r['label'] or '—'} "
                f"| `{r['bbox_source']}` "
                f"| {hf_str} "
                f"| {r['span_count']} "
                f"| {r['caption_text_preview'][:80]} "
                f"| {r['content_text_preview'][:100]} |"
            )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _paper_id_safe(paper_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", paper_id)


def export_object_audit(
    *,
    db_path: str | Path,
    paper_id: str,
    out_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Run object audit and write object_audit_<paper_id_safe>.csv + .md.

    Args:
        db_path:   Path to the SQLite database.
        paper_id:  Paper to audit (required).
        out_dir:   Output directory. Defaults to config.REPORT_DIR.

    Returns:
        (csv_path, md_path) — absolute paths of written files.
    """
    db_path = Path(db_path).resolve()
    out_dir_path = Path(out_dir).resolve() if out_dir else REPORT_DIR
    out_dir_path.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        page_heights = _load_page_heights(conn, paper_id)
        objects = _load_objects(conn, paper_id)
    finally:
        conn.close()

    records = [_build_object_audit_record(obj, page_heights) for obj in objects]

    generated_at = datetime.now().isoformat(timespec="seconds")
    safe = _paper_id_safe(paper_id)

    csv_path = out_dir_path / f"object_audit_{safe}.csv"
    md_path = out_dir_path / f"object_audit_{safe}.md"

    _write_object_audit_csv(records, csv_path)
    _write_object_audit_md(
        records, paper_id, md_path,
        generated_at=generated_at,
        csv_path=csv_path,
    )

    return csv_path, md_path
