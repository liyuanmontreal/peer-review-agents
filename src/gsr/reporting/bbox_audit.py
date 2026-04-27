"""BBox refinement audit report for figure/table evidence objects.

Reads existing evidence_objects rows for a paper and summarises:
  - how many figures/tables exist
  - how many were refined by a detector backend
  - what bbox_source each object ended up with
  - which objects still fell back to caption-only / anchor-only behaviour
  - optional per-object detector confidence / match metadata

Produces two artifacts:
  bbox_audit_<paper_id_safe>.md   — Markdown summary
  bbox_audit_<paper_id_safe>.csv  — Per-object detail table

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
# Constants
# ---------------------------------------------------------------------------

# bbox_source values that count as "refined by a detector"
_DETECTOR_SOURCES: frozenset[str] = frozenset({
    "paddlex_layout",
    "groundingdino",
    "rtdetr",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_json_loads(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def _safe_short_text(text: str | None, max_chars: int = 120) -> str:
    if not text:
        return ""
    t = str(text).replace("\n", " ").strip()
    return t[:max_chars]


def _normalize_bbox_source(meta: dict[str, Any]) -> str:
    """Infer a best-effort normalized bbox_source from metadata_json.

    Priority:
      1. Explicit 'bbox_source' key  → use directly
      2. 'detected_bbox' present and non-empty + detect which detector prefix
         is present in metadata → infer from prefix
      3. 'object_bbox_confidence' = 'caption_only' → 'caption_only'
      4. 'highlight_mode' present → use as hint
      5. fallback → 'unknown'
    """
    # 1. Explicit field
    src = meta.get("bbox_source")
    if src:
        return str(src)

    # 2. Detected bbox present — infer from metadata prefix keys
    detected = meta.get("detected_bbox")
    if detected:
        for detector in ("paddlex_layout", "dino", "rtdetr"):
            if any(k.startswith(detector) for k in meta):
                if detector == "dino":
                    return "groundingdino"
                return detector
        return "detected_unknown"

    # 3. Caption / confidence signals
    conf = meta.get("object_bbox_confidence") or ""
    if conf in ("caption_only", "caption"):
        return "caption_only"
    if conf == "inferred":
        return "caption_anchor"

    # 4. highlight_mode
    hm = meta.get("highlight_mode") or ""
    if hm:
        return f"highlight_{hm}"

    return "unknown"


def _is_refined(meta: dict[str, Any], bbox_source: str) -> bool:
    """Return True if the object has a detector-backed refined bbox."""
    if bbox_source in _DETECTOR_SOURCES:
        return True
    # Also count if detected_bbox is present regardless of source field
    detected = meta.get("detected_bbox")
    if detected and isinstance(detected, (list, tuple)) and len(detected) == 4:
        return True
    # Explicit matched flags
    for detector in ("paddlex_layout", "dino", "rtdetr"):
        if meta.get(f"{detector}_matched") is True:
            return True
    return False


def _detector_name_from_meta(meta: dict[str, Any], bbox_source: str) -> str:
    """Best-effort detector name from metadata keys."""
    if bbox_source in _DETECTOR_SOURCES:
        return bbox_source
    for detector in ("paddlex_layout", "dino", "rtdetr"):
        if any(k.startswith(detector) for k in meta):
            if detector == "dino":
                return "groundingdino"
            return detector
    return ""


def _detector_score(meta: dict[str, Any], bbox_source: str) -> str:
    """Extract detector confidence score from metadata."""
    for key in (
        f"{bbox_source}_score",
        "dino_score",
        "rtdetr_score",
        "paddlex_layout_score",
    ):
        v = meta.get(key)
        if v is not None:
            try:
                return f"{float(v):.3f}"
            except (TypeError, ValueError):
                return str(v)
    return ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_objects(
    conn: sqlite3.Connection,
    paper_id: str,
    include_text: bool = False,
) -> list[dict[str, Any]]:
    """Load evidence objects from SQLite, expanding metadata_json inline."""
    type_filter = "" if include_text else "AND object_type IN ('figure', 'table')"
    rows = conn.execute(
        f"""
        SELECT id, paper_id, object_type, label,
               page, page_start, page_end, section,
               caption_text, bbox_json, metadata_json
        FROM evidence_objects
        WHERE paper_id = ?
          {type_filter}
        ORDER BY object_type, page, label
        """,
        (paper_id,),
    ).fetchall()

    result: list[dict[str, Any]] = []
    for (obj_id, pid, otype, label, page, page_start, page_end,
         section, caption_text, bbox_json, metadata_json_str) in rows:
        meta = _safe_json_loads(metadata_json_str)
        bbox_source = _normalize_bbox_source(meta)
        refined = _is_refined(meta, bbox_source)
        result.append({
            "object_id": obj_id,
            "paper_id": pid,
            "object_type": otype or "",
            "label": label or "",
            "page": page or page_start or 0,
            "page_start": page_start or 0,
            "page_end": page_end or 0,
            "section": section or "",
            "caption_text": caption_text or "",
            "bbox_json": bbox_json or "",
            "meta": meta,
            "bbox_source": bbox_source,
            "is_refined": refined,
        })
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_bbox_audit_stats(
    conn: sqlite3.Connection,
    paper_id: str,
    include_text: bool = False,
) -> dict[str, Any]:
    objects = _load_objects(conn, paper_id, include_text=include_text)

    total = len(objects)
    by_type: dict[str, dict[str, int]] = {}
    for obj in objects:
        ot = obj["object_type"]
        if ot not in by_type:
            by_type[ot] = {"total": 0, "refined": 0}
        by_type[ot]["total"] += 1
        if obj["is_refined"]:
            by_type[ot]["refined"] += 1

    total_refined = sum(1 for o in objects if o["is_refined"])
    source_dist: dict[str, int] = dict(
        Counter(o["bbox_source"] for o in objects).most_common()
    )

    # Detector score averages by source
    score_by_source: dict[str, list[float]] = {}
    for obj in objects:
        if not obj["is_refined"]:
            continue
        src = obj["bbox_source"]
        s = _detector_score(obj["meta"], src)
        if s:
            try:
                score_by_source.setdefault(src, []).append(float(s))
            except ValueError:
                pass
    avg_score_by_source = {
        src: round(sum(vals) / len(vals), 3)
        for src, vals in score_by_source.items()
    }

    not_refined = [o for o in objects if not o["is_refined"]]

    return {
        "paper_id": paper_id,
        "total": total,
        "total_refined": total_refined,
        "by_type": by_type,
        "source_dist": source_dist,
        "avg_score_by_source": avg_score_by_source,
        "not_refined": not_refined,
        "objects": objects,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_bbox_audit_md(
    stats: dict[str, Any],
    *,
    generated_at: str | None = None,
    csv_path: Path | None = None,
    md_path: Path | None = None,
) -> str:
    ts = generated_at or datetime.now().isoformat(timespec="seconds")
    paper_id = stats["paper_id"]
    total = stats["total"]
    total_refined = stats["total_refined"]
    by_type = stats["by_type"]
    source_dist = stats["source_dist"]
    avg_scores = stats["avg_score_by_source"]
    not_refined = stats["not_refined"]

    lines: list[str] = []
    lines.append(f"# BBox Audit — {paper_id}")
    lines.append("")
    lines.append(f"**Generated:** {ts}")
    lines.append("")

    if total == 0:
        lines.append("_No figure/table evidence objects found for this paper._")
        return "\n".join(lines)

    # --- Summary ---
    lines.append("## Summary")
    lines.append("")
    refine_pct = 100.0 * total_refined / total if total > 0 else 0.0
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Total objects (figure + table) | {total} |")
    for ot, counts in sorted(by_type.items()):
        lines.append(f"| Total {ot}s | {counts['total']} |")
    for ot, counts in sorted(by_type.items()):
        t = counts["total"]
        r = counts["refined"]
        pct = 100.0 * r / t if t > 0 else 0.0
        lines.append(f"| Refined {ot}s | {r} / {t} ({pct:.1f}%) |")
    lines.append(f"| Overall refined | {total_refined} / {total} ({refine_pct:.1f}%) |")
    lines.append("")

    # --- BBox source distribution ---
    lines.append("## BBox Source Distribution")
    lines.append("")
    lines.append("| bbox_source | count |")
    lines.append("|---|---:|")
    for src, cnt in source_dist.items():
        lines.append(f"| `{src}` | {cnt} |")
    lines.append("")

    # --- By object type ---
    lines.append("## By Object Type")
    lines.append("")
    lines.append("| object_type | total | refined | refine_rate |")
    lines.append("|---|---:|---:|---:|")
    for ot, counts in sorted(by_type.items()):
        t = counts["total"]
        r = counts["refined"]
        pct = f"{100.0 * r / t:.1f}%" if t > 0 else "N/A"
        lines.append(f"| {ot} | {t} | {r} | {pct} |")
    lines.append("")

    # --- Detector notes ---
    if avg_scores:
        lines.append("## Detector-Specific Notes")
        lines.append("")
        lines.append("| detector | avg_score | refined_objects |")
        lines.append("|---|---:|---:|")
        for src, avg in sorted(avg_scores.items()):
            cnt = source_dist.get(src, 0)
            lines.append(f"| `{src}` | {avg} | {cnt} |")
        lines.append("")

    # --- Not refined ---
    lines.append("## Not-Refined Objects")
    lines.append("")
    if not not_refined:
        lines.append("_All objects were refined by a detector backend._")
    else:
        lines.append(
            f"{len(not_refined)} object(s) were NOT refined "
            "(bbox_source is not a detector backend):"
        )
        lines.append("")
        lines.append("| label | object_type | page | bbox_source | caption_preview |")
        lines.append("|---|---|---|---|---|")
        for obj in not_refined:
            cap = _safe_short_text(obj["caption_text"], 80)
            lines.append(
                f"| {obj['label'] or '—'} | {obj['object_type']} "
                f"| {obj['page']} | `{obj['bbox_source']}` | {cap} |"
            )
    lines.append("")

    # --- Paths ---
    lines.append("## Output Files")
    lines.append("")
    if csv_path:
        lines.append(f"- **CSV:** `{csv_path}`")
    if md_path:
        lines.append(f"- **Markdown:** `{md_path}`")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "paper_id",
    "object_id",
    "object_type",
    "label",
    "page",
    "page_start",
    "page_end",
    "section",
    "bbox_source",
    "is_refined",
    "has_detected_bbox",
    "has_bbox_json",
    "highlight_mode",
    "object_bbox_confidence",
    "detector_name",
    "detector_score",
    "match_mode",
    "caption_preview",
    "bbox_json_preview",
    "detected_bbox_preview",
]


def _write_csv(objects: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for obj in objects:
            meta = obj["meta"]
            src = obj["bbox_source"]
            detected = meta.get("detected_bbox")
            writer.writerow({
                "paper_id": obj["paper_id"],
                "object_id": obj["object_id"],
                "object_type": obj["object_type"],
                "label": obj["label"],
                "page": obj["page"],
                "page_start": obj["page_start"],
                "page_end": obj["page_end"],
                "section": obj["section"],
                "bbox_source": src,
                "is_refined": 1 if obj["is_refined"] else 0,
                "has_detected_bbox": 1 if detected else 0,
                "has_bbox_json": 1 if obj["bbox_json"] else 0,
                "highlight_mode": meta.get("highlight_mode") or "",
                "object_bbox_confidence": meta.get("object_bbox_confidence") or "",
                "detector_name": _detector_name_from_meta(meta, src),
                "detector_score": _detector_score(meta, src),
                "match_mode": (
                    meta.get(f"{src}_match_mode")
                    or meta.get("dino_match_mode")
                    or ""
                ),
                "caption_preview": _safe_short_text(obj["caption_text"], 120),
                "bbox_json_preview": _safe_short_text(obj["bbox_json"], 80),
                "detected_bbox_preview": (
                    str([round(v, 1) for v in detected])
                    if detected and isinstance(detected, (list, tuple))
                    else ""
                ),
            })


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _paper_id_safe(paper_id: str) -> str:
    """Convert paper_id to a safe filename fragment."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", paper_id)


def export_bbox_audit(
    *,
    db_path: str | Path,
    paper_id: str,
    out_dir: str | Path | None = None,
    include_text: bool = False,
) -> tuple[Path, Path]:
    """Run bbox audit and write bbox_audit_<paper_id_safe>.csv + .md.

    Args:
        db_path:      Path to the SQLite database.
        paper_id:     Paper to audit (required).
        out_dir:      Output directory. Defaults to config.REPORT_DIR.
        include_text: If True, also include text_chunk objects (default False).

    Returns:
        (csv_path, md_path) — absolute paths of written files.
    """
    db_path = Path(db_path).resolve()
    out_dir_path = Path(out_dir).resolve() if out_dir else REPORT_DIR
    out_dir_path.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        stats = compute_bbox_audit_stats(conn, paper_id, include_text=include_text)
    finally:
        conn.close()

    generated_at = datetime.now().isoformat(timespec="seconds")
    safe = _paper_id_safe(paper_id)

    csv_path = out_dir_path / f"bbox_audit_{safe}.csv"
    md_path = out_dir_path / f"bbox_audit_{safe}.md"

    _write_csv(stats["objects"], csv_path)

    md_text = render_bbox_audit_md(
        stats,
        generated_at=generated_at,
        csv_path=csv_path,
        md_path=md_path,
    )
    md_path.write_text(md_text, encoding="utf-8")

    return csv_path, md_path
