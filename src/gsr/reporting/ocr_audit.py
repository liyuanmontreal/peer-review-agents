"""OCR audit report for figure evidence objects — Phase 1.2.

Reads evidence_objects.metadata_json from SQLite and produces:
- Aggregate stats (total, attempted, accepted, rejected_after_inference, never_attempted)
- Skip reason and bbox confidence distributions
- Per-bbox-confidence grouped summary
- Per-figure CSV (ocr_audit.csv)
- Markdown summary (ocr_audit.md)

Status buckets
--------------
accepted                  — OCR ran and text was accepted
rejected_after_inference  — OCR ran but result was rejected by quality filter
never_attempted           — OCR did not run (no bbox, caption_only, dependency missing, etc.)

These match the taxonomy in ocr_lighton.py.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from gsr.config import REPORT_DIR

# ---------------------------------------------------------------------------
# Status bucket constants
# ---------------------------------------------------------------------------

ACCEPTED = "accepted"
REJECTED_AFTER_INFERENCE = "rejected_after_inference"
NEVER_ATTEMPTED = "never_attempted"

# Skip reasons that indicate OCR was never attempted (figure_ocr_attempted=False)
_NEVER_ATTEMPTED_REASONS: frozenset[str] = frozenset({
    "caption_only",
    "no_bbox",
    "bbox_too_small",
    "dependency_missing",
    "render_failed",
    "disabled",  # use_figure_ocr=False at index time
    # legacy names (pre-1.2), kept for backward compat with older indexed data
    "pymupdf_not_installed",
    "lightonai_not_installed",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ocr_status(row: dict[str, Any]) -> str:
    """Classify a figure row into one of three status buckets."""
    attempted = row.get("figure_ocr_attempted", False)
    text = row.get("figure_ocr_text") or ""
    if not attempted:
        return NEVER_ATTEMPTED
    if text:
        return ACCEPTED
    return REJECTED_AFTER_INFERENCE


def _load_figure_rows(
    conn: sqlite3.Connection,
    paper_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load figure evidence objects from SQLite, expanding metadata_json."""
    if paper_id:
        rows = conn.execute(
            """
            SELECT paper_id, label, page, metadata_json
            FROM evidence_objects
            WHERE object_type = 'figure'
              AND paper_id = ?
            ORDER BY page, label
            """,
            (paper_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT paper_id, label, page, metadata_json
            FROM evidence_objects
            WHERE object_type = 'figure'
            ORDER BY paper_id, page, label
            """
        ).fetchall()

    result: list[dict[str, Any]] = []
    for pid, label, page, metadata_json_str in rows:
        try:
            meta = json.loads(metadata_json_str) if metadata_json_str else {}
        except Exception:
            meta = {}
        result.append({
            "paper_id": pid,
            "label": label or "",
            "page": page or 0,
            **meta,
        })
    return result


# ---------------------------------------------------------------------------
# Core stats computation
# ---------------------------------------------------------------------------

def compute_ocr_audit_stats(
    conn: sqlite3.Connection,
    paper_id: str | None = None,
) -> dict[str, Any]:
    """Compute aggregated OCR audit statistics.

    Args:
        conn:     Open SQLite connection.
        paper_id: If given, scope to one paper. Otherwise all papers.

    Returns:
        Dict with aggregate counts, distributions, grouped breakdown, and raw rows.
        Rows have a '_status' key added in-place.
    """
    rows = _load_figure_rows(conn, paper_id)
    total = len(rows)

    if total == 0:
        return {
            "total": 0,
            "attempted": 0,
            "accepted": 0,
            "rejected_after_inference": 0,
            "never_attempted": 0,
            "accepted_rate_among_attempted": None,
            "skip_reason_distribution": {},
            "bbox_confidence_distribution": {},
            "grouped_by_bbox_confidence": {},
            "rows": [],
        }

    for row in rows:
        row["_status"] = _ocr_status(row)

    counts = Counter(row["_status"] for row in rows)
    accepted = counts[ACCEPTED]
    rejected = counts[REJECTED_AFTER_INFERENCE]
    never = counts[NEVER_ATTEMPTED]
    attempted = accepted + rejected
    accepted_rate = (accepted / attempted) if attempted > 0 else None

    skip_dist: dict[str, int] = dict(
        Counter(
            (row.get("figure_ocr_skip_reason") or "none")
            for row in rows
        ).most_common()
    )

    bbox_dist: dict[str, int] = dict(
        Counter(
            (row.get("object_bbox_confidence") or "unknown")
            for row in rows
        ).most_common()
    )

    # Grouped by object_bbox_confidence
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {
        "total": 0,
        ACCEPTED: 0,
        REJECTED_AFTER_INFERENCE: 0,
        NEVER_ATTEMPTED: 0,
        "attempted": 0,
    })
    for row in rows:
        conf = row.get("object_bbox_confidence") or "unknown"
        status = row["_status"]
        grouped[conf]["total"] += 1
        grouped[conf][status] += 1
        if status in (ACCEPTED, REJECTED_AFTER_INFERENCE):
            grouped[conf]["attempted"] += 1

    # Error type distribution (inference_failed rows only)
    inference_failed_rows = [
        r for r in rows
        if r.get("figure_ocr_skip_reason") == "inference_failed"
    ]
    error_type_dist: dict[str, int] = dict(
        Counter(
            r.get("figure_ocr_error_type") or "unknown"
            for r in inference_failed_rows
        ).most_common()
    )
    # Up to 5 unique sample messages (truncated at 150 chars each)
    seen_msgs: set[str] = set()
    error_message_samples: list[str] = []
    for r in inference_failed_rows:
        msg = (r.get("figure_ocr_error_message") or "").strip()[:150]
        if msg and msg not in seen_msgs:
            seen_msgs.add(msg)
            error_message_samples.append(msg)
        if len(error_message_samples) >= 5:
            break

    # --- Latency statistics (attempted figures only, requires figure_ocr_elapsed_s) ---
    elapsed_rows = [
        r for r in rows
        if r.get("figure_ocr_attempted") and r.get("figure_ocr_elapsed_s") is not None
    ]
    latency_stats: dict[str, Any] | None = None
    if elapsed_rows:
        elapsed_vals = sorted(float(r["figure_ocr_elapsed_s"]) for r in elapsed_rows)
        n_timed = len(elapsed_vals)
        total_el = sum(elapsed_vals)
        mean_el = total_el / n_timed
        median_el = elapsed_vals[n_timed // 2]
        p95_el = elapsed_vals[max(0, int(0.95 * n_timed) - 1)]

        # Slowest 5: sort attempted rows by elapsed desc
        slowest = sorted(elapsed_rows, key=lambda r: float(r.get("figure_ocr_elapsed_s", 0)), reverse=True)[:5]
        slowest_5 = [
            {
                "label": r.get("label", ""),
                "page": r.get("page", ""),
                "elapsed_s": round(float(r["figure_ocr_elapsed_s"]), 1),
                "outcome": r.get("figure_ocr_outcome") or r.get("_status", ""),
            }
            for r in slowest
        ]

        latency_stats = {
            "n_timed": n_timed,
            "total_elapsed_s": round(total_el, 1),
            "mean_elapsed_s": round(mean_el, 1),
            "median_elapsed_s": round(median_el, 1),
            "p95_elapsed_s": round(p95_el, 1),
            "slowest_5": slowest_5,
        }

    return {
        "total": total,
        "attempted": attempted,
        "accepted": accepted,
        "rejected_after_inference": rejected,
        "never_attempted": never,
        "accepted_rate_among_attempted": accepted_rate,
        "skip_reason_distribution": skip_dist,
        "bbox_confidence_distribution": bbox_dist,
        "grouped_by_bbox_confidence": dict(grouped),
        "inference_failed_count": len(inference_failed_rows),
        "error_type_distribution": error_type_dist,
        "error_message_samples": error_message_samples,
        "latency_stats": latency_stats,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Interpretation bullets (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _generate_interpretation(stats: dict[str, Any]) -> list[str]:
    """Generate short deterministic interpretation bullets from computed stats."""
    bullets: list[str] = []
    total = stats["total"]
    attempted = stats["attempted"]
    accepted = stats["accepted"]
    never = stats["never_attempted"]
    rejected = stats["rejected_after_inference"]
    rate = stats["accepted_rate_among_attempted"]

    if total == 0:
        return ["No figure evidence objects found."]

    # 1. Never-attempted coverage
    never_pct = 100 * never / total
    bullets.append(
        f"{never_pct:.0f}% of figures ({never}/{total}) were never attempted "
        "— OCR did not run, typically due to caption_only bbox or missing dependencies."
    )

    # 2. Acceptance rate among attempted
    if attempted == 0:
        bullets.append("No OCR inference was attempted for any figure in this set.")
    elif rate is not None:
        bullets.append(
            f"Of {attempted} attempted figures, {accepted} were accepted "
            f"({100 * rate:.0f}% acceptance rate)."
        )

    # 3. Caption_only dominance
    skip_dist = stats["skip_reason_distribution"]
    caption_only_count = skip_dist.get("caption_only", 0)
    if total > 0 and caption_only_count / total > 0.35:
        bullets.append(
            f"caption_only is the leading skip reason ({caption_only_count}/{total} figures) — "
            "these figures have no reliable crop region because only a caption span was found."
        )

    # 4. Rejected-after-inference detail
    if rejected > 0 and attempted > 0:
        rej_pct = 100 * rejected / attempted
        # find top rejection reason (exclude never-attempted and "none")
        top_rej = next(
            (r for r in skip_dist if r not in _NEVER_ATTEMPTED_REASONS and r != "none"),
            None,
        )
        note = f" (most common: {top_rej})" if top_rej else ""
        bullets.append(
            f"{rejected} figures passed inference but were rejected by quality filter "
            f"({rej_pct:.0f}% of attempted){note}."
        )

    # 5. High-confidence bbox outcome
    grouped = stats["grouped_by_bbox_confidence"]
    high = grouped.get("high", {})
    if high.get("total", 0) > 0:
        h_total = high["total"]
        h_accepted = high.get(ACCEPTED, 0)
        h_never = high.get(NEVER_ATTEMPTED, 0)
        bullets.append(
            f"Figures with 'high' bbox confidence: {h_accepted}/{h_total} accepted, "
            f"{h_never} never attempted — "
            + ("high-confidence bboxes show better OCR coverage." if h_accepted > h_never else
               "even high-confidence bboxes have significant never-attempted figures.")
        )

    return bullets


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_ocr_audit_md(
    stats: dict[str, Any],
    *,
    paper_id: str | None = None,
    generated_at: str | None = None,
) -> str:
    """Render a Markdown OCR audit report from precomputed stats."""
    ts = generated_at or datetime.now().isoformat(timespec="seconds")
    lines: list[str] = []

    lines.append("# Figure OCR Audit Report (Phase 1)")
    lines.append("")
    if paper_id:
        lines.append(f"**Paper:** `{paper_id}`  ")
    lines.append(f"**Generated:** {ts}  ")
    lines.append("**Source:** Phase 1 LightOnOCR-2 figure text recovery")
    lines.append("")

    total = stats["total"]
    if total == 0:
        lines.append("_No figure evidence objects found._")
        return "\n".join(lines)

    attempted = stats["attempted"]
    accepted = stats["accepted"]
    rejected = stats["rejected_after_inference"]
    never = stats["never_attempted"]
    rate = stats["accepted_rate_among_attempted"]
    rate_str = f"{100 * rate:.1f}%" if rate is not None else "N/A"

    # --- Aggregate summary ---
    lines.append("## Aggregate Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---|")
    lines.append(f"| Total figures | {total} |")
    lines.append(f"| OCR attempted | {attempted} |")
    lines.append(f"| Accepted | {accepted} |")
    lines.append(f"| Rejected after inference | {rejected} |")
    lines.append(f"| Never attempted | {never} |")
    lines.append(f"| Acceptance rate (among attempted) | {rate_str} |")
    lines.append("")

    # --- Skip reason distribution ---
    lines.append("## Skip Reason Distribution")
    lines.append("")
    skip_dist = stats["skip_reason_distribution"]
    if skip_dist:
        lines.append("| Skip Reason | Count |")
        lines.append("|---|---|")
        for reason, count in skip_dist.items():
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("_No skip reasons recorded._")
    lines.append("")

    # --- BBox confidence distribution ---
    lines.append("## BBox Confidence Distribution")
    lines.append("")
    bbox_dist = stats["bbox_confidence_distribution"]
    if bbox_dist:
        lines.append("| BBox Confidence | Count |")
        lines.append("|---|---|")
        for conf, count in bbox_dist.items():
            lines.append(f"| `{conf}` | {count} |")
    else:
        lines.append("_No bbox confidence data recorded._")
    lines.append("")

    # --- Grouped by bbox_confidence ---
    lines.append("## OCR Outcomes by BBox Confidence")
    lines.append("")
    grouped = stats["grouped_by_bbox_confidence"]
    if grouped:
        lines.append("| BBox Confidence | Total | Attempted | Accepted | Rejected | Never Attempted |")
        lines.append("|---|---|---|---|---|---|")
        # canonical order first, then any extras
        canonical_order = ["high", "inferred", "caption_only", "unknown"]
        seen: set[str] = set()
        for conf in canonical_order + [k for k in grouped if k not in canonical_order]:
            if conf in seen or conf not in grouped:
                continue
            seen.add(conf)
            g = grouped[conf]
            lines.append(
                f"| `{conf}` | {g['total']} | {g['attempted']} "
                f"| {g.get(ACCEPTED, 0)} | {g.get(REJECTED_AFTER_INFERENCE, 0)} "
                f"| {g.get(NEVER_ATTEMPTED, 0)} |"
            )
    else:
        lines.append("_No grouped data available._")
    lines.append("")

    # --- Inference failure diagnostics ---
    inference_failed = stats.get("inference_failed_count", 0)
    if inference_failed > 0:
        lines.append("## Inference Failure Diagnostics")
        lines.append("")
        lines.append(f"**Total inference_failed:** {inference_failed}")
        lines.append("")

        error_dist = stats.get("error_type_distribution", {})
        if error_dist:
            lines.append("### Top OCR Error Types")
            lines.append("")
            lines.append("| Error Type | Count |")
            lines.append("|---|---|")
            for etype, count in error_dist.items():
                lines.append(f"| `{etype}` | {count} |")
            lines.append("")

        samples = stats.get("error_message_samples", [])
        if samples:
            lines.append("### Sample Error Messages")
            lines.append("")
            for msg in samples:
                lines.append(f"- `{msg}`")
            lines.append("")

    # --- Latency statistics ---
    latency = stats.get("latency_stats")
    if latency:
        lines.append("## Latency Statistics (Attempted Figures)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Figures timed | {latency['n_timed']} |")
        lines.append(f"| Total elapsed | {latency['total_elapsed_s']}s |")
        lines.append(f"| Mean per figure | {latency['mean_elapsed_s']}s |")
        lines.append(f"| Median per figure | {latency['median_elapsed_s']}s |")
        lines.append(f"| p95 per figure | {latency['p95_elapsed_s']}s |")
        lines.append("")
        slowest = latency.get("slowest_5", [])
        if slowest:
            lines.append("### Slowest 5 Figures")
            lines.append("")
            lines.append("| Label | Page | Elapsed | Outcome |")
            lines.append("|---|---|---|---|")
            for s in slowest:
                lines.append(
                    f"| `{s['label']}` | {s['page']} | {s['elapsed_s']}s | {s['outcome']} |"
                )
            lines.append("")
    else:
        lines.append("## Latency Statistics")
        lines.append("")
        lines.append(
            "_No timing data available. Re-run `gsr retrieve --figure-ocr --force` "
            "to populate `figure_ocr_elapsed_s` metadata._"
        )
        lines.append("")

    # --- Interpretation ---
    lines.append("## Interpretation Notes")
    lines.append("")
    for bullet in _generate_interpretation(stats):
        lines.append(f"- {bullet}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write per-figure audit rows to CSV."""
    fieldnames = [
        "paper_id",
        "label",
        "page",
        "object_bbox_confidence",
        "figure_ocr_attempted",
        "figure_ocr_outcome",
        "figure_ocr_status",
        "figure_ocr_skip_reason",
        "figure_ocr_quality",
        "figure_ocr_elapsed_s",
        "figure_ocr_max_new_tokens",
        "figure_ocr_image_size",
        "figure_ocr_resized_size",
        "figure_ocr_text_preview",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            text = row.get("figure_ocr_text") or ""
            preview = text[:120].replace("\n", " ") if text else ""
            img_size = row.get("figure_ocr_image_size")
            rsz_size = row.get("figure_ocr_resized_size")
            writer.writerow({
                "paper_id": row.get("paper_id", ""),
                "label": row.get("label", ""),
                "page": row.get("page", ""),
                "object_bbox_confidence": row.get("object_bbox_confidence") or "",
                "figure_ocr_attempted": row.get("figure_ocr_attempted", False),
                "figure_ocr_outcome": row.get("figure_ocr_outcome") or "",
                "figure_ocr_status": row.get("_status", ""),
                "figure_ocr_skip_reason": row.get("figure_ocr_skip_reason") or "",
                "figure_ocr_quality": row.get("figure_ocr_quality") if row.get("figure_ocr_quality") is not None else "",
                "figure_ocr_elapsed_s": row.get("figure_ocr_elapsed_s") if row.get("figure_ocr_elapsed_s") is not None else "",
                "figure_ocr_max_new_tokens": row.get("figure_ocr_max_new_tokens") if row.get("figure_ocr_max_new_tokens") is not None else "",
                "figure_ocr_image_size": f"{img_size[0]}x{img_size[1]}" if img_size else "",
                "figure_ocr_resized_size": f"{rsz_size[0]}x{rsz_size[1]}" if rsz_size else "",
                "figure_ocr_text_preview": preview,
            })


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_ocr_audit(
    *,
    db_path: str | Path,
    paper_id: str | None = None,
    out_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Run OCR audit and write ocr_audit.csv + ocr_audit.md.

    Args:
        db_path:  Path to the SQLite database.
        paper_id: Scope to one paper, or None for all papers.
        out_dir:  Output directory. Defaults to config.REPORT_DIR.

    Returns:
        (csv_path, md_path) — absolute paths of written files.
    """
    db_path = Path(db_path).resolve()
    out_dir_path = Path(out_dir).resolve() if out_dir else REPORT_DIR
    out_dir_path.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        stats = compute_ocr_audit_stats(conn, paper_id=paper_id)
    finally:
        conn.close()

    generated_at = datetime.now().isoformat(timespec="seconds")

    csv_path = out_dir_path / "ocr_audit.csv"
    _write_csv(stats["rows"], csv_path)

    md_path = out_dir_path / "ocr_audit.md"
    md_text = render_ocr_audit_md(stats, paper_id=paper_id, generated_at=generated_at)
    md_path.write_text(md_text, encoding="utf-8")

    return csv_path, md_path
