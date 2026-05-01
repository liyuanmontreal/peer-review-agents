"""Read-only dataset health audit for the Koala/GSR competition database.

Produces a Markdown report, a JSON report, and a console summary.
Never writes to or mutates the DB.

Usage:
    python scripts/dataset_audit.py --db ./workspace/koala_agent.db
    python scripts/dataset_audit.py --db PATH --out-md PATH --out-json PATH
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema-safe helpers
# ---------------------------------------------------------------------------

def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}
    except Exception:
        return set()


def _count(conn: sqlite3.Connection, table: str) -> int:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> Any:
    try:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[tuple]:
    try:
        return conn.execute(sql, params).fetchall()
    except Exception:
        return []


def _has_col(cols: set[str], name: str) -> bool:
    return name in cols


# ---------------------------------------------------------------------------
# Section collectors
# ---------------------------------------------------------------------------

_KEY_TABLES = [
    "koala_papers",
    "koala_comments",
    "koala_extracted_claims",
    "koala_claim_verifications",
    "koala_reactive_drafts",
    "koala_agent_actions",
    "koala_karma_ledger",
    "koala_verdict_state",
    # tables referenced in spec but not yet in schema — reported as missing
    "decision_states",
    "verdict_drafts",
    "evidence_results",
    "novelty_assessments",
    "severe_issue_reports",
]


def _section_table_inventory(conn: sqlite3.Connection) -> dict[str, Any]:
    all_tables = _tables(conn)
    inventory: dict[str, int | None] = {}
    for t in sorted(all_tables):
        inventory[t] = _count(conn, t)
    missing = [t for t in _KEY_TABLES if t not in all_tables]
    return {
        "all_tables": inventory,
        "key_tables_present": [t for t in _KEY_TABLES if t in all_tables],
        "key_tables_missing": missing,
    }


def _section_paper_coverage(conn: sqlite3.Connection, present: set[str]) -> dict[str, Any]:
    total = _count(conn, "koala_papers") if "koala_papers" in present else 0

    def _count_papers_with(table: str, join_col: str = "paper_id") -> int | None:
        if table not in present or "koala_papers" not in present:
            return None
        val = _scalar(
            conn,
            f"SELECT COUNT(DISTINCT {join_col}) FROM {table}",
        )
        return int(val) if val is not None else 0

    papers_with_comments = _count_papers_with("koala_comments")
    papers_with_claims = _count_papers_with("koala_extracted_claims")
    papers_with_verifications = _count_papers_with("koala_claim_verifications")
    papers_with_drafts = _count_papers_with("koala_reactive_drafts")

    papers_with_verdict_state: int | None = None
    if "koala_verdict_state" in present:
        vs_cols = _columns(conn, "koala_verdict_state")
        if "has_our_participation" in vs_cols:
            val = _scalar(
                conn,
                "SELECT COUNT(*) FROM koala_verdict_state WHERE has_our_participation=1",
            )
            papers_with_verdict_state = int(val) if val is not None else 0

    papers_with_live_activity: int | None = None
    if "koala_agent_actions" in present:
        val = _scalar(
            conn,
            "SELECT COUNT(DISTINCT paper_id) FROM koala_agent_actions "
            "WHERE status='success' AND action_type IN ('reactive_comment','seed_comment','verdict_submission')",
        )
        papers_with_live_activity = int(val) if val is not None else 0

    return {
        "total_papers": total,
        "papers_with_comments": papers_with_comments,
        "papers_with_extracted_claims": papers_with_claims,
        "papers_with_claim_verifications": papers_with_verifications,
        "papers_with_reactive_drafts": papers_with_drafts,
        "papers_with_our_participation": papers_with_verdict_state,
        "papers_with_live_activity": papers_with_live_activity,
    }


def _section_comment_coverage(conn: sqlite3.Connection, present: set[str]) -> dict[str, Any]:
    if "koala_comments" not in present:
        return {"available": False}
    cols = _columns(conn, "koala_comments")
    total = _count(conn, "koala_comments")
    unique_papers = _scalar(conn, "SELECT COUNT(DISTINCT paper_id) FROM koala_comments") or 0

    unique_authors: int | None = None
    if _has_col(cols, "author_agent_id"):
        unique_authors = int(
            _scalar(conn, "SELECT COUNT(DISTINCT author_agent_id) FROM koala_comments WHERE author_agent_id!=''") or 0
        )

    top_level: int | None = None
    replies: int | None = None
    if _has_col(cols, "parent_id"):
        top_level = int(_scalar(conn, "SELECT COUNT(*) FROM koala_comments WHERE parent_id IS NULL") or 0)
        replies = int(_scalar(conn, "SELECT COUNT(*) FROM koala_comments WHERE parent_id IS NOT NULL") or 0)

    citable: int | None = None
    if _has_col(cols, "is_citable"):
        citable = int(_scalar(conn, "SELECT COUNT(*) FROM koala_comments WHERE is_citable=1") or 0)

    ours: int | None = None
    if _has_col(cols, "is_ours"):
        ours = int(_scalar(conn, "SELECT COUNT(*) FROM koala_comments WHERE is_ours=1") or 0)

    return {
        "available": True,
        "total_comments": total,
        "unique_papers_with_comments": int(unique_papers),
        "unique_authors": unique_authors,
        "top_level_comments": top_level,
        "reply_comments": replies,
        "citable_comments": citable,
        "our_comments": ours,
    }


def _normalize_verdict(raw: str) -> str:
    v = (raw or "").strip().lower()
    if v in ("refuted", "contradicted", "contradiction"):
        return "refuted"
    if v == "supported":
        return "supported"
    if v == "insufficient_evidence":
        return "insufficient_evidence"
    if v == "not_verifiable":
        return "not_verifiable"
    return "unknown"


def _section_claim_coverage(conn: sqlite3.Connection, present: set[str]) -> dict[str, Any]:
    if "koala_extracted_claims" not in present:
        return {"available": False}
    cols = _columns(conn, "koala_extracted_claims")
    total = _count(conn, "koala_extracted_claims")
    unique_papers = int(
        _scalar(conn, "SELECT COUNT(DISTINCT paper_id) FROM koala_extracted_claims") or 0
    )
    unique_source_comments: int | None = None
    if _has_col(cols, "comment_id"):
        unique_source_comments = int(
            _scalar(conn, "SELECT COUNT(DISTINCT comment_id) FROM koala_extracted_claims") or 0
        )

    avg_per_paper: float | None = None
    if unique_papers > 0:
        avg_per_paper = round(total / unique_papers, 2)

    category_dist: dict[str, int] | None = None
    if _has_col(cols, "category"):
        rows = _rows(conn, "SELECT category, COUNT(*) FROM koala_extracted_claims GROUP BY category ORDER BY 2 DESC")
        category_dist = {(r[0] or "null"): int(r[1]) for r in rows}

    return {
        "available": True,
        "total_extracted_claims": total,
        "unique_papers_with_claims": unique_papers,
        "unique_source_comments": unique_source_comments,
        "avg_claims_per_paper": avg_per_paper,
        "category_distribution": category_dist,
    }


def _section_verification_distribution(conn: sqlite3.Connection, present: set[str]) -> dict[str, Any]:
    if "koala_claim_verifications" not in present:
        return {"available": False}
    cols = _columns(conn, "koala_claim_verifications")
    total = _count(conn, "koala_claim_verifications")
    if not _has_col(cols, "verdict"):
        return {"available": True, "total_verified": total, "verdict_column_missing": True}

    raw_rows = _rows(conn, "SELECT verdict, COUNT(*) FROM koala_claim_verifications GROUP BY verdict")
    normalized: dict[str, int] = {}
    for raw_verdict, cnt in raw_rows:
        label = _normalize_verdict(raw_verdict)
        normalized[label] = normalized.get(label, 0) + int(cnt)

    refuted_rate = round(normalized.get("refuted", 0) / total, 3) if total else 0.0
    insufficient_rate = round(normalized.get("insufficient_evidence", 0) / total, 3) if total else 0.0
    distinct_labels = len([k for k, v in normalized.items() if v > 0])

    return {
        "available": True,
        "total_verified": total,
        "normalized_label_counts": normalized,
        "refuted_rate": refuted_rate,
        "insufficient_rate": insufficient_rate,
        "distinct_normalized_labels": distinct_labels,
        "raw_verdict_distribution": {(r[0] or "null"): int(r[1]) for r in raw_rows},
    }


def _extract_json_field(conn: sqlite3.Connection, table: str, col: str, json_key: str) -> list[tuple[str, int]]:
    """Group-count a JSON string field value from a table column."""
    try:
        rows = conn.execute(
            f"SELECT json_extract({col}, '$.{json_key}'), COUNT(*) "
            f"FROM {table} GROUP BY json_extract({col}, '$.{json_key}')"
        ).fetchall()
        return [(str(r[0]) if r[0] is not None else "null", int(r[1])) for r in rows]
    except Exception:
        return []


def _section_action_telemetry(conn: sqlite3.Connection, present: set[str]) -> dict[str, Any]:
    if "koala_agent_actions" not in present:
        return {"available": False}
    cols = _columns(conn, "koala_agent_actions")
    total = _count(conn, "koala_agent_actions")

    type_status_matrix: dict[str, dict[str, int]] = {}
    rows = _rows(conn, "SELECT action_type, status, COUNT(*) FROM koala_agent_actions GROUP BY action_type, status")
    for action_type, status, cnt in rows:
        at = action_type or "null"
        s = status or "null"
        type_status_matrix.setdefault(at, {})[s] = int(cnt)

    live_comments = int(
        _scalar(conn,
            "SELECT COUNT(*) FROM koala_agent_actions "
            "WHERE status='success' AND action_type IN ('reactive_comment','seed_comment')") or 0
    )
    live_verdicts = int(
        _scalar(conn,
            "SELECT COUNT(*) FROM koala_agent_actions "
            "WHERE status='success' AND action_type='verdict_submission'") or 0
    )

    reason_dist: dict[str, int] | None = None
    if _has_col(cols, "details"):
        skip_reason_rows = _extract_json_field(conn, "koala_agent_actions", "details", "skip_reason")
        blocked_reason_rows = _extract_json_field(conn, "koala_agent_actions", "details", "blocked_reason")
        all_reasons: dict[str, int] = {}
        for val, cnt in skip_reason_rows + blocked_reason_rows:
            if val != "null":
                all_reasons[val] = all_reasons.get(val, 0) + cnt
        if all_reasons:
            reason_dist = all_reasons

        target_reasons = [
            "window_skip", "dedup", "live_disabled", "allowlist_required",
            "budget_exhausted", "gate_failed",
        ]
        targeted: dict[str, int] = {}
        for reason in target_reasons:
            val = _scalar(
                conn,
                "SELECT COUNT(*) FROM koala_agent_actions "
                f"WHERE json_extract(details, '$.skip_reason') = ? "
                f"OR json_extract(details, '$.blocked_reason') LIKE ?",
                (reason, f"%{reason}%"),
            )
            targeted[reason] = int(val or 0)

    return {
        "available": True,
        "total_actions": total,
        "action_type_status_matrix": type_status_matrix,
        "live_comment_count": live_comments,
        "live_verdict_count": live_verdicts,
        "reason_distribution": reason_dist,
        "targeted_reason_counts": targeted if _has_col(cols, "details") else None,
    }


def _section_eligibility_telemetry(conn: sqlite3.Connection, present: set[str]) -> dict[str, Any]:
    if "koala_verdict_state" not in present:
        return {"available": False}
    cols = _columns(conn, "koala_verdict_state")
    total = _count(conn, "koala_verdict_state")

    state_dist: dict[str, int] | None = None
    if _has_col(cols, "eligibility_state"):
        rows = _rows(conn, "SELECT eligibility_state, COUNT(*) FROM koala_verdict_state GROUP BY eligibility_state")
        state_dist = {(r[0] or "null"): int(r[1]) for r in rows}

    submitted_count: int | None = None
    if _has_col(cols, "submitted"):
        submitted_count = int(_scalar(conn, "SELECT COUNT(*) FROM koala_verdict_state WHERE submitted=1") or 0)

    eligible_count: int | None = None
    if _has_col(cols, "eligibility_state"):
        eligible_count = int(
            _scalar(conn,
                "SELECT COUNT(*) FROM koala_verdict_state "
                "WHERE eligibility_state IN ('ELIGIBLE_READY','ELIGIBLE_LOW_CONFIDENCE','SUBMITTED')") or 0
        )

    citable_dist: dict[str, int] | None = None
    if _has_col(cols, "distinct_citable_other_agents"):
        rows = _rows(conn,
            "SELECT distinct_citable_other_agents, COUNT(*) FROM koala_verdict_state "
            "GROUP BY distinct_citable_other_agents ORDER BY 1")
        citable_dist = {str(r[0]): int(r[1]) for r in rows}

    heat_band_dist: dict[str, int] | None = None
    if "koala_agent_actions" in present and _has_col(_columns(conn, "koala_agent_actions"), "details"):
        rows = _rows(conn,
            "SELECT json_extract(details, '$.heat_band'), COUNT(*) "
            "FROM koala_agent_actions WHERE json_extract(details, '$.heat_band') IS NOT NULL "
            "GROUP BY json_extract(details, '$.heat_band')")
        if rows:
            heat_band_dist = {(r[0] or "null"): int(r[1]) for r in rows}

    return {
        "available": True,
        "total_verdict_state_rows": total,
        "eligibility_state_distribution": state_dist,
        "submitted_count": submitted_count,
        "eligible_or_submitted_count": eligible_count,
        "distinct_citable_agents_distribution": citable_dist,
        "heat_band_distribution": heat_band_dist,
    }


def _section_readiness(data: dict[str, Any]) -> dict[str, Any]:
    papers = data["paper_coverage"].get("total_papers", 0) or 0
    claims = data["claim_coverage"].get("total_extracted_claims", 0) or 0
    verifs = data["verification_distribution"].get("total_verified", 0) or 0
    distinct_labels = data["verification_distribution"].get("distinct_normalized_labels", 0) or 0
    live_comments = data["action_telemetry"].get("live_comment_count", 0) or 0
    live_verdicts = data["action_telemetry"].get("live_verdict_count", 0) or 0
    live_total = live_comments + live_verdicts

    if papers >= 50 and claims >= 500 and verifs >= 500 and distinct_labels >= 3:
        dataset_color = "green"
        dataset_note = "Target coverage reached across papers, claims, and verifications."
    elif papers >= 20 and claims >= 100 and verifs >= 100:
        dataset_color = "yellow"
        dataset_note = "Partial coverage — approaching usefulness but not yet at target volume."
    else:
        dataset_color = "red"
        dataset_note = "Insufficient data collected. Dataset not yet useful for GSR reuse."

    if live_total >= 20:
        live_color = "green"
    elif live_total > 0:
        live_color = "yellow"
    else:
        live_color = "red"

    bottlenecks = []
    if papers < 20:
        bottlenecks.append(f"paper coverage too low ({papers} < 20)")
    if claims < 100:
        bottlenecks.append(f"extracted claims too low ({claims} < 100)")
    if verifs < 100:
        bottlenecks.append(f"claim verifications too low ({verifs} < 100)")
    if distinct_labels < 3:
        bottlenecks.append(f"verification label diversity too low ({distinct_labels} distinct labels)")
    if live_total == 0:
        bottlenecks.append("zero live interactions posted")

    strengths = []
    if papers >= 50:
        strengths.append(f"paper coverage ({papers} papers)")
    if claims >= 500:
        strengths.append(f"claim grounding ({claims} extracted claims)")
    if verifs >= 500:
        strengths.append(f"verification depth ({verifs} verified claims)")
    if live_total >= 20:
        strengths.append(f"live interaction volume ({live_total} live actions)")
    if not strengths:
        if papers > 0:
            strengths.append("paper sync is working")
        if claims > 0:
            strengths.append("claim extraction pipeline is producing output")

    gsr_reuse_note = _gsr_reuse_note(claims, verifs, distinct_labels, live_total)

    return {
        "overall_dataset_color": dataset_color,
        "overall_dataset_note": dataset_note,
        "live_interaction_color": live_color,
        "live_comments": live_comments,
        "live_verdicts": live_verdicts,
        "gsr_reuse_note": gsr_reuse_note,
        "primary_bottlenecks": bottlenecks if bottlenecks else ["none — all green"],
        "strongest_data_layers": strengths if strengths else ["none — all layers sparse"],
    }


def _gsr_reuse_note(claims: int, verifs: int, distinct_labels: int, live_total: int) -> str:
    if claims >= 500 and verifs >= 500 and distinct_labels >= 3:
        if live_total >= 20:
            return (
                "Claim grounding is strong and live social outcomes are present. "
                "Dataset is ready for GSR reuse."
            )
        return (
            "Claim grounding is already strong but live social outcomes are still sparse. "
            "Useful for evidence-checking tasks; trajectory / outcome modeling needs more live data."
        )
    if claims >= 100 and verifs >= 100:
        return (
            "Claim and verification data are accumulating. "
            "Partial reuse possible for claim-extraction tuning; "
            "full GSR reuse requires more volume and label diversity."
        )
    return (
        "Claim-level data is too sparse for meaningful GSR reuse. "
        "Priority: increase paper coverage and run more reactive analysis cycles."
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

_COLOR_EMOJI = {"green": "🟢 GREEN", "yellow": "🟡 YELLOW", "red": "🔴 RED"}


def _render_markdown(data: dict[str, Any], db_path: str) -> str:
    lines: list[str] = []
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines += [
        f"# Dataset Health Audit",
        f"",
        f"**DB:** `{db_path}`  ",
        f"**Generated:** {now_str}",
        f"",
    ]

    # A. Table inventory
    inv = data["table_inventory"]
    lines += ["## A. Table Inventory", ""]
    lines += ["| Table | Rows |", "|---|---|"]
    for tbl, cnt in sorted(inv["all_tables"].items()):
        lines.append(f"| `{tbl}` | {cnt} |")
    lines.append("")
    if inv["key_tables_missing"]:
        lines += [
            "**Missing key tables** (expected by design spec but not in schema):",
            "",
        ]
        for t in inv["key_tables_missing"]:
            lines.append(f"- `{t}`")
        lines.append("")

    # B. Paper coverage
    pc = data["paper_coverage"]
    lines += ["## B. Paper-Level Coverage", "", "| Metric | Value |", "|---|---|"]
    for k, v in pc.items():
        label = k.replace("_", " ").title()
        lines.append(f"| {label} | {v if v is not None else 'N/A'} |")
    lines.append("")

    # C. Comment coverage
    cc = data["comment_coverage"]
    lines += ["## C. Comment-Level Coverage", ""]
    if not cc.get("available"):
        lines.append("_Table `koala_comments` not found._")
    else:
        lines += ["| Metric | Value |", "|---|---|"]
        for k, v in cc.items():
            if k == "available":
                continue
            label = k.replace("_", " ").title()
            lines.append(f"| {label} | {v if v is not None else 'N/A'} |")
    lines.append("")

    # D. Claim coverage
    clm = data["claim_coverage"]
    lines += ["## D. Claim-Level Coverage", ""]
    if not clm.get("available"):
        lines.append("_Table `koala_extracted_claims` not found._")
    else:
        lines += ["| Metric | Value |", "|---|---|"]
        for k, v in clm.items():
            if k in ("available", "category_distribution"):
                continue
            label = k.replace("_", " ").title()
            lines.append(f"| {label} | {v if v is not None else 'N/A'} |")
        if clm.get("category_distribution"):
            lines += ["", "**Category distribution:**", ""]
            lines += ["| Category | Count |", "|---|---|"]
            for cat, cnt in sorted(clm["category_distribution"].items(), key=lambda x: -x[1]):
                lines.append(f"| `{cat}` | {cnt} |")
    lines.append("")

    # E. Verification distribution
    vd = data["verification_distribution"]
    lines += ["## E. Verification Label Distribution", ""]
    if not vd.get("available"):
        lines.append("_Table `koala_claim_verifications` not found._")
    elif vd.get("verdict_column_missing"):
        lines.append(f"_Total verified: {vd['total_verified']} — verdict column unavailable._")
    else:
        lines += ["| Metric | Value |", "|---|---|"]
        lines.append(f"| Total Verified | {vd['total_verified']} |")
        lines.append(f"| Refuted Rate | {vd['refuted_rate']:.1%} |")
        lines.append(f"| Insufficient Rate | {vd['insufficient_rate']:.1%} |")
        lines.append(f"| Distinct Normalized Labels | {vd['distinct_normalized_labels']} |")
        lines += ["", "**Normalized label counts:**", "", "| Label | Count |", "|---|---|"]
        for label, cnt in sorted(vd.get("normalized_label_counts", {}).items(), key=lambda x: -x[1]):
            lines.append(f"| `{label}` | {cnt} |")
    lines.append("")

    # F. Action telemetry
    at = data["action_telemetry"]
    lines += ["## F. Action / Trajectory Telemetry", ""]
    if not at.get("available"):
        lines.append("_Table `koala_agent_actions` not found._")
    else:
        lines += [
            f"- **Total actions:** {at['total_actions']}",
            f"- **Live comments posted:** {at['live_comment_count']}",
            f"- **Live verdicts submitted:** {at['live_verdict_count']}",
            "",
        ]
        matrix = at.get("action_type_status_matrix", {})
        if matrix:
            all_statuses = sorted({s for d in matrix.values() for s in d})
            header = "| Action Type | " + " | ".join(all_statuses) + " |"
            sep = "|---|" + "|---|" * len(all_statuses)
            lines += ["**Action × Status matrix:**", "", header, sep]
            for atype, statuses in sorted(matrix.items()):
                row = f"| `{atype}` | " + " | ".join(str(statuses.get(s, 0)) for s in all_statuses) + " |"
                lines.append(row)
            lines.append("")
        targeted = at.get("targeted_reason_counts")
        if targeted:
            lines += ["**Targeted skip/block reason counts:**", "", "| Reason | Count |", "|---|---|"]
            for reason, cnt in targeted.items():
                lines.append(f"| `{reason}` | {cnt} |")
            lines.append("")
    lines.append("")

    # G. Eligibility telemetry
    et = data["eligibility_telemetry"]
    lines += ["## G. Eligibility / Strategy Telemetry", ""]
    if not et.get("available"):
        lines.append("_Table `koala_verdict_state` not found._")
    else:
        lines += [
            f"- **Total verdict state rows:** {et['total_verdict_state_rows']}",
            f"- **Submitted verdicts:** {et.get('submitted_count', 'N/A')}",
            f"- **Eligible or submitted:** {et.get('eligible_or_submitted_count', 'N/A')}",
            "",
        ]
        if et.get("eligibility_state_distribution"):
            lines += ["**Eligibility state distribution:**", "", "| State | Count |", "|---|---|"]
            for state, cnt in sorted(et["eligibility_state_distribution"].items(), key=lambda x: -x[1]):
                lines.append(f"| `{state}` | {cnt} |")
            lines.append("")
        if et.get("distinct_citable_agents_distribution"):
            lines += ["**Distinct citable other-agent distribution:**", "", "| Count | Papers |", "|---|---|"]
            for n, papers in sorted(et["distinct_citable_agents_distribution"].items(), key=lambda x: int(x[0])):
                lines.append(f"| {n} | {papers} |")
            lines.append("")
        if et.get("heat_band_distribution"):
            lines += ["**Heat band distribution (from verdict_draft details):**", "", "| Band | Count |", "|---|---|"]
            for band, cnt in sorted(et["heat_band_distribution"].items(), key=lambda x: -x[1]):
                lines.append(f"| `{band}` | {cnt} |")
            lines.append("")
    lines.append("")

    # H. Readiness judgement
    rd = data["readiness"]
    lines += ["## H. Dataset Readiness Judgement", ""]
    overall = _COLOR_EMOJI.get(rd["overall_dataset_color"], rd["overall_dataset_color"])
    live = _COLOR_EMOJI.get(rd["live_interaction_color"], rd["live_interaction_color"])
    lines += [
        f"| Dimension | Status |",
        f"|---|---|",
        f"| Overall Dataset Readiness | **{overall}** |",
        f"| Live Interaction Readiness | **{live}** |",
        f"",
        f"**Note:** {rd['overall_dataset_note']}",
        f"",
        f"**GSR Reuse:** {rd['gsr_reuse_note']}",
        f"",
        f"**Primary bottlenecks:**",
        f"",
    ]
    for b in rd["primary_bottlenecks"]:
        lines.append(f"- {b}")
    lines += [
        "",
        "**Strongest available data layers:**",
        "",
    ]
    for s in rd["strongest_data_layers"]:
        lines.append(f"- {s}")
    lines.append("")

    return "\n".join(lines)


def _console_summary(data: dict[str, Any], db_path: str, md_path: str, json_path: str) -> None:
    rd = data["readiness"]
    color_map = {"green": "GREEN", "yellow": "YELLOW", "red": "RED"}
    overall = color_map.get(rd["overall_dataset_color"], rd["overall_dataset_color"]).upper()
    live = color_map.get(rd["live_interaction_color"], rd["live_interaction_color"]).upper()
    bottleneck = "; ".join(rd["primary_bottlenecks"][:2]) or "none"

    print(f"\n{'='*60}")
    print(f"Dataset Audit Summary")
    print(f"{'='*60}")
    print(f"  DB path:      {db_path}")
    print(f"  Markdown:     {md_path}")
    print(f"  JSON:         {json_path}")
    print(f"  Overall:      {overall}")
    print(f"  Live posts:   {live}  ({rd['live_comments']} comments, {rd['live_verdicts']} verdicts)")
    print(f"  Bottleneck:   {bottleneck}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_audit(db_path: str) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        present = _tables(conn)

        table_inventory = _section_table_inventory(conn)
        paper_coverage = _section_paper_coverage(conn, present)
        comment_coverage = _section_comment_coverage(conn, present)
        claim_coverage = _section_claim_coverage(conn, present)
        verification_distribution = _section_verification_distribution(conn, present)
        action_telemetry = _section_action_telemetry(conn, present)
        eligibility_telemetry = _section_eligibility_telemetry(conn, present)

        data: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "db_path": db_path,
            "table_inventory": table_inventory,
            "paper_coverage": paper_coverage,
            "comment_coverage": comment_coverage,
            "claim_coverage": claim_coverage,
            "verification_distribution": verification_distribution,
            "action_telemetry": action_telemetry,
            "eligibility_telemetry": eligibility_telemetry,
        }
        data["readiness"] = _section_readiness(data)
        return data
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only dataset health audit for Koala/GSR competition DB.")
    parser.add_argument("--db", default="./workspace/koala_agent.db", help="Path to SQLite DB")
    parser.add_argument("--out-md", default="./workspace/reports/dataset_audit.md", help="Markdown output path")
    parser.add_argument("--out-json", default="./workspace/reports/dataset_audit.json", help="JSON output path")
    args = parser.parse_args()

    db_path = str(Path(args.db).resolve())

    if not Path(db_path).exists():
        print(f"[audit] DB not found at {db_path} — creating empty schema for audit.", file=sys.stderr)
        conn = sqlite3.connect(db_path)
        conn.close()

    data = run_audit(db_path)

    md_path = Path(args.out_md).resolve()
    json_path = Path(args.out_json).resolve()
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    md_path.write_text(_render_markdown(data, db_path), encoding="utf-8")
    json_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    _console_summary(data, db_path, str(md_path), str(json_path))


if __name__ == "__main__":
    main()
