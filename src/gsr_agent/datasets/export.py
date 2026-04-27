"""Dataset export: write competition DB contents to structured JSONL + markdown.

Exports all available tables into a versioned directory layout. Read-only:
does not modify any DB rows, does not make API calls, does not post anything.

Public API
----------
export_competition_dataset(db, out_dir) -> dict[str, int]

CLI
---
    python -m gsr_agent.datasets.export --db workspace/koala_agent.db
                                         --out competition_exports
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage.db import KoalaDB

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _try_parse_json(value: Any) -> Any:
    """Return parsed JSON when value is a JSON string; otherwise return as-is."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in ("{", "["):
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return value


def _row_to_dict(row) -> dict:
    """Convert a sqlite3.Row (or dict) to a plain dict, parsing JSON string fields."""
    d = dict(row)
    return {k: _try_parse_json(v) for k, v in d.items()}


def _write_jsonl(path: Path, rows: List[dict]) -> int:
    """Write rows to JSONL, one JSON object per line. Returns rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return len(rows)


def _query(conn, sql: str) -> List[dict]:
    """Execute a read-only SQL query; return empty list on any error."""
    try:
        cursor = conn.execute(sql)
        return [_row_to_dict(row) for row in cursor.fetchall()]
    except Exception as exc:
        log.warning("Export query skipped (%s): %s", type(exc).__name__, exc)
        return []


def _count(conn, table: str) -> int:
    """Return row count for table; 0 if table does not exist."""
    try:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        return int(row["n"]) if row else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Per-section export helpers
# ---------------------------------------------------------------------------

def _export_raw(conn, raw_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    papers = _query(conn, "SELECT * FROM koala_papers ORDER BY paper_id")
    counts["papers.jsonl"] = _write_jsonl(raw_dir / "papers.jsonl", papers)

    comments = _query(
        conn,
        "SELECT * FROM koala_comments ORDER BY paper_id, created_at, comment_id",
    )
    counts["comments.jsonl"] = _write_jsonl(raw_dir / "comments.jsonl", comments)

    actions = _query(
        conn,
        "SELECT * FROM koala_agent_actions ORDER BY paper_id, created_at, action_id",
    )
    counts["actions.jsonl"] = _write_jsonl(raw_dir / "actions.jsonl", actions)

    karma = _query(
        conn,
        "SELECT * FROM koala_karma_ledger ORDER BY paper_id, created_at, ledger_id",
    )
    counts["karma_ledger.jsonl"] = _write_jsonl(raw_dir / "karma_ledger.jsonl", karma)

    verdict_state = _query(conn, "SELECT * FROM koala_verdict_state ORDER BY paper_id")
    counts["verdict_state.jsonl"] = _write_jsonl(
        raw_dir / "verdict_state.jsonl", verdict_state
    )

    return counts


def _export_gsr_structured(conn, gsr_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    claims = _query(
        conn,
        "SELECT * FROM koala_extracted_claims ORDER BY paper_id, comment_id, created_at",
    )
    counts["extracted_claims.jsonl"] = _write_jsonl(
        gsr_dir / "extracted_claims.jsonl", claims
    )

    verifications = _query(
        conn,
        "SELECT * FROM koala_claim_verifications "
        "ORDER BY paper_id, comment_id, created_at",
    )
    counts["claim_verifications.jsonl"] = _write_jsonl(
        gsr_dir / "claim_verifications.jsonl", verifications
    )

    drafts = _query(
        conn,
        "SELECT * FROM koala_reactive_drafts ORDER BY paper_id, created_at, draft_id",
    )
    counts["reactive_drafts.jsonl"] = _write_jsonl(
        gsr_dir / "reactive_drafts.jsonl", drafts
    )

    return counts


def _export_policy(conn, policy_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    run_summary = _query(
        conn,
        """
        SELECT
            p.paper_id,
            p.title,
            p.state,
            p.open_time,
            p.review_end_time,
            p.verdict_end_time,
            vs.eligibility_state,
            vs.submitted,
            vs.distinct_citable_other_agents,
            vs.internal_confidence,
            COUNT(a.action_id) AS action_count
        FROM koala_papers p
        LEFT JOIN koala_verdict_state vs ON vs.paper_id = p.paper_id
        LEFT JOIN koala_agent_actions a ON a.paper_id = p.paper_id
        GROUP BY p.paper_id
        ORDER BY p.paper_id
        """,
    )
    counts["run_summary.jsonl"] = _write_jsonl(
        policy_dir / "run_summary.jsonl", run_summary
    )

    action_traces = _query(
        conn,
        """
        SELECT
            a.action_id,
            a.paper_id,
            a.action_type,
            a.external_id,
            a.github_file_url,
            a.created_at,
            a.status,
            a.error_message,
            a.details
        FROM koala_agent_actions a
        ORDER BY a.paper_id, a.created_at, a.action_id
        """,
    )
    counts["action_traces.jsonl"] = _write_jsonl(
        policy_dir / "action_traces.jsonl", action_traces
    )

    return counts


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _compute_stats(conn) -> Dict[str, int]:
    return {
        "papers": _count(conn, "koala_papers"),
        "comments": _count(conn, "koala_comments"),
        "comments_ours": _count_where(conn, "koala_comments", "is_ours = 1"),
        "comments_citable_other": _count_where(
            conn, "koala_comments", "is_ours = 0 AND is_citable = 1"
        ),
        "actions": _count(conn, "koala_agent_actions"),
        "karma_entries": _count(conn, "koala_karma_ledger"),
        "extracted_claims": _count(conn, "koala_extracted_claims"),
        "claim_verifications": _count(conn, "koala_claim_verifications"),
        "reactive_drafts": _count(conn, "koala_reactive_drafts"),
        "verdict_state_rows": _count(conn, "koala_verdict_state"),
    }


def _count_where(conn, table: str, where: str) -> int:
    try:
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE {where}"
        ).fetchone()
        return int(row["n"]) if row else 0
    except Exception:
        return 0


def _write_dataset_card(
    path: Path,
    generated_at: str,
    file_counts: Dict[str, int],
    stats: Dict[str, int],
) -> None:
    files_section = "\n".join(
        f"- `{fname}`: {n} rows" for fname, n in sorted(file_counts.items())
    )
    lines = [
        "# Dataset Card — Koala GSR Competition Trajectory",
        "",
        "## Purpose",
        "Training and evaluation dataset derived from a GSR (Ground Source Review) "
        "competition agent trajectory. Contains paper metadata, reviewer comments, "
        "agent actions, claim extraction, verification results, and reactive drafts.",
        "",
        "## Source",
        "- **Agent**: Koala GSR Competition Agent (peer-review-agents)",
        "- **Competition**: Koala Science GSR competition",
        f"- **Generated**: {generated_at}",
        "",
        "## Files Exported",
        files_section,
        "",
        "## Summary Statistics",
        f"- Papers: {stats['papers']}",
        f"- Total comments: {stats['comments']}",
        f"- Our comments: {stats['comments_ours']}",
        f"- Citable other-agent comments: {stats['comments_citable_other']}",
        f"- Agent actions logged: {stats['actions']}",
        f"- Karma ledger entries: {stats['karma_entries']}",
        f"- Extracted claims: {stats['extracted_claims']}",
        f"- Claim verifications: {stats['claim_verifications']}",
        f"- Reactive drafts: {stats['reactive_drafts']}",
        f"- Verdict state rows: {stats['verdict_state_rows']}",
        "",
        "## Known Limitations",
        "- Dataset reflects a single competition run; may not generalize.",
        "- Claim extraction and verification are heuristic (no LLM calls in export).",
        "- Verdict scores from `heuristic_v0` only; no calibrated model scores.",
        "- Missing tables produce empty JSONL files (0 rows) — not an error.",
        "- Timestamps are ISO-8601 UTC strings from SQLite storage.",
        "",
        "## Privacy and Safety Note",
        "**Do not publish publicly without review.** This dataset contains competition "
        "discussion text, reviewer comments, and agent outputs from the Koala platform. "
        "Review for sensitive content, PII, and competition rules compliance before "
        "any public release.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_stats(path: Path, stats: Dict[str, int], generated_at: str) -> None:
    lines = [
        "# Export Summary Statistics",
        "",
        f"Generated: {generated_at}",
        "",
        "| Metric | Count |",
        "|--------|------:|",
        f"| Papers | {stats['papers']} |",
        f"| Total comments | {stats['comments']} |",
        f"| Our comments | {stats['comments_ours']} |",
        f"| Citable other-agent comments | {stats['comments_citable_other']} |",
        f"| Agent actions | {stats['actions']} |",
        f"| Karma ledger entries | {stats['karma_entries']} |",
        f"| Extracted claims | {stats['extracted_claims']} |",
        f"| Claim verifications | {stats['claim_verifications']} |",
        f"| Reactive drafts | {stats['reactive_drafts']} |",
        f"| Verdict state rows | {stats['verdict_state_rows']} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main export entry point
# ---------------------------------------------------------------------------

def export_competition_dataset(db: "KoalaDB", out_dir: str) -> Dict[str, int]:
    """Export all competition DB tables to JSONL + markdown reports.

    Creates the output directory structure if it does not exist. Missing
    optional DB tables produce empty JSONL files (0 rows) rather than errors.
    Does not modify any DB rows.

    Args:
        db:      KoalaDB instance (read-only access via _conn)
        out_dir: root output directory

    Returns:
        Dict mapping filename → row count for every file written.
    """
    root = Path(out_dir)
    generated_at = datetime.now(timezone.utc).isoformat()
    conn = db._conn

    counts: Dict[str, int] = {}

    raw_counts = _export_raw(conn, root / "raw")
    counts.update(raw_counts)

    gsr_counts = _export_gsr_structured(conn, root / "gsr_structured")
    counts.update(gsr_counts)

    policy_counts = _export_policy(conn, root / "policy")
    counts.update(policy_counts)

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stats = _compute_stats(conn)
    _write_dataset_card(reports_dir / "dataset_card.md", generated_at, counts, stats)
    _write_summary_stats(reports_dir / "summary_stats.md", stats, generated_at)

    log.info(
        "[export] DONE out=%s files=%d total_rows=%d",
        out_dir,
        len(counts),
        sum(counts.values()),
    )
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint: export competition DB to JSONL dataset files.

    Usage:
        python -m gsr_agent.datasets.export --db workspace/koala_agent.db
                                             --out competition_exports
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Export GSR competition DB to JSONL dataset files"
    )
    parser.add_argument(
        "--db",
        default="workspace/koala_agent.db",
        help="Path to SQLite database (default: workspace/koala_agent.db)",
    )
    parser.add_argument(
        "--out",
        default="competition_exports",
        help="Output root directory (default: competition_exports)",
    )
    args = parser.parse_args()

    from ..storage.db import KoalaDB

    db = KoalaDB(args.db)
    try:
        print(f"[export] START db={args.db} out={args.out}")
        counts = export_competition_dataset(db, args.out)
        total = sum(counts.values())
        print(f"[export] DONE files={len(counts)} total_rows={total}")
        for fname, n in sorted(counts.items()):
            print(f"  {fname}: {n} rows")
    finally:
        db.close()


if __name__ == "__main__":
    main()
