"""Phase 7: Run Summary — dry-run inspection tool for paper/action readiness.

Builds a per-paper summary dict from local SQLite state with no Koala API
calls and no LLM calls.  All data is derived from the local DB.

Public API
----------
build_paper_summary(paper_row, db, now)      -> dict
build_run_summary(db, now=None, paper_ids=None) -> list[dict]
write_run_summary_markdown(summary, path)
write_run_summary_jsonl(summary, path)

CLI
---
    python -m gsr_agent.reporting.run_summary [--db PATH] [--out DIR]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from ..rules.timeline import get_micro_phase, get_paper_phase
from ..rules.verdict_eligibility import MIN_DISTINCT_OTHER_AGENTS
from ..strategy.aggressive_mode import is_aggressive_mode
from ..strategy.heat import paper_heat_band
from ..strategy.opportunity_manager import PREFERRED_COMMENT_MIN, SATURATED_COMMENT_THRESHOLD

if TYPE_CHECKING:
    from ..storage.db import KoalaDB

_STRONG_SIGNAL_THRESHOLD: float = 0.75

_VALID_ACTIONS = frozenset({
    "consider_verdict_draft",
    "consider_reactive_comment",
    "consider_seed_comment",
    "seed_or_verdict_candidate",
    "not_eligible_no_own_comment",
    "not_eligible_window",
    "run_reactive_analysis",
    "skip_too_cold",
    "skip_too_crowded",
    "skip_no_signal",
    "monitor",
})


def _parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return None


def _compute_verdict_eligibility(
    heat_band: str,
    strongest_conf: Optional[float],
    distinct_other_agents: int,
    *,
    aggressive: bool = False,
) -> tuple[bool, str]:
    """Mirror of verdict_assembly Gate 1 + Gate 2 using count-based inputs.

    Returns (eligible, reason_code) without requiring full reactive result objects.
    In aggressive mode, saturated and crowded heat-band vetoes are bypassed so
    that high-comment papers remain verdict-eligible in the final-24h window.
    """
    strong_signal = (
        strongest_conf is not None and strongest_conf >= _STRONG_SIGNAL_THRESHOLD
    )

    if heat_band == "saturated":
        if not aggressive:
            return False, "saturated_low_value_v0"
        # Aggressive: bypass saturated veto; fall through to citation gate.
        if distinct_other_agents < MIN_DISTINCT_OTHER_AGENTS:
            return False, "insufficient_distinct_other_agent_citations"
        return True, "eligible"

    if not strong_signal:
        if heat_band == "cold":
            return False, "cold_no_override"
        if heat_band == "crowded":
            if not aggressive:
                return False, "crowded_no_override"
            # Aggressive: bypass crowded veto; fall through to citation gate.
        else:
            return False, "no_react_signal"

    if distinct_other_agents < MIN_DISTINCT_OTHER_AGENTS:
        return False, "insufficient_distinct_other_agent_citations"

    return True, "eligible"


def _recommended_action(
    *,
    verdict_eligible: bool,
    has_react_candidate: bool,
    heat_band: str,
    citable_other: int,
    comments_analyzed: int,
    total_comments: int = 0,
    is_seed_window: bool = False,
    aggressive: bool = False,
    has_own_comment: bool = False,
) -> str:
    """Return the recommended next action string for a paper.

    Priority order (highest to lowest):
      consider_verdict_draft    — verdict gate passed; draft ready for review
      consider_reactive_comment — Phase 5A found a react candidate
      consider_seed_comment     — SEED_WINDOW paper with social proof (1–12 comments)
      seed_or_verdict_candidate — aggressive mode: saturated paper with own comment + citations
      not_eligible_no_own_comment — aggressive mode: saturated, no own comment yet
      not_eligible_window       — aggressive mode: saturated, other eligibility gap
      skip_too_crowded          — saturated band, low marginal value (normal mode)
      skip_too_cold             — cold band, no social proof
      run_reactive_analysis     — other comments exist but haven't been analysed
      skip_no_signal            — analysis done, no reactive signal found
      monitor                   — default; paper is active but no clear action
    """
    if verdict_eligible:
        return "consider_verdict_draft"
    if has_react_candidate:
        return "consider_reactive_comment"
    if is_seed_window and PREFERRED_COMMENT_MIN <= total_comments <= SATURATED_COMMENT_THRESHOLD:
        return "consider_seed_comment"
    if heat_band == "saturated":
        if aggressive:
            if has_own_comment and citable_other >= MIN_DISTINCT_OTHER_AGENTS:
                return "seed_or_verdict_candidate"
            if not has_own_comment:
                return "not_eligible_no_own_comment"
            return "not_eligible_window"
        return "skip_too_crowded"
    if heat_band == "cold":
        return "skip_too_cold"
    if citable_other > 0 and comments_analyzed == 0:
        return "run_reactive_analysis"
    if not has_react_candidate:
        return "skip_no_signal"
    return "monitor"


def build_paper_summary(paper_row: dict, db: "KoalaDB", now: datetime) -> dict:
    """Build a summary dict for a single paper using local DB data only.

    Args:
        paper_row: row dict from db.get_papers()
        db:        local SQLite state store
        now:       current UTC datetime

    Returns:
        Dict with all Phase 7 summary fields. Missing data yields null/0.
    """
    paper_id = paper_row["paper_id"]
    open_time = _parse_dt(paper_row.get("open_time", ""))

    phase = get_paper_phase(now, open_time).value if open_time else "UNKNOWN"
    micro_phase = get_micro_phase(now, open_time).value if open_time else "UNKNOWN"

    stats = db.get_comment_stats(paper_id)
    distinct_n = db.get_distinct_other_agent_count(paper_id)
    heat_band = paper_heat_band(distinct_n)

    reactive = db.get_phase5a_stats(paper_id)
    strongest_conf = db.get_strongest_contradiction_confidence(paper_id)
    has_react_candidate = reactive["react_count"] > 0

    _aggressive = is_aggressive_mode()
    verdict_eligible, verdict_reason_code = _compute_verdict_eligibility(
        heat_band, strongest_conf, distinct_n, aggressive=_aggressive
    )

    latest_action = db.get_latest_action_for_paper(paper_id)
    latest_artifact_url = (
        latest_action.get("github_file_url") if latest_action else None
    )

    action = _recommended_action(
        verdict_eligible=verdict_eligible,
        has_react_candidate=has_react_candidate,
        heat_band=heat_band,
        citable_other=stats["citable_other"],
        comments_analyzed=reactive["comments_analyzed"],
        total_comments=stats["total"],
        is_seed_window=(micro_phase == "SEED_WINDOW"),
        aggressive=_aggressive,
        has_own_comment=stats["ours"] > 0,
    )

    return {
        "paper_id": paper_id,
        "title": paper_row.get("title") or "",
        "phase": phase,
        "micro_phase": micro_phase,
        "heat_band": heat_band,
        "distinct_citable_other_agents": distinct_n,
        "comment_counts": {
            "total": stats["total"],
            "ours": stats["ours"],
            "citable_other": stats["citable_other"],
        },
        "reactive_stats": {
            "react": reactive["react_count"],
            "skip": reactive["skip_count"],
            "unclear": reactive["unclear_count"],
        },
        "strongest_contradiction_confidence": strongest_conf,
        "has_best_reactive_candidate": has_react_candidate,
        "verdict_eligibility": {
            "eligible": verdict_eligible,
            "reason_code": verdict_reason_code,
        },
        "latest_artifact_url": latest_artifact_url,
        "recommended_next_action": action,
    }


def build_run_summary(
    db: "KoalaDB",
    now: Optional[datetime] = None,
    paper_ids: Optional[List[str]] = None,
) -> List[dict]:
    """Build a per-paper summary list from local DB state.

    Args:
        db:        local SQLite state store
        now:       current UTC datetime (defaults to utcnow when None)
        paper_ids: restrict to these paper IDs; None means all tracked papers

    Returns:
        List of paper summary dicts ordered by open_time descending.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    papers = db.get_papers(paper_ids)
    return [build_paper_summary(p, db, now) for p in papers]


def write_run_summary_markdown(summary: List[dict], path: "str | Path") -> None:
    """Write a human-readable markdown run-summary report.

    Args:
        summary: output of build_run_summary
        path:    destination file path; parent directories are created automatically
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    now_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# GSR Agent Run Summary",
        f"Generated: {now_str} | Papers: {len(summary)}",
        "",
    ]

    for s in summary:
        lines.append("---")
        lines.append("")
        title = s["title"] or s["paper_id"]
        lines.append(f"## {s['paper_id']} — {title}")
        lines.append(f"- Phase: {s['phase']} | Micro: {s['micro_phase']}")
        lines.append(
            f"- Heat: {s['heat_band']}"
            f" | Distinct other agents: {s['distinct_citable_other_agents']}"
        )
        c = s["comment_counts"]
        lines.append(
            f"- Comments: total={c['total']}, ours={c['ours']},"
            f" citable_other={c['citable_other']}"
        )
        r = s["reactive_stats"]
        conf = s["strongest_contradiction_confidence"]
        conf_str = f"{conf:.2f}" if conf is not None else "n/a"
        lines.append(
            f"- Reactive: react={r['react']}, skip={r['skip']}, unclear={r['unclear']}"
            f" | Strongest conf: {conf_str}"
        )
        candidate_str = "YES" if s["has_best_reactive_candidate"] else "no"
        lines.append(f"- Best reactive candidate: {candidate_str}")
        v = s["verdict_eligibility"]
        if s.get("recommended_next_action") == "consider_seed_comment":
            eligible_str = "N/A (seed window)"
        elif v["eligible"]:
            eligible_str = "ELIGIBLE"
        else:
            eligible_str = f"NOT ELIGIBLE ({v['reason_code']})"
        lines.append(f"- Verdict: {eligible_str}")
        if "reactive_live_posted" in s:
            live_reason = s.get("reactive_live_reason") or "n/a"
            lines.append(
                f"- Reactive live: posted={s['reactive_live_posted']},"
                f" reason={live_reason}"
            )
        if "seed_live_posted" in s or "seed_draft_created" in s:
            seed_posted = s.get("seed_live_posted", False)
            seed_draft = s.get("seed_draft_created", False)
            seed_reason = s.get("seed_live_reason") or (
                "live_posted" if seed_posted
                else "draft_created" if seed_draft
                else "no_seed"
            )
            lines.append(
                f"- Seed live: posted={seed_posted}, draft={seed_draft},"
                f" reason={seed_reason}"
            )
        artifact = s["latest_artifact_url"] or "(none)"
        lines.append(f"- Latest artifact: {artifact}")
        lines.append(f"- **→ {s['recommended_next_action']}**")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_run_summary_jsonl(summary: List[dict], path: "str | Path") -> None:
    """Write one JSON object per paper per line (JSONL format).

    Args:
        summary: output of build_run_summary
        path:    destination file path; parent directories are created automatically
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in summary:
            f.write(json.dumps(s, default=str) + "\n")


def main() -> None:
    """CLI entrypoint: write run-summary reports to workspace/reports/.

    Usage:
        python -m gsr_agent.reporting.run_summary [--db PATH] [--out DIR]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="GSR Agent Phase 7 — Run Summary (dry-run only, no Koala writes)"
    )
    parser.add_argument(
        "--db", default="./workspace/koala_agent.db", help="Path to SQLite database"
    )
    parser.add_argument(
        "--out", default="./workspace/reports", help="Output directory for reports"
    )
    args = parser.parse_args()

    from ..storage.db import KoalaDB

    db = KoalaDB(args.db)
    try:
        now = datetime.now(timezone.utc)
        summary = build_run_summary(db, now)
    finally:
        db.close()

    out_dir = Path(args.out)
    md_path = out_dir / "run_summary.md"
    jsonl_path = out_dir / "run_summary.jsonl"

    write_run_summary_markdown(summary, md_path)
    write_run_summary_jsonl(summary, jsonl_path)

    print(
        f"[run_summary] {len(summary)} papers"
        f" | md → {md_path} | jsonl → {jsonl_path}"
    )
    for s in summary:
        print(
            f"  {s['paper_id'][:24]:24s}"
            f" heat={s['heat_band']:10s}"
            f" action={s['recommended_next_action']}"
        )


if __name__ == "__main__":
    main()
