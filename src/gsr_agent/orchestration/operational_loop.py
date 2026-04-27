"""Phase 8: Operational Loop — dry-run orchestrator.

Wires Phase 5A (reactive analysis) → Phase 5B (reactive comment) → Phase 6
(verdict assembly) → Phase 7 (run summary) into a single top-level loop.

Phase 8.5: dedup gates prevent regenerating the same reactive/verdict drafts
on repeated runs.

Phase 9: controlled live reactive posting. --live-reactive + KOALA_RUN_MODE=live
+ test_mode=False allows at most ONE reactive comment per loop run.

Phase 9.5: live-reactive observability counters.

Phase 10: controlled live verdict submission. --live-verdict + KOALA_RUN_MODE=live
+ test_mode=False + explicit paper_ids (allowlist) allows at most ONE verdict per
loop run. More conservative than reactive live: requires explicit paper allowlist.
Verdict live is independent of reactive live.

Default is always dry-run. submit_verdict is never called without all gates passing.

Public API
----------
run_operational_loop(db, now, *, paper_ids=None, max_papers=None,
                     karma_remaining=100.0, output_dir="./workspace/reports",
                     test_mode=True, live_reactive=False, live_verdict=False) -> dict

CLI
---
    python -m gsr_agent.orchestration.operational_loop [--db PATH] [--out DIR]
        [--max-papers N] [--paper-id ID ...] [--live-reactive] [--live-verdict]
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING

from ..artifacts.github import (
    get_run_mode,
    is_github_publish_configured,
    publish_verdict_artifact,
    validate_artifact_for_live_action,
)
from ..commenting.orchestrator import plan_and_post_reactive_comment
from ..koala.errors import KoalaWindowClosedError
from ..commenting.reactive_analysis import (
    analyze_reactive_candidates_for_paper,
    select_best_reactive_candidate_for_paper,
)
from ..koala.models import Paper
from ..reporting.run_summary import (
    build_run_summary,
    write_run_summary_jsonl,
    write_run_summary_markdown,
)
from ..rules.timeline import SAFETY_BUFFER_S, compute_phase_window
from ..rules.verdict_assembly import (
    VerdictEligibilityResult,
    build_verdict_draft_for_paper,
    evaluate_verdict_eligibility,
    plan_verdict_for_paper,
    select_distinct_other_agent_citations,
)
from ..rules.verdict_scoring import VerdictScore

if TYPE_CHECKING:
    from ..storage.db import KoalaDB

log = logging.getLogger(__name__)

_DEDUP_HOURS: float = 12.0


def run_preflight_checks(
    db_path: str,
    output_dir: str,
    *,
    live_reactive: bool = False,
    live_verdict: bool = False,
    paper_ids: Optional[List[str]] = None,
) -> tuple[bool, List[str]]:
    """Validate environment before starting the loop.

    Checks DB parent directory, output directory, and — when any live flag is
    set — KOALA_RUN_MODE, KOALA_API_TOKEN, and GitHub publish configuration.
    For live_verdict, also requires explicit paper_ids.

    Returns:
        (ok, failures) — ok is True when no failures; failures is a list of
        human-readable problem descriptions.
    """
    failures: List[str] = []

    db_parent = Path(db_path).parent
    if not db_parent.exists():
        failures.append(f"DB parent directory does not exist: {db_parent}")

    out_dir = Path(output_dir)
    if not out_dir.exists():
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            failures.append(f"Cannot create output directory {out_dir}: {exc}")

    if live_reactive or live_verdict:
        run_mode = get_run_mode()
        if run_mode != "live":
            failures.append(
                f"KOALA_RUN_MODE must be 'live' for live actions (current: {run_mode!r})"
            )
        if not os.environ.get("KOALA_API_TOKEN"):
            failures.append("KOALA_API_TOKEN is not set; required for live mode")
        if not is_github_publish_configured():
            failures.append(
                "KOALA_ARTIFACT_MODE=github and KOALA_GITHUB_REPO must be set for live mode"
            )

    if live_verdict and paper_ids is None:
        failures.append("--live-verdict requires explicit --paper-id; no allowlist provided")

    return len(failures) == 0, failures


class _DryRunClient:
    """Stub Koala client for test_mode=True. Never makes network calls."""

    def post_comment(
        self,
        paper_id: str,
        body: str,
        github_file_url: str,
        *,
        thread_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        return f"dry-run-{paper_id[:8]}"

    def submit_verdict(self, *args, **kwargs) -> None:
        raise RuntimeError("submit_verdict must never be called in dry-run mode")


def _parse_dt(value: str) -> datetime:
    """Parse ISO-8601 UTC string; fall back to utcnow if invalid."""
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)


def _paper_from_row(row: dict) -> Paper:
    """Reconstruct a Paper dataclass from a koala_papers DB row dict."""
    raw_delib = row.get("deliberating_at")
    return Paper(
        paper_id=row["paper_id"],
        title=row.get("title", ""),
        open_time=_parse_dt(row.get("open_time", "")),
        review_end_time=_parse_dt(row.get("review_end_time", "")),
        verdict_end_time=_parse_dt(row.get("verdict_end_time", "")),
        state=row.get("state", "REVIEW_ACTIVE"),
        pdf_url=row.get("pdf_url", ""),
        local_pdf_path=row.get("local_pdf_path"),
        deliberating_at=_parse_dt(raw_delib) if raw_delib else None,
    )


def _validate_verdict_score(score: Optional[float]) -> Optional[str]:
    """Return None when score is valid for live submission, or an error reason code.

    A valid verdict score is a non-None numeric (int or float, not bool) in [0.0, 10.0].
    """
    if score is None:
        return "missing_verdict_score"
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return "invalid_verdict_score"
    if not (0.0 <= float(score) <= 10.0):
        return "invalid_verdict_score"
    return None


def _submit_live_verdict(
    paper: Paper,
    db: "KoalaDB",
    reactive_results: list,
    now: datetime,
    live_client: Any,
    eligibility: VerdictEligibilityResult,
    *,
    score: Optional[float] = None,
    verdict_score: Optional[VerdictScore] = None,
) -> tuple[bool, str]:
    """Publish a real verdict artifact and submit live to the Koala API.

    Called only after all live-verdict gates have already passed. Returns
    (True, "live_submitted") on success or (False, reason_code) when blocked.
    Score validation is the first gate — any call without a real numeric score
    in [0.0, 10.0] is rejected before touching citations, drafts, or the API.

    Args:
        paper:            the paper to submit a verdict for
        db:               local SQLite state store
        reactive_results: Phase 5A results (used for verdict draft)
        now:              current UTC datetime
        live_client:      real KoalaClient instance
        eligibility:      Gate 1 result already computed by the outer pipeline
        score:            explicit numeric verdict score; None blocks submission

    Returns:
        (True, "live_submitted") on success; (False, reason_code) otherwise.
    """
    score_error = _validate_verdict_score(score)
    if score_error:
        log.info("[verdict] LIVE_SKIP reason=%s paper=%s", score_error, paper.paper_id)
        return False, score_error

    citations = select_distinct_other_agent_citations(paper.paper_id, db)
    if not citations:
        log.info(
            "[verdict] LIVE_SKIP reason=insufficient_citations paper=%s", paper.paper_id
        )
        return False, "insufficient_citations"

    draft = build_verdict_draft_for_paper(
        paper, eligibility, reactive_results, db, now,
        valid_citations=citations, verdict_score=verdict_score,
    )
    if not draft:
        log.info("[verdict] LIVE_SKIP reason=no_draft paper=%s", paper.paper_id)
        return False, "no_draft"

    cited_ids = [c["comment_id"] for c in citations]

    github_file_url = publish_verdict_artifact(
        paper.paper_id, score=score, body=draft, cited_ids=cited_ids, test_mode=False
    )
    validate_artifact_for_live_action(github_file_url)

    live_client.submit_verdict(
        paper.paper_id,
        score=score,
        cited_comment_ids=cited_ids,
        github_file_url=github_file_url,
    )

    db.log_action(
        paper_id=paper.paper_id,
        action_type="verdict_submission",
        github_file_url=github_file_url,
        status="success",
        details={
            "run_mode": "live",
            "cited_comment_ids": cited_ids,
            "artifact_url": github_file_url,
            "cited_count": len(cited_ids),
        },
    )
    log.info(
        "[verdict] LIVE_SUBMITTED paper=%s artifact=%s cited=%d score=%.2f",
        paper.paper_id, github_file_url, len(cited_ids), score,
    )
    return True, "live_submitted"


def _process_paper(
    paper: Paper,
    client: _DryRunClient,
    db: "KoalaDB",
    karma_remaining: float,
    now: datetime,
    test_mode: bool,
    *,
    live_reactive: bool = False,
    live_budget_remaining: int = 0,
    live_client: Optional[Any] = None,
    live_verdict: bool = False,
    verdict_live_budget_remaining: int = 0,
    allowlisted: bool = False,
    score: Optional[float] = None,
) -> dict:
    """Run the full Phase 5→6 pipeline for a single paper.

    Returns a rich outcome dict with both new status fields and legacy boolean
    fields for backward compatibility with run_operational_loop counters.

    reactive_status:   "none"|"dry_run"|"live_posted"|"dedup_skipped"|"skipped"
    verdict_status:    "ineligible"|"dry_run"|"live_submitted"|"dedup_skipped"|"skipped"
    reactive_live_reason: "no_candidate"|"live_disabled"|"live_gate_failed"|
                          "live_budget_exhausted"|"dedup_skipped"|"live_posted"
    verdict_live_reason:  "no_eligible_verdict"|"live_disabled"|"live_gate_failed"|
                          "live_budget_exhausted"|"dedup_skipped"|"allowlist_required"|
                          "missing_verdict_score"|"invalid_verdict_score"|"live_submitted"
    """
    # Phase window — never rely on paper.state alone; verify with timestamps.
    phase_window = compute_phase_window(
        now, paper.state, paper.open_time, paper.deliberating_at
    )
    comment_phase_ok = (
        phase_window.phase == "comment" and phase_window.seconds_left > SAFETY_BUFFER_S
    )
    verdict_phase_ok = (
        phase_window.phase == "verdict" and phase_window.seconds_left > SAFETY_BUFFER_S
    )
    window_decision = (
        "comment" if comment_phase_ok
        else "verdict" if verdict_phase_ok
        else "skip"
    )
    log.info(
        "[window] paper=%s status=%s phase=%s seconds_left=%.0f ends_at=%s decision=%s",
        paper.paper_id, paper.state, phase_window.phase,
        phase_window.seconds_left, phase_window.ends_at.isoformat(), window_decision,
    )

    if phase_window.seconds_left <= 0:
        return {
            "paper_id": paper.paper_id,
            "reactive_status": "window_skip",
            "reactive_reason": "window_closed",
            "reactive_artifact": None,
            "reactive_live_posted": False,
            "reactive_live_reason": "window_closed",
            "verdict_status": "window_skip",
            "verdict_reason": "window_closed",
            "verdict_artifact": None,
            "verdict_live_submitted": False,
            "verdict_live_reason": "window_closed",
            "has_reactive_candidate": False,
            "reactive_draft_created": False,
            "verdict_eligible": False,
            "verdict_draft_created": False,
            "window_skipped": True,
        }

    reactive_results = analyze_reactive_candidates_for_paper(paper.paper_id, db)
    candidate = select_best_reactive_candidate_for_paper(paper.paper_id, reactive_results, db)

    reactive_status = "none"
    reactive_reason: Optional[str] = None
    reactive_artifact: Optional[str] = None
    reactive_live_posted = False
    reactive_live_reason: Optional[str] = None

    if candidate is None:
        reactive_live_reason = "no_candidate"
    elif db.has_recent_reactive_action_for_comment(paper.paper_id, candidate.comment_id, now):
        reactive_status = "dedup_skipped"
        reactive_reason = "recent_reactive_action"
        reactive_live_reason = "dedup_skipped"
    else:
        env_run_mode = get_run_mode() if not test_mode else "test"
        live_allowed = (
            live_reactive
            and not test_mode
            and env_run_mode == "live"
            and live_budget_remaining > 0
            and live_client is not None
            and comment_phase_ok
        )

        if not live_reactive:
            reactive_live_reason = "live_disabled"
        elif test_mode:
            reactive_live_reason = "live_gate_failed"
        elif env_run_mode != "live":
            reactive_live_reason = "live_gate_failed"
        elif live_budget_remaining <= 0:
            reactive_live_reason = "live_budget_exhausted"
        elif live_client is None:
            reactive_live_reason = "live_gate_failed"
        elif not comment_phase_ok:
            reactive_live_reason = "window_skip"

        if live_allowed:
            try:
                comment_id = plan_and_post_reactive_comment(
                    paper, candidate, live_client, db, karma_remaining, now, test_mode=False
                )
            except KoalaWindowClosedError:
                log.warning(
                    "[window] paper=%s 409_window_closed tool=post_comment", paper.paper_id
                )
                comment_id = None
                reactive_live_reason = "window_closed"
                reactive_status = "window_skip"
            if comment_id is not None:
                reactive_status = "live_posted"
                reactive_artifact = comment_id
                reactive_live_posted = True
                reactive_live_reason = "live_posted"
            elif reactive_live_reason != "window_closed":
                reactive_live_reason = "live_gate_failed"
                reactive_status = "skipped"
        else:
            comment_id = plan_and_post_reactive_comment(
                paper, candidate, client, db, karma_remaining, now, test_mode=test_mode
            )
            if comment_id is not None:
                reactive_status = "dry_run"
                reactive_artifact = comment_id
            else:
                reactive_status = "skipped"

    _rlog_reason = reactive_reason
    if not _rlog_reason and reactive_live_reason not in (None, "no_candidate", "live_posted"):
        _rlog_reason = reactive_live_reason

    if reactive_status == "live_posted":
        log.info("[loop] REACTIVE paper=%s status=live_posted", paper.paper_id)
    elif _rlog_reason:
        log.info(
            "[loop] REACTIVE paper=%s status=%s reason=%s",
            paper.paper_id, reactive_status, _rlog_reason,
        )
    else:
        log.info("[loop] REACTIVE paper=%s status=%s", paper.paper_id, reactive_status)

    # ---------------------------------------------------------------------------
    # Verdict path — dry-run by default; live only when all gates pass.
    # submit_verdict is never called unless live_verdict_allowed is True.
    # ---------------------------------------------------------------------------
    eligibility = evaluate_verdict_eligibility(paper, db, reactive_results)
    verdict_status = "ineligible"
    verdict_reason: Optional[str] = None
    verdict_artifact: Optional[str] = None
    verdict_live_submitted = False
    verdict_live_reason: Optional[str] = None

    if not eligibility.eligible:
        verdict_live_reason = "no_eligible_verdict"
    elif db.has_recent_verdict_action_for_paper(paper.paper_id, now):
        verdict_status = "dedup_skipped"
        verdict_reason = "recent_verdict_action"
        verdict_live_reason = "dedup_skipped"
    else:
        verdict_result = plan_verdict_for_paper(
            paper, db, reactive_results, now, test_mode=test_mode
        )
        verdict_artifact = verdict_result.get("artifact_url")
        verdict_status = "dry_run" if verdict_artifact else "skipped"

        effective_score: Optional[float] = score if score is not None else verdict_result.get("score")
        verdict_score_obj: Optional[VerdictScore] = verdict_result.get("verdict_score")

        env_run_mode_v = get_run_mode() if not test_mode else "test"
        live_verdict_allowed = (
            live_verdict
            and not test_mode
            and env_run_mode_v == "live"
            and allowlisted
            and verdict_live_budget_remaining > 0
            and live_client is not None
            and verdict_artifact is not None
            and verdict_phase_ok
        )

        if not live_verdict:
            verdict_live_reason = "live_disabled"
        elif test_mode:
            verdict_live_reason = "live_gate_failed"
        elif env_run_mode_v != "live":
            verdict_live_reason = "live_gate_failed"
        elif not allowlisted:
            verdict_live_reason = "allowlist_required"
        elif verdict_live_budget_remaining <= 0:
            verdict_live_reason = "live_budget_exhausted"
        elif live_client is None:
            verdict_live_reason = "live_gate_failed"
        elif verdict_artifact is None:
            verdict_live_reason = "no_eligible_verdict"
        elif not verdict_phase_ok:
            verdict_live_reason = "window_skip"

        if live_verdict_allowed:
            try:
                submitted, submit_reason = _submit_live_verdict(
                    paper, db, reactive_results, now, live_client, eligibility,
                    score=effective_score, verdict_score=verdict_score_obj,
                )
            except KoalaWindowClosedError:
                log.warning(
                    "[window] paper=%s 409_window_closed tool=submit_verdict", paper.paper_id
                )
                submitted = False
                submit_reason = "window_closed"
                verdict_live_reason = "window_closed"
            if submitted:
                verdict_status = "live_submitted"
                verdict_live_submitted = True
                verdict_live_reason = "live_submitted"
            elif verdict_live_reason == "window_closed":
                verdict_status = "window_skip"
            else:
                if submit_reason in ("missing_verdict_score", "invalid_verdict_score"):
                    verdict_live_reason = submit_reason
                else:
                    verdict_live_reason = "no_eligible_verdict"

    _vlog_reason = verdict_reason
    if not _vlog_reason and verdict_live_reason not in (None, "no_eligible_verdict", "live_submitted"):
        _vlog_reason = verdict_live_reason

    if verdict_status == "live_submitted":
        log.info("[loop] VERDICT paper=%s status=live_submitted", paper.paper_id)
    elif _vlog_reason:
        log.info(
            "[loop] VERDICT paper=%s status=%s reason=%s",
            paper.paper_id, verdict_status, _vlog_reason,
        )
    else:
        log.info("[loop] VERDICT paper=%s status=%s", paper.paper_id, verdict_status)

    return {
        "paper_id": paper.paper_id,
        "reactive_status": reactive_status,
        "reactive_reason": reactive_reason,
        "reactive_artifact": reactive_artifact,
        "reactive_live_posted": reactive_live_posted,
        "reactive_live_reason": reactive_live_reason,
        "verdict_status": verdict_status,
        "verdict_reason": verdict_reason,
        "verdict_artifact": verdict_artifact,
        "verdict_live_submitted": verdict_live_submitted,
        "verdict_live_reason": verdict_live_reason,
        # Legacy fields — consumed by run_operational_loop counters
        "has_reactive_candidate": candidate is not None,
        "reactive_draft_created": reactive_status == "dry_run",
        "verdict_eligible": eligibility.eligible,
        "verdict_draft_created": verdict_status == "dry_run",
    }


def run_operational_loop(
    db: "KoalaDB",
    now: datetime,
    *,
    paper_ids: Optional[List[str]] = None,
    max_papers: Optional[int] = None,
    karma_remaining: float = 100.0,
    output_dir: "str | Path" = "./workspace/reports",
    test_mode: bool = True,
    live_reactive: bool = False,
    live_verdict: bool = False,
) -> dict:
    """Top-level orchestrator for tracked papers.

    Args:
        db:              local SQLite state store
        now:             current UTC datetime
        paper_ids:       restrict to these paper IDs; None means all tracked papers
        max_papers:      cap the number of papers processed
        karma_remaining: karma budget for comment actions
        output_dir:      directory to write run-summary reports
        test_mode:       if True (default), use stub client; always dry-run
        live_reactive:   if True, allow at most one live reactive post per run
                         (requires KOALA_RUN_MODE=live and test_mode=False)
        live_verdict:    if True, allow at most one live verdict submission per run
                         (requires KOALA_RUN_MODE=live, test_mode=False, and explicit
                         paper_ids as allowlist; more conservative than live_reactive)

    Returns:
        Aggregate counter dict. ``errors`` is a list[dict]; ``errors_count`` is its
        length. ``live_reactive_posts`` and ``live_verdict_submissions`` count actual
        live actions taken this run.
    """
    client = _DryRunClient()

    live_client: Optional[Any] = None
    if (live_reactive or live_verdict) and not test_mode:
        from ..koala.client import KoalaClient
        live_client = KoalaClient()

    # Allowlist gate for verdict live: explicit paper_ids required.
    allowlisted = paper_ids is not None

    papers = db.get_papers(paper_ids)
    papers_seen = len(papers)

    if max_papers is not None:
        papers = papers[:max_papers]

    counters: dict = {
        "papers_seen": papers_seen,
        "papers_processed": 0,
        "reactive_candidates_found": 0,
        "reactive_drafts_created": 0,
        "verdicts_eligible": 0,
        "verdict_drafts_created": 0,
        "live_reactive_posts": 0,
        "live_verdict_submissions": 0,
        "verdict_live_missing_score": 0,
        "verdict_live_invalid_score": 0,
        # Phase 9.5 observability counters
        "reactive_dedup_skipped": 0,
        "reactive_live_eligible": 0,
        "reactive_live_posted": 0,
        "live_budget_exhausted": 0,
        "reactive_live_gate_failed": 0,
        "skipped": 0,
        "window_skipped": 0,
        "errors": [],
        "errors_count": 0,
        "summary_path": "",
    }

    live_budget_used = False
    verdict_live_budget_used = False
    paper_live_results: dict = {}

    for row in papers:
        paper_id = row["paper_id"]
        try:
            paper = _paper_from_row(row)
            live_budget_remaining = 0 if live_budget_used else 1
            verdict_live_budget_remaining = 0 if verdict_live_budget_used else 1
            result = _process_paper(
                paper, client, db, karma_remaining, now, test_mode,
                live_reactive=live_reactive,
                live_budget_remaining=live_budget_remaining,
                live_client=live_client,
                live_verdict=live_verdict,
                verdict_live_budget_remaining=verdict_live_budget_remaining,
                allowlisted=allowlisted,
            )
            counters["papers_processed"] += 1
            if result["has_reactive_candidate"]:
                counters["reactive_candidates_found"] += 1
            if result["reactive_draft_created"]:
                counters["reactive_drafts_created"] += 1

            live_reason = result.get("reactive_live_reason")
            if result.get("reactive_status") == "dedup_skipped":
                counters["reactive_dedup_skipped"] += 1
            if live_reason in ("live_posted", "live_gate_failed", "live_budget_exhausted"):
                counters["reactive_live_eligible"] += 1
            if result.get("reactive_live_posted"):
                live_budget_used = True
                counters["live_reactive_posts"] += 1
                counters["reactive_live_posted"] += 1
            if live_reason == "live_budget_exhausted":
                counters["live_budget_exhausted"] += 1
            if live_reason == "live_gate_failed":
                counters["reactive_live_gate_failed"] += 1

            if result["verdict_eligible"]:
                counters["verdicts_eligible"] += 1
            if result["verdict_draft_created"]:
                counters["verdict_drafts_created"] += 1
            if result.get("verdict_live_submitted"):
                verdict_live_budget_used = True
                counters["live_verdict_submissions"] += 1
            vlive_reason = result.get("verdict_live_reason")
            if vlive_reason == "missing_verdict_score":
                counters["verdict_live_missing_score"] += 1
            elif vlive_reason == "invalid_verdict_score":
                counters["verdict_live_invalid_score"] += 1

            if result.get("window_skipped"):
                counters["window_skipped"] += 1
            elif not result["reactive_draft_created"] and not result["verdict_draft_created"]:
                counters["skipped"] += 1

            paper_live_results[paper_id] = {
                "reactive_live_posted": result.get("reactive_live_posted", False),
                "reactive_live_reason": result.get("reactive_live_reason"),
                "reactive_status": result.get("reactive_status"),
                "verdict_status": result.get("verdict_status"),
                "verdict_live_submitted": result.get("verdict_live_submitted", False),
                "verdict_live_reason": result.get("verdict_live_reason"),
            }
        except Exception as exc:
            log.exception("[loop] ERROR processing paper=%s", paper_id)
            counters["errors"].append({
                "paper_id": paper_id,
                "stage": "process_paper",
                "error": str(exc),
            })
            counters["errors_count"] += 1

    log.info(
        "[loop] DONE papers_seen=%d processed=%d reactive=%d live_r=%d "
        "verdicts=%d live_v=%d errors=%d",
        counters["papers_seen"],
        counters["papers_processed"],
        counters["reactive_drafts_created"],
        counters["live_reactive_posts"],
        counters["verdict_drafts_created"],
        counters["live_verdict_submissions"],
        counters["errors_count"],
    )

    out_dir = Path(output_dir)
    summary = build_run_summary(db, now, paper_ids)
    for entry in summary:
        if entry["paper_id"] in paper_live_results:
            entry.update(paper_live_results[entry["paper_id"]])
    md_path = out_dir / "run_summary.md"
    jsonl_path = out_dir / "run_summary.jsonl"
    write_run_summary_markdown(summary, md_path)
    write_run_summary_jsonl(summary, jsonl_path)
    counters["summary_path"] = str(md_path)

    return counters


def main() -> None:
    """CLI entrypoint: run operational loop and write summary reports.

    Usage:
        python -m gsr_agent.orchestration.operational_loop [--db PATH] [--out DIR]
            [--max-papers N] [--paper-id ID ...] [--live-reactive] [--live-verdict]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="GSR Agent Phase 8–10 — Operational Loop"
    )
    parser.add_argument(
        "--db", default="./workspace/koala_agent.db", help="Path to SQLite database"
    )
    parser.add_argument(
        "--out", default="./workspace/reports", help="Output directory for reports"
    )
    parser.add_argument(
        "--max-papers", type=int, default=None, help="Cap number of papers processed"
    )
    parser.add_argument(
        "--paper-id", action="append", dest="paper_ids",
        help="Restrict to specific paper IDs (repeatable)",
    )
    parser.add_argument(
        "--live-reactive",
        action="store_true",
        default=False,
        help=(
            "Allow at most one live reactive comment per run. "
            "Requires KOALA_RUN_MODE=live and test_mode=False."
        ),
    )
    parser.add_argument(
        "--live-verdict",
        action="store_true",
        default=False,
        help=(
            "Allow at most one live verdict submission per run. "
            "Requires KOALA_RUN_MODE=live, test_mode=False, and --paper-id (allowlist)."
        ),
    )
    args = parser.parse_args()

    ok, failures = run_preflight_checks(
        args.db, args.out,
        live_reactive=args.live_reactive,
        live_verdict=args.live_verdict,
        paper_ids=args.paper_ids,
    )
    if not ok:
        for msg in failures:
            print(f"[preflight] FAIL: {msg}")
        return

    mode = "live" if (args.live_reactive or args.live_verdict) else "dry_run"
    papers_label = "all" if args.paper_ids is None else str(len(args.paper_ids))
    print(
        f"[loop] START mode={mode}"
        f" reactive={args.live_reactive}"
        f" verdict={args.live_verdict}"
        f" papers={papers_label}"
    )

    from ..storage.db import KoalaDB

    db = KoalaDB(args.db)
    try:
        now = datetime.now(timezone.utc)
        counters = run_operational_loop(
            db,
            now,
            paper_ids=args.paper_ids,
            max_papers=args.max_papers,
            output_dir=args.out,
            live_reactive=args.live_reactive,
            live_verdict=args.live_verdict,
        )
    finally:
        db.close()

    print(
        f"[loop] papers_seen={counters['papers_seen']}"
        f" processed={counters['papers_processed']}"
        f" reactive_drafts={counters['reactive_drafts_created']}"
        f" live_posts={counters['live_reactive_posts']}"
        f" verdict_drafts={counters['verdict_drafts_created']}"
        f" live_verdicts={counters['live_verdict_submissions']}"
        f" errors={counters['errors_count']}"
    )
    print(
        f"[loop] live_reactive_posts={counters['live_reactive_posts']}"
        f" eligible={counters['reactive_live_eligible']}"
        f" dedup={counters['reactive_dedup_skipped']}"
        f" budget_exhausted={counters['live_budget_exhausted']}"
        f" gate_failed={counters['reactive_live_gate_failed']}"
    )
    print(
        f"[loop] live_verdict_submissions={counters['live_verdict_submissions']}"
        f" missing_score={counters['verdict_live_missing_score']}"
        f" invalid_score={counters['verdict_live_invalid_score']}"
    )
    print(f"[loop] summary → {counters['summary_path']}")


if __name__ == "__main__":
    main()
