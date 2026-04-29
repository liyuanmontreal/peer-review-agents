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

Phase 10.5: automatic live verdict (gsr_agent only). --live-verdict-auto bypasses
the explicit paper_ids allowlist. Instead, the loop scans for VERDICT_READY papers
(deliberating phase + prior own comment + ≥3 distinct citeable other-agent comments)
and selects the best candidate automatically. At most ONE verdict is submitted per
loop run. Safety gates from Phase 10 remain active.

Default is always dry-run. submit_verdict is never called without all gates passing.

Public API
----------
run_operational_loop(db, now, *, paper_ids=None, max_papers=None,
                     karma_remaining=100.0, output_dir="./workspace/reports",
                     test_mode=True, live_reactive=False, live_verdict=False,
                     live_verdict_auto=False) -> dict

CLI
---
    python -m gsr_agent.orchestration.operational_loop [--db PATH] [--out DIR]
        [--max-papers N] [--paper-id ID ...] [--live-reactive] [--live-verdict]
        [--live-verdict-auto]
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
from ..commenting.orchestrator import plan_and_post_reactive_comment, plan_and_post_seed_comment
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
from ..rules.timeline import SAFETY_BUFFER_S, MicroPhase, compute_phase_window, get_micro_phase
from ..rules.verdict_assembly import (
    VerdictEligibilityResult,
    build_verdict_draft_for_paper,
    evaluate_verdict_eligibility,
    plan_verdict_for_paper,
    select_distinct_other_agent_citations,
)
from ..rules.verdict_scoring import VerdictScore
from ..strategy.opportunity_manager import (
    CANDIDATE_BUDGET,
    EXTENDED_COMMENT_MAX,
    MIN_VERDICT_CITATIONS,
    OPPORTUNITY_PRIORITY,
    PREFERRED_COMMENT_MIN,
    PREFERRED_COMMENT_MAX,
    SATURATED_COMMENT_THRESHOLD,
    PaperOpportunity,
    classify_paper_opportunity,
)

if TYPE_CHECKING:
    from ..storage.db import KoalaDB

log = logging.getLogger(__name__)

_DEDUP_HOURS: float = 12.0

# Endgame live-action budgets per loop run.
_LIVE_COMMENT_BUDGET: int = 3   # max live reactive/seed posts per loop
_LIVE_VERDICT_BUDGET: int = 2   # max live verdict submissions per loop


def run_preflight_checks(
    db_path: str,
    output_dir: str,
    *,
    live_reactive: bool = False,
    live_verdict: bool = False,
    live_verdict_auto: bool = False,
    paper_ids: Optional[List[str]] = None,
) -> tuple[bool, List[str]]:
    """Validate environment before starting the loop.

    Checks DB parent directory, output directory, and — when any live flag is
    set — KOALA_RUN_MODE, KOALA_API_TOKEN, and GitHub publish configuration.
    For live_verdict, also requires explicit paper_ids. live_verdict_auto does
    not require paper_ids (candidate selection is automatic).

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

    if live_reactive or live_verdict or live_verdict_auto:
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
        abstract=row.get("abstract", ""),
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
    opportunity: Optional["PaperOpportunity"] = None,
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
    _stats_for_log = db.get_comment_stats(paper.paper_id)
    _micro_for_log = get_micro_phase(now, paper.open_time).value
    _has_seed_candidate_log = False
    _recent_seed_log = False
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
    # Seed comment path — when no reactive candidate and paper is in SEED window
    # ---------------------------------------------------------------------------
    seed_draft_created = False
    seed_live_posted = False
    seed_live_reason: Optional[str] = None

    # Allow seed path when micro-phase is SEED_WINDOW even if comment_phase_ok
    # is False (e.g. paper.state == "NEW" from schema default makes
    # compute_phase_window return "expired").
    seed_window_ok = (
        get_micro_phase(now, paper.open_time) == MicroPhase.SEED_WINDOW
        and phase_window.seconds_left > SAFETY_BUFFER_S
    )
    if candidate is None and (comment_phase_ok or seed_window_ok):
        _participated = db.has_prior_participation(paper.paper_id)
        _opp = classify_paper_opportunity(paper, _participated, karma_remaining, now)
        _has_seed_candidate_log = (_opp == PaperOpportunity.SEED)
        if _opp != PaperOpportunity.SEED:
            seed_live_reason = "no_candidate"
        elif db.has_recent_seed_action_for_paper(paper.paper_id, now):
            _recent_seed_log = True
            seed_live_reason = "recent_action"
            log.info(
                "[competition] seed_skipped paper_id=%s reason=recent_action",
                paper.paper_id,
            )
        else:
            env_run_mode_s = get_run_mode() if not test_mode else "test"
            live_seed_allowed = (
                live_reactive
                and not test_mode
                and env_run_mode_s == "live"
                and live_budget_remaining > 0
                and live_client is not None
                and seed_window_ok
            )
            if not live_reactive:
                seed_live_reason = "live_disabled"
            elif test_mode:
                seed_live_reason = "seed_gate_test_mode"
            elif env_run_mode_s != "live":
                seed_live_reason = "seed_gate_run_mode"
            elif live_budget_remaining <= 0:
                seed_live_reason = "seed_gate_budget"
            elif live_client is None:
                seed_live_reason = "seed_gate_no_client"
            elif not seed_window_ok:
                seed_live_reason = "seed_gate_window"

            if live_seed_allowed:
                _seed_id, _skip_reason = None, None
                try:
                    _seed_id, _skip_reason = plan_and_post_seed_comment(
                        paper, live_client, db, karma_remaining, now, test_mode=False
                    )
                except KoalaWindowClosedError:
                    log.warning(
                        "[window] paper=%s 409_window_closed tool=post_seed_comment",
                        paper.paper_id,
                    )
                    seed_live_reason = "window_closed"
                if _seed_id is not None:
                    seed_live_posted = True
                    seed_live_reason = "live_posted"
                    log.info("[competition] seed_live_posted paper_id=%s", paper.paper_id)
                elif seed_live_reason != "window_closed":
                    seed_live_reason = _skip_reason or "live_gate_failed"
                    log.info(
                        "[competition] seed_skipped paper_id=%s reason=%s",
                        paper.paper_id, seed_live_reason,
                    )
            else:
                _seed_id, _skip_reason = plan_and_post_seed_comment(
                    paper, client, db, karma_remaining, now, test_mode=test_mode
                )
                if _seed_id is not None:
                    seed_draft_created = True
                    seed_live_reason = "draft_created"
                    log.info(
                        "[competition] seed_draft_created paper_id=%s", paper.paper_id
                    )
    else:
        seed_live_reason = "no_candidate"

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

    _seed_status_log = (
        "live_posted" if seed_live_posted
        else "draft_created" if seed_draft_created
        else "no_candidate" if seed_live_reason == "no_candidate"
        else "skipped"
    )
    log.info(
        "[competition] decision paper_id=%s opportunity=%s phase=%s micro=%s "
        "total_comments=%s recent_seed=%s has_seed_candidate=%s "
        "seed_status=%s seed_reason=%s reactive_status=%s reactive_reason=%s "
        "verdict_status=%s verdict_reason=%s",
        paper.paper_id,
        opportunity.value if opportunity is not None else "unknown",
        phase_window.phase,
        _micro_for_log,
        _stats_for_log.get("total", 0),
        _recent_seed_log,
        _has_seed_candidate_log,
        _seed_status_log,
        seed_live_reason,
        reactive_status,
        reactive_reason or reactive_live_reason,
        verdict_status,
        verdict_reason or verdict_live_reason,
    )
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
        "seed_draft_created": seed_draft_created,
        "seed_live_posted": seed_live_posted,
        "seed_live_reason": seed_live_reason,
        "opportunity": opportunity,
    }


def _write_verdict_opportunities_report(
    opportunities: list[dict],
    out_dir: Path,
) -> str:
    """Write verdict_opportunities.md; return path string."""
    path = out_dir / "verdict_opportunities.md"
    lines = ["# Verdict Opportunities\n\n"]
    if not opportunities:
        lines.append("No verdict opportunities found this run.\n")
    else:
        lines.append("| paper_id | title | phase | citeable | own_comment | action |\n")
        lines.append("|----------|-------|-------|----------|-------------|--------|\n")
        for opp in opportunities:
            lines.append(
                f"| {opp['paper_id']} | {opp['title'][:40]} | {opp['phase']}"
                f" | {opp['citeable_count']} | {'Y' if opp['has_own_comment'] else 'N'}"
                f" | {opp['recommended_action']} |\n"
            )
    path.write_text("".join(lines))
    return str(path)


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
    live_verdict_auto: bool = False,
) -> dict:
    """Top-level orchestrator for tracked papers.

    Args:
        db:                local SQLite state store
        now:               current UTC datetime
        paper_ids:         restrict to these paper IDs; None means all tracked papers
        max_papers:        cap the number of papers processed
        karma_remaining:   karma budget for comment actions
        output_dir:        directory to write run-summary reports
        test_mode:         if True (default), use stub client; always dry-run
        live_reactive:     if True, allow at most one live reactive post per run
                           (requires KOALA_RUN_MODE=live and test_mode=False)
        live_verdict:      if True, allow at most one live verdict submission per run
                           (requires KOALA_RUN_MODE=live, test_mode=False, and explicit
                           paper_ids as allowlist; more conservative than live_reactive)
        live_verdict_auto: if True, automatically select and submit at most one verdict
                           per run without requiring explicit paper_ids. Candidate must
                           be in deliberating phase with prior own comment and ≥3 distinct
                           citeable other-agent comments. (gsr_agent only)

    Returns:
        Aggregate counter dict. ``errors`` is a list[dict]; ``errors_count`` is its
        length. ``live_reactive_posts`` and ``live_verdict_submissions`` count actual
        live actions taken this run.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = _DryRunClient()

    live_client: Optional[Any] = None
    if (live_reactive or live_verdict or live_verdict_auto) and not test_mode:
        from ..koala.client import KoalaClient
        live_client = KoalaClient()
        from ..koala.sync import sync_all_active_state
        sync_all_active_state(live_client, db)

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
        "seed_drafts_created": 0,
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
        # Seed decision counters
        "seed_posted": 0,
        "no_seed_candidate": 0,
        "seed_skipped_recent_action": 0,
        "seed_skipped_low_signal": 0,
        "seed_skipped_missing_abstract": 0,
        "seed_skipped_moderation": 0,
        "seed_skipped_live_gate": 0,
        "errors": [],
        "errors_count": 0,
        "summary_path": "",
    }

    live_comments_used = 0
    verdict_submissions_used = 0
    paper_live_results: dict = {}

    # Pre-classify all papers once for priority-based selection.
    _classified = []
    for _row in papers:
        _p = _paper_from_row(_row)
        _participated = db.has_prior_participation(_p.paper_id)
        _opp = classify_paper_opportunity(_p, _participated, karma_remaining, now)
        _stats = db.get_comment_stats(_p.paper_id)
        _recent_seed = (
            _opp == PaperOpportunity.SEED
            and db.has_recent_seed_action_for_paper(_p.paper_id, now)
        )
        _classified.append((_row, _p, _opp, _stats, _recent_seed))

    # Verdict opportunity scan: deliberating + VERDICT_READY papers.
    _verdict_opportunities: list[dict] = []
    for _r, _p, _o, _s, _ in _classified:
        _is_verdict_phase = (
            _o == PaperOpportunity.VERDICT_READY
            or _p.state == "deliberating"
            or _p.deliberating_at is not None
        )
        if not _is_verdict_phase:
            continue
        _citable_n = _s.get("citable_other", 0)
        _has_own = db.has_prior_participation(_p.paper_id)
        if _citable_n < MIN_VERDICT_CITATIONS:
            log.info(
                "[competition] verdict_blocked paper_id=%s reason=insufficient_verdict_citations count=%d required=%d",
                _p.paper_id, _citable_n, MIN_VERDICT_CITATIONS,
            )
        elif not _has_own:
            log.info(
                "[competition] verdict_blocked paper_id=%s reason=no_prior_own_comment",
                _p.paper_id,
            )
        else:
            log.info(
                "[competition] verdict_ready paper_id=%s citeable_comments=%d ours=Y",
                _p.paper_id, _citable_n,
            )
            _verdict_opportunities.append({
                "paper_id": _p.paper_id,
                "title": _p.title,
                "phase": _p.state,
                "citeable_count": _citable_n,
                "has_own_comment": True,
                "recommended_action": "submit_verdict",
            })

    _verdict_valid = [
        (_r, _p, _o, _s)
        for _r, _p, _o, _s, _ in _classified
        if _o == PaperOpportunity.VERDICT_READY
        and _s.get("citable_other", 0) >= MIN_VERDICT_CITATIONS
    ]
    log.info(
        "[competition] verdict_scan papers=%d candidates=%d",
        len(papers), len(_verdict_valid),
    )
    if _verdict_valid:
        _best_v = _verdict_valid[0]
        log.info(
            "[competition] selected_verdict_candidate paper_id=%s citeable_comments=%d",
            _best_v[1].paper_id,
            _best_v[3].get("citable_other", 0),
        )
    else:
        log.info("[competition] no_viable_verdict_candidates")
    _write_verdict_opportunities_report(_verdict_opportunities, out_dir)

    # Phase 10.5: auto-verdict candidate selection (gsr_agent only).
    # Candidates already satisfy: deliberating phase + own comment + ≥MIN_VERDICT_CITATIONS.
    _auto_verdict_paper_id: Optional[str] = None
    if live_verdict_auto:
        if _verdict_opportunities:
            _best_auto = _verdict_opportunities[0]
            _auto_verdict_paper_id = _best_auto["paper_id"]
            log.info(
                "[competition] selected_auto_verdict paper_id=%s citeable_comments=%d",
                _auto_verdict_paper_id, _best_auto["citeable_count"],
            )
        else:
            log.info("[competition] auto_verdict_skipped reason=no_verdict_ready_candidates")

    # Sort by opportunity priority; SEED papers further sorted by 4-tier crowding rank.
    # Tier 0 (best): 3–10 other comments — highest verdict-conversion probability.
    # Tier 1: 11–14 — still eligible, above fallback.
    # Tier 2: 1–2 — some social proof, below preferred.
    # Tier 3 (worst): 0 — cold-start, lowest fallback.
    def _sort_key(item):
        _, _p, _o, _s, _recent = item
        _base = OPPORTUNITY_PRIORITY[_o]
        if _o == PaperOpportunity.SEED:
            _n = _s.get("total", 0)
            if _n > SATURATED_COMMENT_THRESHOLD:
                return (OPPORTUNITY_PRIORITY[PaperOpportunity.SKIP], 0, 0)
            _r = 1 if _recent else 0
            if PREFERRED_COMMENT_MIN <= _n <= PREFERRED_COMMENT_MAX:
                return (_base, _r, 0)  # tier 1: 3–10 (highest)
            if PREFERRED_COMMENT_MAX < _n <= EXTENDED_COMMENT_MAX:
                return (_base, _r, 1)  # tier 2: 11–14
            if 1 <= _n < PREFERRED_COMMENT_MIN:
                return (_base, _r, 2)  # tier 3: 1–2
            return (_base, _r, 3)      # tier 4: cold-start (0)
        return (_base, 0, 0)

    _sorted = sorted(_classified, key=_sort_key)
    _opp_by_id = {_p.paper_id: _o for _, _p, _o, _, _ in _classified}

    # Filter SEED papers by crowding thresholds and apply candidate budget.
    _candidates: list = []
    _cold_start_admitted = False
    _has_nonzero_seed_candidate = False
    for _r, _p, _o, _s, _ in _sorted:
        if _o == PaperOpportunity.SKIP:
            continue
        _n = _s.get("total", 0)
        if _o == PaperOpportunity.SEED:
            if _n > SATURATED_COMMENT_THRESHOLD:
                log.info(
                    "[competition] saturated_comments paper_id=%s comment_count=%d",
                    _p.paper_id, _n,
                )
                continue
            if _n == 0:
                _abstract = (_p.abstract or "").strip()
                if (
                    _cold_start_admitted
                    or _has_nonzero_seed_candidate
                    or not _abstract
                    or len(_abstract) < 40
                    or len(_abstract.split()) < 6
                ):
                    log.info(
                        "[competition] skip_too_cold paper_id=%s comment_count=%d",
                        _p.paper_id, _n,
                    )
                    continue
                _cold_start_admitted = True
                log.info("[competition] cold_start_seed_fallback paper_id=%s", _p.paper_id)
                log.info(
                    "[competition] selected_cold_start_seed_candidate paper_id=%s", _p.paper_id,
                )
            else:
                _has_nonzero_seed_candidate = True
                if PREFERRED_COMMENT_MIN <= _n <= PREFERRED_COMMENT_MAX:
                    log.info(
                        "[competition] seed_candidate_preferred_band paper_id=%s comment_count=%d",
                        _p.paper_id, _n,
                    )
                log.info("[competition] selected_seed_candidate paper_id=%s", _p.paper_id)
        _candidates.append(_r)

    _inspected = min(len(_candidates), CANDIDATE_BUDGET)
    log.info("[competition] candidate_budget inspected=%d max=%d", _inspected, CANDIDATE_BUDGET)
    papers = _candidates[:CANDIDATE_BUDGET]

    # Ensure auto-verdict candidate is in the processing list even if outside CANDIDATE_BUDGET.
    if _auto_verdict_paper_id is not None:
        if not any(r["paper_id"] == _auto_verdict_paper_id for r in papers):
            _auto_row = next(
                (_r for _r, _p, _o, _s, _ in _classified if _p.paper_id == _auto_verdict_paper_id),
                None,
            )
            if _auto_row is not None:
                papers = [_auto_row, *papers]

    for row in papers:
        paper_id = row["paper_id"]
        try:
            paper = _paper_from_row(row)
            live_budget_remaining = max(0, _LIVE_COMMENT_BUDGET - live_comments_used)
            verdict_live_budget_remaining = max(0, _LIVE_VERDICT_BUDGET - verdict_submissions_used)
            _is_auto_candidate = live_verdict_auto and paper_id == _auto_verdict_paper_id
            _paper_live_verdict = live_verdict or _is_auto_candidate
            _paper_allowlisted = allowlisted or _is_auto_candidate
            result = _process_paper(
                paper, client, db, karma_remaining, now, test_mode,
                opportunity=_opp_by_id.get(paper_id),
                live_reactive=live_reactive,
                live_budget_remaining=live_budget_remaining,
                live_client=live_client,
                live_verdict=_paper_live_verdict,
                verdict_live_budget_remaining=verdict_live_budget_remaining,
                allowlisted=_paper_allowlisted,
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
                live_comments_used += 1
                counters["live_reactive_posts"] += 1
                counters["reactive_live_posted"] += 1
            if result.get("seed_draft_created"):
                counters["seed_drafts_created"] += 1
            if result.get("seed_live_posted"):
                live_comments_used += 1
                counters["live_reactive_posts"] += 1
            _seed_reason = result.get("seed_live_reason")
            if _seed_reason == "live_posted":
                counters["seed_posted"] += 1
            elif _seed_reason == "no_candidate":
                counters["no_seed_candidate"] += 1
            elif _seed_reason == "recent_action":
                counters["seed_skipped_recent_action"] += 1
            elif _seed_reason in ("seed_plan_low_signal", "seed_plan_empty_draft"):
                counters["seed_skipped_low_signal"] += 1
            elif _seed_reason == "seed_plan_missing_abstract":
                counters["seed_skipped_missing_abstract"] += 1
            elif _seed_reason == "seed_plan_moderation_low_effort":
                counters["seed_skipped_moderation"] += 1
            elif _seed_reason not in (None, "draft_created"):
                counters["seed_skipped_live_gate"] += 1
            if live_reason == "live_budget_exhausted":
                counters["live_budget_exhausted"] += 1
            if live_reason == "live_gate_failed":
                counters["reactive_live_gate_failed"] += 1

            if result["verdict_eligible"]:
                counters["verdicts_eligible"] += 1
            if result["verdict_draft_created"]:
                counters["verdict_drafts_created"] += 1
            if result.get("verdict_live_submitted"):
                verdict_submissions_used += 1
                counters["live_verdict_submissions"] += 1
            vlive_reason = result.get("verdict_live_reason")
            if vlive_reason == "missing_verdict_score":
                counters["verdict_live_missing_score"] += 1
            elif vlive_reason == "invalid_verdict_score":
                counters["verdict_live_invalid_score"] += 1

            if _is_auto_candidate:
                if result.get("verdict_live_submitted"):
                    log.info("[competition] auto_verdict_submitted paper_id=%s", paper_id)
                else:
                    _auto_skip_reason = vlive_reason or "unknown"
                    log.info(
                        "[competition] auto_verdict_skipped reason=%s", _auto_skip_reason
                    )

            if result.get("window_skipped"):
                counters["window_skipped"] += 1
            elif (
                not result["reactive_draft_created"]
                and not result["verdict_draft_created"]
                and not result.get("seed_draft_created")
                and not result.get("seed_live_posted")
            ):
                counters["skipped"] += 1

            paper_live_results[paper_id] = {
                "reactive_live_posted": result.get("reactive_live_posted", False),
                "reactive_live_reason": result.get("reactive_live_reason"),
                "reactive_status": result.get("reactive_status"),
                "verdict_status": result.get("verdict_status"),
                "verdict_live_submitted": result.get("verdict_live_submitted", False),
                "verdict_live_reason": result.get("verdict_live_reason"),
                "seed_draft_created": result.get("seed_draft_created", False),
                "seed_live_posted": result.get("seed_live_posted", False),
                "seed_live_reason": result.get("seed_live_reason"),
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
    log.info(
        "[loop] seed_summary seed_posted=%d no_candidate=%d "
        "skipped_recent=%d skipped_low_signal=%d skipped_missing_abstract=%d "
        "skipped_moderation=%d skipped_live_gate=%d",
        counters["seed_posted"],
        counters["no_seed_candidate"],
        counters["seed_skipped_recent_action"],
        counters["seed_skipped_low_signal"],
        counters["seed_skipped_missing_abstract"],
        counters["seed_skipped_moderation"],
        counters["seed_skipped_live_gate"],
    )

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
    parser.add_argument(
        "--live-verdict-auto",
        action="store_true",
        default=False,
        help=(
            "Automatically select and submit at most one live verdict per run "
            "without requiring --paper-id. Candidate must be in deliberating phase "
            "with prior own comment and ≥3 distinct citeable other-agent comments. "
            "Requires KOALA_RUN_MODE=live and test_mode=False. gsr_agent only."
        ),
    )
    args = parser.parse_args()

    ok, failures = run_preflight_checks(
        args.db, args.out,
        live_reactive=args.live_reactive,
        live_verdict=args.live_verdict,
        live_verdict_auto=args.live_verdict_auto,
        paper_ids=args.paper_ids,
    )
    if not ok:
        for msg in failures:
            print(f"[preflight] FAIL: {msg}")
        return

    mode = "live" if (args.live_reactive or args.live_verdict or args.live_verdict_auto) else "dry_run"
    papers_label = "all" if args.paper_ids is None else str(len(args.paper_ids))
    print(
        f"[loop] START mode={mode}"
        f" reactive={args.live_reactive}"
        f" verdict={args.live_verdict}"
        f" verdict_auto={args.live_verdict_auto}"
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
            test_mode=not (args.live_reactive or args.live_verdict or args.live_verdict_auto),
            live_reactive=args.live_reactive,
            live_verdict=args.live_verdict,
            live_verdict_auto=args.live_verdict_auto,
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
