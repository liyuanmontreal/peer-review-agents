"""Phase 6: Verdict Assembly v0 — dry-run-only verdict draft pipeline.

Produces a deterministic markdown verdict draft and persists it as an artifact.
NEVER submits a live verdict to the Koala API; always dry-run.

Eligibility logic (Phase 6.5 — two-gate design)
-------------------------------------------------
Gate 1 — heat-band + strong reactive signal (evaluate_verdict_eligibility):
  The `eligible` flag is True when heat-band and reactive signal together
  suggest a verdict attempt is worth making. Saturated is always False.

Gate 2 — citation correctness (select_distinct_other_agent_citations):
  A draft is only valid when >= MIN_DISTINCT_OTHER_AGENTS (3) distinct
  other-agent citable comments exist. Duplicate comments from the same agent
  count as one citation. Self-comments are excluded at the DB layer.

If Gate 1 passes but Gate 2 fails, plan_verdict_for_paper logs SKIP with
reason=insufficient_distinct_other_agent_citations and produces no artifact.

Run mode
--------
Phase 6 is unconditionally dry-run. plan_verdict_for_paper always returns
without calling the Koala API, regardless of KOALA_RUN_MODE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from ..artifacts.github import publish_verdict_artifact
from ..commenting.reactive_analysis import ReactiveAnalysisResult
from ..rules.verdict_eligibility import MIN_DISTINCT_OTHER_AGENTS
from ..rules.verdict_scoring import VerdictScore, score_verdict_heuristic_v0
from ..strategy.heat import paper_heat_band

if TYPE_CHECKING:
    from ..koala.models import Paper
    from ..storage.db import KoalaDB

log = logging.getLogger(__name__)

_STRONG_SIGNAL_THRESHOLD: float = 0.75
_CONTRADICTION_VERDICTS = frozenset({"refuted", "contradicted", "contradiction"})


def _is_contradiction(verdict: str) -> bool:
    return (verdict or "").strip().lower() in _CONTRADICTION_VERDICTS


def _max_react_confidence(result: ReactiveAnalysisResult) -> float:
    return max(
        (
            float(v.get("confidence") or 0)
            for v in result.verifications
            if _is_contradiction(v.get("verdict", ""))
        ),
        default=0.0,
    )


@dataclass
class VerdictEligibilityResult:
    """Output of evaluate_verdict_eligibility (Gate 1)."""

    eligible: bool
    reason_code: str  # "eligible" | "no_react_signal" | "cold_no_override" | "crowded_no_override" | "saturated_low_value_v0"
    heat_band: str
    distinct_citable_other_agents: int
    strongest_contradiction_confidence: Optional[float]
    selected_candidates: List[ReactiveAnalysisResult] = field(default_factory=list)


def evaluate_verdict_eligibility(
    paper: "Paper",
    db: "KoalaDB",
    reactive_results: List[ReactiveAnalysisResult],
) -> VerdictEligibilityResult:
    """Gate 1: heat-band + strong reactive signal eligibility check.

    Args:
        paper:            the paper to evaluate
        db:               local SQLite state store (provides citable_other count)
        reactive_results: Phase 5A analysis results for the paper

    Returns:
        VerdictEligibilityResult with eligible flag, reason code, and
        selected candidates (top-3 react results) when eligible.
    """
    stats = db.get_comment_stats(paper.paper_id)
    n = stats["citable_other"]
    band = paper_heat_band(n)

    react_candidates = [r for r in reactive_results if r.recommendation == "react"]
    if react_candidates:
        strongest_conf: Optional[float] = max(_max_react_confidence(r) for r in react_candidates)
    else:
        strongest_conf = None

    strong_signal = strongest_conf is not None and strongest_conf >= _STRONG_SIGNAL_THRESHOLD

    if band == "saturated":
        return VerdictEligibilityResult(
            eligible=False,
            reason_code="saturated_low_value_v0",
            heat_band=band,
            distinct_citable_other_agents=n,
            strongest_contradiction_confidence=strongest_conf,
            selected_candidates=[],
        )

    if not strong_signal:
        if band == "cold":
            reason_code = "cold_no_override"
        elif band == "crowded":
            reason_code = "crowded_no_override"
        else:
            reason_code = "no_react_signal"
        return VerdictEligibilityResult(
            eligible=False,
            reason_code=reason_code,
            heat_band=band,
            distinct_citable_other_agents=n,
            strongest_contradiction_confidence=strongest_conf,
            selected_candidates=[],
        )

    return VerdictEligibilityResult(
        eligible=True,
        reason_code="eligible",
        heat_band=band,
        distinct_citable_other_agents=n,
        strongest_contradiction_confidence=strongest_conf,
        selected_candidates=react_candidates[:3],
    )


def select_distinct_other_agent_citations(
    paper_id: str,
    db: "KoalaDB",
    min_count: int = MIN_DISTINCT_OTHER_AGENTS,
) -> List[dict]:
    """Gate 2: select one citation per distinct other agent, stably ordered by comment_id.

    Relies on db.get_citable_other_comments_for_paper which already excludes
    self-comments (is_ours=0) and non-citable comments (is_citable=1).
    When an agent has multiple comments, the earliest by created_at is chosen
    (the DB method returns rows ordered by created_at).
    Comments with a missing or empty author_agent_id are excluded.

    Args:
        paper_id:  the paper whose citations to select
        db:        local SQLite state store
        min_count: minimum distinct agents required (default: MIN_DISTINCT_OTHER_AGENTS=3)

    Returns:
        List of selected comment dicts sorted by comment_id when distinct-agent
        count >= min_count. Empty list when the count is insufficient.
    """
    all_comments = db.get_citable_other_comments_for_paper(paper_id)
    seen_agents: set[str] = set()
    selected: List[dict] = []
    for c in all_comments:
        agent_id = c.get("author_agent_id") or ""
        if agent_id and agent_id not in seen_agents:
            seen_agents.add(agent_id)
            selected.append(c)
    if len(selected) < min_count:
        return []
    return sorted(selected, key=lambda c: c.get("comment_id", ""))


_BAND_RATIONALE: dict[str, str] = {
    "cold": "Cold paper — low activity, but strong reactive signal overrides soft penalty.",
    "warm": "Warm paper — moderate activity; strong reactive signal detected.",
    "goldilocks": "Goldilocks paper — optimal participation level for verdict.",
    "crowded": "Crowded paper — high activity, but strong reactive signal overrides soft penalty.",
}


def build_verdict_draft_for_paper(
    paper: "Paper",
    eligibility: VerdictEligibilityResult,
    reactive_results: List[ReactiveAnalysisResult],
    db: "KoalaDB",
    now: datetime,
    *,
    valid_citations: Optional[List[dict]] = None,
    verdict_score: Optional[VerdictScore] = None,
) -> Optional[str]:
    """Build a deterministic markdown verdict draft. No LLM calls.

    Args:
        paper:            the paper being evaluated
        eligibility:      output of evaluate_verdict_eligibility
        reactive_results: Phase 5A results (used for evidence detail)
        db:               local SQLite state store (used only when valid_citations is None)
        now:              current UTC datetime (for timestamp in draft)
        valid_citations:  pre-selected citation dicts from select_distinct_other_agent_citations.
                          When None, the helper is called internally.

    Returns:
        Markdown string, or None if fewer than MIN_DISTINCT_OTHER_AGENTS
        distinct citable other-agent comments are available.
    """
    citations = valid_citations
    if citations is None:
        citations = select_distinct_other_agent_citations(paper.paper_id, db)
    if not citations:
        return None

    n = eligibility.distinct_citable_other_agents
    lines = [
        "# Verdict Draft (DRY-RUN — not submitted)",
        f"Paper: {paper.paper_id} — {paper.title}",
        f"Generated: {now.isoformat()}",
        f"Heat band: {eligibility.heat_band}",
        f"Distinct citable other agents: {n}",
        "",
        "## Why this paper is eligible",
        f"- {_BAND_RATIONALE.get(eligibility.heat_band, eligibility.heat_band)}",
    ]
    if eligibility.strongest_contradiction_confidence is not None:
        lines.append(
            f"- Strongest contradiction confidence: "
            f"{eligibility.strongest_contradiction_confidence:.2f}"
        )
    lines.append("")

    lines.append("## Citable other-agent comments (3 minimum required for submission)")
    for c in citations:
        agent = c.get("author_agent_id", "unknown")
        lines.append(f"- Comment {c['comment_id']} by agent {agent}")
    lines.append("")

    lines.append("## GSR reactive evidence summary")
    if eligibility.selected_candidates:
        for r in eligibility.selected_candidates:
            conf = _max_react_confidence(r)
            lines.append(
                f"- Comment {r.comment_id}: refuted claim detected (confidence {conf:.2f})"
            )
    else:
        lines.append("- No refuted claims detected.")
    lines.append("")

    if verdict_score is not None:
        lines.append("## Verdict Score")
        lines.append(f"- Score: {verdict_score.score:.1f} / 10.0")
        lines.append(f"- Source: {verdict_score.score_source}")
        lines.append(f"- Confidence: {verdict_score.confidence:.4f}")
        lines.append(f"- Rationale: {verdict_score.rationale}")
        lines.append("")

    lines.append("## Proposed verdict rationale")
    lines.append(
        "- Reactive evidence suggests at least one claim in citable other-agent "
        "comments is contradicted by the paper."
    )
    lines.append("- Manual review and score assignment required before any live submission.")
    lines.append("")

    lines.append("## Suggested next step")
    lines.append("- Manual review required before any live verdict submission.")

    return "\n".join(lines)


def plan_verdict_for_paper(
    paper: "Paper",
    db: "KoalaDB",
    reactive_results: List[ReactiveAnalysisResult],
    now: datetime,
    *,
    test_mode: bool = False,
) -> dict:
    """Evaluate eligibility, build a verdict draft artifact, and persist it.

    Phase 6 is always dry-run: no Koala API call is made regardless of
    KOALA_RUN_MODE or test_mode.

    Steps:
      1. Gate 1 — heat-band + reactive signal (evaluate_verdict_eligibility).
         If not eligible: log SKIP and return status="skipped".
      2. Gate 2 — citation correctness (select_distinct_other_agent_citations).
         If < MIN_DISTINCT_OTHER_AGENTS distinct other agents: log SKIP
         reason=insufficient_distinct_other_agent_citations.
      3. Build deterministic markdown draft (build_verdict_draft_for_paper).
      4. Publish artifact (always test_mode=True — never a real GitHub push).
      5. Log dry_run action in DB.
      6. Return result dict with artifact_url and status="dry_run".

    Args:
        paper:            the paper to evaluate
        db:               local SQLite state store
        reactive_results: Phase 5A analysis results for the paper
        now:              current UTC datetime
        test_mode:        reserved for unit-test callers; does not affect
                          dry-run behaviour (Phase 6 is always dry-run)

    Returns:
        dict with keys: paper_id, eligible, reason_code, heat_band,
        distinct_citable_other_agents, artifact_url, status.
    """
    log.info("[verdict] START paper=%s", paper.paper_id)

    eligibility = evaluate_verdict_eligibility(paper, db, reactive_results)

    if not eligibility.eligible:
        log.info(
            "[verdict] SKIP reason=%s heat=%s citable_other=%d",
            eligibility.reason_code,
            eligibility.heat_band,
            eligibility.distinct_citable_other_agents,
        )
        return {
            "paper_id": paper.paper_id,
            "eligible": False,
            "reason_code": eligibility.reason_code,
            "heat_band": eligibility.heat_band,
            "distinct_citable_other_agents": eligibility.distinct_citable_other_agents,
            "artifact_url": None,
            "status": "skipped",
            "score": None,
            "verdict_score": None,
        }

    log.info(
        "[verdict] ELIGIBLE heat=%s citable_other=%d strongest_conf=%.2f",
        eligibility.heat_band,
        eligibility.distinct_citable_other_agents,
        eligibility.strongest_contradiction_confidence or 0.0,
    )

    citations = select_distinct_other_agent_citations(paper.paper_id, db)
    if not citations:
        log.info(
            "[verdict] SKIP reason=insufficient_distinct_other_agent_citations "
            "heat=%s citable_other=%d",
            eligibility.heat_band,
            eligibility.distinct_citable_other_agents,
        )
        return {
            "paper_id": paper.paper_id,
            "eligible": False,
            "reason_code": "insufficient_distinct_other_agent_citations",
            "heat_band": eligibility.heat_band,
            "distinct_citable_other_agents": eligibility.distinct_citable_other_agents,
            "artifact_url": None,
            "status": "skipped",
            "score": None,
            "verdict_score": None,
        }

    verdict_score = score_verdict_heuristic_v0(paper, reactive_results, db)
    log.info(
        "[verdict] SCORE paper=%s score=%.1f source=%s confidence=%.4f",
        paper.paper_id,
        verdict_score.score,
        verdict_score.score_source,
        verdict_score.confidence,
    )

    draft = build_verdict_draft_for_paper(
        paper, eligibility, reactive_results, db, now,
        valid_citations=citations, verdict_score=verdict_score,
    )
    if draft is None:
        log.info(
            "[verdict] SKIP reason=insufficient_distinct_other_agent_citations paper=%s",
            paper.paper_id,
        )
        return {
            "paper_id": paper.paper_id,
            "eligible": False,
            "reason_code": "insufficient_distinct_other_agent_citations",
            "heat_band": eligibility.heat_band,
            "distinct_citable_other_agents": eligibility.distinct_citable_other_agents,
            "artifact_url": None,
            "status": "skipped",
            "score": None,
            "verdict_score": None,
        }

    cited_ids = [c["comment_id"] for c in citations]
    artifact_url = publish_verdict_artifact(
        paper.paper_id,
        score=verdict_score.score,
        body=draft,
        cited_ids=cited_ids,
        test_mode=True,
    )

    log.info("[verdict] DRY_RUN artifact=%s", artifact_url)

    db.log_action(
        paper_id=paper.paper_id,
        action_type="verdict_draft",
        github_file_url=artifact_url,
        status="dry_run",
        details={
            "reason_code": eligibility.reason_code,
            "heat_band": eligibility.heat_band,
            "distinct_citable_other_agents": eligibility.distinct_citable_other_agents,
            "strongest_contradiction_confidence": eligibility.strongest_contradiction_confidence,
            "selected_candidate_ids": [c.comment_id for c in eligibility.selected_candidates],
            "cited_comment_ids": cited_ids,
            "artifact_url": artifact_url,
            "is_test_url": True,
            "publish_status": "dry_run",
            "blocked_reason": "Phase 6 is always dry-run; live verdict submission not implemented.",
        },
    )

    log.info("[verdict] DONE paper=%s", paper.paper_id)

    return {
        "paper_id": paper.paper_id,
        "eligible": True,
        "reason_code": eligibility.reason_code,
        "heat_band": eligibility.heat_band,
        "distinct_citable_other_agents": eligibility.distinct_citable_other_agents,
        "artifact_url": artifact_url,
        "status": "dry_run",
        "score": verdict_score.score,
        "verdict_score": verdict_score,
    }
