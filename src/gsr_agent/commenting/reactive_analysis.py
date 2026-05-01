"""Phase 5A: reactive fact-check dry-run orchestrator.

Reads citable other-agent comments, extracts challengeable claims via GSR,
verifies them against paper evidence, and produces a draft suggestion.
All output is persisted to DB; NO Koala API writes are made (always dry-run).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from ..adapters.gsr_runner import extract_claims_from_koala_comment, retrieve_and_verify_claims
from ..strategy.heat import paper_heat_band

if TYPE_CHECKING:
    from ..storage.db import KoalaDB

log = logging.getLogger(__name__)

_MIN_REACT_CONFIDENCE = 0.5

_DRY_RUN_HEADER = "[DRY-RUN — not posted]"

# Verdict values that are treated as "this claim is contradicted by the paper".
# GSR currently produces "refuted" via VerificationResponse Literal.
# "contradicted" and "contradiction" appear in historical GSR DB rows and in the
# reporting layer's _normalize_verdict(); we handle them here for robustness.
_CONTRADICTION_VERDICTS = frozenset({"refuted", "contradicted", "contradiction"})


def _is_contradiction_like_verdict(verdict: str) -> bool:
    """Return True when verdict signals the paper contradicts the claim."""
    return (verdict or "").strip().lower() in _CONTRADICTION_VERDICTS


@dataclass
class ReactiveAnalysisResult:
    comment_id: str
    paper_id: str
    recommendation: str  # "react" | "skip" | "unclear"
    claims: List[dict] = field(default_factory=list)
    verifications: List[dict] = field(default_factory=list)
    draft_text: Optional[str] = None
    skip_reason: Optional[str] = None
    thread_id: Optional[str] = None  # source comment's thread (for threaded replies)


def analyze_reactive_opportunity_for_comment(
    comment_id: str,
    comment_text: str,
    paper_id: str,
    *,
    db: Optional["KoalaDB"] = None,
    workspace: Optional[Path] = None,
    aggressive_mode: bool = False,
) -> ReactiveAnalysisResult:
    """Extract and verify claims from a citable comment. Always dry-run.

    Steps:
      1. Extract challengeable claims from comment_text via GSR.
      2. Fast-reject with recommendation=skip when no claims are found.
      3. Verify each claim against paper_chunks evidence.
      4. Compute recommendation: react | skip | unclear | evidence_sparse.
         evidence_sparse is only produced when aggressive_mode=True and all
         verifications returned insufficient_evidence.
      5. Build draft text when recommendation is react or evidence_sparse.
      6. Persist results to db when db is provided.
    """
    claims = extract_claims_from_koala_comment(comment_text, paper_id, workspace)

    if not claims:
        log.info(
            "[comment_decision] paper_id=%s comment=%s path=reactive_short "
            "decision=skip reason=no_claims_extracted",
            paper_id, comment_id,
        )
        result = ReactiveAnalysisResult(
            comment_id=comment_id,
            paper_id=paper_id,
            recommendation="skip",
            skip_reason="no claims extracted",
        )
        if db is not None:
            _persist(db, comment_id, paper_id, result)
        return result

    # Enrich each claim with routing metadata.
    for i, c in enumerate(claims):
        c.setdefault("id", f"phase5a_{paper_id}_{comment_id}_{i}")
        c["paper_id"] = paper_id
        c["review_id"] = comment_id

    verifications = retrieve_and_verify_claims(paper_id, claims, workspace)

    recommendation, skip_reason = _compute_recommendation(verifications)

    # Aggressive mode: when all verifications lack evidence but claims exist,
    # promote to evidence_sparse so the paper isn't silently dropped.
    if (
        aggressive_mode
        and recommendation == "unclear"
        and _all_insufficient_evidence(verifications)
    ):
        recommendation = "evidence_sparse"
        skip_reason = None

    log.info(
        "[comment_decision] paper_id=%s comment=%s path=reactive_short "
        "decision=%s reason=%s claim_count=%d verdict_counts=%s aggressive=%s",
        paper_id, comment_id,
        "post" if recommendation in ("react", "evidence_sparse") else "skip",
        skip_reason or recommendation,
        len(claims), _verdict_counts(verifications), aggressive_mode,
    )

    draft_text: Optional[str] = None
    if recommendation == "react":
        draft_text = _build_draft_text(comment_id, claims, verifications)
    elif recommendation == "evidence_sparse":
        draft_text = _build_evidence_sparse_draft(comment_id, claims)

    result = ReactiveAnalysisResult(
        comment_id=comment_id,
        paper_id=paper_id,
        recommendation=recommendation,
        claims=claims,
        verifications=verifications,
        draft_text=draft_text,
        skip_reason=skip_reason,
    )

    if db is not None:
        _persist(db, comment_id, paper_id, result)

    return result


def analyze_reactive_candidates_for_paper(
    paper_id: str,
    db: "KoalaDB",
    *,
    workspace: Optional[Path] = None,
    aggressive_mode: bool = False,
) -> List[ReactiveAnalysisResult]:
    """Run reactive analysis on every citable other-agent comment for a paper."""
    comments = db.get_citable_other_comments_for_paper(paper_id)
    results = []
    for comment in comments:
        result = analyze_reactive_opportunity_for_comment(
            comment_id=comment["comment_id"],
            comment_text=comment["text"],
            paper_id=paper_id,
            db=db,
            workspace=workspace,
            aggressive_mode=aggressive_mode,
        )
        result.thread_id = comment.get("thread_id")
        results.append(result)
        if aggressive_mode and result.recommendation in ("react", "evidence_sparse"):
            break
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_recommendation(verifications: List[dict]) -> tuple[str, Optional[str]]:
    if not verifications:
        return "skip", "no verifications produced"

    contradicted = [
        v for v in verifications
        if _is_contradiction_like_verdict(v.get("verdict", ""))
        and float(v.get("confidence") or 0) >= _MIN_REACT_CONFIDENCE
    ]
    if contradicted:
        return "react", None

    supported_count = sum(1 for v in verifications if v.get("verdict") == "supported")
    if supported_count == len(verifications):
        return "skip", "all claims supported by paper"

    return "unclear", None


def _all_insufficient_evidence(verifications: List[dict]) -> bool:
    """Return True when every verification returned insufficient_evidence."""
    return bool(verifications) and all(
        v.get("verdict", "") == "insufficient_evidence" for v in verifications
    )


def _build_evidence_sparse_draft(
    comment_id: str,
    claims: List[dict],
) -> str:
    """Build a lightweight draft when claims exist but no paper evidence was retrieved."""
    lines = [
        _DRY_RUN_HEADER,
        f"Claim review (evidence-sparse) for comment {comment_id}:",
        "",
        "The following claims were identified in the reviewed comment but could not be "
        "verified against the paper's evidence base. These points may warrant closer scrutiny:",
        "",
    ]
    for c in claims[:3]:
        claim_text = c.get("claim_text", "")
        if claim_text:
            lines.append(f"- {claim_text}")
    lines.append("")
    return "\n".join(lines).rstrip()


def _build_draft_text(
    comment_id: str,
    claims: List[dict],
    verifications: List[dict],
) -> str:
    claim_by_id = {c["id"]: c for c in claims}
    refuted = [
        v for v in verifications
        if _is_contradiction_like_verdict(v.get("verdict", ""))
        and float(v.get("confidence") or 0) >= _MIN_REACT_CONFIDENCE
    ]
    lines = [
        _DRY_RUN_HEADER,
        f"Reactive fact-check draft for comment {comment_id}:",
        "",
    ]
    for v in refuted:
        claim_id = v.get("claim_id") or v.get("id", "")
        c = claim_by_id.get(claim_id, {})
        claim_text = c.get("claim_text") or v.get("claim_text", "")
        conf = float(v.get("confidence") or 0)
        lines.append(f"Claim: {claim_text}")
        lines.append(f"Verdict: refuted (confidence {conf:.2f})")
        reasoning = v.get("reasoning") or ""
        if reasoning:
            lines.append(f"Reasoning: {reasoning}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _verdict_counts(verifications: List[dict]) -> dict:
    counts: dict = {}
    for v in verifications:
        verd = v.get("verdict", "unknown")
        counts[verd] = counts.get(verd, 0) + 1
    return counts


def _draft_id(comment_id: str) -> str:
    return hashlib.sha256(f"draft:{comment_id}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Phase 5B: runtime selector
# ---------------------------------------------------------------------------

_STRONG_CONTRADICTION_OVERRIDE_CONFIDENCE: float = 0.75


def _max_contradiction_confidence(result: ReactiveAnalysisResult) -> float:
    """Return the highest contradiction confidence among a result's verifications."""
    return max(
        (
            float(v.get("confidence") or 0)
            for v in result.verifications
            if _is_contradiction_like_verdict(v.get("verdict", ""))
        ),
        default=0.0,
    )


def select_best_reactive_candidate(
    results: List[ReactiveAnalysisResult],
    distinct_citable_other_agents: int,
    *,
    aggressive_mode: bool = False,
) -> Optional[ReactiveAnalysisResult]:
    """Select the highest-value reactive candidate from Phase 5A outputs.

    Picks at most one candidate per paper using:
      1. recommendation == "react" (only actionable results considered)
      2. Strongest contradiction confidence across refuted verifications
      3. Heat-band preference: goldilocks/warm return the best candidate;
         cold/crowded/saturated return only when contradiction confidence
         >= _STRONG_CONTRADICTION_OVERRIDE_CONFIDENCE (soft penalty, not hard ban).

    In aggressive_mode, falls back to evidence_sparse candidates when no react
    candidate is available or when a react candidate is suppressed by heat band.

    Args:
        results:                       Phase 5A results for a single paper.
        distinct_citable_other_agents: other-agent citable comment count for
                                       the paper (heat-band input).
        aggressive_mode:               when True, include evidence_sparse as fallback.

    Returns:
        The best ReactiveAnalysisResult with recommendation "react" or
        "evidence_sparse" (aggressive_mode only), or None.
    """
    react_candidates = [r for r in results if r.recommendation == "react"]
    sparse_candidates = (
        [r for r in results if r.recommendation == "evidence_sparse"]
        if aggressive_mode else []
    )

    if not react_candidates:
        return sparse_candidates[0] if sparse_candidates else None

    best = max(react_candidates, key=_max_contradiction_confidence)
    band = paper_heat_band(distinct_citable_other_agents)

    if band in ("goldilocks", "warm"):
        return best

    # cold, crowded, saturated: soft penalty — allow override with strong contradiction
    if _max_contradiction_confidence(best) >= _STRONG_CONTRADICTION_OVERRIDE_CONFIDENCE:
        return best

    # In aggressive mode, fall back to evidence_sparse when react is suppressed by heat
    return sparse_candidates[0] if sparse_candidates else None


def select_best_reactive_candidate_for_paper(
    paper_id: str,
    results: List[ReactiveAnalysisResult],
    db: "KoalaDB",
    *,
    aggressive_mode: bool = False,
) -> Optional[ReactiveAnalysisResult]:
    """Wrapper around select_best_reactive_candidate that reads crowding from DB.

    Fetches the current distinct citable other-agent count via
    db.get_comment_stats(paper_id)["citable_other"], then delegates to
    select_best_reactive_candidate().

    Args:
        paper_id:       the paper to select for
        results:        Phase 5A analysis results for the paper
        db:             local SQLite state store
        aggressive_mode: when True, include evidence_sparse candidates as fallback.

    Returns:
        The best ReactiveAnalysisResult, or None.
    """
    stats = db.get_comment_stats(paper_id)
    return select_best_reactive_candidate(
        results, stats["citable_other"], aggressive_mode=aggressive_mode
    )


# ---------------------------------------------------------------------------
# Internal persistence helpers
# ---------------------------------------------------------------------------

def _persist(
    db: "KoalaDB",
    comment_id: str,
    paper_id: str,
    result: ReactiveAnalysisResult,
) -> None:
    db.clear_phase5a_for_comment(comment_id)

    for c in result.claims:
        db.insert_extracted_claim({
            "claim_id": c["id"],
            "comment_id": comment_id,
            "paper_id": paper_id,
            "claim_text": c.get("claim_text", ""),
            "category": c.get("category"),
            "confidence": c.get("confidence"),
            "challengeability": c.get("challengeability"),
            "binary_question": c.get("binary_question"),
        })

    for v in result.verifications:
        db.insert_claim_verification({
            "verification_id": v.get("id", ""),
            "claim_id": v.get("claim_id") or v.get("id", ""),
            "comment_id": comment_id,
            "paper_id": paper_id,
            "verdict": v.get("verdict", "insufficient_evidence"),
            "confidence": v.get("confidence"),
            "reasoning": v.get("reasoning"),
            "supporting_quote": v.get("supporting_quote"),
            "model_id": v.get("model_id"),
        })

    analysis_summary = {
        "recommendation": result.recommendation,
        "claim_count": len(result.claims),
        "verdict_counts": _verdict_counts(result.verifications),
        "skip_reason": result.skip_reason,
    }

    db.insert_reactive_draft({
        "draft_id": _draft_id(comment_id),
        "comment_id": comment_id,
        "paper_id": paper_id,
        "recommendation": result.recommendation,
        "draft_text": result.draft_text,
        "analysis_json": json.dumps(analysis_summary),
    })
