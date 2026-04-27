"""Phase A MVP: heuristic_v0 verdict score source.

Conservative score in [3.0, 7.0] derived from reactive analysis evidence.
No LLM calls. Extremes (<3.0, >8.0) are reserved for future evidence modules.

Public API
----------
score_verdict_heuristic_v0(paper, reactive_results, db) -> VerdictScore
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..commenting.reactive_analysis import ReactiveAnalysisResult
    from ..koala.models import Paper
    from ..storage.db import KoalaDB

_STRONG_CONF: float = 0.85
_MEDIUM_CONF: float = 0.65


@dataclass(frozen=True)
class VerdictScore:
    """Verdict score with provenance metadata."""

    score: float       # [0.0, 10.0]; heuristic_v0 always in [3.0, 7.0]
    score_source: str  # "heuristic_v0"
    confidence: float  # [0.0, 1.0]
    rationale: str


def score_verdict_heuristic_v0(
    paper: "Paper",
    reactive_results: "List[ReactiveAnalysisResult]",
    db: "KoalaDB",
) -> VerdictScore:
    """Conservative heuristic verdict score from reactive analysis evidence.

    Tiers (no extremes below 3.0 or above 8.0 without future evidence modules):
      conf >= 0.85  → score 3.5  (strong contradiction, weak-reject)
      conf 0.65–0.84 → score 4.5  (moderate contradiction, borderline)
      no react + unclear > 0 → score 5.5  (ambiguous signals, neutral)
      otherwise     → score 6.5  (no contradiction found, weak-accept)

    Args:
        paper:            the paper being scored
        reactive_results: Phase 5A analysis results (unused directly; DB is primary source)
        db:               local SQLite state store

    Returns:
        VerdictScore with score, score_source, confidence, and rationale.
    """
    paper_id = paper.paper_id
    strongest_conf = db.get_strongest_contradiction_confidence(paper_id)
    stats = db.get_phase5a_stats(paper_id)

    if strongest_conf is not None and strongest_conf >= _STRONG_CONF:
        return VerdictScore(
            score=3.5,
            score_source="heuristic_v0",
            confidence=round(strongest_conf, 4),
            rationale=(
                f"Strong contradiction signal detected (confidence {strongest_conf:.2f}); "
                "high-confidence evidence challenges core claims — conservative weak-reject."
            ),
        )

    if strongest_conf is not None and strongest_conf >= _MEDIUM_CONF:
        return VerdictScore(
            score=4.5,
            score_source="heuristic_v0",
            confidence=round(strongest_conf, 4),
            rationale=(
                f"Moderate contradiction signal detected (confidence {strongest_conf:.2f}); "
                "evidence challenges claims without conclusive refutation — borderline."
            ),
        )

    react_count = stats["react_count"]
    unclear_count = stats["unclear_count"]

    if react_count == 0 and unclear_count > 0:
        return VerdictScore(
            score=5.5,
            score_source="heuristic_v0",
            confidence=0.40,
            rationale=(
                f"Ambiguous reviewer signals ({unclear_count} unclear, 0 refuted); "
                "insufficient evidence for a directional verdict — neutral."
            ),
        )

    return VerdictScore(
        score=6.5,
        score_source="heuristic_v0",
        confidence=0.35,
        rationale=(
            "No significant contradicting evidence found in reviewer comments; "
            "paper appears generally supported — conservative weak-accept."
        ),
    )
