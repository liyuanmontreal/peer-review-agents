"""Phase 6 / 6.5: Verdict Assembly — eligibility, citation correctness, draft, orchestration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from gsr_agent.commenting.reactive_analysis import ReactiveAnalysisResult
from gsr_agent.koala.models import Paper
from gsr_agent.rules.verdict_assembly import (
    VerdictEligibilityResult,
    build_verdict_draft_for_paper,
    evaluate_verdict_eligibility,
    plan_verdict_for_paper,
    select_distinct_other_agent_citations,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
_PAPER_ID = "paper-p6-001"


def _make_paper(paper_id: str = _PAPER_ID) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        open_time=_NOW,
        review_end_time=_NOW + timedelta(hours=48),
        verdict_end_time=_NOW + timedelta(hours=72),
        state="REVIEW_ACTIVE",
    )


def _make_react_result(
    comment_id: str = "cmt-001",
    confidence: float = 0.8,
    paper_id: str = _PAPER_ID,
) -> ReactiveAnalysisResult:
    verif = {
        "verdict": "refuted",
        "confidence": confidence,
        "claim_id": f"claim-{comment_id}",
        "reasoning": "Paper section 3 contradicts the claim.",
    }
    return ReactiveAnalysisResult(
        comment_id=comment_id,
        paper_id=paper_id,
        recommendation="react",
        verifications=[verif],
        draft_text=f"[DRY-RUN — not posted]\nReactive fact-check for {comment_id}",
    )


def _make_skip_result(comment_id: str = "cmt-skip") -> ReactiveAnalysisResult:
    return ReactiveAnalysisResult(
        comment_id=comment_id,
        paper_id=_PAPER_ID,
        recommendation="skip",
        skip_reason="all claims supported by paper",
    )


def _make_db(citable_other: int = 2, *, distinct_agents: Optional[int] = None) -> MagicMock:
    """Build a mock KoalaDB.

    citable_other controls get_comment_stats["citable_other"] (used for heat-band).
    distinct_agents controls how many unique author_agent_ids get_citable_other_comments_for_paper
    returns; defaults to citable_other when not set.
    """
    db = MagicMock()
    db.get_comment_stats.return_value = {
        "total": citable_other + 1,
        "ours": 1,
        "citable_other": citable_other,
    }
    n = distinct_agents if distinct_agents is not None else citable_other
    db.get_citable_other_comments_for_paper.return_value = [
        {
            "comment_id": f"cmt-other-{i}",
            "paper_id": _PAPER_ID,
            "author_agent_id": f"agent-{i}",
            "thread_id": f"thread-{i}",
            "text": f"Comment text {i}",
        }
        for i in range(n)
    ]
    db.get_strongest_contradiction_confidence.return_value = None
    db.get_phase5a_stats.return_value = {
        "react_count": 0,
        "skip_count": 0,
        "unclear_count": 0,
        "comments_analyzed": 0,
    }
    return db


def _make_valid_citations(n: int = 3) -> List[dict]:
    """Build n citation dicts with distinct author_agent_ids, sorted by comment_id."""
    return [
        {
            "comment_id": f"cmt-cite-{i:03d}",
            "author_agent_id": f"cite-agent-{i}",
            "thread_id": None,
            "text": f"Citation text {i}",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# A. evaluate_verdict_eligibility — heat-band × signal matrix
# ---------------------------------------------------------------------------

class TestEvaluateVerdictEligibility:
    """Endgame model: only saturated (15+ other agents) is ineligible.
    All other heat bands pass Gate 1 regardless of contradiction signal strength.
    Gate 2 (MIN_DISTINCT_OTHER_AGENTS) is the social-proof enforcer.
    """

    def test_goldilocks_strong_reactive_is_eligible(self):
        paper = _make_paper()
        db = _make_db(citable_other=5)
        result = evaluate_verdict_eligibility(paper, db, [_make_react_result(confidence=0.80)])
        assert result.eligible is True
        assert result.reason_code == "eligible"
        assert result.heat_band == "goldilocks"

    def test_warm_any_signal_is_eligible(self):
        paper = _make_paper()
        db = _make_db(citable_other=1)
        result = evaluate_verdict_eligibility(paper, db, [_make_react_result(confidence=0.40)])
        assert result.eligible is True
        assert result.reason_code == "eligible"
        assert result.heat_band == "warm"

    def test_cold_no_react_is_eligible(self):
        """Cold papers pass Gate 1 (Gate 2 handles social-proof enforcement)."""
        paper = _make_paper()
        db = _make_db(citable_other=0)
        result = evaluate_verdict_eligibility(paper, db, [])
        assert result.eligible is True
        assert result.reason_code == "eligible"
        assert result.heat_band == "cold"

    def test_goldilocks_weak_reactive_is_eligible(self):
        """Endgame: non-saturated papers are eligible regardless of signal strength."""
        paper = _make_paper()
        db = _make_db(citable_other=5)
        result = evaluate_verdict_eligibility(paper, db, [_make_react_result(confidence=0.30)])
        assert result.eligible is True
        assert result.reason_code == "eligible"
        assert result.heat_band == "goldilocks"

    def test_crowded_any_signal_is_eligible(self):
        """Endgame: crowded (11–14) papers pass Gate 1."""
        paper = _make_paper()
        db = _make_db(citable_other=12)
        result = evaluate_verdict_eligibility(paper, db, [_make_react_result(confidence=0.40)])
        assert result.eligible is True
        assert result.reason_code == "eligible"
        assert result.heat_band == "crowded"

    def test_saturated_strong_reactive_is_not_eligible(self):
        """Saturated (15+) is always ineligible regardless of signal strength."""
        paper = _make_paper()
        db = _make_db(citable_other=15)
        result = evaluate_verdict_eligibility(paper, db, [_make_react_result(confidence=0.90)])
        assert result.eligible is False
        assert result.reason_code == "saturated_low_value_v0"
        assert result.heat_band == "saturated"

    def test_saturated_at_boundary_15(self):
        paper = _make_paper()
        db = _make_db(citable_other=15)
        result = evaluate_verdict_eligibility(paper, db, [])
        assert result.eligible is False
        assert result.heat_band == "saturated"

    def test_not_saturated_at_boundary_14(self):
        paper = _make_paper()
        db = _make_db(citable_other=14)
        result = evaluate_verdict_eligibility(paper, db, [])
        assert result.eligible is True
        assert result.heat_band == "crowded"

    def test_goldilocks_no_react_is_eligible(self):
        """Endgame: goldilocks papers are eligible without reactive signal."""
        paper = _make_paper()
        db = _make_db(citable_other=5)
        result = evaluate_verdict_eligibility(paper, db, [_make_skip_result()])
        assert result.eligible is True
        assert result.reason_code == "eligible"
        assert result.heat_band == "goldilocks"

    def test_selected_candidates_capped_at_three(self):
        paper = _make_paper()
        db = _make_db(citable_other=5)
        react_results = [
            _make_react_result(comment_id=f"cmt-{i}", confidence=0.80)
            for i in range(5)
        ]
        result = evaluate_verdict_eligibility(paper, db, react_results)
        assert result.eligible is True
        assert len(result.selected_candidates) == 3

    def test_selected_candidates_empty_when_saturated(self):
        paper = _make_paper()
        db = _make_db(citable_other=15)
        result = evaluate_verdict_eligibility(paper, db, [_make_react_result(confidence=0.90)])
        assert result.eligible is False
        assert result.selected_candidates == []

    def test_strongest_contradiction_confidence_set_when_eligible(self):
        paper = _make_paper()
        db = _make_db(citable_other=5)
        result = evaluate_verdict_eligibility(paper, db, [_make_react_result(confidence=0.82)])
        assert result.strongest_contradiction_confidence == pytest.approx(0.82)

    def test_strongest_contradiction_confidence_is_none_when_no_react(self):
        paper = _make_paper()
        db = _make_db(citable_other=5)
        result = evaluate_verdict_eligibility(paper, db, [])
        assert result.strongest_contradiction_confidence is None
        assert result.eligible is True

    def test_strongest_confidence_is_max_across_candidates(self):
        paper = _make_paper()
        db = _make_db(citable_other=5)
        results = [
            _make_react_result(comment_id="cmt-a", confidence=0.76),
            _make_react_result(comment_id="cmt-b", confidence=0.91),
        ]
        result = evaluate_verdict_eligibility(paper, db, results)
        assert result.strongest_contradiction_confidence == pytest.approx(0.91)

    def test_heat_band_14_is_crowded_eligible(self):
        paper = _make_paper()
        db = _make_db(citable_other=14)
        result = evaluate_verdict_eligibility(paper, db, [_make_react_result(confidence=0.95)])
        assert result.heat_band == "crowded"
        assert result.eligible is True

    def test_heat_band_10_is_goldilocks_eligible(self):
        paper = _make_paper()
        db = _make_db(citable_other=10)
        result = evaluate_verdict_eligibility(paper, db, [_make_react_result(confidence=0.60)])
        assert result.heat_band == "goldilocks"
        assert result.eligible is True

    def test_empty_reactive_results_goldilocks_eligible(self):
        """Endgame: goldilocks + no reactive evidence → still eligible (weak signal ok)."""
        paper = _make_paper()
        db = _make_db(citable_other=5)
        result = evaluate_verdict_eligibility(paper, db, [])
        assert result.eligible is True
        assert result.reason_code == "eligible"
        assert result.heat_band == "goldilocks"


# ---------------------------------------------------------------------------
# B. select_distinct_other_agent_citations — citation correctness gate
# ---------------------------------------------------------------------------

class TestSelectDistinctOtherAgentCitations:
    def _db(self, comments: list) -> MagicMock:
        db = MagicMock()
        db.get_citable_other_comments_for_paper.return_value = comments
        return db

    def _c(self, comment_id: str, agent_id: str) -> dict:
        return {
            "comment_id": comment_id,
            "author_agent_id": agent_id,
            "thread_id": None,
            "text": f"text for {comment_id}",
        }

    def test_3_distinct_agents_returns_3(self):
        db = self._db([self._c("a", "ag-1"), self._c("b", "ag-2"), self._c("c", "ag-3")])
        result = select_distinct_other_agent_citations(_PAPER_ID, db)
        assert len(result) == 3

    def test_duplicate_agent_comments_count_once(self):
        db = self._db([
            self._c("cmt-a1", "agent-A"),
            self._c("cmt-a2", "agent-A"),  # same agent
            self._c("cmt-b", "agent-B"),
            self._c("cmt-c", "agent-C"),
        ])
        result = select_distinct_other_agent_citations(_PAPER_ID, db)
        assert len(result) == 3
        agent_ids = [c["author_agent_id"] for c in result]
        assert agent_ids.count("agent-A") == 1

    def test_3_comments_2_agents_returns_empty(self):
        """3 comments but only 2 distinct agents → below min_count=3 → empty."""
        db = self._db([
            self._c("cmt-1", "agent-A"),
            self._c("cmt-2", "agent-A"),
            self._c("cmt-3", "agent-B"),
        ])
        result = select_distinct_other_agent_citations(_PAPER_ID, db)
        assert result == []

    def test_4_comments_3_distinct_agents_valid(self):
        db = self._db([
            self._c("cmt-a1", "agent-A"),
            self._c("cmt-a2", "agent-A"),  # duplicate
            self._c("cmt-b", "agent-B"),
            self._c("cmt-c", "agent-C"),
        ])
        result = select_distinct_other_agent_citations(_PAPER_ID, db)
        assert len(result) == 3

    def test_zero_comments_returns_empty(self):
        result = select_distinct_other_agent_citations(_PAPER_ID, self._db([]))
        assert result == []

    def test_stable_ordering_by_comment_id(self):
        db = self._db([
            self._c("cmt-zzz", "agent-Z"),
            self._c("cmt-aaa", "agent-A"),
            self._c("cmt-mmm", "agent-M"),
        ])
        result = select_distinct_other_agent_citations(_PAPER_ID, db)
        ids = [c["comment_id"] for c in result]
        assert ids == sorted(ids)

    def test_earliest_comment_chosen_per_agent(self):
        """DB returns rows ordered by created_at; first occurrence per agent is selected."""
        db = self._db([
            self._c("cmt-early", "agent-A"),
            self._c("cmt-late", "agent-A"),  # later, same agent
            self._c("cmt-b", "agent-B"),
            self._c("cmt-c", "agent-C"),
        ])
        result = select_distinct_other_agent_citations(_PAPER_ID, db)
        agent_a = [c for c in result if c["author_agent_id"] == "agent-A"]
        assert len(agent_a) == 1
        assert agent_a[0]["comment_id"] == "cmt-early"

    def test_comment_without_agent_id_excluded(self):
        db = self._db([
            {"comment_id": "cmt-none", "author_agent_id": None, "thread_id": None, "text": ""},
            {"comment_id": "cmt-empty", "author_agent_id": "", "thread_id": None, "text": ""},
            self._c("cmt-b", "agent-B"),
            self._c("cmt-c", "agent-C"),
        ])
        result = select_distinct_other_agent_citations(_PAPER_ID, db)
        assert result == []  # only 2 valid agents, below min_count=3

    def test_min_count_param_allows_relaxed_threshold(self):
        db = self._db([self._c("cmt-a", "agent-A"), self._c("cmt-b", "agent-B")])
        result = select_distinct_other_agent_citations(_PAPER_ID, db, min_count=2)
        assert len(result) == 2

    def test_uses_citable_other_method_not_all_comments(self):
        """Verifies the DB method that excludes self-comments (is_ours=0) is used."""
        db = self._db([self._c("a", "ag-1"), self._c("b", "ag-2"), self._c("c", "ag-3")])
        select_distinct_other_agent_citations(_PAPER_ID, db)
        db.get_citable_other_comments_for_paper.assert_called_once_with(_PAPER_ID)

    def test_result_contains_comment_id_and_author_fields(self):
        db = self._db([self._c("a", "ag-1"), self._c("b", "ag-2"), self._c("c", "ag-3")])
        result = select_distinct_other_agent_citations(_PAPER_ID, db)
        for c in result:
            assert "comment_id" in c
            assert "author_agent_id" in c

    def test_exactly_min_count_agents_valid(self):
        """Exactly 3 distinct agents meets the minimum."""
        db = self._db([self._c("a", "ag-1"), self._c("b", "ag-2"), self._c("c", "ag-3")])
        result = select_distinct_other_agent_citations(_PAPER_ID, db)
        assert len(result) == 3

    def test_one_below_min_count_returns_empty(self):
        db = self._db([self._c("a", "ag-1"), self._c("b", "ag-2")])
        result = select_distinct_other_agent_citations(_PAPER_ID, db)
        assert result == []


# ---------------------------------------------------------------------------
# C. build_verdict_draft_for_paper — markdown structure
# ---------------------------------------------------------------------------

class TestBuildVerdictDraft:
    def _run(self, confidence: float = 0.82) -> Optional[str]:
        paper = _make_paper()
        db = _make_db(citable_other=3)
        eligibility = VerdictEligibilityResult(
            eligible=True,
            reason_code="eligible",
            heat_band="goldilocks",
            distinct_citable_other_agents=3,
            strongest_contradiction_confidence=confidence,
            selected_candidates=[_make_react_result(confidence=confidence)],
        )
        return build_verdict_draft_for_paper(
            paper, eligibility, [], db, _NOW, valid_citations=_make_valid_citations()
        )

    def test_draft_contains_header(self):
        assert "# Verdict Draft (DRY-RUN — not submitted)" in self._run()

    def test_draft_contains_paper_id(self):
        assert _PAPER_ID in self._run()

    def test_draft_contains_heat_band(self):
        assert "goldilocks" in self._run()

    def test_draft_contains_eligible_section(self):
        assert "## Why this paper is eligible" in self._run()

    def test_draft_contains_citable_comments_section(self):
        assert "## Citable other-agent comments" in self._run()

    def test_draft_contains_reactive_evidence_section(self):
        assert "## GSR reactive evidence summary" in self._run()

    def test_draft_contains_verdict_rationale_section(self):
        assert "## Proposed verdict rationale" in self._run()

    def test_draft_contains_suggested_next_step(self):
        draft = self._run()
        assert "## Suggested next step" in draft
        assert "Manual review required" in draft

    def test_draft_shows_confidence_value(self):
        assert "0.82" in self._run(confidence=0.82)

    def test_draft_shows_candidate_comment_id(self):
        assert "cmt-001" in self._run()

    def test_draft_shows_citable_comment_agent_names(self):
        draft = self._run()
        assert "cite-agent-0" in draft
        assert "cite-agent-1" in draft
        assert "cite-agent-2" in draft

    def test_draft_shows_citation_comment_ids(self):
        draft = self._run()
        assert "cmt-cite-000" in draft
        assert "cmt-cite-001" in draft

    def test_draft_contains_generated_timestamp(self):
        assert _NOW.isoformat() in self._run()

    def test_draft_shows_no_claims_when_no_selected_candidates(self):
        paper = _make_paper()
        db = _make_db(citable_other=3)
        eligibility = VerdictEligibilityResult(
            eligible=True,
            reason_code="eligible",
            heat_band="goldilocks",
            distinct_citable_other_agents=3,
            strongest_contradiction_confidence=0.80,
            selected_candidates=[],
        )
        draft = build_verdict_draft_for_paper(
            paper, eligibility, [], db, _NOW, valid_citations=_make_valid_citations()
        )
        assert draft is not None
        assert "No refuted claims detected" in draft

    def test_build_returns_none_when_no_valid_citations_in_db(self):
        """Without explicit valid_citations, the helper is called; returns None for < 3 agents."""
        paper = _make_paper()
        db = _make_db(citable_other=2)  # 2 distinct agents → helper returns []
        eligibility = VerdictEligibilityResult(
            eligible=True,
            reason_code="eligible",
            heat_band="goldilocks",
            distinct_citable_other_agents=2,
            strongest_contradiction_confidence=0.80,
            selected_candidates=[_make_react_result()],
        )
        draft = build_verdict_draft_for_paper(paper, eligibility, [], db, _NOW)
        assert draft is None

    def test_build_returns_none_when_explicit_empty_citations(self):
        """Explicit empty valid_citations → None (caller bypass; still fails guard)."""
        paper = _make_paper()
        db = _make_db(citable_other=3)
        eligibility = VerdictEligibilityResult(
            eligible=True,
            reason_code="eligible",
            heat_band="goldilocks",
            distinct_citable_other_agents=3,
            strongest_contradiction_confidence=0.80,
            selected_candidates=[_make_react_result()],
        )
        draft = build_verdict_draft_for_paper(
            paper, eligibility, [], db, _NOW, valid_citations=[]
        )
        assert draft is None

    def test_draft_is_not_none_with_valid_citations(self):
        assert self._run() is not None


# ---------------------------------------------------------------------------
# D. plan_verdict_for_paper — orchestration
# ---------------------------------------------------------------------------

class TestPlanVerdictForPaper:
    def test_ineligible_returns_skipped_status(self):
        paper = _make_paper()
        db = _make_db(citable_other=15)
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.90)], _NOW)
        assert result["status"] == "skipped"
        assert result["eligible"] is False

    def test_ineligible_returns_no_artifact_url(self):
        paper = _make_paper()
        db = _make_db(citable_other=15)
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.90)], _NOW)
        assert result["artifact_url"] is None

    def test_ineligible_does_not_log_action(self):
        paper = _make_paper()
        db = _make_db(citable_other=15)
        plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.90)], _NOW)
        db.log_action.assert_not_called()

    def test_ineligible_returns_reason_code(self):
        paper = _make_paper()
        db = _make_db(citable_other=15)
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.90)], _NOW)
        assert result["reason_code"] == "saturated_low_value_v0"

    def test_eligible_returns_dry_run_status(self):
        paper = _make_paper()
        db = _make_db(citable_other=3)
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        assert result["status"] == "dry_run"
        assert result["eligible"] is True

    def test_eligible_returns_artifact_url(self):
        paper = _make_paper()
        db = _make_db(citable_other=3)
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        assert result["artifact_url"] is not None
        assert len(result["artifact_url"]) > 0

    def test_eligible_logs_action_once(self):
        paper = _make_paper()
        db = _make_db(citable_other=3)
        plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        db.log_action.assert_called_once()

    def test_eligible_logs_action_with_dry_run_status(self):
        paper = _make_paper()
        db = _make_db(citable_other=3)
        plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        _, kwargs = db.log_action.call_args
        assert kwargs["status"] == "dry_run"
        assert kwargs["action_type"] == "verdict_draft"

    def test_eligible_logs_action_with_paper_id(self):
        paper = _make_paper()
        db = _make_db(citable_other=3)
        plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        _, kwargs = db.log_action.call_args
        assert kwargs["paper_id"] == _PAPER_ID

    def test_artifact_url_is_test_mode_url(self):
        """Phase 6 always publishes with test_mode=True."""
        from gsr_agent.artifacts.github import is_test_mode_artifact_url
        paper = _make_paper()
        db = _make_db(citable_other=3)
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        assert is_test_mode_artifact_url(result["artifact_url"])

    def test_eligible_returns_paper_id(self):
        paper = _make_paper()
        db = _make_db(citable_other=3)
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        assert result["paper_id"] == _PAPER_ID

    def test_eligible_returns_heat_band(self):
        paper = _make_paper()
        db = _make_db(citable_other=3)
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        assert result["heat_band"] == "goldilocks"

    def test_eligible_returns_distinct_count(self):
        paper = _make_paper()
        db = _make_db(citable_other=3)
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        assert result["distinct_citable_other_agents"] == 3

    def test_test_mode_true_still_dry_run(self):
        """test_mode=True is always dry-run for Phase 6."""
        paper = _make_paper()
        db = _make_db(citable_other=3)
        result = plan_verdict_for_paper(
            paper, db, [_make_react_result(confidence=0.80)], _NOW, test_mode=True
        )
        assert result["status"] == "dry_run"

    def test_no_react_results_still_produces_dry_run(self):
        """Endgame: goldilocks papers with valid citations produce dry_run even without reactive signal."""
        paper = _make_paper()
        db = _make_db(citable_other=3)
        result = plan_verdict_for_paper(paper, db, [], _NOW)
        assert result["status"] == "dry_run"
        assert result["eligible"] is True

    def test_details_in_log_action_contain_reason_code(self):
        paper = _make_paper()
        db = _make_db(citable_other=3)
        plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        _, kwargs = db.log_action.call_args
        assert kwargs["details"]["reason_code"] == "eligible"

    def test_details_in_log_action_contain_heat_band(self):
        paper = _make_paper()
        db = _make_db(citable_other=3)
        plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        _, kwargs = db.log_action.call_args
        assert kwargs["details"]["heat_band"] == "goldilocks"

    def test_details_contain_cited_comment_ids(self):
        paper = _make_paper()
        db = _make_db(citable_other=3)
        plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        _, kwargs = db.log_action.call_args
        assert "cited_comment_ids" in kwargs["details"]
        assert len(kwargs["details"]["cited_comment_ids"]) == 3

    # --- Citation gate tests (Phase 6.5) ---

    def test_warm_band_1_agent_fails_citation_gate(self):
        """Warm heat band (1 distinct agent) passes Gate 1 but fails Gate 2 (< 3 agents)."""
        paper = _make_paper()
        db = _make_db(citable_other=1)  # 1 distinct agent in DB
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        assert result["status"] == "skipped"
        assert result["reason_code"] == "insufficient_distinct_other_agent_citations"
        assert result["artifact_url"] is None

    def test_citation_gate_skip_does_not_log_action(self):
        """No artifact logged when citation gate fails."""
        paper = _make_paper()
        db = _make_db(citable_other=1)
        plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        db.log_action.assert_not_called()

    def test_eligible_band_with_only_2_distinct_agents_skips(self):
        """Gate 1 passes (goldilocks + strong signal) but Gate 2 fails (2 distinct < 3)."""
        paper = _make_paper()
        db = _make_db(citable_other=3, distinct_agents=2)
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        assert result["status"] == "skipped"
        assert result["reason_code"] == "insufficient_distinct_other_agent_citations"
        assert result["artifact_url"] is None
        db.log_action.assert_not_called()

    def test_3_comments_2_agents_skips_citation_gate(self):
        """Duplicate agent comments: 3 total but only 2 distinct → citation gate fails."""
        paper = _make_paper()
        db = MagicMock()
        db.get_comment_stats.return_value = {"total": 4, "ours": 1, "citable_other": 3}
        db.get_citable_other_comments_for_paper.return_value = [
            {"comment_id": "cmt-1", "author_agent_id": "agent-A", "thread_id": None, "text": ""},
            {"comment_id": "cmt-2", "author_agent_id": "agent-A", "thread_id": None, "text": ""},
            {"comment_id": "cmt-3", "author_agent_id": "agent-B", "thread_id": None, "text": ""},
        ]
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        assert result["status"] == "skipped"
        assert result["reason_code"] == "insufficient_distinct_other_agent_citations"

    def test_artifact_produced_when_3_distinct_citations_exist(self):
        """Gate 1 + Gate 2 both pass → artifact produced, status dry_run."""
        paper = _make_paper()
        db = _make_db(citable_other=3)  # 3 distinct agents satisfies citation gate
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        assert result["status"] == "dry_run"
        assert result["artifact_url"] is not None

    def test_4_comments_3_distinct_agents_produces_artifact(self):
        """4 comments but 3 distinct agents → citation gate passes, artifact produced."""
        paper = _make_paper()
        db = MagicMock()
        db.get_comment_stats.return_value = {"total": 5, "ours": 1, "citable_other": 4}
        db.get_citable_other_comments_for_paper.return_value = [
            {"comment_id": "cmt-1", "author_agent_id": "agent-A", "thread_id": None, "text": ""},
            {"comment_id": "cmt-2", "author_agent_id": "agent-A", "thread_id": None, "text": ""},
            {"comment_id": "cmt-3", "author_agent_id": "agent-B", "thread_id": None, "text": ""},
            {"comment_id": "cmt-4", "author_agent_id": "agent-C", "thread_id": None, "text": ""},
        ]
        db.get_strongest_contradiction_confidence.return_value = None
        db.get_phase5a_stats.return_value = {
            "react_count": 0, "skip_count": 0, "unclear_count": 0, "comments_analyzed": 0,
        }
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.80)], _NOW)
        assert result["status"] == "dry_run"
        assert result["artifact_url"] is not None

    def test_goldilocks_with_sufficient_citations_produces_artifact(self):
        """Goldilocks + >= 3 distinct citations → full success path (no strong signal needed)."""
        paper = _make_paper()
        db = _make_db(citable_other=5)  # goldilocks band (3-10), 5 distinct agents
        result = plan_verdict_for_paper(paper, db, [_make_react_result(confidence=0.40)], _NOW)
        assert result["status"] == "dry_run"
        assert result["heat_band"] == "goldilocks"


# ---------------------------------------------------------------------------
# E. Aggressive-mode overrides (KOALA_AGGRESSIVE_FINAL_24H=1)
# ---------------------------------------------------------------------------

class TestAggressiveModeVerdictEligibility:
    """In aggressive mode, the saturated-band veto must not block verdicts."""

    def test_saturated_normal_mode_is_ineligible(self):
        """Normal mode: saturated (15+) is always ineligible — baseline unchanged."""
        paper = _make_paper()
        db = _make_db(citable_other=15)
        result = evaluate_verdict_eligibility(paper, db, [_make_react_result(confidence=0.90)])
        assert result.eligible is False
        assert result.reason_code == "saturated_low_value_v0"

    def test_saturated_aggressive_mode_is_eligible(self):
        """Aggressive mode: saturated paper with sufficient citations passes Gate 1."""
        paper = _make_paper()
        db = _make_db(citable_other=15)
        result = evaluate_verdict_eligibility(
            paper, db, [_make_react_result(confidence=0.90)], aggressive=True
        )
        assert result.eligible is True
        assert result.reason_code == "eligible"
        assert result.heat_band == "saturated"

    def test_saturated_aggressive_no_reactive_still_eligible(self):
        """Aggressive mode: saturated paper with no reactive signal still passes Gate 1."""
        paper = _make_paper()
        db = _make_db(citable_other=20)
        result = evaluate_verdict_eligibility(paper, db, [], aggressive=True)
        assert result.eligible is True
        assert result.heat_band == "saturated"

    def test_plan_verdict_saturated_aggressive_produces_dry_run(self):
        """Aggressive mode: plan_verdict_for_paper on saturated paper produces dry_run artifact."""
        paper = _make_paper()
        db = _make_db(citable_other=15)  # 15 distinct agents in DB, satisfies Gate 2
        result = plan_verdict_for_paper(
            paper, db, [_make_react_result(confidence=0.90)], _NOW, aggressive=True
        )
        assert result["status"] == "dry_run"
        assert result["eligible"] is True
        assert result["artifact_url"] is not None
        assert result["heat_band"] == "saturated"

    def test_plan_verdict_saturated_aggressive_insufficient_citations_skips(self):
        """Aggressive mode: Gate 1 passes for saturated but Gate 2 fails (<3 distinct agents)."""
        paper = _make_paper()
        db = _make_db(citable_other=15, distinct_agents=2)
        result = plan_verdict_for_paper(
            paper, db, [_make_react_result(confidence=0.90)], _NOW, aggressive=True
        )
        assert result["status"] == "skipped"
        assert result["reason_code"] == "insufficient_distinct_other_agent_citations"
        assert result["artifact_url"] is None

    def test_plan_verdict_normal_mode_saturated_still_skips(self):
        """Normal mode: saturated paper must still be rejected (no regression)."""
        paper = _make_paper()
        db = _make_db(citable_other=15)
        result = plan_verdict_for_paper(
            paper, db, [_make_react_result(confidence=0.90)], _NOW, aggressive=False
        )
        assert result["status"] == "skipped"
        assert result["reason_code"] == "saturated_low_value_v0"
