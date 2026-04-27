"""Tests for Phase A MVP: heuristic_v0 verdict score and draft integration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from gsr_agent.koala.models import Paper
from gsr_agent.rules.verdict_assembly import plan_verdict_for_paper
from gsr_agent.rules.verdict_scoring import VerdictScore, score_verdict_heuristic_v0

_NOW = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
_PAPER_ID = "paper-score-001"


def _make_paper(paper_id: str = _PAPER_ID) -> Paper:
    _open = _NOW - timedelta(hours=60)  # 60h before _NOW → verdict window open (12h remaining)
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        open_time=_open,
        review_end_time=_open + timedelta(hours=48),
        verdict_end_time=_open + timedelta(hours=72),
        state="VERDICT_ACTIVE",
    )


def _make_db(strongest_conf=None, react_count=0, unclear_count=0) -> MagicMock:
    db = MagicMock()
    db.get_strongest_contradiction_confidence.return_value = strongest_conf
    db.get_phase5a_stats.return_value = {
        "react_count": react_count,
        "skip_count": 0,
        "unclear_count": unclear_count,
        "comments_analyzed": react_count + unclear_count,
    }
    return db


# ---------------------------------------------------------------------------
# TestVerdictScoreDataclass
# ---------------------------------------------------------------------------

class TestVerdictScoreDataclass:

    def test_fields_are_accessible(self):
        vs = VerdictScore(score=5.0, score_source="heuristic_v0", confidence=0.5, rationale="ok")
        assert vs.score == 5.0
        assert vs.score_source == "heuristic_v0"
        assert vs.confidence == 0.5
        assert vs.rationale == "ok"

    def test_is_frozen(self):
        vs = VerdictScore(score=5.0, score_source="heuristic_v0", confidence=0.5, rationale="ok")
        with pytest.raises((AttributeError, TypeError)):
            vs.score = 6.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestHeuristicV0StrongContradiction
# ---------------------------------------------------------------------------

class TestHeuristicV0StrongContradiction:

    def test_conf_at_threshold_returns_strong_tier(self):
        db = _make_db(strongest_conf=0.85)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.score == 3.5

    def test_high_conf_also_strong_tier(self):
        db = _make_db(strongest_conf=0.97)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.score == 3.5

    def test_strong_tier_source(self):
        db = _make_db(strongest_conf=0.90)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.score_source == "heuristic_v0"

    def test_strong_tier_confidence_matches_input(self):
        db = _make_db(strongest_conf=0.91)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.confidence == round(0.91, 4)

    def test_strong_tier_rationale_nonempty(self):
        db = _make_db(strongest_conf=0.88)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert len(result.rationale) > 0


# ---------------------------------------------------------------------------
# TestHeuristicV0MediumContradiction
# ---------------------------------------------------------------------------

class TestHeuristicV0MediumContradiction:

    def test_conf_at_medium_threshold_returns_medium_tier(self):
        db = _make_db(strongest_conf=0.65)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.score == 4.5

    def test_conf_below_strong_above_medium_is_medium_tier(self):
        db = _make_db(strongest_conf=0.75)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.score == 4.5

    def test_medium_tier_confidence_matches_input(self):
        db = _make_db(strongest_conf=0.72)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.confidence == round(0.72, 4)

    def test_medium_tier_source(self):
        db = _make_db(strongest_conf=0.70)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.score_source == "heuristic_v0"


# ---------------------------------------------------------------------------
# TestHeuristicV0Unclear
# ---------------------------------------------------------------------------

class TestHeuristicV0Unclear:

    def test_no_react_with_unclear_returns_neutral(self):
        db = _make_db(strongest_conf=None, react_count=0, unclear_count=2)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.score == 5.5

    def test_unclear_tier_confidence_is_conservative(self):
        db = _make_db(strongest_conf=None, react_count=0, unclear_count=1)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.confidence == 0.40

    def test_nonzero_react_with_unclear_does_not_enter_unclear_tier(self):
        db = _make_db(strongest_conf=None, react_count=1, unclear_count=2)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.score == 6.5

    def test_unclear_tier_source(self):
        db = _make_db(strongest_conf=None, react_count=0, unclear_count=3)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.score_source == "heuristic_v0"


# ---------------------------------------------------------------------------
# TestHeuristicV0Supported
# ---------------------------------------------------------------------------

class TestHeuristicV0Supported:

    def test_no_signals_returns_weak_accept(self):
        db = _make_db(strongest_conf=None, react_count=0, unclear_count=0)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.score == 6.5

    def test_supported_tier_confidence_is_conservative(self):
        db = _make_db(strongest_conf=None, react_count=0, unclear_count=0)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.confidence == 0.35

    def test_below_medium_conf_falls_through_to_supported(self):
        db = _make_db(strongest_conf=0.50)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.score == 6.5

    def test_supported_tier_source(self):
        db = _make_db(strongest_conf=None)
        result = score_verdict_heuristic_v0(_make_paper(), [], db)
        assert result.score_source == "heuristic_v0"


# ---------------------------------------------------------------------------
# TestHeuristicV0Invariants — parametrized across all 4 tiers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strongest_conf,react_count,unclear_count,expected_score", [
    (0.90, 0, 0, 3.5),
    (0.70, 0, 0, 4.5),
    (None, 0, 2, 5.5),
    (None, 0, 0, 6.5),
])
def test_score_always_in_heuristic_range(strongest_conf, react_count, unclear_count, expected_score):
    db = _make_db(strongest_conf=strongest_conf, react_count=react_count, unclear_count=unclear_count)
    result = score_verdict_heuristic_v0(_make_paper(), [], db)
    assert result.score == expected_score
    assert 3.0 <= result.score <= 7.0


@pytest.mark.parametrize("strongest_conf,react_count,unclear_count", [
    (0.90, 0, 0),
    (0.70, 0, 0),
    (None, 0, 2),
    (None, 0, 0),
])
def test_source_always_heuristic_v0(strongest_conf, react_count, unclear_count):
    db = _make_db(strongest_conf=strongest_conf, react_count=react_count, unclear_count=unclear_count)
    result = score_verdict_heuristic_v0(_make_paper(), [], db)
    assert result.score_source == "heuristic_v0"


@pytest.mark.parametrize("strongest_conf,react_count,unclear_count", [
    (0.90, 0, 0),
    (0.70, 0, 0),
    (None, 0, 2),
    (None, 0, 0),
])
def test_confidence_in_range(strongest_conf, react_count, unclear_count):
    db = _make_db(strongest_conf=strongest_conf, react_count=react_count, unclear_count=unclear_count)
    result = score_verdict_heuristic_v0(_make_paper(), [], db)
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.parametrize("strongest_conf,react_count,unclear_count", [
    (0.90, 0, 0),
    (0.70, 0, 0),
    (None, 0, 2),
    (None, 0, 0),
])
def test_rationale_nonempty(strongest_conf, react_count, unclear_count):
    db = _make_db(strongest_conf=strongest_conf, react_count=react_count, unclear_count=unclear_count)
    result = score_verdict_heuristic_v0(_make_paper(), [], db)
    assert isinstance(result.rationale, str) and len(result.rationale) > 0


# ---------------------------------------------------------------------------
# TestBuildDraftWithScore — draft section rendered / absent
# ---------------------------------------------------------------------------

class TestBuildDraftWithScore:

    def _build_draft(self, verdict_score=None):
        from gsr_agent.rules.verdict_assembly import (
            VerdictEligibilityResult,
            build_verdict_draft_for_paper,
        )
        eligibility = VerdictEligibilityResult(
            eligible=True,
            reason_code="eligible",
            heat_band="goldilocks",
            distinct_citable_other_agents=4,
            strongest_contradiction_confidence=0.85,
            selected_candidates=[],
        )
        citations = [
            {"comment_id": f"cmt-{i}", "author_agent_id": f"agent-{i}"}
            for i in range(3)
        ]
        db = MagicMock()
        return build_verdict_draft_for_paper(
            _make_paper(), eligibility, [], db, _NOW,
            valid_citations=citations, verdict_score=verdict_score,
        )

    def test_without_score_no_verdict_score_section(self):
        draft = self._build_draft(verdict_score=None)
        assert "## Verdict Score" not in draft

    def test_with_score_section_present(self):
        vs = VerdictScore(score=3.5, score_source="heuristic_v0", confidence=0.88, rationale="Strong contradiction.")
        draft = self._build_draft(verdict_score=vs)
        assert "## Verdict Score" in draft

    def test_with_score_shows_score_value(self):
        vs = VerdictScore(score=4.5, score_source="heuristic_v0", confidence=0.72, rationale="Moderate.")
        draft = self._build_draft(verdict_score=vs)
        assert "4.5" in draft

    def test_with_score_shows_source(self):
        vs = VerdictScore(score=6.5, score_source="heuristic_v0", confidence=0.35, rationale="Supported.")
        draft = self._build_draft(verdict_score=vs)
        assert "heuristic_v0" in draft

    def test_with_score_shows_confidence(self):
        vs = VerdictScore(score=3.5, score_source="heuristic_v0", confidence=0.9100, rationale="Strong.")
        draft = self._build_draft(verdict_score=vs)
        assert "0.9100" in draft

    def test_with_score_shows_rationale(self):
        vs = VerdictScore(score=5.5, score_source="heuristic_v0", confidence=0.40, rationale="Ambiguous signals detected.")
        draft = self._build_draft(verdict_score=vs)
        assert "Ambiguous signals detected." in draft


# ---------------------------------------------------------------------------
# TestPlanVerdictIntegration — plan_verdict_for_paper returns score fields
# ---------------------------------------------------------------------------

class TestPlanVerdictIntegration:

    def _make_plan_db(self, strongest_conf=None):
        db = _make_db(strongest_conf=strongest_conf)
        db.get_comment_stats.return_value = {"total": 5, "ours": 1, "citable_other": 4}
        db.get_citable_other_comments_for_paper.return_value = [
            {"comment_id": f"cmt-{i}", "author_agent_id": f"agent-{i}"}
            for i in range(4)
        ]
        return db

    def _make_react_result(self):
        from gsr_agent.commenting.reactive_analysis import ReactiveAnalysisResult
        return ReactiveAnalysisResult(
            comment_id="cmt-001",
            paper_id=_PAPER_ID,
            recommendation="react",
            verifications=[{"verdict": "refuted", "confidence": 0.9, "claim_id": "c1", "reasoning": "r"}],
            draft_text="draft",
        )

    @patch("gsr_agent.rules.verdict_assembly.publish_verdict_artifact", return_value="https://gh/dry.md")
    def test_dry_run_result_has_score_key(self, mock_pub):
        db = self._make_plan_db(strongest_conf=0.90)
        result = plan_verdict_for_paper(_make_paper(), db, [self._make_react_result()], _NOW)
        assert "score" in result

    @patch("gsr_agent.rules.verdict_assembly.publish_verdict_artifact", return_value="https://gh/dry.md")
    def test_dry_run_result_has_verdict_score_key(self, mock_pub):
        db = self._make_plan_db(strongest_conf=0.90)
        result = plan_verdict_for_paper(_make_paper(), db, [self._make_react_result()], _NOW)
        assert "verdict_score" in result

    @patch("gsr_agent.rules.verdict_assembly.publish_verdict_artifact", return_value="https://gh/dry.md")
    def test_score_in_heuristic_range(self, mock_pub):
        db = self._make_plan_db(strongest_conf=0.90)
        result = plan_verdict_for_paper(_make_paper(), db, [self._make_react_result()], _NOW)
        assert 3.0 <= result["score"] <= 7.0

    @patch("gsr_agent.rules.verdict_assembly.publish_verdict_artifact", return_value="https://gh/dry.md")
    def test_verdict_score_is_verdict_score_instance(self, mock_pub):
        db = self._make_plan_db(strongest_conf=0.90)
        result = plan_verdict_for_paper(_make_paper(), db, [self._make_react_result()], _NOW)
        assert isinstance(result["verdict_score"], VerdictScore)

    @patch("gsr_agent.rules.verdict_assembly.publish_verdict_artifact", return_value="https://gh/dry.md")
    def test_publish_called_with_heuristic_score(self, mock_pub):
        db = self._make_plan_db(strongest_conf=0.90)
        plan_verdict_for_paper(_make_paper(), db, [self._make_react_result()], _NOW)
        _, kwargs = mock_pub.call_args
        assert kwargs["score"] == 3.5

    def test_skipped_result_has_none_score(self):
        db = MagicMock()
        db.get_comment_stats.return_value = {"total": 1, "ours": 1, "citable_other": 0}
        result = plan_verdict_for_paper(_make_paper(), db, [], _NOW)
        assert result["score"] is None
        assert result["verdict_score"] is None


# ---------------------------------------------------------------------------
# TestLiveVerdictUsesHeuristicScore — score flows from plan → _submit_live_verdict
# ---------------------------------------------------------------------------

class TestLiveVerdictUsesHeuristicScore:

    _MOD = "gsr_agent.orchestration.operational_loop"

    def _make_process_db(self):
        db = MagicMock()
        db.has_recent_reactive_action_for_comment.return_value = False
        db.has_recent_verdict_action_for_paper.return_value = False
        return db

    @patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[])
    @patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None)
    @patch(f"{_MOD}.evaluate_verdict_eligibility")
    @patch(f"{_MOD}.plan_verdict_for_paper")
    @patch(f"{_MOD}._submit_live_verdict", return_value=(True, "live_submitted"))
    @patch(f"{_MOD}.get_run_mode", return_value="live")
    def test_score_from_plan_passed_to_submit(
        self, mock_mode, mock_submit, mock_plan, mock_elig, mock_cand, mock_analyze
    ):
        from gsr_agent.orchestration.operational_loop import _process_paper, _DryRunClient
        from gsr_agent.rules.verdict_assembly import VerdictEligibilityResult

        heuristic_score = VerdictScore(
            score=3.5, score_source="heuristic_v0", confidence=0.91, rationale="Strong."
        )
        mock_plan.return_value = {
            "artifact_url": "https://gh/dry.md",
            "status": "dry_run",
            "score": 3.5,
            "verdict_score": heuristic_score,
        }
        mock_elig.return_value = VerdictEligibilityResult(
            eligible=True, reason_code="eligible", heat_band="goldilocks",
            distinct_citable_other_agents=4, strongest_contradiction_confidence=0.91,
        )
        db = self._make_process_db()
        live_client = MagicMock()

        _process_paper(
            _make_paper(), _DryRunClient(), db, 100.0, _NOW, False,
            live_verdict=True, verdict_live_budget_remaining=1,
            live_client=live_client, allowlisted=True,
        )

        _, submit_kwargs = mock_submit.call_args
        assert submit_kwargs.get("score") == 3.5
        assert submit_kwargs.get("verdict_score") is heuristic_score

    @patch(f"{_MOD}.analyze_reactive_candidates_for_paper", return_value=[])
    @patch(f"{_MOD}.select_best_reactive_candidate_for_paper", return_value=None)
    @patch(f"{_MOD}.evaluate_verdict_eligibility")
    @patch(f"{_MOD}.plan_verdict_for_paper")
    @patch(f"{_MOD}._submit_live_verdict", return_value=(False, "missing_verdict_score"))
    @patch(f"{_MOD}.get_run_mode", return_value="live")
    def test_missing_score_from_plan_blocks_submit(
        self, mock_mode, mock_submit, mock_plan, mock_elig, mock_cand, mock_analyze
    ):
        from gsr_agent.orchestration.operational_loop import _process_paper, _DryRunClient
        from gsr_agent.rules.verdict_assembly import VerdictEligibilityResult

        mock_plan.return_value = {
            "artifact_url": "https://gh/dry.md",
            "status": "dry_run",
            "score": None,
            "verdict_score": None,
        }
        mock_elig.return_value = VerdictEligibilityResult(
            eligible=True, reason_code="eligible", heat_band="goldilocks",
            distinct_citable_other_agents=4, strongest_contradiction_confidence=None,
        )
        db = self._make_process_db()
        live_client = MagicMock()

        result = _process_paper(
            _make_paper(), _DryRunClient(), db, 100.0, _NOW, False,
            live_verdict=True, verdict_live_budget_remaining=1,
            live_client=live_client, allowlisted=True,
        )

        assert result["verdict_live_submitted"] is False
        assert result["verdict_live_reason"] == "missing_verdict_score"
