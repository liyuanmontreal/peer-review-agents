"""Tests for Phase 5A reactive fact-check dry-run pipeline."""

import json
import pytest

from gsr_agent.storage.db import KoalaDB
from gsr_agent.commenting.reactive_analysis import (
    ReactiveAnalysisResult,
    analyze_reactive_opportunity_for_comment,
    analyze_reactive_candidates_for_paper,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    koala_db = KoalaDB(str(tmp_path / "test_koala.db"))
    yield koala_db
    koala_db.close()


_PAPER_ID = "paper-001"
_COMMENT_ID = "comment-001"
_COMMENT_TEXT = "The model achieves 95% accuracy on MNIST and beats all baselines."

_FAKE_CLAIM = {
    "id": "claim-001",
    "paper_id": _PAPER_ID,
    "review_id": _COMMENT_ID,
    "claim_text": "The model achieves 95% accuracy on MNIST",
    "verbatim_quote": "achieves 95% accuracy on MNIST",
    "claim_type": "factual",
    "confidence": 0.85,
    "challengeability": 0.9,
    "category": "results",
    "binary_question": "Does the model achieve 95% accuracy on MNIST?",
    "why_challengeable": "Specific numeric claim verifiable from paper.",
}

_FAKE_REFUTED_VERIF = {
    "id": "verif-001",
    "claim_id": "claim-001",
    "paper_id": _PAPER_ID,
    "review_id": _COMMENT_ID,
    "verdict": "refuted",
    "confidence": 0.8,
    "reasoning": "Table 3 shows 87% accuracy, not 95%.",
    "supporting_quote": "87% accuracy on MNIST",
    "evidence": [],
    "model_id": "gpt-4o",
    "status": "success",
    "error": None,
}

_FAKE_SUPPORTED_VERIF = {
    "id": "verif-001",
    "claim_id": "claim-001",
    "paper_id": _PAPER_ID,
    "review_id": _COMMENT_ID,
    "verdict": "supported",
    "confidence": 0.9,
    "reasoning": "Table 3 confirms 95% accuracy.",
    "supporting_quote": "95% accuracy on MNIST",
    "evidence": [],
    "model_id": "gpt-4o",
    "status": "success",
    "error": None,
}

_FAKE_INSUFFICIENT_VERIF = {
    "id": "verif-001",
    "claim_id": "claim-001",
    "paper_id": _PAPER_ID,
    "review_id": _COMMENT_ID,
    "verdict": "insufficient_evidence",
    "confidence": 0.4,
    "reasoning": "No relevant evidence found.",
    "supporting_quote": None,
    "evidence": [],
    "model_id": "gpt-4o",
    "status": "success",
    "error": None,
}


# ---------------------------------------------------------------------------
# Section A: schema + DB helpers
# ---------------------------------------------------------------------------

def test_phase5a_tables_created(db):
    tables = {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "koala_extracted_claims" in tables
    assert "koala_claim_verifications" in tables
    assert "koala_reactive_drafts" in tables


def test_insert_extracted_claim_round_trip(db):
    db.insert_extracted_claim({
        "claim_id": "c-1",
        "comment_id": "cmt-1",
        "paper_id": "p-1",
        "claim_text": "The model achieves 95% accuracy",
        "category": "results",
        "confidence": 0.9,
        "challengeability": 0.8,
        "binary_question": "Does the model achieve 95% accuracy?",
    })
    row = db._conn.execute(
        "SELECT * FROM koala_extracted_claims WHERE claim_id='c-1'"
    ).fetchone()
    assert row is not None
    assert row["claim_text"] == "The model achieves 95% accuracy"
    assert row["category"] == "results"
    assert abs(row["confidence"] - 0.9) < 1e-9


def test_insert_claim_verification_round_trip(db):
    db.insert_claim_verification({
        "verification_id": "v-1",
        "claim_id": "c-1",
        "comment_id": "cmt-1",
        "paper_id": "p-1",
        "verdict": "refuted",
        "confidence": 0.75,
        "reasoning": "Paper shows different numbers.",
        "supporting_quote": "87% accuracy",
        "model_id": "gpt-4o",
    })
    row = db._conn.execute(
        "SELECT * FROM koala_claim_verifications WHERE verification_id='v-1'"
    ).fetchone()
    assert row is not None
    assert row["verdict"] == "refuted"
    assert abs(row["confidence"] - 0.75) < 1e-9


def test_insert_reactive_draft_round_trip(db):
    db.insert_reactive_draft({
        "draft_id": "d-1",
        "comment_id": "cmt-1",
        "paper_id": "p-1",
        "recommendation": "react",
        "draft_text": "[DRY-RUN — not posted]\nDraft text here",
        "analysis_json": json.dumps({"recommendation": "react", "claim_count": 1}),
    })
    row = db._conn.execute(
        "SELECT * FROM koala_reactive_drafts WHERE draft_id='d-1'"
    ).fetchone()
    assert row is not None
    assert row["recommendation"] == "react"
    assert "[DRY-RUN" in row["draft_text"]


def test_clear_phase5a_for_comment(db):
    db.insert_extracted_claim({
        "claim_id": "c-1", "comment_id": "cmt-1", "paper_id": "p-1",
        "claim_text": "text", "category": None, "confidence": None,
        "challengeability": None, "binary_question": None,
    })
    db.insert_claim_verification({
        "verification_id": "v-1", "claim_id": "c-1", "comment_id": "cmt-1",
        "paper_id": "p-1", "verdict": "refuted", "confidence": 0.8,
        "reasoning": None, "supporting_quote": None, "model_id": None,
    })
    db.insert_reactive_draft({
        "draft_id": "d-1", "comment_id": "cmt-1", "paper_id": "p-1",
        "recommendation": "react", "draft_text": None, "analysis_json": None,
    })
    db.clear_phase5a_for_comment("cmt-1")
    assert db._conn.execute(
        "SELECT COUNT(*) FROM koala_extracted_claims WHERE comment_id='cmt-1'"
    ).fetchone()[0] == 0
    assert db._conn.execute(
        "SELECT COUNT(*) FROM koala_claim_verifications WHERE comment_id='cmt-1'"
    ).fetchone()[0] == 0
    assert db._conn.execute(
        "SELECT COUNT(*) FROM koala_reactive_drafts WHERE comment_id='cmt-1'"
    ).fetchone()[0] == 0


def test_get_citable_other_comments_for_paper(db):
    from gsr_agent.koala.models import Comment, Paper
    from datetime import datetime, timezone

    _open = datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)
    paper = Paper(
        paper_id="p-1", title="T", abstract="A", full_text="",
        pdf_url="", open_time=_open,
        review_end_time=_open, verdict_end_time=_open,
        state="REVIEW_ACTIVE",
    )
    db.upsert_paper(paper)

    citable = Comment(
        comment_id="cmt-citable", paper_id="p-1", thread_id=None, parent_id=None,
        author_agent_id="other-agent", text="A claim.", created_at=_open,
        is_ours=False, is_citable=True,
    )
    not_citable = Comment(
        comment_id="cmt-not-citable", paper_id="p-1", thread_id=None, parent_id=None,
        author_agent_id="other-agent-2", text="Another.", created_at=_open,
        is_ours=False, is_citable=False,
    )
    ours = Comment(
        comment_id="cmt-ours", paper_id="p-1", thread_id=None, parent_id=None,
        author_agent_id="us", text="Our comment.", created_at=_open,
        is_ours=True, is_citable=True,
    )
    for c in (citable, not_citable, ours):
        db.upsert_comment(c)

    results = db.get_citable_other_comments_for_paper("p-1")
    assert len(results) == 1
    assert results[0]["comment_id"] == "cmt-citable"


# ---------------------------------------------------------------------------
# Section B: fast-reject — no claims extracted
# ---------------------------------------------------------------------------

def test_no_claims_returns_skip(monkeypatch, db):
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, db=db
    )
    assert result.recommendation == "skip"
    assert result.skip_reason is not None
    assert not result.claims
    assert not result.verifications
    assert result.draft_text is None


def test_no_claims_skip_persisted_to_db(monkeypatch, db):
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [],
    )
    analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, db=db
    )
    analysis = db.get_reactive_analysis_for_comment(_COMMENT_ID)
    assert analysis is not None
    assert analysis["recommendation"] == "skip"
    assert len(analysis["claims"]) == 0


# ---------------------------------------------------------------------------
# Section C: core dry-run — refuted claim → react
# ---------------------------------------------------------------------------

def test_refuted_claim_returns_react(monkeypatch):
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_REFUTED_VERIF.copy()],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID
    )
    assert result.recommendation == "react"
    assert result.draft_text is not None
    assert "[DRY-RUN" in result.draft_text


def test_react_draft_contains_claim_text(monkeypatch):
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_REFUTED_VERIF.copy()],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID
    )
    assert "95% accuracy" in result.draft_text


def test_react_result_persisted(monkeypatch, db):
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_REFUTED_VERIF.copy()],
    )
    analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, db=db
    )
    analysis = db.get_reactive_analysis_for_comment(_COMMENT_ID)
    assert analysis is not None
    assert analysis["recommendation"] == "react"
    assert len(analysis["claims"]) == 1
    assert len(analysis["verifications"]) == 1
    assert analysis["draft"]["draft_text"] is not None
    assert "[DRY-RUN" in analysis["draft"]["draft_text"]


def test_react_low_confidence_not_react(monkeypatch):
    low_conf_refuted = {**_FAKE_REFUTED_VERIF.copy(), "confidence": 0.3}
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [low_conf_refuted],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID
    )
    assert result.recommendation != "react"


# ---------------------------------------------------------------------------
# Section D: all claims supported → skip
# ---------------------------------------------------------------------------

def test_all_supported_returns_skip(monkeypatch):
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_SUPPORTED_VERIF.copy()],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID
    )
    assert result.recommendation == "skip"
    assert result.skip_reason is not None
    assert result.draft_text is None


def test_all_supported_two_claims_returns_skip(monkeypatch):
    claim2 = {**_FAKE_CLAIM.copy(), "id": "claim-002", "claim_text": "Also correct"}
    verif2 = {**_FAKE_SUPPORTED_VERIF.copy(), "id": "verif-002", "claim_id": "claim-002"}
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy(), claim2],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_SUPPORTED_VERIF.copy(), verif2],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID
    )
    assert result.recommendation == "skip"


# ---------------------------------------------------------------------------
# Section E: unclear / mixed results
# ---------------------------------------------------------------------------

def test_insufficient_evidence_returns_unclear(monkeypatch):
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_INSUFFICIENT_VERIF.copy()],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID
    )
    assert result.recommendation == "unclear"
    assert result.draft_text is None


def test_mixed_supported_and_insufficient_returns_unclear(monkeypatch):
    claim2 = {**_FAKE_CLAIM.copy(), "id": "claim-002", "claim_text": "Another claim"}
    insuff2 = {**_FAKE_INSUFFICIENT_VERIF.copy(), "id": "verif-002", "claim_id": "claim-002"}
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy(), claim2],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_SUPPORTED_VERIF.copy(), insuff2],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID
    )
    assert result.recommendation == "unclear"


# ---------------------------------------------------------------------------
# Section F: idempotent rerun
# ---------------------------------------------------------------------------

def test_idempotent_rerun_clears_previous(monkeypatch, db):
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_REFUTED_VERIF.copy()],
    )
    analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, db=db
    )
    analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, db=db
    )
    # Should have exactly 1 claim, not 2
    count = db._conn.execute(
        "SELECT COUNT(*) FROM koala_extracted_claims WHERE comment_id=?",
        (_COMMENT_ID,),
    ).fetchone()[0]
    assert count == 1


def test_idempotent_rerun_recommendation_can_change(monkeypatch, db):
    calls = {"n": 0}

    def _extract(*a, **kw):
        calls["n"] += 1
        return [_FAKE_CLAIM.copy()]

    def _verify_first(*a, **kw):
        return [_FAKE_REFUTED_VERIF.copy()]

    def _verify_second(*a, **kw):
        return [_FAKE_SUPPORTED_VERIF.copy()]

    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        _extract,
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        _verify_first,
    )
    analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, db=db
    )
    assert db.get_reactive_analysis_for_comment(_COMMENT_ID)["recommendation"] == "react"

    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        _verify_second,
    )
    analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, db=db
    )
    assert db.get_reactive_analysis_for_comment(_COMMENT_ID)["recommendation"] == "skip"


# ---------------------------------------------------------------------------
# Section G: analyze_reactive_candidates_for_paper
# ---------------------------------------------------------------------------

def test_candidates_returns_empty_for_no_citable_comments(monkeypatch, db):
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [],
    )
    results = analyze_reactive_candidates_for_paper(_PAPER_ID, db)
    assert results == []


def test_candidates_processes_all_citable_comments(monkeypatch, db):
    from gsr_agent.koala.models import Comment, Paper
    from datetime import datetime, timezone

    _open = datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)
    paper = Paper(
        paper_id=_PAPER_ID, title="T", abstract="A", full_text="",
        pdf_url="", open_time=_open,
        review_end_time=_open, verdict_end_time=_open,
        state="REVIEW_ACTIVE",
    )
    db.upsert_paper(paper)

    for i in range(3):
        c = Comment(
            comment_id=f"cmt-{i}", paper_id=_PAPER_ID, thread_id=None, parent_id=None,
            author_agent_id=f"agent-{i}", text=f"Claim text {i}", created_at=_open,
            is_ours=False, is_citable=True,
        )
        db.upsert_comment(c)

    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [],
    )
    results = analyze_reactive_candidates_for_paper(_PAPER_ID, db)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, ReactiveAnalysisResult)
        assert r.paper_id == _PAPER_ID


def test_candidates_skips_non_citable_comments(monkeypatch, db):
    from gsr_agent.koala.models import Comment, Paper
    from datetime import datetime, timezone

    _open = datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)
    paper = Paper(
        paper_id=_PAPER_ID, title="T", abstract="A", full_text="",
        pdf_url="", open_time=_open,
        review_end_time=_open, verdict_end_time=_open,
        state="REVIEW_ACTIVE",
    )
    db.upsert_paper(paper)

    citable = Comment(
        comment_id="cmt-citable", paper_id=_PAPER_ID, thread_id=None, parent_id=None,
        author_agent_id="other", text="Claim text", created_at=_open,
        is_ours=False, is_citable=True,
    )
    not_citable = Comment(
        comment_id="cmt-not", paper_id=_PAPER_ID, thread_id=None, parent_id=None,
        author_agent_id="other2", text="Not citable", created_at=_open,
        is_ours=False, is_citable=False,
    )
    db.upsert_comment(citable)
    db.upsert_comment(not_citable)

    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [],
    )
    results = analyze_reactive_candidates_for_paper(_PAPER_ID, db)
    assert len(results) == 1
    assert results[0].comment_id == "cmt-citable"


# ---------------------------------------------------------------------------
# Section H: get_reactive_analysis_for_comment (DB round-trip)
# ---------------------------------------------------------------------------

def test_get_reactive_analysis_returns_none_before_any_run(db):
    assert db.get_reactive_analysis_for_comment("nonexistent-comment") is None


def test_get_reactive_analysis_full_round_trip(monkeypatch, db):
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_REFUTED_VERIF.copy()],
    )
    analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, db=db
    )
    analysis = db.get_reactive_analysis_for_comment(_COMMENT_ID)
    assert analysis is not None
    assert analysis["recommendation"] == "react"
    assert len(analysis["claims"]) == 1
    assert analysis["claims"][0]["claim_text"] == _FAKE_CLAIM["claim_text"]
    assert len(analysis["verifications"]) == 1
    assert analysis["verifications"][0]["verdict"] == "refuted"
    assert analysis["draft"] is not None
    assert analysis["draft"]["draft_text"] is not None
    assert "[DRY-RUN" in analysis["draft"]["draft_text"]


def test_get_reactive_analysis_analysis_json_parseable(monkeypatch, db):
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_REFUTED_VERIF.copy()],
    )
    analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, db=db
    )
    analysis = db.get_reactive_analysis_for_comment(_COMMENT_ID)
    summary = json.loads(analysis["draft"]["analysis_json"])
    assert summary["recommendation"] == "react"
    assert summary["claim_count"] == 1
    assert "refuted" in summary["verdict_counts"]


def test_get_reactive_analysis_skip_round_trip(monkeypatch, db):
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [],
    )
    analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, db=db
    )
    analysis = db.get_reactive_analysis_for_comment(_COMMENT_ID)
    assert analysis is not None
    assert analysis["recommendation"] == "skip"
    assert analysis["claims"] == []
    assert analysis["verifications"] == []
    assert analysis["draft"]["draft_text"] is None


# ---------------------------------------------------------------------------
# Phase 5A.5 audit: verdict alias handling (_is_contradiction_like_verdict)
# ---------------------------------------------------------------------------

from gsr_agent.commenting.reactive_analysis import _is_contradiction_like_verdict


def test_is_contradiction_like_refuted():
    assert _is_contradiction_like_verdict("refuted") is True


def test_is_contradiction_like_contradicted():
    assert _is_contradiction_like_verdict("contradicted") is True


def test_is_contradiction_like_contradiction():
    assert _is_contradiction_like_verdict("contradiction") is True


def test_is_contradiction_like_supported_is_false():
    assert _is_contradiction_like_verdict("supported") is False


def test_is_contradiction_like_insufficient_evidence_is_false():
    assert _is_contradiction_like_verdict("insufficient_evidence") is False


def test_is_contradiction_like_not_verifiable_is_false():
    assert _is_contradiction_like_verdict("not_verifiable") is False


def test_is_contradiction_like_empty_is_false():
    assert _is_contradiction_like_verdict("") is False


def test_contradicted_verdict_triggers_react(monkeypatch):
    """Historical 'contradicted' label (from older GSR rows) must trigger react."""
    contradicted_verif = {**_FAKE_REFUTED_VERIF.copy(), "verdict": "contradicted"}
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [contradicted_verif],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID
    )
    assert result.recommendation == "react"
    assert result.draft_text is not None
    assert "[DRY-RUN" in result.draft_text


def test_contradiction_verdict_triggers_react(monkeypatch):
    contradiction_verif = {**_FAKE_REFUTED_VERIF.copy(), "verdict": "contradiction"}
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [contradiction_verif],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID
    )
    assert result.recommendation == "react"


def test_supported_verdict_does_not_trigger_react(monkeypatch):
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_SUPPORTED_VERIF.copy()],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID
    )
    assert result.recommendation != "react"


def test_insufficient_evidence_does_not_trigger_react(monkeypatch):
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_INSUFFICIENT_VERIF.copy()],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID
    )
    assert result.recommendation != "react"


# ---------------------------------------------------------------------------
# Phase 5A.5 audit: get_phase5a_stats
# ---------------------------------------------------------------------------

def _populate_phase5a_rows(db, paper_id: str, comment_id: str, verdict: str) -> None:
    db.insert_extracted_claim({
        "claim_id": f"c-{comment_id}", "comment_id": comment_id,
        "paper_id": paper_id, "claim_text": "Some claim",
        "category": "results", "confidence": 0.8,
        "challengeability": 0.9, "binary_question": None,
    })
    db.insert_claim_verification({
        "verification_id": f"v-{comment_id}", "claim_id": f"c-{comment_id}",
        "comment_id": comment_id, "paper_id": paper_id,
        "verdict": verdict, "confidence": 0.75,
        "reasoning": "Test", "supporting_quote": None, "model_id": "gpt-4o",
    })
    rec = "react" if verdict in ("refuted", "contradicted") else "skip"
    db.insert_reactive_draft({
        "draft_id": f"d-{comment_id}", "comment_id": comment_id,
        "paper_id": paper_id, "recommendation": rec,
        "draft_text": None, "analysis_json": "{}",
    })


def test_phase5a_stats_empty_db(db):
    stats = db.get_phase5a_stats()
    assert stats["comments_analyzed"] == 0
    assert stats["claims_extracted"] == 0
    assert stats["claims_verified"] == 0
    assert stats["react_count"] == 0
    assert stats["skip_count"] == 0
    assert stats["unclear_count"] == 0


def test_phase5a_stats_mixed_verdicts(db):
    _populate_phase5a_rows(db, "p-001", "cmt-refuted", "refuted")
    _populate_phase5a_rows(db, "p-001", "cmt-supported", "supported")
    _populate_phase5a_rows(db, "p-001", "cmt-insufficient", "insufficient_evidence")

    stats = db.get_phase5a_stats()
    assert stats["comments_analyzed"] == 3
    assert stats["claims_extracted"] == 3
    assert stats["claims_verified"] == 3
    assert stats["react_count"] == 1
    assert stats["skip_count"] == 2
    assert stats["contradicted_count"] == 1  # only "refuted"
    assert stats["supported_count"] == 1
    assert stats["insufficient_count"] == 1


def test_phase5a_stats_paper_id_filter(db):
    _populate_phase5a_rows(db, "p-001", "cmt-a", "refuted")
    _populate_phase5a_rows(db, "p-002", "cmt-b", "supported")

    stats_p1 = db.get_phase5a_stats(paper_id="p-001")
    assert stats_p1["comments_analyzed"] == 1
    assert stats_p1["contradicted_count"] == 1
    assert stats_p1["supported_count"] == 0

    stats_p2 = db.get_phase5a_stats(paper_id="p-002")
    assert stats_p2["comments_analyzed"] == 1
    assert stats_p2["contradicted_count"] == 0
    assert stats_p2["supported_count"] == 1

    stats_all = db.get_phase5a_stats()
    assert stats_all["comments_analyzed"] == 2


def test_phase5a_stats_contradicted_label_counted(db):
    """Historical 'contradicted' verdict is included in contradicted_count."""
    _populate_phase5a_rows(db, "p-001", "cmt-old", "contradicted")
    stats = db.get_phase5a_stats()
    assert stats["contradicted_count"] == 1


def test_phase5a_stats_output_has_all_required_keys(db):
    stats = db.get_phase5a_stats()
    required = {
        "comments_analyzed", "claims_extracted", "claims_verified",
        "react_count", "skip_count", "unclear_count",
        "contradicted_count", "supported_count",
        "insufficient_count", "verification_error_count",
    }
    assert required.issubset(stats.keys())


# ---------------------------------------------------------------------------
# Section I: evidence-sparse aggressive-mode path
# ---------------------------------------------------------------------------

def test_evidence_sparse_normal_mode_still_unclear(monkeypatch):
    """Normal mode (aggressive_mode=False): all insufficient_evidence → 'unclear', no draft."""
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_INSUFFICIENT_VERIF.copy()],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID
    )
    assert result.recommendation == "unclear"
    assert result.draft_text is None


def test_evidence_sparse_aggressive_returns_evidence_sparse(monkeypatch):
    """Aggressive mode: all insufficient_evidence + claims → recommendation=evidence_sparse."""
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_INSUFFICIENT_VERIF.copy()],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, aggressive_mode=True
    )
    assert result.recommendation == "evidence_sparse"
    assert result.draft_text is not None
    assert "[DRY-RUN" in result.draft_text


def test_evidence_sparse_draft_contains_claim_text(monkeypatch):
    """Evidence-sparse draft includes the extracted claim text."""
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_INSUFFICIENT_VERIF.copy()],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, aggressive_mode=True
    )
    assert "95% accuracy" in result.draft_text


def test_evidence_sparse_requires_all_insufficient(monkeypatch):
    """If any verdict is not insufficient_evidence, do not produce evidence_sparse."""
    claim2 = {**_FAKE_CLAIM.copy(), "id": "claim-002", "claim_text": "Another claim"}
    mixed = [_FAKE_SUPPORTED_VERIF.copy(), _FAKE_INSUFFICIENT_VERIF.copy()]
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy(), claim2],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: mixed,
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, aggressive_mode=True
    )
    assert result.recommendation == "unclear"
    assert result.draft_text is None


def test_evidence_sparse_multiple_claims_aggressive(monkeypatch):
    """Multiple claims all returning insufficient_evidence → evidence_sparse in aggressive mode."""
    claim2 = {**_FAKE_CLAIM.copy(), "id": "claim-002", "claim_text": "Another verifiable claim"}
    insuff2 = {**_FAKE_INSUFFICIENT_VERIF.copy(), "id": "verif-002", "claim_id": "claim-002"}
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy(), claim2],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_INSUFFICIENT_VERIF.copy(), insuff2],
    )
    result = analyze_reactive_opportunity_for_comment(
        _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, aggressive_mode=True
    )
    assert result.recommendation == "evidence_sparse"
    assert result.draft_text is not None


def test_evidence_sparse_logs_comment_decision(monkeypatch, caplog):
    """[comment_decision] log is emitted for the evidence_sparse path."""
    import logging
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.extract_claims_from_koala_comment",
        lambda *a, **kw: [_FAKE_CLAIM.copy()],
    )
    monkeypatch.setattr(
        "gsr_agent.commenting.reactive_analysis.retrieve_and_verify_claims",
        lambda *a, **kw: [_FAKE_INSUFFICIENT_VERIF.copy()],
    )
    with caplog.at_level(logging.INFO, logger="gsr_agent.commenting.reactive_analysis"):
        analyze_reactive_opportunity_for_comment(
            _COMMENT_ID, _COMMENT_TEXT, _PAPER_ID, aggressive_mode=True
        )
    assert any("[comment_decision]" in r.message for r in caplog.records)
    assert any("decision=post" in r.message for r in caplog.records)
