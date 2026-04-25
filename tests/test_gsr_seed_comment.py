"""Tests for gsr_agent.commenting.seed_comment and adapters.gsr_runner."""

from datetime import datetime, timedelta, timezone

import pytest

from gsr_agent.adapters.gsr_runner import (
    PaperIndex,
    SeedEvidenceCandidate,
    get_paper_summary_sections,
    get_seed_evidence_candidates,
    index_paper_for_koala,
)
from gsr_agent.commenting.seed_comment import (
    choose_best_seed_comment,
    generate_seed_comment_candidates,
    score_seed_comment_candidate,
)
from gsr_agent.koala.models import Paper
from gsr_agent.rules.moderation import check_moderation

UTC = timezone.utc
_OPEN = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)


def _make_paper(abstract: str = "", full_text: str = "", domains=None) -> Paper:
    from gsr_agent.rules.timeline import compute_paper_windows
    w = compute_paper_windows(_OPEN)
    return Paper(
        paper_id="paper-001",
        title="Advances in Gradient Estimation",
        open_time=w.open_time,
        review_end_time=w.review_end_time,
        verdict_end_time=w.verdict_end_time,
        state="REVIEW_ACTIVE",
        abstract=abstract,
        full_text=full_text,
        domains=domains or ["ML"],
    )


# ---------------------------------------------------------------------------
# index_paper_for_koala
# ---------------------------------------------------------------------------

def test_index_paper_copies_basic_fields():
    paper = _make_paper(abstract="A new approach to X.")
    idx = index_paper_for_koala(paper)
    assert idx.paper_id == "paper-001"
    assert idx.title == "Advances in Gradient Estimation"
    assert idx.abstract == "A new approach to X."
    assert idx.domains == ["ML"]


def test_index_paper_empty_abstract():
    paper = _make_paper(abstract="")
    idx = index_paper_for_koala(paper)
    assert idx.abstract == ""


def test_index_paper_sections_is_dict():
    paper = _make_paper()
    idx = index_paper_for_koala(paper)
    assert isinstance(idx.sections, dict)


# ---------------------------------------------------------------------------
# get_paper_summary_sections (Tier 2 — workspace-based)
# ---------------------------------------------------------------------------

def test_summary_sections_returns_structured_result_when_db_missing(tmp_path):
    from gsr_agent.adapters.gsr_runner import PaperSummarySections
    result = get_paper_summary_sections("paper-001", workspace=tmp_path)
    assert isinstance(result, PaperSummarySections)
    assert result.ok is False
    assert result.sections == {}


def test_summary_sections_abstract_available_via_paper_index():
    paper = _make_paper(abstract="Main contribution: faster convergence.")
    idx = index_paper_for_koala(paper)
    # Abstract lives on PaperIndex.abstract (Tier 1); Tier 2 reads from GSR DB.
    assert idx.abstract == "Main contribution: faster convergence."


# ---------------------------------------------------------------------------
# get_seed_evidence_candidates
# ---------------------------------------------------------------------------

def test_no_candidates_for_empty_abstract():
    paper = _make_paper(abstract="")
    idx = index_paper_for_koala(paper)
    candidates = get_seed_evidence_candidates(idx)
    assert candidates == []


def test_candidates_returned_for_nonempty_abstract():
    paper = _make_paper(abstract="We propose a novel method for few-shot learning.")
    idx = index_paper_for_koala(paper)
    candidates = get_seed_evidence_candidates(idx)
    assert len(candidates) >= 1


def test_candidate_has_confidence_field():
    paper = _make_paper(abstract="Main claim: X outperforms Y.")
    idx = index_paper_for_koala(paper)
    cands = get_seed_evidence_candidates(idx)
    for c in cands:
        assert 0.0 <= c.confidence <= 1.0


def test_candidate_has_location_field():
    paper = _make_paper(abstract="Abstract text.")
    idx = index_paper_for_koala(paper)
    cands = get_seed_evidence_candidates(idx)
    for c in cands:
        assert isinstance(c.location, str)
        assert len(c.location) > 0


# ---------------------------------------------------------------------------
# generate_seed_comment_candidates
# ---------------------------------------------------------------------------

def test_generate_returns_list():
    paper = _make_paper(abstract="We present a novel approach to contrastive learning.")
    idx = index_paper_for_koala(paper)
    candidates = generate_seed_comment_candidates(idx)
    assert isinstance(candidates, list)


def test_generate_returns_at_least_one_for_paper_with_abstract():
    paper = _make_paper(abstract="We propose a method that achieves state-of-the-art results.")
    idx = index_paper_for_koala(paper)
    candidates = generate_seed_comment_candidates(idx)
    assert len(candidates) >= 1


def test_generate_returns_empty_for_no_abstract():
    paper = _make_paper(abstract="")
    idx = index_paper_for_koala(paper)
    candidates = generate_seed_comment_candidates(idx)
    assert candidates == []


def test_generate_candidates_are_strings():
    paper = _make_paper(abstract="Interesting abstract about deep learning.")
    idx = index_paper_for_koala(paper)
    for c in generate_seed_comment_candidates(idx):
        assert isinstance(c, str)
        assert len(c) > 0


def test_generate_candidates_pass_moderation():
    paper = _make_paper(abstract="We introduce a new benchmark for evaluating language models.")
    idx = index_paper_for_koala(paper)
    for candidate in generate_seed_comment_candidates(idx):
        passes, reason = check_moderation(candidate)
        assert passes, f"Candidate failed moderation: {reason!r}\nCandidate: {candidate!r}"


def test_generate_candidates_are_not_empty_strings():
    paper = _make_paper(abstract="A study on attention mechanisms in transformers.")
    idx = index_paper_for_koala(paper)
    for c in generate_seed_comment_candidates(idx):
        assert c.strip() != ""


# ---------------------------------------------------------------------------
# score_seed_comment_candidate
# ---------------------------------------------------------------------------

def test_score_returns_float():
    score = score_seed_comment_candidate("An interesting observation about the paper.", "paper-001")
    assert isinstance(score, float)


def test_score_is_between_0_and_1():
    for body in [
        "Thank you for this paper.",
        "The ablation in §4 appears to omit a baseline.",
        "",
    ]:
        score = score_seed_comment_candidate(body, "paper-001")
        assert 0.0 <= score <= 1.0, f"Score out of range for {body!r}: {score}"


def test_empty_comment_scores_zero():
    assert score_seed_comment_candidate("", "paper-001") == 0.0


def test_whitespace_comment_scores_zero():
    assert score_seed_comment_candidate("   ", "paper-001") == 0.0


def test_substantive_comment_scores_higher_than_empty():
    substantive = "The proposed method in §3 makes a strong assumption about data stationarity that may not hold in practice."
    empty_score = score_seed_comment_candidate("", "paper-001")
    sub_score = score_seed_comment_candidate(substantive, "paper-001")
    assert sub_score > empty_score


# ---------------------------------------------------------------------------
# choose_best_seed_comment
# ---------------------------------------------------------------------------

def test_choose_best_returns_none_for_empty_list():
    assert choose_best_seed_comment([]) is None


def test_choose_best_returns_string_for_nonempty_list():
    result = choose_best_seed_comment(["Candidate A.", "Candidate B with more detail."])
    assert isinstance(result, str)


def test_choose_best_selects_from_candidates():
    candidates = ["Short.", "Longer and more substantive comment about the methodology."]
    result = choose_best_seed_comment(candidates)
    assert result in candidates
