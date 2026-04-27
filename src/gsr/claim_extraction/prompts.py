"""Prompt templates and Pydantic response models for claim extraction."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ClaimType(str, Enum):
    factual = "factual"
    subjective = "subjective"
    procedural = "procedural"
    comparative = "comparative"


class ClaimCategory(str, Enum):
    """rubric taxonomy for challengeable claims (strict 6-category)."""

    methodology = "methodology"
    results = "results"
    comparison = "comparison"
    literature = "literature"
    scope = "scope"
    reproducibility = "reproducibility"


class ExtractedClaim(BaseModel):
    """A single atomic claim extracted from a review."""

    claim_text: str = Field(
        description="A clear, self-contained restatement of the claim.",
    )
    verbatim_quote: str = Field(
        description="The exact span from the review that supports this claim.",
    )
    claim_type: Literal["factual", "subjective", "procedural", "comparative"] = Field(
        description="Category of the claim.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Extraction confidence (0-1).",
    )
    category: Literal[
        "methodology",
        "results",
        "comparison",
        "literature",
        "scope",
        "reproducibility",
    ] = Field(description="Challengeable-claim taxonomy category.")
    challengeability: float = Field(
        ge=0.0,
        le=1.0,
        description="How strongly this is binary-checkable by reading the paper (0-1).",
    )
    binary_question: str = Field(
        description="Yes/No question version of the claim, answerable by reading the paper.",
    )
    why_challengeable: str | None = Field(
        default=None,
        description="One sentence explaining why this claim is challengeable.",
    )


class ExtractionResponse(BaseModel):
    """LLM response containing a list of extracted claims."""
    claims: list[ExtractedClaim] = Field(default_factory=list)


class ReviewExtractedClaim(ExtractedClaim):
    source_field: Literal[
        "summary",
        "strengths",
        "weaknesses",
        "questions",
    ] = Field(
        description="Which section of the review the claim comes from."
    )


class ReviewExtractionResponse(BaseModel):
    claims: list[ReviewExtractedClaim] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
    You are an expert scientific peer-review analyst.

    Your job is to extract ONLY strictly challengeable factual claims from a review section.

    Definition (challengeable claim):

    A challengeable claim MUST:
    - Be a factual assertion made by the reviewer.
    - Assert the presence, absence, correctness, or incorrectness of a specific item.
    - Be verifiable as YES/NO by directly reading the paper.
    - Refer to something concretely identifiable (e.g., specific ablation, table, figure, metric, baseline, dataset, citation, training detail).

    A claim is NOT challengeable if:
    - It is an opinion or value judgment (e.g., "not convincing", "weak contribution").
    - It is vague (e.g., "experiments could be improved").
    - It is a suggestion or request without asserting a fact.
    - It requires interpretation rather than checking explicit presence/absence.

    Important:
    If a claim cannot be resolved by directly checking the paper content, DO NOT extract it.

    For each extracted claim, output:

    1) claim_text:
    A clear, self-contained reformulation of the claim.

    2) verbatim_quote:
    The exact span from the review.

    3) category:
    One of:
    - methodology (missing/incorrect experimental design or ablations)
    - results (numbers, error bars, statistical tests, quantitative evidence)
    - comparison (baselines, fairness of comparison, prior methods)
    - literature (citations, originality claims)
    - scope (datasets, domains, languages, evaluation conditions)
    - reproducibility (code, hyperparameters, training details, artifacts)

    4) confidence:
    Extraction confidence (0-1).

    5) challengeability:
    1.0 = explicit presence/absence claim
    0.7-0.9 = numeric or comparison mismatch
    0.5-0.7 = specific but slightly interpretive
    <0.5 should NOT be extracted.

    6) binary_question:
    Reformulate the claim as a strict YES/NO question.

    7) why_challengeable:
    One short sentence explaining why this can be checked directly in the paper.

    Return ONLY valid JSON matching the required schema.
    No commentary.
    """

USER_PROMPT_TEMPLATE = """\
Paper title: {paper_title}

Review section ({field_name}):
\"\"\"
{review_text}
\"\"\"

Extract all atomic claims from the review section above.\
"""

REVIEW_USER_PROMPT_TEMPLATE = """\
Paper title: {paper_title}

Review:

[summary]
{summary}

[strengths]
{strengths}

[weaknesses]
{weaknesses}

[questions]
{questions}

You are extracting challengeable factual claims from a peer review.

A challengeable factual claim is a statement that:
- asserts something about the paper's methods, results, comparisons, literature positioning, scope, or reproducibility
- can potentially be verified, contradicted, or judged as insufficiently supported by reading the paper
- is not merely a subjective preference, vague opinion, or generic praise/criticism

IMPORTANT: This is NOT a summarization task.
Do NOT return only the main points.
Return all challengeable factual claims from all provided review fields.

Process all provided review fields independently.
For each provided field:
- inspect every sentence
- extract all challengeable factual claims
- preserve source_field
- do not merge claims across fields

For each field:
- If the field is empty or missing, skip it.
- Independently inspect the field sentence by sentence.
- Extract all challengeable factual claims that could be checked against the paper.
- Preserve the source_field for every extracted claim.
- Prefer recall over aggressive deduplication.
- Do NOT skip a claim just because a similar claim appears in another field.
- Do NOT merge claims across different fields unless they are literally the same proposition.
- If a sentence contains multiple distinct factual propositions, extract them as separate claims.
- If a claim is expressed across multiple adjacent sentences, you may combine them into one claim.
- Keep claims even if they are tentative, hedged, or phrased as reviewer concerns/questions, as long as they make a checkable factual assertion.
- Especially pay attention to claims in weaknesses and questions, which are often easy to miss.

Important:
- Each claim MUST include source_field.
- source_field must be exactly one of:
  - summary
  - strengths
  - weaknesses
  - questions

Output requirements:
- Return JSON only.
- Return an array of claim objects.
- Do not include explanations outside JSON.
"""