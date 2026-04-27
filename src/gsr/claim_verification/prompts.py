"""Prompt templates and Pydantic response models for claim verification."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    supported = "supported"
    refuted = "refuted"
    insufficient_evidence = "insufficient_evidence"
    not_verifiable = "not_verifiable"


class VerificationResponse(BaseModel):
    verdict: Literal[
        "supported",
        "refuted",
        "insufficient_evidence",
        "not_verifiable",
    ] = Field(
        description=(
            "Overall verdict: 'supported' if the evidence confirms the claim, "
            "'refuted' if it contradicts it, 'insufficient_evidence' if the "
            "evidence does not contain enough information, or 'not_verifiable' "
            "if the claim is opinion-based or inherently untestable."
        ),
    )
    reasoning: str = Field(
        description=(
            "A concise explanation (2–4 sentences) citing the most relevant "
            "evidence items."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the verdict (0 = uncertain, 1 = certain).",
    )
    supporting_quote: str = Field(
        default="",
        description=(
            "The single most relevant passage or caption excerpt supporting the verdict. "
            "Empty if no single quote stands out."
        ),
    )


SYSTEM_PROMPT = """\
You are an expert scientific fact-checker specialising in machine-learning research.

You will be given:
1. A claim extracted from a peer review, along with its verbatim source quote.
2. A numbered list of evidence items retrieved from the paper being reviewed.

The evidence may include:
- Text evidence: normal passages from the paper text.
- Table evidence: a table object, including its label (e.g. Table 3), caption,
  and extracted or nearby text.
- Figure evidence: a figure object, including its label (e.g. Figure 2), caption,
  and nearby explanatory text.

Your task is to determine whether the paper's own content SUPPORTS, REFUTES,
or provides INSUFFICIENT EVIDENCE for the claim.

Verdict definitions
-------------------
- supported             – The evidence directly and unambiguously confirms the claim.
- refuted               – The evidence directly contradicts the claim.
- insufficient_evidence – The evidence does not contain enough information to decide.
- not_verifiable        – The claim is an opinion, value judgement, or otherwise
                          untestable against the paper content.

Rules
-----
- Reason ONLY from the provided evidence items. Do not use outside knowledge.
- Prefer a conservative verdict: choose 'insufficient_evidence' over 'supported'
  or 'refuted' when the evidence is ambiguous.
- Set confidence low (< 0.4) whenever the evidence is sparse, indirect, or only
  partially relevant.
- If the claim explicitly references a table or figure number (e.g. "Table 3",
  "Fig. 2"), prioritise evidence items whose labels match that reference.
- For claims about performance, comparisons, ablations, or numerical results,
  table evidence is often more probative than nearby prose.
- For claims about architecture, pipeline, framework, modules, or qualitative
  examples, figure evidence may be more probative than nearby prose.
- Do not ignore captions. Captions often state the intended meaning of a table
  or figure.
- If multiple evidence items conflict, explain which one is more directly relevant
  to the claim.
- Return ONLY valid JSON matching the required schema. Do not include any extra text.
"""

USER_PROMPT_TEMPLATE = """\
## Claim to verify

Claim text: {claim_text}
Verbatim quote from review: "{verbatim_quote}"
Claim type: {claim_type}

## Evidence items from the paper

{evidence_block}

Based solely on the evidence items above, classify the claim and explain your reasoning.
"""


MAX_TEXT_LEN_TEXT = 1400
MAX_TEXT_LEN_TABLE = 1800
MAX_TEXT_LEN_FIGURE = 1400


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def _truncate(text: str, limit: int) -> str:
    text = _clean_text(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _normalize_object_type(item: dict) -> str:
    raw = str(item.get("object_type") or item.get("evidence_type") or "").strip().lower()
    if raw == "table":
        return "table"
    if raw == "figure":
        return "figure"
    return "text"


def _pick_label(item: dict) -> str:
    return _clean_text(
        item.get("label")
        or item.get("evidence_label")
        or item.get("object_label")
        or ""
    )


def _pick_caption(item: dict) -> str:
    return _clean_text(item.get("caption_text") or item.get("caption") or "")


def _pick_section(item: dict) -> str:
    return _clean_text(item.get("section") or "unknown")


def _pick_page(item: dict) -> str:
    page = item.get("page")
    if page in (None, "", 0):
        page = item.get("page_num")
    if page in (None, "", 0):
        page = "?"
    return str(page)


def _pick_text(item: dict, object_type: str) -> str:
    text = _clean_text(
        item.get("text")
        or item.get("content_text")
        or item.get("retrieval_text")
        or ""
    )
    if object_type == "table":
        return _truncate(text, MAX_TEXT_LEN_TABLE)
    if object_type == "figure":
        return _truncate(text, MAX_TEXT_LEN_FIGURE)
    return _truncate(text, MAX_TEXT_LEN_TEXT)


def _format_metadata_lines(item: dict) -> list[str]:
    lines: list[str] = []
    lines.append(f"Section: {_pick_section(item)}")
    lines.append(f"Page: {_pick_page(item)}")

    label = _pick_label(item)
    if label:
        lines.append(f"Label: {label}")

    if item.get("reference_matched"):
        lines.append("Reference match: yes")
    elif isinstance(item.get("reference_boost"), (int, float)) and item.get("reference_boost", 0.0) > 0:
        lines.append(f"Reference boost: {float(item.get('reference_boost')):.2f}")

    return lines


def _format_text_item(rank: int, item: dict) -> str:
    meta = _format_metadata_lines(item)
    text = _pick_text(item, "text")

    parts = [f"[Evidence E{rank} | Type: Text]"]
    parts.extend(meta)
    parts.append("Content:")
    parts.append(text or "(No text available)")
    return "\n".join(parts)


def _format_table_item(rank: int, item: dict) -> str:
    meta = _format_metadata_lines(item)
    caption = _pick_caption(item)
    text = _pick_text(item, "table")

    parts = [f"[Evidence E{rank} | Type: Table]"]
    parts.extend(meta)
    if caption:
        parts.append("Caption:")
        parts.append(caption)
    parts.append("Content:")
    parts.append(text or "(No extracted table text available)")
    return "\n".join(parts)


def _format_figure_item(rank: int, item: dict) -> str:
    meta = _format_metadata_lines(item)
    caption = _pick_caption(item)
    text = _pick_text(item, "figure")
    ocr_text = _truncate(_clean_text(item.get("figure_ocr_text") or ""), MAX_TEXT_LEN_FIGURE)

    parts = [f"[Evidence E{rank} | Type: Figure]"]
    parts.extend(meta)
    if caption:
        parts.append("Caption:")
        parts.append(caption)

    # Figure semantic enrichment block (Phase P2 — Qwen2.5-VL)
    sem = item.get("figure_semantics")
    if sem and isinstance(sem, dict):
        summary = _truncate(_clean_text(sem.get("figure_summary") or ""), 300)
        if summary:
            parts.append(f"Figure summary: {summary}")
        key_obs = sem.get("key_observations") or []
        if key_obs:
            parts.append("Key observations:")
            for obs in key_obs[:3]:
                parts.append(f"- {_truncate(_clean_text(str(obs)), 150)}")
        poss_use = sem.get("possible_verification_use") or []
        if poss_use:
            parts.append("Possible verification use:")
            for u in poss_use[:2]:
                parts.append(f"- {_truncate(_clean_text(str(u)), 150)}")
        if sem.get("needs_precise_ocr") is True:
            parts.append("Needs precise OCR: yes")

    if ocr_text:
        parts.append("Figure text (OCR):")
        parts.append(ocr_text)
    parts.append("Content:")
    parts.append(text or "(No extracted figure context available)")
    return "\n".join(parts)


def format_evidence_item(rank: int, item: dict) -> str:
    object_type = _normalize_object_type(item)
    if object_type == "table":
        return _format_table_item(rank, item)
    if object_type == "figure":
        return _format_figure_item(rank, item)
    return _format_text_item(rank, item)


def format_evidence_block(evidence_chunks: list[dict]) -> str:
    if not evidence_chunks:
        return "(No evidence items retrieved.)"
    return "\n\n".join(
        format_evidence_item(i + 1, c)
        for i, c in enumerate(evidence_chunks)
    )


def build_messages(
    claim: dict,
    evidence_chunks: list[dict],
    figure_images: dict[str, str] | None = None,
) -> list[dict]:
    """Build the LLM message list for claim verification.

    Args:
        claim: The claim dict.
        evidence_chunks: Prepared evidence items.
        figure_images: Optional mapping of {evidence_object_id: base64_png}.
            When provided and non-empty, the user message is built as a
            multimodal content list (text + image_url blocks).
            When None or empty, the original text-only format is preserved
            exactly — backward compatible with all existing callers.

    Image format: OpenAI / LiteLLM vision API —
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<b64>"}}
    """
    evidence_block = format_evidence_block(evidence_chunks)
    user_content = USER_PROMPT_TEMPLATE.format(
        claim_text=claim.get("claim_text", ""),
        verbatim_quote=claim.get("verbatim_quote", ""),
        claim_type=claim.get("claim_type", "unknown"),
        evidence_block=evidence_block,
    )

    # Text-only path — preserves exact existing behaviour.
    if not figure_images:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # Multimodal path — text block first, then one image block per attached figure.
    user_content_blocks: list[dict] = [{"type": "text", "text": user_content}]
    for evidence_object_id, b64_png in figure_images.items():
        user_content_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_png}"},
        })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content_blocks},
    ]