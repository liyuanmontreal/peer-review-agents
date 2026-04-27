# src/gsr/claim_extraction/scorer.py

from __future__ import annotations

import os
import re
import time
import logging
from dataclasses import dataclass
from typing import Optional

from huggingface_hub import InferenceClient

log = logging.getLogger(__name__)


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"

PROMPT_TEMPLATE = """You are evaluating whether an extracted claim is FAITHFULLY supported by the review text.

Important rule:
Score the claim based ONLY on what is explicitly stated or very directly implied in the review.
Do NOT reward the claim for being reasonable, plausible, or a good summary if it adds interpretation, exaggeration, stronger wording, or broader conclusions than the review.

Scoring rubric:
5 = The claim is a near-restatement of a specific explicit statement in the review. It is highly faithful and adds no meaningful new interpretation.
4 = The claim is supported with only minor paraphrase differences. It stays close to the review and does not materially strengthen or broaden the meaning.
3 = The claim is partly supported, but it adds noticeable interpretation, generalization, stronger wording, or mild overstatement.
2 = The claim has weak support and introduces substantial interpretation drift, missing nuance, or incorrect emphasis.
1 = The claim is unsupported, misleading, contradicted, or hallucinated relative to the review.

Use a strict standard:
- Use score 5 very sparingly.
- Only assign 5 when the claim is almost a direct restatement of a specific statement in the review.
- If the claim is even slightly stronger, broader, more definite, or more interpretive than the review, do NOT assign 5.
- If the claim is stronger or more generalized than the review, prefer 3 instead of 4.
- If the claim adds a conclusion not clearly present in the review, prefer 2 or 3 instead of 4.

Examples:

Example 1
Review:
The paper lacks comparisons with recent diffusion-based baselines.

Claim:
The review says the paper lacks comparisons with recent diffusion-based baselines.

Answer:
5

Example 2
Review:
The experiments are somewhat limited in scope.

Claim:
The experimental scope is limited.

Answer:
4

Example 3
Review:
The experiments are somewhat limited in scope.

Claim:
The experimental evaluation is insufficient.

Answer:
3

Example 4
Review:
The ablation study could be expanded.

Claim:
The paper has no meaningful ablation study.

Answer:
2

Now evaluate the following case.

Review:
{review}

Claim:
{claim}

Answer:
"""


@dataclass
class ClaimScorerConfig:
    enabled: bool = False
    provider: str = "featherless-ai"
    model: str = DEFAULT_MODEL
    api_key_env: str = "HF_TOKEN"
    max_new_tokens: int = 8
    temperature: float = 0.0
    timeout: Optional[float] = None


class ClaimScorer:
    def __init__(self, cfg: ClaimScorerConfig):
        self.cfg = cfg
        self._client: Optional[InferenceClient] = None

    def is_enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def _get_client(self) -> InferenceClient:
        if self._client is None:
            api_key = os.environ.get(self.cfg.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing API key env var: {self.cfg.api_key_env}"
                )
            self._client = InferenceClient(
                provider=self.cfg.provider,
                api_key=api_key,
                timeout=self.cfg.timeout,
            )
        return self._client

    @staticmethod
    def _normalize_text(text: str, max_chars: int = 4000) -> str:
        text = (text or "").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > max_chars:
            text = text[:max_chars]
        return text

    @staticmethod
    def _parse_score(text: str) -> Optional[int]:
        if not text:
            return None

        text = text.strip()

        patterns = [
            r'^\s*([1-5])\s*$',
            r'score\s*[:=]?\s*([1-5])',
            r'rating\s*[:=]?\s*([1-5])',
            r'\b([1-5])\b',
        ]

        for p in patterns:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))

        return None

    @staticmethod
    def _classify_error(e: Exception) -> tuple[str, str]:
        detail = f"{type(e).__name__}: {e}"
        lower = detail.lower()

        if "429" in lower or "rate limit" in lower or "too many requests" in lower:
            return "rate_limit", detail
        if "timeout" in lower or "timed out" in lower or "readtimeout" in lower:
            return "timeout", detail
        if "401" in lower or "403" in lower or "unauthorized" in lower or "forbidden" in lower:
            return "auth_error", detail
        if "502" in lower or "503" in lower or "504" in lower or "server error" in lower:
            return "http_error", detail

        return "scoring_exception", detail

    @staticmethod
    def score_to_norm(score: Optional[int]) -> Optional[float]:
        if score is None:
            return None
        mapping = {
            1: 0.10,
            2: 0.30,
            3: 0.50,
            4: 0.70,
            5: 0.90,
        }
        return mapping.get(score)

    def score_claim(self, review_text: str, claim_text: str) -> dict:
        if not self.cfg.enabled:
            return {
                "calibrated_score": None,
                "calibrated_score_norm": None,
                "calibrated_score_raw_text": None,
                "calibration_error": None,
                "calibration_error_detail": None,
            }

        review_text = self._normalize_text(review_text)
        claim_text = self._normalize_text(claim_text, max_chars=1000)

        if not review_text or not claim_text:
            return {
                "calibrated_score": None,
                "calibrated_score_norm": None,
                "calibrated_score_raw_text": None,
                "calibration_error": "empty_review_or_claim",
                "calibration_error_detail": None,
            }

        prompt = PROMPT_TEMPLATE.format(
            review=review_text,
            claim=claim_text,
        )

        client = self._get_client()

        raw_text = None
        last_exc = None

        for attempt in range(3):
            try:
                result = client.text_generation(
                    prompt,
                    model=self.cfg.model,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                )

                raw_text = str(result).strip()
                score = self._parse_score(raw_text)

                return {
                    "calibrated_score": score,
                    "calibrated_score_norm": self.score_to_norm(score),
                    "calibrated_score_raw_text": raw_text,
                    "calibration_error": None if score is not None else "parse_failed",
                    "calibration_error_detail": None if score is not None else raw_text,
                }

            except Exception as e:
                last_exc = e
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue

        err_code, err_detail = self._classify_error(last_exc)

        log.exception("Claim scoring failed")
        return {
            "calibrated_score": None,
            "calibrated_score_norm": None,
            "calibrated_score_raw_text": raw_text,
            "calibration_error": err_code,
            "calibration_error_detail": err_detail,
        }