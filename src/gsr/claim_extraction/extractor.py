"""Main orchestrator for claim extraction."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from concurrent.futures import ThreadPoolExecutor, as_completed

from gsr.utils.timing import timed
from gsr.claim_extraction.llm import complete, get_model_id
from gsr.claim_extraction.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    REVIEW_USER_PROMPT_TEMPLATE,
    ExtractionResponse,
    ReviewExtractionResponse,
)
from gsr.claim_extraction.scorer import ClaimScorer, ClaimScorerConfig
from gsr.claim_extraction.field_policy import (
    canonicalize_field_name,
    should_extract_field,
    FIELD_POLICY_VERSION,
)

log = logging.getLogger(__name__)

DEFAULT_FIELDS = ("weaknesses", "strengths", "questions", "summary")

# ---------------------------------------------------------------------------
# Threshold defaults (CLI can override these)
# ---------------------------------------------------------------------------

MIN_CHALLENGEABILITY: float = 0.6
MIN_CONFIDENCE: float = 0.5

ALLOWED_CATEGORIES = {
    "methodology",
    "results",
    "comparison",
    "literature",
    "scope",
    "reproducibility",
}

CATEGORY_ALIASES = {
    "methods": "methodology",
    "method": "methodology",
    "experiment": "methodology",
    "experiments": "methodology",
    "evaluation": "results",
    "result": "results",
    "baselines": "comparison",
    "baseline": "comparison",
    "related_work": "literature",
    "related work": "literature",
    "citations": "literature",
    "citation": "literature",
    "dataset": "scope",
    "datasets": "scope",
    "limitations": "scope",
    "reproducible": "reproducibility",
    "reproducibility/replicability": "reproducibility",
    "code": "reproducibility",
    "implementation": "reproducibility",
}

ALLOWED_REVIEW_FIELDS = {"summary", "strengths", "weaknesses", "questions"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_category(cat: str | None) -> str | None:
    if not cat:
        return None
    c = str(cat).strip().lower().replace("-", "_")
    c = CATEGORY_ALIASES.get(c, c)
    return c if c in ALLOWED_CATEGORIES else None


def normalize_source_field(raw: str | None) -> str | None:
    if not raw:
        return None

    s = str(raw).strip().lower()
    mapping = {
        "summary": "summary",
        "strength": "strengths",
        "strengths": "strengths",
        "weakness": "weaknesses",
        "weaknesses": "weaknesses",
        "question": "questions",
        "questions": "questions",
    }
    return mapping.get(s)


def _config_hash(
    *,
    model_id: str,
    fields: tuple[str, ...],
    min_confidence: float,
    min_challengeability: float,
) -> str:
    raw = (
        f"{model_id}|{','.join(fields)}"
        f"|min_conf={min_confidence}|min_chal={min_challengeability}"
        f"|field_policy_version={FIELD_POLICY_VERSION}"
    )
    log.debug("extraction config field_policy_version=%s", FIELD_POLICY_VERSION)
    return hashlib.sha256(raw.encode()).hexdigest()


def _resolve_extraction_fields(
    review: dict[str, Any],
    default_fields: tuple[str, ...],
) -> list[tuple[str, str]]:
    """Return ``[(canonical_field, raw_field_key), ...]`` to extract from.

    When ``review["raw_fields"]`` (JSON blob from DB) is present, the full set
    of content fields is evaluated against the field policy.  Otherwise falls
    back to *default_fields* (backward-compatible path).

    Side-effect: overlays text from raw_fields onto the review dict so that
    ``review.get(canonical_field)`` works for newly discovered fields.
    """
    raw_fields_json = review.get("raw_fields")
    if not raw_fields_json:
        return [(f, f) for f in default_fields]

    try:
        raw_fields: dict[str, Any] = json.loads(raw_fields_json)
    except Exception:
        return [(f, f) for f in default_fields]

    canonical_to_raw: dict[str, str] = {}
    for raw_key, raw_text in raw_fields.items():
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue
        canonical = canonicalize_field_name(raw_key)
        if not canonical:
            continue
        # First raw key seen for this canonical wins
        if canonical not in canonical_to_raw:
            canonical_to_raw[canonical] = raw_key
            # Overlay onto review dict if not already a named DB column with content
            if not review.get(canonical):
                review[canonical] = raw_text

    result: list[tuple[str, str]] = []
    for canonical, raw_key in canonical_to_raw.items():
        text = review.get(canonical, "")
        include, reason = should_extract_field(canonical, text)
        log.info(
            "field_policy review=%s field=%s raw=%r include=%s reason=%s",
            review.get("id", "?"),
            canonical,
            raw_key,
            include,
            reason,
        )
        if include:
            result.append((canonical, raw_key))

    return result


def _load_reviews(
    conn: sqlite3.Connection,
    limit: int | None = None,
    paper_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load reviews from SQLite, joining paper title."""
    # Conditionally include raw_fields if the column exists
    review_cols = {row[1] for row in conn.execute("PRAGMA table_info(reviews)")}
    extra = ", r.raw_fields" if "raw_fields" in review_cols else ""

    query = (
        "SELECT r.id, r.paper_id, p.title AS paper_title, "
        f"r.summary, r.strengths, r.weaknesses, r.questions{extra} "
        "FROM reviews r JOIN papers p ON r.paper_id = p.id"
    )
    params: list[Any] = []

    if paper_id is not None:
        query += " WHERE r.paper_id = ?"
        params.append(paper_id)

    query += " ORDER BY r.id"

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    conn.row_factory = sqlite3.Row
    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def _already_processed(conn: sqlite3.Connection, review_id: str, config_hash: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM extraction_runs WHERE review_id=? AND config_hash=? AND status='success' LIMIT 1",
        (review_id, config_hash),
    ).fetchone()
    return row is not None


def _claim_id(review_id: str, source_field: str, claim_index: int, config_hash: str) -> str:
    s = f"{review_id}|{source_field}|{claim_index}|{config_hash}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Post-filtering
# ---------------------------------------------------------------------------

def _keep_claim(c: dict[str, Any], *, min_confidence: float, min_challengeability: float) -> bool:
    if (c.get("challengeability") or 0.0) < min_challengeability:
        return False
    if (c.get("confidence") or 0.0) < min_confidence:
        return False
    return True


# ---------------------------------------------------------------------------
# Single-field extraction
# ---------------------------------------------------------------------------

def _extract_claims_from_field(
    review_id: str,
    paper_title: str,
    field_name: str,
    review_text: str,
    cfg_hash: str,
    field_name_raw: str | None = None,
    model: str | None = None,
    min_confidence: float = MIN_CONFIDENCE,
    min_challengeability: float = MIN_CHALLENGEABILITY,
    on_field_done=None,
) -> list[dict[str, Any]]:
    if not review_text or not review_text.strip():
        if on_field_done:
            try:
                on_field_done(
                    review_id=review_id,
                    field_name=field_name,
                    raw_count=0,
                    kept_count=0,
                    skipped_empty=True,
                )
            except Exception:
                pass
        return []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            paper_title=paper_title or "Unknown",
            field_name=field_name,
            review_text=review_text,
        )},
    ]

    response: ExtractionResponse = complete(
        messages=messages,
        response_format=ExtractionResponse,
        model=model,
    )

    raw_claims: list[dict[str, Any]] = []

    for idx, claim in enumerate(getattr(response, "claims", []) or []):
        raw_cat = claim.category
        cat = normalize_category(raw_cat)
        if cat is None:
            log.debug(
                "Drop claim due to category: raw=%r allowed=%s",
                raw_cat,
                sorted(ALLOWED_CATEGORIES),
            )
            continue

        raw_claims.append({
            "id": _claim_id(review_id, field_name, idx, cfg_hash),
            "source_field": field_name,
            "source_field_raw": field_name_raw,
            "claim_index": idx,
            "claim_text": claim.claim_text,
            "verbatim_quote": claim.verbatim_quote,
            "claim_type": claim.claim_type,
            "confidence": claim.confidence,
            "category": cat,
            "challengeability": claim.challengeability,
            "binary_question": claim.binary_question,
            "why_challengeable": getattr(claim, "why_challengeable", None),
        })

    kept_claims = [
        c for c in raw_claims
        if _keep_claim(
            c,
            min_confidence=min_confidence,
            min_challengeability=min_challengeability,
        )
    ]

    log.info(
        "extract_field review=%s field=%s raw=%d kept=%d (min_conf=%.2f min_chal=%.2f) model=%s",
        review_id,
        field_name,
        len(raw_claims),
        len(kept_claims),
        float(min_confidence),
        float(min_challengeability),
        model or "default",
    )

    if len(raw_claims) == 0:
        log.warning("LLM returned 0 claims: review=%s field=%s", review_id, field_name)

    if on_field_done:
        try:
            on_field_done(
                review_id=review_id,
                field_name=field_name,
                raw_count=len(raw_claims),
                kept_count=len(kept_claims),
                skipped_empty=False,
            )
        except Exception:
            pass

    return kept_claims


# ---------------------------------------------------------------------------
# Single-review extraction (one LLM call for all fields)
# ---------------------------------------------------------------------------

def _extract_claims_from_review(
    review: dict[str, Any],
    cfg_hash: str,
    model: str | None = None,
    min_confidence: float = MIN_CONFIDENCE,
    min_challengeability: float = MIN_CHALLENGEABILITY,
) -> list[dict[str, Any]]:
    review_id = review["id"]
    paper_title = review.get("paper_title", "")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": REVIEW_USER_PROMPT_TEMPLATE.format(
                paper_title=paper_title or "Unknown",
                summary=review.get("summary") or "",
                strengths=review.get("strengths") or "",
                weaknesses=review.get("weaknesses") or "",
                questions=review.get("questions") or "",
            ),
        },
    ]

    response: ReviewExtractionResponse = complete(
        messages=messages,
        response_format=ReviewExtractionResponse,
        model=model,
    )

    raw_claims: list[dict[str, Any]] = []

    for idx, claim in enumerate(getattr(response, "claims", []) or []):
        source_field = normalize_source_field(getattr(claim, "source_field", None))
        if source_field is None:
            log.warning(
                "Drop claim due to invalid source_field: review=%s raw_source_field=%r",
                review_id,
                getattr(claim, "source_field", None),
            )
            continue

        raw_cat = claim.category
        cat = normalize_category(raw_cat)
        if cat is None:
            log.debug(
                "Drop claim due to category: raw=%r allowed=%s",
                raw_cat,
                sorted(ALLOWED_CATEGORIES),
            )
            continue

        raw_claims.append({
            "id": _claim_id(review_id, source_field, idx, cfg_hash),
            "source_field": source_field,
            "claim_index": idx,
            "claim_text": claim.claim_text,
            "verbatim_quote": claim.verbatim_quote,
            "claim_type": claim.claim_type,
            "confidence": claim.confidence,
            "category": cat,
            "challengeability": claim.challengeability,
            "binary_question": claim.binary_question,
            "why_challengeable": getattr(claim, "why_challengeable", None),
        })

    kept_claims = [
        c for c in raw_claims
        if _keep_claim(
            c,
            min_confidence=min_confidence,
            min_challengeability=min_challengeability,
        )
    ]

    log.info(
        "extract_review_single_call review=%s raw=%d kept=%d (min_conf=%.2f min_chal=%.2f) model=%s",
        review_id,
        len(raw_claims),
        len(kept_claims),
        float(min_confidence),
        float(min_challengeability),
        model or "default",
    )

    if len(raw_claims) == 0:
        log.warning("LLM returned 0 claims: review=%s mode=review", review_id)

    return kept_claims


# ---------------------------------------------------------------------------
# Per-review extraction
# ---------------------------------------------------------------------------

from datetime import datetime, timezone
from typing import Any

def extract_review_claims(
    review: dict[str, Any],
    model: str | None = None,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
    min_confidence: float = MIN_CONFIDENCE,
    min_challengeability: float = MIN_CHALLENGEABILITY,
    scorer: ClaimScorer | None = None,
    on_field_done=None,
    extract_mode: str = "field",
) -> dict[str, Any]:
    """Extract claims from one review.

    Modes:
    - field: existing behavior, one LLM call per field
    - review: one LLM call per full review, with source_field returned by model
    - grouped: two review-level calls
        group 1 = summary + strengths
        group 2 = weaknesses + questions
    """
    review_id = review["id"]
    model_id = get_model_id(model)

    with timed(
        "extract_review_build_config_hash",
        review_id=review_id,
        model_id=model_id,
        fields=",".join(fields),
    ):
        cfg = _config_hash(
            model_id=model_id,
            fields=fields,
            min_confidence=min_confidence,
            min_challengeability=min_challengeability,
        )

    started_at = datetime.now(timezone.utc).isoformat()
    all_claims: list[dict[str, Any]] = []

    try:
        # ------------------------------------------------------------
        # Mode A: review-level single call
        # ------------------------------------------------------------
        if extract_mode == "review":
            with timed(
                "extract_review_single_call_total",
                review_id=review_id,
                field_count=len(fields),
            ):
                all_claims = _extract_claims_from_review(
                    review=review,
                    cfg_hash=cfg,
                    model=model,
                    min_confidence=min_confidence,
                    min_challengeability=min_challengeability,
                )

            log.info(
                "Review extraction done: review_id=%s mode=review claims=%d",
                review_id,
                len(all_claims),
            )

            if scorer is not None and scorer.is_enabled():
                with timed(
                    "extract_review_score_claims",
                    review_id=review_id,
                    claim_count=len(all_claims),
                ):
                    field_text_map = {
                        f: (review.get(f) or "")
                        for f in fields
                    }
                    for c in all_claims:
                        review_text = field_text_map.get(c.get("source_field"), "")
                        scoring = scorer.score_claim(
                            review_text=review_text,
                            claim_text=c.get("claim_text", ""),
                        )
                        c.update(scoring)

        # ------------------------------------------------------------
        # Mode B: grouped review-level extraction (2 calls)
        # ------------------------------------------------------------
        elif extract_mode == "grouped":
            group_specs = [
                ("summary_strengths", ("summary", "strengths")),
                ("weaknesses_questions", ("weaknesses", "questions")),
            ]

            with timed(
                "extract_review_grouped_total",
                review_id=review_id,
                field_count=len(fields),
                group_count=2,
            ):
                for group_name, group_fields in group_specs:
                    active_group_fields = tuple(f for f in group_fields if f in fields)

                    if not active_group_fields:
                        log.info(
                            "Grouped extraction skip: review_id=%s group=%s reason=no_active_fields",
                            review_id,
                            group_name,
                        )
                        continue

                    group_review = dict(review)
                    for f in DEFAULT_FIELDS:
                        if f not in active_group_fields:
                            group_review[f] = ""

                    if not any((group_review.get(f) or "").strip() for f in active_group_fields):
                        log.info(
                            "Grouped extraction skip: review_id=%s group=%s reason=all_empty fields=%s",
                            review_id,
                            group_name,
                            ",".join(active_group_fields),
                        )
                        continue

                    with timed(
                        "extract_review_group_call_total",
                        review_id=review_id,
                        group_name=group_name,
                        fields=",".join(active_group_fields),
                    ):
                        group_claims = _extract_claims_from_review(
                            review=group_review,
                            cfg_hash=cfg,
                            model=model,
                            min_confidence=min_confidence,
                            min_challengeability=min_challengeability,
                        )

                    log.info(
                        "Grouped extraction done: review_id=%s group=%s fields=%s claims=%d",
                        review_id,
                        group_name,
                        ",".join(active_group_fields),
                        len(group_claims),
                    )

                    if scorer is not None and scorer.is_enabled():
                        with timed(
                            "extract_review_group_score_claims",
                            review_id=review_id,
                            group_name=group_name,
                            claim_count=len(group_claims),
                        ):
                            field_text_map = {
                                f: (review.get(f) or "")
                                for f in active_group_fields
                            }
                            for c in group_claims:
                                review_text = field_text_map.get(c.get("source_field"), "")
                                scoring = scorer.score_claim(
                                    review_text=review_text,
                                    claim_text=c.get("claim_text", ""),
                                )
                                c.update(scoring)

                    all_claims.extend(group_claims)

            log.info(
                "Review extraction done: review_id=%s mode=grouped claims=%d",
                review_id,
                len(all_claims),
            )

        # ------------------------------------------------------------
        # Mode C: existing field-level extraction
        # ------------------------------------------------------------
        else:
            # _resolve_extraction_fields uses field_policy when raw_fields is
            # present; otherwise falls back to the caller-supplied `fields` tuple.
            fields_to_extract = _resolve_extraction_fields(review, fields)

            with timed(
                "extract_review_fields_loop_total",
                review_id=review_id,
                field_count=len(fields_to_extract),
            ):
                for field, field_raw in fields_to_extract:
                    text = review.get(field)

                    if not text:
                        with timed(
                            "extract_field_empty",
                            review_id=review_id,
                            field_name=field,
                        ):
                            if on_field_done:
                                try:
                                    on_field_done(
                                        review_id=review["id"],
                                        field_name=field,
                                        raw_count=0,
                                        kept_count=0,
                                        skipped_empty=True,
                                    )
                                except Exception:
                                    pass
                        continue

                    with timed(
                        "extract_field_total",
                        review_id=review_id,
                        field_name=field,
                        text_len=len(text),
                    ):
                        with timed(
                            "extract_field_claims_call",
                            review_id=review_id,
                            field_name=field,
                            text_len=len(text),
                        ):
                            field_claims = _extract_claims_from_field(
                                review_id=review["id"],
                                paper_title=review.get("paper_title", ""),
                                field_name=field,
                                field_name_raw=field_raw if field_raw != field else None,
                                review_text=text,
                                model=model,
                                min_confidence=min_confidence,
                                min_challengeability=min_challengeability,
                                cfg_hash=cfg,
                                on_field_done=on_field_done,
                            )

                        log.info(
                            "Field extraction done: review_id=%s field=%s claims=%d",
                            review_id,
                            field,
                            len(field_claims),
                        )

                        if scorer is not None and scorer.is_enabled():
                            with timed(
                                "extract_field_score_claims",
                                review_id=review_id,
                                field_name=field,
                                claim_count=len(field_claims),
                            ):
                                for c in field_claims:
                                    scoring = scorer.score_claim(
                                        review_text=text,
                                        claim_text=c.get("claim_text", ""),
                                    )
                                    c.update(scoring)

                        with timed(
                            "extract_field_extend_results",
                            review_id=review_id,
                            field_name=field,
                            claim_count=len(field_claims),
                        ):
                            all_claims.extend(field_claims)

        return {
            "review_id": review["id"],
            "paper_id": review["paper_id"],
            "model_id": model_id,
            "status": "success",
            "error": None,
            "claims": all_claims,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "min_confidence": min_confidence,
            "min_challengeability": min_challengeability,
            "fields": ",".join(fields),
            "config_hash": cfg,
        }

    except Exception as exc:
        if extract_mode in {"review", "grouped"}:
            log.exception(
                "Non-field extraction failed for review %s, falling back to field mode: %s",
                review["id"],
                exc,
            )
            return extract_review_claims(
                review,
                model=model,
                fields=fields,
                min_confidence=min_confidence,
                min_challengeability=min_challengeability,
                scorer=scorer,
                on_field_done=on_field_done,
                extract_mode="field",
            )

        log.error("Extraction failed for review %s: %s", review["id"], exc)
        return {
            "review_id": review["id"],
            "paper_id": review["paper_id"],
            "model_id": model_id,
            "status": "error",
            "error": str(exc),
            "claims": [],
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "min_confidence": min_confidence,
            "min_challengeability": min_challengeability,
            "fields": ",".join(fields),
            "config_hash": cfg,
        }

# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_all_claims(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    paper_id: str | None = None,
    model: str | None = None,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
    min_confidence: float = MIN_CONFIDENCE,
    min_challengeability: float = MIN_CHALLENGEABILITY,
    force: bool = False,
    delay: float = 0.1,
    score_claims: bool = False,
    scorer_model: str = "meta-llama/Llama-3.1-8B",
    max_workers: int = 2,
    extract_mode: str = "field",
) -> dict[str, Any]:
    """Extract claims from all (or a subset of) reviews in the database.

    Returns a summary dict:
    ``{"reviews_processed", "claims_extracted", "errors", "results"}``.
    """
    with timed(
        "extract_all_claims_total",
        paper_id=paper_id,
        limit=limit,
        model=model,
        fields=",".join(fields),
        score_claims=score_claims,
        force=force,
        max_workers=max_workers,
    ):
        with timed(
            "extract_init_scorer",
            score_claims=score_claims,
            scorer_model=scorer_model,
        ):
            scorer = ClaimScorer(
                ClaimScorerConfig(
                    enabled=score_claims,
                    model=scorer_model,
                    provider="featherless-ai",
                    api_key_env="HF_TOKEN",
                )
            )

        log.debug("extract scorer: score_claims=%s enabled=%s model=%s", score_claims, scorer.is_enabled(), scorer.cfg.model)

        from gsr.claim_extraction.storage import ensure_experiment_schema, create_experiment

        with timed("extract_ensure_experiment_schema", paper_id=paper_id):
            ensure_experiment_schema(conn)

        with timed("extract_create_experiment", paper_id=paper_id):
            experiment_id = create_experiment(
                conn,
                paper_id=paper_id,
                command="gsr extract",
                params={
                    "model": model,
                    "fields": list(fields),
                    "min_confidence": min_confidence,
                    "min_challengeability": min_challengeability,
                    "max_workers": max_workers,
                    "extract_mode": extract_mode,
                },
            )

        with timed("extract_load_reviews", paper_id=paper_id, limit=limit):
            reviews = _load_reviews(conn, limit=limit, paper_id=paper_id)

        if len(reviews) == 0:
            log.info("[extract] SKIP no_reviews paper_id=%s", paper_id)
        else:
            log.info("[extract] START paper_id=%s reviews=%d", paper_id, len(reviews))
        log.debug("extract loaded reviews: paper_id=%s count=%d", paper_id, len(reviews))

        total_claims = 0
        errors = 0

        model_id = get_model_id(model)

        with timed(
            "extract_build_config_hash",
            model_id=model_id,
            fields=",".join(fields),
        ):
            cfg = _config_hash(
                model_id=model_id,
                fields=fields,
                min_confidence=min_confidence,
                min_challengeability=min_challengeability,
            )

        with timed(
            "extract_filter_reviews",
            raw_review_count=len(reviews),
            force=force,
        ):
            reviews_to_process: list[dict[str, Any]] = []
            skipped = 0
            for review in reviews:
                if not force and _already_processed(conn, review["id"], cfg):
                    skipped += 1
                    log.debug(
                        "Skipping already-processed review %s (cfg=%s)",
                        review["id"],
                        cfg[:8],
                    )
                    continue
                reviews_to_process.append(review)

        total_reviews = len(reviews_to_process)

        log.info(
            "Extraction queue prepared: total=%d, skipped=%d, force=%s, max_workers=%d, extract_mode=%s",
            total_reviews, skipped, force, max_workers, extract_mode,
        )

        #print(f"GSR_PROGRESS extract_claims 0 {total_reviews}", flush=True)
        #print(f"GSR_PROGRESS prepare extraction...", flush=True)

        if total_reviews == 0:
            if skipped > 0:
                log.info("[extract] SKIP already_cached paper_id=%s skipped=%d", paper_id, skipped)
            return {
                "reviews_processed": 0,
                "claims_extracted": 0,
                "errors": 0,
                "results": [],
                "experiment_id": experiment_id,
                "reviews_skipped": skipped,
            }

        ordered_results: list[dict[str, Any] | None] = [None] * total_reviews
        completed = 0

        def _run_one(review: dict[str, Any]) -> dict[str, Any]:
            return extract_review_claims(
                review,
                model=model,
                fields=fields,
                min_confidence=min_confidence,
                min_challengeability=min_challengeability,
                scorer=scorer,
                extract_mode=extract_mode,
            )

        with timed(
            "extract_review_loop_total",
            paper_id=paper_id,
            total_reviews=total_reviews,
            fields=",".join(fields),
            max_workers=max_workers,
        ):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_meta = {}

                for idx, review in enumerate(reviews_to_process):
                    review_id = review["id"]
                    log.info(
                        "Submitting review extraction: review_id=%s (%d/%d)",
                        review_id,
                        idx + 1,
                        total_reviews,
                    )

                    future = executor.submit(_run_one, review)
                    future_to_meta[future] = (idx, review_id)

                    if delay > 0 and idx < total_reviews - 1:
                        time.sleep(delay)

                for future in as_completed(future_to_meta):
                    idx, review_id = future_to_meta[future]

                    with timed(
                        "extract_review_future_collect",
                        review_id=review_id,
                        idx=idx + 1,
                        total=total_reviews,
                    ):
                        try:
                            result = future.result()
                        except Exception as exc:
                            log.exception(
                                "Unhandled extraction exception for review %s: %s",
                                review_id,
                                exc,
                            )
                            review = reviews_to_process[idx]
                            result = {
                                "review_id": review["id"],
                                "paper_id": review["paper_id"],
                                "model_id": model_id,
                                "status": "error",
                                "error": str(exc),
                                "claims": [],
                                "started_at": None,
                                "finished_at": datetime.now(timezone.utc).isoformat(),
                                "min_confidence": min_confidence,
                                "min_challengeability": min_challengeability,
                                "fields": ",".join(fields),
                                "config_hash": cfg,
                            }

                        result["config_hash"] = cfg
                        result["fields"] = ",".join(fields)
                        result["min_confidence"] = min_confidence
                        result["min_challengeability"] = min_challengeability
                        result["experiment_id"] = experiment_id

                        ordered_results[idx] = result

                        if result.get("status") == "success":
                            claim_count = len(result.get("claims", []))
                            total_claims += claim_count
                            log.info(
                                "Review extraction success: review_id=%s claims=%d",
                                review_id,
                                claim_count,
                            )
                        else:
                            errors += 1
                            log.warning(
                                "Review extraction failed: review_id=%s status=%s error=%s",
                                review_id,
                                result.get("status"),
                                result.get("error"),
                            )

                    completed += 1
                    print(f"GSR_PROGRESS extract_claims {completed} {total_reviews}", flush=True)

        results = [r for r in ordered_results if r is not None]

        log.info(
            "[extract] DONE paper_id=%s reviews=%d claims=%d errors=%d skipped=%d",
            paper_id,
            len(results),
            total_claims,
            errors,
            skipped,
        )

        return {
            "reviews_processed": len(results),
            "claims_extracted": total_claims,
            "errors": errors,
            "results": results,
            "experiment_id": experiment_id,
            "reviews_skipped": skipped,
        }