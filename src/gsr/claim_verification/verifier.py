"""Main orchestrator for claim verification – Module 4.

For each claim this module:
1. Loads cached mixed evidence (text chunks + evidence objects).
2. Calls an LLM to classify the claim as supported / refuted /
   insufficient_evidence / not_verifiable.
3. Returns structured verification results ready for storage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

from gsr.claim_extraction.llm import complete, get_model_id
from gsr.claim_verification.alignment import select_best_span_ids_for_supporting_quote
from gsr.claim_verification.prompts import VerificationResponse, build_messages
from gsr.utils.timing import timed

log = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5


def log_timing(stage: str, t0: float, **kwargs: Any) -> None:
    """Structured timing log helper."""
    dt = time.perf_counter() - t0
    if kwargs:
        extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
        log.info("[timing] %s dt=%.3fs %s", stage, dt, extras)
    else:
        log.info("[timing] %s dt=%.3fs", stage, dt)


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def _make_verification_id(claim_id: str, model_id: str) -> str:
    raw = f"{claim_id}:{model_id}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_verifiable_claims(
    conn: sqlite3.Connection,
    *,
    paper_id: str | None = None,
    review_id: str | None = None,
    experiment_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    conditions: list[str] = []
    params: list[Any] = []

    if paper_id:
        conditions.append("c.paper_id = ?")
        params.append(paper_id)

    if review_id:
        conditions.append("c.review_id = ?")
        params.append(review_id)

    if experiment_id:
        conditions.append("er.experiment_id = ?")
        params.append(experiment_id)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    query = f"""
        SELECT
            c.id,
            c.review_id,
            c.paper_id,
            c.source_field,
            c.claim_text,
            c.verbatim_quote,
            c.claim_type,
            c.confidence AS extraction_confidence
        FROM claims c
        JOIN extraction_runs er ON er.id = c.extraction_run_id
        {where}
        ORDER BY c.paper_id, c.id
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    conn.row_factory = sqlite3.Row
    rows = conn.execute(query, params).fetchall()
    conn.row_factory = None
    return [dict(row) for row in rows]


def _already_verified(
    conn: sqlite3.Connection,
    claim_id: str,
    model_id: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM verification_results
        WHERE claim_id = ? AND model_id = ? AND status = 'success'
        LIMIT 1
        """,
        (claim_id, model_id),
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Evidence prep helpers
# ---------------------------------------------------------------------------

def _claim_prefers_table(claim_text: str) -> bool:
    q = (claim_text or "").lower()
    return any(
        kw in q
        for kw in [
            "table",
            "tab.",
            "ablation",
            "baseline",
            "compare",
            "comparison",
            "result",
            "results",
            "performance",
            "score",
            "accuracy",
            "f1",
        ]
    )


def _claim_prefers_figure(claim_text: str) -> bool:
    q = (claim_text or "").lower()
    return any(
        kw in q
        for kw in [
            "figure",
            "fig.",
            "architecture",
            "pipeline",
            "framework",
            "module",
            "overview",
            "diagram",
            "qualitative",
        ]
    )


def _extract_explicit_refs(claim_text: str) -> dict[str, set[str]]:
    import re

    text = claim_text or ""
    refs = {"figure": set(), "table": set()}

    for m in re.finditer(r"\b(?:Figure|Fig\.)\s*(\d+)\b", text, flags=re.I):
        n = m.group(1)
        refs["figure"].add(f"Figure {n}")
        refs["figure"].add(f"Fig. {n}")

    for m in re.finditer(r"\bTable\s*(\d+)\b", text, flags=re.I):
        n = m.group(1)
        refs["table"].add(f"Table {n}")

    return refs


def _label_matches_any(label: str | None, candidates: set[str]) -> bool:
    if not label:
        return False
    norm = label.strip().lower()
    return any(norm == c.strip().lower() for c in candidates)

import re

def _extract_explicit_refs(claim_text: str) -> dict[str, set[str]]:
    text = claim_text or ""
    refs = {"figure": set(), "table": set()}

    for m in re.finditer(r"\b(?:Figure|Fig\.)\s*(\d+)\b", text, flags=re.I):
        n = m.group(1)
        refs["figure"].add(f"Figure {n}")
        refs["figure"].add(f"Fig. {n}")

    for m in re.finditer(r"\bTable\s*(\d+)\b", text, flags=re.I):
        n = m.group(1)
        refs["table"].add(f"Table {n}")

    return refs

def _prepare_evidence_for_prompt(
    claim: dict[str, Any],
    evidence_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Final pre-multimodal prompt preparation:
    - drop empty items
    - normalize fields
    - strongly prioritize explicit Figure/Table object matches
    - then add one nearby text chunk as supplement
    - then sort remaining evidence
    """
    claim_text = claim.get("claim_text", "") or ""
    prefers_table = _claim_prefers_table(claim_text)
    prefers_figure = _claim_prefers_figure(claim_text)
    explicit_refs = _extract_explicit_refs(claim_text)

    cleaned: list[dict[str, Any]] = []
    for ev in evidence_items or []:
        text = (
            ev.get("text")
            or ev.get("content_text")
            or ev.get("caption_text")
            or ""
        ).strip()

        caption_text = (ev.get("caption_text") or ev.get("caption") or "").strip()

        if not text and not caption_text:
            continue

        ev2 = dict(ev)
        ev2["text"] = text or caption_text
        ev2["caption_text"] = caption_text or None
        ev2["object_type"] = ev.get("object_type") or "text_chunk"
        ev2["evidence_type"] = ev2["object_type"]
        ev2["reference_boost"] = float(ev.get("reference_boost") or 0.0)
        ev2["reference_matched"] = bool(
            ev.get("reference_matched") or ev2["reference_boost"] > 0
        )
        cleaned.append(ev2)

    # 1) Find explicit object matches first
    explicit_object_matches: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []

    for ev in cleaned:
        obj_type = ev.get("object_type") or "text_chunk"
        label = ev.get("label") or ev.get("evidence_label")

        is_explicit_obj = (
            (obj_type == "figure" and _label_matches_any(label, explicit_refs["figure"]))
            or
            (obj_type == "table" and _label_matches_any(label, explicit_refs["table"]))
        )

        if is_explicit_obj:
            explicit_object_matches.append(ev)
        else:
            remaining.append(ev)

    # 2) Keep at most one supplemental text chunk for each explicit object match
    supplemental_texts: list[dict[str, Any]] = []
    if explicit_object_matches:
        used_ids = {
            (ev.get("evidence_object_id"), ev.get("chunk_id"))
            for ev in explicit_object_matches
        }

        for ev in remaining:
            if (ev.get("object_type") or "text_chunk") != "text_chunk":
                continue
            # prefer a nearby text chunk on same page
            same_page = False
            for obj in explicit_object_matches:
                if ev.get("page") == obj.get("page") and ev.get("page") is not None:
                    same_page = True
                    break
            if same_page:
                supplemental_texts.append(ev)
                break

    # remove chosen supplement from remaining
    supp_ids = {(ev.get("evidence_object_id"), ev.get("chunk_id")) for ev in supplemental_texts}
    remaining = [
        ev for ev in remaining
        if (ev.get("evidence_object_id"), ev.get("chunk_id")) not in supp_ids
    ]

    # 3) Sort the rest as before
    def _priority(ev: dict[str, Any]) -> tuple[float, float, float]:
        obj_type = ev.get("object_type") or "text_chunk"
        ref = 1.0 if ev.get("reference_matched") else 0.0
        score = float(ev.get("score") or 0.0)

        type_bonus = 0.0
        if prefers_table and obj_type == "table":
            type_bonus += 0.6
        if prefers_figure and obj_type == "figure":
            type_bonus += 0.6
        if obj_type == "text_chunk":
            type_bonus += 0.1

        return (ref, type_bonus, score)

    explicit_object_matches.sort(
        key=lambda ev: float(ev.get("score") or 0.0),
        reverse=True,
    )
    remaining.sort(key=_priority, reverse=True)

    # 4) Final order: explicit object -> supplemental text -> rest
    return explicit_object_matches + supplemental_texts + remaining



# ---------------------------------------------------------------------------
# Phase A — Multimodal figure image attachment helpers
# ---------------------------------------------------------------------------

def _should_attach_figure_image(
    ev: dict[str, Any],
    claim_text: str,
    already_attached: int,
) -> bool:
    """Conservative V1 attach policy — returns True only when all gates pass.

    Gates (all must hold):
    1. object_type is "figure"
    2. asset_path is a non-empty string
    3. crop_quality is not "low" (from metadata_json / evidence dict)
    4. at most 0 images already attached this claim (v1 cap: 1 image per claim)
    5. claim text signals figure relevance OR the evidence was explicitly referenced
    """
    if already_attached >= 1:
        return False
    if ev.get("object_type") != "figure":
        return False
    asset_path = ev.get("asset_path")
    if not asset_path:
        return False

    # Reject low-quality crops (crop_quality comes from metadata stored in the ev dict
    # via load_cached_evidence_mixed; fall back permissively if key is absent).
    crop_quality = (ev.get("crop_quality") or "").lower()
    if crop_quality == "low":
        return False

    # Relevance gate: prefer attaching when the claim text mentions figure language
    # OR the evidence was explicitly reference-matched (strong signal).
    if ev.get("reference_matched"):
        return True
    if _claim_prefers_figure(claim_text):
        return True

    return False


def _load_image_as_base64(path: str) -> str | None:
    """Read a PNG file and return a base64-encoded string, or None on failure."""
    import base64
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception as exc:
        log.warning("[verify-mm] failed to load image %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Single-claim verification
# ---------------------------------------------------------------------------

def verify_claim(
    claim: dict[str, Any],
    evidence_chunks: list[dict],
    model: str | None = None,
) -> dict[str, Any]:
    model_id = get_model_id(model)
    verified_at = datetime.now(timezone.utc).isoformat()

    try:
        prepared_evidence = _prepare_evidence_for_prompt(claim, evidence_chunks)

        log.info(
            "[verify] claim_id=%s evidence_types=%s labels=%s",
            claim.get("id"),
            [e.get("object_type") for e in prepared_evidence],
            [
                e.get("label") or e.get("evidence_label")
                for e in prepared_evidence
                if (e.get("label") or e.get("evidence_label"))
            ],
        )

        if not prepared_evidence:
            structured_evidence = normalize_structured_evidence(evidence_chunks or [])
            return {
                "id": _make_verification_id(claim["id"], model_id),
                "claim_id": claim["id"],
                "paper_id": claim["paper_id"],
                "review_id": claim["review_id"],
                "verdict": "insufficient_evidence",
                "reasoning": "No usable evidence text available after loading cached retrieval results.",
                "confidence": 0.0,
                "supporting_quote": None,
                "evidence": structured_evidence,
                "evidence_chunk_ids": [
                    c["chunk_id"] for c in structured_evidence if c.get("chunk_id")
                ],
                "model_id": model_id,
                "status": "success",
                "error": None,
                "verified_at": verified_at,
            }

        # --- Phase A: attach figure image only if rank-#1 evidence is a valid figure ---
        # Policy: examine only prepared_evidence[0]. Do not scan further items.
        claim_text = claim.get("claim_text", "") or ""
        figure_images: dict[str, str] = {}  # {evidence_object_id: base64_png}
        _fig_attach_stats: dict[str, int] = {
            "attached_top1_figure": 0,
            "skipped_top1_no_evidence": 0,
            "skipped_top1_not_figure": 0,
            "skipped_top1_missing_asset_path": 0,
            "skipped_top1_file_not_found": 0,
            "skipped_top1_low_crop_quality": 0,
        }

        _top1 = prepared_evidence[0] if prepared_evidence else None

        # Pre-compute log fields (safe defaults when top-1 absent or non-figure)
        _top1_ev_id = (_top1.get("evidence_object_id") or _top1.get("id") or "") if _top1 else ""
        _top1_label = (_top1.get("label") or _top1.get("evidence_label") or "") if _top1 else ""
        _top1_asset_path = (_top1.get("asset_path") or "") if _top1 else ""
        _top1_crop_quality = ((_top1.get("crop_quality") or "").lower() or "N/A") if _top1 else "N/A"
        _top1_ref_matched = bool(
            _top1.get("reference_matched") or float(_top1.get("reference_boost") or 0) > 0
        ) if _top1 else False
        _top1_prefers_fig = _claim_prefers_figure(claim_text)
        _top1_file_exists = bool(_top1_asset_path) and os.path.isfile(_top1_asset_path)

        if _top1 is None:
            _fa_label = "top1_no_evidence"
            _fig_attach_stats["skipped_top1_no_evidence"] += 1
        elif _top1.get("object_type") != "figure":
            _fa_label = "top1_not_figure"
            _fig_attach_stats["skipped_top1_not_figure"] += 1
        else:
            if not _top1_asset_path:
                _fa_label = "top1_missing_asset_path"
                _fig_attach_stats["skipped_top1_missing_asset_path"] += 1
            elif not _top1_file_exists:
                _fa_label = "top1_file_not_found"
                _fig_attach_stats["skipped_top1_file_not_found"] += 1
            elif _top1_crop_quality == "low":
                _fa_label = "top1_low_crop_quality"
                _fig_attach_stats["skipped_top1_low_crop_quality"] += 1
            else:
                b64 = _load_image_as_base64(_top1_asset_path)
                if b64:
                    figure_images[_top1_ev_id] = b64
                    _fig_attach_stats["attached_top1_figure"] += 1
                    _fa_label = "top1_figure"
                else:
                    _fig_attach_stats["skipped_top1_file_not_found"] += 1
                    _fa_label = "top1_file_not_found"

        log.info(
            "[figure-attach] claim_id=%s ev_id=%s label=%r "
            "asset_path_present=%s file_exists=%s crop_quality=%s "
            "reference_matched=%s claim_prefers_figure=%s "
            "decision=%s skip_reason=%s",
            claim.get("id"), _top1_ev_id, _top1_label,
            bool(_top1_asset_path), _top1_file_exists, _top1_crop_quality,
            _top1_ref_matched, _top1_prefers_fig,
            "attached" if _fa_label == "top1_figure" else "skipped",
            _fa_label,
        )

        attached_figure_ids = list(figure_images.keys())
        used_multimodal = bool(figure_images)

        log.info(
            "[verify] claim_id=%s figure_images_attached=%d",
            claim.get("id"), len(figure_images),
        )

        # Build messages — multimodal when figure images are available.
        messages = build_messages(claim, prepared_evidence, figure_images=figure_images or None)

        # Call LLM — fall back to text-only if multimodal call fails.
        try:
            response: VerificationResponse = complete(
                messages=messages,
                response_format=VerificationResponse,
                model=model,
                temperature=0.0,
            )
        except Exception as mm_exc:
            if used_multimodal:
                log.warning(
                    "[verify] multimodal call failed for claim_id=%s, retrying text-only: %s",
                    claim.get("id"), mm_exc,
                )
                text_only_messages = build_messages(claim, prepared_evidence)
                response = complete(
                    messages=text_only_messages,
                    response_format=VerificationResponse,
                    model=model,
                    temperature=0.0,
                )
                used_multimodal = False
                attached_figure_ids = []
            else:
                raise

        structured_evidence = normalize_structured_evidence(prepared_evidence)

        return {
            "id": _make_verification_id(claim["id"], model_id),
            "claim_id": claim["id"],
            "paper_id": claim["paper_id"],
            "review_id": claim["review_id"],
            "verdict": response.verdict,
            "reasoning": response.reasoning,
            "confidence": response.confidence,
            "supporting_quote": response.supporting_quote,
            "evidence": structured_evidence,
            "evidence_chunk_ids": [
                c["chunk_id"] for c in structured_evidence if c.get("chunk_id")
            ],
            "model_id": model_id,
            "status": "success",
            "error": None,
            "verified_at": verified_at,
            # Phase A: multimodal audit fields
            "used_multimodal": used_multimodal,
            "attached_figure_ids": attached_figure_ids,
            "attached_figure_count": len(attached_figure_ids),
            "figure_attach_stats": _fig_attach_stats,
        }

    except Exception as exc:
        log.error("Verification failed for claim '%s': %s", claim["id"], exc)
        return {
            "id": _make_verification_id(claim["id"], model_id),
            "claim_id": claim["id"],
            "paper_id": claim["paper_id"],
            "review_id": claim["review_id"],
            "verdict": None,
            "reasoning": None,
            "confidence": None,
            "supporting_quote": None,
            "evidence": [],
            "evidence_chunk_ids": [],
            "model_id": model_id,
            "status": "error",
            "error": str(exc),
            "verified_at": verified_at,
        }


# ---------------------------------------------------------------------------
# Batch verification
# ---------------------------------------------------------------------------

def _verify_one_claim_llm_only(
    *,
    claim: dict[str, Any],
    evidence: list[dict[str, Any]],
    model: str | None,
    evidence_source: str,
) -> dict[str, Any]:
    claim_id = claim["id"]

    t0 = time.perf_counter()
    result = verify_claim(claim, evidence, model=model)
    log_timing(
        "claim_llm_verify",
        t0,
        claim_id=claim_id,
        evidence_count=len(evidence),
        source=evidence_source,
        model=model or "default",
    )
    return result


def _maybe_enrich_figure_semantics_for_claim(
    claim: dict[str, Any],
    evidence: list[dict[str, Any]],
    conn: sqlite3.Connection,
    pdf_path: str,
    config_hash: str,
    *,
    enable_top: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pre-enrich figure evidence with Qwen2.5-VL semantics (Phase P2).

    Trigger policy:
    - Always: if claim has an explicit figure reference (Figure N / Fig. N),
      enrich the matching figure evidence object(s).
    - With enable_top=True: also enrich the top-ranked figure even without
      an explicit reference.

    Returns (enriched_evidence_list, stats_dict).
    stats_dict keys: triggered, trigger, skip_reason, enriched,
                     cache_hit, fresh_success, failed, explicit_ref, top_fallback.
    """
    from gsr.claim_verification.figure_escalation import (
        check_escalation_trigger,
        pick_top_figure_evidence,
    )
    from gsr.paper_retrieval.vision.figure_semantics_qwen import (
        ensure_figure_semantics_for_evidence_object,
    )

    claim_id = claim.get("id")
    stats: dict[str, Any] = {
        "triggered": False,
        "trigger": None,
        "skip_reason": None,
        "enriched": 0,
        "cache_hit": 0,
        "fresh_success": 0,
        "failed": 0,
        "explicit_ref": 0,
        "top_fallback": 0,
    }

    # Check for explicit figure reference (always; no verdict needed)
    trigger_result = check_escalation_trigger(
        claim,
        evidence,
        first_pass_verdict=None,
        first_pass_confidence=None,
        enable_heuristic=False,
    )

    figures_to_enrich: list[dict[str, Any]] = []

    if trigger_result["should_escalate"]:
        # Explicit figure reference detected — find matching figure(s) by label
        explicit_refs = _extract_explicit_refs(claim.get("claim_text", "") or "")
        for ev in evidence:
            if ev.get("object_type") == "figure":
                label = ev.get("label") or ""
                if _label_matches_any(label, explicit_refs["figure"]):
                    figures_to_enrich.append(ev)
        # Fallback: no label match but explicit ref present → use top figure
        if not figures_to_enrich:
            top_fig = pick_top_figure_evidence(evidence)
            if top_fig:
                figures_to_enrich = [top_fig]
        stats["triggered"] = True
        stats["trigger"] = "explicit_figure_ref"
        stats["explicit_ref"] = len(figures_to_enrich)
        log.info(
            "[figure-sem] claim=%s trigger=explicit_figure_ref figures=%s",
            claim_id, [f.get("label") for f in figures_to_enrich],
        )
    elif enable_top and trigger_result.get("figure_present"):
        # Top-figure fallback mode: enrich top figure even without explicit ref
        top_fig = pick_top_figure_evidence(evidence)
        if top_fig:
            figures_to_enrich = [top_fig]
            stats["triggered"] = True
            stats["trigger"] = "top_figure_fallback"
            stats["top_fallback"] = 1
            log.info(
                "[figure-sem] claim=%s trigger=top_figure_fallback figure=%s",
                claim_id, top_fig.get("label"),
            )
        else:
            stats["skip_reason"] = "no_figure"
    else:
        stats["skip_reason"] = trigger_result.get("skip_reason") or "trigger_not_met"

    if not figures_to_enrich:
        return evidence, stats

    enriched_evidence = list(evidence)
    seen_ids: set[str] = set()

    for fig_ev in figures_to_enrich:
        ev_id = fig_ev.get("evidence_object_id") or ""
        if not ev_id or ev_id in seen_ids:
            continue
        seen_ids.add(ev_id)

        updated_ev, cache_status = ensure_figure_semantics_for_evidence_object(
            conn, fig_ev, pdf_path, config_hash=config_hash,
        )

        # Replace the original evidence item in the list
        for i, ev in enumerate(enriched_evidence):
            if ev.get("evidence_object_id") == ev_id:
                enriched_evidence[i] = updated_ev
                break

        stats["enriched"] += 1
        if cache_status == "cache_hit_positive":
            stats["cache_hit"] += 1
        elif cache_status == "fresh_success":
            stats["fresh_success"] += 1
        else:
            stats["failed"] += 1

        log.info(
            "[figure-sem] figure=%s page=%s cache_status=%s",
            fig_ev.get("label"), fig_ev.get("page"), cache_status,
        )

    return enriched_evidence, stats


def verify_all_claims(
    conn: sqlite3.Connection,
    *,
    paper_id: str | None = None,
    review_id: str | None = None,
    experiment_id: str | None = None,
    limit: int | None = None,
    model: str | None = None,
    top_k: int = _DEFAULT_TOP_K,
    embedding_model_id: str | None = None,
    force: bool = False,
    delay: float = 0.0,
    allow_live_retrieval: bool = False,
    require_cached_evidence: bool = True,
    max_workers: int = 2,
    selective_figure_ocr: bool = False,
    enable_heuristic_figure_escalation: bool = False,
    selective_figure_semantic: bool = False,
    enable_figure_semantic_top: bool = False,
) -> dict[str, Any]:
    model_id = get_model_id(model)

    raw_claims = _load_verifiable_claims(
        conn,
        paper_id=paper_id,
        experiment_id=experiment_id,
        review_id=review_id,
        limit=limit,
    )

    log.debug("verify loaded claims: paper_id=%s count=%d", paper_id, len(raw_claims))

    claims: list[dict[str, Any]] = []
    skipped_verified = 0
    for claim in raw_claims:
        if not force and _already_verified(conn, claim["id"], model_id):
            skipped_verified += 1
            log.debug("Skipping already-verified claim '%s'.", claim["id"])
            continue
        claims.append(claim)

    total = len(claims)

    if total == 0:
        log.info("[verify] SKIP no_claims paper_id=%s skipped_already_verified=%d", paper_id, skipped_verified)
    else:
        log.info("[verify] START paper_id=%s claims=%d skipped_already_verified=%d", paper_id, total, skipped_verified)
    log.debug(
        "verify queue: total=%d skipped=%d force=%s allow_live=%s require_cached=%s workers=%d",
        total,
        skipped_verified,
        force,
        allow_live_retrieval,
        require_cached_evidence,
        max_workers,
    )

    print("GSR_PROGRESS verify_prepare 0 1", flush=True)

    _emb_model = None
    _emb_model_id = embedding_model_id or "allenai/specter2_base"

    results: list[dict[str, Any]] = []
    verdicts: dict[str, int] = {}
    errors = 0
    cache_hits = 0
    live_retrievals = 0
    missing_evidence = 0
    missing_evidence_claim_ids: list[str] = []
    claims_skipped_missing_evidence = 0
    completed = 0

    # Figure semantics counters (Phase P2)
    _figsem_attempted = 0
    _figsem_cache_hit = 0
    _figsem_fresh_success = 0
    _figsem_fresh_failed = 0
    _figsem_explicit_ref = 0
    _figsem_top_fallback = 0

    # Selective figure OCR counters (Phase 2.2 expanded)
    _sel_escalated = 0
    _sel_cache_hit_pos = 0
    _sel_cache_hit_neg = 0
    _sel_fresh_ocr = 0
    _sel_ocr_unavailable = 0
    _sel_skipped_rank_policy = 0
    _sel_skipped_lexical = 0
    _sel_verdict_changed = 0
    _sel_confidence_improved_only = 0
    _sel_no_change = 0
    _sel_escalated_figure_ranks: list[int] = []
    _sel_explicit_ref_escalations = 0
    _sel_heuristic_escalations = 0
    _sel_explicit_ref_verdict_changed = 0
    _sel_heuristic_verdict_changed = 0

    if total == 0:
        return {
            "claims_processed": 0,
            "verdicts": {},
            "errors": 0,
            "results": [],
            "cache_hits": 0,
            "live_retrievals": 0,
            "missing_evidence": 0,
            "missing_evidence_claim_ids": [],
            "claims_skipped_missing_evidence": 0,
        }

    # --- Selective figure semantic setup (Phase P2) ---
    _figsem_config_hash: str = ""
    _figsem_enabled = selective_figure_semantic
    if _figsem_enabled:
        try:
            from gsr.paper_retrieval.vision.figure_semantics_qwen import (  # type: ignore[import]
                make_figsem_config_hash,
                _DEFAULT_MAX_NEW_TOKENS as _FIGSEM_MAX_TOKENS,
                _RENDER_SCALE as _FIGSEM_SCALE,
                _get_model_id as _figsem_get_model_id,
            )
            _figsem_config_hash = make_figsem_config_hash(
                model_id=_figsem_get_model_id(),
                render_scale=_FIGSEM_SCALE,
                max_new_tokens=_FIGSEM_MAX_TOKENS,
            )
        except ImportError:
            _figsem_config_hash = "qwen25vl_figsem_v1_default"
        log.info(
            "[figure-sem] selective mode enabled config_hash=%s top_enabled=%s",
            _figsem_config_hash, enable_figure_semantic_top,
        )

    # --- Selective figure OCR setup ---
    _ocr_config_hash: str = ""
    _pdf_path_cache: dict[str, str | None] = {}
    if selective_figure_ocr:
        try:
            from gsr.paper_retrieval.vision.ocr_lighton import (  # type: ignore[import]
                _DEFAULT_MAX_NEW_TOKENS,
                _DEFAULT_TIMEOUT_SECONDS,
                _MAX_CROP_LONGEST_SIDE,
                _RENDER_SCALE,
            )
            from gsr.claim_verification.figure_escalation import make_ocr_config_hash
            _ocr_config_hash = make_ocr_config_hash(
                max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                max_crop_longest_side=_MAX_CROP_LONGEST_SIDE,
                render_scale=_RENDER_SCALE,
            )
        except ImportError:
            _ocr_config_hash = "lightonocr2_default"
        log.info(
            "[verify-figure-ocr] selective mode enabled config_hash=%s heuristic_enabled=%s",
            _ocr_config_hash, enable_heuristic_figure_escalation,
        )

    loop_t0 = time.perf_counter()

    # Figure attachment diagnostic counters aggregated across all claims
    _fa_counters: dict[str, int] = {
        "attached_top1_figure": 0,
        "skipped_top1_no_evidence": 0,
        "skipped_top1_not_figure": 0,
        "skipped_top1_missing_asset_path": 0,
        "skipped_top1_file_not_found": 0,
        "skipped_top1_low_crop_quality": 0,
    }

    def _accumulate_fa_stats(result: dict[str, Any]) -> None:
        fa = result.get("figure_attach_stats") or {}
        for k in _fa_counters:
            _fa_counters[k] += fa.get(k, 0)

    def _finalize_future(
        future,
        meta: tuple[int, dict[str, Any], str, float],
    ) -> None:
        nonlocal completed, errors, results, verdicts

        i, claim, evidence_source, claim_t0 = meta

        try:
            result = future.result()
        except Exception as exc:
            log.exception(
                "Verification worker failed for claim '%s' (%d/%d): %s",
                claim["id"], i, total, exc,
            )
            result = {
                "status": "error",
                "claim_id": claim["id"],
                "claim_text": claim.get("claim_text", ""),
                "verdict": None,
                "reasoning": f"Worker exception: {exc}",
                "evidence": [],
                "supporting_quote": None,
            }

        t0 = time.perf_counter()
        if result["status"] == "success" and result.get("supporting_quote"):
            for ev in result.get("evidence", []):
                ev["aligned_span_ids"] = select_best_span_ids_for_supporting_quote(
                    conn=conn,
                    supporting_quote=result["supporting_quote"],
                    candidate_span_ids=ev.get("span_ids", []),
                    top_n=3,
                    min_score=0.25,
                    expand_neighbors=True,
                )
        else:
            for ev in result.get("evidence", []):
                ev["aligned_span_ids"] = ev.get("span_ids", [])

        log_timing(
            "claim_align_supporting_quote",
            t0,
            claim_id=claim["id"],
            status=result["status"],
            evidence_count=len(result.get("evidence", [])),
        )

        results.append(result)
        _accumulate_fa_stats(result)

        if result["status"] == "success":
            v = result["verdict"]
            verdicts[v] = verdicts.get(v, 0) + 1
        else:
            errors += 1

        if result["status"] == "success":
            log.info(
                "Verified claim (%d/%d): claim_id=%s verdict=%s source=%s",
                i,
                total,
                claim["id"],
                result["verdict"],
                evidence_source,
            )
        else:
            log.info(
                "Verification failed (%d/%d): claim_id=%s source=%s",
                i,
                total,
                claim["id"],
                evidence_source,
            )

        completed += 1
        print(f"GSR_PROGRESS verify_claims {completed} {total}", flush=True)

        log_timing(
            "claim_verify_total",
            claim_t0,
            claim_id=claim["id"],
            idx=i,
            total=total,
            source=evidence_source,
        )

    # --- Late Docling table enrichment (runs once per batch, before verification) ---
    # Inspects top-k retrieval results for the claims being verified, finds table
    # evidence objects that were indexed with Docling (docling_enriched=False) but
    # not yet enriched, and populates their content_text with Docling markdown.
    # Skips gracefully when Docling is unavailable or no table candidates exist.
    print("GSR_PROGRESS verify_enrich 0 1", flush=True)
    if claims:
        _docling_claim_ids = [c["id"] for c in claims]
        try:
            from gsr.paper_retrieval.retrieval import enrich_topk_tables_with_docling
            _docling_summary = enrich_topk_tables_with_docling(
                conn, _docling_claim_ids, top_k
            )
            log.info(
                "[docling-late] pre-verify enrichment: enriched=%d failed=%d skipped_already=%d",
                _docling_summary.get("enriched", 0),
                _docling_summary.get("failed", 0),
                _docling_summary.get("skipped_already", 0),
            )
        except Exception as _docling_exc:
            log.warning(
                "[docling-late] pre-verify enrichment failed (non-fatal): %s", _docling_exc
            )

    print("GSR_PROGRESS verify_model 0 1", flush=True)
    print(f"GSR_PROGRESS verify_claims 0 {total}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_to_meta: dict[Any, tuple[int, dict[str, Any], str, float]] = {}

        for i, claim in enumerate(claims, start=1):
            claim_t0 = time.perf_counter()

            t0 = time.perf_counter()
            evidence = load_cached_evidence_mixed(
                conn,
                claim["id"],
                top_k,
                model_id=_emb_model_id,
            )
            log_timing(
                "claim_load_cached_evidence_mix",
                t0,
                claim_id=claim["id"],
                top_k=top_k,
            )

            evidence_source = "cache"

            if evidence:
                cache_hits += 1
            else:
                missing_evidence += 1
                missing_evidence_claim_ids.append(claim["id"])

                log.warning(
                    "Missing cached evidence for claim (%d/%d): claim_id=%s paper_id=%s",
                    i,
                    total,
                    claim["id"],
                    claim.get("paper_id"),
                )

                if allow_live_retrieval:
                    evidence_source = "live"
                    live_retrievals += 1

                    t0 = time.perf_counter()
                    evidence, _emb_model, _emb_model_id = _retrieve_live(
                        conn,
                        claim,
                        top_k=top_k,
                        embedding_model=_emb_model,
                        embedding_model_id=_emb_model_id,
                    )
                    log_timing(
                        "claim_retrieve_live",
                        t0,
                        claim_id=claim["id"],
                        top_k=top_k,
                        embedding_model_id=_emb_model_id,
                    )

                    if not evidence and require_cached_evidence:
                        claims_skipped_missing_evidence += 1
                        log.info(
                            "Skipping verification (%d/%d): claim_id=%s reason=missing_evidence_after_live source=live",
                            i,
                            total,
                            claim["id"],
                        )
                        completed += 1
                        print(f"GSR_PROGRESS verify_claims {completed} {total}", flush=True)
                        log_timing(
                            "claim_verify_total",
                            claim_t0,
                            claim_id=claim["id"],
                            idx=i,
                            total=total,
                            source="missing_evidence_after_live",
                        )
                        continue
                else:
                    claims_skipped_missing_evidence += 1
                    log.info(
                        "Skipping verification (%d/%d): claim_id=%s reason=missing_evidence source=cache",
                        i,
                        total,
                        claim["id"],
                    )
                    completed += 1
                    print(f"GSR_PROGRESS verify_claims {completed} {total}", flush=True)
                    log_timing(
                        "claim_verify_total",
                        claim_t0,
                        claim_id=claim["id"],
                        idx=i,
                        total=total,
                        source="missing_evidence",
                    )
                    continue

            log.info(
                "Prepared claim '%s' (%d/%d) with %d evidence chunks. source=%s",
                claim["id"], i, total, len(evidence), evidence_source,
            )

            # --- Shared PDF path lookup (for semantics and/or OCR) ---
            _pdf_path: str | None = None
            if _figsem_enabled or selective_figure_ocr:
                _pid = claim.get("paper_id", "")
                if _pid not in _pdf_path_cache:
                    _pdf_path_cache[_pid] = _lookup_pdf_path(conn, _pid)
                _pdf_path = _pdf_path_cache[_pid]

            # --- Selective figure semantic enrichment (Phase P2, sequential) ---
            if _figsem_enabled and _pdf_path:
                evidence, _sem_stats = _maybe_enrich_figure_semantics_for_claim(
                    claim, evidence, conn, _pdf_path, _figsem_config_hash,
                    enable_top=enable_figure_semantic_top,
                )
                if _sem_stats.get("triggered"):
                    _figsem_attempted += 1
                    _figsem_cache_hit += _sem_stats.get("cache_hit", 0)
                    _figsem_fresh_success += _sem_stats.get("fresh_success", 0)
                    _figsem_fresh_failed += _sem_stats.get("failed", 0)
                    _figsem_explicit_ref += min(_sem_stats.get("explicit_ref", 0), 1)
                    _figsem_top_fallback += min(_sem_stats.get("top_fallback", 0), 1)

            # --- Selective figure OCR path (sequential, in main thread) ---
            if selective_figure_ocr:
                result = _verify_claim_selective_ocr(
                    claim=claim,
                    evidence=evidence,
                    conn=conn,
                    pdf_path=_pdf_path,
                    model=model,
                    ocr_config_hash=_ocr_config_hash,
                    enable_heuristic_figure_escalation=enable_heuristic_figure_escalation,
                )

                # Track selective stats (Phase 2.2 expanded)
                vmeta = result.get("verify_meta") or {}
                if vmeta.get("escalation_attempted"):
                    _sel_escalated += 1
                    cs = vmeta.get("escalated_ocr_cache_status", "")
                    if cs == "cache_hit_positive":
                        _sel_cache_hit_pos += 1
                    elif cs == "cache_hit_negative":
                        _sel_cache_hit_neg += 1
                    elif cs == "fresh_ocr":
                        _sel_fresh_ocr += 1
                    outcome = vmeta.get("escalation_outcome", "")
                    if outcome == "ocr_unavailable":
                        _sel_ocr_unavailable += 1
                    elif outcome == "verdict_changed":
                        _sel_verdict_changed += 1
                    elif outcome == "confidence_improved_only":
                        _sel_confidence_improved_only += 1
                    elif outcome == "no_change":
                        _sel_no_change += 1
                    rank = vmeta.get("escalated_figure_rank")
                    if rank is not None:
                        _sel_escalated_figure_ranks.append(rank)
                    # explicit ref vs heuristic split
                    trigger_val = vmeta.get("trigger") or ""
                    if trigger_val == "explicit_figure_ref":
                        _sel_explicit_ref_escalations += 1
                        if outcome == "verdict_changed":
                            _sel_explicit_ref_verdict_changed += 1
                    elif trigger_val:
                        _sel_heuristic_escalations += 1
                        if outcome == "verdict_changed":
                            _sel_heuristic_verdict_changed += 1
                elif vmeta.get("rejected_by_rank_policy"):
                    _sel_skipped_rank_policy += 1
                elif vmeta.get("rejected_by_lexical_policy"):
                    _sel_skipped_lexical += 1

                # Span alignment (same as _finalize_future)
                t0 = time.perf_counter()
                if result["status"] == "success" and result.get("supporting_quote"):
                    for ev in result.get("evidence", []):
                        ev["aligned_span_ids"] = select_best_span_ids_for_supporting_quote(
                            conn=conn,
                            supporting_quote=result["supporting_quote"],
                            candidate_span_ids=ev.get("span_ids", []),
                            top_n=3,
                            min_score=0.25,
                            expand_neighbors=True,
                        )
                else:
                    for ev in result.get("evidence", []):
                        ev["aligned_span_ids"] = ev.get("span_ids", [])
                log_timing(
                    "claim_align_supporting_quote",
                    t0,
                    claim_id=claim["id"],
                    status=result["status"],
                    evidence_count=len(result.get("evidence", [])),
                )

                results.append(result)
                _accumulate_fa_stats(result)
                if result["status"] == "success":
                    v = result["verdict"]
                    verdicts[v] = verdicts.get(v, 0) + 1
                    log.info(
                        "Verified claim (%d/%d): claim_id=%s verdict=%s source=%s mode=%s",
                        i, total, claim["id"], result["verdict"],
                        evidence_source,
                        vmeta.get("verify_mode", "baseline_only"),
                    )
                else:
                    errors += 1
                    log.info(
                        "Verification failed (%d/%d): claim_id=%s source=%s",
                        i, total, claim["id"], evidence_source,
                    )
                completed += 1
                print(f"GSR_PROGRESS verify_claims {completed} {total}", flush=True)
                log_timing(
                    "claim_verify_total",
                    claim_t0,
                    claim_id=claim["id"],
                    idx=i,
                    total=total,
                    source=evidence_source,
                )
                if delay > 0 and i < total:
                    time.sleep(delay)
                continue
            # --- End selective path ---

            future = executor.submit(
                _verify_one_claim_llm_only,
                claim=claim,
                evidence=evidence,
                model=model,
                evidence_source=evidence_source,
            )
            future_to_meta[future] = (i, claim, evidence_source, claim_t0)

            if len(future_to_meta) >= max(1, max_workers):
                done_now = [f for f in list(future_to_meta.keys()) if f.done()]
                if not done_now:
                    done_now, _ = wait(
                        future_to_meta.keys(),
                        return_when=FIRST_COMPLETED,
                    )

                for done_f in done_now:
                    meta = future_to_meta.pop(done_f)
                    _finalize_future(done_f, meta)

            if delay > 0 and i < total:
                t0 = time.perf_counter()
                time.sleep(delay)
                log_timing(
                    "claim_delay",
                    t0,
                    claim_id=claim["id"],
                    delay=delay,
                )

        for future in as_completed(list(future_to_meta.keys())):
            meta = future_to_meta[future]
            _finalize_future(future, meta)

    log_timing(
        "verify_claim_loop_total",
        loop_t0,
        paper_id=paper_id or "ALL",
        total_claims=total,
        top_k=top_k,
        model=model or "default",
        allow_live_retrieval=allow_live_retrieval,
        require_cached_evidence=require_cached_evidence,
        max_workers=max_workers,
        selective_figure_ocr=selective_figure_ocr,
    )

    log.info(
        "[verify] DONE paper_id=%s claims=%d errors=%d missing_evidence=%d",
        paper_id,
        len(results),
        errors,
        missing_evidence,
    )
    log.debug(
        "verify finished: processed=%d errors=%d cache_hits=%d live=%d missing=%d skipped_missing=%d",
        len(results),
        errors,
        cache_hits,
        live_retrievals,
        missing_evidence,
        claims_skipped_missing_evidence,
    )

    log.info(
        "[figure-attach] summary total_claims=%d attached_top1_figure=%d "
        "skipped_top1_no_evidence=%d skipped_top1_not_figure=%d "
        "skipped_top1_missing_asset_path=%d skipped_top1_file_not_found=%d "
        "skipped_top1_low_crop_quality=%d",
        total,
        _fa_counters["attached_top1_figure"],
        _fa_counters["skipped_top1_no_evidence"],
        _fa_counters["skipped_top1_not_figure"],
        _fa_counters["skipped_top1_missing_asset_path"],
        _fa_counters["skipped_top1_file_not_found"],
        _fa_counters["skipped_top1_low_crop_quality"],
    )

    if selective_figure_ocr:
        _avg_rank = (
            sum(_sel_escalated_figure_ranks) / len(_sel_escalated_figure_ranks)
            if _sel_escalated_figure_ranks else None
        )
        log.info(
            "[verify-figure-ocr] summary escalated=%d cache_hit_pos=%d cache_hit_neg=%d "
            "fresh=%d ocr_unavailable=%d skipped_rank=%d skipped_lexical=%d "
            "verdict_changed=%d confidence_only=%d no_change=%d "
            "explicit_ref=%d heuristic=%d avg_rank=%s",
            _sel_escalated, _sel_cache_hit_pos, _sel_cache_hit_neg,
            _sel_fresh_ocr, _sel_ocr_unavailable, _sel_skipped_rank_policy, _sel_skipped_lexical,
            _sel_verdict_changed, _sel_confidence_improved_only, _sel_no_change,
            _sel_explicit_ref_escalations, _sel_heuristic_escalations,
            f"{_avg_rank:.1f}" if _avg_rank is not None else "N/A",
        )

    return {
        "claims_processed": len(results),
        "verdicts": verdicts,
        "errors": errors,
        "results": results,
        "cache_hits": cache_hits,
        "live_retrievals": live_retrievals,
        "missing_evidence": missing_evidence,
        "missing_evidence_claim_ids": missing_evidence_claim_ids,
        "claims_skipped_missing_evidence": claims_skipped_missing_evidence,
        "selective_figure_ocr_stats": {
            "enabled": True,
            "escalated": _sel_escalated,
            "cache_hit_positive": _sel_cache_hit_pos,
            "cache_hit_negative": _sel_cache_hit_neg,
            "fresh_ocr": _sel_fresh_ocr,
            "ocr_unavailable": _sel_ocr_unavailable,
            "skipped_rank_policy": _sel_skipped_rank_policy,
            "skipped_lexical": _sel_skipped_lexical,
            "verdict_changed": _sel_verdict_changed,
            "confidence_improved_only": _sel_confidence_improved_only,
            "no_change": _sel_no_change,
            "explicit_ref_escalations": _sel_explicit_ref_escalations,
            "heuristic_escalations": _sel_heuristic_escalations,
            "explicit_ref_verdict_changed": _sel_explicit_ref_verdict_changed,
            "heuristic_verdict_changed": _sel_heuristic_verdict_changed,
            "avg_escalated_figure_rank": (
                sum(_sel_escalated_figure_ranks) / len(_sel_escalated_figure_ranks)
                if _sel_escalated_figure_ranks else None
            ),
        } if selective_figure_ocr else {"enabled": False},
        "figure_semantics_stats": {
            "enabled": True,
            "attempted": _figsem_attempted,
            "cache_hit": _figsem_cache_hit,
            "fresh_success": _figsem_fresh_success,
            "fresh_failed": _figsem_fresh_failed,
            "explicit_ref": _figsem_explicit_ref,
            "top_fallback": _figsem_top_fallback,
        } if selective_figure_semantic else {"enabled": False},
    }


# ---------------------------------------------------------------------------
# Internal: live evidence retrieval
# ---------------------------------------------------------------------------

def _lookup_pdf_path(conn: sqlite3.Connection, paper_id: str) -> str | None:
    """Return the local PDF path for a paper, or None if not available."""
    row = conn.execute(
        "SELECT pdf_path FROM papers WHERE id = ? LIMIT 1", (paper_id,)
    ).fetchone()
    return row[0] if row and row[0] else None


def _verify_claim_selective_ocr(
    *,
    claim: dict[str, Any],
    evidence: list[dict[str, Any]],
    conn: sqlite3.Connection,
    pdf_path: str | None,
    model: str | None,
    ocr_config_hash: str,
    enable_heuristic_figure_escalation: bool = False,
) -> dict[str, Any]:
    """Two-pass verification with selective figure OCR escalation (Phase 2.1).

    Pass 1: standard verify_claim using caption-first evidence.
    Escalation check: rank-gated rule-based trigger (Phase 2.1 strict policy).
    Pass 2 (if escalated and OCR available): re-run verify_claim with OCR-enriched evidence.

    Returns the final result dict enriched with a ``verify_meta`` key carrying
    full audit metadata for effectiveness analysis.
    """
    from gsr.claim_verification.figure_escalation import (
        check_escalation_trigger,
        ensure_figure_ocr_for_evidence_object,
        pick_top_figure_evidence,
    )

    claim_id = claim["id"]

    # --- First pass ---
    t0 = time.perf_counter()
    first_pass = verify_claim(claim, evidence, model=model)
    log_timing(
        "claim_llm_verify_first_pass",
        t0,
        claim_id=claim_id,
        evidence_count=len(evidence),
    )

    first_verdict = first_pass.get("verdict")
    first_conf = first_pass.get("confidence")

    log.info(
        "[verify-figure-ocr] claim=%s first_pass verdict=%s conf=%s",
        claim_id, first_verdict, first_conf,
    )

    # --- Escalation check (returns structured dict) ---
    trigger_result = check_escalation_trigger(
        claim,
        evidence,
        first_pass_verdict=first_verdict,
        enable_heuristic=enable_heuristic_figure_escalation,
        first_pass_confidence=first_conf,
    )

    figure_present = trigger_result["figure_present"]
    top_figure_rank = trigger_result["top_figure_rank"]
    top_figure_label = trigger_result["top_figure_label"]
    explicit_figure_ref_detected = trigger_result["explicit_figure_ref_detected"]
    rejected_by_rank_policy = trigger_result["rejected_by_rank_policy"]
    skip_reason = trigger_result["skip_reason"]

    # Common lexical audit fields for all verify_meta dicts in this function
    _lex = {
        "rejected_by_lexical_policy": trigger_result.get("rejected_by_lexical_policy", False),
        "lexical_gate_applied": trigger_result.get("lexical_gate_applied", False),
        "lexical_gate_passed": trigger_result.get("lexical_gate_passed", False),
        "lexical_strong_cues": trigger_result.get("lexical_strong_cues", []),
        "lexical_weak_cues": trigger_result.get("lexical_weak_cues", []),
    }

    # No escalation — return first pass with non-escalation audit meta
    if not trigger_result["should_escalate"]:
        first_pass["verify_meta"] = {
            "verify_mode": "baseline_only",
            "selective_figure_ocr_enabled": True,
            "escalation_attempted": False,
            "figure_present": figure_present,
            "explicit_figure_ref_detected": explicit_figure_ref_detected,
            "top_figure_rank": top_figure_rank,
            "top_figure_label": top_figure_label,
            "rejected_by_rank_policy": rejected_by_rank_policy,
            "skipped_by_policy": True,
            "skip_reason": skip_reason,
            "trigger": trigger_result.get("trigger"),
            "first_pass_verdict": first_verdict,
            "first_pass_confidence": first_conf,
            **_lex,
        }
        return first_pass

    # Escalation triggered but pdf_path unavailable → ocr_unavailable
    if not pdf_path:
        log.warning(
            "[verify-figure-ocr] claim=%s trigger=%s but pdf_path unavailable",
            claim_id, trigger_result["trigger"],
        )
        first_pass["verify_meta"] = {
            "verify_mode": "escalated_figure_ocr",
            "selective_figure_ocr_enabled": True,
            "escalation_attempted": True,
            "escalation_trigger": trigger_result["trigger"],
            "trigger": trigger_result["trigger"],
            "figure_present": True,
            "explicit_figure_ref_detected": explicit_figure_ref_detected,
            "escalated_figure_label": top_figure_label,
            "escalated_figure_rank": top_figure_rank,
            "escalated_ocr_cache_status": "error",
            "first_pass_verdict": first_verdict,
            "first_pass_confidence": first_conf,
            "second_pass_verdict": None,
            "second_pass_confidence": None,
            "verdict_changed": False,
            "confidence_delta": None,
            "confidence_improved": False,
            "second_pass_used_ocr_text": False,
            "escalation_outcome": "ocr_unavailable",
            **_lex,
        }
        return first_pass

    top_fig = pick_top_figure_evidence(evidence)
    if not top_fig:
        # Defensive: should not happen if figure_present == True
        first_pass["verify_meta"] = {
            "verify_mode": "baseline_only",
            "selective_figure_ocr_enabled": True,
            "escalation_attempted": False,
            "figure_present": False,
            "explicit_figure_ref_detected": explicit_figure_ref_detected,
            "top_figure_rank": None,
            "top_figure_label": None,
            "rejected_by_rank_policy": False,
            "skipped_by_policy": True,
            "skip_reason": "no_figure",
            "trigger": None,
            "first_pass_verdict": first_verdict,
            "first_pass_confidence": first_conf,
            **_lex,
        }
        return first_pass

    fig_label = top_fig.get("label") or top_figure_label or "?"
    log.info(
        "[verify-figure-ocr] claim=%s trigger=%s top_figure=%s rank=%s",
        claim_id, trigger_result["trigger"], fig_label, top_figure_rank,
    )

    # --- Ensure OCR for top figure ---
    enriched_fig, cache_status = ensure_figure_ocr_for_evidence_object(
        conn, top_fig, pdf_path, config_hash=ocr_config_hash
    )

    log.info(
        "[verify-figure-ocr] claim=%s cache_status=%s figure=%s",
        claim_id, cache_status, fig_label,
    )

    # Determine if OCR text is actually available for the prompt
    ocr_text = enriched_fig.get("figure_ocr_text") or ""
    second_pass_used_ocr_text = cache_status in {"cache_hit_positive", "fresh_ocr"} and bool(ocr_text)

    # OCR unavailable → return first pass with ocr_unavailable outcome
    if not second_pass_used_ocr_text:
        first_pass["verify_meta"] = {
            "verify_mode": "escalated_figure_ocr",
            "selective_figure_ocr_enabled": True,
            "escalation_attempted": True,
            "escalation_trigger": trigger_result["trigger"],
            "trigger": trigger_result["trigger"],
            "figure_present": True,
            "explicit_figure_ref_detected": explicit_figure_ref_detected,
            "escalated_figure_label": fig_label,
            "escalated_figure_rank": top_figure_rank,
            "escalated_ocr_cache_status": cache_status,
            "first_pass_verdict": first_verdict,
            "first_pass_confidence": first_conf,
            "second_pass_verdict": None,
            "second_pass_confidence": None,
            "verdict_changed": False,
            "confidence_delta": None,
            "confidence_improved": False,
            "second_pass_used_ocr_text": False,
            "escalation_outcome": "ocr_unavailable",
            **_lex,
        }
        return first_pass

    # OCR available — rebuild evidence list with enriched figure, run second pass
    top_fig_id = top_fig.get("evidence_object_id")
    enriched_evidence: list[dict[str, Any]] = [
        enriched_fig if (ev.get("evidence_object_id") == top_fig_id and ev.get("object_type") == "figure")
        else ev
        for ev in evidence
    ]

    # --- Second pass ---
    t0 = time.perf_counter()
    second_pass = verify_claim(claim, enriched_evidence, model=model)
    log_timing(
        "claim_llm_verify_second_pass",
        t0,
        claim_id=claim_id,
        evidence_count=len(enriched_evidence),
        trigger=trigger_result["trigger"],
        cache_status=cache_status,
    )

    second_verdict = second_pass.get("verdict")
    second_conf = second_pass.get("confidence")

    # Compute confidence_delta and confidence_improved
    if first_conf is not None and second_conf is not None:
        confidence_delta: float | None = second_conf - first_conf
        confidence_improved = confidence_delta > 0.05
    else:
        confidence_delta = None
        confidence_improved = False

    verdict_changed = first_verdict != second_verdict

    # Classify escalation_outcome (mutually exclusive)
    if verdict_changed:
        escalation_outcome = "verdict_changed"
    elif confidence_improved:
        escalation_outcome = "confidence_improved_only"
    else:
        escalation_outcome = "no_change"

    log.info(
        "[verify-figure-ocr] claim=%s second_pass verdict=%s conf=%s verdict_changed=%s outcome=%s",
        claim_id, second_verdict, second_conf, verdict_changed, escalation_outcome,
    )

    second_pass["verify_meta"] = {
        "verify_mode": "escalated_figure_ocr",
        "selective_figure_ocr_enabled": True,
        "escalation_attempted": True,
        "escalation_trigger": trigger_result["trigger"],
        "trigger": trigger_result["trigger"],
        "figure_present": True,
        "explicit_figure_ref_detected": explicit_figure_ref_detected,
        "escalated_figure_label": fig_label,
        "escalated_figure_rank": top_figure_rank,
        "escalated_ocr_cache_status": cache_status,
        "first_pass_verdict": first_verdict,
        "first_pass_confidence": first_conf,
        "second_pass_verdict": second_verdict,
        "second_pass_confidence": second_conf,
        "verdict_changed": verdict_changed,
        "confidence_delta": confidence_delta,
        "confidence_improved": confidence_improved,
        "second_pass_used_ocr_text": second_pass_used_ocr_text,
        "escalation_outcome": escalation_outcome,
        **_lex,
    }
    return second_pass


def _retrieve_live(
    conn: sqlite3.Connection,
    claim: dict[str, Any],
    *,
    top_k: int,
    embedding_model,
    embedding_model_id: str | None,
) -> tuple[list[dict], Any, str | None]:
    try:
        from gsr.paper_retrieval.retrieval import retrieve_evidence_for_claim
        from gsr.paper_retrieval.storage.storage import save_retrieval_results

        log.info(
            "Live retrieval start: claim_id=%s paper_id=%s top_k=%d embedding_model_id=%s has_model=%s",
            claim["id"],
            claim["paper_id"],
            top_k,
            embedding_model_id,
            embedding_model is not None,
        )

        results, embedding_model, embedding_model_id = retrieve_evidence_for_claim(
            claim,
            claim["paper_id"],
            conn,
            top_k=top_k,
            model=embedding_model,
            model_id=embedding_model_id,
        )

        cache_model_id = embedding_model_id or "allenai/specter2_base"

        with conn:
            conn.execute(
                """
                DELETE FROM retrieval_results
                WHERE claim_id = ? AND model_id = ?
                """,
                (claim["id"], cache_model_id),
            )

        save_retrieval_results(claim["id"], results, cache_model_id, conn)

        log.info(
            "Live retrieval saved: claim_id=%s rows=%d cache_model_id=%s",
            claim["id"],
            len(results),
            cache_model_id,
        )

        return results, embedding_model, cache_model_id

    except Exception as exc:
        log.warning(
            "Live retrieval failed for claim '%s' (paper '%s'): %s",
            claim["id"], claim["paper_id"], exc,
        )
        return [], embedding_model, embedding_model_id

def load_cached_evidence_mixed(
    conn: sqlite3.Connection,
    claim_id: str,
    top_k: int,
    model_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load cached retrieval results, supporting BOTH:
    1) multimodal-lite evidence_objects rows
    2) text-only paper_chunks fallback
    """
    conditions = ["rr.claim_id = ?"]
    params: list[Any] = [claim_id]

    if model_id:
        conditions.append("rr.model_id = ?")
        params.append(model_id)

    where = " AND ".join(conditions)

    query = f"""
        SELECT
            rr.rank,
            rr.chunk_id,
            rr.evidence_object_id,
            rr.object_type,
            rr.label,
            rr.model_id,
            rr.bm25_score,
            rr.semantic_score,
            rr.reference_boost,
            rr.combined_score AS score,

            eo.section AS eo_section,
            eo.page AS eo_page,
            eo.page_start AS eo_page_start,
            eo.page_end AS eo_page_end,
            eo.caption_text,
            eo.content_text,
            eo.bbox_json,
            eo.span_ids_json AS eo_span_ids_json,
            eo.metadata_json AS eo_metadata_json,
            eo.asset_path,

            pc.section AS pc_section,
            pc.page AS pc_page,
            pc.text AS pc_text,
            pc.span_ids_json AS pc_span_ids_json

        FROM retrieval_results rr
        LEFT JOIN evidence_objects eo ON eo.id = rr.evidence_object_id
        LEFT JOIN paper_chunks pc ON pc.id = rr.chunk_id
        WHERE {where}
        ORDER BY rr.rank
        LIMIT ?
    """
    params.append(top_k)

    conn.row_factory = sqlite3.Row
    rows = conn.execute(query, params).fetchall()
    conn.row_factory = None

    out: list[dict[str, Any]] = []

    for row in rows:
        d = dict(row)

        raw_span_ids_json = d.get("eo_span_ids_json") or d.get("pc_span_ids_json") or "[]"
        try:
            span_ids = json.loads(raw_span_ids_json)
        except Exception:
            span_ids = []

        try:
            bbox = json.loads(d["bbox_json"]) if d.get("bbox_json") else None
        except Exception:
            bbox = None

        try:
            eo_metadata = json.loads(d["eo_metadata_json"]) if d.get("eo_metadata_json") else {}
        except Exception:
            eo_metadata = {}
        # figure_ocr_* schema — only populated for figure evidence objects
        ocr_text = eo_metadata.get("figure_ocr_text") or ""
        ocr_attempted = eo_metadata.get("figure_ocr_attempted", False)
        ocr_skip_reason = eo_metadata.get("figure_ocr_skip_reason")
        ocr_quality = eo_metadata.get("figure_ocr_quality")
        bbox_confidence = eo_metadata.get("object_bbox_confidence")
        # Phase A: figure crop signals (only populated for figure evidence objects)
        crop_quality = eo_metadata.get("crop_quality")

        text = (
            d.get("content_text")
            or d.get("caption_text")
            or d.get("pc_text")
            or ""
        )

        section = d.get("eo_section") or d.get("pc_section")
        page = d.get("eo_page") or d.get("eo_page_start") or d.get("pc_page")
        page_start = d.get("eo_page_start") or d.get("pc_page")
        page_end = d.get("eo_page_end") or d.get("pc_page")

        object_type = d.get("object_type")
        if not object_type:
            object_type = "text_chunk" if d.get("chunk_id") else "unknown"

        reference_boost = d.get("reference_boost") or 0.0

        out.append(
            {
                "rank": d.get("rank"),
                "chunk_id": d.get("chunk_id"),
                "evidence_object_id": d.get("evidence_object_id"),
                "object_type": object_type,
                "evidence_type": object_type,
                "label": d.get("label"),
                "evidence_label": d.get("label") or f"E{d.get('rank')}",
                "model_id": d.get("model_id"),
                "bm25_score": d.get("bm25_score"),
                "semantic_score": d.get("semantic_score"),
                "reference_boost": reference_boost,
                "reference_matched": bool(reference_boost > 0),
                "score": d.get("score"),
                "section": section,
                "page": page,
                "page_num": page,
                "page_start": page_start,
                "page_end": page_end,
                "caption_text": d.get("caption_text"),
                "caption": d.get("caption_text"),
                "content_text": d.get("content_text"),
                "text": text,
                "bbox": bbox,
                "object_bbox": bbox,
                "span_ids": span_ids,
                # figure_ocr_* fields (empty/None for non-figure evidence)
                "figure_ocr_text": ocr_text,
                "figure_ocr_attempted": ocr_attempted,
                "figure_ocr_skip_reason": ocr_skip_reason,
                "figure_ocr_quality": ocr_quality,
                "object_bbox_confidence": bbox_confidence,
                # Phase A: figure crop signals (None/absent for non-figure evidence)
                "asset_path": d.get("asset_path"),
                "crop_quality": crop_quality,
            }
        )

    if not out:
        log.warning(
            "load_cached_evidence_mixed miss: claim_id=%s model_id=%s top_k=%d",
            claim_id,
            model_id,
            top_k,
        )
    else:
        log.info(
            "load_cached_evidence_mixed hit: claim_id=%s model_id=%s rows=%d first_source=%s",
            claim_id,
            model_id,
            len(out),
            out[0].get("object_type"),
        )

    return out

def normalize_structured_evidence(evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    structured = []
    for ev in evidence_items:
        reference_boost = ev.get("reference_boost", 0.0) or 0.0
        structured.append(
            {
                "chunk_id": ev.get("chunk_id"),
                "evidence_object_id": ev.get("evidence_object_id") or ev.get("chunk_id"),
                "object_type": ev.get("object_type") or "text_chunk",
                "evidence_type": ev.get("object_type") or "text_chunk",
                "label": ev.get("label") or ev.get("evidence_label"),
                "evidence_label": ev.get("evidence_label") or ev.get("label"),
                "rank": ev.get("rank"),
                "score": ev.get("score"),
                "section": ev.get("section"),
                "page": ev.get("page") or ev.get("page_start"),
                "page_num": ev.get("page") or ev.get("page_start"),
                "page_start": ev.get("page_start"),
                "page_end": ev.get("page_end"),
                "caption_text": ev.get("caption_text"),
                "caption": ev.get("caption") or ev.get("caption_text"),
                "text": (
                    ev.get("text")
                    or ev.get("content_text")
                    or ev.get("caption_text")
                    or ""
                ),
                "bbox": ev.get("bbox") or ev.get("object_bbox"),
                "object_bbox": ev.get("bbox") or ev.get("object_bbox"),
                "span_ids": ev.get("span_ids", []),
                "reference_boost": reference_boost,
                "reference_matched": bool(
                    ev.get("reference_matched") or reference_boost > 0
                ),
                # figure_ocr_* debug/evaluation fields (empty/None for non-figure)
                "figure_ocr_text": ev.get("figure_ocr_text") or "",
                "figure_ocr_attempted": ev.get("figure_ocr_attempted", False),
                "figure_ocr_skip_reason": ev.get("figure_ocr_skip_reason"),
                "figure_ocr_quality": ev.get("figure_ocr_quality"),
                "object_bbox_confidence": ev.get("object_bbox_confidence"),
            }
        )
    return structured
