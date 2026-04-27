"""Claim-to-evidence retrieval – Module 3, Step 5.

Upgraded V1 multimodal-lite retrieval:
- Prefer unified evidence_objects (text_chunk / table / figure)
- Fall back to paper_chunks if evidence_objects are not available
- Apply explicit-reference boosting when the claim mentions:
  - Table 3 / Table 3.1
  - Figure 2 / Fig. 2
  - Section 3 / Section 3.2 / Sec. 3.2

Ranking:
- BM25 on retrieval_text
- Semantic cosine similarity on stored embeddings
- Reference-aware boost
- Small type prior for result/comparison/ablation-style claims
- Hard include for explicit Figure/Table label matches

Returned evidence dicts now support mixed evidence:
- text_chunk
- table
- figure
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from typing import Any

import numpy as np

from .parsing.reference_parser import compute_reference_boost

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _retrieve_live(
    conn: sqlite3.Connection,
    claim: dict[str, Any],
    *,
    top_k: int,
    embedding_model,
    embedding_model_id: str | None,
) -> tuple[list[dict], Any, str | None]:
    """Retrieve evidence on-the-fly via Module 3 if no cached results exist.

    Returns:
        (results, embedding_model, embedding_model_id)
    """
    try:
        from .storage.storage import save_retrieval_results

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

        print(f"Live retrieval found {len(results)} evidence objects for claim '{claim['id']}'.")
        return results, embedding_model, cache_model_id

    except Exception as exc:
        log.warning(
            "Live retrieval failed for claim '%s' (paper '%s'): %s",
            claim["id"], claim["paper_id"], exc,
        )
        return [], embedding_model, embedding_model_id


def retrieve_evidence_for_claim(
    claim: dict,
    paper_id: str,
    conn,
    *,
    top_k: int = 5,
    bm25_weight: float = 0.35,
    semantic_weight: float = 0.45,
    reference_weight: float = 0.20,
    model=None,
    model_id: str | None = None,
) -> tuple[list[dict], object, str | None]:
    """Retrieve top-k evidence objects for one claim.

    Returns:
        (results, model, model_id)

    Strategy:
    1) Prefer evidence_objects if present
    2) Otherwise fall back to text-only paper_chunks
    3) Blend BM25 + semantic + explicit-reference boost
    4) Hard include explicit Figure/Table object matches when present
    """
    from .storage.embeddings import load_embedding_model

    claim_id = claim.get("id")
    query = (claim.get("claim_text") or "").strip()

    if not query:
        return [], model, model_id

    evidence_objects = _load_evidence_objects_for_paper(conn, paper_id)
    using_evidence_objects = bool(evidence_objects)

    if not evidence_objects:
        evidence_objects = _load_chunks_as_evidence(conn, paper_id)

    if not evidence_objects:
        log.warning(
            "retrieve_evidence_for_claim: no evidence found for paper_id=%s claim_id=%s",
            paper_id,
            claim_id,
        )
        return [], model, model_id

    texts = [ev["retrieval_text"] for ev in evidence_objects]

    # BM25
    bm25_scores = _compute_bm25(query, texts)
    bm25_norm = _min_max_normalize(bm25_scores)

    resolved_model_id = model_id or "allenai/specter2_base"
    stored_embeddings = _load_embeddings_for_paper(
        conn,
        paper_id,
        resolved_model_id,
        using_evidence_objects=using_evidence_objects,
    )

    semantic_scores = np.zeros(len(evidence_objects), dtype=float)
    semantic_norm = np.zeros(len(evidence_objects), dtype=float)

    if stored_embeddings and len(stored_embeddings) == len(evidence_objects):
        if model is None:
            log.info(
                "retrieve_evidence_for_claim: loading embedding model for claim_id=%s model_id=%s",
                claim_id,
                resolved_model_id,
            )
            model, resolved_model_id = load_embedding_model(resolved_model_id)

        try:
            emb_matrix = np.array(stored_embeddings, dtype=float)
            query_vec = np.array(model.encode([query])[0], dtype=float)
            semantic_scores = _cosine_similarity_batch(query_vec, emb_matrix)
            semantic_norm = _min_max_normalize(semantic_scores)
        except Exception as exc:
            log.warning(
                "Semantic retrieval failed; falling back to BM25 + reference-only: claim_id=%s paper_id=%s err=%s",
                claim_id,
                paper_id,
                exc,
            )
            semantic_scores = np.zeros(len(evidence_objects), dtype=float)
            semantic_norm = np.zeros(len(evidence_objects), dtype=float)
    else:
        log.info(
            "retrieve_evidence_for_claim: no aligned stored embeddings for paper_id=%s model_id=%s -> lexical retrieval only (+ reference boost)",
            paper_id,
            resolved_model_id,
        )

    # Explicit reference boost: Table 3.1 / Figure 2 / Section 3.2
    reference_boost_map = compute_reference_boost(query, evidence_objects)
    reference_scores = np.array(
        [reference_boost_map.get(ev["id"], 0.0) for ev in evidence_objects],
        dtype=float,
    )

    # Small type prior
    type_priors = np.array(
        [_type_prior(query, ev.get("object_type")) for ev in evidence_objects],
        dtype=float,
    )

    combined_scores = (
        bm25_weight * bm25_norm
        + semantic_weight * semantic_norm
        + reference_weight * reference_scores
        + 0.05 * type_priors
    )

    # Build candidate rows first
    candidates: list[dict[str, Any]] = []
    for idx, ev in enumerate(evidence_objects):
        row = _format_evidence_result(
            evidence=ev,
            bm25_score=float(bm25_scores[idx]),
            semantic_score=float(semantic_scores[idx]) if len(semantic_scores) else 0.0,
            reference_boost=float(reference_scores[idx]),
            combined_score=float(combined_scores[idx]),
        )
        candidates.append(row)

    # Sort by combined score descending
    candidates.sort(key=lambda r: float(r.get("combined_score") or 0.0), reverse=True)

    # HARD INCLUDE explicit Figure/Table object matches
    candidates = _force_include_explicit_object_matches(query, candidates)

    # Final top-k with diversity caps
    results: list[dict] = []
    type_counts: dict[str, int] = {}

    for cand in candidates:
        obj_type = cand.get("object_type") or "text_chunk"

        # Diversity cap so top-k is not all adjacent text chunks
        if obj_type == "text_chunk" and type_counts.get(obj_type, 0) >= 3:
            continue
        if obj_type == "table" and type_counts.get(obj_type, 0) >= 2:
            continue
        if obj_type == "figure" and type_counts.get(obj_type, 0) >= 1:
            continue

        cand["rank"] = len(results) + 1
        results.append(cand)

        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        if len(results) >= top_k:
            break

    log.info(
        "retrieve_evidence_for_claim: claim_id=%s paper_id=%s evidence=%d top_k=%d returned=%d model_id=%s has_model=%s has_embeddings=%s using_evidence_objects=%s refs=%d",
        claim_id,
        paper_id,
        len(evidence_objects),
        top_k,
        len(results),
        resolved_model_id,
        model is not None,
        bool(stored_embeddings),
        using_evidence_objects,
        len(reference_boost_map),
    )

    return results, model, resolved_model_id


def retrieve_evidence_for_claims(
    claims: list[dict],
    paper_id: str,
    conn: sqlite3.Connection,
    *,
    top_k: int = 5,
    bm25_weight: float = 0.35,
    semantic_weight: float = 0.45,
    reference_weight: float = 0.20,
    model=None,
    model_id: str | None = None,
) -> dict[str, list[dict]]:
    """Retrieve evidence for multiple claims in one call.

    Loads the embedding model once and reuses it across all claims.
    """
    resolved_model_id = model_id or "allenai/specter2_base"

    if model is None:
        embeddings = _load_embeddings_for_paper(
            conn,
            paper_id,
            resolved_model_id,
            using_evidence_objects=True,
        )
        if not embeddings:
            embeddings = _load_embeddings_for_paper(
                conn,
                paper_id,
                resolved_model_id,
                using_evidence_objects=False,
            )
        if embeddings:
            from .storage.embeddings import load_embedding_model
            model, resolved_model_id = load_embedding_model(resolved_model_id)

    results: dict[str, list[dict]] = {}
    for claim in claims:
        claim_id = claim.get("id", "")
        evs, model, resolved_model_id = retrieve_evidence_for_claim(
            claim,
            paper_id,
            conn,
            top_k=top_k,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            reference_weight=reference_weight,
            model=model,
            model_id=resolved_model_id,
        )
        results[claim_id] = evs
    return results


def retrieve_all_claims_for_paper(
    conn: sqlite3.Connection,
    *,
    paper_id: str,
    experiment_id: str | int | None = None,
    top_k: int = 5,
    embedding_model_id: str | None = None,
    force: bool = False,
    use_bbox_refine: bool = False,
    bbox_detector: str = "groundingdino",
) -> dict[str, Any]:
    """Build retrieval_results cache for all claims of a paper.

    Supported filtering modes:
    - experiment_id is an int / numeric string:
        interpreted as claims.extraction_run_id
    - experiment_id is a non-numeric string:
        interpreted as extraction_runs.experiment_id via JOIN
    """
    from .storage.storage import save_retrieval_results

    # [retrieve-timing] phase clocks
    _t_total_start = time.monotonic()
    _t_db_init_start = time.monotonic()  # covers col introspection + claims load

    resolved_model_id = embedding_model_id or "allenai/specter2_base"

    claim_cols = _get_table_columns(conn, "claims")
    run_cols = _get_table_columns(conn, "extraction_runs")

    has_extraction_run_id = "extraction_run_id" in claim_cols
    has_run_experiment_id = "experiment_id" in run_cols

    base_conditions = ["c.paper_id = ?"]
    params: list[Any] = [paper_id]
    join_extraction_runs = False

    if experiment_id is not None:
        try:
            extraction_run_id = int(experiment_id)
            if has_extraction_run_id:
                base_conditions.append("c.extraction_run_id = ?")
                params.append(extraction_run_id)
            else:
                log.warning(
                    "claims table has no extraction_run_id column; ignoring numeric experiment_id filter=%r",
                    experiment_id,
                )
        except (TypeError, ValueError):
            if has_extraction_run_id and has_run_experiment_id:
                join_extraction_runs = True
                base_conditions.append("er.experiment_id = ?")
                params.append(str(experiment_id))
            else:
                log.warning(
                    "Cannot apply string experiment_id filter=%r because "
                    "claims.extraction_run_id or extraction_runs.experiment_id is missing; "
                    "falling back to paper_id-only retrieval.",
                    experiment_id,
                )

    where = " AND ".join(base_conditions)

    select_cols = [
        "c.id",
        "c.review_id",
        "c.paper_id",
        "c.source_field",
        "c.claim_index",
        "c.claim_text",
        "c.verbatim_quote",
        "c.claim_type",
        "c.confidence",
        "c.category",
        "c.challengeability",
        "c.binary_question",
    ]

    if "why_challengeable" in claim_cols:
        select_cols.append("c.why_challengeable")
    if "model_id" in claim_cols:
        select_cols.append("c.model_id")
    if "extracted_at" in claim_cols:
        select_cols.append("c.extracted_at")
    if has_extraction_run_id:
        select_cols.append("c.extraction_run_id")

    from_sql = "FROM claims c"
    if join_extraction_runs:
        from_sql += " JOIN extraction_runs er ON c.extraction_run_id = er.id"

    sql = f"""
        SELECT {", ".join(select_cols)}
        {from_sql}
        WHERE {where}
        ORDER BY c.review_id, c.source_field, c.claim_index
    """

    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.row_factory = None

    claims = [dict(r) for r in rows]

    for c in claims:
        if "why_challengeable" not in c:
            c["why_challengeable"] = None
        if "model_id" not in c:
            c["model_id"] = None
        if "extracted_at" not in c:
            c["extracted_at"] = None
        if "extraction_run_id" not in c:
            c["extraction_run_id"] = None

    total_claims = len(claims)
    print(f"GSR_PROGRESS retrieve_claims 0 {total_claims}", flush=True)

    log.debug(
        "retrieve loaded claims: paper_id=%s experiment_id=%s model_id=%s count=%d",
        paper_id,
        experiment_id,
        resolved_model_id,
        total_claims,
    )
    if total_claims == 0:
        log.info("[retrieve] SKIP no_claims paper_id=%s", paper_id)
    else:
        log.info("[retrieve] START paper_id=%s claims=%d model_id=%s", paper_id, total_claims, resolved_model_id)

    _t_db_init = round(time.monotonic() - _t_db_init_start, 3)
    log.info(
        "[timing] stage=retrieve_db_init elapsed=%.3fs paper=%s claims=%d",
        _t_db_init, paper_id, total_claims,
    )
    _t_loop_start = time.monotonic()

    claims_processed = 0
    claims_skipped = 0
    cache_hits = 0
    errors = 0
    results_written = 0

    shared_model = None
    shared_model_id = resolved_model_id

    for idx, claim in enumerate(claims, start=1):
        claim_id = claim["id"]
        claim_text = (claim.get("claim_text") or "").strip().replace("\n", " ")
        claim_preview = claim_text[:100] + ("..." if len(claim_text) > 100 else "")

        if not force:
            existing = conn.execute(
                """
                SELECT COUNT(*)
                FROM retrieval_results
                WHERE claim_id = ? AND model_id = ?
                """,
                (claim_id, shared_model_id),
            ).fetchone()[0]

            if existing > 0:
                claims_skipped += 1
                cache_hits += 1

                log.info(
                    "Skipping retrieval for cached claim (%d/%d): claim_id=%s model_id=%s existing_rows=%d text=%r",
                    idx,
                    total_claims,
                    claim_id,
                    shared_model_id,
                    existing,
                    claim_preview,
                )
                print(f"GSR_PROGRESS retrieve_claims {idx} {total_claims}", flush=True)
                continue

        log.info(
            "Retrieving evidence (%d/%d): claim_id=%s text=%r",
            idx,
            total_claims,
            claim_id,
            claim_preview,
        )

        try:
            results, shared_model, shared_model_id = retrieve_evidence_for_claim(
                claim,
                paper_id,
                conn,
                top_k=top_k,
                model=shared_model,
                model_id=shared_model_id,
            )

            with conn:
                conn.execute(
                    """
                    DELETE FROM retrieval_results
                    WHERE claim_id = ? AND model_id = ?
                    """,
                    (claim_id, shared_model_id),
                )

            save_retrieval_results(claim_id, results, shared_model_id, conn)

            claims_processed += 1
            results_written += len(results)

            log.info(
                "Cached retrieval for claim (%d/%d): claim_id=%s rows=%d model_id=%s",
                idx,
                total_claims,
                claim_id,
                len(results),
                shared_model_id,
            )
            print(f"GSR_PROGRESS retrieve_claims {idx} {total_claims}", flush=True)

        except Exception as exc:
            errors += 1
            log.exception(
                "Failed retrieval caching for claim (%d/%d): claim_id=%s paper_id=%s error=%s",
                idx,
                total_claims,
                claim_id,
                paper_id,
                exc,
            )
            print(f"GSR_PROGRESS retrieve_claims {idx} {total_claims}", flush=True)

    _t_loop = round(time.monotonic() - _t_loop_start, 3)
    log.info(
        "[timing] stage=retrieve_loop elapsed=%.3fs paper=%s claims=%d avg_per_claim=%.3fs",
        _t_loop, paper_id, total_claims, _t_loop / max(total_claims, 1),
    )

    summary = {
        "claims_total": total_claims,
        "claims_processed": claims_processed,
        "claims_skipped": claims_skipped,
        "cache_hits": cache_hits,
        "errors": errors,
        "results_written": results_written,
    }

    # [retrieve-timing] bbox refine phase accumulators (set if refine actually runs)
    _t_refine = 0.0
    _t_refine_ids = 0
    _t_refine_pages = 0
    _t_refine_refined = 0

    # --- Selective bbox refinement (post-retrieval, paper-level batch) -------
    # Collect all figure/table evidence_object_ids present in retrieval_results
    # for this paper (includes cached results from prior runs), then refine only
    # those objects.  Already-attempted objects are skipped inside the function.
    if use_bbox_refine:
        _t_refine_start = time.monotonic()
        try:
            _pdf_row = conn.execute(
                "SELECT pdf_path FROM papers WHERE id = ?", (paper_id,)
            ).fetchone()
            _pdf_path = _pdf_row[0] if _pdf_row else None

            if _pdf_path:
                _hit_rows = conn.execute(
                    """
                    SELECT DISTINCT rr.evidence_object_id
                    FROM retrieval_results rr
                    JOIN claims c ON c.id = rr.claim_id
                    WHERE c.paper_id = ?
                      AND rr.object_type IN ('figure', 'table')
                      AND rr.evidence_object_id IS NOT NULL
                    """,
                    (paper_id,),
                ).fetchall()
                _hit_ids = [r[0] for r in _hit_rows]

                if _hit_ids:
                    # Count distinct pages for the selected evidence objects (pre-refine log)
                    _hit_id_placeholders = ",".join("?" * len(_hit_ids))
                    _page_rows = conn.execute(
                        f"""
                        SELECT DISTINCT COALESCE(page, page_start, 0)
                        FROM evidence_objects
                        WHERE id IN ({_hit_id_placeholders})
                          AND object_type IN ('figure', 'table')
                        """,
                        _hit_ids,
                    ).fetchall()
                    _distinct_pages = sorted(r[0] for r in _page_rows)
                    log.info(
                        "[retrieve] selective bbox refinement START: "
                        "claims_processed=%d distinct_figure_table_ids=%d "
                        "distinct_pages=%d pages=%s detector=%s",
                        claims_processed, len(_hit_ids),
                        len(_distinct_pages), _distinct_pages, bbox_detector,
                    )
                    log.info("SELECTIVE_REFINEMENT_CALL_CONFIRMED")
                    from .vision.bbox_refinement import refine_bbox_for_evidence_ids
                    _refine_stats = refine_bbox_for_evidence_ids(
                        _hit_ids, _pdf_path, conn, detector=bbox_detector,
                    )
                    summary["bbox_refine_stats"] = _refine_stats
                    _t_refine_ids = len(_hit_ids)
                    _t_refine_pages = _refine_stats.get("bbox_refine_pages_processed", 0)
                    _t_refine_refined = _refine_stats.get("bbox_refine_accepted", 0)
                    log.info(
                        "[retrieve] selective bbox refinement DONE: "
                        "refined_object_count=%d skipped_already_attempted=%d "
                        "skipped_non_visual=%d pages_processed=%d elapsed_s=%.2f",
                        _refine_stats.get("bbox_refine_accepted", 0),
                        _refine_stats.get("bbox_refine_skipped_already_attempted", 0),
                        _refine_stats.get("bbox_refine_skipped_non_visual", 0),
                        _refine_stats.get("bbox_refine_pages_processed", 0),
                        _refine_stats.get("bbox_refine_elapsed_s", 0.0),
                    )
                else:
                    log.info(
                        "[retrieve] selective bbox refinement: no figure/table hits in retrieval_results."
                    )
            else:
                log.warning(
                    "[retrieve] selective bbox refinement requested but pdf_path not found "
                    "for paper_id=%s; skipping.", paper_id,
                )
        except Exception as exc:
            log.warning(
                "[retrieve] selective bbox refinement failed (non-fatal): %s", exc,
            )
        _t_refine = round(time.monotonic() - _t_refine_start, 3)
        log.info(
            "[timing] stage=retrieve_bbox_refine elapsed=%.3fs ids=%d pages=%d refined=%d",
            _t_refine, _t_refine_ids, _t_refine_pages, _t_refine_refined,
        )

    if total_claims > 0:
        log.info(
            "[retrieve] DONE paper_id=%s claims=%d processed=%d cached=%d errors=%d",
            paper_id,
            total_claims,
            claims_processed,
            cache_hits,
            errors,
        )
    _t_total = round(time.monotonic() - _t_total_start, 3)
    log.info(
        "[timing] stage=retrieve_total elapsed=%.3fs paper=%s claims=%d "
        "db_init=%.3fs loop=%.3fs bbox_refine=%.3fs",
        _t_total, paper_id, total_claims, _t_db_init, _t_loop, _t_refine,
    )

    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_bm25(query: str, texts: list[str]) -> np.ndarray:
    """Return BM25Okapi scores for *query* against each text in *texts*."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "rank-bm25 is required for BM25 retrieval: pip install rank-bm25"
        ) from exc

    tokenized = [_bm25_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_bm25_tokenize(query))
    return np.array(scores, dtype=float)


def _bm25_tokenize(text: str) -> list[str]:
    """Tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _cosine_similarity_batch(
    query_vec: np.ndarray,
    emb_matrix: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between *query_vec* and each row of *emb_matrix*."""
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0.0:
        return np.zeros(len(emb_matrix), dtype=float)
    m_norms = np.linalg.norm(emb_matrix, axis=1)
    denom = np.where(m_norms == 0.0, 1e-10, m_norms) * q_norm
    return np.dot(emb_matrix, query_vec) / denom


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Scale *scores* to [0, 1]; returns zeros if all values are identical."""
    if len(scores) == 0:
        return scores
    s_min, s_max = scores.min(), scores.max()
    if s_max == s_min:
        return np.zeros_like(scores, dtype=float)
    return (scores - s_min) / (s_max - s_min)


def _type_prior(query: str, object_type: str | None) -> float:
    """Small prior to help result/comparison claims surface tables/figures."""
    q = query.lower()

    if object_type in {"table", "figure"} and re.search(
        r"\b(table|tab\.|figure|fig\.|ablation|baseline|compare|comparison|result|results|performance|qualitative)\b",
        q,
    ):
        return 1.0

    if object_type == "text_chunk" and re.search(
        r"\b(section|method|proof|assumption|definition|theorem)\b",
        q,
    ):
        return 0.4

    return 0.0


def _extract_explicit_refs(claim_text: str) -> dict[str, set[str]]:
    """Extract explicit Table/Figure references from claim text."""
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


def _force_include_explicit_object_matches(
    claim_text: str,
    ranked_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Guarantee that explicit Figure/Table object matches are moved to the front.

    Example:
    - claim mentions "Figure 9"
    - ranked results include:
        * text_chunk mentioning Figure 9
        * figure object labeled Figure 9
      -> move the figure object ahead of text_chunk
    """
    refs = _extract_explicit_refs(claim_text)

    explicit_objects: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []

    for r in ranked_results:
        obj_type = str(r.get("object_type") or "").lower()
        label = r.get("label")

        is_match = (
            (obj_type == "figure" and _label_matches_any(label, refs["figure"]))
            or
            (obj_type == "table" and _label_matches_any(label, refs["table"]))
        )

        if is_match:
            explicit_objects.append(r)
        else:
            others.append(r)

    if not explicit_objects:
        return ranked_results

    # Highest-score explicit objects first
    explicit_objects.sort(
        key=lambda r: float(r.get("combined_score") or 0.0),
        reverse=True,
    )

    # Deduplicate while preserving order: explicit objects first, then others
    final: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()

    for r in explicit_objects + others:
        key = (r.get("evidence_object_id"), r.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        final.append(r)

    return final


def _format_evidence_result(
    *,
    evidence: dict,
    bm25_score: float,
    semantic_score: float,
    reference_boost: float,
    combined_score: float,
) -> dict:
    """Build one returned evidence dict from a loaded evidence row."""
    obj_type = evidence.get("object_type") or "text_chunk"

    return {
        "evidence_object_id": evidence["id"],
        "chunk_id": evidence.get("chunk_id") or (evidence["id"] if obj_type == "text_chunk" else None),
        "object_type": obj_type,
        "label": evidence.get("label"),
        "paper_id": evidence["paper_id"],
        "section": evidence.get("section"),
        "section_number": evidence.get("section_number"),
        "page": evidence.get("page") or evidence.get("page_start"),
        "page_start": evidence.get("page_start"),
        "page_end": evidence.get("page_end"),
        "span_ids": evidence.get("span_ids"),
        "caption_text": evidence.get("caption_text"),
        "text": evidence.get("content_text") or evidence["retrieval_text"],
        "bm25_score": bm25_score,
        "semantic_score": semantic_score,
        "reference_boost": reference_boost,
        "reference_matched": reference_boost > 0,
        "combined_score": combined_score,
    }


def _load_evidence_objects_for_paper(
    conn: sqlite3.Connection,
    paper_id: str,
) -> list[dict]:
    """Load all evidence_objects for *paper_id* ordered by id.

    Returns [] if evidence_objects table does not exist or has no rows.
    """
    table_cols = _get_table_columns(conn, "evidence_objects")
    if not table_cols:
        return []

    sql = """
        SELECT
            id,
            paper_id,
            object_type,
            label,
            page,
            page_start,
            page_end,
            section,
            section_number,
            caption_text,
            retrieval_text,
            content_text,
            span_ids_json
        FROM evidence_objects
        WHERE paper_id = ?
        ORDER BY id
    """
    cur = conn.execute(sql, (paper_id,))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]

    for row in rows:
        row["span_ids"] = json.loads(row["span_ids_json"]) if row.get("span_ids_json") else None
        row.pop("span_ids_json", None)

    return rows


def _load_chunks_as_evidence(
    conn: sqlite3.Connection,
    paper_id: str,
) -> list[dict]:
    """Fallback path: load paper_chunks and expose them as text_chunk evidence."""
    chunks = _load_chunks_for_paper(conn, paper_id)
    out: list[dict] = []

    for chunk in chunks:
        out.append(
            {
                "id": chunk["id"],
                "chunk_id": chunk["id"],
                "paper_id": chunk["paper_id"],
                "object_type": "text_chunk",
                "label": None,
                "page": chunk.get("page"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "section": chunk.get("section"),
                "section_number": _extract_section_number(chunk.get("section")),
                "caption_text": None,
                "retrieval_text": chunk["text"],
                "content_text": chunk["text"],
                "span_ids": chunk.get("span_ids"),
            }
        )
    return out


def _extract_section_number(section: str | None) -> str | None:
    if not section:
        return None
    m = re.match(r"^(\d+(?:\.\d+)*)\b", section.strip())
    return m.group(1) if m else None


def _load_chunks_for_paper(
    conn: sqlite3.Connection,
    paper_id: str,
) -> list[dict]:
    """Load all chunks for *paper_id* ordered by chunk_index."""
    chunk_cols = _get_table_columns(conn, "paper_chunks")

    has_page_start = "page_start" in chunk_cols
    has_page_end = "page_end" in chunk_cols
    has_span_ids_json = "span_ids_json" in chunk_cols

    select_cols = [
        "id",
        "paper_id",
        "chunk_index",
        "section",
        "page",
        "text",
        "char_start",
        "char_end",
    ]

    if has_page_start:
        select_cols.append("page_start")
    if has_page_end:
        select_cols.append("page_end")
    if has_span_ids_json:
        select_cols.append("span_ids_json")

    sql = f"""
        SELECT {", ".join(select_cols)}
        FROM paper_chunks
        WHERE paper_id = ?
        ORDER BY chunk_index
    """

    cur = conn.execute(sql, (paper_id,))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]

    for row in rows:
        if "page_start" not in row:
            row["page_start"] = None
        if "page_end" not in row:
            row["page_end"] = None

        if "span_ids_json" in row:
            row["span_ids"] = json.loads(row["span_ids_json"]) if row["span_ids_json"] else None
            del row["span_ids_json"]
        else:
            row["span_ids"] = None

    return rows


def _load_embeddings_for_paper(
    conn: sqlite3.Connection,
    paper_id: str,
    model_id: str | None = None,
    *,
    using_evidence_objects: bool = True,
) -> list[list[float]]:
    """Load stored embeddings for a paper.

    Priority:
    - evidence_embeddings joined to evidence_objects, if using_evidence_objects=True
    - fallback to chunk_embeddings joined to paper_chunks
    """
    resolved_model_id = model_id

    if using_evidence_objects and _get_table_columns(conn, "evidence_embeddings"):
        if resolved_model_id is None:
            row = conn.execute(
                """
                SELECT ee.model_id
                FROM evidence_embeddings ee
                JOIN evidence_objects eo ON ee.evidence_object_id = eo.id
                WHERE eo.paper_id = ?
                ORDER BY ee.embedded_at DESC
                LIMIT 1
                """,
                (paper_id,),
            ).fetchone()
            if row is not None:
                resolved_model_id = row[0]

        if resolved_model_id is not None:
            cur = conn.execute(
                """
                SELECT ee.embedding
                FROM evidence_embeddings ee
                JOIN evidence_objects eo ON ee.evidence_object_id = eo.id
                WHERE eo.paper_id = ? AND ee.model_id = ?
                ORDER BY eo.id
                """,
                (paper_id, resolved_model_id),
            )
            rows = cur.fetchall()
            if rows:
                return [json.loads(row[0]) for row in rows]

    # fallback to legacy chunk embeddings
    resolved_model_id = model_id
    if resolved_model_id is None:
        row = conn.execute(
            """
            SELECT ce.model_id
            FROM chunk_embeddings ce
            JOIN paper_chunks pc ON ce.chunk_id = pc.id
            WHERE pc.paper_id = ?
            ORDER BY ce.embedded_at DESC
            LIMIT 1
            """,
            (paper_id,),
        ).fetchone()
        if row is None:
            return []
        resolved_model_id = row[0]

    cur = conn.execute(
        """
        SELECT ce.embedding
        FROM chunk_embeddings ce
        JOIN paper_chunks pc ON ce.chunk_id = pc.id
        WHERE pc.paper_id = ? AND ce.model_id = ?
        ORDER BY pc.chunk_index
        """,
        (paper_id, resolved_model_id),
    )

    rows = cur.fetchall()
    if not rows:
        return []
    return [json.loads(row[0]) for row in rows]


def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Return the set of column names for a SQLite table.

    Returns an empty set if the table does not exist.
    """
    try:
        cur = conn.execute(f"PRAGMA table_info({table_name})")
        return {row[1] for row in cur.fetchall()}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Late Docling table enrichment (top-k selective)
# ---------------------------------------------------------------------------

def enrich_topk_tables_with_docling(
    conn: sqlite3.Connection,
    claim_ids: list[str],
    top_k: int,
) -> dict[str, Any]:
    """Enrich top-k table evidence objects with Docling markdown before verification.

    For the given claim_ids, inspects cached retrieval_results (rank <= top_k),
    collects table evidence objects not yet Docling-enriched, runs Docling on
    the needed pages, and writes markdown into evidence_objects.content_text.

    This is a no-op when docling is not installed or no table candidates exist.

    Returns:
        {"enriched": int, "failed": int, "skipped_already": int}
    """
    from .parsing.parser_docling import extract_tables_for_pages, docling_available
    from .storage.storage import (
        update_table_evidence_docling_enrichment,
        mark_table_evidence_docling_failed,
    )

    if not docling_available():
        log.debug("[docling-late] docling not installed — skipping late enrichment")
        return {"enriched": 0, "failed": 0, "skipped_already": 0}

    if not claim_ids:
        return {"enriched": 0, "failed": 0, "skipped_already": 0}

    _t0_total = time.monotonic()

    # 1. Query retrieval_results for top-k table candidates across all claim_ids.
    placeholders = ",".join("?" for _ in claim_ids)
    rows = conn.execute(
        f"""
        SELECT DISTINCT rr.evidence_object_id,
               eo.paper_id, eo.page, eo.label, eo.caption_text, eo.metadata_json
        FROM retrieval_results rr
        JOIN evidence_objects eo ON eo.id = rr.evidence_object_id
        WHERE rr.claim_id IN ({placeholders})
          AND rr.rank <= ?
          AND rr.evidence_object_id IS NOT NULL
          AND eo.object_type = 'table'
        """,
        [*claim_ids, top_k],
    ).fetchall()

    if not rows:
        log.debug(
            "[docling-late] no table candidates for %d claims top_k=%d",
            len(claim_ids), top_k,
        )
        return {"enriched": 0, "failed": 0, "skipped_already": 0}

    # 2. Filter out already-enriched tables; group remainder by paper_id.
    candidates: list[dict[str, Any]] = []
    skipped_already = 0

    for ev_id, paper_id, page, label, caption_text, meta_json in rows:
        meta: dict = {}
        if meta_json:
            try:
                meta = json.loads(meta_json)
            except Exception:
                pass
        if meta.get("docling_enriched") is True:
            skipped_already += 1
            continue
        candidates.append({
            "ev_id": ev_id,
            "paper_id": paper_id,
            "page": page or 0,
            "label": label or "",
            "caption_text": caption_text or "",
        })

    if not candidates:
        log.info(
            "[docling-late] DONE all_already_enriched skipped_already=%d", skipped_already
        )
        return {"enriched": 0, "failed": 0, "skipped_already": skipped_already}

    # 3. Group by paper_id.
    from collections import defaultdict
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_paper[c["paper_id"]].append(c)

    log.info(
        "[docling-late] START papers=%d candidate_tables=%d",
        len(by_paper), len(candidates),
    )

    enriched = 0
    failed = 0

    for paper_id, paper_cands in by_paper.items():
        # Look up pdf_path from papers table.
        row = conn.execute(
            "SELECT pdf_path FROM papers WHERE id = ? LIMIT 1", (paper_id,)
        ).fetchone()
        if not row or not row[0]:
            log.warning("[docling-late] pdf_path not found for paper=%s", paper_id)
            for c in paper_cands:
                mark_table_evidence_docling_failed(conn, c["ev_id"], "pdf_path_not_found")
                failed += 1
            continue

        pdf_path = row[0]
        pages_needed = sorted({c["page"] for c in paper_cands if c["page"] > 0})

        log.info(
            "[docling-late] START paper=%s candidate_tables=%d candidate_pages=%d",
            paper_id, len(paper_cands), len(pages_needed),
        )

        # 4. Run Docling for needed pages (full-doc run, filtered by page).
        try:
            _t0_page = time.monotonic()
            page_table_map = extract_tables_for_pages(pdf_path, pages_needed)
            _elapsed = time.monotonic() - _t0_page
            extracted_total = sum(len(v) for v in page_table_map.values())
            log.info(
                "[docling-late] PAGE page=%s requested=%d extracted=%d elapsed=%.3fs",
                pages_needed, len(pages_needed), extracted_total, _elapsed,
            )
        except Exception as exc:
            log.warning("[docling-late] Docling failed for paper=%s: %s", paper_id, exc)
            for c in paper_cands:
                mark_table_evidence_docling_failed(conn, c["ev_id"], str(exc)[:200])
                failed += 1
            continue

        # 5. Match each candidate to the best extracted table on its page.
        for cand in paper_cands:
            page_tables = page_table_map.get(cand["page"], [])
            matched = _match_docling_table(cand, page_tables)

            if matched is None:
                mark_table_evidence_docling_failed(conn, cand["ev_id"], "no_match")
                failed += 1
                log.debug(
                    "[docling-late] no_match ev_id=%s label=%r page=%s",
                    cand["ev_id"], cand["label"], cand["page"],
                )
                continue

            markdown = matched["markdown"]
            match_mode = matched["match_mode"]
            update_table_evidence_docling_enrichment(
                conn,
                cand["ev_id"],
                markdown_text=markdown,
                metadata_updates={
                    "docling_match_mode": match_mode,
                    "docling_source_page": cand["page"],
                },
            )
            enriched += 1
            log.debug(
                "[docling-late] enriched ev_id=%s label=%r page=%s mode=%s chars=%d",
                cand["ev_id"], cand["label"], cand["page"], match_mode, len(markdown),
            )

    _total_elapsed = time.monotonic() - _t0_total
    log.info(
        "[docling-late] DONE paper=%s enriched=%d failed=%d skipped_already=%d elapsed=%.3fs",
        list(by_paper.keys())[0] if len(by_paper) == 1 else "multi",
        enriched, failed, skipped_already, _total_elapsed,
    )
    return {"enriched": enriched, "failed": failed, "skipped_already": skipped_already}


def _match_docling_table(
    cand: dict[str, Any],
    page_tables: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Match a table candidate to a Docling-extracted table on the same page.

    Priority (first match wins):
    1. exact_label_page  — case-insensitive exact label match
    2. label_page        — normalized label match (trailing punctuation stripped)
    3. caption_page      — caption word overlap >= 4 words
    4. page_only         — single table on the page (last resort)

    Returns the matched table dict with an added "match_mode" key, or None.
    """
    if not page_tables:
        return None

    cand_label = (cand.get("label") or "").strip().lower()
    cand_caption = (cand.get("caption_text") or "").strip().lower()

    def _norm(s: str) -> str:
        return re.sub(r"[:\.\s]+$", "", s.strip().lower())

    # 1) Exact label
    for t in page_tables:
        t_label = (t.get("label") or "").strip().lower()
        if cand_label and t_label and cand_label == t_label:
            return {**t, "match_mode": "exact_label_page"}

    # 2) Normalized label
    cand_label_norm = _norm(cand_label)
    for t in page_tables:
        t_label_norm = _norm((t.get("label") or "").strip().lower())
        if cand_label_norm and t_label_norm and cand_label_norm == t_label_norm:
            return {**t, "match_mode": "label_page"}

    # 3) Caption word overlap (>= 4 shared words)
    if cand_caption:
        cand_words = set(re.findall(r"\w+", cand_caption))
        best_overlap = 0
        best_t: dict[str, Any] | None = None
        for t in page_tables:
            t_cap = (t.get("caption_text") or "").strip().lower()
            if not t_cap:
                continue
            t_words = set(re.findall(r"\w+", t_cap))
            overlap = len(cand_words & t_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_t = t
        if best_overlap >= 4 and best_t is not None:
            return {**best_t, "match_mode": "caption_page"}

    # 4) Page-only fallback
    if len(page_tables) == 1:
        return {**page_tables[0], "match_mode": "page_only"}

    return None