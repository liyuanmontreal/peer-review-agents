"""Phase P1.5 — GroundingDINO bbox refinement for figure/table evidence objects.

After evidence objects have been built and persisted (caption-first identity
from PyMuPDF + caption_extractor), this module optionally refines their
bounding boxes by matching GroundingDINO detections against existing objects.

Design rules:
  - Does NOT change object identity (id, label, page, object_type).
  - Does NOT change retrieval_text, content_text, caption_text.
  - Does NOT recompute embeddings.
  - Stores all results in evidence_objects.metadata_json only (no schema change).
  - Falls back gracefully if GroundingDINO is unavailable or detection fails.

Public entry point:
  refine_bbox_for_paper(paper_id, pdf_path, conn) → summary dict
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_RENDER_SCALE = 2.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _area(bbox: list[float]) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ---------------------------------------------------------------------------
# Page area helper (cached per call via dict in caller)
# ---------------------------------------------------------------------------

def _page_area_pdf(pdf_path: str, page_0idx: int) -> float:
    """Return page area in PDF coordinate units, or 0 on error."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page = doc[page_0idx]
        area = page.rect.width * page.rect.height
        doc.close()
        return area
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _best_match(
    obj_bbox: list[float],
    obj_type: str,
    usable_detections: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (best_detection, rejection_reason) from pre-filtered usable detections.

    Candidates are detections of the same type as the evidence object.
    The nearest candidate (by centre-point Euclidean distance) is chosen.
    Area / coverage filtering is done upstream by filter_detections().
    """
    anchor = _center(obj_bbox)
    candidates = [d for d in usable_detections if d["label"] == obj_type]

    if not candidates:
        return None, "no_candidates_of_type"

    best = min(candidates, key=lambda d: _euclidean(anchor, _center(d["bbox"])))
    return best, None


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_meta(conn: sqlite3.Connection, obj_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT metadata_json FROM evidence_objects WHERE id = ?", (obj_id,)
    ).fetchone()
    if not row or not row[0]:
        return {}
    try:
        return json.loads(row[0])
    except Exception:
        return {}


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to native Python types.

    Defensive pass applied right before json.dumps so that any detector
    backend that leaks numpy scalars (e.g. np.float32) into metadata dicts
    does not cause 'Object of type float32 is not JSON serializable'.
    """
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    # numpy scalar or 0-d array: .item() gives the native Python equivalent
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _save_meta(
    conn: sqlite3.Connection, obj_id: str, meta: dict[str, Any]
) -> None:
    with conn:
        conn.execute(
            "UPDATE evidence_objects SET metadata_json = ? WHERE id = ?",
            (json.dumps(_to_jsonable(meta), ensure_ascii=False), obj_id),
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def refine_bbox_for_paper(
    paper_id: str,
    pdf_path: str,
    conn: sqlite3.Connection,
    *,
    render_scale: float = _DEFAULT_RENDER_SCALE,
    detector: str = "groundingdino",
) -> dict[str, Any]:
    """Run bbox refinement for all figure/table evidence objects belonging to *paper_id*.

    Loads existing evidence objects from DB, runs per-page detection using the
    selected detector backend, matches detections to objects, and persists results
    to metadata_json only (no schema / identity changes).

    Args:
        detector: Backend to use — "groundingdino" (default) or "rtdetr".

    Returns a summary dict with keys:
      bbox_refine_enabled, bbox_refine_available, bbox_refine_detector,
      bbox_refine_attempted, bbox_refine_accepted, bbox_refine_rejected
    """
    # --- Detector dispatch --------------------------------------------------
    if detector == "rtdetr":
        from .rtdetr_detector import (
            rtdetr_available as _detector_available,
            detect_on_page,
            filter_detections,
            _MODEL_ID,
        )
        _backend_name = "rtdetr"
        _meta_prefix = "rtdetr"
        _raw_label_key = "rtdetr_label"
    elif detector == "paddlex_layout":
        from .paddlex_layout_detector import (
            paddlex_layout_available as _detector_available,
            detect_on_page,
            filter_detections,
            _MODEL_ID,
        )
        _backend_name = "paddlex_layout"
        _meta_prefix = "paddlex_layout"
        _raw_label_key = "paddlex_layout_label"
    else:
        from .grounding_dino import (
            grounding_dino_available as _detector_available,
            detect_on_page,
            filter_detections,
            _MODEL_ID,
        )
        _backend_name = "groundingdino"
        _meta_prefix = "dino"
        _raw_label_key = "dino_label"

    from ..storage.storage import load_evidence_objects_for_paper

    _base = {"bbox_refine_enabled": True, "bbox_refine_detector": detector}

    if not _detector_available():
        log.warning(
            "[bbox-refine] detector=%s unavailable; skipping refinement for paper '%s'.",
            detector, paper_id,
        )
        return {**_base, "bbox_refine_available": False,
                "bbox_refine_attempted": 0, "bbox_refine_accepted": 0,
                "bbox_refine_rejected": 0}

    log.info(
        "[bbox-refine] setup enabled=True detector=%s model=%s render_scale=%.1f paper=%s",
        detector, _MODEL_ID, render_scale, paper_id,
    )

    all_objects = load_evidence_objects_for_paper(paper_id, conn)
    targets = [o for o in all_objects if o["object_type"] in ("figure", "table")]

    n_figures = sum(1 for o in targets if o["object_type"] == "figure")
    n_tables = sum(1 for o in targets if o["object_type"] == "table")

    if not targets:
        return {**_base, "bbox_refine_available": True,
                "bbox_refine_attempted": 0, "bbox_refine_accepted": 0,
                "bbox_refine_rejected": 0}

    # Group objects by 0-based page index so we render each page once
    by_page: dict[int, list[dict[str, Any]]] = {}
    for obj in targets:
        pg = (obj.get("page") or obj.get("page_start") or 1) - 1
        by_page.setdefault(pg, []).append(obj)

    attempted = accepted = rejected = unavailable = 0
    # Detection-quality counters
    raw_detections_total = 0
    filtered_oversize = filtered_too_small = filtered_invalid = 0
    no_detection_pages = pages_with_raw = pages_with_usable = 0
    match_failures = 0

    now = datetime.now(timezone.utc).isoformat()
    page_area_cache: dict[int, float] = {}

    # --- Meta-update helpers (parameterized by detector backend) -------------
    _mp = _meta_prefix  # "dino" or "rtdetr"

    def _meta_fail(reason: str, *, attempted: bool = True, include_scale: bool = False) -> dict:
        d: dict = {
            f"{_mp}_attempted": attempted,
            f"{_mp}_matched": False,
            f"{_mp}_rejected_reason": reason,
            f"{_mp}_last_updated_at": now,
        }
        if include_scale:
            d[f"{_mp}_render_scale"] = render_scale
        return d

    def _meta_accept(best_det: dict, dist: float) -> dict:
        return {
            "detected_bbox": best_det["bbox"],
            "bbox_source": _backend_name,
            "object_bbox_confidence": "refined",
            f"{_mp}_score": best_det["score"],
            f"{_mp}_label": best_det.get(_raw_label_key, best_det["label"]),
            f"{_mp}_model": best_det.get("model", ""),
            f"{_mp}_match_mode": "same_page_nearest_anchor",
            f"{_mp}_match_distance": round(dist, 2),
            f"{_mp}_attempted": True,
            f"{_mp}_matched": True,
            f"{_mp}_render_scale": render_scale,
            f"{_mp}_last_updated_at": now,
        }

    sorted_pages = sorted(by_page.items())
    _compat_failed = False  # set on first processor compatibility error

    for pg0, page_objs in sorted_pages:
        if _compat_failed:
            # Fast-path: mark remaining objects unavailable without re-running detection
            for obj in page_objs:
                meta = _load_meta(conn, obj["id"])
                meta.update(_meta_fail("compat_failure_skipped", attempted=False))
                _save_meta(conn, obj["id"], meta)
                unavailable += 1
            continue

        if pg0 not in page_area_cache:
            page_area_cache[pg0] = _page_area_pdf(pdf_path, pg0)
        page_area = page_area_cache[pg0]

        try:
            raw_dets = detect_on_page(
                pdf_path, pg0, render_scale=render_scale
            )
        except RuntimeError as exc:
            err_str = str(exc)
            if "compatibility failure" in err_str or "unexpected keyword" in err_str:
                log.error(
                    "[bbox-refine] Processor compatibility failure on page=%d paper=%s — "
                    "aborting refinement for this paper. %s",
                    pg0 + 1, paper_id, exc,
                )
                _compat_failed = True
            else:
                log.warning(
                    "[bbox-refine] Detection failed page=%d paper=%s: %s",
                    pg0 + 1, paper_id, exc,
                )
                
                log.exception(
                    "[bbox-refine] Detection failed page=%d paper=%s error=%s",
                    pg0 + 1,
                    paper_id,
                    str(exc)
                )
            for obj in page_objs:
                meta = _load_meta(conn, obj["id"])
                meta.update(_meta_fail("detection_exception"))
                _save_meta(conn, obj["id"], meta)
                unavailable += 1
            continue
        except Exception as exc:
            log.warning(
                "[bbox-refine] Detection failed page=%d paper=%s: %s",
                pg0 + 1, paper_id, exc,
            )
            log.exception(
                    "[bbox-refine] Detection failed page=%d paper=%s error=%s",
                    pg0 + 1,
                    paper_id,
                    str(exc)
            )
            for obj in page_objs:
                meta = _load_meta(conn, obj["id"])
                meta.update(_meta_fail("detection_exception"))
                _save_meta(conn, obj["id"], meta)
                unavailable += 1
            continue

        # Page-level detection quality accounting
        usable_dets, rejected_dets = filter_detections(raw_dets, page_area)
        raw_detections_total += len(raw_dets)
        if len(raw_dets) == 0:
            no_detection_pages += 1
        else:
            pages_with_raw += 1
        if usable_dets:
            pages_with_usable += 1
        for r in rejected_dets:
            reason = r.get("reject_reason", "")
            if reason == "page_coverage_too_large":
                filtered_oversize += 1
            elif reason == "area_too_small":
                filtered_too_small += 1
            elif reason == "invalid_bbox":
                filtered_invalid += 1

        # Determine page-level rejection reason (used when all candidates were filtered)
        def _page_reject_reason() -> str:
            if not raw_dets:
                return "no_raw_detection"
            if not usable_dets:
                reasons = {r.get("reject_reason") for r in rejected_dets}
                if reasons <= {"page_coverage_too_large"}:
                    return "only_oversize_candidates"
                if reasons <= {"area_too_small"}:
                    return "only_too_small_candidates"
                return "all_candidates_filtered"
            return ""  # usable dets exist — reason depends on per-object matching

        page_level_reason = _page_reject_reason()

        for obj in page_objs:
            obj_id = obj["id"]
            obj_type = obj["object_type"]
            obj_label = obj.get("label")
            obj_page_1 = pg0 + 1

            obj_bbox = obj.get("bbox")
            if not obj_bbox or len(obj_bbox) < 4 or _area(obj_bbox) == 0:
                # No usable anchor bbox — skip this object
                log.debug(
                    "[bbox-refine] object id=%s label=%s page=%d no anchor bbox; skipping",
                    obj_id, obj_label, obj_page_1,
                )
                meta = _load_meta(conn, obj_id)
                meta.update(_meta_fail("no_anchor_bbox"))
                _save_meta(conn, obj_id, meta)
                unavailable += 1
                continue

            attempted += 1
            same_type_usable = sum(1 for d in usable_dets if d["label"] == obj_type)
            best_det, match_reason = _best_match(obj_bbox, obj_type, usable_dets)

            if best_det is None:
                rejected += 1
                if page_level_reason:
                    final_reason = page_level_reason
                else:
                    # Usable dets exist but none of the right type
                    final_reason = match_reason or "candidates_exist_but_no_anchor_match"
                    match_failures += 1
                log.info(
                    "[bbox-refine] object type=%s label=%s page=%d "
                    "raw=%d usable=%d same_type=%d accepted=False reason=%s",
                    obj_type, obj_label, obj_page_1,
                    len(raw_dets), len(usable_dets), same_type_usable,
                    final_reason,
                )
                meta = _load_meta(conn, obj_id)
                meta.update(_meta_fail(final_reason, include_scale=True))
                _save_meta(conn, obj_id, meta)
            else:
                accepted += 1
                dist = _euclidean(_center(obj_bbox), _center(best_det["bbox"]))
                log.info(
                    "[bbox-refine] object type=%s label=%s page=%d "
                    "raw=%d usable=%d same_type=%d accepted=True score=%.2f dist=%.1f",
                    obj_type, obj_label, obj_page_1,
                    len(raw_dets), len(usable_dets), same_type_usable,
                    best_det["score"], dist,
                )
                meta = _load_meta(conn, obj_id)
                meta.update(_meta_accept(best_det, dist))
                _save_meta(conn, obj_id, meta)

    log.info(
        "[bbox-refine] summary detector=%s figures=%d tables=%d attempted=%d "
        "accepted=%d rejected=%d unavailable=%d "
        "raw_total=%d oversize=%d too_small=%d "
        "pages_raw=%d pages_usable=%d no_det_pages=%d match_failures=%d",
        detector, n_figures, n_tables, attempted, accepted, rejected, unavailable,
        raw_detections_total, filtered_oversize, filtered_too_small,
        pages_with_raw, pages_with_usable, no_detection_pages, match_failures,
    )

    return {
        **_base,
        "bbox_refine_available": True,
        "bbox_refine_attempted": attempted,
        "bbox_refine_accepted": accepted,
        "bbox_refine_rejected": rejected,
        "bbox_refine_raw_detections": raw_detections_total,
        "bbox_refine_filtered_oversize": filtered_oversize,
        "bbox_refine_filtered_too_small": filtered_too_small,
        "bbox_refine_filtered_invalid": filtered_invalid,
        "bbox_refine_no_detection_pages": no_detection_pages,
        "bbox_refine_pages_with_raw": pages_with_raw,
        "bbox_refine_pages_with_usable": pages_with_usable,
        "bbox_refine_match_failures": match_failures,
        "bbox_refine_compat_failure": int(_compat_failed),
    }


# ---------------------------------------------------------------------------
# Selective entry point — refine only specific evidence object IDs
# ---------------------------------------------------------------------------

def _load_evidence_objects_by_ids(
    conn: sqlite3.Connection,
    ids: list[str],
) -> list[dict[str, Any]]:
    """Load specific evidence objects from DB by ID list (minimal fields for refinement)."""
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    cur = conn.execute(
        f"""
        SELECT id, paper_id, object_type, label, page, page_start, page_end,
               bbox_json, metadata_json
        FROM evidence_objects
        WHERE id IN ({placeholders})
        ORDER BY COALESCE(page, page_start, 99999), id
        """,
        ids,
    )
    out = []
    for r in cur.fetchall():
        out.append({
            "id": r[0], "paper_id": r[1], "object_type": r[2], "label": r[3],
            "page": r[4], "page_start": r[5], "page_end": r[6],
            "bbox": json.loads(r[7]) if r[7] else None,
            "metadata": json.loads(r[8]) if r[8] else {},
        })
    return out


def refine_bbox_for_evidence_ids(
    evidence_object_ids: list[str],
    pdf_path: str,
    conn: sqlite3.Connection,
    *,
    render_scale: float = _DEFAULT_RENDER_SCALE,
    detector: str = "groundingdino",
) -> dict[str, Any]:
    """Selective bbox refinement: process only the given evidence object IDs.

    Designed for post-retrieval use — pass the unique figure/table IDs that
    appeared in top-k retrieval results for a paper.

    Objects already attempted by this detector backend are skipped so the
    same object is not refined repeatedly across claims in the same run.
    Original bbox_json is never overwritten; results land in metadata_json
    (detected_bbox + bbox_source + per-detector audit fields).

    Returns the same summary dict shape as refine_bbox_for_paper.
    """
    # --- Detector dispatch (mirrors refine_bbox_for_paper) ------------------
    if detector == "rtdetr":
        from .rtdetr_detector import (
            rtdetr_available as _detector_available,
            detect_on_page,
            filter_detections,
            _MODEL_ID,
        )
        _backend_name = "rtdetr"
        _meta_prefix = "rtdetr"
        _raw_label_key = "rtdetr_label"
    elif detector == "paddlex_layout":
        from .paddlex_layout_detector import (
            paddlex_layout_available as _detector_available,
            detect_on_page,
            filter_detections,
            _MODEL_ID,
        )
        _backend_name = "paddlex_layout"
        _meta_prefix = "paddlex_layout"
        _raw_label_key = "paddlex_layout_label"
    else:
        from .grounding_dino import (
            grounding_dino_available as _detector_available,
            detect_on_page,
            filter_detections,
            _MODEL_ID,
        )
        _backend_name = "groundingdino"
        _meta_prefix = "dino"
        _raw_label_key = "dino_label"

    _mp = _meta_prefix
    _base = {"bbox_refine_enabled": True, "bbox_refine_detector": detector}

    if not _detector_available():
        log.warning(
            "[bbox-refine-selective] detector=%s unavailable; skipping.", detector,
        )
        return {**_base, "bbox_refine_available": False,
                "bbox_refine_attempted": 0, "bbox_refine_accepted": 0,
                "bbox_refine_rejected": 0}

    # Deduplicate input IDs while preserving first-seen order
    seen: set[str] = set()
    unique_ids: list[str] = []
    for i in evidence_object_ids:
        if i and i not in seen:
            seen.add(i)
            unique_ids.append(i)

    if not unique_ids:
        return {**_base, "bbox_refine_available": True,
                "bbox_refine_attempted": 0, "bbox_refine_accepted": 0,
                "bbox_refine_rejected": 0}

    all_candidates = _load_evidence_objects_by_ids(conn, unique_ids)

    # Keep only figure/table objects not yet attempted by this detector backend
    visual_candidates = [o for o in all_candidates if o["object_type"] in ("figure", "table")]
    targets = [
        o for o in visual_candidates
        if not o.get("metadata", {}).get(f"{_mp}_attempted")
    ]
    skipped_non_visual = len(all_candidates) - len(visual_candidates)
    skipped_already_attempted = len(visual_candidates) - len(targets)

    # Group by 0-based page index so each page is rendered exactly once
    by_page: dict[int, list[dict[str, Any]]] = {}
    for obj in targets:
        pg = (obj.get("page") or obj.get("page_start") or 1) - 1
        by_page.setdefault(pg, []).append(obj)

    distinct_pages = sorted(by_page.keys())
    _t0 = time.monotonic()

    log.info(
        "[bbox-refine-selective] START detector=%s model=%s render_scale=%.1f "
        "input_ids=%d unique=%d candidates=%d "
        "skipped_non_visual=%d skipped_already_attempted=%d targets=%d "
        "distinct_pages=%d pages=%s",
        detector, _MODEL_ID, render_scale,
        len(evidence_object_ids), len(unique_ids), len(all_candidates),
        skipped_non_visual, skipped_already_attempted, len(targets),
        len(distinct_pages), distinct_pages,
    )

    if not targets:
        return {**_base, "bbox_refine_available": True,
                "bbox_refine_attempted": 0, "bbox_refine_accepted": 0,
                "bbox_refine_rejected": 0,
                "bbox_refine_skipped_non_visual": skipped_non_visual,
                "bbox_refine_skipped_already_attempted": skipped_already_attempted}

    now = datetime.now(timezone.utc).isoformat()
    page_area_cache: dict[int, float] = {}

    def _meta_fail(reason: str, *, attempted: bool = True, include_scale: bool = False) -> dict:
        d: dict = {
            f"{_mp}_attempted": attempted,
            f"{_mp}_matched": False,
            f"{_mp}_rejected_reason": reason,
            f"{_mp}_last_updated_at": now,
        }
        if include_scale:
            d[f"{_mp}_render_scale"] = render_scale
        return d

    def _meta_accept(best_det: dict, dist: float) -> dict:
        return {
            "detected_bbox": best_det["bbox"],
            "bbox_source": _backend_name,
            "object_bbox_confidence": "refined",
            f"{_mp}_score": best_det["score"],
            f"{_mp}_label": best_det.get(_raw_label_key, best_det["label"]),
            f"{_mp}_model": best_det.get("model", ""),
            f"{_mp}_match_mode": "same_page_nearest_anchor",
            f"{_mp}_match_distance": round(dist, 2),
            f"{_mp}_attempted": True,
            f"{_mp}_matched": True,
            f"{_mp}_render_scale": render_scale,
            f"{_mp}_last_updated_at": now,
        }

    attempted = accepted = rejected = unavailable = 0
    raw_detections_total = filtered_oversize = filtered_too_small = filtered_invalid = 0
    no_detection_pages = pages_with_raw = pages_with_usable = 0
    match_failures = 0
    _compat_failed = False

    for pg0, page_objs in sorted(by_page.items()):
        if _compat_failed:
            for obj in page_objs:
                meta = _load_meta(conn, obj["id"])
                meta.update(_meta_fail("compat_failure_skipped", attempted=False))
                _save_meta(conn, obj["id"], meta)
                unavailable += 1
            continue

        if pg0 not in page_area_cache:
            page_area_cache[pg0] = _page_area_pdf(pdf_path, pg0)
        page_area = page_area_cache[pg0]

        try:
            raw_dets = detect_on_page(pdf_path, pg0, render_scale=render_scale)
        except RuntimeError as exc:
            err_str = str(exc)
            if "compatibility failure" in err_str or "unexpected keyword" in err_str:
                log.error(
                    "[bbox-refine-selective] Processor compatibility failure page=%d — aborting. %s",
                    pg0 + 1, exc,
                )
                _compat_failed = True
            else:
                log.warning(
                    "[bbox-refine-selective] Detection failed page=%d: %s", pg0 + 1, exc,
                )
            for obj in page_objs:
                meta = _load_meta(conn, obj["id"])
                meta.update(_meta_fail("detection_exception"))
                _save_meta(conn, obj["id"], meta)
                unavailable += 1
            continue
        except Exception as exc:
            log.warning(
                "[bbox-refine-selective] Detection failed page=%d: %s", pg0 + 1, exc,
            )
            for obj in page_objs:
                meta = _load_meta(conn, obj["id"])
                meta.update(_meta_fail("detection_exception"))
                _save_meta(conn, obj["id"], meta)
                unavailable += 1
            continue

        usable_dets, rejected_dets = filter_detections(raw_dets, page_area)
        raw_detections_total += len(raw_dets)
        if len(raw_dets) == 0:
            no_detection_pages += 1
        else:
            pages_with_raw += 1
        if usable_dets:
            pages_with_usable += 1
        for r in rejected_dets:
            reason = r.get("reject_reason", "")
            if reason == "page_coverage_too_large":
                filtered_oversize += 1
            elif reason == "area_too_small":
                filtered_too_small += 1
            elif reason == "invalid_bbox":
                filtered_invalid += 1

        def _page_reject_reason() -> str:
            if not raw_dets:
                return "no_raw_detection"
            if not usable_dets:
                reasons = {r.get("reject_reason") for r in rejected_dets}
                if reasons <= {"page_coverage_too_large"}:
                    return "only_oversize_candidates"
                if reasons <= {"area_too_small"}:
                    return "only_too_small_candidates"
                return "all_candidates_filtered"
            return ""

        page_level_reason = _page_reject_reason()

        for obj in page_objs:
            obj_id = obj["id"]
            obj_type = obj["object_type"]
            obj_label = obj.get("label")
            obj_page_1 = pg0 + 1

            obj_bbox = obj.get("bbox")
            if not obj_bbox or len(obj_bbox) < 4 or _area(obj_bbox) == 0:
                log.debug(
                    "[bbox-refine-selective] id=%s label=%s page=%d no anchor bbox; skipping",
                    obj_id, obj_label, obj_page_1,
                )
                meta = _load_meta(conn, obj_id)
                meta.update(_meta_fail("no_anchor_bbox"))
                _save_meta(conn, obj_id, meta)
                unavailable += 1
                continue

            attempted += 1
            same_type_usable = sum(1 for d in usable_dets if d["label"] == obj_type)
            best_det, match_reason = _best_match(obj_bbox, obj_type, usable_dets)

            if best_det is None:
                rejected += 1
                if page_level_reason:
                    final_reason = page_level_reason
                else:
                    final_reason = match_reason or "candidates_exist_but_no_anchor_match"
                    match_failures += 1
                log.info(
                    "[bbox-refine-selective] type=%s label=%s page=%d "
                    "raw=%d usable=%d same_type=%d accepted=False reason=%s",
                    obj_type, obj_label, obj_page_1,
                    len(raw_dets), len(usable_dets), same_type_usable, final_reason,
                )
                meta = _load_meta(conn, obj_id)
                meta.update(_meta_fail(final_reason, include_scale=True))
                _save_meta(conn, obj_id, meta)
            else:
                accepted += 1
                dist = _euclidean(_center(obj_bbox), _center(best_det["bbox"]))
                log.info(
                    "[bbox-refine-selective] type=%s label=%s page=%d "
                    "raw=%d usable=%d same_type=%d accepted=True score=%.2f dist=%.1f",
                    obj_type, obj_label, obj_page_1,
                    len(raw_dets), len(usable_dets), same_type_usable,
                    best_det["score"], dist,
                )
                meta = _load_meta(conn, obj_id)
                meta.update(_meta_accept(best_det, dist))
                _save_meta(conn, obj_id, meta)

    _elapsed = round(time.monotonic() - _t0, 2)
    log.info(
        "[bbox-refine-selective] DONE detector=%s "
        "refined_object_count=%d skipped_already_attempted=%d skipped_non_visual=%d "
        "pages_processed=%d elapsed_s=%.2f | "
        "attempted=%d accepted=%d rejected=%d unavailable=%d "
        "pages_raw=%d pages_usable=%d no_det_pages=%d match_failures=%d",
        detector,
        accepted, skipped_already_attempted, skipped_non_visual,
        len(distinct_pages), _elapsed,
        attempted, accepted, rejected, unavailable,
        pages_with_raw, pages_with_usable, no_detection_pages, match_failures,
    )

    return {
        **_base,
        "bbox_refine_available": True,
        "bbox_refine_attempted": attempted,
        "bbox_refine_accepted": accepted,
        "bbox_refine_rejected": rejected,
        "bbox_refine_skipped_non_visual": skipped_non_visual,
        "bbox_refine_skipped_already_attempted": skipped_already_attempted,
        "bbox_refine_pages_processed": len(distinct_pages),
        "bbox_refine_elapsed_s": _elapsed,
        "bbox_refine_raw_detections": raw_detections_total,
        "bbox_refine_filtered_oversize": filtered_oversize,
        "bbox_refine_filtered_too_small": filtered_too_small,
        "bbox_refine_filtered_invalid": filtered_invalid,
        "bbox_refine_no_detection_pages": no_detection_pages,
        "bbox_refine_pages_with_raw": pages_with_raw,
        "bbox_refine_pages_with_usable": pages_with_usable,
        "bbox_refine_match_failures": match_failures,
        "bbox_refine_compat_failure": int(_compat_failed),
    }
