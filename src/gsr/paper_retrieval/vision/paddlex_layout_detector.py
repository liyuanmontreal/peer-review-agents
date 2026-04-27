"""PaddleX layout detection backend for GSR bbox refinement (experimental).

Provides the same public interface as grounding_dino / rtdetr_detector so that
bbox_refinement.py can dispatch to it with minimal changes.

Model choice:
  Default: PicoDet-L_layout_17cls  (document-layout specialist, 17 classes)
  Override: GSR_PADDLEX_LAYOUT_MODEL environment variable
  Future:   RT-DETR-H_layout_17cls can be set via env var without code changes.

This backend is bbox refinement ONLY:
  - Does NOT create evidence objects.
  - Does NOT change identity / retrieval / verification semantics.
  - Stores results in metadata_json only (detected_bbox, bbox_source, paddlex_layout_*).

Coordinate convention (same as grounding_dino.py and rtdetr_detector.py):
  All returned bboxes are in PDF page coordinate space [x0, y0, x1, y1]
  (origin at top-left, y increasing downward), consistent with evidence_objects bbox_json.

PaddleX dependency:
  pip install paddlex
  If paddlex is not installed, paddlex_layout_available() returns False and
  all operations degrade gracefully with a clear log message.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "PicoDet-L_layout_17cls"
_MODEL_NAME = os.environ.get("GSR_PADDLEX_LAYOUT_MODEL", _DEFAULT_MODEL_NAME)

# Alias for compatibility with bbox_refinement.py / bbox-smoke (which imports _MODEL_ID)
_MODEL_ID = _MODEL_NAME

_DEFAULT_CONFIDENCE_THRESHOLD = 0.5
_DEFAULT_RENDER_SCALE = 2.0

# Structural quality filters — identical to GroundingDINO / RT-DETR for fair comparison
_FILTER_MIN_AREA = 300.0
_FILTER_MAX_COVERAGE = 0.85


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------
#
# PaddleX document-layout models expose 17 classes.  We normalize them to the
# three coarse types our pipeline understands:
#   "figure"  — image regions that should be overlaid in the PDF
#   "table"   — table regions
#   "caption" — caption annotations (debug / hint only, NOT usable candidates)
#   None      — everything else (text, title, header, footer …) → ignored
#
# All matching is case-insensitive substring matching so it is robust to minor
# label wording differences across model variants.

# Fragments that make a label a hard ignore (checked first)
_IGNORE_FRAGMENTS: frozenset[str] = frozenset({
    "header",
    "footer",
    "footnote",
    "page number",
    "pagenumber",
    "page-number",
    "section header",
    "section-header",
    "text",
    "title",
    "reference",
    "equation",
    "formula",
    "list",
    "index",
    "abstract",
    "toc",
    "contents",
})

_FIGURE_FRAGMENTS: frozenset[str] = frozenset({
    "figure", "image", "picture", "photo",
    "illustration", "chart", "graph", "diagram", "plot",
})

_TABLE_FRAGMENTS: frozenset[str] = frozenset({"table"})

_CAPTION_FRAGMENTS: frozenset[str] = frozenset({"caption"})


def _normalize_label(raw_label: str) -> str | None:
    """Map a PaddleX layout class label to 'figure' | 'table' | 'caption' | None.

    Returns None for classes that should be skipped entirely (text, headers, etc.).
    Returns 'caption' for caption annotations — these are surfaced in detections
    for debug / smoke output but are excluded from usable candidates by
    filter_detections().
    """
    rl = raw_label.strip().lower()

    # Hard-ignore list checked first
    for frag in _IGNORE_FRAGMENTS:
        if frag in rl:
            return None

    # "Figure Note" / "Table Note" — notes are not usable candidates
    if "note" in rl:
        return None

    # Table (checked before figure to avoid "table caption" matching figure)
    if any(frag in rl for frag in _TABLE_FRAGMENTS):
        if any(frag in rl for frag in _CAPTION_FRAGMENTS):
            return "caption"
        return "table"

    # Figure / image / chart / diagram
    if any(frag in rl for frag in _FIGURE_FRAGMENTS):
        if any(frag in rl for frag in _CAPTION_FRAGMENTS):
            return "caption"
        return "figure"

    # Pure caption label (e.g. "Caption" without table/figure qualifier)
    if any(frag in rl for frag in _CAPTION_FRAGMENTS):
        return "caption"

    return None


# ---------------------------------------------------------------------------
# Module-level singleton (lazy load)
# ---------------------------------------------------------------------------

_model = None
_model_loaded_name: str | None = None
_available: bool | None = None


def paddlex_layout_available() -> bool:
    """Return True if PaddleX can be imported.

    Safe to call multiple times — result is cached after the first check.
    """
    global _available
    if _available is not None:
        return _available
    try:
        import paddlex  # noqa: F401
        _available = True
    except ImportError:
        _available = False
        log.warning(
            "[bbox-refine] PaddleX unavailable (paddlex not installed); "
            "bbox refinement with paddlex_layout will be skipped. "
            "Install with: pip install paddlex"
        )
    return _available


def _ensure_model(model_name: str = _MODEL_NAME):
    """Load (or return cached) PaddleX model."""
    global _model, _model_loaded_name
    if _model is not None and _model_loaded_name == model_name:
        return _model
    from paddlex import create_model  # type: ignore[import]
    log.info("[bbox-refine] Loading PaddleX layout model: %s", model_name)
    _model = create_model(model_name)
    _model_loaded_name = model_name
    log.info("[bbox-refine] PaddleX layout model loaded: %s", model_name)
    return _model


# ---------------------------------------------------------------------------
# Detection output helpers
# ---------------------------------------------------------------------------

def _extract_boxes_from_result(result: Any) -> list[dict[str, Any]]:
    """Robustly extract the raw box list from a PaddleX prediction result.

    PaddleX 3.x results expose boxes via multiple access paths depending on
    the installed version.  We try them all so the code works across minor API
    variants.
    """
    # Preferred: direct attribute
    if hasattr(result, "boxes"):
        boxes = result.boxes
        if boxes is not None:
            return list(boxes)

    # json_fields dict (PaddleX >= 3.0 common path)
    if hasattr(result, "json_fields"):
        try:
            jf = result.json_fields
            return list(jf.get("boxes", []))
        except Exception:
            pass

    # to_dict() conversion
    if hasattr(result, "to_dict"):
        try:
            d = result.to_dict()
            return list(d.get("boxes", []))
        except Exception:
            pass

    # Last resort: dict-like iteration
    try:
        d = dict(result)
        return list(d.get("boxes", []))
    except Exception:
        pass

    return []


def _box_fields(box: Any) -> tuple[list[float], str, float]:
    """Return (coordinate, label, score) from a box dict or object."""
    if isinstance(box, dict):
        coord = box.get("coordinate", [])
        label = str(box.get("label", ""))
        score = float(box.get("score", 0.0))
    else:
        coord = list(getattr(box, "coordinate", []))
        label = str(getattr(box, "label", ""))
        score = float(getattr(box, "score", 0.0))
    return coord, label, score


# ---------------------------------------------------------------------------
# Public interface (mirrors grounding_dino.py / rtdetr_detector.py)
# ---------------------------------------------------------------------------

def filter_detections(
    detections: list[dict[str, Any]],
    page_area: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split raw detections into usable and rejected.

    Uses identical area / coverage thresholds to the GroundingDINO and
    RT-DETR paths for an apples-to-apples comparison.

    Caption detections are placed in rejected with reject_reason='caption_hint_only'
    — they are surfaced in smoke-test output as debug hints but are NOT used for
    bbox refinement.

    Returns:
        (usable, rejected)

        Possible reject_reason values:
          caption_hint_only | non_candidate_label |
          invalid_bbox | area_too_small | page_coverage_too_large
    """
    usable: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for det in detections:
        norm_label = det.get("label")

        if norm_label == "caption":
            rejected.append({**det, "reject_reason": "caption_hint_only"})
            continue

        if norm_label not in ("figure", "table"):
            rejected.append({**det, "reject_reason": "non_candidate_label"})
            continue

        bbox = det.get("bbox", [])
        if not bbox or len(bbox) < 4:
            rejected.append({**det, "reject_reason": "invalid_bbox"})
            continue

        x0, y0, x1, y1 = bbox
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)

        if area < _FILTER_MIN_AREA:
            rejected.append({**det, "reject_reason": "area_too_small"})
            continue

        if page_area > 0 and (area / page_area) > _FILTER_MAX_COVERAGE:
            rejected.append({**det, "reject_reason": "page_coverage_too_large"})
            continue

        usable.append(det)

    return usable, rejected


def detect_on_page(
    pdf_path: str,
    page_num: int,
    *,
    render_scale: float = _DEFAULT_RENDER_SCALE,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    model_name: str = _MODEL_NAME,
) -> list[dict[str, Any]]:
    """Detect figure/table regions on one PDF page using PaddleX layout detection.

    Returns detections in the same schema as grounding_dino.detect_on_page /
    rtdetr_detector.detect_on_page — compatible with bbox_refinement.py:

      bbox                    [x0, y0, x1, y1] in PDF page coordinates
      label                   "figure" | "table" | "caption"
      score                   float — detection confidence
      model                   str   — model name
      paddlex_layout_label    str   — raw model class label (backend-specific key)

    Caption detections (label="caption") are included so they are visible in
    the smoke test output as debug hints.  filter_detections() excludes them
    from the usable set.

    Classes that normalize to None (text, title, header, footer …) are
    silently discarded.
    """
    import numpy as np
    import fitz
    from PIL import Image

    doc = fitz.open(pdf_path)
    page = doc[page_num]
    page_w_pdf = page.rect.width
    page_h_pdf = page.rect.height
    mat = fitz.Matrix(render_scale, render_scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_w, img_h = pix.width, pix.height
    doc.close()

    img = Image.frombytes("RGB", [img_w, img_h], pix.samples)
    img_array = np.array(img)

    model = _ensure_model(model_name)

    # predict() returns a generator; materialise it once
    results = list(model.predict(img_array))

    detections: list[dict[str, Any]] = []

    for result in results:
        for box in _extract_boxes_from_result(result):
            coord, raw_label, score = _box_fields(box)

            if score < confidence_threshold:
                continue
            if not coord or len(coord) < 4:
                continue

            norm_label = _normalize_label(raw_label)
            if norm_label is None:
                continue  # text / title / header / footer — not a usable class

            # PaddleX coordinate format: [x_left, y_top, x_right, y_bottom] in image pixels.
            # Explicitly cast to native float so downstream JSON serialization never sees
            # numpy scalar types (np.float32 is not JSON-serializable).
            x0_px, y0_px, x1_px, y1_px = coord[0], coord[1], coord[2], coord[3]
            x0 = float(max(0.0, min(float(x0_px) / render_scale, page_w_pdf)))
            y0 = float(max(0.0, min(float(y0_px) / render_scale, page_h_pdf)))
            x1 = float(max(0.0, min(float(x1_px) / render_scale, page_w_pdf)))
            y1 = float(max(0.0, min(float(y1_px) / render_scale, page_h_pdf)))

            detections.append({
                "bbox": [x0, y0, x1, y1],
                "label": norm_label,
                "score": float(score),
                "model": model_name,
                "paddlex_layout_label": raw_label,
            })

    return detections
