"""RT-DETR-based figure/table region detection for PDF pages — Phase P1.6 spike.

Provides the same public interface as grounding_dino.detect_on_page / filter_detections
so that bbox_refinement.py can swap backends for a controlled comparison.

Model choice:
  Default: PekingU/rtdetr_r50vd_coco_o365 (RT-DETR trained on COCO + Objects365)
  Override: GSR_RTDETR_MODEL environment variable

LIMITATION (important for interpreting results):
  This is a general-purpose detector trained on natural-image data (COCO + Objects365).
  It is NOT document-layout specialized. Key implications:
    - "table" / "dining table" classes exist in COCO/O365 → table detection may work.
    - No explicit "figure", "chart", "graph", or "diagram" class exists in COCO/O365.
      Figure detection relies on approximate mappings from visually similar classes
      (e.g. "picture", "painting", "poster"). These will rarely match scientific plots.
  This limitation is expected and is precisely what the P1.6 spike is measuring.
  A document-layout RT-DETR checkpoint (if available) would be configurable via
  GSR_RTDETR_MODEL once identified.

Coordinate convention (same as grounding_dino.py):
  All returned bboxes are in PDF page coordinate space [x0, y0, x1, y1]
  (origin at top-left, y increasing downward), consistent with evidence_objects bbox_json.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "PekingU/rtdetr_r50vd_coco_o365"
_MODEL_ID = os.environ.get("GSR_RTDETR_MODEL", _DEFAULT_MODEL_ID)

_DEFAULT_CONFIDENCE_THRESHOLD = 0.5
_DEFAULT_RENDER_SCALE = 2.0

# Structural quality filters — identical values to GroundingDINO path for fair comparison
_FILTER_MIN_AREA = 300.0
_FILTER_MAX_COVERAGE = 0.85

# Label mapping: RT-DETR/COCO/O365 class name → coarse type.
# COCO+O365 have "table" / "dining table" but no explicit figure/chart/graph classes.
# Figure mapping uses approximate visual analogues — see module docstring for limitation.
_TABLE_LABEL_KEYWORDS = frozenset({"table", "dining table"})
_FIGURE_LABEL_KEYWORDS = frozenset({
    "figure", "chart", "graph", "diagram", "plot",
    "picture", "painting", "drawing", "illustration",
    "poster", "sign", "artwork", "photo", "photograph",
    "screen", "monitor", "tv", "television",
})

# Module-level singletons — loaded once and reused
_processor = None
_model = None
_model_loaded_id: str | None = None
_available: bool | None = None


def _map_label(raw_label: str) -> str | None:
    """Map a model class label to 'figure' | 'table' | None (rejected).

    Uses substring matching to handle label variants across COCO / O365 naming.
    Returns None for classes that don't map to figure or table.
    """
    rl = raw_label.strip().lower()
    # Check table first (more specific substring)
    if rl in _TABLE_LABEL_KEYWORDS or "table" in rl:
        return "table"
    for kw in _FIGURE_LABEL_KEYWORDS:
        if kw in rl:
            return "figure"
    return None


def filter_detections(
    detections: list[dict[str, Any]],
    page_area: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split raw detections into usable and rejected.

    Uses identical filter thresholds to the GroundingDINO path so that
    the comparison between detectors is apples-to-apples.

    Returns:
        ``(usable, rejected)`` — rejected dicts carry a ``reject_reason`` key.

        Possible ``reject_reason`` values:
          ``invalid_bbox`` | ``area_too_small`` | ``page_coverage_too_large``
    """
    usable: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for det in detections:
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


def rtdetr_available() -> bool:
    """Return True if RT-DETR (via transformers) can be imported."""
    global _available
    if _available is not None:
        return _available
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, RTDetrForObjectDetection  # noqa: F401
        _available = True
    except ImportError:
        _available = False
        log.warning(
            "[bbox-refine] RT-DETR unavailable "
            "(missing torch or transformers.RTDetrForObjectDetection); "
            "bbox refinement with rtdetr will be skipped."
        )
    return _available


def _ensure_model(model_id: str = _MODEL_ID):
    global _processor, _model, _model_loaded_id
    if _model is not None and _model_loaded_id == model_id:
        return _processor, _model
    from transformers import AutoImageProcessor, RTDetrForObjectDetection
    log.info("[bbox-refine] Loading RT-DETR model: %s", model_id)
    _processor = AutoImageProcessor.from_pretrained(model_id)
    _model = RTDetrForObjectDetection.from_pretrained(model_id)
    _model_loaded_id = model_id

    id2label = getattr(_model.config, "id2label", {})
    figure_classes = [v for v in id2label.values() if _map_label(v) == "figure"]
    table_classes = [v for v in id2label.values() if _map_label(v) == "table"]
    log.info(
        "[bbox-refine] RT-DETR loaded: model=%s total_classes=%d "
        "mapped_figure=%s mapped_table=%s",
        model_id, len(id2label), figure_classes, table_classes,
    )
    return _processor, _model


def detect_on_page(
    pdf_path: str,
    page_num: int,
    *,
    render_scale: float = _DEFAULT_RENDER_SCALE,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    model_id: str = _MODEL_ID,
) -> list[dict[str, Any]]:
    """Detect figure/table regions on one PDF page using RT-DETR.

    Returns detections in the same schema as grounding_dino.detect_on_page:
      bbox          [x0, y0, x1, y1] in PDF page coordinates
      label         "figure" | "table"
      score         float — detection confidence
      model         str   — model identifier
      rtdetr_label  str   — raw model class label (backend-specific key)
    """
    import torch
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

    processor, model = _ensure_model(model_id)
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[img_h, img_w]])
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=confidence_threshold,
    )[0]

    id2label = model.config.id2label
    detections: list[dict[str, Any]] = []

    for score, label_id, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        raw_label = id2label.get(label_id.item(), str(label_id.item()))
        coarse = _map_label(raw_label)
        if coarse is None:
            continue  # not a figure/table class — skip

        px = box.tolist()  # [x0, y0, x1, y1] in image pixels
        x0 = max(0.0, min(px[0] / render_scale, page_w_pdf))
        y0 = max(0.0, min(px[1] / render_scale, page_h_pdf))
        x1 = max(0.0, min(px[2] / render_scale, page_w_pdf))
        y1 = max(0.0, min(px[3] / render_scale, page_h_pdf))

        detections.append({
            "bbox": [x0, y0, x1, y1],
            "label": coarse,
            "score": float(score),
            "model": model_id,
            "rtdetr_label": raw_label,
        })

    return detections
