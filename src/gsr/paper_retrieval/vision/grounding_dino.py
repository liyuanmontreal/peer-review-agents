"""GroundingDINO-based figure/table region detection for PDF pages — Phase P1.5.

Wraps the HuggingFace transformers GroundingDINO API with lazy loading and
clean fallback when the model or dependency is not available.

Coordinate convention (IMPORTANT):
  All returned bboxes are in PDF page coordinate space [x0, y0, x1, y1]
  (origin at page top-left, y increasing downward), consistent with the
  bbox_json format used throughout evidence_objects.

  Pages are rendered at ``render_scale`` (default 2.0).
  Conversion: pdf_coord = pixel_coord / render_scale.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_MODEL_ID = "IDEA-Research/grounding-dino-base"

# Detection prompt — covers the main figure/table object types we care about.
# GroundingDINO uses period-separated noun phrases.
_TEXT_PROMPT = "figure . table . chart . graph . diagram . plot ."

_DEFAULT_BOX_THRESHOLD = 0.35   # raised from 0.35 — prefer fewer, higher-confidence detections
_DEFAULT_TEXT_THRESHOLD = 0.25
_DEFAULT_RENDER_SCALE = 2.0

# Which model output labels map to each coarse type
_FIGURE_LABELS = frozenset({"figure", "chart", "graph", "diagram", "plot"})
_TABLE_LABELS = frozenset({"table"})

# Detection quality filters applied in filter_detections()
_FILTER_MIN_AREA = 300.0        # minimum PDF-coordinate area; smaller boxes are noise
_FILTER_MAX_COVERAGE = 0.85     # reject detections covering > 85 % of the page area

# Module-level singletons — loaded once and reused
_processor = None
_model = None
_model_loaded_id: str | None = None
_available: bool | None = None

# Detected at first model load; None = not yet checked
_postprocess_param_names: list[str] | None = None


def filter_detections(
    detections: list[dict[str, Any]],
    page_area: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split raw detections into usable and rejected.

    Applies structural quality filters so that callers can distinguish:
      - raw detections present but all oversized
      - raw detections present but all too small
      - raw detections absent entirely

    Args:
        detections: Raw detections from :func:`detect_on_page`.
        page_area:  Page area in PDF coordinate units (width × height).

    Returns:
        ``(usable, rejected)`` — rejected dicts have a ``reject_reason`` key.

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


def grounding_dino_available() -> bool:
    """Return True if GroundingDINO (via transformers) can be imported."""
    global _available
    if _available is not None:
        return _available
    try:
        import torch  # noqa: F401
        from transformers import (  # noqa: F401
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
        )
        _available = True
    except ImportError:
        _available = False
        log.warning(
            "[bbox-refine] GroundingDINO unavailable "
            "(missing torch/transformers); bbox refinement will be skipped."
        )
    return _available


def _ensure_model(model_id: str = _MODEL_ID):
    global _processor, _model, _model_loaded_id, _postprocess_param_names
    if _model is not None and _model_loaded_id == model_id:
        return _processor, _model
    import inspect
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    log.info("[bbox-refine] Loading GroundingDINO model: %s", model_id)
    _processor = AutoProcessor.from_pretrained(model_id)
    _model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    _model_loaded_id = model_id
    # Introspect postprocess params once for compatibility wrapper
    method = getattr(_processor, "post_process_grounded_object_detection", None)
    if method:
        _postprocess_param_names = list(inspect.signature(method).parameters.keys())
    else:
        _postprocess_param_names = []
    log.info(
        "[bbox-refine] GroundingDINO loaded: model=%s processor=%s postprocess_params=%s",
        model_id,
        type(_processor).__name__,
        _postprocess_param_names,
    )
    return _processor, _model


def _postprocess_grounding_outputs(
    processor,
    outputs,
    input_ids,
    *,
    box_threshold: float,
    text_threshold: float,
    target_sizes: list,
) -> list:
    """Compatibility wrapper for processor.post_process_grounded_object_detection.

    Introspects the method signature and only passes supported keyword args,
    handling API differences across transformers versions:
      - older: box_threshold=, text_threshold=
      - newer (installed): threshold=, text_threshold=

    Raises RuntimeError with a descriptive message if the call still fails.
    """
    method = getattr(processor, "post_process_grounded_object_detection", None)
    if method is None:
        raise RuntimeError(
            f"Processor {type(processor).__name__} has no "
            "post_process_grounded_object_detection method."
        )

    import inspect
    supported = list(inspect.signature(method).parameters.keys())

    kwargs: dict = {"target_sizes": target_sizes}
    # Map threshold kwarg to whatever the installed version accepts
    if "threshold" in supported:
        kwargs["threshold"] = box_threshold
    elif "box_threshold" in supported:
        kwargs["box_threshold"] = box_threshold

    if "text_threshold" in supported:
        kwargs["text_threshold"] = text_threshold

    try:
        return method(outputs, input_ids, **kwargs)
    except TypeError as exc:
        raise RuntimeError(
            f"[bbox-refine] processor compatibility failure: {exc}; "
            f"processor={type(processor).__name__} "
            f"supported={supported} attempted_kwargs={list(kwargs.keys())}"
        ) from exc


def detect_on_page(
    pdf_path: str,
    page_num: int,
    *,
    render_scale: float = _DEFAULT_RENDER_SCALE,
    box_threshold: float = _DEFAULT_BOX_THRESHOLD,
    text_threshold: float = _DEFAULT_TEXT_THRESHOLD,
    text_prompt: str = _TEXT_PROMPT,
    model_id: str = _MODEL_ID,
) -> list[dict[str, Any]]:
    """Detect figure/table regions on one PDF page using GroundingDINO.

    Args:
        pdf_path:      Path to the PDF file.
        page_num:      0-based page index.
        render_scale:  Render the page at this scale factor (default 2.0).
        box_threshold: GroundingDINO box confidence threshold.
        text_threshold: GroundingDINO text threshold.
        text_prompt:   Detection prompt string.
        model_id:      HuggingFace model identifier.

    Returns:
        List of detection dicts. Each dict contains:
          bbox        [x0, y0, x1, y1] in PDF page coordinates
          label       "figure" | "table"
          score       float — detection confidence
          model       str   — model identifier
          dino_label  str   — raw label from model
    """
    import torch
    import fitz  # PyMuPDF — already a project dependency
    from PIL import Image

    # --- Render page ---
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    page_w_pdf = page.rect.width
    page_h_pdf = page.rect.height
    mat = fitz.Matrix(render_scale, render_scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_w, img_h = pix.width, pix.height
    doc.close()

    img = Image.frombytes("RGB", [img_w, img_h], pix.samples)

    # --- Run GroundingDINO ---
    processor, model = _ensure_model(model_id)
    inputs = processor(images=img, text=text_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    all_results = _postprocess_grounding_outputs(
        processor,
        outputs,
        inputs["input_ids"],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(img_h, img_w)],
    )
    results = all_results[0]

    # --- Parse + convert coordinates ---
    detections: list[dict[str, Any]] = []
    for box, score, label in zip(
        results["boxes"], results["scores"], results["labels"]
    ):
        raw_label = label.strip().lower()

        if raw_label in _TABLE_LABELS:
            coarse = "table"
        elif raw_label in _FIGURE_LABELS:
            coarse = "figure"
        else:
            continue  # unknown label — skip

        px = box.tolist()  # [x0, y0, x1, y1] in image pixels

        # Convert image pixels → PDF coordinates
        x0 = max(0.0, min(px[0] / render_scale, page_w_pdf))
        y0 = max(0.0, min(px[1] / render_scale, page_h_pdf))
        x1 = max(0.0, min(px[2] / render_scale, page_w_pdf))
        y1 = max(0.0, min(px[3] / render_scale, page_h_pdf))

        detections.append({
            "bbox": [x0, y0, x1, y1],
            "label": coarse,
            "score": float(score),
            "model": model_id,
            "dino_label": raw_label,
        })

    return detections
