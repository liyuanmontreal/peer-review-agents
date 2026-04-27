"""Qwen2.5-VL selective figure semantic enrichment — Phase P2.

Provides two public helpers used by verify_all_claims() when
``selective_figure_semantic=True``:

- ``make_figsem_config_hash()``  — stable string for cache validity
- ``ensure_figure_semantics_for_evidence_object()`` — cache check + VLM + persist

Results are stored in ``evidence_objects.metadata_json`` under ``figure_semantics_*``
keys.  Evidence identity, retrieval ranking, and embeddings are not changed.

Cache design
------------
Semantic results are persisted under a ``figure_semantics_config_hash`` key so the
cache is invalidated when model or render parameters change.  Terminal negative
outcomes (render_failed, dependency_missing, etc.) are also cached to prevent
repeated failed calls within the same config.

Model selection
---------------
Default: ``Qwen/Qwen2.5-VL-7B-Instruct``.
Override via ``GSR_FIGSEM_MODEL`` environment variable (e.g. to use the 3B variant).

Runtime dependencies
--------------------
    PyMuPDF (fitz)               — pip install pymupdf
    torch                        — pip install torch
    transformers >= 4.49         — pip install --upgrade transformers
    Pillow                       — pip install Pillow

All are optional; if unavailable the function returns a graceful failure.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

_RENDER_SCALE = 2.0          # 2× → ~144 DPI for a typical 72-DPI PDF
_MIN_BBOX_DIM = 4.0          # minimum width/height in PDF points to attempt crop
_MAX_CROP_LONGEST_SIDE = 1120  # pixels; larger crops increase inference latency
_DEFAULT_MAX_NEW_TOKENS: int = 512
_DEFAULT_TIMEOUT_SECONDS: float = 180.0

# Module-level model cache (lazy-loaded once per process)
_model_cache: dict[str, Any] = {}

# Terminal negative outcomes that should not be retried when config hash matches
_TERMINAL_NEGATIVE_OUTCOMES = frozenset({
    "skipped_no_bbox",
    "skipped_small_crop",
    "render_failed",
    "dependency_missing",
    "failed",
    "unavailable",
    "timeout",
    "inference_failed",
})

_ALLOWED_FIGURE_TYPES = frozenset({
    "bar_chart",
    "line_chart",
    "scatter_plot",
    "heatmap",
    "qualitative_examples",
    "architecture_diagram",
    "workflow_diagram",
    "table_like_visual",
    "other",
})


# ---------------------------------------------------------------------------
# Config hash
# ---------------------------------------------------------------------------

def make_figsem_config_hash(
    *,
    model_id: str,
    render_scale: float,
    max_new_tokens: int,
) -> str:
    """Return a stable string used as the figure-semantics cache validity key.

    Changing any of these parameters invalidates the cache.
    """
    slug = model_id.split("/")[-1].lower().replace("-", "").replace(".", "")
    return (
        f"qwen25vl_figsem_v1"
        f"_{slug}"
        f"_scale{render_scale:.1f}"
        f"_tok{max_new_tokens}"
    )


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def qwen25vl_available() -> bool:
    """Return True if all dependencies for Qwen2.5-VL inference are installed."""
    try:
        import fitz  # noqa: F401
    except ImportError:
        return False
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    try:
        import transformers  # noqa: F401
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
        return True
    except (ImportError, AttributeError):
        return False


def _get_model_id() -> str:
    return os.environ.get("GSR_FIGSEM_MODEL", _DEFAULT_MODEL_ID)


# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------

def _load_qwen_model() -> tuple[Any, Any, str]:
    """Load (or return cached) Qwen2.5-VL model, processor, and device string.

    Model is loaded once per process and cached in _model_cache.
    Device selection: CUDA > MPS > CPU.
    """
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["processor"], _model_cache["device"]

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model_id = _get_model_id()
    log.info("[figure-sem] loading model=%s", model_id)

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)

    _model_cache.update({
        "model": model,
        "processor": processor,
        "device": device,
        "dtype": dtype,
        "model_id": model_id,
    })

    log.info("[figure-sem] model loaded device=%s", device)
    return model, processor, device


# ---------------------------------------------------------------------------
# Crop selection
# ---------------------------------------------------------------------------

def _pick_figure_bbox(
    metadata: dict[str, Any],
    evidence_item: dict[str, Any],
) -> tuple[list[float] | None, str]:
    """Select the best available bbox for a figure crop.

    Priority order:
    1. detected_bbox in metadata_json (from GroundingDINO if present)
    2. bbox from the evidence_item (canonical figure bbox from evidence_objects)
    3. inferred_bbox in metadata_json
    4. caption bbox with conservative upward expansion (caption_fallback)

    Returns (bbox_or_None, source_label).
    """
    # 1. GroundingDINO detected bbox
    detected = metadata.get("detected_bbox")
    if detected and len(detected) == 4:
        return list(float(v) for v in detected), "detected_bbox"

    # 2. Evidence item bbox (canonical figure bbox stored on the object)
    ev_bbox = evidence_item.get("bbox")
    if ev_bbox and len(ev_bbox) == 4:
        return list(float(v) for v in ev_bbox), "object_bbox"

    # 3. inferred_bbox in metadata
    for key in ("inferred_bbox", "figure_inferred_bbox"):
        inferred = metadata.get(key)
        if inferred and len(inferred) == 4:
            return list(float(v) for v in inferred), "inferred_bbox"

    # 4. Caption bbox — expand upward conservatively (figure is above caption)
    cap_bbox = metadata.get("caption_bbox")
    if cap_bbox and len(cap_bbox) == 4:
        x0, y0, x1, y1 = (float(v) for v in cap_bbox)
        height = y1 - y0
        expansion = max(height * 8.0, 100.0)
        return [x0, max(0.0, y0 - expansion), x1, y1], "caption_fallback"

    return None, "none"


# ---------------------------------------------------------------------------
# PDF crop rendering
# ---------------------------------------------------------------------------

def _render_figure_crop(
    pdf_path: str | Path,
    page: int,
    bbox: list[float],
    scale: float = _RENDER_SCALE,
) -> bytes | None:
    """Render a figure region from a PDF page and return PNG bytes.

    Returns None on failure (missing dep, bad bbox, render error).
    """
    try:
        import fitz
    except ImportError:
        return None

    x0, y0, x1, y1 = bbox
    if (x1 - x0) < _MIN_BBOX_DIM or (y1 - y0) < _MIN_BBOX_DIM:
        return None

    try:
        doc = fitz.open(str(pdf_path))
        fitz_page = doc[page - 1]  # fitz is 0-indexed
        clip = fitz.Rect(x0, y0, x1, y1)
        mat = fitz.Matrix(scale, scale)
        pix = fitz_page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
        img_bytes: bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception as exc:
        log.warning("[figure-sem] crop render failed page=%s bbox=%s: %s", page, bbox, exc)
        return None


# ---------------------------------------------------------------------------
# Qwen2.5-VL prompt
# ---------------------------------------------------------------------------

_FIGSEM_PROMPT = """\
You are analyzing a figure from a machine learning research paper.
Identify the following and respond in JSON only. Be concise and conservative.
Do not hallucinate data that is not clearly visible. If uncertain, say "uncertain".

Respond with exactly this JSON schema — no extra text, no markdown fences:
{
  "figure_type": "<one of: bar_chart, line_chart, scatter_plot, heatmap, qualitative_examples, architecture_diagram, workflow_diagram, table_like_visual, other>",
  "figure_summary": "<1-2 sentences describing what the figure shows>",
  "key_observations": ["<observation 1>", "<observation 2>"],
  "possible_verification_use": ["<how this figure could support or refute a paper claim>"],
  "needs_precise_ocr": <true or false>
}

Rules:
- figure_type: pick the single best match.
- figure_summary: describe only what is clearly visible. Do not infer beyond the figure.
- key_observations: up to 3 short observations. Only include if clearly evidenced by the figure.
- possible_verification_use: 1-2 items about what factual claims this figure could verify.
- needs_precise_ocr: true only if the figure contains important embedded text or numbers that are hard to read at this resolution.
"""


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def _parse_figsem_response(raw_text: str) -> dict[str, Any]:
    """Parse the JSON response from Qwen2.5-VL into a validated dict."""
    raw = raw_text.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    data: dict[str, Any] = {}
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                pass

    figure_type = str(data.get("figure_type") or "other").lower().strip()
    if figure_type not in _ALLOWED_FIGURE_TYPES:
        figure_type = "other"

    summary = str(data.get("figure_summary") or "").strip()[:400]

    key_obs_raw = data.get("key_observations") or []
    key_obs = (
        [str(x).strip()[:200] for x in key_obs_raw[:3] if x]
        if isinstance(key_obs_raw, list)
        else []
    )

    poss_use_raw = data.get("possible_verification_use") or []
    poss_use = (
        [str(x).strip()[:200] for x in poss_use_raw[:2] if x]
        if isinstance(poss_use_raw, list)
        else []
    )

    needs_ocr_raw = data.get("needs_precise_ocr")
    if isinstance(needs_ocr_raw, bool):
        needs_ocr = needs_ocr_raw
    else:
        needs_ocr = str(needs_ocr_raw).lower().strip() in ("true", "yes", "1")

    return {
        "figure_type": figure_type,
        "figure_summary": summary,
        "key_observations": key_obs,
        "possible_verification_use": poss_use,
        "needs_precise_ocr": needs_ocr,
    }


def _call_qwen25vl(
    img_bytes: bytes,
    *,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Call Qwen2.5-VL on PNG image bytes and return the parsed semantic dict.

    Wraps inference in a best-effort timeout via ThreadPoolExecutor.
    Raises TimeoutError or re-raises inference exceptions to the caller.
    """
    import concurrent.futures

    def _run() -> dict[str, Any]:
        from PIL import Image as PILImage
        import torch

        model, processor, device = _load_qwen_model()

        pil_image = PILImage.open(BytesIO(img_bytes)).convert("RGB")

        # Resize if crop exceeds longest-side cap
        longest = max(pil_image.size)
        if longest > _MAX_CROP_LONGEST_SIDE:
            factor = _MAX_CROP_LONGEST_SIDE / longest
            new_w = max(1, int(pil_image.width * factor))
            new_h = max(1, int(pil_image.height * factor))
            pil_image = pil_image.resize((new_w, new_h), PILImage.LANCZOS)

        # Build conversation with a single image + instruction
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": _FIGSEM_PROMPT},
                ],
            }
        ]
        text_prompt: str = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text_prompt],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        )

        dtype = _model_cache.get("dtype", torch.float32)
        inputs = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }

        log.debug("[figure-sem] generate starting max_new_tokens=%d", max_new_tokens)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        log.debug("[figure-sem] generate done")

        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        raw_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
        return _parse_figsem_response(raw_text)

    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = _executor.submit(_run)
    _executor.shutdown(wait=False)
    try:
        return fut.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(
            f"Qwen2.5-VL inference exceeded {timeout_seconds:.0f}s budget"
        )


# ---------------------------------------------------------------------------
# Metadata persistence helper
# ---------------------------------------------------------------------------

def _persist_figsem_metadata(
    conn: sqlite3.Connection,
    evidence_object_id: str,
    metadata: dict[str, Any],
    *,
    status: str,
    config_hash: str,
    bbox_source: str = "none",
    model_id: str = "",
    elapsed_s: float | None = None,
    semantics: dict[str, Any] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    """Persist figure-semantics fields into evidence_objects.metadata_json."""
    now = datetime.now(timezone.utc).isoformat()
    updates: dict[str, Any] = {
        "figure_semantics_model": model_id,
        "figure_semantics_config_hash": config_hash,
        "figure_semantics_cached": True,
        "figure_semantics_last_updated_at": now,
        "figure_semantics_status": status,
        "figure_semantics_bbox_source": bbox_source,
        "figure_semantics_render_scale": _RENDER_SCALE,
    }
    if elapsed_s is not None:
        updates["figure_semantics_elapsed_s"] = round(elapsed_s, 2)
    if error_type:
        updates["figure_semantics_error_type"] = error_type
    if error_message:
        updates["figure_semantics_error_message"] = error_message
    if semantics:
        # Flatten semantic fields directly into metadata
        updates.update(semantics)

    metadata.update(updates)
    with conn:
        conn.execute(
            "UPDATE evidence_objects SET metadata_json = ? WHERE id = ?",
            (json.dumps(metadata, ensure_ascii=False), evidence_object_id),
        )


def _extract_semantics_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Pull figure_semantics fields out of a cached metadata dict."""
    return {
        "figure_type": metadata.get("figure_type"),
        "figure_summary": metadata.get("figure_summary"),
        "key_observations": metadata.get("key_observations") or [],
        "possible_verification_use": metadata.get("possible_verification_use") or [],
        "needs_precise_ocr": metadata.get("needs_precise_ocr"),
    }


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

def ensure_figure_semantics_for_evidence_object(
    conn: sqlite3.Connection,
    evidence_item: dict[str, Any],
    pdf_path: str,
    *,
    config_hash: str,
) -> tuple[dict[str, Any], str]:
    """Ensure Qwen2.5-VL semantic enrichment is available for a figure evidence object.

    Steps:
    1. Load current metadata from ``evidence_objects``.
    2. If ``figure_semantics_config_hash`` matches and status is ``success``:
       return the cached semantics (cache_hit_positive).
    3. If config hash matches and status is a terminal negative: skip
       re-inference (cache_hit_negative).
    4. Otherwise: run Qwen2.5-VL, persist result, and return enriched item.

    Returns:
        (updated_evidence_item, cache_status) where cache_status is one of:
        - ``"cache_hit_positive"`` — cached semantics reused
        - ``"cache_hit_negative"`` — cached terminal negative, skip
        - ``"fresh_success"``      — fresh inference succeeded
        - ``"fresh_failed"``       — fresh inference failed (non-fatal)
        - ``"error"``              — unexpected problem (missing ID / DB row)

    Side effects:
        On fresh inference: updates ``evidence_objects.metadata_json``.
        Does NOT update ``evidence_embeddings`` — retrieval is unchanged.
    """
    evidence_object_id = evidence_item.get("evidence_object_id")
    if not evidence_object_id:
        log.warning("[figure-sem] evidence_item missing evidence_object_id")
        return evidence_item, "error"

    # --- Load current DB state ---
    row = conn.execute(
        "SELECT metadata_json, bbox_json FROM evidence_objects WHERE id = ?",
        (evidence_object_id,),
    ).fetchone()
    if not row:
        log.warning("[figure-sem] evidence_object id=%s not in DB", evidence_object_id)
        return evidence_item, "error"

    try:
        metadata: dict[str, Any] = json.loads(row[0]) if row[0] else {}
    except Exception:
        metadata = {}

    try:
        db_bbox = json.loads(row[1]) if row[1] else None
    except Exception:
        db_bbox = None

    # Merge bbox from DB if evidence_item doesn't have it
    ev = dict(evidence_item)
    if not ev.get("bbox") and db_bbox:
        ev["bbox"] = db_bbox

    label = ev.get("label") or "?"
    page = ev.get("page") or ev.get("page_start") or 1

    # --- Cache check ---
    cached_hash = metadata.get("figure_semantics_config_hash")
    cached_status = metadata.get("figure_semantics_status")

    if cached_hash == config_hash and cached_status:
        if cached_status == "success":
            log.info(
                "[figure-sem] cache_hit_positive figure=%s id=%s",
                label, evidence_object_id,
            )
            ev["figure_semantics"] = _extract_semantics_from_metadata(metadata)
            return ev, "cache_hit_positive"

        if cached_status in _TERMINAL_NEGATIVE_OUTCOMES:
            log.info(
                "[figure-sem] cache_hit_negative figure=%s id=%s status=%s",
                label, evidence_object_id, cached_status,
            )
            return ev, "cache_hit_negative"

    # --- Availability ---
    if not qwen25vl_available():
        log.warning("[figure-sem] Qwen2.5-VL unavailable figure=%s", label)
        _persist_figsem_metadata(
            conn, evidence_object_id, metadata,
            status="unavailable",
            config_hash=config_hash,
            error_type="dependency_missing",
            error_message="Qwen2.5-VL or dependencies not installed",
        )
        return ev, "fresh_failed"

    # --- Bbox selection ---
    bbox, bbox_source = _pick_figure_bbox(metadata, ev)

    log.info(
        "[figure-sem] figure=%s page=%s id=%s bbox_source=%s config=%s",
        label, page, evidence_object_id, bbox_source, config_hash,
    )

    if bbox is None:
        log.info("[figure-sem] figure=%s no_bbox skipping", label)
        _persist_figsem_metadata(
            conn, evidence_object_id, metadata,
            status="skipped_no_bbox",
            config_hash=config_hash,
            bbox_source=bbox_source,
        )
        return ev, "fresh_failed"

    # --- Render crop ---
    img_bytes = _render_figure_crop(pdf_path, page, bbox, scale=_RENDER_SCALE)
    if img_bytes is None:
        log.info("[figure-sem] figure=%s render_failed", label)
        _persist_figsem_metadata(
            conn, evidence_object_id, metadata,
            status="render_failed",
            config_hash=config_hash,
            bbox_source=bbox_source,
        )
        return ev, "fresh_failed"

    # --- Inference ---
    model_id = _get_model_id()
    t0 = time.monotonic()
    try:
        sem: dict[str, Any] = _call_qwen25vl(img_bytes)
        elapsed = time.monotonic() - t0
        log.info(
            "[figure-sem] figure=%s status=success type=%s needs_ocr=%s elapsed=%.1fs",
            label, sem.get("figure_type"), sem.get("needs_precise_ocr"), elapsed,
        )
        _persist_figsem_metadata(
            conn, evidence_object_id, metadata,
            status="success",
            config_hash=config_hash,
            bbox_source=bbox_source,
            model_id=model_id,
            elapsed_s=elapsed,
            semantics=sem,
        )
        ev["figure_semantics"] = sem
        return ev, "fresh_success"

    except Exception as exc:
        elapsed = time.monotonic() - t0
        error_type = type(exc).__name__
        error_msg = str(exc)[:400]
        log.warning(
            "[figure-sem] figure=%s inference failed [%s]: %s",
            label, error_type, error_msg,
        )
        _persist_figsem_metadata(
            conn, evidence_object_id, metadata,
            status="failed",
            config_hash=config_hash,
            bbox_source=bbox_source,
            model_id=model_id,
            elapsed_s=elapsed,
            error_type=error_type,
            error_message=error_msg,
        )
        return ev, "fresh_failed"
