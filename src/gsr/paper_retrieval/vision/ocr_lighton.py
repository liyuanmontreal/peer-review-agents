"""LightOnOCR-2 figure text recovery for GSR — Phase 1.1 / 1.2.

Renders a figure region from the source PDF (via PyMuPDF), runs LightOnOCR-2
via Hugging Face transformers local inference, then applies text cleaning and
quality filtering before returning.

Runtime dependencies
--------------------
    PyMuPDF (fitz)      — pip install pymupdf
    torch               — pip install torch
    transformers >= 5.0 — pip install transformers
    Pillow              — pip install Pillow

The model is loaded lazily on first use and cached for the process lifetime.
Model: lightonai/LightOnOCR-2-1B

Integration rules:
- Only called for canonical figure evidence objects (caption-defined identity).
- Only runs when bbox_confidence is "high" or "inferred" — "caption_only" objects
  have no reliable crop region.
- All public output uses the figure_ocr_* schema; generic names like "ocr_text"
  are intentionally avoided so future table OCR or semantic extraction can coexist.

Output schema returned by ocr_figure_region():
    figure_ocr_text        — cleaned, filtered OCR text (empty string if rejected)
    figure_ocr_model       — model identifier string
    figure_ocr_quality     — confidence/quality score (float or None)
    figure_ocr_attempted   — True if inference was attempted
    figure_ocr_skip_reason — structured skip reason code (see taxonomy below)
    figure_ocr_source      — fixed "lightonocr2"

Skip reason taxonomy (figure_ocr_skip_reason)
----------------------------------------------
Never attempted (figure_ocr_attempted = False):
    caption_only        — bbox_confidence is caption_only; no reliable crop region
    no_bbox             — evidence object has no bbox
    bbox_too_small      — bbox dimensions below minimum threshold
    dependency_missing  — PyMuPDF, torch, or transformers not installed
    render_failed       — PDF crop rendering raised an error
    disabled            — use_figure_ocr=False at index time

Rejected after inference (figure_ocr_attempted = True):
    inference_failed    — OCR model raised an exception during inference
    empty               — cleaned text is empty
    too_short           — cleaned text below minimum character threshold
    low_quality         — model-reported confidence below threshold
    noisy_text          — printable character ratio below threshold
    caption_duplicate   — OCR output has too much token overlap with caption

If OCR succeeds and text is accepted, figure_ocr_skip_reason is None.
"""
from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# --- Model ---
_MODEL_ID = "lightonai/LightOnOCR-2-1B"

# --- Rendering ---
_RENDER_SCALE = 2.0          # 2× → ~144 DPI for a typical 72-DPI PDF
_MIN_BBOX_DIM = 4.0          # minimum width/height in PDF points to attempt a crop

# --- Inference guardrails (CPU-safe defaults) ---
# Lower max_new_tokens dramatically reduces per-figure latency on CPU.
# 256 tokens ≈ 200 words, sufficient for typical figure content.
_DEFAULT_MAX_NEW_TOKENS: int = 256
# Per-figure wall-clock timeout (seconds). A best-effort ThreadPoolExecutor
# wrapper is used because torch on CPU releases the GIL during tensor ops.
# The background thread may outlive the timeout, but processing continues.
_DEFAULT_TIMEOUT_SECONDS: float = 120.0
# Crop resize cap — longest side in pixels. Large crops are the primary
# source of inference latency spikes on CPU.
_MAX_CROP_LONGEST_SIDE: int = 1536

# --- Quality filtering thresholds ---
_MIN_OCR_CHARS = 20          # reject below this after cleaning
_MAX_OCR_CHARS = 1200        # hard truncation ceiling
_MIN_PRINTABLE_RATIO = 0.60  # fraction of chars that should be printable alphanum/punct
_MIN_CONFIDENCE = 0.20       # reject if model reports confidence below this
_CAPTION_OVERLAP_REJECT = 0.85  # reject if OCR token overlap with caption exceeds this

# --- Module-level model cache (lazy-loaded once per process) ---
_ocr_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_ocr_text(raw: str) -> str:
    """Normalize whitespace, collapse blank lines, dedupe adjacent repeated lines.

    Does not alter content beyond whitespace normalization and obvious structural
    clean-up.  Truncates at _MAX_OCR_CHARS.
    """
    if not raw:
        return ""

    text = raw.strip()
    text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)        # at most one blank line between blocks

    # Remove adjacent duplicate lines (common OCR artefact on ruled lines)
    lines = text.split("\n")
    deduped: list[str] = []
    prev_stripped = None
    for line in lines:
        stripped = line.strip()
        if stripped != prev_stripped:
            deduped.append(line)
        prev_stripped = stripped
    text = "\n".join(deduped).strip()

    if len(text) > _MAX_OCR_CHARS:
        text = text[: _MAX_OCR_CHARS - 3].rstrip() + "..."

    return text


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

def _filter_ocr_quality(
    text: str,
    confidence: float | None,
    caption_text: str | None,
) -> tuple[bool, str | None]:
    """Apply lightweight quality gate to cleaned OCR text.

    Returns (keep: bool, rejection_reason: str | None).
    Favors precision: it is better to drop noisy OCR than to pollute prompts.

    Rejection conditions (any one is sufficient):
    1. Empty after cleaning.
    2. Too short to be meaningful (< _MIN_OCR_CHARS chars).
    3. Model-reported confidence below _MIN_CONFIDENCE.
    4. Low printable-character ratio — signal of junk / encoding artefacts.
    5. High token overlap with caption — OCR is just repeating caption text.
    """
    if not text:
        return False, "empty"

    if len(text) < _MIN_OCR_CHARS:
        return False, "too_short"

    if confidence is not None and confidence < _MIN_CONFIDENCE:
        return False, "low_quality"

    # Printable-ratio guard: count chars that are ordinary alphanum / punct / space
    printable = re.sub(r"[^\w\s.,;:!\?\-\+\=\%\$\#\@\(\)\[\]\/\\\"\'`~^&*]", "", text)
    if len(text) > 0 and len(printable) / len(text) < _MIN_PRINTABLE_RATIO:
        return False, "noisy_text"

    # Caption-duplicate guard: reject if OCR adds nothing beyond the caption
    if caption_text:
        def _tokens(t: str) -> set[str]:
            return {w.lower() for w in re.split(r"\W+", t) if len(w) >= 3}

        ocr_toks = _tokens(text)
        cap_toks = _tokens(caption_text)
        if ocr_toks and cap_toks:
            overlap = len(ocr_toks & cap_toks) / len(ocr_toks)
            if overlap > _CAPTION_OVERLAP_REJECT:
                return False, "caption_duplicate"

    return True, None


# ---------------------------------------------------------------------------
# Environment availability
# ---------------------------------------------------------------------------

def get_lighton_ocr_env_status() -> dict[str, Any]:
    """Return a dict describing OCR runtime environment readiness.

    Useful for debugging dependency issues without running inference.

    Returns::

        {
            "pymupdf_available": bool,
            "torch_available": bool,
            "transformers_available": bool,
            "lightonocr_classes_available": bool,
            "ocr_backend_ready": bool,   # True only if all four above are True
            "detail": str,               # human-readable summary
        }
    """
    status: dict[str, Any] = {
        "pymupdf_available": False,
        "torch_available": False,
        "transformers_available": False,
        "lightonocr_classes_available": False,
        "ocr_backend_ready": False,
        "detail": "",
    }

    missing: list[str] = []

    try:
        import fitz  # noqa: F401
        status["pymupdf_available"] = True
    except ImportError:
        missing.append("pymupdf (pip install pymupdf)")

    try:
        import torch  # noqa: F401
        status["torch_available"] = True
    except ImportError:
        missing.append("torch (pip install torch)")

    try:
        import transformers  # noqa: F401
        status["transformers_available"] = True
    except ImportError:
        missing.append("transformers (pip install transformers)")

    if status["transformers_available"]:
        try:
            from transformers import (  # noqa: F401
                LightOnOcrForConditionalGeneration,
                LightOnOcrProcessor,
            )
            status["lightonocr_classes_available"] = True
        except ImportError:
            missing.append(
                "LightOnOCR classes not found in transformers "
                "(upgrade: pip install --upgrade transformers)"
            )

    status["ocr_backend_ready"] = all([
        status["pymupdf_available"],
        status["torch_available"],
        status["lightonocr_classes_available"],
    ])

    if status["ocr_backend_ready"]:
        status["detail"] = f"Ready — model will load from '{_MODEL_ID}' on first call."
    else:
        status["detail"] = "Not ready — missing: " + "; ".join(missing)

    return status


def lighton_ocr_available() -> bool:
    """Return True if all dependencies for LightOnOCR-2 transformers inference are installed."""
    env = get_lighton_ocr_env_status()
    if not env["ocr_backend_ready"]:
        log.debug("LightOnOCR-2 not available: %s", env["detail"])
    return env["ocr_backend_ready"]


# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------

def _load_ocr_model(force_cpu: bool = False) -> tuple[Any, Any, str]:
    """Load (or return cached) LightOnOCR-2 model, processor, and device string.

    Model is loaded once per process and cached in _ocr_cache.
    Device selection: CUDA > MPS > CPU.

    When force_cpu=True, always load on CPU and do NOT update the cache —
    intended for smoke-test / debug runs only.
    """
    if not force_cpu and "model" in _ocr_cache:
        return _ocr_cache["model"], _ocr_cache["processor"], _ocr_cache["device"]

    import torch
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

    if force_cpu:
        device = "cpu"
        dtype = torch.float32
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # bfloat16 unreliable on MPS
    else:
        device = "cpu"
        dtype = torch.float32

    log.info("Loading LightOnOCR-2 model '%s' on device=%s dtype=%s", _MODEL_ID, device, dtype)

    model = LightOnOcrForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    processor = LightOnOcrProcessor.from_pretrained(_MODEL_ID)

    if not force_cpu:
        _ocr_cache["model"] = model
        _ocr_cache["processor"] = processor
        _ocr_cache["device"] = device
        _ocr_cache["dtype"] = dtype

    log.info("LightOnOCR-2 model loaded successfully (device=%s).", device)
    return model, processor, device


# ---------------------------------------------------------------------------
# LightOnOCR-2 transformers inference adapter
# ---------------------------------------------------------------------------

def _call_lighton_ocr(
    img_bytes: bytes,
    force_cpu: bool = False,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
) -> tuple[str, float | None]:
    """Invoke LightOnOCR-2 on PNG image bytes via Hugging Face transformers.

    Returns (raw_text, confidence_or_None).
    LightOnOCR-2 does not expose a confidence score — quality is None.

    Args:
        img_bytes:       PNG image bytes to run OCR on.
        force_cpu:       If True, load/run on CPU regardless of available hardware.
                         Bypasses the model cache — for smoke-test/debug use only.
        max_new_tokens:  Token generation cap. Lower values reduce CPU latency.

    Raises:
        ImportError: if transformers or torch is not installed.
        Exception:   any model-level error propagated to caller.
    """
    from PIL import Image as PILImage
    import torch

    model, processor, device = _load_ocr_model(force_cpu=force_cpu)
    dtype = torch.float32 if force_cpu else _ocr_cache.get("dtype", torch.float32)

    pil_image = PILImage.open(BytesIO(img_bytes)).convert("RGB")

    # Step 1: get the formatted text prompt string only (do NOT tokenize yet).
    # Calling apply_chat_template with tokenize=True and image/tensor kwargs triggers a
    # ProcessorMixin check in newer transformers that rejects kwargs passed directly to
    # processor.__call__ instead of via processor_kwargs={}. Splitting into two steps avoids
    # this entirely: apply_chat_template only formats text, processor() handles everything else.
    conversation = [
        {"role": "user", "content": [{"type": "image"}]}
    ]
    text_prompt: str = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Step 2: tokenize + encode image together via processor(text, images, return_tensors).
    # This is the correct modern-transformers path for VLM processors.
    inputs = processor(
        text=[text_prompt],
        images=[pil_image],
        return_tensors="pt",
    )

    # Move inputs to device with correct dtype
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    log.debug("[figure-ocr] model.generate starting (max_new_tokens=%d) ...", max_new_tokens)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    log.debug("[figure-ocr] model.generate done.")

    # Decode only the newly generated tokens (strip the prompt)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True).strip()

    return raw_text, None  # LightOnOCR-2 does not expose a confidence score


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_EMPTY_RESULT: dict[str, Any] = {
    "figure_ocr_text": "",
    "figure_ocr_model": _MODEL_ID,
    "figure_ocr_quality": None,
    "figure_ocr_attempted": False,
    "figure_ocr_skip_reason": None,
    "figure_ocr_source": "lightonocr2",
}


def ocr_figure_region(
    pdf_path: str | Path,
    page: int,
    bbox: list[float] | None,
    *,
    caption_text: str | None = None,
    scale: float = _RENDER_SCALE,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    timeout_seconds: float | None = _DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Render a figure region from a PDF page and run LightOnOCR-2.

    Args:
        pdf_path:        Path to the source PDF.
        page:            1-indexed page number (matches evidence object convention).
        bbox:            [x0, y0, x1, y1] in PDF user-space points.
        caption_text:    Caption of the figure — used for caption-duplicate filtering.
        scale:           Render scale (default 2.0 ≈ 144 DPI).
        max_new_tokens:  Token generation cap. Defaults to _DEFAULT_MAX_NEW_TOKENS (256).
                         Lower values cut CPU latency significantly.
        timeout_seconds: Per-figure wall-clock timeout. None = no limit.
                         Uses a ThreadPoolExecutor best-effort guard (torch releases
                         the GIL on CPU, so the thread can be interrupted). If timeout
                         fires, returns inference_failed / TimeoutError.

    Returns:
        Dict with the figure_ocr_* schema (see module docstring).
        figure_ocr_text is always a string (empty when rejected/skipped).
        figure_ocr_attempted is True only when inference ran regardless of outcome.
        Additional diagnostic fields when inference ran:
          figure_ocr_image_size    — [w, h] of original rendered crop
          figure_ocr_resized_size  — [w, h] after resize, or None if not resized
          figure_ocr_max_new_tokens — generation cap used
          figure_ocr_timeout_s     — timeout budget used (None if unlimited)
    """

    def _skip(reason: str) -> dict[str, Any]:
        return {**_EMPTY_RESULT, "figure_ocr_skip_reason": reason}

    # --- Pre-flight guards ---
    if not bbox or len(bbox) != 4:
        return _skip("no_bbox")

    x0, y0, x1, y1 = bbox
    if (x1 - x0) < _MIN_BBOX_DIM or (y1 - y0) < _MIN_BBOX_DIM:
        return _skip("bbox_too_small")

    # --- Render crop with PyMuPDF ---
    try:
        import fitz  # PyMuPDF — project dependency
    except ImportError:
        return _skip("dependency_missing")

    try:
        doc = fitz.open(str(pdf_path))
        fitz_page = doc[page - 1]          # fitz is 0-indexed
        clip = fitz.Rect(x0, y0, x1, y1)
        mat = fitz.Matrix(scale, scale)
        pix = fitz_page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
        img_bytes: bytes = pix.tobytes("png")
        doc.close()
    except Exception as exc:
        log.warning("Figure crop render failed (page=%s bbox=%s): %s", page, bbox, exc)
        return _skip("render_failed")

    # --- Resize crop if it exceeds the longest-side cap ---
    # Large crops are the primary source of latency spikes on CPU.
    from PIL import Image as PILImage
    pil_img = PILImage.open(BytesIO(img_bytes)).convert("RGB")
    original_size: list[int] = list(pil_img.size)          # [w, h]
    resized_size: list[int] | None = None
    longest = max(pil_img.size)
    if longest > _MAX_CROP_LONGEST_SIDE:
        factor = _MAX_CROP_LONGEST_SIDE / longest
        new_w = max(1, int(pil_img.width * factor))
        new_h = max(1, int(pil_img.height * factor))
        pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
        resized_size = [new_w, new_h]
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        log.info(
            "[figure-ocr] crop resized page=%s: %dx%d -> %dx%d",
            page, original_size[0], original_size[1], new_w, new_h,
        )

    # Metadata collected during this call, included in all non-skip returns.
    _diag: dict[str, Any] = {
        "figure_ocr_image_size": original_size,
        "figure_ocr_resized_size": resized_size,
        "figure_ocr_max_new_tokens": max_new_tokens,
        "figure_ocr_timeout_s": timeout_seconds,
    }

    # --- Inference (with optional best-effort timeout) ---
    try:
        if timeout_seconds is not None:
            import concurrent.futures
            _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            _fut = _executor.submit(_call_lighton_ocr, img_bytes, False, max_new_tokens)
            # shutdown(wait=False) so we don't block on cleanup after a timeout
            _executor.shutdown(wait=False)
            try:
                raw_text, confidence = _fut.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"LightOnOCR-2 inference exceeded {timeout_seconds:.0f}s budget"
                )
        else:
            raw_text, confidence = _call_lighton_ocr(img_bytes, max_new_tokens=max_new_tokens)
    except ImportError:
        return {**_skip("dependency_missing"), **_diag}
    except Exception as exc:
        error_type = type(exc).__name__
        error_msg = str(exc)[:400]
        log.warning(
            "LightOnOCR-2 inference failed (page=%s): [%s] %s",
            page, error_type, error_msg,
        )
        # Inference was attempted but raised — mark attempted=True with inference_failed reason.
        # Capture lightweight exception diagnostics for audit visibility.
        return {
            **_EMPTY_RESULT,
            "figure_ocr_model": _MODEL_ID,
            "figure_ocr_attempted": True,
            "figure_ocr_skip_reason": "inference_failed",
            "figure_ocr_error_type": error_type,
            "figure_ocr_error_message": error_msg,
            "figure_ocr_source": "lightonocr2",
            **_diag,
        }

    # At this point inference was attempted.
    result_base: dict[str, Any] = {
        **_EMPTY_RESULT,
        "figure_ocr_model": _MODEL_ID,
        "figure_ocr_quality": confidence,
        "figure_ocr_attempted": True,
        "figure_ocr_source": "lightonocr2",
        **_diag,
    }

    # --- Clean ---
    cleaned = _clean_ocr_text(raw_text)

    # --- Quality filter ---
    keep, rejection_reason = _filter_ocr_quality(cleaned, confidence, caption_text)
    if not keep:
        log.debug(
            "LightOnOCR-2 output rejected: page=%s reason=%s", page, rejection_reason
        )
        return {
            **result_base,
            "figure_ocr_text": "",
            "figure_ocr_skip_reason": rejection_reason,
        }

    log.debug(
        "LightOnOCR-2 accepted: page=%s chars=%d quality=%s",
        page, len(cleaned), confidence,
    )
    return {
        **result_base,
        "figure_ocr_text": cleaned,
        "figure_ocr_skip_reason": None,
    }


# ---------------------------------------------------------------------------
# Single-image smoke test (debug / diagnostic helper)
# ---------------------------------------------------------------------------

def ocr_smoke_test(
    pdf_path: str | Path,
    page: int,
    bbox: list[float],
    *,
    caption_text: str | None = None,
    force_cpu: bool = False,
    scale: float = _RENDER_SCALE,
) -> dict[str, Any]:
    """Run OCR on a single figure crop and return a diagnostic dict.

    Intended for debug/smoke-test use.  Does not update any database.

    Args:
        pdf_path:     Path to the source PDF.
        page:         1-indexed page number.
        bbox:         [x0, y0, x1, y1] in PDF points.
        caption_text: Optional caption text for duplicate filtering.
        force_cpu:    If True, force CPU inference (bypasses model cache).
        scale:        Render scale (default 2.0).

    Returns a dict with::

        env                 — get_lighton_ocr_env_status() result
        device_selected     — "cpu" / "cuda" / "mps" (actual inference device)
        crop_rendered       — bool
        image_size          — [width, height] in pixels, or None
        ocr_attempted       — bool
        ocr_status          — "accepted" | "rejected" | "failed" | "not_attempted"
        ocr_text_preview    — first 200 chars if accepted, else ""
        error_type          — exception class name if inference failed, else None
        error_message       — truncated error string if inference failed, else None
        figure_ocr_result   — full figure_ocr_* result dict
    """
    out: dict[str, Any] = {
        "env": get_lighton_ocr_env_status(),
        "device_selected": "cpu" if force_cpu else None,
        "crop_rendered": False,
        "image_size": None,
        "ocr_attempted": False,
        "ocr_status": "not_attempted",
        "ocr_text_preview": "",
        "error_type": None,
        "error_message": None,
        "figure_ocr_result": {},
    }

    # --- Render crop ---
    try:
        import fitz
        from PIL import Image as PILImage
    except ImportError as exc:
        out["error_type"] = type(exc).__name__
        out["error_message"] = str(exc)[:400]
        return out

    try:
        doc = fitz.open(str(pdf_path))
        fitz_page = doc[page - 1]
        x0, y0, x1, y1 = bbox
        clip = fitz.Rect(x0, y0, x1, y1)
        mat = fitz.Matrix(scale, scale)
        pix = fitz_page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
        img_bytes: bytes = pix.tobytes("png")
        doc.close()
        out["crop_rendered"] = True

        pil_img = PILImage.open(BytesIO(img_bytes))
        out["image_size"] = list(pil_img.size)  # [w, h]
    except Exception as exc:
        out["error_type"] = type(exc).__name__
        out["error_message"] = str(exc)[:400]
        return out

    # --- Determine device that will be used ---
    if not force_cpu:
        try:
            import torch
            if torch.cuda.is_available():
                out["device_selected"] = "cuda"
            elif torch.backends.mps.is_available():
                out["device_selected"] = "mps"
            else:
                out["device_selected"] = "cpu"
        except ImportError:
            out["device_selected"] = "unknown"

    # --- Run OCR ---
    out["ocr_attempted"] = True
    try:
        raw_text, confidence = _call_lighton_ocr(img_bytes, force_cpu=force_cpu)
    except Exception as exc:
        out["ocr_status"] = "failed"
        out["error_type"] = type(exc).__name__
        out["error_message"] = str(exc)[:400]
        return out

    cleaned = _clean_ocr_text(raw_text)
    keep, rejection_reason = _filter_ocr_quality(cleaned, confidence, caption_text)

    if keep:
        out["ocr_status"] = "accepted"
        out["ocr_text_preview"] = cleaned[:200]
    else:
        out["ocr_status"] = "rejected"

    out["figure_ocr_result"] = {
        "figure_ocr_text": cleaned if keep else "",
        "figure_ocr_model": _MODEL_ID,
        "figure_ocr_quality": confidence,
        "figure_ocr_attempted": True,
        "figure_ocr_skip_reason": rejection_reason,
        "figure_ocr_source": "lightonocr2",
    }
    return out
