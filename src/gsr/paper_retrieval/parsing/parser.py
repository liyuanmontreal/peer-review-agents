"""PDF parsing for academic papers – Module 3, Step 1.

Extracts structured sections from downloaded PDFs using PyMuPDF (fitz).

This module now supports two parsing modes:

V1:
    - plain-text extraction via ``page.get_text("text")``
    - section segmentation over page text
    - compatible with the existing text-only chunking pipeline

V2:
    - layout-aware extraction via ``page.get_text("dict")``
    - line-level spans with bounding boxes (bbox)
    - section segmentation over spans
    - designed for bbox-aware chunking / evidence rendering
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section-heading detection
# ---------------------------------------------------------------------------

# Common academic paper section names (with optional numbering prefix).
_KNOWN_HEADINGS = re.compile(
    r"^(?:\d+\.?\d*\.?\s+)?("
    r"abstract|introduction|related work|background|preliminaries|"
    r"problem (statement|formulation)|notation|"
    r"method(?:ology)?s?|approach|model|framework|architecture|"
    r"experiment(?:s|al (setup|design|results)?)?|"
    r"evaluation|results?|analysis|ablation|"
    r"discussion|limitation(?:s)?|future work|"
    r"conclusion(?:s)?|summary|"
    r"references?|bibliography|appendix|supplementary|acknowledgments?"
    r")\b",
    re.IGNORECASE,
)

# Numbered section like "1 Introduction" or "2.1 Method" or "A.1 Proofs"
_NUMBERED_SECTION = re.compile(
    r"^(?:[A-Z]\.)?(?:\d+)(?:\.\d+)*\.?\s+[A-Z]"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_paper_pdf(pdf_path: str | Path, paper_id: str) -> dict:
    """Parse an academic PDF and return a structured document dict.

    Uses PyMuPDF for text extraction, then applies heuristic section
    detection to segment the document into named sections.

    Args:
        pdf_path: Path to the PDF file.
        paper_id: Identifier of the paper (must match ``papers.id`` in DB).

    Returns:
        dict with keys:
            ``paper_id`` (str),
            ``pdf_path`` (str),
            ``n_pages`` (int),
            ``sections`` (list[dict]):
                each section has ``heading`` (str), ``page`` (int),
                ``text`` (str).

    Raises:
        ImportError: If PyMuPDF is not installed.
        ValueError: If the PDF cannot be opened.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF parsing: pip install pymupdf"
        ) from exc

    pdf_path = Path(pdf_path)
    log.debug("Parsing PDF: %s (paper_id=%s)", pdf_path, paper_id)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Failed to open PDF '{pdf_path}': {exc}") from exc

    try:
        pages: list[dict] = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")  # plain text extraction
            pages.append({"page": page_num, "text": text})
    finally:
        doc.close()

    sections = _segment_sections(pages)

    log.info(
        "Parsed '%s': %d pages → %d sections",
        pdf_path.name, len(pages), len(sections),
    )
    return {
        "paper_id": paper_id,
        "pdf_path": str(pdf_path),
        "n_pages": len(pages),
        "sections": sections,
    }


def parse_paper_pdf_v2(pdf_path: str | Path, paper_id: str) -> dict:
    """Parse an academic PDF and return a layout-aware structured document dict.

    V2 parser extracts line-level spans with bounding boxes (bbox) using
    PyMuPDF's ``get_text("dict")`` and then segments sections based on
    heading heuristics over spans.

    Args:
        pdf_path: Path to the PDF file.
        paper_id: Identifier of the paper (must match ``papers.id`` in DB).

    Returns:
        dict with keys:
            ``paper_id`` (str),
            ``pdf_path`` (str),
            ``n_pages`` (int),
            ``spans`` (list[dict]),
            ``sections`` (list[dict])

        Span dicts contain:
            ``id`` (str),
            ``paper_id`` (str),
            ``page_num`` (int),
            ``span_index`` (int),
            ``text`` (str),
            ``bbox`` (list[float]),
            ``block_index`` (int),
            ``line_index`` (int)

        Section dicts contain:
            ``heading`` (str),
            ``page`` (int),
            ``text`` (str),
            ``span_ids`` (list[str])

    Raises:
        ImportError: If PyMuPDF is not installed.
        ValueError: If the PDF cannot be opened.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF parsing: pip install pymupdf"
        ) from exc

    pdf_path = Path(pdf_path)
    log.debug("Parsing PDF V2: %s (paper_id=%s)", pdf_path, paper_id)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Failed to open PDF '{pdf_path}': {exc}") from exc

    try:
        spans = _extract_line_spans_from_doc(doc, paper_id)
        sections = _segment_sections_from_spans(spans)
        spans = attach_section_labels_to_spans(spans, sections)
    finally:
        doc.close()

    log.info(
        "Parsed V2 '%s': %d pages → %d spans → %d sections",
        pdf_path.name,
        _infer_n_pages(spans),
        len(spans),
        len(sections),
    )
    return {
        "paper_id": paper_id,
        "pdf_path": str(pdf_path),
        "n_pages": _infer_n_pages(spans),
        "spans": spans,
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# V1 internal helpers
# ---------------------------------------------------------------------------

def _segment_sections(pages: list[dict]) -> list[dict]:
    """Segment page-level text into named sections using heading heuristics."""
    sections: list[dict] = []
    current_heading = "preamble"
    current_page = 1
    current_lines: list[str] = []

    for entry in pages:
        page_num = entry["page"]
        for line in entry["text"].split("\n"):
            stripped = line.strip()

            if _is_section_heading(stripped):
                # Flush accumulated lines into current section.
                _flush_section(sections, current_heading, current_page, current_lines)
                current_heading = _normalize_heading(stripped)
                current_page = page_num
                current_lines = []
            else:
                current_lines.append(line)

    # Flush final section.
    _flush_section(sections, current_heading, current_page, current_lines)

    # Fallback: treat the entire doc as one section if detection yielded nothing.
    if not sections:
        all_text = "\n".join(e["text"] for e in pages).strip()
        if all_text:
            sections = [{"heading": "body", "page": 1, "text": all_text}]

    return sections


def _flush_section(
    sections: list[dict],
    heading: str,
    page: int,
    lines: list[str],
) -> None:
    """Append a section dict if the accumulated text is non-empty."""
    text = "\n".join(lines).strip()
    if text:
        sections.append({"heading": heading, "page": page, "text": text})


# ---------------------------------------------------------------------------
# V2 internal helpers
# ---------------------------------------------------------------------------

def _extract_line_spans_from_doc(doc: Any, paper_id: str) -> list[dict[str, Any]]:
    """Extract line-level spans with bbox from an open PyMuPDF document."""
    spans: list[dict[str, Any]] = []
    global_span_index = 0

    for page_idx, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")

        for block_idx, block in enumerate(page_dict.get("blocks", [])):
            if block.get("type") != 0:
                # Skip non-text blocks (images, drawings, etc.)
                continue

            for line_idx, line in enumerate(block.get("lines", [])):
                text_parts: list[str] = []
                boxes: list[list[float]] = []

                for span in line.get("spans", []):
                    txt = _normalize_span_text(span.get("text", ""))
                    if not txt:
                        continue
                    text_parts.append(txt)
                    boxes.append(list(span["bbox"]))

                if not text_parts or not boxes:
                    continue

                line_text = _normalize_span_text(" ".join(text_parts))
                if not line_text:
                    continue

                span_id = f"{paper_id}_p{page_idx}_s{global_span_index}"

                spans.append(
                    {
                        "id": span_id,
                        "paper_id": paper_id,
                        "page_num": page_idx,
                        "span_index": global_span_index,
                        "text": line_text,
                        "bbox": _merge_bboxes(boxes),
                        "block_index": block_idx,
                        "line_index": line_idx,
                    }
                )
                global_span_index += 1

    return spans


def _segment_sections_from_spans(
    spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Segment sections using heading heuristics over line-level spans."""
    sections: list[dict[str, Any]] = []
    current_heading = "preamble"
    current_page = 1
    current_span_ids: list[str] = []
    current_lines: list[str] = []

    for span in spans:
        text = span["text"].strip()

        if _is_section_heading(text):
            _flush_section_v2(
                sections,
                heading=current_heading,
                page=current_page,
                lines=current_lines,
                span_ids=current_span_ids,
            )
            current_heading = _normalize_heading(text)
            current_page = span["page_num"]
            current_lines = []
            current_span_ids = []
        else:
            current_lines.append(text)
            current_span_ids.append(span["id"])

    _flush_section_v2(
        sections,
        heading=current_heading,
        page=current_page,
        lines=current_lines,
        span_ids=current_span_ids,
    )

    if not sections and spans:
        all_text = "\n".join(s["text"] for s in spans).strip()
        if all_text:
            sections = [
                {
                    "heading": "body",
                    "page": 1,
                    "text": all_text,
                    "span_ids": [s["id"] for s in spans],
                }
            ]

    return sections


def _flush_section_v2(
    sections: list[dict[str, Any]],
    *,
    heading: str,
    page: int,
    lines: list[str],
    span_ids: list[str],
) -> None:
    """Append a V2 section dict if the accumulated text is non-empty."""
    text = "\n".join(lines).strip()
    if text:
        sections.append(
            {
                "heading": heading,
                "page": page,
                "text": text,
                "span_ids": list(span_ids),
            }
        )


def attach_section_labels_to_spans(
    spans: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach a representative section label to each span in-place.

    This makes later chunking / reporting easier because span-level metadata
    already carries the section heading.
    """
    span_map = {s["id"]: s for s in spans}

    for sec in sections:
        heading = sec["heading"]
        for span_id in sec.get("span_ids", []):
            if span_id in span_map:
                span_map[span_id]["section"] = heading

    return spans


def _normalize_span_text(text: str) -> str:
    """Normalize whitespace for a single extracted span/line."""
    return " ".join((text or "").strip().split())


def _merge_bboxes(boxes: list[list[float]]) -> list[float]:
    """Merge multiple bbox rectangles into one enclosing rectangle."""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return [x0, y0, x1, y1]


def _infer_n_pages(spans: list[dict[str, Any]]) -> int:
    """Infer page count from extracted spans."""
    if not spans:
        return 0
    return max(s["page_num"] for s in spans)


# ---------------------------------------------------------------------------
# Shared heading helpers
# ---------------------------------------------------------------------------

def _is_section_heading(line: str) -> bool:
    """Return True if *line* looks like an academic paper section heading."""
    if not line or len(line) > 80:
        return False
    if _KNOWN_HEADINGS.match(line):
        return True
    if _NUMBERED_SECTION.match(line):
        return True
    return False


def _normalize_heading(line: str) -> str:
    """Strip leading numbering and lowercase a heading line."""
    heading = re.sub(r"^(?:[A-Z]\.)?(?:\d+)(?:\.\d+)*\.?\s+", "", line)
    return heading.lower().strip()