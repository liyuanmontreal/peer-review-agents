"""Parser routing for the PDF parsing pipeline.

Routing strategy
----------------
Layer 1 — PyMuPDF V2 (always runs):
    Extracts line-level spans with bbox coordinates.
    This is the grounding layer: source of truth for PDF red-box overlays.

Layer 2 — Docling (optional, primary semantic parser):
    Provides better structural understanding:
    - table content (structured, markdown-exportable)
    - figure/caption detection
    - reading-order-aware section segmentation
    Runs when: prefer_docling=True AND docling is installed.
    Falls back gracefully if Docling raises an error.

Layer 3 — PaddleOCR (future, page-level fallback):
    Only invoked for pages with poor extracted text quality.
    Not implemented in MVP.

The router always returns a NormalizedDocument. When Docling is used, its
tables/figures are attached. Otherwise NormalizedDocument.tables/figures
remain empty and the caption_extractor path in evidence_builder fills them.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .normalized_document import NormalizedDocument, NormalizedSection
from .parser import parse_paper_pdf_v2

log = logging.getLogger(__name__)

# Minimum average chars-per-span below which we consider a page "text-poor".
# Reserved for future PaddleOCR fallback.
_TEXT_POOR_THRESHOLD = 8.0


def route(
    pdf_path: str | Path,
    paper_id: str,
    *,
    prefer_docling: bool = True,
) -> NormalizedDocument:
    """Parse a PDF and return a NormalizedDocument.

    Always runs PyMuPDF V2 first for span/bbox grounding.
    Optionally runs Docling on top for richer structure.

    Args:
        pdf_path: Path to the PDF file.
        paper_id: Paper identifier.
        prefer_docling: Try Docling if installed. Fallback to PyMuPDF-only on error.

    Returns:
        NormalizedDocument. source_parser is "docling+pymupdf" or "pymupdf".
    """
    pdf_path = Path(pdf_path)

    # --- Layer 1: PyMuPDF V2 (grounding) ---
    log.info("parse_router: running PyMuPDF V2 for %s (paper_id=%s)", pdf_path.name, paper_id)
    v2_result = parse_paper_pdf_v2(str(pdf_path), paper_id)
    spans: list[dict[str, Any]] = v2_result["spans"]
    v2_sections = v2_result.get("sections", [])
    n_pages: int = v2_result.get("n_pages", 0)

    normalized_sections = _sections_from_v2(v2_sections, paper_id)

    # --- Layer 2: Docling (optional semantic enrichment) ---
    if prefer_docling:
        try:
            from .parser_docling import docling_available, parse_with_docling

            if docling_available():
                log.info("parse_router: running Docling for %s", pdf_path.name)
                doc = parse_with_docling(str(pdf_path), paper_id, pymupdf_spans=spans)

                # Docling sections override V2 sections when available
                sections = doc.sections if doc.sections else normalized_sections

                return NormalizedDocument(
                    paper_id=paper_id,
                    pdf_path=str(pdf_path),
                    n_pages=n_pages,
                    source_parser="docling+pymupdf",
                    spans=spans,
                    sections=sections,
                    tables=doc.tables,
                    figures=doc.figures,
                    parser_metadata={
                        **doc.parser_metadata,
                        "v2_n_spans": len(spans),
                        "v2_n_sections": len(v2_sections),
                    },
                )
            else:
                log.debug("parse_router: Docling not installed, using PyMuPDF-only")

        except Exception as exc:
            log.warning(
                "parse_router: Docling failed for '%s' (%s) — falling back to PyMuPDF-only",
                pdf_path.name,
                exc,
            )

    # --- PyMuPDF-only path ---
    # tables / figures will be populated later by evidence_builder via caption_extractor
    return NormalizedDocument(
        paper_id=paper_id,
        pdf_path=str(pdf_path),
        n_pages=n_pages,
        source_parser="pymupdf",
        spans=spans,
        sections=normalized_sections,
        tables=[],
        figures=[],
        parser_metadata={
            "v2_n_spans": len(spans),
            "v2_n_sections": len(v2_sections),
        },
    )


def text_quality_score(page_spans: list[dict[str, Any]]) -> float:
    """Compute a simple text-quality score for a page's spans.

    Returns average chars per span. Low values indicate a scan-like or
    image-heavy page that might benefit from OCR.

    Reserved for future PaddleOCR fallback routing.
    """
    if not page_spans:
        return 0.0
    total_chars = sum(len((s.get("text") or "").strip()) for s in page_spans)
    return total_chars / len(page_spans)


def identify_poor_quality_pages(
    spans: list[dict[str, Any]],
    *,
    threshold: float = _TEXT_POOR_THRESHOLD,
) -> list[int]:
    """Return page numbers (1-indexed) with poor text extraction quality.

    These pages are candidates for PaddleOCR fallback (Phase 2).
    """
    from collections import defaultdict

    pages: dict[int, list[dict]] = defaultdict(list)
    for span in spans:
        pages[span.get("page_num", 0)].append(span)

    poor: list[int] = []
    for page_num, page_spans in sorted(pages.items()):
        score = text_quality_score(page_spans)
        if score < threshold:
            poor.append(page_num)
            log.debug(
                "parse_router: page %d text_quality=%.1f (threshold=%.1f) — candidate for OCR",
                page_num, score, threshold,
            )

    return poor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sections_from_v2(
    v2_sections: list[dict[str, Any]],
    paper_id: str,
) -> list[NormalizedSection]:
    """Convert V2 section dicts to NormalizedSection objects."""
    import re
    _num_re = re.compile(r"^(\d+(?:\.\d+)*)\b")

    out: list[NormalizedSection] = []
    for idx, sec in enumerate(v2_sections):
        heading = sec.get("heading", "")
        m = _num_re.match(heading.strip())
        section_number = m.group(1) if m else None
        out.append(
            NormalizedSection(
                id=f"{paper_id}_sec_{idx}",
                heading=heading,
                section_number=section_number,
                page=sec.get("page", 1),
                text=sec.get("text", ""),
                span_ids=sec.get("span_ids", []),
            )
        )
    return out
