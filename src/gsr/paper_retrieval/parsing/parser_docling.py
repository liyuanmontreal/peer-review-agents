"""Docling-based semantic parser for academic PDFs.

Docling provides better structural parsing than plain PyMuPDF:
- Reading-order-aware section segmentation
- Structured table extraction (markdown / row data)
- Figure and caption detection
- Multi-column layout handling

Usage:
    from gsr.paper_retrieval.parser_docling import parse_with_docling, docling_available

    if docling_available():
        result = parse_with_docling(pdf_path, paper_id, pymupdf_spans=spans)

Integration rules:
- PyMuPDF V2 spans ALWAYS run first (grounding layer for bboxes / UI red-boxes).
- Docling enriches NormalizedTable and NormalizedFigure objects.
- If Docling fails on any document, the caller falls back to PyMuPDF caption_extractor.
- Do NOT let Docling replace span-level grounding.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from .normalized_document import (
    NormalizedDocument,
    NormalizedFigure,
    NormalizedSection,
    NormalizedTable,
)

log = logging.getLogger(__name__)

_SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\b")


def docling_available() -> bool:
    """Return True if the docling package is installed."""
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        return False


def parse_with_docling(
    pdf_path: str | Path,
    paper_id: str,
    *,
    pymupdf_spans: list[dict[str, Any]],
) -> NormalizedDocument:
    """Parse a PDF with Docling and return a NormalizedDocument.

    PyMuPDF spans must be pre-computed and passed in; they remain the grounding
    layer. Docling output is used to enrich table and figure structure.

    Args:
        pdf_path: Path to the PDF file.
        paper_id: Paper identifier (must match papers.id in DB).
        pymupdf_spans: Pre-extracted V2 spans from parse_paper_pdf_v2().

    Returns:
        NormalizedDocument with Docling-enriched tables/figures.

    Raises:
        ImportError: If docling is not installed.
        RuntimeError: If Docling conversion fails.
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except ImportError as exc:
        raise ImportError(
            "docling is required for Docling parsing: pip install docling"
        ) from exc

    pdf_path = Path(pdf_path)
    log.info("Docling parse start: %s (paper_id=%s)", pdf_path.name, paper_id)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False          # rely on embedded text; OCR is PaddleOCR's job
    pipeline_options.do_table_structure = True

    try:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        doc = result.document
    except Exception as exc:
        raise RuntimeError(f"Docling conversion failed for '{pdf_path}': {exc}") from exc

    n_pages = _infer_n_pages(pymupdf_spans)

    sections = _extract_sections(doc, paper_id)
    tables = _extract_tables(doc, paper_id, pymupdf_spans=pymupdf_spans)
    figures = _extract_figures(doc, paper_id, pymupdf_spans=pymupdf_spans)

    log.info(
        "Docling parse done: %s — %d sections, %d tables, %d figures",
        pdf_path.name,
        len(sections),
        len(tables),
        len(figures),
    )

    return NormalizedDocument(
        paper_id=paper_id,
        pdf_path=str(pdf_path),
        n_pages=n_pages,
        source_parser="docling+pymupdf",
        spans=pymupdf_spans,
        sections=sections,
        tables=tables,
        figures=figures,
        parser_metadata={"docling_version": _get_docling_version()},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_docling_version() -> str:
    try:
        import docling
        return getattr(docling, "__version__", "unknown")
    except Exception:
        return "unknown"


def _infer_n_pages(spans: list[dict[str, Any]]) -> int:
    if not spans:
        return 0
    return max(s.get("page_num", 0) for s in spans)


def _section_number(heading: str | None) -> str | None:
    if not heading:
        return None
    m = _SECTION_NUM_RE.match((heading or "").strip())
    return m.group(1) if m else None


def _clean(text: str | None) -> str:
    return " ".join((text or "").split()).strip()


def _extract_sections(doc: Any, paper_id: str) -> list[NormalizedSection]:
    """Extract section structure from a Docling document."""
    sections: list[NormalizedSection] = []
    idx = 0

    try:
        for item, _ in doc.iterate_items():
            # Docling uses SectionHeaderItem for headings
            from docling.datamodel.document import SectionHeaderItem, TextItem
            if isinstance(item, SectionHeaderItem):
                heading = _clean(item.text)
                page = _item_page(item)
                text = _collect_section_text(doc, item)
                sections.append(
                    NormalizedSection(
                        id=f"{paper_id}_sec_{idx}",
                        heading=heading,
                        section_number=_section_number(heading),
                        page=page,
                        text=text,
                        span_ids=[],
                    )
                )
                idx += 1
    except Exception as exc:
        log.warning("Docling section extraction failed: %s — falling back to empty sections", exc)

    return sections


def _collect_section_text(doc: Any, header_item: Any) -> str:
    """Collect body text immediately following a section header."""
    lines: list[str] = []
    found_header = False

    try:
        from docling.datamodel.document import SectionHeaderItem, TextItem
        for item, _ in doc.iterate_items():
            if item is header_item:
                found_header = True
                continue
            if found_header:
                if isinstance(item, SectionHeaderItem):
                    break
                if isinstance(item, TextItem):
                    t = _clean(item.text)
                    if t:
                        lines.append(t)
    except Exception:
        pass

    return " ".join(lines)


def _item_page(item: Any) -> int:
    """Extract 1-indexed page number from a Docling item."""
    try:
        prov = item.prov[0] if item.prov else None
        if prov:
            return int(prov.page_no)
    except Exception:
        pass
    return 1


def _item_bbox(item: Any) -> list[float] | None:
    """Extract [x0, y0, x1, y1] bbox from a Docling item (in PDF points)."""
    try:
        prov = item.prov[0] if item.prov else None
        if prov and prov.bbox:
            b = prov.bbox
            return [float(b.l), float(b.t), float(b.r), float(b.b)]
    except Exception:
        pass
    return None


def _find_nearest_pymupdf_spans(
    bbox: list[float] | None,
    page: int,
    pymupdf_spans: list[dict[str, Any]],
    *,
    tolerance: float = 20.0,
) -> list[str]:
    """Return span IDs from pymupdf_spans that overlap with the given bbox on the given page."""
    if not bbox or not pymupdf_spans:
        return []

    x0, y0, x1, y1 = bbox
    matched: list[str] = []

    for span in pymupdf_spans:
        if span.get("page_num") != page:
            continue
        sb = span.get("bbox")
        if not sb or len(sb) != 4:
            continue
        sx0, sy0, sx1, sy1 = sb

        # Check for overlap with tolerance
        if sx1 < x0 - tolerance or sx0 > x1 + tolerance:
            continue
        if sy1 < y0 - tolerance or sy0 > y1 + tolerance:
            continue

        matched.append(span["id"])

    return matched


def _extract_tables(
    doc: Any,
    paper_id: str,
    *,
    pymupdf_spans: list[dict[str, Any]],
) -> list[NormalizedTable]:
    """Extract tables from a Docling document.

    Docling provides structured table data. We convert it to retrieval-friendly
    markdown text and map back to PyMuPDF span IDs for grounding.
    """
    tables: list[NormalizedTable] = []
    idx = 0

    try:
        from docling.datamodel.document import TableItem
        for item, _ in doc.iterate_items():
            if not isinstance(item, TableItem):
                continue

            page = _item_page(item)
            bbox = _item_bbox(item)

            # Convert table to markdown for retrieval (pass doc for current API)
            content_text = _table_to_text(item, doc)
            if not content_text:
                idx += 1
                continue

            # caption_text is a method in current Docling, not a property
            caption = _item_caption_text(item, doc)

            # Infer label from caption prefix (Table 1 / Table 2.1)
            label, caption_body = _split_label_from_caption(caption)
            if not label:
                label = f"Table {idx + 1}"

            # Find PyMuPDF spans that overlap this table region for grounding
            span_ids = _find_nearest_pymupdf_spans(bbox, page, pymupdf_spans)

            tables.append(
                NormalizedTable(
                    id=f"{paper_id}_docling_table_{idx}",
                    label=label,
                    caption=caption_body or caption or None,
                    page=page,
                    section=None,
                    section_number=None,
                    content_text=content_text,
                    bbox=bbox,
                    span_ids=span_ids,
                    source_parser="docling",
                    metadata={"docling_item_ref": str(getattr(item, "self_ref", ""))},
                )
            )
            idx += 1

    except Exception as exc:
        log.warning("Docling table extraction failed: %s", exc)

    return tables


def _item_caption_text(item: Any, doc: Any) -> str:
    """Safely call item.caption_text(doc) — it's a method, not a property."""
    try:
        fn = getattr(item, "caption_text", None)
        if callable(fn):
            return _clean(fn(doc))
    except Exception:
        pass
    return ""


def _table_to_text(table_item: Any, doc: Any) -> str:
    """Convert a Docling TableItem to retrieval-friendly text.

    Tries markdown export first (requires doc argument in current API),
    then falls back to flattened cell text.
    """
    # Try Docling's export_to_markdown — current API requires doc argument
    try:
        md = table_item.export_to_markdown(doc)
        if isinstance(md, str) and md.strip():
            return _clean(md)
    except Exception:
        pass

    # Fallback: flatten cell data from table_item.data.table_cells
    try:
        table_data = table_item.data
        if table_data and hasattr(table_data, "table_cells"):
            grid: dict[tuple[int, int], str] = {}
            for cell in table_data.table_cells:
                r = getattr(cell, "start_row_offset_idx", 0)
                c = getattr(cell, "start_col_offset_idx", 0)
                grid[(r, c)] = _clean(cell.text)
            if grid:
                max_row = max(r for r, _ in grid) + 1
                max_col = max(c for _, c in grid) + 1
                rows: list[str] = []
                for r in range(max_row):
                    row_cells = [grid.get((r, c), "") for c in range(max_col)]
                    rows.append(" | ".join(row_cells))
                return "\n".join(rows)
    except Exception:
        pass

    return ""


def _extract_figures(
    doc: Any,
    paper_id: str,
    *,
    pymupdf_spans: list[dict[str, Any]],
) -> list[NormalizedFigure]:
    """Extract figures from a Docling document."""
    figures: list[NormalizedFigure] = []
    idx = 0

    try:
        from docling.datamodel.document import PictureItem
        for item, _ in doc.iterate_items():
            if not isinstance(item, PictureItem):
                continue

            page = _item_page(item)
            bbox = _item_bbox(item)

            caption = _item_caption_text(item, doc)
            label, caption_body = _split_label_from_caption(caption)
            if not label:
                label = f"Figure {idx + 1}"

            span_ids = _find_nearest_pymupdf_spans(bbox, page, pymupdf_spans)

            # For v1 text-based verification, use caption as content_text
            content_text = caption or f"{label} on page {page}"

            figures.append(
                NormalizedFigure(
                    id=f"{paper_id}_docling_figure_{idx}",
                    label=label,
                    caption=caption_body or caption or None,
                    page=page,
                    section=None,
                    section_number=None,
                    content_text=content_text,
                    bbox=bbox,
                    span_ids=span_ids,
                    source_parser="docling",
                    metadata={"docling_item_ref": str(getattr(item, "self_ref", ""))},
                )
            )
            idx += 1

    except Exception as exc:
        log.warning("Docling figure extraction failed: %s", exc)

    return figures


def extract_tables_for_pages(
    pdf_path: str | Path,
    pages: list[int],
) -> dict[int, list[dict]]:
    """Run Docling on a page-scoped temporary PDF and return table data for the requested pages.

    Builds a temporary PDF containing only the requested pages (in ascending original
    order), runs Docling exclusively on that subset, then maps extracted table page
    numbers back to the original page numbers before returning.

    This replaces the previous full-document Docling approach and is substantially
    faster when only a small number of pages need enrichment (typical for top-k late
    enrichment before verification).

    Args:
        pdf_path: Path to the PDF.
        pages: 1-indexed original page numbers to include.

    Returns:
        {original_page_num: [{"label": str, "caption_text": str, "markdown": str, "bbox": list | None}]}
        Keys are ORIGINAL page numbers, not temp-PDF page numbers.

    Raises:
        ImportError: If docling or PyMuPDF is not installed.
        RuntimeError: If Docling conversion fails.
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.document import TableItem
    except ImportError as exc:
        raise ImportError(
            "docling is required for late table enrichment: pip install docling"
        ) from exc

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF (fitz) is required for page-scoped Docling enrichment: pip install pymupdf"
        ) from exc

    pdf_path = Path(pdf_path)

    if not pages:
        return {}

    # Sort and deduplicate; preserve ascending order for temp PDF page numbering.
    sorted_pages = sorted(set(pages))

    tmp_path: str | None = None
    t0 = time.perf_counter()

    try:
        src_doc = fitz.open(str(pdf_path))
        n_src_pages = src_doc.page_count  # total pages in the source PDF

        # Filter to valid 1-based page numbers; skip silently if out of range.
        valid_pages = [p for p in sorted_pages if 1 <= p <= n_src_pages]
        if not valid_pages:
            src_doc.close()
            log.warning(
                "[docling-pages] all requested pages out of range for '%s' "
                "(n_pages=%d, requested=%s)",
                pdf_path.name, n_src_pages, sorted_pages,
            )
            return {}

        # Build temp PDF with only the valid pages.
        # temp_to_orig maps 1-based temp page number → 1-based original page number.
        temp_to_orig: dict[int, int] = {}
        tmp_doc = fitz.open()
        for temp_idx, orig_page in enumerate(valid_pages):
            # fitz uses 0-based page indexes; our pages list is 1-based.
            tmp_doc.insert_pdf(src_doc, from_page=orig_page - 1, to_page=orig_page - 1)
            temp_to_orig[temp_idx + 1] = orig_page  # Docling returns 1-based page numbers

        src_doc.close()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        tmp_doc.save(tmp_path)
        tmp_doc.close()

        t_build = time.perf_counter() - t0
        log.info(
            "[docling-pages] START '%s': requested=%d valid=%d temp_pages=%d build=%.2fs",
            pdf_path.name, len(pages), len(valid_pages), len(valid_pages), t_build,
        )

        # Run Docling exclusively on the temporary page-scoped PDF.
        t_docling = time.perf_counter()
        try:
            converter = DocumentConverter()
            result = converter.convert(tmp_path)
            doc = result.document
        except Exception as exc:
            raise RuntimeError(
                f"Docling conversion failed for page-scoped temp PDF of '{pdf_path}': {exc}"
            ) from exc

        log.info(
            "[docling-pages] DONE '%s': docling=%.2fs total=%.2fs",
            pdf_path.name,
            time.perf_counter() - t_docling,
            time.perf_counter() - t0,
        )

        # Iterate Docling items; remap temp page numbers to original page numbers.
        n_temp_pages = len(valid_pages)
        out: dict[int, list[dict]] = {}
        idx = 0

        try:
            for item, _ in doc.iterate_items():
                if not isinstance(item, TableItem):
                    idx += 1
                    continue

                temp_page = _item_page(item)
                orig_page = temp_to_orig.get(temp_page)
                if orig_page is None:
                    # Docling returned a page number outside the temp PDF range — skip safely.
                    log.warning(
                        "[docling-pages] skipping table at temp_page=%d "
                        "(out of range 1..%d) for '%s'",
                        temp_page, n_temp_pages, pdf_path.name,
                    )
                    idx += 1
                    continue

                markdown = _table_to_text(item, doc)
                caption = _item_caption_text(item, doc)
                label, _cap_body = _split_label_from_caption(caption)
                if not label:
                    label = f"Table {idx + 1}"

                out.setdefault(orig_page, []).append({
                    "label": label,
                    "caption_text": caption,
                    "markdown": markdown,
                    "bbox": _item_bbox(item),
                })
                idx += 1
        except Exception as exc:
            log.warning("extract_tables_for_pages iteration failed: %s", exc)

        return out

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


_LABEL_PREFIX_RE = re.compile(
    r"^((?:Table|Tab\.|Figure|Fig\.)\s+\d+(?:\.\d+)*)[.:]\s*",
    re.IGNORECASE,
)


def _split_label_from_caption(caption: str | None) -> tuple[str | None, str | None]:
    """Split 'Table 3: some caption text' into ('Table 3', 'some caption text')."""
    if not caption:
        return None, None
    m = _LABEL_PREFIX_RE.match(caption.strip())
    if m:
        label = _clean(m.group(1))
        # Normalize Fig. → Figure
        if label.lower().startswith("fig."):
            label = "Figure" + label[4:]
        body = caption[m.end():].strip()
        return label, body or None
    return None, caption or None
