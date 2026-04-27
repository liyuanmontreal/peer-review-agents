"""Docling parser adapter.

Reuses the existing gsr parse_with_docling() + parse_paper_pdf_v2() stack.
PyMuPDF runs first for grounding (spans/bboxes), Docling enriches the
structural layer (tables, reading order, figure detection).

Text chunks are produced from the span stream (same as the PyMuPDF adapter)
but with Docling boundary hints applied (P1.7b), giving higher-quality segment
boundaries.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from gsr.paper_retrieval.parser_benchmark.adapters.base import BaseParserAdapter
from gsr.paper_retrieval.parser_benchmark.schema import (
    FigureObject,
    ParsedDocument,
    TableObject,
    TextChunk,
)

log = logging.getLogger(__name__)


class DoclingAdapter(BaseParserAdapter):
    """Adapter wrapping the GSR Docling+PyMuPDF combined stack."""

    name = "docling"

    def is_available(self) -> tuple[bool, str]:
        try:
            import fitz  # noqa: F401
        except ImportError:
            return False, "PyMuPDF (fitz) is not installed"
        try:
            import docling  # noqa: F401
            return True, ""
        except ImportError:
            return False, "docling is not installed (pip install docling)"

    def parse(self, pdf_path: str | Path, paper_id: str) -> ParsedDocument:
        from gsr.paper_retrieval.parsing.parser import parse_paper_pdf_v2
        from gsr.paper_retrieval.parsing.parser_docling import parse_with_docling
        from gsr.paper_retrieval.parsing.chunking import chunk_paper_v2_from_spans

        pdf_path = Path(pdf_path)
        doc = ParsedDocument(
            paper_id=paper_id,
            parser_name=self.name,
            pdf_path=str(pdf_path),
        )

        # ----------------------------------------------------------------
        # Step 1: PyMuPDF V2 spans (grounding layer — always required)
        # ----------------------------------------------------------------
        v2 = parse_paper_pdf_v2(str(pdf_path), paper_id)
        spans: list[dict] = v2.get("spans", [])
        n_pages: int = v2.get("n_pages", 0)
        doc.pages = {"n_pages": n_pages}

        # ----------------------------------------------------------------
        # Step 2: Docling enrichment
        # ----------------------------------------------------------------
        ndoc = parse_with_docling(str(pdf_path), paper_id, pymupdf_spans=spans)

        # ----------------------------------------------------------------
        # Step 3: text chunks — V2 span-based with Docling boundary hints
        # ----------------------------------------------------------------
        raw_chunks = chunk_paper_v2_from_spans(
            spans,
            chunk_size=8,
            chunk_overlap=2,
            min_chars=40,
            docling_doc=ndoc,
        )
        for idx, rc in enumerate(raw_chunks):
            doc.text_chunks.append(TextChunk.from_chunk_dict(rc, paper_id, idx))

        # ----------------------------------------------------------------
        # Step 4: tables from Docling NormalizedTable objects
        # ----------------------------------------------------------------
        for idx, tbl in enumerate(ndoc.tables):
            bbox_json = json.dumps(tbl.bbox) if tbl.bbox else None
            retrieval_parts = []
            if tbl.label:
                retrieval_parts.append(tbl.label)
            if tbl.caption:
                retrieval_parts.append(f"Caption: {tbl.caption}")
            if tbl.content_text:
                retrieval_parts.append(tbl.content_text)

            doc.tables.append(TableObject(
                id=tbl.id or f"{paper_id}_bench_tbl_{idx}",
                retrieval_text="\n".join(retrieval_parts),
                label=tbl.label,
                page=tbl.page,
                caption_text=tbl.caption,
                table_markdown=tbl.content_text if tbl.content_text and "|" in tbl.content_text else None,
                bbox_json=bbox_json,
                metadata={
                    "section": tbl.section,
                    "source_parser": tbl.source_parser,
                    **tbl.metadata,
                },
            ))

        # ----------------------------------------------------------------
        # Step 5: figures from Docling NormalizedFigure objects
        # ----------------------------------------------------------------
        for idx, fig in enumerate(ndoc.figures):
            bbox_json = json.dumps(fig.bbox) if fig.bbox else None
            retrieval_parts = []
            if fig.label:
                retrieval_parts.append(fig.label)
            if fig.caption:
                retrieval_parts.append(f"Caption: {fig.caption}")

            doc.figures.append(FigureObject(
                id=fig.id or f"{paper_id}_bench_fig_{idx}",
                retrieval_text="\n".join(retrieval_parts),
                label=fig.label,
                page=fig.page,
                caption_text=fig.caption,
                bbox_json=bbox_json,
                image_path=fig.asset_path,
                metadata={
                    "section": fig.section,
                    "source_parser": fig.source_parser,
                    **fig.metadata,
                },
            ))

        log.debug(
            "docling adapter: paper=%s chunks=%d figures=%d tables=%d",
            paper_id, len(doc.text_chunks), len(doc.figures), len(doc.tables),
        )
        return doc
