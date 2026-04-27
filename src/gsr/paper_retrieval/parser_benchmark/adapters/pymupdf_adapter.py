"""PyMuPDF parser adapter.

This is the required baseline adapter.  It reuses the existing production
parsing stack (parse_paper_pdf_v2, chunk_paper_v2_from_spans,
extract_captions_from_spans) rather than duplicating complex heuristics.
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


class PyMuPDFAdapter(BaseParserAdapter):
    """Baseline adapter using the existing GSR PyMuPDF V2 pipeline."""

    name = "pymupdf"

    def is_available(self) -> tuple[bool, str]:
        try:
            import fitz  # noqa: F401
            return True, ""
        except ImportError:
            return False, "PyMuPDF (fitz) is not installed"

    def parse(self, pdf_path: str | Path, paper_id: str) -> ParsedDocument:
        from gsr.paper_retrieval.parsing.parser import parse_paper_pdf_v2
        from gsr.paper_retrieval.parsing.chunking import chunk_paper_v2_from_spans
        from gsr.paper_retrieval.parsing.caption_extractor import extract_captions_from_spans

        pdf_path = Path(pdf_path)
        doc = ParsedDocument(
            paper_id=paper_id,
            parser_name=self.name,
            pdf_path=str(pdf_path),
        )

        # ----------------------------------------------------------------
        # Step 1: parse with PyMuPDF V2 → spans
        # ----------------------------------------------------------------
        v2 = parse_paper_pdf_v2(str(pdf_path), paper_id)
        spans: list[dict] = v2.get("spans", [])
        n_pages: int = v2.get("n_pages", 0)
        doc.pages = {"n_pages": n_pages}

        # ----------------------------------------------------------------
        # Step 2: produce text chunks (V2 span-based)
        # ----------------------------------------------------------------
        raw_chunks = chunk_paper_v2_from_spans(
            spans,
            chunk_size=8,
            chunk_overlap=2,
            min_chars=40,
        )
        for idx, rc in enumerate(raw_chunks):
            doc.text_chunks.append(TextChunk.from_chunk_dict(rc, paper_id, idx))

        # ----------------------------------------------------------------
        # Step 3: extract figures / tables via caption-first logic
        # ----------------------------------------------------------------
        captions = extract_captions_from_spans(spans)
        fig_idx = 0
        tbl_idx = 0
        for cap in captions:
            kind = cap.get("kind")
            label = cap.get("label")
            page_num = cap.get("page_num")
            caption_text = cap.get("caption_text", "")
            bbox = cap.get("bbox")
            bbox_json = json.dumps(bbox) if bbox is not None else None

            retrieval_text_parts = []
            if label:
                retrieval_text_parts.append(label)
            if caption_text:
                retrieval_text_parts.append(f"Caption: {caption_text}")
            retrieval_text = "\n".join(retrieval_text_parts)

            if kind == "figure":
                doc.figures.append(FigureObject(
                    id=f"{paper_id}_bench_fig_{fig_idx}",
                    retrieval_text=retrieval_text,
                    label=label,
                    page=page_num,
                    caption_text=caption_text or None,
                    bbox_json=bbox_json,
                    metadata={"span_ids": cap.get("span_ids"), "section": cap.get("section")},
                ))
                fig_idx += 1
            elif kind == "table":
                doc.tables.append(TableObject(
                    id=f"{paper_id}_bench_tbl_{tbl_idx}",
                    retrieval_text=retrieval_text,
                    label=label,
                    page=page_num,
                    caption_text=caption_text or None,
                    bbox_json=bbox_json,
                    metadata={"span_ids": cap.get("span_ids"), "section": cap.get("section")},
                ))
                tbl_idx += 1

        log.debug(
            "pymupdf adapter: paper=%s chunks=%d figures=%d tables=%d",
            paper_id, len(doc.text_chunks), len(doc.figures), len(doc.tables),
        )
        return doc
