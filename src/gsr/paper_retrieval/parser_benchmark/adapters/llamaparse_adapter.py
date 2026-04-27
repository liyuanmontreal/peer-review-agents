"""LlamaParse adapter (optional, cloud-based).

LlamaParse is a cloud API service.  This adapter is intentionally lightweight:
- It is NEVER required for the benchmark to run.
- It skips automatically when credentials are absent.
- Do NOT add this as a required dependency.

To use:
    1. pip install llama-parse
    2. Set LLAMA_CLOUD_API_KEY in your .env file
    3. Run: gsr benchmark-parse --pdf paper.pdf --parsers llamaparse

Limitations:
- Requires internet access and a LlamaCloud API key.
- Rate limits and latency vary by plan.
- Figure/table extraction quality depends on LlamaParse plan tier.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from gsr.paper_retrieval.parser_benchmark.adapters.base import BaseParserAdapter
from gsr.paper_retrieval.parser_benchmark.schema import (
    FigureObject,
    ParsedDocument,
    TableObject,
    TextChunk,
)

log = logging.getLogger(__name__)

_FIG_CAPTION = re.compile(r"^\s*(Figure|Fig\.)\s+(\d+(?:\.\d+)*)\b[.:]\s*(.*)", re.IGNORECASE)
_TBL_CAPTION = re.compile(r"^\s*(Table|Tab\.)\s+(\d+(?:\.\d+)*)\b[.:]\s*(.*)", re.IGNORECASE)
_APPROX_CHUNK_WORDS = 200
_MIN_CHUNK_CHARS = 40


class LlamaParseAdapter(BaseParserAdapter):
    """Optional adapter for LlamaParse cloud API."""

    name = "llamaparse"

    def is_available(self) -> tuple[bool, str]:
        try:
            import llama_parse  # noqa: F401
        except ImportError:
            return False, (
                "llama-parse package is not installed "
                "(pip install llama-parse)"
            )
        api_key = os.environ.get("LLAMA_CLOUD_API_KEY", "").strip()
        if not api_key:
            return False, (
                "LLAMA_CLOUD_API_KEY is not set. "
                "Set this environment variable to enable LlamaParse."
            )
        return True, ""

    def parse(self, pdf_path: str | Path, paper_id: str) -> ParsedDocument:
        from llama_parse import LlamaParse  # type: ignore[import]

        pdf_path = Path(pdf_path)
        doc = ParsedDocument(
            paper_id=paper_id,
            parser_name=self.name,
            pdf_path=str(pdf_path),
        )

        api_key = os.environ.get("LLAMA_CLOUD_API_KEY", "").strip()
        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            verbose=False,
        )

        try:
            documents = parser.load_data(str(pdf_path))
        except Exception as exc:
            doc.metadata["llamaparse_error"] = str(exc)
            log.warning("LlamaParse API call failed: %s", exc)
            return doc

        # Concatenate all pages
        full_text = "\n\n".join(d.text for d in documents if hasattr(d, "text"))
        doc.metadata["llamaparse_n_documents"] = len(documents)

        _parse_markdown_into_doc(full_text, paper_id, doc)
        log.debug(
            "llamaparse adapter: paper=%s chunks=%d figures=%d tables=%d",
            paper_id, len(doc.text_chunks), len(doc.figures), len(doc.tables),
        )
        return doc


# ---------------------------------------------------------------------------
# Shared markdown parser (same pattern as marker_adapter)
# ---------------------------------------------------------------------------

def _parse_markdown_into_doc(markdown: str, paper_id: str, doc: ParsedDocument) -> None:
    lines = markdown.splitlines()
    fig_idx = tbl_idx = chunk_idx = 0

    table_line_indices: set[int] = set()
    table_blocks: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            table_blocks.append((i, j, "\n".join(lines[i:j])))
            for k in range(i, j):
                table_line_indices.add(k)
            i = j
        else:
            i += 1

    for start, _end, md_block in table_blocks:
        label: str | None = None
        caption_text: str | None = None
        if start > 0:
            prev = lines[start - 1].strip()
            m = _TBL_CAPTION.match(prev)
            if m:
                label = f"Table {m.group(2)}"
                caption_text = m.group(3).strip() or prev

        parts = []
        if label:
            parts.append(label)
        if caption_text:
            parts.append(f"Caption: {caption_text}")
        parts.append(md_block[:1000])

        doc.tables.append(TableObject(
            id=f"{paper_id}_bench_tbl_{tbl_idx}",
            retrieval_text="\n".join(parts),
            label=label,
            page=None,
            caption_text=caption_text,
            table_markdown=md_block,
        ))
        tbl_idx += 1

    body_lines: list[str] = []
    for i, line in enumerate(lines):
        if i in table_line_indices:
            continue
        m_fig = _FIG_CAPTION.match(line)
        if m_fig:
            label = f"Figure {m_fig.group(2)}"
            caption_text = m_fig.group(3).strip()
            parts = [label]
            if caption_text:
                parts.append(f"Caption: {caption_text}")
            doc.figures.append(FigureObject(
                id=f"{paper_id}_bench_fig_{fig_idx}",
                retrieval_text="\n".join(parts),
                label=label,
                page=None,
                caption_text=caption_text or None,
            ))
            fig_idx += 1
            continue
        body_lines.append(line)

    words = " ".join(body_lines).split()
    for start in range(0, max(1, len(words)), _APPROX_CHUNK_WORDS):
        chunk_words = words[start: start + _APPROX_CHUNK_WORDS]
        text = " ".join(chunk_words).strip()
        if len(text) < _MIN_CHUNK_CHARS:
            continue
        doc.text_chunks.append(TextChunk(
            id=f"{paper_id}_bench_chunk_{chunk_idx}",
            text=text,
            char_len=len(text),
            metadata={"word_start": start, "source": "llamaparse_markdown"},
        ))
        chunk_idx += 1
