"""Marker parser adapter.

Marker (https://github.com/VikParuchuri/marker) converts PDFs to structured
Markdown with table/figure detection.  It can be used as:

1. Python package:  ``from marker.convert import convert_single_pdf``
2. CLI tool:        ``marker_single input.pdf --output_dir out/``

This adapter tries the Python API first, falls back to the CLI, and degrades
gracefully to text-chunk-only output when neither is available.

Limitations (recorded in benchmark metadata):
- Marker's table output is Markdown; no structured row data is available.
- Figure bbox is not reliably available from Marker output alone.
- Page numbers inside Marker output use their own scheme and may not align
  with PyMuPDF page numbers perfectly.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
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

# Marker markdown table pattern: line starts with |
_MD_TABLE_BLOCK = re.compile(r"(\|.+\|\n)+", re.MULTILINE)
# Caption-like prefix patterns.
# Strip leading markdown bold (**) before matching, e.g. "**Table 1:**" → "Table 1:"
_FIG_CAPTION = re.compile(r"^\s*\*{0,2}(Figure|Fig\.)\s+(\d+(?:\.\d+)*)\b\*{0,2}[.:]\s*(.*)", re.IGNORECASE)
_TBL_CAPTION = re.compile(r"^\s*\*{0,2}(Table|Tab\.)\s+(\d+(?:\.\d+)*)\b\*{0,2}[.:]\s*(.*)", re.IGNORECASE)

_MIN_CHUNK_CHARS = 40
_APPROX_CHUNK_WORDS = 200   # word-based sliding window fallback


def _marker_python_available() -> bool:
    try:
        import marker  # noqa: F401
        return True
    except ImportError:
        return False


def _marker_cli_available() -> bool:
    return shutil.which("marker_single") is not None or shutil.which("marker") is not None


class MarkerAdapter(BaseParserAdapter):
    """Adapter for the Marker PDF-to-markdown tool."""

    name = "marker"

    def is_available(self) -> tuple[bool, str]:
        if _marker_python_available():
            return True, ""
        if _marker_cli_available():
            return True, ""
        return False, (
            "Marker is not installed. "
            "Install with: pip install marker-pdf  "
            "or ensure 'marker_single' / 'marker' CLI is on PATH"
        )

    def parse(self, pdf_path: str | Path, paper_id: str) -> ParsedDocument:
        pdf_path = Path(pdf_path)
        doc = ParsedDocument(
            paper_id=paper_id,
            parser_name=self.name,
            pdf_path=str(pdf_path),
        )

        markdown_text, parse_meta = self._get_markdown(pdf_path)
        doc.metadata.update(parse_meta)

        if not markdown_text:
            doc.metadata["warning"] = "Marker produced empty output"
            return doc

        # ----------------------------------------------------------------
        # Parse markdown into chunks / figures / tables
        # ----------------------------------------------------------------
        _parse_markdown_into_doc(markdown_text, paper_id, doc)
        log.debug(
            "marker adapter: paper=%s chunks=%d figures=%d tables=%d",
            paper_id, len(doc.text_chunks), len(doc.figures), len(doc.tables),
        )
        return doc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_markdown(self, pdf_path: Path) -> tuple[str, dict[str, Any]]:
        """Return (markdown_text, metadata_dict).

        Tries Python API first, then CLI.
        """
        if _marker_python_available():
            return self._run_python_api(pdf_path)
        if _marker_cli_available():
            return self._run_cli(pdf_path)
        raise RuntimeError("Marker is not available (checked both Python API and CLI)")

    def _run_python_api(self, pdf_path: Path) -> tuple[str, dict[str, Any]]:
        meta: dict[str, Any] = {"marker_mode": "python_api"}
        try:
            # Marker ≥ 1.x API
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered

            models = create_model_dict()
            converter = PdfConverter(artifact_dict=models)
            rendered = converter(str(pdf_path))
            text, _, _ = text_from_rendered(rendered)
            meta["marker_api"] = "marker>=1.x"
            return text, meta
        except ImportError:
            pass
        except Exception as exc:
            log.warning("Marker 1.x API failed: %s", exc)
            meta["marker_api_error"] = str(exc)

        try:
            # Marker 0.x API
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models

            model_lst = load_all_models()
            full_text, _images, _metadata = convert_single_pdf(str(pdf_path), model_lst)
            meta["marker_api"] = "marker<1.x"
            return full_text, meta
        except Exception as exc:
            log.warning("Marker 0.x API failed: %s — trying CLI", exc)
            meta["marker_python_error"] = str(exc)

        # Fall through to CLI
        if _marker_cli_available():
            return self._run_cli(pdf_path)

        return "", meta

    def _run_cli(self, pdf_path: Path) -> tuple[str, dict[str, Any]]:
        meta: dict[str, Any] = {"marker_mode": "cli"}
        cli_cmd = "marker_single" if shutil.which("marker_single") else "marker"

        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [cli_cmd, str(pdf_path), "--output_dir", tmp_dir]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            except subprocess.TimeoutExpired:
                meta["cli_error"] = "timeout (300s)"
                return "", meta
            except Exception as exc:
                meta["cli_error"] = str(exc)
                return "", meta

            if result.returncode != 0:
                meta["cli_returncode"] = result.returncode
                meta["cli_stderr"] = result.stderr[:500]
                return "", meta

            # Find generated markdown file
            md_files = list(Path(tmp_dir).rglob("*.md"))
            if not md_files:
                meta["cli_error"] = "no .md file produced"
                return "", meta

            text = md_files[0].read_text(encoding="utf-8", errors="replace")
            meta["cli_output_file"] = md_files[0].name
            return text, meta


# ---------------------------------------------------------------------------
# Markdown parsing helpers
# ---------------------------------------------------------------------------

def _parse_markdown_into_doc(markdown: str, paper_id: str, doc: ParsedDocument) -> None:
    """Parse Marker markdown output into text chunks, figures, and tables."""
    lines = markdown.splitlines()
    fig_idx = tbl_idx = chunk_idx = 0

    # -- Extract markdown table blocks first (remove them from body text) --
    tables_removed: list[tuple[int, int, str]] = []  # (start_line, end_line, md_text)
    i = 0
    table_line_indices: set[int] = set()
    while i < len(lines):
        if lines[i].startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            md_block = "\n".join(lines[i:j])
            tables_removed.append((i, j, md_block))
            for k in range(i, j):
                table_line_indices.add(k)
            i = j
        else:
            i += 1

    # Build table objects from extracted blocks
    # Marker sometimes puts captions before, sometimes after the table block.
    for start, _end, md_block in tables_removed:
        caption_text: str | None = None
        label: str | None = None
        # Check line immediately before the block
        if start > 0:
            prev = lines[start - 1].strip()
            m = _TBL_CAPTION.match(prev)
            if m:
                label = f"Table {m.group(2)}"
                caption_text = m.group(3).strip() or prev
        # If not found before, check line immediately after the block
        if label is None and _end < len(lines):
            nxt = lines[_end].strip()
            m = _TBL_CAPTION.match(nxt)
            if m:
                label = f"Table {m.group(2)}"
                caption_text = m.group(3).strip() or nxt

        retrieval_parts = []
        if label:
            retrieval_parts.append(label)
        if caption_text:
            retrieval_parts.append(f"Caption: {caption_text}")
        retrieval_parts.append(md_block[:1000])  # cap table content

        doc.tables.append(TableObject(
            id=f"{paper_id}_bench_tbl_{tbl_idx}",
            retrieval_text="\n".join(retrieval_parts),
            label=label,
            page=None,
            caption_text=caption_text,
            table_markdown=md_block,
            metadata={"marker_line_start": start},
        ))
        tbl_idx += 1

    # -- Scan non-table lines for figure captions and body text --
    body_lines: list[str] = []
    for i, line in enumerate(lines):
        if i in table_line_indices:
            continue

        # Check for figure caption
        m_fig = _FIG_CAPTION.match(line)
        if m_fig:
            label = f"Figure {m_fig.group(2)}"
            caption_text = m_fig.group(3).strip()
            retrieval_parts = [label]
            if caption_text:
                retrieval_parts.append(f"Caption: {caption_text}")
            doc.figures.append(FigureObject(
                id=f"{paper_id}_bench_fig_{fig_idx}",
                retrieval_text="\n".join(retrieval_parts),
                label=label,
                page=None,
                caption_text=caption_text or None,
                metadata={"marker_line": i},
            ))
            fig_idx += 1
            continue

        body_lines.append(line)

    # -- Chunk body text by word count --
    body_text = "\n".join(body_lines)
    words = body_text.split()
    for start in range(0, max(1, len(words)), _APPROX_CHUNK_WORDS):
        chunk_words = words[start: start + _APPROX_CHUNK_WORDS]
        text = " ".join(chunk_words).strip()
        if len(text) < _MIN_CHUNK_CHARS:
            continue
        doc.text_chunks.append(TextChunk(
            id=f"{paper_id}_bench_chunk_{chunk_idx}",
            text=text,
            char_len=len(text),
            metadata={"word_start": start, "source": "marker_markdown"},
        ))
        chunk_idx += 1
