"""Metrics computation for benchmark runs.

Takes a ParsedDocument and computes a flat metrics dictionary suitable for
CSV summary rows and Markdown report tables.
"""
from __future__ import annotations

import statistics
from typing import Any

from gsr.paper_retrieval.parser_benchmark.schema import ParsedDocument


def compute_metrics(
    doc: ParsedDocument,
    *,
    parse_runtime_sec: float,
    parser_available: bool = True,
    parser_error: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Return a flat metrics dict for one parser/document pair.

    All values are JSON-serialisable (int, float, str, bool, None).
    """
    m: dict[str, Any] = {
        # Identity
        "paper_id": doc.paper_id,
        "parser_name": doc.parser_name,
        "pdf_path": doc.pdf_path,
        # Runtime
        "parse_runtime_sec": round(parse_runtime_sec, 3),
        "parser_available": parser_available,
        "parser_error": parser_error,
        "notes": notes,
    }

    # ------------------------------------------------------------------
    # Text chunk metrics
    # ------------------------------------------------------------------
    chunks = doc.text_chunks
    m["num_text_chunks"] = len(chunks)

    if chunks:
        char_lens = [c.char_len for c in chunks]
        m["avg_chunk_chars"] = round(statistics.mean(char_lens), 1)
        m["median_chunk_chars"] = round(statistics.median(char_lens), 1)
        m["min_chunk_chars"] = min(char_lens)
        m["max_chunk_chars"] = max(char_lens)
        m["total_text_chars"] = sum(char_lens)
        m["empty_or_tiny_chunks_count"] = sum(1 for l in char_lens if l < 40)
        with_section = sum(1 for c in chunks if c.section)
        m["chunks_with_section_count"] = with_section
        m["chunks_with_section_rate"] = round(with_section / len(chunks), 4) if chunks else 0.0
    else:
        m["avg_chunk_chars"] = 0.0
        m["median_chunk_chars"] = 0.0
        m["min_chunk_chars"] = 0
        m["max_chunk_chars"] = 0
        m["total_text_chars"] = 0
        m["empty_or_tiny_chunks_count"] = 0
        m["chunks_with_section_count"] = 0
        m["chunks_with_section_rate"] = 0.0

    # ------------------------------------------------------------------
    # Table metrics
    # ------------------------------------------------------------------
    tables = doc.tables
    m["num_tables"] = len(tables)
    m["tables_with_label_count"] = sum(1 for t in tables if t.label)
    m["tables_with_caption_count"] = sum(1 for t in tables if t.caption_text)
    m["tables_with_nonempty_markdown_count"] = sum(
        1 for t in tables if t.table_markdown and len(t.table_markdown.strip()) > 0
    )
    if tables:
        md_lens = [
            len(t.table_markdown) for t in tables if t.table_markdown
        ]
        m["avg_table_markdown_chars"] = round(statistics.mean(md_lens), 1) if md_lens else 0.0
    else:
        m["avg_table_markdown_chars"] = 0.0

    # ------------------------------------------------------------------
    # Figure metrics
    # ------------------------------------------------------------------
    figures = doc.figures
    m["num_figures"] = len(figures)
    m["figures_with_label_count"] = sum(1 for f in figures if f.label)
    m["figures_with_caption_count"] = sum(1 for f in figures if f.caption_text)
    m["figures_with_bbox_count"] = sum(1 for f in figures if f.bbox_json)
    m["figures_with_image_path_count"] = sum(1 for f in figures if f.image_path)

    return m


def aggregate_metrics(
    per_doc_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-document metrics into a parser-level summary.

    For multi-PDF runs, averages numeric fields across all successfully
    parsed documents.
    """
    if not per_doc_metrics:
        return {}

    numeric_keys = [
        "parse_runtime_sec",
        "num_text_chunks",
        "avg_chunk_chars",
        "median_chunk_chars",
        "min_chunk_chars",
        "max_chunk_chars",
        "total_text_chars",
        "empty_or_tiny_chunks_count",
        "chunks_with_section_count",
        "chunks_with_section_rate",
        "num_tables",
        "tables_with_label_count",
        "tables_with_caption_count",
        "tables_with_nonempty_markdown_count",
        "avg_table_markdown_chars",
        "num_figures",
        "figures_with_label_count",
        "figures_with_caption_count",
        "figures_with_bbox_count",
        "figures_with_image_path_count",
    ]

    agg: dict[str, Any] = {
        "parser_name": per_doc_metrics[0]["parser_name"],
        "num_pdfs": len(per_doc_metrics),
        "num_errors": sum(1 for m in per_doc_metrics if m.get("parser_error")),
    }

    for key in numeric_keys:
        vals = [m[key] for m in per_doc_metrics if key in m and m[key] is not None]
        if vals:
            agg[f"mean_{key}"] = round(statistics.mean(vals), 3)
        else:
            agg[f"mean_{key}"] = None

    return agg
