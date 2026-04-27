"""Report generation for benchmark runs.

Produces:
- per-parser JSON files   (full ParsedDocument serialisation)
- one combined CSV summary
- one Markdown report
"""
from __future__ import annotations

import csv
import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

from gsr.paper_retrieval.parser_benchmark.schema import ParsedDocument


# ---------------------------------------------------------------------------
# Per-parser JSON
# ---------------------------------------------------------------------------

def write_parsed_doc_json(doc: ParsedDocument, output_dir: Path) -> Path:
    """Serialise a ParsedDocument to a JSON file in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{doc.paper_id}__{doc.parser_name}.json"
    out_path = output_dir / fname
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Combined CSV summary
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "paper_id",
    "parser_name",
    "parse_runtime_sec",
    "parser_available",
    "parser_error",
    # text chunks
    "num_text_chunks",
    "avg_chunk_chars",
    "median_chunk_chars",
    "min_chunk_chars",
    "max_chunk_chars",
    "total_text_chars",
    "empty_or_tiny_chunks_count",
    "chunks_with_section_count",
    "chunks_with_section_rate",
    # tables
    "num_tables",
    "tables_with_label_count",
    "tables_with_caption_count",
    "tables_with_nonempty_markdown_count",
    "avg_table_markdown_chars",
    # figures
    "num_figures",
    "figures_with_label_count",
    "figures_with_caption_count",
    "figures_with_bbox_count",
    "figures_with_image_path_count",
    "notes",
]


def write_summary_csv(
    all_metrics: list[dict[str, Any]],
    output_dir: Path,
    filename: str = "summary.csv",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in all_metrics:
            writer.writerow({k: row.get(k, "") for k in _CSV_COLUMNS})
    return out_path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown_report(
    all_metrics: list[dict[str, Any]],
    availability: dict[str, tuple[bool, str]],
    output_dir: Path,
    run_id: str = "",
    filename: str = "report.md",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    lines = _build_report_lines(all_metrics, availability, run_id)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _build_report_lines(
    all_metrics: list[dict[str, Any]],
    availability: dict[str, tuple[bool, str]],
    run_id: str,
) -> list[str]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    lines.append("# PDF Parser Benchmark Report")
    lines.append("")
    if run_id:
        lines.append(f"**Run ID:** `{run_id}`  ")
    lines.append(f"**Generated:** {now}")
    lines.append("")

    # ── Parser availability ──────────────────────────────────────────────
    lines.append("## Parser Availability")
    lines.append("")
    lines.append("| Parser | Available | Notes |")
    lines.append("|--------|-----------|-------|")
    for parser_name, (avail, reason) in sorted(availability.items()):
        status = "✓" if avail else "✗"
        lines.append(f"| {parser_name} | {status} | {reason or '—'} |")
    lines.append("")

    if not all_metrics:
        lines.append("*No benchmark results to display.*")
        return lines

    # ── Summary table ────────────────────────────────────────────────────
    lines.append("## Summary Table")
    lines.append("")

    # Group by paper_id → parser_name
    paper_ids = list(dict.fromkeys(m["paper_id"] for m in all_metrics))
    parser_names = list(dict.fromkeys(m["parser_name"] for m in all_metrics))

    for paper_id in paper_ids:
        rows = [m for m in all_metrics if m["paper_id"] == paper_id]
        lines.append(f"### Paper: `{paper_id}`")
        lines.append("")
        lines.append("#### Text Chunks")
        lines.append("")
        lines.append(
            "| Parser | # Chunks | Avg Chars | Median Chars | "
            "Total Chars | Section Rate | Tiny (<40) | Runtime (s) |"
        )
        lines.append(
            "|--------|----------|-----------|--------------|"
            "-------------|--------------|------------|-------------|"
        )
        for m in rows:
            if not m.get("parser_available"):
                lines.append(
                    f"| {m['parser_name']} | *unavailable* | — | — | — | — | — | — |"
                )
                continue
            if m.get("parser_error"):
                lines.append(
                    f"| {m['parser_name']} | *error* | — | — | — | — | — | — |"
                )
                continue
            lines.append(
                f"| {m['parser_name']} "
                f"| {m.get('num_text_chunks', 0)} "
                f"| {m.get('avg_chunk_chars', 0):.0f} "
                f"| {m.get('median_chunk_chars', 0):.0f} "
                f"| {m.get('total_text_chars', 0):,} "
                f"| {m.get('chunks_with_section_rate', 0):.1%} "
                f"| {m.get('empty_or_tiny_chunks_count', 0)} "
                f"| {m.get('parse_runtime_sec', 0):.2f} |"
            )
        lines.append("")
        lines.append("#### Tables")
        lines.append("")
        lines.append(
            "| Parser | # Tables | w/ Label | w/ Caption | w/ Markdown | Avg MD Chars |"
        )
        lines.append(
            "|--------|----------|----------|------------|-------------|--------------|"
        )
        for m in rows:
            if not m.get("parser_available") or m.get("parser_error"):
                continue
            lines.append(
                f"| {m['parser_name']} "
                f"| {m.get('num_tables', 0)} "
                f"| {m.get('tables_with_label_count', 0)} "
                f"| {m.get('tables_with_caption_count', 0)} "
                f"| {m.get('tables_with_nonempty_markdown_count', 0)} "
                f"| {m.get('avg_table_markdown_chars', 0):.0f} |"
            )
        lines.append("")
        lines.append("#### Figures")
        lines.append("")
        lines.append(
            "| Parser | # Figures | w/ Label | w/ Caption | w/ BBox | w/ Image |"
        )
        lines.append(
            "|--------|-----------|----------|------------|---------|----------|"
        )
        for m in rows:
            if not m.get("parser_available") or m.get("parser_error"):
                continue
            lines.append(
                f"| {m['parser_name']} "
                f"| {m.get('num_figures', 0)} "
                f"| {m.get('figures_with_label_count', 0)} "
                f"| {m.get('figures_with_caption_count', 0)} "
                f"| {m.get('figures_with_bbox_count', 0)} "
                f"| {m.get('figures_with_image_path_count', 0)} |"
            )
        lines.append("")

    # ── Parser notes / limitations ───────────────────────────────────────
    lines.append("## Parser Notes")
    lines.append("")
    parser_notes = {
        "pymupdf": textwrap.dedent("""\
            - **Grounding:** Full span-level bbox grounding (used by UI red-boxes).
            - **Text:** V2 span-based chunking with P1.7 boundary rules.
            - **Tables:** Caption-first identity; no markdown table content.
            - **Figures:** Caption-first identity; bbox from caption span region.
            - **Limitation:** Table content_text is caption + region text, not structured rows.
        """),
        "docling": textwrap.dedent("""\
            - **Grounding:** PyMuPDF V2 spans are still the bbox source of truth.
            - **Text:** V2 span-based chunking + Docling P1.7b boundary hints.
            - **Tables:** Docling structured markdown extraction (best table quality).
            - **Figures:** Docling figure detection with best-effort page/bbox.
            - **Limitation:** Docling may fail on some PDFs; falls back to PyMuPDF only.
        """),
        "marker": textwrap.dedent("""\
            - **Grounding:** No span-level bbox (Marker outputs plain markdown).
            - **Text:** Word-based sliding window over Marker markdown body text.
            - **Tables:** Extracted from Markdown table blocks; no structured rows.
            - **Figures:** Caption regex detection only; no bbox data.
            - **Limitation:** Page numbers not reliably available. No PDF coordinate grounding.
        """),
        "llamaparse": textwrap.dedent("""\
            - **Grounding:** No span-level bbox (cloud API returns markdown).
            - **Text:** Word-based sliding window over LlamaParse markdown output.
            - **Tables / Figures:** Caption regex + markdown table block detection.
            - **Limitation:** Requires internet + LLAMA_CLOUD_API_KEY. Rate-limited.
              Not suitable for offline/local-only workflows.
        """),
    }

    for parser_name in parser_names:
        note = parser_notes.get(parser_name, "No notes available for this parser.")
        lines.append(f"### {parser_name}")
        lines.append("")
        lines.append(note)
        lines.append("")

    return lines
