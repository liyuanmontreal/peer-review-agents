"""Benchmark runner.

Orchestrates parser adapters over one or more PDFs, collects metrics, and
writes output artefacts.

Usage:
    runner = BenchmarkRunner(output_dir="workspace/benchmarks/pdf_parsing/run1")
    runner.run_single(pdf_path="paper.pdf", paper_id="my_paper")
    runner.run_dir(pdf_dir="path/to/pdfs/")
"""
from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from gsr.paper_retrieval.parser_benchmark.schema import ParsedDocument
from gsr.paper_retrieval.parser_benchmark.evaluator import compute_metrics
from gsr.paper_retrieval.parser_benchmark.report import (
    write_parsed_doc_json,
    write_summary_csv,
    write_markdown_report,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

def _load_adapters(requested: list[str] | None = None):
    """Instantiate and return all (or selected) adapters in priority order."""
    from gsr.paper_retrieval.parser_benchmark.adapters.pymupdf_adapter import PyMuPDFAdapter
    from gsr.paper_retrieval.parser_benchmark.adapters.docling_adapter import DoclingAdapter
    from gsr.paper_retrieval.parser_benchmark.adapters.marker_adapter import MarkerAdapter
    from gsr.paper_retrieval.parser_benchmark.adapters.llamaparse_adapter import LlamaParseAdapter

    all_adapters = [
        PyMuPDFAdapter(),
        DoclingAdapter(),
        MarkerAdapter(),
        LlamaParseAdapter(),
    ]

    if requested:
        requested_lower = {r.lower() for r in requested}
        all_adapters = [a for a in all_adapters if a.name in requested_lower]

    return all_adapters


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Run the PDF parser benchmark and write results to output_dir.

    Args:
        output_dir: Directory to write JSON / CSV / Markdown artefacts.
        parsers: Optional list of parser names to run.  Defaults to all.
        verbose: If True, emit per-chunk debug lines.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        parsers: list[str] | None = None,
        verbose: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapters = _load_adapters(parsers)
        self.verbose = verbose

        # Resolved at check time
        self._availability: dict[str, tuple[bool, str]] = {}
        self._all_metrics: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single(
        self,
        pdf_path: str | Path,
        paper_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Benchmark all parsers on a single PDF.  Returns list of metric dicts."""
        pdf_path = Path(pdf_path)
        if paper_id is None:
            paper_id = pdf_path.stem

        log.info("=== Benchmark: %s (paper_id=%s) ===", pdf_path.name, paper_id)
        metrics_for_pdf: list[dict[str, Any]] = []

        for adapter in self.adapters:
            avail, reason = adapter.is_available()
            self._availability[adapter.name] = (avail, reason)

            if not avail:
                log.info("  [%s] SKIPPED — %s", adapter.name, reason)
                # Record as unavailable row
                empty_doc = ParsedDocument(
                    paper_id=paper_id,
                    parser_name=adapter.name,
                    pdf_path=str(pdf_path),
                )
                m = compute_metrics(
                    empty_doc,
                    parse_runtime_sec=0.0,
                    parser_available=False,
                    parser_error=None,
                    notes=reason,
                )
                metrics_for_pdf.append(m)
                self._all_metrics.append(m)
                continue

            log.info("  [%s] running ...", adapter.name)
            t0 = time.perf_counter()
            doc: ParsedDocument | None = None
            error_str: str | None = None

            try:
                adapter.setup()
                doc = adapter.parse(pdf_path, paper_id)
            except Exception as exc:
                error_str = f"{type(exc).__name__}: {exc}"
                log.warning("  [%s] FAILED — %s", adapter.name, error_str)
                if self.verbose:
                    traceback.print_exc()
                doc = ParsedDocument(
                    paper_id=paper_id,
                    parser_name=adapter.name,
                    pdf_path=str(pdf_path),
                    metadata={"error": error_str},
                )

            runtime = time.perf_counter() - t0
            log.info(
                "  [%s] done — %.2fs  chunks=%d  tables=%d  figures=%d",
                adapter.name,
                runtime,
                len(doc.text_chunks),
                len(doc.tables),
                len(doc.figures),
            )

            # Write per-parser JSON
            try:
                json_path = write_parsed_doc_json(doc, self.output_dir / "parsed")
                log.debug("  [%s] wrote %s", adapter.name, json_path)
            except Exception as exc:
                log.warning("  [%s] failed to write JSON: %s", adapter.name, exc)

            m = compute_metrics(
                doc,
                parse_runtime_sec=runtime,
                parser_available=True,
                parser_error=error_str,
            )
            metrics_for_pdf.append(m)
            self._all_metrics.append(m)

        return metrics_for_pdf

    def run_dir(self, pdf_dir: str | Path) -> list[dict[str, Any]]:
        """Benchmark all PDFs in a directory."""
        pdf_dir = Path(pdf_dir)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            log.warning("No PDF files found in %s", pdf_dir)
            return []

        log.info("Found %d PDF(s) in %s", len(pdf_files), pdf_dir)
        all_metrics: list[dict[str, Any]] = []
        for pdf_path in pdf_files:
            metrics = self.run_single(pdf_path)
            all_metrics.extend(metrics)

        return all_metrics

    def write_reports(self, run_id: str = "") -> dict[str, Path]:
        """Write CSV + Markdown reports from accumulated metrics.

        Returns dict of {report_type: path}.
        """
        if not self._all_metrics:
            log.warning("No metrics collected — skipping report generation")
            return {}

        paths: dict[str, Path] = {}

        csv_path = write_summary_csv(self._all_metrics, self.output_dir)
        log.info("CSV summary: %s", csv_path)
        paths["csv"] = csv_path

        md_path = write_markdown_report(
            self._all_metrics,
            self._availability,
            self.output_dir,
            run_id=run_id,
        )
        log.info("Markdown report: %s", md_path)
        paths["markdown"] = md_path

        return paths


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_benchmark(
    *,
    pdf_path: str | Path | None = None,
    pdf_dir: str | Path | None = None,
    output_dir: str | Path,
    parsers: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, Path]:
    """One-shot benchmark runner.  Returns report paths dict."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / run_id
    runner = BenchmarkRunner(output_dir, parsers=parsers, verbose=verbose)

    if pdf_path:
        runner.run_single(pdf_path)
    elif pdf_dir:
        runner.run_dir(pdf_dir)
    else:
        raise ValueError("Provide either pdf_path or pdf_dir")

    return runner.write_reports(run_id=run_id)
