"""PDF Parser Benchmark Framework.

Compares multiple PDF parsing / chunking pipelines for the downstream task of
evidence-grounded scientific claim verification.

Entry point:
    from gsr.paper_retrieval.parser_benchmark.runner import BenchmarkRunner
    runner = BenchmarkRunner(output_dir="workspace/benchmarks/pdf_parsing/run1")
    runner.run(pdf_path="path/to/paper.pdf", paper_id="paper123")

CLI:
    gsr benchmark-parse --pdf path/to/paper.pdf
    gsr benchmark-parse --pdf-dir path/to/pdfs/
"""
from __future__ import annotations
