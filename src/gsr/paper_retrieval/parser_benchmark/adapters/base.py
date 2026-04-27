"""Abstract base class for all parser adapters."""
from __future__ import annotations

import abc
from pathlib import Path

from gsr.paper_retrieval.parser_benchmark.schema import ParsedDocument


class BaseParserAdapter(abc.ABC):
    """All adapters must implement this interface.

    Adapters should be stateless and re-entrant — a single instance may be
    called for multiple PDFs in the same benchmark run.
    """

    #: Human-readable name used in reports (e.g. "pymupdf", "docling")
    name: str = "base"

    @abc.abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Return (available, reason_if_not).

        ``reason_if_not`` is shown in the benchmark report when ``available``
        is False.  Return an empty string when available.
        """

    @abc.abstractmethod
    def parse(self, pdf_path: str | Path, paper_id: str) -> ParsedDocument:
        """Parse a PDF and return a unified ParsedDocument.

        Implementations must NOT raise for recoverable errors (e.g. missing
        optional sections).  Fatal errors (file not found, corrupt PDF) should
        propagate so the runner can record them.
        """

    # ------------------------------------------------------------------
    # Optional: adapters may override to perform one-time setup
    # ------------------------------------------------------------------

    def setup(self) -> None:  # noqa: B027
        """Called once before the first parse() call."""
