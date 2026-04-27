"""Internal normalized document representation.

This module defines the intermediate data types that sit between raw parser
output (PyMuPDF V2, Docling) and our pipeline's evidence objects.

Design principles:
- PyMuPDF spans are always the grounding layer (bbox, page coordinates).
- Higher-level parsers (Docling) enrich the structure (tables, reading order).
- Normalized objects must map cleanly back to PyMuPDF span IDs for UI red-boxes.
- This layer is internal: consumers should work with EvidenceObject, not these.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedSection:
    id: str
    heading: str
    section_number: str | None
    page: int
    text: str
    span_ids: list[str] = field(default_factory=list)


@dataclass
class NormalizedTable:
    """A table object extracted from the paper.

    content_text is the retrieval-friendly representation:
    - From Docling: markdown or flattened row text
    - From caption_extractor fallback: caption + nearby lines
    """
    id: str
    label: str | None           # "Table 1", "Table 2.1", etc.
    caption: str | None
    page: int
    section: str | None
    section_number: str | None
    content_text: str           # for retrieval / verification
    bbox: list[float] | None
    span_ids: list[str] = field(default_factory=list)
    source_parser: str = "pymupdf"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedFigure:
    """A figure object extracted from the paper.

    For v1 (text-only verification), content_text = caption + nearby discussion.
    For future multimodal: asset_path would hold the extracted image.
    """
    id: str
    label: str | None           # "Figure 1", "Fig. 3", etc.
    caption: str | None
    page: int
    section: str | None
    section_number: str | None
    content_text: str           # caption + nearby discussion (text-based v1)
    bbox: list[float] | None
    span_ids: list[str] = field(default_factory=list)
    asset_path: str | None = None   # future: extracted image file
    source_parser: str = "pymupdf"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedDocument:
    """Normalized representation of a parsed paper.

    Always built from PyMuPDF V2 spans (grounding layer).
    Optionally enriched by Docling (better tables, reading order).

    Downstream consumers (evidence_builder, chunking) should read from this
    rather than calling parser-specific functions directly.
    """
    paper_id: str
    pdf_path: str
    n_pages: int
    source_parser: str          # "pymupdf" / "docling+pymupdf"

    # PyMuPDF V2 spans — always present, source of truth for bboxes / grounding
    spans: list[dict[str, Any]] = field(default_factory=list)

    # Section structure (from whichever parser ran)
    sections: list[NormalizedSection] = field(default_factory=list)

    # Structured table and figure objects (enriched by Docling when available)
    tables: list[NormalizedTable] = field(default_factory=list)
    figures: list[NormalizedFigure] = field(default_factory=list)

    # Parser-specific metadata for debugging
    parser_metadata: dict[str, Any] = field(default_factory=dict)
