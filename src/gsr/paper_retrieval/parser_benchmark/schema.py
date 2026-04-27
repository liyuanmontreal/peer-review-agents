"""Unified intermediate representation for parser benchmark output.

All parser adapters must return a ParsedDocument containing TextChunk,
FigureObject, and TableObject instances.  This schema is intentionally
aligned with the GSR EvidenceObject model so that benchmark outputs can
be directly mapped into the production pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Leaf types
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    id: str
    text: str
    char_len: int
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    bbox_json: str | None = None       # JSON-serialised [x0, y0, x1, y1]
    span_ids_json: str | None = None   # JSON-serialised list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chunk_dict(cls, d: dict[str, Any], paper_id: str, idx: int) -> "TextChunk":
        """Construct from a gsr chunking.py V2 chunk dict."""
        text = d.get("text", "")
        bbox = d.get("bbox")
        span_ids = d.get("span_ids")
        return cls(
            id=d.get("id") or f"{paper_id}_bench_chunk_{idx}",
            text=text,
            char_len=len(text),
            page_start=d.get("page_start") or d.get("page"),
            page_end=d.get("page_end") or d.get("page"),
            section=d.get("section"),
            bbox_json=json.dumps(bbox) if bbox is not None else None,
            span_ids_json=json.dumps(span_ids) if span_ids is not None else None,
            metadata=d.get("metadata") or {},
        )


@dataclass
class FigureObject:
    id: str
    retrieval_text: str
    label: str | None = None
    page: int | None = None
    caption_text: str | None = None
    bbox_json: str | None = None
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableObject:
    id: str
    retrieval_text: str
    label: str | None = None
    page: int | None = None
    caption_text: str | None = None
    table_markdown: str | None = None
    bbox_json: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level document
# ---------------------------------------------------------------------------

@dataclass
class ParsedDocument:
    paper_id: str
    parser_name: str
    pdf_path: str
    text_chunks: list[TextChunk] = field(default_factory=list)
    figures: list[FigureObject] = field(default_factory=list)
    tables: list[TableObject] = field(default_factory=list)
    pages: dict[str, Any] = field(default_factory=dict)   # optional page metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d
