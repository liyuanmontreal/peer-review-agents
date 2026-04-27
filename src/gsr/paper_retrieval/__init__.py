"""Paper retrieval – Module 3.

Retrieve evidence chunks from papers using a two-stage pipeline:

1. **PDF parsing** (PyMuPDF)  – extract section-structured text.
2. **Chunking**               – split into overlapping character windows.
3. **Embedding** (SPECTER2)   – encode chunks with a scientific LM.
4. **Indexing**               – persist chunks + embeddings to SQLite.
5. **Retrieval** (BM25 + cosine) – rank chunks for a given claim.

Typical usage::

    import sqlite3
    from gsr.paper_retrieval import index_paper, retrieve_evidence_for_claim
    from gsr.data_collection.storage import init_db
    from gsr.paper_retrieval.storage import init_retrieval_db

    conn = init_db()
    init_retrieval_db(conn)

    # Index one paper (parse → chunk → embed → save)
    summary = index_paper(
        paper_id="abc123",
        pdf_path="/data/pdf/ICLR.cc/abc123.pdf",
        conn=conn,
    )

    # Retrieve evidence for a claim
    claim = {"id": "...", "claim_text": "The model achieves 95% accuracy on CIFAR-10."}
    evidence = retrieve_evidence_for_claim(claim, paper_id="abc123", conn=conn)
"""
from __future__ import annotations

import logging
import sqlite3

log = logging.getLogger(__name__)


from .parsing.chunking import chunk_paper, chunk_paper_v2_from_spans
from .storage.embeddings import embed_chunks, load_embedding_model
from .evidence.evidence_builder import build_evidence_objects_for_paper
from .parsing.parser import parse_paper_pdf, parse_paper_pdf_v2
from .retrieval import (
    retrieve_evidence_for_claim,
    retrieve_evidence_for_claims,
    retrieve_all_claims_for_paper,
)
from .storage.storage import (
    get_embedded_paper_ids,
    get_indexed_paper_ids,
    init_retrieval_db,
    save_chunks,
    save_embeddings,
    save_evidence_embeddings,
    save_evidence_objects,
    save_pdf_spans,
    save_retrieval_results,
)

__all__ = [
    # Steps
    "parse_paper_pdf",
    "parse_paper_pdf_v2",
    "chunk_paper",
    "chunk_paper_v2_from_spans",
    "embed_chunks",
    "load_embedding_model",
    "build_evidence_objects_for_paper",
    "index_paper",
    "index_all_papers",
    # Retrieval
    "retrieve_evidence_for_claim",
    "retrieve_evidence_for_claims",
    "retrieve_all_claims_for_paper",
    # Storage helpers
    "init_retrieval_db",
    "save_chunks",
    "save_embeddings",
    "save_evidence_embeddings",
    "save_evidence_objects",
    "save_pdf_spans",
    "save_retrieval_results",
    "get_indexed_paper_ids",
    "get_embedded_paper_ids",
]


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def index_paper(
    paper_id: str,
    pdf_path: str,
    conn: sqlite3.Connection,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embed: bool = True,
    model=None,
    model_id: str | None = None,
    layout_aware: bool = False,
    use_docling: bool = False,
    use_figure_ocr: bool = False,
    use_bbox_refine: bool = False,
    bbox_detector: str = "groundingdino",
) -> dict:
    """Parse, chunk, embed, and persist a single paper.

    Parsing modes:

    - V1 (default, layout_aware=False):
        Plain-text section parsing + character-window chunking.
        No PDF spans, no figure/table evidence objects.

    - V2 (layout_aware=True):
        PyMuPDF span-level parsing + span-window chunking + pdf_spans persistence.
        Builds text_chunk / table / figure evidence objects via caption_extractor.
        This is the recommended mode.

    - V2 + Docling (layout_aware=True, use_docling=True):
        Same as V2 but additionally runs Docling for better table structure and
        figure extraction. Falls back to V2-only if Docling is not installed or fails.

    Args:
        paper_id: Must match a row in ``papers.id``.
        pdf_path: Absolute path to the downloaded PDF.
        conn: Open SQLite connection (must have retrieval tables created via
            :func:`~gsr.paper_retrieval.storage.init_retrieval_db`).
        chunk_size: Target chunk size.
            - V1: characters
            - V2: spans (approx line units)
        chunk_overlap: Overlap between consecutive chunks.
            - V1: characters
            - V2: spans
        embed: Whether to compute and store embeddings.
        model: Pre-loaded ``SentenceTransformer`` model; loaded lazily if *None*.
        model_id: HuggingFace model ID string.
        layout_aware: If True, use the V2 layout-aware pipeline.
        use_docling: If True (and layout_aware=True), run Docling for enriched
            table/figure extraction. Requires ``pip install docling``.

    Returns:
        Summary dict with keys: paper_id, layout_aware, source_parser, n_sections,
        n_spans, n_chunks, n_evidence_objects, embedded, model_id.
    """

    spans: list | None = None
    docling_tables: list | None = None
    source_parser = "pymupdf_v1"
    refine_summary: dict = {"bbox_refine_enabled": False}

    if layout_aware:
        # Signal: parse/chunk phase starting (1 1 = this phase is the unit of work)
        print("GSR_PROGRESS index_prepare 1 1", flush=True)
        if use_docling:
            # PyMuPDF V2 only at index time — Docling is deferred to late enrichment
            # (enrich_topk_tables_with_docling) and runs only for top-k table candidates
            # before verification. Full-document Docling at index time is too slow.
            from .parsing.parse_router import route as _route
            norm_doc = _route(pdf_path, paper_id, prefer_docling=False)
            spans = norm_doc.spans
            source_parser = norm_doc.source_parser
            n_sections = len(norm_doc.sections)
            # docling_tables intentionally left None — tables get content_text=""
            # at build time and are enriched lazily before verify.
        else:
            parsed = parse_paper_pdf_v2(pdf_path, paper_id)
            spans = parsed["spans"]
            source_parser = "pymupdf_v2"
            n_sections = len(parsed.get("sections", []))

        save_pdf_spans(spans, conn)

        chunks = chunk_paper_v2_from_spans(
            spans,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            docling_doc=norm_doc if use_docling else None,
        )

        n_spans = len(spans)
    else:
        parsed = parse_paper_pdf(pdf_path, paper_id)
        chunks = chunk_paper(
            parsed,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        n_sections = len(parsed["sections"])
        n_spans = None

    save_chunks(chunks, conn, chunk_size=chunk_size)

    # Build evidence objects (text_chunk + figure + table) when span data is available.
    # This is the primary evidence layer consumed by retrieval and verification.
    evidence_objects = []
    if layout_aware and spans is not None:
        # Signal: parse + chunk complete; evidence object construction starting
        print("GSR_PROGRESS index_evidence 1 1", flush=True)
        evidence_objects = build_evidence_objects_for_paper(
            paper_id=paper_id,
            chunks=chunks,
            spans=spans,
            docling_tables=docling_tables,
            pdf_path=pdf_path,
            use_figure_ocr=use_figure_ocr,
        )
        save_evidence_objects(evidence_objects, conn)
        log.info(
            "[index] OBJECTS paper_id=%s objects=%d text_chunk=%d table=%d figure=%d",
            paper_id,
            len(evidence_objects),
            sum(1 for e in evidence_objects if e.object_type == "text_chunk"),
            sum(1 for e in evidence_objects if e.object_type == "table"),
            sum(1 for e in evidence_objects if e.object_type == "figure"),
        )

        # Phase P1.5: optional GroundingDINO bbox refinement.
        # Runs after evidence objects are saved so identity is stable.
        # Updates metadata_json only — no identity/retrieval changes.
        if use_bbox_refine and pdf_path:
            from .vision.bbox_refinement import refine_bbox_for_paper
            refine_summary = refine_bbox_for_paper(paper_id, pdf_path, conn, detector=bbox_detector)
            log.info(
                "[bbox-refine] paper='%s' attempted=%d accepted=%d rejected=%d",
                paper_id,
                refine_summary.get("bbox_refine_attempted", 0),
                refine_summary.get("bbox_refine_accepted", 0),
                refine_summary.get("bbox_refine_rejected", 0),
            )
        else:
            refine_summary = {"bbox_refine_enabled": False}

    actual_model_id: str | None = None
    did_embed = False

    if embed and chunks:
        # Signal: evidence objects ready; embedding starting (typically the slowest phase)
        print("GSR_PROGRESS index_embed 1 1", flush=True)
        if model is None:
            model, model_id = load_embedding_model(model_id)

        # Skip legacy chunk_embeddings when evidence_embeddings will be generated.
        # Retrieval always prefers evidence_embeddings when present; chunk_embeddings
        # are only needed as a fallback for non-layout-aware or evidence-object-free runs.
        skip_chunk_embed = layout_aware and bool(evidence_objects)

        # Unconditional decision log — proves this patched code path executed and
        # shows the exact values that drove the skip decision.
        log.info(
            "[embed] DECISION paper='%s' layout_aware=%s evidence_objects=%d "
            "chunks=%d skip_chunk_embed=%s",
            paper_id,
            layout_aware,
            len(evidence_objects),
            len(chunks),
            skip_chunk_embed,
        )

        if skip_chunk_embed:
            log.info(
                "[embed] SKIP chunk_embeddings reason=using_evidence_embeddings "
                "evidence_objects=%d",
                len(evidence_objects),
            )
        else:
            reason = (
                "layout_aware=False"
                if not layout_aware
                else "evidence_objects_empty"
            )
            log.info(
                "[embed] RUN chunk_embeddings reason=%s layout_aware=%s chunks=%d",
                reason,
                layout_aware,
                len(chunks),
            )
            embeddings, actual_model_id = embed_chunks(
                chunks, model=model, model_id=model_id
            )
            save_embeddings(chunks, embeddings, actual_model_id, conn)

        did_embed = True

        # Embed evidence objects and store in evidence_embeddings.
        # Retrieval prefers evidence_embeddings over chunk_embeddings when present.
        if evidence_objects:
            ev_texts = [{"text": ev.retrieval_text} for ev in evidence_objects]
            ev_embeddings, actual_model_id = embed_chunks(
                ev_texts, model=model, model_id=actual_model_id or model_id
            )
            ev_dicts = [{"id": ev.id} for ev in evidence_objects]
            save_evidence_embeddings(ev_dicts, ev_embeddings, actual_model_id, conn)
            log.info(
                "[embed] evidence_embeddings paper='%s' count=%d model=%s",
                paper_id,
                len(ev_embeddings),
                actual_model_id,
            )

    return {
        "paper_id": paper_id,
        "layout_aware": layout_aware,
        "source_parser": source_parser,
        "n_sections": n_sections,
        "n_spans": n_spans,
        "n_chunks": len(chunks),
        "n_evidence_objects": len(evidence_objects),
        "embedded": did_embed,
        "model_id": actual_model_id,
        **refine_summary,
    }

def index_all_papers(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    paper_id: str | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embed: bool = True,
    model_id: str | None = None,
    force: bool = False,
    layout_aware: bool = False,
    use_docling: bool = False,
    use_figure_ocr: bool = False,
    use_bbox_refine: bool = False,
    bbox_detector: str = "groundingdino",
) -> dict:

    """Index all papers that have a downloaded PDF.

    Queries ``papers`` for rows with a non-null ``pdf_path``, skips any
    that are already indexed (unless *force* is ``True``), and calls
    :func:`index_paper` for each remaining paper.

    Args:
        conn: Open SQLite connection with retrieval tables initialised.
        limit: Maximum number of papers to process.
        paper_id: If given, process only this paper.
        chunk_size: Chunk size in characters.
        chunk_overlap: Overlap in characters.
        embed: Whether to compute embeddings.
        model_id: Embedding model to use.
        force: Re-index papers that already have chunks.

    Returns:
        Summary dict::

            {
                "papers_processed": int,
                "total_chunks":     int,
                "errors":           int,
                "results":          list[dict],
            }
    """
    # Fetch papers with PDFs from the DB.
    query = "SELECT id, pdf_path FROM papers WHERE pdf_path IS NOT NULL"
    params: list = []
    if paper_id:
        query += " AND id = ?"
        params.append(paper_id)
    if limit:
        query += f" LIMIT {int(limit)}"

    cur = conn.execute(query, params)
    rows = cur.fetchall()

    already_indexed = get_indexed_paper_ids(conn) if not force else set()

    # Load embedding model once to reuse across all papers.
    emb_model = None
    if embed:
        try:
            emb_model, model_id = load_embedding_model(model_id)
        except Exception as exc:
            log.warning("Could not load embedding model (%s); disabling embed.", exc)
            embed = False

    results: list[dict] = []
    errors = 0
    total_chunks = 0

    for pid, pdf_path in rows:
        if pid in already_indexed and not force:
            log.debug("Skipping already-indexed paper '%s'.", pid)
            continue

        try:
            summary = index_paper(
                pid,
                pdf_path,
                conn,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embed=embed,
                model=emb_model,
                model_id=model_id,
                layout_aware=layout_aware,
                use_docling=use_docling,
                use_figure_ocr=use_figure_ocr,
                use_bbox_refine=use_bbox_refine,
                bbox_detector=bbox_detector,
            )
            results.append(summary)
            total_chunks += summary["n_chunks"]

            log.info(
                "[index] DONE paper_id=%s chunks=%d evidence_objects=%d source_parser=%s embedded=%s",
                pid,
                summary["n_chunks"],
                summary.get("n_evidence_objects", 0),
                summary.get("source_parser", "unknown"),
                summary["embedded"],
            )
        except Exception as exc:
            log.warning("Failed to index paper '%s': %s", pid, exc)
            errors += 1

    return {
        "papers_processed": len(results),
        "total_chunks": total_chunks,
        "errors": errors,
        "results": results,
    }
