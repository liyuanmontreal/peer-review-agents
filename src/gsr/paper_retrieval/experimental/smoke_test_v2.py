from __future__ import annotations

import argparse
import re
import json
import sqlite3
from pathlib import Path

from ..parsing.parser import parse_paper_pdf_v2
from ..parsing.chunking import chunk_paper_v2_from_spans
from ..storage.storage import (
    init_retrieval_db,
    save_pdf_spans,
    save_chunks,
)
from ..retrieval import retrieve_evidence_for_claim
from gsr.reporting.annotate_pdf import annotate_pdf_with_evidence


def _connect_db(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))

def _safe_filename(name: str, max_len: int = 120) -> str:
    """
    Convert arbitrary string to a Windows-safe filename stem.
    Replaces illegal filename chars with '_'.
    """
    # Windows illegal chars: <>:"/\\|?*
    safe = re.sub(r'[<>:"/\\|?*]+', "_", name)
    safe = safe.strip(" .")
    if not safe:
        safe = "untitled"
    return safe[:max_len]

def _maybe_embed_and_save(
    *,
    chunks: list[dict],
    conn: sqlite3.Connection,
    model_id: str | None,
) -> None:
    """
    Optional embedding step.
    If sentence-transformers / your embedding module is ready, this will run.
    Otherwise it safely skips, and retrieval will fall back to BM25-only.
    """
    if not chunks:
        return

    if not model_id:
        print("[INFO] No model_id provided; skip embeddings. Retrieval will use BM25-only.")
        return

    try:
        from ..storage.embeddings import load_embedding_model, embed_texts
    except Exception as exc:
        print(f"[WARN] Embedding module unavailable, skip embeddings: {exc}")
        return

    try:
        from ..storage.storage import save_embeddings
    except Exception as exc:
        print(f"[WARN] save_embeddings unavailable, skip embeddings: {exc}")
        return

    print(f"[INFO] Loading embedding model: {model_id}")
    model, resolved_model_id = load_embedding_model(model_id)

    texts = [c["text"] for c in chunks]
    print(f"[INFO] Embedding {len(texts)} chunks ...")
    embeddings = embed_texts(texts, model, resolved_model_id)

    save_embeddings(chunks, embeddings, resolved_model_id, conn)
    print(f"[INFO] Saved embeddings with model_id={resolved_model_id}")


def run_smoke_test_v2(
    *,
    pdf_path: str | Path,
    db_path: str | Path,
    paper_id: str,
    claim_text: str,
    output_dir: str | Path,
    top_k: int = 3,
    chunk_size: int = 8,
    chunk_overlap: int = 2,
    model_id: str | None = None,
) -> None:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_paper_id = _safe_filename(paper_id)

    annotated_pdf_path = output_dir / f"{safe_paper_id}_annotated.pdf"
    retrieval_json_path = output_dir / f"{safe_paper_id}_retrieval_results.json"


    conn = _connect_db(db_path)
    try:
        print("[STEP 1] init_retrieval_db")
        init_retrieval_db(conn)

        print("[STEP 2] parse_paper_pdf_v2")
        parsed = parse_paper_pdf_v2(pdf_path, paper_id)
        spans = parsed["spans"]
        sections = parsed["sections"]

        print(f"  pages    = {parsed['n_pages']}")
        print(f"  spans    = {len(spans)}")
        print(f"  sections = {len(sections)}")

        print("[STEP 3] save_pdf_spans")
        save_pdf_spans(spans, conn)

        print("[STEP 4] chunk_paper_v2_from_spans")
        chunks = chunk_paper_v2_from_spans(
            spans,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        print(f"  chunks   = {len(chunks)}")

        if chunks:
            print("  first chunk preview:")
            print("  ----------------------------------------")
            print(chunks[0]["text"][:500])
            print("  ----------------------------------------")
            print(f"  first chunk span_ids[:5] = {chunks[0].get('span_ids', [])[:5]}")

        print("[STEP 5] save_chunks")
        save_chunks(chunks, conn, chunk_size=chunk_size)

        print("[STEP 6] optional embeddings")
        _maybe_embed_and_save(
            chunks=chunks,
            conn=conn,
            model_id=model_id,
        )

        claim = {
            "id": f"{paper_id}_smoke_claim",
            "claim_text": claim_text,
        }

        print("[STEP 7] retrieve_evidence_for_claim")
        results = retrieve_evidence_for_claim(
            claim=claim,
            paper_id=paper_id,
            conn=conn,
            top_k=top_k,
            model_id=model_id,
        )

        print(f"  retrieved = {len(results)}")
        for i, r in enumerate(results, start=1):
            print(f"\n  [{i}] chunk_id={r['chunk_id']}")
            print(f"      section={r.get('section')}")
            print(f"      page={r.get('page')} page_start={r.get('page_start')} page_end={r.get('page_end')}")
            print(f"      bm25={r['bm25_score']:.4f} semantic={r['semantic_score']:.4f} combined={r['combined_score']:.4f}")
            print(f"      span_ids[:5]={ (r.get('span_ids') or [])[:5] }")
            preview = r["text"].replace("\n", " ")
            print(f"      text={preview[:300]}")

        print("[STEP 8] save retrieval results json")
        retrieval_json_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  wrote: {retrieval_json_path}")

        if results:
            print("[STEP 9] annotate_pdf_with_evidence")
            annotate_pdf_with_evidence(
                pdf_path=pdf_path,
                output_path=annotated_pdf_path,
                paper_id=paper_id,
                evidence_items=results,
                conn=conn,
            )
            print(f"  wrote: {annotated_pdf_path}")
        else:
            print("[WARN] No retrieval results, skip PDF annotation.")

    finally:
        conn.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a V2 end-to-end smoke test.")
    p.add_argument("--pdf", required=True, help="Path to input paper PDF")
    p.add_argument("--db", required=True, help="Path to sqlite DB (e.g. workspace/gsr.db)")
    p.add_argument("--paper-id", required=True, help="Paper ID")
    p.add_argument("--claim", required=True, help="Claim text used for retrieval")
    p.add_argument("--out-dir", required=True, help="Directory for outputs")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=8, help="Chunk size in spans")
    p.add_argument("--chunk-overlap", type=int, default=2, help="Chunk overlap in spans")
    p.add_argument("--model-id", default=None, help="Optional embedding model id")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run_smoke_test_v2(
        pdf_path=args.pdf,
        db_path=args.db,
        paper_id=args.paper_id,
        claim_text=args.claim,
        output_dir=args.out_dir,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()