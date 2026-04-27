"""Claim verification – Module 4.

RAG / entailment-style verification of extracted claims against evidence
retrieved from the paper being reviewed.

Pipeline position
-----------------
Module 2 (claim_extraction) → **Module 4 (claim_verification)** ← Module 3 (paper_retrieval)

For each verifiable claim the module:

1. Looks up cached evidence chunks in ``retrieval_results`` (Module 3 output).
2. Falls back to live retrieval via
   :func:`~gsr.paper_retrieval.retrieval.retrieve_evidence_for_claim` if no
   cache exists.
3. Sends claim + evidence to an LLM and receives a structured
   :class:`~gsr.claim_verification.prompts.VerificationResponse`.
4. Persists the verdict, reasoning, and confidence to
   ``verification_results`` in ``gsr.db``.

Typical usage::

    from gsr.data_collection.storage import init_db
    from gsr.claim_verification import (
        init_verification_db,
        verify_all_claims,
        save_verification_results,
    )

    conn = init_db()
    init_verification_db(conn)

    summary = verify_all_claims(conn, paper_id="abc123", top_k=5)
    save_verification_results(summary["results"], conn)

    print(summary["verdicts"])
    # {'supported': 12, 'refuted': 3, 'insufficient_evidence': 7}
"""
from __future__ import annotations

from gsr.claim_verification.prompts import (
    Verdict,
    VerificationResponse,
    build_messages,
    format_evidence_block,
)
from gsr.claim_verification.storage import (
    get_verdict_summary,
    init_verification_db,
    load_verification_results,
    save_verification_results,
)
from gsr.claim_verification.verifier import (
    verify_all_claims,
    verify_claim,
)

__all__ = [
    # Prompts & models
    "Verdict",
    "VerificationResponse",
    "build_messages",
    "format_evidence_block",
    # Verification
    "verify_claim",
    "verify_all_claims",
    # Storage
    "init_verification_db",
    "save_verification_results",
    "load_verification_results",
    "get_verdict_summary",
]
