"""Semantic text embedding for evidence retrieval –  Module 3,Step 3.

Uses ``sentence-transformers`` with the SPECTER2 model, which is trained on
scientific paper abstracts and citations and therefore well-suited for
domain-accurate retrieval in ML/AI literature.

A lightweight fallback (``all-MiniLM-L6-v2``) is used automatically if the
SPECTER2 weights cannot be downloaded (e.g. in offline environments).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

# Default: SPECTER2 – fine-tuned on scientific text (Allen AI).
DEFAULT_EMBEDDING_MODEL = "allenai/specter2_base"

# Fallback: much smaller, general-purpose, no scientific fine-tuning.
_FALLBACK_MODEL = "all-MiniLM-L6-v2"

# Environment-variable override (mirrors the GSR_LLM_MODEL pattern).
_ENV_KEY = "GSR_EMBEDDING_MODEL"


def get_embedding_model_id(model_id: str | None = None) -> str:
    """Return the embedding model ID from argument, env-var, or default."""
    return model_id or os.environ.get(_ENV_KEY, DEFAULT_EMBEDDING_MODEL)


def load_embedding_model(model_id: str | None = None) -> tuple:
    """Load a SentenceTransformer model.

    Args:
        model_id: HuggingFace model ID.  Defaults to
            ``GSR_EMBEDDING_MODEL`` env-var or :data:`DEFAULT_EMBEDDING_MODEL`.

    Returns:
        ``(model, resolved_model_id)`` — a ready-to-use
        ``SentenceTransformer`` instance and the string identifier
        that was actually loaded.

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required: pip install sentence-transformers"
        ) from exc

    resolved = get_embedding_model_id(model_id)
    log.info("Loading embedding model: %s", resolved)

    try:
        model = SentenceTransformer(resolved)
        return model, resolved
    except Exception as exc:
        log.warning(
            "Could not load '%s' (%s); falling back to '%s'",
            resolved, exc, _FALLBACK_MODEL,
        )
        model = SentenceTransformer(_FALLBACK_MODEL)
        return model, _FALLBACK_MODEL


def embed_chunks(
    chunks: list[dict],
    model=None,
    model_id: str | None = None,
    *,
    batch_size: int = 64,
    show_progress: bool = False,
) -> tuple[list[list[float]], str]:
    """Compute embeddings for a list of chunk dicts.

    Args:
        chunks: Chunk dicts produced by :func:`~gsr.paper_retrieval.chunking.chunk_paper`.
            Each must have a ``"text"`` key.
        model: Pre-loaded ``SentenceTransformer`` model.  Loaded lazily if
            *None*.
        model_id: Model identifier string (used for DB storage).
        batch_size: Inference batch size.
        show_progress: Show a ``tqdm`` progress bar during encoding.

    Returns:
        ``(embeddings, resolved_model_id)`` where ``embeddings[i]`` is a
        ``list[float]`` for ``chunks[i]``.
    """
    if not chunks:
        resolved = get_embedding_model_id(model_id)
        return [], resolved

    if model is None:
        model, model_id = load_embedding_model(model_id)
    elif model_id is None:
        model_id = get_embedding_model_id()

    texts = [c["text"] for c in chunks]
    log.info("Embedding %d chunks with model '%s'", len(texts), model_id)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    return [v.tolist() for v in vectors], model_id


def embed_query(
    query: str,
    model,
    model_id: str | None = None,
) -> list[float]:
    """Embed a single query string for retrieval.

    Args:
        query: The text to embed (e.g. a claim string).
        model: Pre-loaded ``SentenceTransformer`` model.
        model_id: Unused; kept for API symmetry.

    Returns:
        Embedding as a plain Python list of floats.
    """
    vectors = model.encode([query], convert_to_numpy=True)
    return vectors[0].tolist()
