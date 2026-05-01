"""Paper and comment sync helpers — pull from Koala API into local DB."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .client import KoalaClient
    from ..koala.models import Paper
    from ..storage.db import KoalaDB

log = logging.getLogger(__name__)


def sync_paper(client: "KoalaClient", db: "KoalaDB", paper_id: str) -> None:
    """Sync a single paper and its comments from the Koala API to local DB."""
    paper = client.get_paper(paper_id)
    if paper is None:
        log.warning("[sync_paper] paper not found: %s", paper_id)
        return
    db.upsert_paper(paper)
    for comment in client.list_comments(paper_id):
        db.upsert_comment(comment)
    log.debug("[sync_paper] synced paper=%s", paper_id)


def sync_all_papers(client: "KoalaClient", db: "KoalaDB") -> int:
    """Sync all active papers and their comments. Returns count of synced papers."""
    papers = client.list_active_papers()
    for paper in papers:
        db.upsert_paper(paper)
        for comment in client.list_comments(paper.paper_id):
            db.upsert_comment(comment)
    log.info("[sync_all_papers] synced %d papers", len(papers))
    return len(papers)


def sync_active_papers(client: "KoalaClient", db: "KoalaDB") -> List["Paper"]:
    """Fetch and persist all active papers. Returns the list of Paper objects."""
    papers = client.list_active_papers()
    for paper in papers:
        db.upsert_paper(paper)
    log.info("[sync_active_papers] synced %d active papers", len(papers))
    return papers


def sync_paper_comments(
    client: "KoalaClient",
    db: "KoalaDB",
    paper_id: str,
    *,
    agent_id: str = "",
) -> int:
    """Fetch and persist comments for *paper_id*. Returns count of comments synced.

    Comments authored by *agent_id* (or KOALA_AGENT_ID env) are marked
    ``is_ours=True``.
    """
    our_id = agent_id or os.environ.get("KOALA_AGENT_ID", "")
    comments = client.list_comments(paper_id)
    for comment in comments:
        if our_id and comment.author_agent_id == our_id:
            comment.is_ours = True
        comment.is_citable = not comment.is_ours and bool(comment.author_agent_id)
        db.upsert_comment(comment)
    citable_count = sum(1 for c in comments if c.is_citable)
    log.info(
        "[competition] synced_comments paper_id=%s count=%d citable_other=%d",
        paper_id, len(comments), citable_count,
    )
    return len(comments)


def sync_all_active_state(
    client: "KoalaClient",
    db: "KoalaDB",
    *,
    agent_id: str = "",
) -> dict:
    """Sync all active papers and their comments into the local DB.

    Returns a summary dict: {papers: N, comments: N}.
    """
    papers = sync_active_papers(client, db)
    total_comments = 0
    for paper in papers:
        total_comments += sync_paper_comments(client, db, paper.paper_id, agent_id=agent_id)
    log.info(
        "[sync_all_active_state] synced papers=%d comments=%d",
        len(papers), total_comments,
    )
    return {"papers": len(papers), "comments": total_comments}
