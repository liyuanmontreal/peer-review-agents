"""Tests for Phase 4B sync layer and KoalaClient env-based config."""

import os
from unittest.mock import MagicMock, patch

import pytest

from gsr_agent.koala.client import KoalaClient, _DEFAULT_API_BASE
from gsr_agent.koala.models import Comment, Paper
from gsr_agent.koala.sync import (
    sync_active_papers,
    sync_all_active_state,
    sync_paper_comments,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper(paper_id: str = "p-001") -> Paper:
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    return Paper(
        paper_id=paper_id,
        title="Test Paper",
        open_time=now,
        review_end_time=now + timedelta(hours=48),
        verdict_end_time=now + timedelta(hours=72),
        state="REVIEW_ACTIVE",
    )


def _make_comment(comment_id: str = "c-001", paper_id: str = "p-001", author: str = "agent-x") -> Comment:
    from datetime import datetime, timezone
    return Comment(
        comment_id=comment_id,
        paper_id=paper_id,
        author_agent_id=author,
        text="Hello.",
        created_at=datetime.now(timezone.utc),
    )


def _make_client(papers=None, comments=None):
    client = MagicMock(spec=KoalaClient)
    client.list_active_papers.return_value = papers or []
    client.list_comments.return_value = comments or []
    return client


def _make_db():
    db = MagicMock()
    return db


# ---------------------------------------------------------------------------
# KoalaClient env-based config
# ---------------------------------------------------------------------------

def test_client_uses_koala_api_base_url_env(monkeypatch):
    monkeypatch.setenv("KOALA_API_BASE_URL", "https://test.koala.io/api/v1")
    monkeypatch.setenv("KOALA_API_TOKEN", "")
    client = KoalaClient(test_mode=True)
    assert client._base == "https://test.koala.io/api/v1"


def test_client_falls_back_to_default_base_when_env_not_set(monkeypatch):
    monkeypatch.delenv("KOALA_API_BASE_URL", raising=False)
    client = KoalaClient(test_mode=True)
    assert client._base == _DEFAULT_API_BASE.rstrip("/")


def test_client_explicit_api_base_overrides_env(monkeypatch):
    monkeypatch.setenv("KOALA_API_BASE_URL", "https://env.koala.io/api")
    client = KoalaClient(api_base="https://explicit.koala.io/api", test_mode=True)
    assert client._base == "https://explicit.koala.io/api"


def test_client_uses_koala_api_token_env(monkeypatch):
    monkeypatch.setenv("KOALA_API_TOKEN", "env-token-123")
    client = KoalaClient(test_mode=False)
    assert client._token == "env-token-123"


def test_client_explicit_token_overrides_env(monkeypatch):
    monkeypatch.setenv("KOALA_API_TOKEN", "env-token")
    client = KoalaClient(api_token="explicit-token", test_mode=False)
    assert client._token == "explicit-token"


def test_client_requires_token_without_env(monkeypatch):
    monkeypatch.delenv("KOALA_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="api_token"):
        KoalaClient(api_token="", test_mode=False)


# ---------------------------------------------------------------------------
# KoalaClient configurable endpoints
# ---------------------------------------------------------------------------

def test_ep_active_papers_uses_env(monkeypatch):
    monkeypatch.setenv("KOALA_ENDPOINT_ACTIVE_PAPERS", "/custom/papers/")
    client = KoalaClient(test_mode=True)
    assert client._ep_active_papers == "/custom/papers/"


def test_ep_active_papers_default(monkeypatch):
    monkeypatch.delenv("KOALA_ENDPOINT_ACTIVE_PAPERS", raising=False)
    client = KoalaClient(test_mode=True)
    assert client._ep_active_papers == "/papers/"


def test_ep_post_comment_uses_env(monkeypatch):
    monkeypatch.setenv("KOALA_ENDPOINT_POST_COMMENT", "/custom/comments/")
    client = KoalaClient(test_mode=True)
    assert client._ep_post_comment == "/custom/comments/"


def test_ep_post_comment_default(monkeypatch):
    monkeypatch.delenv("KOALA_ENDPOINT_POST_COMMENT", raising=False)
    client = KoalaClient(test_mode=True)
    assert client._ep_post_comment == "/comments/"


# ---------------------------------------------------------------------------
# KoalaClient retry/backoff
# ---------------------------------------------------------------------------

def test_retry_succeeds_after_transient_500():
    import json
    import urllib.error

    client = KoalaClient(api_token="tok", test_mode=False, _retry_delay_s=0)
    good_resp = MagicMock()
    good_resp.read.return_value = json.dumps([]).encode()
    good_resp.__enter__ = lambda s: s
    good_resp.__exit__ = MagicMock(return_value=False)

    err_500 = urllib.error.HTTPError(url="", code=500, msg="Error", hdrs=None, fp=None)
    err_500.read = lambda: b"error"

    call_count = [0]

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] < 2:
            raise err_500
        return good_resp

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.list_active_papers()

    assert result == []
    assert call_count[0] == 2  # one failure, one success


def test_retry_exhausts_all_attempts_on_persistent_500():
    import urllib.error
    from gsr_agent.koala.errors import KoalaAPIError

    client = KoalaClient(api_token="tok", test_mode=False, _retry_delay_s=0)
    err_500 = urllib.error.HTTPError(url="", code=500, msg="Error", hdrs=None, fp=None)
    err_500.read = lambda: b"error"

    call_count = [0]

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        raise err_500

    with patch("urllib.request.urlopen", fake_urlopen):
        with pytest.raises(KoalaAPIError):
            client.list_active_papers()

    assert call_count[0] == 3  # _MAX_RETRIES


# ---------------------------------------------------------------------------
# sync_active_papers
# ---------------------------------------------------------------------------

def test_sync_active_papers_calls_upsert_for_each():
    papers = [_make_paper("p-001"), _make_paper("p-002")]
    client = _make_client(papers=papers)
    db = _make_db()

    result = sync_active_papers(client, db)

    assert len(result) == 2
    assert db.upsert_paper.call_count == 2


def test_sync_active_papers_returns_empty_when_none():
    client = _make_client(papers=[])
    db = _make_db()
    result = sync_active_papers(client, db)
    assert result == []
    assert db.upsert_paper.call_count == 0


# ---------------------------------------------------------------------------
# sync_paper_comments
# ---------------------------------------------------------------------------

def test_sync_paper_comments_marks_ours_by_agent_id():
    our_comment = _make_comment("c-001", author="our-agent")
    other_comment = _make_comment("c-002", author="other-agent")
    client = _make_client(comments=[our_comment, other_comment])
    db = _make_db()

    sync_paper_comments(client, db, "p-001", agent_id="our-agent")

    upserted = [call.args[0] for call in db.upsert_comment.call_args_list]
    our = next(c for c in upserted if c.comment_id == "c-001")
    other = next(c for c in upserted if c.comment_id == "c-002")
    assert our.is_ours is True
    assert other.is_ours is False


def test_sync_paper_comments_uses_env_agent_id(monkeypatch):
    monkeypatch.setenv("KOALA_AGENT_ID", "env-agent")
    our_comment = _make_comment("c-001", author="env-agent")
    client = _make_client(comments=[our_comment])
    db = _make_db()

    sync_paper_comments(client, db, "p-001")

    upserted = db.upsert_comment.call_args_list[0].args[0]
    assert upserted.is_ours is True


def test_sync_paper_comments_no_agent_id_leaves_is_ours_false(monkeypatch):
    monkeypatch.delenv("KOALA_AGENT_ID", raising=False)
    comment = _make_comment("c-001", author="some-agent")
    client = _make_client(comments=[comment])
    db = _make_db()

    sync_paper_comments(client, db, "p-001")

    upserted = db.upsert_comment.call_args_list[0].args[0]
    assert upserted.is_ours is False


def test_sync_paper_comments_returns_count():
    comments = [_make_comment("c-001"), _make_comment("c-002"), _make_comment("c-003")]
    client = _make_client(comments=comments)
    db = _make_db()
    count = sync_paper_comments(client, db, "p-001")
    assert count == 3


# ---------------------------------------------------------------------------
# sync_all_active_state
# ---------------------------------------------------------------------------

def test_sync_all_active_state_returns_summary():
    papers = [_make_paper("p-001"), _make_paper("p-002")]
    client = _make_client(papers=papers, comments=[_make_comment()])
    db = _make_db()

    summary = sync_all_active_state(client, db)

    assert summary["papers"] == 2
    assert summary["comments"] == 2  # one comment per paper


def test_sync_all_active_state_empty():
    client = _make_client(papers=[], comments=[])
    db = _make_db()
    summary = sync_all_active_state(client, db)
    assert summary == {"papers": 0, "comments": 0}


# ---------------------------------------------------------------------------
# db.get_comment_stats
# ---------------------------------------------------------------------------

def test_get_comment_stats_returns_expected_counts(tmp_path):
    from datetime import datetime, timezone
    from gsr_agent.storage.db import KoalaDB
    from gsr_agent.koala.models import Comment

    db = KoalaDB(db_path=str(tmp_path / "test.db"))
    now = datetime.now(timezone.utc).isoformat()

    for i in range(3):
        db._conn.execute(
            "INSERT INTO koala_comments (comment_id, paper_id, author_agent_id, text, created_at, is_ours, is_citable) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"c-{i}", "p-001", "agent", "text", now, 0, 0),
        )
    # Add one "ours" citable
    db._conn.execute(
        "INSERT INTO koala_comments (comment_id, paper_id, author_agent_id, text, created_at, is_ours, is_citable) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("c-ours", "p-001", "us", "text", now, 1, 0),
    )
    # Add two citable others
    for i in range(2):
        db._conn.execute(
            "INSERT INTO koala_comments (comment_id, paper_id, author_agent_id, text, created_at, is_ours, is_citable) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"c-citable-{i}", "p-001", f"agent-{i}", "text", now, 0, 1),
        )
    db._conn.commit()

    stats = db.get_comment_stats("p-001")
    assert stats["total"] == 6
    assert stats["ours"] == 1
    assert stats["citable_other"] == 2
    db.close()


def test_get_comment_stats_returns_zeros_for_unknown_paper(tmp_path):
    from gsr_agent.storage.db import KoalaDB
    db = KoalaDB(db_path=str(tmp_path / "test.db"))
    stats = db.get_comment_stats("nonexistent")
    assert stats == {"total": 0, "ours": 0, "citable_other": 0}
    db.close()


# ---------------------------------------------------------------------------
# Audit patch: is_citable, synced_at, thread/parent refresh
# ---------------------------------------------------------------------------

def test_comment_from_api_reads_is_citable_true():
    from datetime import datetime, timezone
    data = {
        "id": "c-cit", "paper_id": "p-001",
        "author_id": "agent-a", "content_markdown": "Nice.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "is_citable": True,
    }
    c = Comment.from_api(data, "p-001")
    assert c.is_citable is True


def test_comment_from_api_reads_is_citable_false():
    from datetime import datetime, timezone
    data = {
        "id": "c-ncit", "paper_id": "p-001",
        "author_id": "agent-b", "content_markdown": "Hmm.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "is_citable": False,
    }
    c = Comment.from_api(data, "p-001")
    assert c.is_citable is False


def test_comment_from_api_defaults_is_citable_false_when_absent():
    from datetime import datetime, timezone
    data = {
        "id": "c-no-cit", "paper_id": "p-001",
        "author_id": "agent-c", "content_markdown": "Text.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    c = Comment.from_api(data, "p-001")
    assert c.is_citable is False


def test_upsert_comment_sets_synced_at(tmp_path):
    from gsr_agent.storage.db import KoalaDB
    db = KoalaDB(db_path=str(tmp_path / "test.db"))
    paper = _make_paper("p-001")
    db.upsert_paper(paper)
    db.upsert_comment(_make_comment("c-001", paper_id="p-001"))

    row = db._conn.execute(
        "SELECT synced_at FROM koala_comments WHERE comment_id='c-001'"
    ).fetchone()
    assert row is not None
    assert row["synced_at"] is not None
    db.close()


def test_upsert_comment_refreshes_thread_and_parent_on_conflict(tmp_path):
    from datetime import datetime, timezone
    from gsr_agent.storage.db import KoalaDB

    db = KoalaDB(db_path=str(tmp_path / "test.db"))
    paper = _make_paper("p-001")
    db.upsert_paper(paper)

    c = Comment(
        comment_id="c-001", paper_id="p-001", author_agent_id="ag",
        text="v1", created_at=datetime.now(timezone.utc),
        thread_id="t-old", parent_id="p-old",
    )
    db.upsert_comment(c)

    c2 = Comment(
        comment_id="c-001", paper_id="p-001", author_agent_id="ag",
        text="v1", created_at=datetime.now(timezone.utc),
        thread_id="t-new", parent_id="p-new",
    )
    db.upsert_comment(c2)

    row = db._conn.execute(
        "SELECT thread_id, parent_id FROM koala_comments WHERE comment_id='c-001'"
    ).fetchone()
    assert row["thread_id"] == "t-new"
    assert row["parent_id"] == "p-new"
    db.close()


def test_no_duplicate_rows_on_resync(tmp_path):
    from gsr_agent.storage.db import KoalaDB

    db = KoalaDB(db_path=str(tmp_path / "test.db"))
    paper = _make_paper("p-001")
    db.upsert_paper(paper)
    comment = _make_comment("c-001", paper_id="p-001")

    db.upsert_comment(comment)
    db.upsert_comment(comment)
    db.upsert_comment(comment)

    count = db._conn.execute(
        "SELECT COUNT(*) FROM koala_comments WHERE comment_id='c-001'"
    ).fetchone()[0]
    assert count == 1
    db.close()


def test_all_comments_stored_regardless_of_is_ours(tmp_path):
    from gsr_agent.storage.db import KoalaDB

    db = KoalaDB(db_path=str(tmp_path / "test.db"))
    paper = _make_paper("p-001")
    db.upsert_paper(paper)

    comments = [
        _make_comment("c-ours", paper_id="p-001", author="our-agent"),
        _make_comment("c-other", paper_id="p-001", author="other-agent"),
    ]
    client = _make_client(comments=comments)

    sync_paper_comments(client, db, "p-001", agent_id="our-agent")

    count = db._conn.execute(
        "SELECT COUNT(*) FROM koala_comments WHERE paper_id='p-001'"
    ).fetchone()[0]
    assert count == 2
    db.close()


def test_citable_other_nonzero_after_sync_with_citable_api_data(tmp_path):
    from datetime import datetime, timezone
    from gsr_agent.storage.db import KoalaDB

    db = KoalaDB(db_path=str(tmp_path / "test.db"))
    paper = _make_paper("p-001")
    db.upsert_paper(paper)

    now = datetime.now(timezone.utc)
    citable_comment = Comment(
        comment_id="c-cit", paper_id="p-001", author_agent_id="other-agent",
        text="Citable comment.", created_at=now, is_citable=True,
    )
    non_citable = Comment(
        comment_id="c-ncit", paper_id="p-001", author_agent_id="other-agent2",
        text="Non-citable.", created_at=now, is_citable=False,
    )
    client = _make_client(comments=[citable_comment, non_citable])

    sync_paper_comments(client, db, "p-001", agent_id="our-agent")

    stats = db.get_comment_stats("p-001")
    assert stats["total"] == 2
    assert stats["citable_other"] == 1
    db.close()


def test_upsert_comment_refreshes_text_on_conflict(tmp_path):
    from datetime import datetime, timezone
    from gsr_agent.storage.db import KoalaDB

    db = KoalaDB(db_path=str(tmp_path / "test.db"))
    paper = _make_paper("p-001")
    db.upsert_paper(paper)

    c1 = Comment(
        comment_id="c-001", paper_id="p-001", author_agent_id="ag",
        text="original text", created_at=datetime.now(timezone.utc),
    )
    db.upsert_comment(c1)

    c2 = Comment(
        comment_id="c-001", paper_id="p-001", author_agent_id="ag",
        text="updated text", created_at=datetime.now(timezone.utc),
    )
    db.upsert_comment(c2)

    row = db._conn.execute(
        "SELECT text FROM koala_comments WHERE comment_id='c-001'"
    ).fetchone()
    assert row["text"] == "updated text"
    db.close()
