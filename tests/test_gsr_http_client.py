"""Tests for gsr_agent.koala.client — KoalaClient HTTP and test_mode."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from gsr_agent.koala.client import KoalaClient
from gsr_agent.koala.errors import KoalaAPIError, KoalaPreflightError, KoalaRateLimitError
from gsr_agent.koala.models import Comment, Paper

UTC = timezone.utc
_REAL_URL = "https://github.com/owner/repo/blob/main/logs/paper-001/artifact.md"

_PAPER_API_DICT = {
    "id": "paper-001",
    "title": "Test Paper",
    "status": "in_review",
    "open_time": "2026-04-24T12:00:00Z",
    "pdf_url": "https://example.com/paper.pdf",
    "abstract": "This paper proposes a new method.",
    "full_text": "",
    "domains": ["ML", "NLP"],
}

_COMMENT_API_DICT = {
    "id": "comment-abc",
    "paper_id": "paper-001",
    "author_id": "agent-x",
    "content_markdown": "Good point about the ablation.",
    "created_at": "2026-04-24T14:00:00Z",
    "thread_id": None,
    "parent_id": None,
}


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_requires_token_in_production():
    with pytest.raises(ValueError, match="api_token"):
        KoalaClient(api_token="", test_mode=False)


def test_accepts_empty_token_in_test_mode():
    client = KoalaClient(test_mode=True)
    assert client._test_mode is True


def test_production_client_stores_token():
    client = KoalaClient(api_token="tok-abc", test_mode=False)
    assert client._token == "tok-abc"


# ---------------------------------------------------------------------------
# test_mode read stubs
# ---------------------------------------------------------------------------

def test_list_active_papers_returns_empty_in_test_mode():
    client = KoalaClient(test_mode=True)
    assert client.list_active_papers() == []


def test_get_paper_returns_none_in_test_mode():
    client = KoalaClient(test_mode=True)
    assert client.get_paper("paper-001") is None


def test_list_comments_returns_empty_in_test_mode():
    client = KoalaClient(test_mode=True)
    assert client.list_comments("paper-001") == []


# ---------------------------------------------------------------------------
# test_mode write stubs
# ---------------------------------------------------------------------------

def test_post_comment_returns_stub_id_in_test_mode():
    client = KoalaClient(test_mode=True)
    cid = client.post_comment("paper-001", "body", _REAL_URL)
    assert isinstance(cid, str)
    assert len(cid) > 0


def test_post_comment_test_mode_rejects_placeholder_url():
    client = KoalaClient(test_mode=True)
    with pytest.raises(KoalaPreflightError, match="github_file_url"):
        client.post_comment("paper-001", "body", "TODO: placeholder")


def test_post_comment_test_mode_rejects_empty_url():
    client = KoalaClient(test_mode=True)
    with pytest.raises(KoalaPreflightError, match="github_file_url"):
        client.post_comment("paper-001", "body", "")


def test_submit_verdict_returns_true_in_test_mode():
    client = KoalaClient(test_mode=True)
    result = client.submit_verdict("paper-001", 7.0, ["c1", "c2", "c3"], _REAL_URL)
    assert result is True


def test_submit_verdict_test_mode_rejects_placeholder_url():
    client = KoalaClient(test_mode=True)
    with pytest.raises(KoalaPreflightError):
        client.submit_verdict("paper-001", 7.0, [], "TODO: placeholder")


# ---------------------------------------------------------------------------
# Paper.from_api
# ---------------------------------------------------------------------------

def test_paper_from_api_parses_id_and_title():
    paper = Paper.from_api(_PAPER_API_DICT)
    assert paper.paper_id == "paper-001"
    assert paper.title == "Test Paper"


def test_paper_from_api_maps_in_review_to_review_active():
    paper = Paper.from_api(_PAPER_API_DICT)
    assert paper.state == "REVIEW_ACTIVE"


def test_paper_from_api_maps_deliberating_to_verdict_active():
    d = {**_PAPER_API_DICT, "status": "deliberating"}
    paper = Paper.from_api(d)
    assert paper.state == "VERDICT_ACTIVE"


def test_paper_from_api_maps_reviewed_to_expired():
    d = {**_PAPER_API_DICT, "status": "reviewed"}
    paper = Paper.from_api(d)
    assert paper.state == "EXPIRED"


def test_paper_from_api_maps_closed_to_expired():
    d = {**_PAPER_API_DICT, "status": "closed"}
    paper = Paper.from_api(d)
    assert paper.state == "EXPIRED"


def test_paper_from_api_parses_open_time():
    paper = Paper.from_api(_PAPER_API_DICT)
    assert paper.open_time.tzinfo is not None
    assert paper.open_time.year == 2026


def test_paper_from_api_falls_back_to_opened_at():
    d = {k: v for k, v in _PAPER_API_DICT.items() if k != "open_time"}
    d["opened_at"] = "2026-04-24T12:00:00Z"
    paper = Paper.from_api(d)
    assert paper.open_time.year == 2026


def test_paper_from_api_populates_content_fields():
    paper = Paper.from_api(_PAPER_API_DICT)
    assert paper.abstract == "This paper proposes a new method."
    assert paper.domains == ["ML", "NLP"]


def test_paper_from_api_defaults_missing_fields():
    d = {"id": "p1", "title": "T", "status": "in_review", "open_time": "2026-04-24T12:00:00Z"}
    paper = Paper.from_api(d)
    assert paper.abstract == ""
    assert paper.domains == []


# ---------------------------------------------------------------------------
# Comment.from_api
# ---------------------------------------------------------------------------

def test_comment_from_api_parses_fields():
    c = Comment.from_api(_COMMENT_API_DICT, "paper-001")
    assert c.comment_id == "comment-abc"
    assert c.paper_id == "paper-001"
    assert c.author_agent_id == "agent-x"
    assert c.text == "Good point about the ablation."


def test_comment_from_api_parses_created_at():
    c = Comment.from_api(_COMMENT_API_DICT, "paper-001")
    assert c.created_at.tzinfo is not None
    assert c.created_at.year == 2026


# ---------------------------------------------------------------------------
# _request — production HTTP (mocked)
# ---------------------------------------------------------------------------

def _make_mock_response(data: dict, status: int = 200):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_list_active_papers_never_sends_status_active():
    """list_active_papers must not request status=active (Koala API returns 422)."""
    client = KoalaClient(api_token="tok", test_mode=False)
    captured_urls: list[str] = []

    def fake_urlopen(req, timeout=None):
        captured_urls.append(req.full_url)
        return _make_mock_response([])

    with patch("urllib.request.urlopen", fake_urlopen):
        client.list_active_papers()

    for url in captured_urls:
        assert "status=active" not in url, f"Found forbidden status=active in {url}"


def test_list_active_papers_fetches_in_review_and_deliberating():
    """list_active_papers must request exactly in_review and deliberating."""
    client = KoalaClient(api_token="tok", test_mode=False)
    captured_urls: list[str] = []

    def fake_urlopen(req, timeout=None):
        captured_urls.append(req.full_url)
        return _make_mock_response([])

    with patch("urllib.request.urlopen", fake_urlopen):
        client.list_active_papers()

    statuses = {url.split("status=")[1].split("&")[0] for url in captured_urls if "status=" in url}
    assert statuses == {"in_review", "deliberating"}


def test_list_active_papers_dedupes_by_paper_id():
    """Papers returned by both status queries are deduplicated by paper_id."""
    client = KoalaClient(api_token="tok", test_mode=False)
    paper_a = {**_PAPER_API_DICT, "id": "paper-001", "status": "in_review"}
    paper_b = {**_PAPER_API_DICT, "id": "paper-002", "status": "deliberating"}

    responses = iter([
        _make_mock_response([paper_a, paper_b]),   # in_review call
        _make_mock_response([paper_a]),              # deliberating call (paper_a duplicate)
    ])

    with patch("urllib.request.urlopen", lambda req, timeout=None: next(responses)):
        papers = client.list_active_papers()

    assert len(papers) == 2
    ids = {p.paper_id for p in papers}
    assert ids == {"paper-001", "paper-002"}


def test_list_active_papers_parses_paginated_response():
    client = KoalaClient(api_token="tok", test_mode=False)
    resp_data = {"results": [_PAPER_API_DICT], "count": 1}

    responses = iter([_make_mock_response(resp_data), _make_mock_response([])])
    with patch("urllib.request.urlopen", lambda req, timeout=None: next(responses)):
        papers = client.list_active_papers()
    assert len(papers) == 1


def test_get_paper_returns_paper():
    client = KoalaClient(api_token="tok", test_mode=False)
    mock_resp = _make_mock_response(_PAPER_API_DICT)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        paper = client.get_paper("paper-001")
    assert paper is not None
    assert paper.paper_id == "paper-001"


def test_get_paper_returns_none_on_404():
    client = KoalaClient(api_token="tok", test_mode=False)
    exc = urllib.error.HTTPError(url="", code=404, msg="Not Found", hdrs=None, fp=None)
    exc.read = lambda: b""
    with patch("urllib.request.urlopen", side_effect=exc):
        result = client.get_paper("paper-missing")
    assert result is None


def test_request_raises_rate_limit_on_429():
    client = KoalaClient(api_token="tok", test_mode=False, _retry_delay_s=0)
    exc = urllib.error.HTTPError(url="", code=429, msg="Too Many", hdrs=None, fp=None)
    exc.read = lambda: b"rate limited"
    with patch("urllib.request.urlopen", side_effect=exc):
        with pytest.raises(KoalaRateLimitError):
            client.list_active_papers()


def test_request_raises_api_error_on_500():
    client = KoalaClient(api_token="tok", test_mode=False, _retry_delay_s=0)
    exc = urllib.error.HTTPError(url="", code=500, msg="Error", hdrs=None, fp=None)
    exc.read = lambda: b"server error"
    with patch("urllib.request.urlopen", side_effect=exc):
        with pytest.raises(KoalaAPIError):
            client.list_active_papers()


def test_post_comment_sends_correct_fields():
    client = KoalaClient(api_token="tok", test_mode=False)
    mock_resp = _make_mock_response({"id": "new-comment-id"})
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        captured["method"] = req.method
        return mock_resp

    with patch("urllib.request.urlopen", fake_urlopen):
        cid = client.post_comment("paper-001", "My comment.", _REAL_URL)

    assert cid == "new-comment-id"
    assert captured["method"] == "POST"
    assert captured["body"]["paper_id"] == "paper-001"
    assert captured["body"]["content_markdown"] == "My comment."
    assert captured["body"]["github_file_url"] == _REAL_URL


def test_post_comment_raises_preflight_on_placeholder():
    client = KoalaClient(api_token="tok", test_mode=False)
    with pytest.raises(KoalaPreflightError, match="github_file_url"):
        client.post_comment("paper-001", "body", "TODO: placeholder")


def test_post_comment_raises_api_error_when_response_has_no_id():
    client = KoalaClient(api_token="tok", test_mode=False)
    mock_resp = _make_mock_response({})
    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(KoalaAPIError, match="comment id"):
            client.post_comment("paper-001", "body", _REAL_URL)


def test_authorization_header_sent():
    client = KoalaClient(api_token="Bearer my-token", test_mode=False)
    mock_resp = _make_mock_response([])
    captured_headers = {}

    def fake_urlopen(req, timeout=None):
        captured_headers.update(dict(req.headers))
        return mock_resp

    with patch("urllib.request.urlopen", fake_urlopen):
        client.list_active_papers()

    assert "Authorization" in captured_headers
    assert captured_headers["Authorization"] == "Bearer my-token"


# ---------------------------------------------------------------------------
# list_comments — KOALA_COMMENTS_LIMIT env var (Phase 5A.5 patch)
# ---------------------------------------------------------------------------

def test_list_comments_default_limit_is_500(monkeypatch):
    """When KOALA_COMMENTS_LIMIT is unset, the request uses limit=500."""
    monkeypatch.delenv("KOALA_COMMENTS_LIMIT", raising=False)
    client = KoalaClient(api_token="tok", test_mode=False)
    captured_params = {}

    def fake_urlopen(req, timeout=None):
        from urllib.parse import urlparse, parse_qs
        captured_params.update(parse_qs(urlparse(req.full_url).query))
        return _make_mock_response([])

    with patch("urllib.request.urlopen", fake_urlopen):
        client.list_comments("paper-001")

    assert captured_params.get("limit") == ["500"]


def test_list_comments_env_limit_overrides_default(monkeypatch):
    """KOALA_COMMENTS_LIMIT env var is used when set."""
    monkeypatch.setenv("KOALA_COMMENTS_LIMIT", "250")
    client = KoalaClient(api_token="tok", test_mode=False)
    captured_params = {}

    def fake_urlopen(req, timeout=None):
        from urllib.parse import urlparse, parse_qs
        captured_params.update(parse_qs(urlparse(req.full_url).query))
        return _make_mock_response([])

    with patch("urllib.request.urlopen", fake_urlopen):
        client.list_comments("paper-001")

    assert captured_params.get("limit") == ["250"]


def test_list_comments_explicit_limit_kwarg_overrides_env(monkeypatch):
    """Explicit limit= kwarg takes precedence over env var."""
    monkeypatch.setenv("KOALA_COMMENTS_LIMIT", "999")
    client = KoalaClient(api_token="tok", test_mode=False)
    captured_params = {}

    def fake_urlopen(req, timeout=None):
        from urllib.parse import urlparse, parse_qs
        captured_params.update(parse_qs(urlparse(req.full_url).query))
        return _make_mock_response([])

    with patch("urllib.request.urlopen", fake_urlopen):
        client.list_comments("paper-001", limit=42)

    assert captured_params.get("limit") == ["42"]


def test_list_comments_test_mode_unchanged(monkeypatch):
    """test_mode still returns empty list regardless of limit."""
    monkeypatch.setenv("KOALA_COMMENTS_LIMIT", "500")
    client = KoalaClient(test_mode=True)
    assert client.list_comments("paper-001") == []
    assert client.list_comments("paper-001", limit=1000) == []
