"""Tests for reva.env — environment accessors."""
from reva.env import koala_base_url, koala_github_repo, koala_write_enabled


def test_koala_base_url_default(monkeypatch):
    monkeypatch.delenv("KOALA_BASE_URL", raising=False)
    assert koala_base_url() == "https://koala.science"


def test_koala_base_url_override(monkeypatch):
    monkeypatch.setenv("KOALA_BASE_URL", "https://staging.koala.science")
    assert koala_base_url() == "https://staging.koala.science"


def test_koala_base_url_empty_string_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("KOALA_BASE_URL", "")
    assert koala_base_url() == "https://koala.science"


def test_koala_base_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("KOALA_BASE_URL", "https://staging.koala.science/")
    assert koala_base_url() == "https://staging.koala.science"


def test_koala_base_url_strips_multiple_trailing_slashes(monkeypatch):
    monkeypatch.setenv("KOALA_BASE_URL", "https://staging.koala.science///")
    assert koala_base_url() == "https://staging.koala.science"


# ── koala_write_enabled ────────────────────────────────────────────────────────

def test_koala_write_enabled_default_is_false(monkeypatch):
    monkeypatch.delenv("KOALA_WRITE_ENABLED", raising=False)
    assert koala_write_enabled() is False


def test_koala_write_enabled_false_string(monkeypatch):
    monkeypatch.setenv("KOALA_WRITE_ENABLED", "false")
    assert koala_write_enabled() is False


def test_koala_write_enabled_empty_string(monkeypatch):
    monkeypatch.setenv("KOALA_WRITE_ENABLED", "")
    assert koala_write_enabled() is False


def test_koala_write_enabled_true_string(monkeypatch):
    monkeypatch.setenv("KOALA_WRITE_ENABLED", "true")
    assert koala_write_enabled() is True


def test_koala_write_enabled_true_uppercase(monkeypatch):
    monkeypatch.setenv("KOALA_WRITE_ENABLED", "TRUE")
    assert koala_write_enabled() is True


def test_koala_write_enabled_true_mixed_case(monkeypatch):
    monkeypatch.setenv("KOALA_WRITE_ENABLED", "True")
    assert koala_write_enabled() is True


# ── koala_github_repo ──────────────────────────────────────────────────────────

def test_koala_github_repo_default_is_empty(monkeypatch):
    monkeypatch.delenv("KOALA_GITHUB_REPO", raising=False)
    assert koala_github_repo() == ""


def test_koala_github_repo_honors_env(monkeypatch):
    monkeypatch.setenv("KOALA_GITHUB_REPO", "https://github.com/owner/repo")
    assert koala_github_repo() == "https://github.com/owner/repo"
