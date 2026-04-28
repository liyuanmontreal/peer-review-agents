-- gsr_agent v4 competition schema
-- Separate from the koala-gsr-agent legacy schema; these tables track the
-- rule-hardened state introduced in Phase 1–3.

CREATE TABLE IF NOT EXISTS koala_papers (
    paper_id            TEXT PRIMARY KEY,
    title               TEXT NOT NULL DEFAULT '',
    abstract            TEXT NOT NULL DEFAULT '',
    open_time           TEXT NOT NULL,                  -- ISO-8601 UTC
    review_end_time     TEXT NOT NULL,                  -- ISO-8601 UTC
    verdict_end_time    TEXT NOT NULL,                  -- ISO-8601 UTC
    state               TEXT NOT NULL DEFAULT 'NEW',    -- NEW | REVIEW_ACTIVE | VERDICT_ACTIVE | EXPIRED
    pdf_url             TEXT NOT NULL DEFAULT '',
    local_pdf_path      TEXT,
    last_synced_at      TEXT NOT NULL                   -- ISO-8601 UTC
);

CREATE TABLE IF NOT EXISTS koala_comments (
    comment_id          TEXT PRIMARY KEY,
    paper_id            TEXT NOT NULL,
    thread_id           TEXT,
    parent_id           TEXT,
    author_agent_id     TEXT NOT NULL DEFAULT '',
    text                TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL,                  -- ISO-8601 UTC
    is_ours             INTEGER NOT NULL DEFAULT 0,
    is_citable          INTEGER NOT NULL DEFAULT 0,
    synced_at           TEXT,                              -- ISO-8601 UTC; NULL for rows pre-dating this column
    FOREIGN KEY (paper_id) REFERENCES koala_papers(paper_id)
);

-- Append-only audit log of every external action attempted.
CREATE TABLE IF NOT EXISTS koala_agent_actions (
    action_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id            TEXT NOT NULL,
    action_type         TEXT NOT NULL,                  -- comment | thread | verdict | skip
    external_id         TEXT,                           -- ID returned by Koala API on success
    github_file_url     TEXT,
    created_at          TEXT NOT NULL,                  -- ISO-8601 UTC
    status              TEXT NOT NULL DEFAULT 'pending', -- pending | success | failed | dry_run
    error_message       TEXT,
    details             TEXT                            -- JSON: run_mode, artifact_mode, is_test_url, etc.
);

-- Immutable karma ledger — one row per karma-consuming action.
CREATE TABLE IF NOT EXISTS koala_karma_ledger (
    ledger_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id            TEXT NOT NULL,
    action_type         TEXT NOT NULL,                  -- comment | thread | verdict
    cost                REAL NOT NULL,
    karma_before        REAL NOT NULL,
    karma_after         REAL NOT NULL,
    created_at          TEXT NOT NULL                   -- ISO-8601 UTC
);

-- Phase 5A: claims extracted from other-agent citable comments.
CREATE TABLE IF NOT EXISTS koala_extracted_claims (
    claim_id            TEXT PRIMARY KEY,
    comment_id          TEXT NOT NULL,
    paper_id            TEXT NOT NULL,
    claim_text          TEXT NOT NULL,
    category            TEXT,
    confidence          REAL,
    challengeability    REAL,
    binary_question     TEXT,
    created_at          TEXT NOT NULL
);

-- Phase 5A: GSR verification results for extracted claims.
CREATE TABLE IF NOT EXISTS koala_claim_verifications (
    verification_id     TEXT PRIMARY KEY,
    claim_id            TEXT NOT NULL,
    comment_id          TEXT NOT NULL,
    paper_id            TEXT NOT NULL,
    verdict             TEXT NOT NULL,              -- supported | refuted | insufficient_evidence | not_verifiable
    confidence          REAL,
    reasoning           TEXT,
    supporting_quote    TEXT,
    model_id            TEXT,
    created_at          TEXT NOT NULL
);

-- Phase 5A: reactive draft suggestions (dry-run only, never posted).
CREATE TABLE IF NOT EXISTS koala_reactive_drafts (
    draft_id            TEXT PRIMARY KEY,
    comment_id          TEXT NOT NULL,
    paper_id            TEXT NOT NULL,
    recommendation      TEXT NOT NULL,             -- react | skip | unclear
    draft_text          TEXT,
    analysis_json       TEXT,                      -- JSON summary of verdict counts + skip_reason
    created_at          TEXT NOT NULL
);

-- Per-paper verdict eligibility state (latest snapshot per paper).
CREATE TABLE IF NOT EXISTS koala_verdict_state (
    paper_id                        TEXT PRIMARY KEY,
    has_our_participation           INTEGER NOT NULL DEFAULT 0,
    distinct_citable_other_agents   INTEGER NOT NULL DEFAULT 0,
    eligibility_state               TEXT NOT NULL DEFAULT 'NOT_PARTICIPATED',
    reachability_score              REAL,
    internal_confidence             REAL,
    submitted                       INTEGER NOT NULL DEFAULT 0,
    skip_reason                     TEXT,
    updated_at                      TEXT NOT NULL       -- ISO-8601 UTC
);
