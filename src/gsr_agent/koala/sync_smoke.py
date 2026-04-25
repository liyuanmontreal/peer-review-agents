"""Sync smoke-test CLI — validates Koala API connectivity and local state sync.

Usage (dry-run, default — no writes to Koala):
    python -m gsr_agent.koala.sync_smoke

Usage (live — requires KOALA_API_TOKEN and KOALA_API_BASE_URL):
    python -m gsr_agent.koala.sync_smoke --live-post

Environment variables required for live mode:
    KOALA_API_BASE_URL    Koala platform base URL
    KOALA_API_TOKEN       bearer token for authentication
    KOALA_AGENT_ID        this agent's ID (for is_ours marking)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sync_smoke")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Koala API sync smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--live-post",
        action="store_true",
        default=False,
        help="Enable live mode (default: dry-run, no Koala writes).",
    )
    parser.add_argument(
        "--db-path",
        default="./workspace/smoke_test.db",
        help="Path to local SQLite database (default: ./workspace/smoke_test.db).",
    )
    parser.add_argument(
        "--agent-id",
        default="",
        help="Agent ID for is_ours marking (falls back to KOALA_AGENT_ID env).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    live = args.live_post
    if live:
        api_base = os.environ.get("KOALA_API_BASE_URL", "")
        api_token = os.environ.get("KOALA_API_TOKEN", "")
        if not api_base or not api_token:
            log.error(
                "--live-post requires KOALA_API_BASE_URL and KOALA_API_TOKEN to be set."
            )
            return 1
    else:
        log.info("Running in dry-run mode (no Koala writes). Pass --live-post to enable.")
        api_base = None
        api_token = ""

    from .client import KoalaClient
    from .sync import sync_all_active_state
    from ..storage.db import KoalaDB

    client = KoalaClient(
        api_base=api_base,
        api_token=api_token,
        test_mode=(not live),
    )
    db = KoalaDB(db_path=args.db_path)

    agent_id = args.agent_id or os.environ.get("KOALA_AGENT_ID", "")

    try:
        summary = sync_all_active_state(client, db, agent_id=agent_id)
    except Exception as exc:
        log.error("sync_all_active_state failed: %s", exc)
        db.close()
        return 1

    log.info("Sync complete: %s", json.dumps(summary))
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
