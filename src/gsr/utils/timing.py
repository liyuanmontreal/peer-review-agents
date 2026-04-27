# src/gsr/utils/timing.py
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

log = logging.getLogger(__name__)

@contextmanager
def timed(stage: str, **meta) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        meta_str = " ".join(f"{k}={v}" for k, v in meta.items() if v is not None)
        log.info("[timing] stage=%s %s elapsed=%.2fs", stage, meta_str, elapsed)