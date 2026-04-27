"""Thin LiteLLM wrapper for claim extraction."""

from __future__ import annotations

import logging
from typing import Any

import litellm

from gsr.config import get_llm_model

log = logging.getLogger(__name__)


def get_model_id(model: str | None = None) -> str:
    """Return the LLM model identifier.

    Uses *model* if provided, otherwise delegates to ``config.get_llm_model()``.
    """
    return model or get_llm_model()


def complete(
    messages: list[dict[str, str]],
    response_format: type | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Any:
    """Call LiteLLM completion and return the parsed content.

    If *response_format* is a Pydantic model class the response is parsed into
    that model; otherwise the raw content string is returned.
    """
    model_id = get_model_id(model)
    log.debug("LLM call: model=%s, messages=%d, temperature=%s", model_id, len(messages), temperature)

    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if response_format is not None:
        kwargs["response_format"] = response_format

    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content

    if response_format is not None and hasattr(response_format, "model_validate_json"):
        return response_format.model_validate_json(content)

    return content
