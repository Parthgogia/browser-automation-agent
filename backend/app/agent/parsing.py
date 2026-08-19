"""Tolerant parsing of model output.

Models asked for JSON return JSON *most* of the time. The rest of the time they
wrap it in a Markdown fence, prefix it with "Here is the plan:", or emit
trailing prose. None of that is worth failing a task over, so this module digs
the object out and falls back to something usable when it genuinely cannot.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: ```json ... ``` or ``` ... ```
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from `text`, or None."""
    if not text:
        return None

    candidates: list[str] = []

    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())

    candidates.append(text.strip())

    # Last resort: the outermost brace-delimited span. Cheap and effective for
    # "Here is the plan: {...}. Let me know if..." responses.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed

    logger.debug("Could not parse JSON from model output: %.200s", text)
    return None


def parse_string_list(value: Any, *, limit: int = 12) -> list[str]:
    """Coerce a model-supplied value into a clean list of short strings."""
    if isinstance(value, str):
        # Occasionally a model returns a newline- or bullet-separated string.
        value = [line for line in re.split(r"[\n;]", value) if line.strip()]
    if not isinstance(value, list):
        return []
    return [
        re.sub(r"^\s*[-*\d.)\s]+", "", str(item)).strip()
        for item in value[:limit]
        if str(item).strip()
    ]
