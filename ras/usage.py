"""Token usage and API cost of an improver session, from the event stream its CLI wrote to the session log.

Codex (``codex exec --json``) reports each turn's tokens and no cost, so the cost is priced
from the same LiteLLM table Harbor prices trials with. Claude Code (``--output-format
stream-json``) ends with a result event that carries its own usage and cost. A log in neither
form, such as one from before the CLIs wrote JSON, yields only the total the CLI printed, if any.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
from pathlib import Path

FIELDS = ("input_tokens", "cached_tokens", "cache_write_tokens", "output_tokens")


def session(log: Path, model: str) -> dict:
    """{input, cached, cache-write and output tokens, cost_usd} for one session log."""
    usage = dict.fromkeys(FIELDS, 0) | {"cost_usd": None}
    if not log.is_file():
        return usage
    reported = None
    for line in log.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "turn.completed":  # codex: input_tokens includes the cached ones
            turn = event.get("usage") or {}
            usage["input_tokens"] += turn.get("input_tokens", 0) - turn.get("cached_input_tokens", 0)
            usage["cached_tokens"] += turn.get("cached_input_tokens", 0)
            usage["output_tokens"] += turn.get("output_tokens", 0)
        elif event.get("type") == "result":  # claude: the session's totals, once, at the end
            total = event.get("usage") or {}
            usage |= {"input_tokens": total.get("input_tokens", 0),
                      "cached_tokens": total.get("cache_read_input_tokens", 0),
                      "cache_write_tokens": total.get("cache_creation_input_tokens", 0),
                      "output_tokens": total.get("output_tokens", 0)}
            reported = event.get("total_cost_usd")
    if not any(usage[f] for f in FIELDS):
        printed = re.findall(r"^tokens used\n([\d,]+)", log.read_text(errors="replace"), re.M)
        usage["total_tokens"] = int(printed[-1].replace(",", "")) if printed else 0
        return usage
    usage["cost_usd"] = reported if reported is not None else price(model, usage)
    return usage


def price(model: str, usage: dict) -> float | None:
    rates = _prices().get(model)
    if not rates or rates.get("input_cost_per_token") is None:
        return None
    return (usage["input_tokens"] * rates["input_cost_per_token"]
            + usage["cached_tokens"] * (rates.get("cache_read_input_token_cost") or rates["input_cost_per_token"])
            + usage["cache_write_tokens"] * (rates.get("cache_creation_input_token_cost") or rates["input_cost_per_token"])
            + usage["output_tokens"] * rates["output_cost_per_token"])


@functools.cache
def _prices() -> dict:
    spec = importlib.util.find_spec("litellm")
    if spec is None or spec.origin is None:
        return {}
    table = Path(spec.origin).parent / "model_prices_and_context_window_backup.json"
    return json.loads(table.read_text()) if table.is_file() else {}
