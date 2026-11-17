import json

import pytest

from ras import usage


def test_codex_turns_are_summed_and_priced(tmp_path, monkeypatch):
    monkeypatch.setattr(usage, "_prices", lambda: {"m": {"input_cost_per_token": 1e-6, "output_cost_per_token": 1e-5,
                                                         "cache_read_input_token_cost": 1e-7}})
    log = tmp_path / "session.log"
    turns = [{"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 800, "output_tokens": 50}},
             {"type": "item.completed", "item": {"type": "agent_message"}},
             {"type": "turn.completed", "usage": {"input_tokens": 2000, "cached_input_tokens": 1500, "output_tokens": 70}}]
    log.write_text("starting\n" + "\n".join(json.dumps(t) for t in turns) + "\n")
    spent = usage.session(log, "m")
    assert (spent["input_tokens"], spent["cached_tokens"], spent["output_tokens"]) == (700, 2300, 120)
    assert spent["cost_usd"] == pytest.approx(700e-6 + 2300e-7 + 120e-5)


def test_claude_reports_its_own_cost(tmp_path):
    log = tmp_path / "session.log"
    result = {"type": "result", "total_cost_usd": 0.42,
              "usage": {"input_tokens": 10, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 90, "output_tokens": 30}}
    log.write_text(json.dumps({"type": "system"}) + "\n" + json.dumps(result) + "\n")
    spent = usage.session(log, "unpriced-model")
    assert spent == {"input_tokens": 10, "cached_tokens": 900, "cache_write_tokens": 90, "output_tokens": 30, "cost_usd": 0.42}


def test_a_plain_log_yields_only_the_printed_total(tmp_path):
    log = tmp_path / "session.log"
    log.write_text("did some work\ntokens used\n135,492\nDone.\n")
    spent = usage.session(log, "m")
    assert spent["total_tokens"] == 135492 and spent["cost_usd"] is None
