import json

from ras import evidence

from conftest import DESIGNS, write_trial


def test_record_seals_the_verifier_and_export_leaves_it_behind(tmp_path, campaign):
    trial = write_trial(tmp_path / "trial", list(DESIGNS)[:8])
    record = evidence.record(trial, tmp_path / "records" / "base-1", campaign, "")
    assert (record / "sealed" / "metrics.json").is_file()
    assert evidence.source(record) == str(trial)

    shown = json.loads((record / "discovery.json").read_text())
    assert set(shown["fields"]) == {"discovery_hits"}
    assert shown["gates"]["discovery_hits"] == {"value": 4, "required": 3.0, "met": True}
    assert "ndcg" not in (record / "discovery.json").read_text()

    stats = json.loads((record / "stats.json").read_text())
    assert stats["agent_minutes"] == 15 and stats["steps"] == 2 and stats["cost_usd"] == 0.3

    exported = evidence.export(record, tmp_path / "workspace" / "base-1")
    names = {path.name for path in exported.rglob("*")}
    assert "sealed" not in names and "metrics.json" not in names and "reward.txt" not in names
