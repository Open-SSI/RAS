"""Run records, and the evidence an improver session may read.

``record`` copies a finished Harbor trial into a run record. The verifier's own
output goes to ``sealed/``; ``verifier.json`` summarizes it for selection and for the
improvers: the official discovery and learning metrics, each gate as a share of its
threshold, the bottleneck, pass or fail, and any verifier error. These are aggregates
over the evaluation set; no hidden label or held-out design ever leaves the verifier.
``discovery.json`` keeps the discovery fields alone. ``export`` copies the allowlisted
files, and nothing else, into a working directory.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from .config import Campaign

# What an improver may read from a run record. feedback.refresh adds feedback.json.
EVIDENCE = ("trajectory.json", "ledger.json", "final_model.js", "stats.json",
            "discovery.json", "verifier.json", "feedback.json", "harness.diff")
SEALED = "sealed"


def record(trial: Path, target: Path, campaign: Campaign, harness_diff: str, harness_name: str = "") -> Path:
    """Copy one Harbor trial into ``target``, which appears only once complete."""
    staging = target.with_name(target.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / SEALED).mkdir(parents=True)
    state = trial / "artifacts" / "state"
    for source, name in ((trial / "agent" / "trajectory.json", "trajectory.json"),
                         (state / "ledger.json", "ledger.json"),
                         (state / "deliverables" / "final_model.js", "final_model.js")):
        if source.is_file():
            shutil.copyfile(source, staging / name)
    for name in ("metrics.json", "reward.txt"):
        if (trial / "verifier" / name).is_file():
            shutil.copyfile(trial / "verifier" / name, staging / SEALED / name)
    own = agent_model(trial)
    if own is not None:
        shutil.copyfile(own, staging / "agent_model.js")
    if (trial / "agent" / "plan").is_dir() and any((trial / "agent" / "plan").glob("round_*.csv")):
        shutil.copytree(trial / "agent" / "plan", staging / "proposals")  # what the planner proposed, round by round
    (staging / SEALED / "trial.txt").write_text(f"{trial}\n")
    metrics = read_json(staging / SEALED / "metrics.json")
    _write(staging / "discovery.json", discovery(metrics, campaign))
    _write(staging / "verifier.json", verifier(metrics, campaign))
    _write(staging / "stats.json", stats(trial))
    (staging / "harness.diff").write_text(harness_diff)
    (staging / "harness.txt").write_text(f"{harness_name}\n")
    staging.rename(target)
    return target


def agent_model(trial: Path) -> Path | None:
    """Where the harness delivered its procedure's model after the agent (ras.agents), the model the agent itself
    delivered last: the captured file whose SHA-256 the lab's ledger lists for the agent's last submission, the one
    before the harness's."""
    log, captured = trial / "agent" / "delivery.txt", trial / "agent" / "agent_models"
    if not log.is_file() or not log.read_text().startswith("exit 0") or not captured.is_dir():
        return None
    entries = sorted(read_json(trial / "artifacts" / "state" / "ledger.json").get("deliverables", []), key=lambda e: e["seq"])
    if len(entries) < 2:
        return None
    path = captured / f"{entries[-2]['sha256']}.js"
    return path if path.is_file() else None


def discovery(metrics: dict, campaign: Campaign) -> dict:
    """The campaign's discovery fields and gates from the verifier's metrics, and nothing else."""
    fields = {key: metrics[key] for key in campaign.discovery if key in metrics}
    gates = {
        key: {"value": fields.get(key), "required": required,
              "met": isinstance(fields.get(key), (int, float)) and fields[key] >= required}
        for key, required in campaign.gates.items()
    }
    return {"fields": fields, "gates": gates, "progress": metrics.get("campaign", {})}


def verifier(metrics: dict, campaign: Campaign) -> dict:
    """The official result: every gate as a share of its threshold, the bottleneck min(D/tD, L/tL), and
    pass or fail. A gate the verifier could not score (an error, a missing model) counts 0."""
    def gate(value, required) -> dict:
        number = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        return {"value": number, "required": required, "share": (number or 0.0) / required}
    discovery_gates = {key: gate(metrics.get(key), required) for key, required in campaign.gates.items()}
    key, required = campaign.learning_gate
    learning_gate = gate(metrics.get(key), required)
    shares = {"discovery": min(g["share"] for g in discovery_gates.values()), "learning": learning_gate["share"]}
    aggregates = {k: v for k, v in metrics.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {"discovery": shares["discovery"], "learning": shares["learning"],
            "bottleneck": min(shares.values()), "passed": bool(metrics.get("passed")),
            "gates": {**discovery_gates, key: learning_gate},
            "metrics": aggregates, "error": metrics.get("error") or (None if metrics else "the verifier wrote no metrics"),
            "progress": metrics.get("campaign", {})}


def stats(trial: Path) -> dict:
    result = read_json(trial / "result.json")
    agent = result.get("agent_result") or {}
    timing = result.get("agent_execution") or {}
    return {
        "cost_usd": agent.get("cost_usd"),
        "input_tokens": agent.get("n_input_tokens"),
        "cache_tokens": agent.get("n_cache_tokens"),
        "output_tokens": agent.get("n_output_tokens"),
        "agent_minutes": _minutes(timing.get("started_at"), timing.get("finished_at")),
        "steps": len(read_json(trial / "agent" / "trajectory.json").get("steps", [])),
        "exception": (result.get("exception_info") or {}).get("exception_type"),
    }


def source(record_dir: Path) -> str:
    """The Harbor trial a record was copied from."""
    return (record_dir / SEALED / "trial.txt").read_text().strip()


def export(record_dir: Path, target: Path) -> Path:
    """Copy a record's allowlisted evidence into an improver's working directory."""
    target.mkdir(parents=True)
    for name in EVIDENCE:
        if (record_dir / name).is_file():
            shutil.copyfile(record_dir / name, target / name)
    return target


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _minutes(start: str | None, end: str | None) -> float | None:
    if not (start and end):
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60
