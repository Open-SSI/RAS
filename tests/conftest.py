"""A tiny campaign on disk, and fake Harbor trials in the layout Harbor writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ras.config import Campaign, Executor, Fit, Objective, Protocol, Scan, Study

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# Twelve designs with distinct sequences; activity() makes a quarter of them class 2 and a quarter class 3.
DESIGNS = {f"variant_{i:04d}": "MK" + "".join(AMINO_ACIDS[(7 * i + j) % 20] for j in range(8))
           for i in range(12)}


@pytest.fixture
def campaign(tmp_path: Path) -> Campaign:
    task = tmp_path / "task"
    (task / "environment" / "data").mkdir(parents=True)
    (task / "instruction.md").write_text("Find active variants, then deliver a model.\n")
    (task / "environment" / "MODEL_FORMAT.md").write_text("predict(sequence) returns a number.\n")
    (task / "tests").mkdir()
    (task / "tests" / "secret.csv").write_text("never copied\n")
    catalog = task / "environment" / "data" / "candidates.csv"
    catalog.write_text("variant_id,sequence,family\n" + "".join(
        f"{design},{sequence},{'ab'[i % 2]}\n" for i, (design, sequence) in enumerate(DESIGNS.items())))
    return Campaign(
        name="toy-campaign", task=task, catalog=catalog, id="variant_id", input="sequence",
        groups=("family",), objective=Objective("activity"),
        discovery=("discovery_hits",), gates={"discovery_hits": 3.0},
        hit=("at_least", 2.0), random_hit_rate=0.25, timeout_multiplier=0.25,
        scan=(Scan(catalog, ("variant_id", "sequence")),), fit=Fit("tokens"),
        learning_gate=("ndcg_at_20", 0.5), ranking={"k": 20, "relevance": "above", "threshold": 1.5},
    )


@pytest.fixture
def study(tmp_path: Path, campaign: Campaign) -> Study:
    executor = Executor("fake", "ras.agents:HarnessedCodex", "fake-model", {}, {}, ("true", "{request}"))
    return Study(tmp_path / "benchmark", "0" * 40, "docker",
                 Protocol(steps=2, recursions=1, base_runs=2, eval_runs=2, probe_runs=1, composed_runs=2),
                 {"fake": executor}, {campaign.name: campaign})


def activity(design: str) -> float:
    return float(int(design[-4:]) % 4)


def write_trial(path: Path, designs: list[str], ndcg: float = 0.5) -> Path:
    """One finished trial: two rounds over ``designs``, a model, and the verifier's output."""
    state = path / "artifacts" / "state"
    (state / "deliverables").mkdir(parents=True)
    (path / "agent").mkdir(parents=True)
    (path / "verifier").mkdir(parents=True)
    half = len(designs) // 2
    jobs = [{"job_id": f"job-{stage:04d}", "stage": stage, "designs": batch, "status": "completed",
             "results": [{"variant_id": d, "activity": activity(d)} for d in batch]}
            for stage, batch in ((1, designs[:half]), (2, designs[half:]))]
    (state / "ledger.json").write_text(json.dumps({"format_version": 1, "jobs": jobs}))
    (state / "deliverables" / "final_model.js").write_text("function predict(s) { return s.length; }\n")
    (path / "agent" / "trajectory.json").write_text(json.dumps({"steps": [{"step_id": 1}, {"step_id": 2}]}))
    hits = sum(activity(d) >= 2 for d in designs)
    (path / "verifier" / "metrics.json").write_text(json.dumps(
        {"discovery_hits": hits, "ndcg_at_20": ndcg, "gate_results": {"discovery_hits": hits >= 3, "ndcg_at_20": ndcg >= 0.5},
         "passed": hits >= 3 and ndcg >= 0.5, "campaign": {"stages_completed": 2}}))  # the toy campaign's gates
    (path / "verifier" / "reward.txt").write_text("0\n")
    (path / "result.json").write_text(json.dumps({
        "agent_result": {"cost_usd": 0.3, "n_input_tokens": 100, "n_cache_tokens": 50, "n_output_tokens": 10},
        "agent_execution": {"started_at": "2026-10-23T10:00:00Z", "finished_at": "2026-10-23T10:15:00Z"},
        "exception_info": None}))
    return path
