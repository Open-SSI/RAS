"""Preflight: every piece a study needs, on every campaign, before a recursion spends model tokens.

Per campaign, with no model involved: the stock harness passes the guard; the configuration restates
the instruction's gates, its ranking metric (ranking.py) and the verifier's model contract
(contract.py); and the benchmark's own runner scores a trivial model on an evaluation-sized set of
designs within the verifier's time limit, so a delivered model's contract check can run. Per
executor, one improver session in the sandbox, whose usage must be readable from its log, and one
smoke trial, whose harness must mount. Everything goes under its own runs directory, never the study's.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from . import contract, guard, harness, ranking, smoke, usage
from .config import Campaign, Executor, Study
from .improver import Improver
from .runner import HarborRunner

PROBE = ("Create a file named RATIONALE.md in this directory whose only content is the line "
         "`preflight ok`. Do nothing else.\n")
TRIVIAL = "function predict(design) { return String(design).length; }\n"


def campaign(study: Study, item: Campaign, runner: HarborRunner, root: Path) -> dict:
    """The model-free checks on one campaign."""
    started = time.monotonic()
    stock = root / "h0"
    if not stock.exists():
        harness.write_stock(stock)
    out = {"guard": guard.check(stock, item)}
    out["config"] = [problem for problem, bad in (
        ("no ranking metric", item.ranking is None), ("no model contract", item.contract is None),
        ("no learning gate", item.learning_gate is None), ("no discovery gate", not item.gates)) if bad]
    if item.ranking:
        out["ranking"] = ranking.describe(item.ranking)
    if item.contract:
        out["contract_inputs"] = len(contract.inputs(item))
        with tempfile.TemporaryDirectory(prefix="ras-preflight-") as work:
            model = Path(work) / "final_model.js"
            model.write_text(TRIVIAL)
            out["contract"] = contract.check(model, item, contract.benchmark_runner(study.runtime))
    out["minutes"] = round((time.monotonic() - started) / 60, 1)
    out["ok"] = (not out["guard"] and not out["config"] and out.get("contract", {}).get("ok", False)
                 and out.get("contract_inputs") == item.contract.designs)
    return out


def improver(executor: Executor, root: Path) -> dict:
    """One trivial improver session in the sandbox: it must run, write its notes, and leave readable usage."""
    workspace = root / "improver" / executor.name
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    (workspace / "STAGE.md").write_text(PROBE)
    started = time.monotonic()
    code = Improver(executor, timeout=600).session(workspace)
    notes = workspace / "RATIONALE.md"
    spent = usage.session(workspace / "session.log", executor.model)
    return {"exit_code": code, "notes": notes.read_text().strip() if notes.is_file() else None, "usage": spent,
            "minutes": round((time.monotonic() - started) / 60, 1),
            "ok": code == 0 and notes.is_file() and spent["cost_usd"] is not None and spent["output_tokens"] > 0}


def trial(study: Study, executor: Executor, root: Path, env: str) -> dict:
    """One smoke trial: the marker harness must mount, and the trial must report its cost."""
    report = smoke.run(study, executor, study.campaigns["biosensor-active-learning"], root, env)
    return {"checks": report["checks"], "stats": report["stats"],
            "ok": all(report["checks"].values()) and report["stats"].get("cost_usd") is not None}


def write(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
