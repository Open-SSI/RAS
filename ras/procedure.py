"""Harness-owned modeling: the procedure an agent builds its delivered model with, and its validation.

A harness owns its modeling procedure in ``scripts/model/``: ``fit.py`` turns ``train.csv``
into ``final_model.js``, and may do so any way that is deterministic. The library copy an
editor starts from adds ``train.py`` and ``spec.json``, which build ``train.csv`` from the
public seed and the lab; the pipeline builds the same rows from the run's ledger.
``validate`` cross-validates the procedure and checks that the delivered model is its
output, in the sandbox with no network.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from . import sandbox
from .config import Campaign

MODEL_DIR = "scripts/model"
TOOLKIT = Path(__file__).with_name("toolkit")
CV_SCRIPT = Path(__file__).with_name("procedure_cv.py")
BENCH = Path(__file__).with_name("bench.py")
RANKING = Path(__file__).with_name("ranking.py")  # the campaign's ranking metric, beside every script that scores
PROBE = 300  # catalog inputs on which the refitted and the delivered models are compared
Runner = Callable[[Path, list[str]], subprocess.CompletedProcess]


def spec(campaign: Campaign) -> dict:
    """What train.py needs to build train.csv inside the agent's container; all of it public."""
    data = campaign.task / "environment" / "data"
    container = lambda path: f"/app/data/{path.relative_to(data).as_posix()}"
    objective = campaign.objective
    return {
        "catalog": container(campaign.catalog), "id": campaign.id, "input": campaign.input,
        "objective": {"field": objective.field, "higher_is_better": objective.higher_is_better,
                      "zero_above": list(objective.zero_above) if objective.zero_above else None},
        "group": list(campaign.fit.group) if campaign.fit else [],
        "seed": [{"file": container(seed.file), "input": seed.input} for seed in campaign.seed],
    }


def library(campaign: Campaign, target: Path) -> Path:
    """The starting procedure an editor may adopt: the toolkit, the train.csv builder, and its spec."""
    shutil.copytree(TOOLKIT, target, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(RANKING, target / RANKING.name)
    (target / "spec.json").write_text(json.dumps(spec(campaign), indent=2) + "\n")
    if campaign.fit:  # the campaign's featurizer and grouping, as the procedure's defaults
        config = {"featurizer": campaign.fit.featurizer, "protocol": "random",
                  "group_normalize": bool(campaign.fit.group), "ranking": campaign.ranking}
        (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return target


def complete(model: Path, campaign: Campaign) -> Path:
    """Give a harness's procedure RAS's own train.py and spec.json, which build train.csv in the agent's container (the
    rows feedback.training_rows builds from the ledger) and which no editor may change: the harness's delivery and its
    planner run train.py, so a procedure without them could be replayed but never delivered."""
    if (model / "fit.py").is_file():
        shutil.copyfile(TOOLKIT / "train.py", model / "train.py")
        (model / "spec.json").write_text(json.dumps(spec(campaign), indent=2) + "\n")
    return model


def scorers(target: Path, metric: dict | None) -> None:
    """The scripts that score procedures, with the campaign's ranking metric beside them."""
    for script in (BENCH, CV_SCRIPT, RANKING):
        shutil.copyfile(script, target / script.name)
    if metric:
        (target / "ranking.json").write_text(json.dumps(metric, indent=2) + "\n")


def audit(model: Path, evidence_dir: Path, run: Runner = sandbox.offline) -> dict:
    """The bench's prior check of a procedure on the evidence rows, in the sandbox with no network."""
    with tempfile.TemporaryDirectory(prefix="ras-audit-") as work:
        root = Path(work)
        root.chmod(0o755)
        shutil.copytree(model, root / "procedure", ignore=shutil.ignore_patterns("__pycache__"))
        scorers(root, None)
        for train in evidence_dir.glob("*/train.csv"):
            (root / "evidence" / train.parent.name).mkdir(parents=True)
            shutil.copyfile(train, root / "evidence" / train.parent.name / "train.csv")
        done = run(root, ["python3", "bench.py", "procedure", "--prior", "--no-cross"])
        lines = done.stdout.strip().splitlines()
        try:
            report = json.loads(lines[-1])["scores"]["prior"]
            if (root / "prior.js").is_file():
                report["source"] = (root / "prior.js").read_text()
            return report
        except (IndexError, ValueError, KeyError):
            tail = (done.stdout + done.stderr).strip().splitlines()[-1:] or ["no output"]
            return {"error": f"the procedure audit did not run: {tail[0][:300]}"}


def validate(harness_tree: Path | None, rows: list[dict], delivered: Path, probe: list[str],
             run: Runner = sandbox.offline, metric: dict | None = None) -> dict:
    """Cross-validation of the harness's procedure by the campaign's ranking ``metric``, and whether it
    produced the delivered model."""
    procedure = harness_tree / MODEL_DIR if harness_tree else None
    if procedure is None or not (procedure / "fit.py").is_file():
        return {"error": "the harness has no modeling procedure (scripts/model/fit.py)"}
    with tempfile.TemporaryDirectory(prefix="ras-cv-") as work:
        root = Path(work)
        root.chmod(0o755)
        shutil.copytree(procedure, root / "procedure", ignore=shutil.ignore_patterns("__pycache__"))
        scorers(root, metric)
        with (root / "train.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if delivered.is_file():
            shutil.copyfile(delivered, root / "delivered.js")
        (root / "probe.json").write_text(json.dumps(probe))
        done = run(root, ["python3", "procedure_cv.py", "--out", "report.json"])
        report = root / "report.json"
        if not report.is_file():
            tail = (done.stdout + done.stderr).strip().splitlines()[-1:] or ["no output"]
            return {"error": f"validation did not run: {tail[0][:300]}"}
        return json.loads(report.read_text())


def local(work: Path, argv: list[str]) -> subprocess.CompletedProcess:
    """Run validation on the host with this interpreter instead of the sandbox; for tests."""
    return subprocess.run([sys.executable, *argv[1:]], cwd=work, capture_output=True, text=True, timeout=3600)
