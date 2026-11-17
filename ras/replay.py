"""Replay: a harness's modeling procedure, refitted on a run another harness made, scored by the official verifier.

Both kinds of lab are deterministic (a replay table, a noiseless twin) and the agent's model takes no seed,
so two runs cannot share an execution seed; what they can share is the trajectory itself. A run's
discovery depends only on what it measured (its ledger), and its learning only on the model it delivered.
A harness that changes nothing but its modeling files (harness.MODELING) gives the agent the same plan,
and its delivered model is its procedure's output on the run's own rows (``verified``). Its run on that
trajectory is then known without running it: refit ``scripts/model/fit.py`` on the trajectory's
train.csv, exactly as ``train.py`` builds it, and score the ledger and that model with the task's own
verifier (``tests/test_outputs.py``), unmodified. The candidate and the incumbent are then compared on
identical measurements, which removes the run-to-run spread of discovery and of the agent's own model
building from the comparison: common random numbers in the strongest form available.

A replay is one counterfactual run: ``verifier.json`` (the same aggregates a run gets), the refitted model,
the verifier's raw output under ``sealed/``, and which run it replayed. Only aggregates leave the verifier.
Replays run one at a time, so the verifier's time limit means what it means in a trial. Nothing here
reaches an agent; a replay only informs selection and, as aggregates, the editors.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from . import evidence, feedback, procedure, sandbox
from .config import Campaign

VERIFY_TIMEOUT = 900  # seconds for the verifier process; its own limit on scoring the model is below this
_ONE_AT_A_TIME = threading.Lock()

Fitter = Callable[[Path, list[str]], subprocess.CompletedProcess]  # sandbox.offline's signature
Verifier = Callable[[Campaign, Path, Path], dict]  # (campaign, ledger, model) -> the verifier's metrics


def official_verifier(campaign: Campaign, ledger: Path, model: Path) -> dict:
    """The task's own verifier, run unmodified in its local mode on one ledger and one model: its metrics.json."""
    with tempfile.TemporaryDirectory(prefix="ras-verify-") as work:
        root = Path(work)
        (root / "state" / "deliverables").mkdir(parents=True)
        (root / "logs").mkdir()
        shutil.copyfile(ledger, root / "state" / "ledger.json")
        shutil.copyfile(model, root / "state" / "deliverables" / "final_model.js")
        env = os.environ | {"ASB_VERIFY_LOCAL": "1", "ASB_VERIFY_STATE_DIR": str(root / "state"),
                            "ASB_VERIFY_LOGS_DIR": str(root / "logs"), "PYTHONDONTWRITEBYTECODE": "1",
                            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        try:  # -B: no bytecode beside the frozen benchmark's tests
            subprocess.run([sys.executable, "-B", str(campaign.task / "tests" / "test_outputs.py")], cwd=root,
                           env=env, capture_output=True, text=True, timeout=VERIFY_TIMEOUT)
        except subprocess.TimeoutExpired:
            return {"error": f"the verifier did not finish within {VERIFY_TIMEOUT} s"}
        return evidence.read_json(root / "logs" / "metrics.json")


def fit(model_dir: Path, rows: list[dict], out: Path, run: Fitter = sandbox.offline) -> str | None:
    """Refit a procedure on ``rows`` in the sandbox, with no network; write its model to ``out``. Returns an error."""
    with tempfile.TemporaryDirectory(prefix="ras-replay-") as work:
        root = Path(work)
        root.chmod(0o755)
        shutil.copytree(model_dir, root / "procedure", ignore=shutil.ignore_patterns("__pycache__"))
        with (root / "train.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        try:
            done = run(root, ["sh", "-c", "cd procedure && python3 fit.py --train ../train.csv --out ../final_model.js"])
        except subprocess.TimeoutExpired:
            return "the procedure did not finish in the sandbox's time limit"
        if not (root / "final_model.js").is_file():
            tail = (done.stdout + done.stderr).strip().splitlines()[-1:] or ["no output"]
            return f"the procedure wrote no model: {tail[0][:300]}"
        shutil.copyfile(root / "final_model.js", out)
    return None


class Replayer:
    """Replays procedures on run records, once each: a replay depends only on the procedure and the record."""

    def __init__(self, campaign: Campaign, fitter: Fitter = sandbox.offline, verifier: Verifier = official_verifier):
        self.campaign, self.fitter, self.verifier = campaign, fitter, verifier

    def score(self, record: Path, model: Path, target: Path) -> dict:
        """A given model on ``record``'s trajectory, scored by the verifier and written to ``target`` like a replay."""
        if (target / "verifier.json").is_file():
            return evidence.read_json(target / "verifier.json")
        staging = target.with_name(target.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        (staging / evidence.SEALED).mkdir(parents=True)
        shutil.copyfile(model, staging / "final_model.js")
        with _ONE_AT_A_TIME:
            metrics = self.verifier(self.campaign, record / "ledger.json", staging / "final_model.js")
        result = evidence.verifier(metrics, self.campaign)
        (staging / evidence.SEALED / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (staging / "verifier.json").write_text(json.dumps(result, indent=2) + "\n")
        (staging / "replayed.txt").write_text(f"{record.name}\n")
        staging.rename(target)
        return result

    def replay(self, tree: Path, record: Path, target: Path) -> dict:
        """``tree``'s procedure on ``record``'s trajectory, written to ``target``; returns its verifier.json."""
        if (target / "verifier.json").is_file():
            return evidence.read_json(target / "verifier.json")
        staging = target.with_name(target.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        (staging / evidence.SEALED).mkdir(parents=True)
        rows = feedback.training_rows(evidence.read_json(record / "ledger.json"), self.campaign)
        with _ONE_AT_A_TIME:
            error = fit(tree / procedure.MODEL_DIR, rows, staging / "final_model.js", self.fitter)
            metrics = {} if error else self.verifier(self.campaign, record / "ledger.json", staging / "final_model.js")
        error = error or metrics.get("error") or (None if metrics else "the verifier wrote no metrics")
        if error:  # the model failed: learning scores 0, as the verifier scores it; discovery is the ledger's own
            result = evidence.read_json(record / "verifier.json")
            key = self.campaign.learning_gate[0]
            gates = (result.get("gates") or {}) | {key: {"value": None, "required": self.campaign.learning_gate[1],
                                                         "share": 0.0}}
            result |= {"learning": 0.0, "bottleneck": 0.0, "passed": False, "error": error, "gates": gates}
        else:
            result = evidence.verifier(metrics, self.campaign)
        (staging / evidence.SEALED / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (staging / "verifier.json").write_text(json.dumps(result, indent=2) + "\n")
        (staging / "replayed.txt").write_text(f"{record.name}\n")
        staging.rename(target)
        return result

