"""Simulated runs: a planning harness's scripted rounds, run through the task's own lab, branched from a real run.

A harness that plans with its own model (harness.planned) leaves round 1 to the agent and proposes every later round
with ``scripts/plan/acquire.py``, which the agent orders unchanged. Those rounds need no model calls, so they can be run
without the agent: in the task's own images (its lab and its agent environment, built from the frozen task), round 1 is
ordered exactly as a real run ordered it, every later round is the planner's proposal, the harness then delivers its
procedure's model as it does after an agent (ras.agents), and the task's own verifier scores the ledger and the model.
The simulated run shares its first round with the real run it branched from, which pairs the two the way a common seed
would: what differs is everything the plan decided after round 1. A real agent's runs of the harness confirm it
(recursion: the agent ordered the proposals). If the planner reports insufficient evidence in a later round, the round
would be the agent's own, which a simulation cannot stand in for: the simulation stops and says so.

A simulated run is written like a replay (verifier.json, ledger.json, final_model.js, the raw verifier output under
sealed/, the proposals) with ``simulated.txt`` naming the run it branched from; only aggregates leave the verifier.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from . import evidence, harness
from .agents import DELIVER
from .config import Campaign
from .replay import official_verifier

INSUFFICIENT = 3  # acquire.py's exit status when the round is the agent's to choose
_ONE_AT_A_TIME = threading.Lock()


def _docker(*args: str, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True, **kwargs)


class Simulator:
    """Runs simulated runs of one campaign, one at a time, in images built once from the frozen task."""

    def __init__(self, campaign: Campaign, verifier=official_verifier):
        self.campaign, self.verifier = campaign, verifier
        self.images = {part: f"ras-sim-{campaign.name}-{part}:frozen" for part in ("main", "lab")}

    def _ensure_images(self) -> None:
        contexts = {"main": self.campaign.task / "environment", "lab": self.campaign.task / "environment" / "lab"}
        for part, tag in self.images.items():
            if _docker("image", "inspect", tag, check=False).returncode:
                _docker("build", "-q", "-t", tag, str(contexts[part]), timeout=3600)

    def run(self, tree: Path, source: Path, target: Path, branch_after: int = 1) -> dict:
        """``tree``'s planned rounds after the first ``branch_after`` rounds of ``source``, written to ``target``."""
        if (target / "verifier.json").is_file():
            return evidence.read_json(target / "verifier.json")
        with _ONE_AT_A_TIME:
            return self._run(tree, source, target, branch_after)

    def _run(self, tree: Path, source: Path, target: Path, branch_after: int) -> dict:
        self._ensure_images()
        tag = target.name.replace("@", "-").lower()[:40]
        net, lab, main = f"ras-sim-{tag}", f"ras-sim-{tag}-lab", f"ras-sim-{tag}-main"
        for name in (lab, main):
            _docker("rm", "-f", name, check=False)
        _docker("network", "rm", net, check=False)
        _docker("network", "create", "--internal", net)
        staging = target.with_name(target.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        (staging / evidence.SEALED).mkdir(parents=True)
        log: list[str] = []
        learning_valid = True
        try:
            _docker("run", "-d", "--name", lab, "--network", net, "--network-alias", "lab", self.images["lab"])
            _docker("run", "-d", "--name", main, "--network", net, "-e", "ASB_LAB_URL=http://lab:8080", "-w", "/app",
                    self.images["main"], "sleep", "infinity")
            _docker("exec", main, "sh", "-c", "for i in $(seq 60); do asb status >/dev/null 2>&1 && exit 0; sleep 1; done; exit 1")
            with tempfile.TemporaryDirectory(prefix="ras-sim-") as work:
                root = Path(work)
                for local, remote in harness.mount(tree, root / "mount", "AGENTS.md"):
                    _docker("exec", main, "mkdir", "-p", str(Path(remote).parent))
                    _docker("cp", str(local), f"{main}:{remote}")
                delivery = harness.delivery(tree, root / "staging")
                ledger = evidence.read_json(source / "ledger.json")
                jobs = sorted((j for j in ledger.get("jobs", []) if j.get("status") == "completed"), key=lambda j: j["stage"])
                stages = len(evidence.read_json(self.campaign.task / "environment" / "lab" / "campaign" / "campaign.json")["stages"])
                status = "complete"
                for stage in range(1, stages + 1):
                    batch = root / f"round_{stage}.csv"
                    if stage <= branch_after:
                        designs = next(j["designs"] for j in jobs if j["stage"] == stage)
                        batch.write_text(f"{self.campaign.id}\n" + "".join(f"{d}\n" for d in designs))
                        _docker("cp", str(batch), f"{main}:/app/{batch.name}")
                        origin = f"round {stage}: {len(designs)} designs as {source.name} ordered them"
                    else:
                        done = _docker("exec", main, "python3", "/app/harness/scripts/plan/acquire.py",
                                       "--out", f"/app/{batch.name}", check=False, timeout=3600)
                        origin = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else done.stderr[-300:]
                        if done.returncode == INSUFFICIENT:
                            status = f"stopped at round {stage}: {origin}"
                            log.append(origin)
                            break
                        if done.returncode:
                            raise RuntimeError(f"acquire.py failed in round {stage}: {done.stderr[-800:]}")
                    _docker("exec", main, "asb", "run", f"/app/{batch.name}", "-o", f"/app/{batch.stem}.out.csv", timeout=1800)
                    log.append(origin)
                if status == "complete" and delivery is not None:
                    _docker("cp", str(delivery), f"{main}:{harness.DELIVERY_MOUNT}")
                    done = _docker("exec", "-u", "root", main, "sh", "-c", DELIVER, check=False, timeout=1800)
                    log.append(f"delivery: exit {done.returncode}")
                elif status == "complete":  # the agent's own model would be delivered, which no simulation has
                    stand_in = root / "stand_in.js"
                    stand_in.write_text("function predict(design) { return 0; }\n")
                    _docker("cp", str(stand_in), f"{main}:/app/stand_in.js")
                    _docker("exec", main, "asb", "deliver", "/app/stand_in.js", check=False)
                    log.append("delivery: a constant stand-in for the agent's model (its learning is not the run's)")
                    learning_valid = False
                _docker("cp", f"{lab}:/state/ledger.json", str(staging / "ledger.json"))
                model = staging / "final_model.js"
                if _docker("cp", f"{lab}:/state/deliverables/final_model.js", str(model), check=False).returncode:
                    model.write_text("function predict(design) { return 0; }\n")  # nothing delivered: learning scores 0
                if _docker("cp", f"{main}:/app/plan", str(staging / "proposals"), check=False).returncode:
                    (staging / "proposals").mkdir(exist_ok=True)
            metrics = self.verifier(self.campaign, staging / "ledger.json", model) if status == "complete" else {}
        finally:
            for name in (lab, main):
                _docker("rm", "-f", name, check=False)
            _docker("network", "rm", net, check=False)
        result = evidence.verifier(metrics, self.campaign) if metrics else {}
        if status != "complete":
            result = {"discovery": 0.0, "learning": 0.0, "bottleneck": 0.0, "passed": False, "gates": {},
                      "error": status, "inconclusive": True}
        elif not learning_valid:  # discovery only: such a run is no observation of the harness's whole result
            result |= {"learning_valid": False, "inconclusive": True}
        (staging / evidence.SEALED / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (staging / "verifier.json").write_text(json.dumps(result, indent=2) + "\n")
        (staging / "discovery.json").write_text(json.dumps(evidence.discovery(metrics, self.campaign), indent=2) + "\n")
        (staging / "simulated.txt").write_text(f"{source.name}\nbranched after round {branch_after}\n" + "\n".join(log) + "\n")
        (staging / "replayed.txt").write_text(f"{source.name}\n")
        staging.rename(target)
        return result
