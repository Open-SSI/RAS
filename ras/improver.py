"""Improver sessions: the executor's own CLI, run headless in the sandbox on a prepared working directory.

A working directory holds STAGE.md (the stage prompt and the feedback notes),
the campaign card, the evidence so far with each run's official verifier result,
and the harness to edit under ``harness/``, with an optional modeling procedure and
the procedure bench under ``library/``. The composer also gets both candidates under
``candidates/``. The card is only what the agent itself was given, and evidence
arrives only through ``evidence.export``: aggregates of the verifier's scoring, never
its hidden labels or held-out designs. The sandbox mounts only the working directory,
so nothing else on the host can reach a session.
"""

from __future__ import annotations

import csv
import json
import shlex
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from . import evidence, feedback, harness, procedure, sandbox
from .config import Campaign, Executor, environment, resolve
from .runner import DryRun

PROMPTS = Path(__file__).with_name("prompts")
STAGES = ("planning", "modeling", "composer")
REQUEST = "Read STAGE.md in this directory and follow it."
# The campaign card: files the agent itself reads, relative to the task.
CARD = ("instruction.md", "environment/MODEL_FORMAT.md", "environment/data/README.md")


def stage_prompt(stage: str, diagnosis: str = "") -> str:
    """The stage's prompt, with the recursion's diagnosis of the current harness where it gives one."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage}; known: {STAGES}")
    if stage == "composer":
        text = (PROMPTS / "composer.md").read_text()
    else:
        scope = (PROMPTS / f"scope_{stage}.md").read_text().strip()
        text = (PROMPTS / "editor.md").read_text().replace("{scope}", scope)
    text = text.replace("{diagnosis}\n\n", f"{diagnosis.strip()}\n\n" if diagnosis.strip() else "")
    return text + "\n" + (PROMPTS / "feedback.md").read_text() + (PROMPTS / "bench.md").read_text()


def notes_path(tree: Path) -> Path:
    """Where a session's RATIONALE.md or DECISIONS.md is kept, next to the harness it produced."""
    return tree.with_name(f"{tree.name}.notes.md")


def prepare(workspace: Path, stage: str, campaign: Campaign, current: Path,
            records: Mapping[str, Path], candidates: Mapping[str, Path] | None = None, diagnosis: str = "") -> Path:
    workspace.mkdir(parents=True)
    (workspace / "STAGE.md").write_text(stage_prompt(stage, diagnosis))
    card = workspace / "campaign"
    card.mkdir()
    for relative in CARD:
        source = campaign.task / relative
        if source.is_file():
            shutil.copyfile(source, card / source.name)
    harness.copy(current, workspace / "harness")
    # A starting modeling procedure the editor may adopt; nothing in library/ reaches the agent unless copied.
    procedure.library(campaign, workspace / "library" / "model-procedure")
    bench = workspace / "library" / "bench"
    bench.mkdir()
    procedure.scorers(bench, campaign.ranking)
    for name, path in sorted(records.items()):
        evidence.export(path, workspace / "evidence" / name)
        write_rows(workspace / "evidence" / name, path, campaign)
    for scope, tree in (candidates or {}).items():
        target = workspace / "candidates" / scope
        harness.copy(tree, target / "harness")
        if notes_path(tree).is_file():
            shutil.copyfile(notes_path(tree), target / "RATIONALE.md")
    return workspace


def write_rows(target: Path, record: Path, campaign: Campaign) -> None:
    """train.csv, the rows a run's model was fitted on (the seed and its measured designs, as
    train.py builds them), and measured.csv, the measured designs alone."""
    ledger = record / "ledger.json"
    if not ledger.is_file():
        return
    ledger_json = json.loads(ledger.read_text())
    for name, rows in (("train.csv", feedback.training_rows(ledger_json, campaign)),
                       ("measured.csv", feedback.measured_rows(ledger_json, campaign))):
        if rows:
            with (target / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


class Improver:
    """Runs one executor's improver sessions in the sandbox."""

    def __init__(self, executor: Executor, timeout: int = 3600, dry_run: bool = False):
        self.executor, self.timeout, self.dry_run = executor, timeout, dry_run
        self._ready = False

    def command(self) -> list[str]:
        parts = list(self.executor.improver)
        if "{request}" not in parts:
            return [*parts, REQUEST]
        return [REQUEST if part == "{request}" else part for part in parts]

    def session(self, workspace: Path, attempt: int = 1) -> int | None:
        """Run one session; return its exit code, or None if it timed out."""
        environ = environment()
        secrets = resolve(self.executor.improver_env, environ)
        command = sandbox.command(workspace, self.command(), secrets)
        if self.dry_run:
            print(shlex.join(command))
            raise DryRun("printed the improver command above and stopped before running it")
        if not self._ready:
            sandbox.ensure()
            self._ready = True
        log = workspace / ("session.log" if attempt == 1 else f"session-{attempt}.log")
        with log.open("w") as handle:
            try:
                return subprocess.run(command, stdin=subprocess.DEVNULL, stdout=handle,
                                      stderr=subprocess.STDOUT, env=environ | secrets,
                                      timeout=self.timeout, check=False).returncode
            except subprocess.TimeoutExpired:
                return None
