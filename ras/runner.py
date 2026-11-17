"""Harbor jobs: run one harness tree on one campaign a fixed number of times.

Each job is written as a Harbor job config and run with ``harbor run -c``.
Harbor resumes a job whose directory already holds the same config, so running
an interrupted job again finishes it instead of starting over. A trial that
failed before its agent did any work is an infrastructure error, as Autonomous
Science Bench's summary defines it; so is one the model provider cut short
(overloaded, rate-limited, out of usage, a server error, a dropped or stalled stream) or the
harness cancelled, since through no doing of its agent it did not get the
campaign's budget. Either is moved to ``failed/`` and run again.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import Campaign, Executor, environment
from .evidence import read_json

# The harbor of this environment, which can import ras.agents.
HARBOR = (str(Path(sys.executable).with_name("harbor")),)
RETRIES = 3        # reruns of a job whose trials hit infrastructure errors
RETRY_DELAY = 300  # seconds before a rerun, so a transient shortage (Docker networks, an overloaded API) can clear
# Harbor's exception types for a trial the model provider cut short, and for one the harness cancelled.
INTERRUPTIONS = {"ApiOverloadedError", "ApiRateLimitError", "ApiUsageLimitError", "ApiInternalServerError",
                 "ApiConnectionClosedError", "ApiResponseStalledError", "CancelledError"}
SETUP_MULTIPLIER = 3.0  # agent setup limit, as a multiple of Harbor's 360 s
AT_ONCE = 3  # trials of one job side by side: verifiers time out, and memory runs short, when many run at once


class DryRun(Exception):
    """Raised instead of starting a job or a session under --dry-run."""


@dataclass(frozen=True)
class Job:
    name: str
    campaign: Campaign
    executor: Executor
    harness: Path
    attempts: int


def job_config(job: Job, jobs_dir: Path, env: str) -> dict:
    agent = {
        "import_path": job.executor.agent,
        "model_name": job.executor.model,
        "kwargs": {**job.executor.kwargs, "harness_dir": str(job.harness.resolve())},
    }
    if job.executor.env:
        agent["env"] = job.executor.env
    return {
        "job_name": job.name,
        "jobs_dir": str(jobs_dir.resolve()),
        "n_attempts": job.attempts,
        "n_concurrent_trials": min(job.attempts, AT_ONCE),
        "environment": {"type": env},
        "agents": [agent],
        "tasks": [{"path": str(job.campaign.task)}],
        "agent_timeout_multiplier": job.campaign.timeout_multiplier,
        # Installing the agent's CLI in many trials at once can exceed Harbor's 360 s setup limit.
        "agent_setup_timeout_multiplier": SETUP_MULTIPLIER,
    }


def infra_error(trial: Path) -> bool:
    """The trial failed before its agent did any work (no billed usage, no completed stage), or
    the model provider or the harness interrupted it."""
    result = read_json(trial / "result.json")
    exception = result.get("exception_info") or {}
    if exception.get("exception_type") in INTERRUPTIONS:
        return True
    if exception and not (trial / "verifier" / "reward.txt").is_file() and "verifier" in str(exception.get("exception_message")):
        return True  # the verifier's environment never started: the run was never scored
    agent = result.get("agent_result") or {}
    jobs = read_json(trial / "artifacts" / "state" / "ledger.json").get("jobs", [])
    return bool(result.get("exception_info") and not agent.get("n_input_tokens")
                and not agent.get("cost_usd") and not any(j.get("status") == "completed" for j in jobs))


def finished(jobs_dir: Path, name: str) -> list[Path]:
    """Trials of a job that wrote their result, other than infrastructure errors, in a stable order."""
    return sorted(path.parent for path in (jobs_dir / name).glob("*/result.json")
                  if not infra_error(path.parent))


def set_aside(jobs_dir: Path, name: str) -> list[Path]:
    """Move a job's infrastructure errors to failed/, so resuming the job runs them again."""
    moved = []
    for result in sorted((jobs_dir / name).glob("*/result.json")):
        if infra_error(result.parent):
            target = jobs_dir / "failed" / name / result.parent.name
            target.parent.mkdir(parents=True, exist_ok=True)
            moved.append(result.parent.rename(target))
    return moved


class HarborRunner:
    """Runs jobs side by side; a job that already has all its trials is not run again."""

    def __init__(self, jobs_dir: Path, env: str, command: Sequence[str] = HARBOR,
                 dry_run: bool = False, retry_delay: float = RETRY_DELAY):
        self.jobs_dir, self.env, self.command, self.dry_run = jobs_dir, env, tuple(command), dry_run
        self.retry_delay = retry_delay

    def run(self, jobs: Sequence[Job]) -> dict[str, list[Path]]:
        for attempt in range(1 + RETRIES):
            pending = [job for job in jobs if len(finished(self.jobs_dir, job.name)) < job.attempts]
            if not pending:
                break
            if any([set_aside(self.jobs_dir, job.name) for job in pending]) and attempt:
                time.sleep(self.retry_delay)
            self._run_once(pending)
        trials = {job.name: finished(self.jobs_dir, job.name)[: job.attempts] for job in jobs}
        short = [job.name for job in jobs if len(trials[job.name]) < job.attempts]
        if short:
            raise RuntimeError(f"harbor jobs still short of trials after {RETRIES} reruns: {short}; "
                               f"see {self.jobs_dir / 'logs'}")
        return trials

    def _run_once(self, jobs: Sequence[Job]) -> None:
        """Start every job, then wait for all of them.

        Harbor's exit code is not trusted; the trials on disk are.
        """
        started = []
        for job in jobs:
            config = self.jobs_dir / "configs" / f"{job.name}.json"
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(json.dumps(job_config(job, self.jobs_dir, self.env), indent=2) + "\n")
            command = [*self.command, "run", "-c", str(config)]
            if self.dry_run:
                print(shlex.join(command))
                continue
            log = self.jobs_dir / "logs" / f"{job.name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as handle:
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                           env=environment())
            started.append(process)
        if self.dry_run:
            raise DryRun("wrote the job configs above and stopped before running them")
        for process in started:
            process.wait()
