"""Smoke trial: one real campaign run with a marker harness, to check the mount end to end.

Before the campaign, the marker harness asks the agent to print a code kept in a
mounted script file and to name the skills and subagents it can use. Finding the
code and the names in the agent's logs shows that the instruction file was read,
the scripts were mounted, and the skills and subagents were registered. The run
then goes through the same record and feedback steps as a recursion's runs.
"""

from __future__ import annotations

from pathlib import Path

from . import evidence, feedback, guard, harness, procedure
from .config import Campaign, Executor, Study
from .runner import HarborRunner, Job

CODE = "RAS-SMOKE-7f3a"
SKILL = "ras-smoke-skill"
SUBAGENT = "ras-smoke-agent"
PROMPT = ("Before you start the campaign, run `cat /app/harness/scripts/smoke.txt` and report what it prints, "
          "then name every skill and subagent you can use, in one line. Then run the campaign as instructed.\n")


def marker_harness(path: Path) -> Path:
    tree = harness.write_stock(path)
    (tree / "system_prompt.md").write_text(PROMPT)
    (tree / "scripts" / "smoke.txt").write_text(f"{CODE}\n")
    (tree / "skills" / SKILL).mkdir()
    (tree / "skills" / SKILL / "SKILL.md").write_text(
        f"---\nname: {SKILL}\ndescription: Confirms that harness skills load. Not needed for the campaign.\n"
        "---\nNothing to do.\n")
    (tree / "agents" / f"{SUBAGENT}.md").write_text(
        f"---\nname: {SUBAGENT}\ndescription: Confirms that harness subagents load. Not needed for the "
        "campaign.\n---\nReply with one word: loaded.\n")
    return tree


def run(study: Study, executor: Executor, campaign: Campaign, root: Path, env: str) -> dict:
    """Run the smoke trial, or reuse a finished one, and report what the agent's logs show."""
    base = root / "smoke" / executor.name / campaign.name
    stock = base / "h0" if (base / "h0").exists() else harness.write_stock(base / "h0")
    tree = base / "harness" if (base / "harness").exists() else marker_harness(base / "harness")
    problems = guard.check(tree, campaign)
    if problems:
        raise SystemExit("the marker harness failed the guard:\n" + "\n".join(problems))
    name = f"smoke-{executor.name}-{campaign.name}"
    (trial,) = HarborRunner(root / "jobs", env).run([Job(name, campaign, executor, tree, 1)])[name]
    record = base / "record"
    if not record.exists():
        evidence.record(trial, record, campaign, harness.diff(stock, tree))
    feedback.write(record, campaign, tree, procedure.validate)
    # agent/setup/ holds the staged harness itself, so it would match every marker.
    logs = "\n".join(path.read_text(errors="replace") for path in sorted((trial / "agent").rglob("*"))
                     if path.is_file() and path.relative_to(trial / "agent").parts[0] != "setup")
    checks = {"instruction file read, scripts mounted": CODE in logs, "skills registered": SKILL in logs}
    if executor.agent.endswith(":HarnessedClaudeCode"):  # Codex has no subagents
        checks["subagents registered"] = SUBAGENT in logs
    return {
        "trial": trial,
        "checks": checks,
        "stats": evidence.read_json(record / "stats.json"),
        "discovery": evidence.read_json(record / "discovery.json")["fields"],
        "verifier": evidence.read_json(record / "sealed" / "metrics.json"),
    }
