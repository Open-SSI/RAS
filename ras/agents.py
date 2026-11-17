"""Harbor agents that run Codex or Claude Code with a RAS harness mounted.

Use them from a job config: ``import_path = "ras.agents:HarnessedCodex"``, with
``kwargs.harness_dir`` set to a harness tree on the host. The stock tree uploads
nothing, so Base runs exactly Harbor's own agent.

A harness that owns a modeling procedure applies it after the agent: once the agent's
run has returned, the packed procedure is uploaded (it does not exist in the container
while the agent works), fitted on the run's own measurements, and its model delivered in
place of the agent's; the lab scores the latest submission. What happened is written to
``delivery.txt`` in the agent's logs. Before that, every JavaScript file the agent wrote
during its run is saved to ``agent_models/`` (named by SHA-256), so the record can keep the
model the agent itself delivered (evidence.record): the same trajectory scored with the
agent's own model is the run of the harness without the procedure (recursion._shadows).
"""

from __future__ import annotations

import shlex
from pathlib import Path

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment

from .harness import DELIVERY_MOUNT, MOUNT, WORKDIR, delivery, mount, planned

DELIVER = (f"python3 {DELIVERY_MOUNT}/deliver_model.py --out {DELIVERY_MOUNT}/final_model.js"
           f" && asb deliver {DELIVERY_MOUNT}/final_model.js")
DELIVERY_TIMEOUT = 1800  # seconds to fit and deliver the procedure's model
START = "/opt/ras-start"  # touched at setup: the agent's files are the ones newer than it
CAPTURE = ("mkdir -p /opt/ras-agent && find /app /tmp /root /home /var/tmp -xdev -type f -name '*.js' -size -4097k "
           f"-newer {START} 2>/dev/null | head -500 | while read -r f; do "
           "cp \"$f\" /opt/ras-agent/$(sha256sum \"$f\" | cut -c1-64).js; done; true")


class _Harnessed:
    INSTRUCTIONS = "AGENTS.md"     # read by the agent from its working directory
    SUBAGENTS: str | None = None   # where the agent looks for subagent files, if it has them

    def __init__(self, *args, harness_dir: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.harness_dir = Path(harness_dir) if harness_dir else None
        self.delivery: Path | None = None

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        if self.harness_dir is None:
            return
        uploads = mount(self.harness_dir, self.logs_dir / "setup" / "harness", self.INSTRUCTIONS)
        self.delivery = delivery(self.harness_dir, self.logs_dir / "setup")
        if self.delivery is not None:
            await self.exec_as_root(environment, command=f"mkdir -p /opt && touch {START}")
        for local, remote in uploads:
            if local.is_dir():
                await environment.upload_dir(local, remote)
            else:
                await environment.upload_file(local, remote)
        remotes = [remote for _, remote in uploads]
        if f"{MOUNT}/skills" in remotes:
            # The parent agent copies skills_dir into its own skills directory when it runs.
            self.skills_dir = f"{MOUNT}/skills"
        if f"{MOUNT}/agents" in remotes and self.SUBAGENTS:
            target = shlex.quote(self.SUBAGENTS)
            await self.exec_as_root(environment, command=f"mkdir -p {target} && cp {MOUNT}/agents/*.md {target}/")
            remotes.append(self.SUBAGENTS)
        if remotes and environment.default_user is not None:
            # Uploads land as root; the agent runs as the environment's default user.
            owner = shlex.quote(str(environment.default_user))
            await self.exec_as_root(environment, command=f"chown -R {owner} {shlex.join(remotes)}")


    async def run(self, instruction, environment: BaseEnvironment, context) -> None:
        await super().run(instruction, environment, context)
        if self.delivery is None and not (self.harness_dir is not None and planned(self.harness_dir)):
            return
        await self.exec_as_root(environment, command=CAPTURE + "; mkdir -p /app/plan", timeout_sec=300)
        await environment.download_dir("/opt/ras-agent", self.logs_dir / "agent_models")
        await environment.download_dir("/app/plan", self.logs_dir / "plan")  # the planner's proposals, if it ran
        if self.delivery is None:  # a planner that leaves the delivered model to the agent
            return
        await environment.upload_dir(self.delivery, DELIVERY_MOUNT)
        result = await self.exec_as_root(environment, command=DELIVER, timeout_sec=DELIVERY_TIMEOUT)
        (self.logs_dir / "delivery.txt").write_text(
            f"exit {result.return_code}\n{result.stdout or ''}\n{result.stderr or ''}")


class HarnessedCodex(_Harnessed, Codex):
    INSTRUCTIONS = "AGENTS.md"


class HarnessedClaudeCode(_Harnessed, ClaudeCode):
    INSTRUCTIONS = "CLAUDE.md"
    SUBAGENTS = f"{WORKDIR}/.claude/agents"

    def __init__(self, *args, subscription: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.subscription = subscription

    def _should_force_oauth(self) -> bool:
        # A kwarg rather than CLAUDE_FORCE_OAUTH in the agent's env: Harbor scrubs every value of a
        # sensitive env key from the trial's files, and "1" is in nearly all of them.
        return self.subscription or super()._should_force_oauth()
