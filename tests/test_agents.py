"""The harnessed agents against a fake environment; the parent agents' own setup is stubbed out."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex

from ras import harness
from ras.agents import HarnessedClaudeCode, HarnessedCodex


class FakeEnvironment:
    default_user = "agent"

    def __init__(self):
        self.uploads: dict[str, list[str] | str] = {}
        self.commands: list[str] = []

    async def upload_dir(self, source, target):
        self.uploads[target] = sorted(p.relative_to(source).as_posix() for p in Path(source).rglob("*") if p.is_file())

    async def upload_file(self, source, target):
        self.uploads[target] = Path(source).read_text()

    async def download_dir(self, source, target):
        self.downloads = getattr(self, "downloads", []) + [(source, str(target))]
        Path(target).mkdir(parents=True, exist_ok=True)

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        self.commands.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")


@pytest.fixture(autouse=True)
def stub_parent_setup(monkeypatch):
    async def setup(self, environment):
        return None
    monkeypatch.setattr(Codex, "setup", setup)
    monkeypatch.setattr(ClaudeCode, "setup", setup)


def tree_with_content(path: Path) -> Path:
    tree = harness.write_stock(path)
    (tree / "system_prompt.md").write_text("Plan all rounds first.\n")
    (tree / "skills" / "batch").mkdir()
    (tree / "skills" / "batch" / "SKILL.md").write_text("---\nname: batch\ndescription: pick a batch\n---\n")
    (tree / "agents" / "critic.md").write_text("---\nname: critic\ndescription: review a batch\n---\n")
    return tree


def setup(agent_class, tmp_path: Path, tree: Path):
    agent = agent_class(logs_dir=tmp_path / "logs", model_name="model", harness_dir=str(tree))
    environment = FakeEnvironment()
    asyncio.run(agent.setup(environment))
    return agent, environment


def test_stock_tree_uploads_nothing(tmp_path):
    agent, environment = setup(HarnessedCodex, tmp_path, harness.write_stock(tmp_path / "h0"))
    assert environment.uploads == {} and environment.commands == [] and agent.skills_dir is None


def test_codex_gets_agents_md_and_skills(tmp_path):
    agent, environment = setup(HarnessedCodex, tmp_path, tree_with_content(tmp_path / "h1"))
    assert "Plan all rounds first." in environment.uploads["/app/AGENTS.md"]
    assert environment.uploads["/app/harness/skills"] == ["batch/SKILL.md"]
    assert agent.skills_dir == "/app/harness/skills"
    assert not any(".claude/agents" in command for command in environment.commands)  # no subagents in Codex


def test_claude_code_gets_claude_md_and_subagents(tmp_path):
    agent, environment = setup(HarnessedClaudeCode, tmp_path, tree_with_content(tmp_path / "h1"))
    assert "/app/CLAUDE.md" in environment.uploads
    assert any("cp /app/harness/agents/*.md /app/.claude/agents/" in command for command in environment.commands)
    assert any(command.endswith("chown -R agent /app/harness/skills /app/harness/agents /app/CLAUDE.md "
                                "/app/.claude/agents") for command in environment.commands)


def test_claude_code_subscription_drops_the_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_FORCE_OAUTH", raising=False)
    assert HarnessedClaudeCode(logs_dir=tmp_path, model_name="model", subscription=True)._should_force_oauth()
    assert not HarnessedClaudeCode(logs_dir=tmp_path, model_name="model")._should_force_oauth()
