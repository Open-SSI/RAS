from pathlib import Path

import pytest

from ras import sandbox
from ras.config import resolve


def test_sessions_see_only_their_workspace_and_secrets_stay_out_of_argv(tmp_path: Path):
    argv = sandbox.command(tmp_path, ["codex", "exec", "go"], {"CODEX_API_KEY": "sk-secret"})
    assert "sk-secret" not in " ".join(argv)
    assert argv[argv.index("-e", argv.index("CODEX_API_KEY") - 1) + 1] == "CODEX_API_KEY"
    mounts = [argv[i + 1] for i, part in enumerate(argv) if part == "-v"]
    assert mounts == [f"{tmp_path.resolve()}:/work"]
    assert argv[argv.index("--network") + 1] == sandbox.NETWORK
    assert f"HTTPS_PROXY={sandbox.PROXY_URL}" in argv
    assert argv[-3:] == ["codex", "exec", "go"]


def test_secret_templates_resolve_or_fail():
    assert resolve({"CODEX_API_KEY": "${OPENAI_API_KEY}"}, {"OPENAI_API_KEY": "k"}) == {"CODEX_API_KEY": "k"}
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set"):
        resolve({"CODEX_API_KEY": "${OPENAI_API_KEY}"}, {"OPENAI_API_KEY": ""})
