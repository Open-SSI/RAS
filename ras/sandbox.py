"""The improver sandbox: a container that sees only its working directory and reaches only the model APIs.

Sessions run in the ``ras-improver`` image, built from the campaigns' own agent
image recipe plus the executors' CLIs, on an internal Docker network whose only
way out is an allowlist proxy for the model APIs. The CLIs' own sandboxes are
off, as Harbor runs them in trials: the container is the sandbox.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from .config import ROOT

IMAGE = "ras-improver:0.1.0"
PROXY_IMAGE = "ras-improver-proxy:0.1.0"
NETWORK = "ras-improver"
PROXY = "ras-improver-proxy"
PROXY_URL = f"http://{PROXY}:8888"
HOME = "/home/ras"


def ensure() -> None:
    """Build the images if missing, then start the internal network and its proxy."""
    for tag, context in ((IMAGE, "improver"), (PROXY_IMAGE, "proxy")):
        _image(tag, context)
    if not _ok("network", "inspect", NETWORK):
        subprocess.run(["docker", "network", "create", "--internal", NETWORK], check=True, capture_output=True)
    if not _ok("container", "inspect", PROXY):
        subprocess.run(["docker", "run", "-d", "--restart", "unless-stopped", "--name", PROXY, PROXY_IMAGE],
                       check=True, capture_output=True)
        subprocess.run(["docker", "network", "connect", NETWORK, PROXY], check=True, capture_output=True)
    elif "true" not in _inspect(PROXY, "{{.State.Running}}"):
        subprocess.run(["docker", "start", PROXY], check=True, capture_output=True)


def command(workspace: Path, argv: list[str], secrets: Mapping[str, str]) -> list[str]:
    """``docker run`` for one session. Secret values travel in the environment, never in argv."""
    # CODEX_HOME must exist before Codex starts; the throwaway home always does.
    # PYTHONDONTWRITEBYTECODE keeps an editor's test runs from leaving .pyc files in the harness.
    variables = {"HOME": HOME, "CODEX_HOME": HOME, "PYTHONDONTWRITEBYTECODE": "1", "HTTPS_PROXY": PROXY_URL,
                 "HTTP_PROXY": PROXY_URL, "https_proxy": PROXY_URL, "http_proxy": PROXY_URL}
    return [
        "docker", "run", "--rm", "--network", NETWORK,
        "--user", f"{os.getuid()}:{os.getgid()}", "--tmpfs", f"{HOME}:rw,exec,mode=1777",
        *[part for key, value in variables.items() for part in ("-e", f"{key}={value}")],
        *[part for key in secrets for part in ("-e", key)],
        "-v", f"{workspace.resolve()}:/work", "-w", "/work", IMAGE, *argv,
    ]


def offline(workdir: Path, argv: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    """Run a command in the sandbox image with no network at all, seeing only ``workdir``."""
    _image(IMAGE, "improver")
    return subprocess.run([
        "docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
        "--tmpfs", f"{HOME}:rw,exec,mode=1777", "-e", f"HOME={HOME}",
        "-v", f"{workdir.resolve()}:/work", "-w", "/work", IMAGE, *argv,
    ], capture_output=True, text=True, timeout=timeout)


def _image(tag: str, context: str) -> None:
    if not _ok("image", "inspect", tag):
        subprocess.run(["docker", "build", "-t", tag, str(ROOT / "docker" / context)], check=True)


def _ok(*args: str) -> bool:
    return subprocess.run(["docker", *args], capture_output=True).returncode == 0


def _inspect(container: str, template: str) -> str:
    return subprocess.run(["docker", "inspect", "-f", template, container], capture_output=True, text=True).stdout
