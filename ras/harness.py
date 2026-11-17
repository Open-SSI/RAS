"""The editable harness: Harness²'s components as files in one directory tree.

Text components are always on: they are rendered, in order, into the file the
agent reads from its working directory (AGENTS.md for Codex, CLAUDE.md for
Claude Code). Skills, scripts and subagents are mounted under /app/harness. A
component that still holds its placeholder is left out, so the stock tree H0
mounts nothing and Base runs the stock agent.

The modeling stage owns its own files (``MODELING``): ``modeling.md``, the always-on
text that says how the delivered model is built, the procedure under ``scripts/model/``
and its skill. Everything else is the experiment plan and the shared prompts. Keeping the
two apart lets an edit to one be attributed, and lets a modeling edit be scored on the
runs another harness with the same plan already made (replay.py).

A harness that owns a procedure (``scripts/model/fit.py``) applies it itself: none of the
modeling files reach the agent, which runs the campaign as it would without them, and
once the agent has finished, the harness uploads the packed procedure (``delivery``) and
delivers the model it fits on the run's own measurements in place of the agent's (the lab
scores the latest submission; ras.agents). Agents that knew a procedure would build their
model, whether or not they could read it, planned differently and chained their hits into
fewer families (on nuclease), so a change to the modeling files
is confined to the delivered model only when the agent never learns of it.
"""

from __future__ import annotations

import difflib
import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

PLACEHOLDER = "<!-- ras:placeholder -->"
WORKDIR = "/app"
MOUNT = f"{WORKDIR}/harness"

# (file, heading, what the lever is for), in rendering order.
TEXT_COMPONENTS = (
    ("system_prompt.md", "System prompt", "Persona and top-level procedure followed on every campaign."),
    ("guardrails.md", "Guardrails", "Hard constraints that hold regardless of campaign or tool."),
    ("memory.md", "Memory", "Durable lessons about running campaigns, not facts about this one."),
    ("tool_guidance.md", "Tool guidance", "When and how to use asb, the shell and the harness scripts."),
    ("modeling.md", "Delivered model", "How the delivered model is built and checked; it governs the delivered "
                                       "model over anything above."),
    ("campaign_notes.md", "Campaign notes", "The only place for what was found in this campaign."),
)
# (directory, what the lever is for)
DIRECTORIES = (
    ("skills", "One directory per skill with a SKILL.md whose front matter gives name and "
               "description; loaded only when invoked."),
    ("scripts", "Deterministic tools the agent runs from the shell instead of reasoning them "
                "out token by token."),
    ("agents", "Subagents, one Markdown file each whose front matter gives name and description "
               "(Claude Code only)."),
)


# The modeling stage's files: modeling.md and everything under these directories.
MODELING = ("modeling.md", "scripts/model/", "skills/modeling-procedure/")
WRAPPER = Path(__file__).with_name("deliver_model.py")  # uploaded beside the packed procedure after the agent finishes
PACKAGE = "model.tar.gz"  # the packed procedure
DELIVERY_MOUNT = "/opt/ras-delivery"  # where the harness puts both, once the agent's run has returned
PLANNER = Path(__file__).with_name("planner")  # model-guided planning: acquire.py and score.js, mounted in scripts/plan/
PLAN_DIR = "scripts/plan"
PLANNING = (
    "Round 1 is yours: choose and order it as you would. From round 2 on, before you order each round, run\n\n"
    "```bash\npython3 /app/harness/scripts/plan/acquire.py --out /app/round_N.csv\n```\n\n"
    "with N the round. It refits this harness's model (`/app/harness/scripts/model/`) on the public seed and every "
    "measurement so far, with bootstrap refits for its uncertainty, and proposes the round: the designs with the highest "
    "predicted value plus uncertainty, and a quarter chosen for their uncertainty among those predicted above the "
    "median, within the campaign's own counting rules. When it writes the file, order exactly that file with `asb run`; "
    "do not edit it. When it reports insufficient evidence, choose the round yourself as you otherwise would. It takes "
    "a few minutes. After the last round, build and deliver your final model as you usually would.")


def is_modeling(name: str) -> bool:
    """Whether a harness file (a path relative to the tree) belongs to the modeling stage."""
    return name in MODELING or name.startswith(tuple(m for m in MODELING if m.endswith("/")))


def write_stock(path: Path) -> Path:
    """Write H0: every component present, holding a placeholder that names its lever."""
    path.mkdir(parents=True)
    for name, heading, purpose in TEXT_COMPONENTS:
        (path / name).write_text(f"{PLACEHOLDER}\n<!-- {heading}: {purpose} -->\n")
    for name, purpose in DIRECTORIES:
        (path / name).mkdir()
        (path / name / "README.md").write_text(f"{PLACEHOLDER}\n<!-- {purpose} -->\n")
    return path


def complete(tree: Path) -> Path:
    """Give a tree written before a component existed that component, holding its placeholder."""
    for name, heading, purpose in TEXT_COMPONENTS:
        if not (tree / name).exists():
            (tree / name).write_text(f"{PLACEHOLDER}\n<!-- {heading}: {purpose} -->\n")
    return tree


def is_placeholder(path: Path) -> bool:
    return path.read_bytes().startswith(PLACEHOLDER.encode())


def packaged(tree: Path) -> bool:
    """Whether the harness owns a modeling procedure, which the harness applies itself after the agent's run."""
    return (tree / "scripts" / "model" / "fit.py").is_file()


def planned(tree: Path) -> bool:
    """Whether the harness plans its rounds with its own model (scripts/plan/acquire.py), which needs the procedure."""
    return (tree / PLAN_DIR / "acquire.py").is_file()


def install_planner(tree: Path, config: dict) -> Path:
    """Add model-guided planning to a tree that owns a procedure: the planner and its configuration, ``plan.json``."""
    target = tree / PLAN_DIR
    target.mkdir(parents=True, exist_ok=True)
    for name in ("acquire.py", "score.js"):
        shutil.copyfile(PLANNER / name, target / name)
    (target / "plan.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return tree


def _shown(tree: Path, directory: str) -> list[Path]:
    """A component directory's files the agent sees: none of the modeling files when the harness owns a procedure,
    unless it plans with it."""
    hidden = packaged(tree) and not planned(tree)
    return [path for path in _content(tree / directory) if not (hidden and is_modeling(path.relative_to(tree).as_posix()))]


def render_instructions(tree: Path) -> str:
    """The always-on text for the agent's instruction file; empty for the stock tree. With a procedure, modeling.md
    is left out, like the rest of the modeling files."""
    sections = [
        f"## {heading}\n\n{(tree / name).read_text().strip()}"
        for name, heading, _ in TEXT_COMPONENTS
        if (tree / name).is_file() and not is_placeholder(tree / name) and not (name == "modeling.md" and packaged(tree))
    ]
    if planned(tree):
        sections.append(f"## Experiment planning\n\n{PLANNING}")
    if _shown(tree, "scripts"):
        sections.append(f"## Harness scripts\n\n`{MOUNT}/scripts/` holds this harness's scripts; "
                        "run them from the shell.")
    return "\n\n".join(sections) + "\n" if sections else ""


def mount(tree: Path, staging: Path, instructions: str) -> list[tuple[Path, str]]:
    """Stage what the container needs from ``tree``; return (local, remote) pairs to upload.

    Placeholders are not staged, so the stock tree yields no uploads at all.
    """
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    uploads = []
    for name, _ in DIRECTORIES:
        files = _shown(tree, name)
        for path in files:
            target = staging / name / path.relative_to(tree / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        if files:
            uploads.append((staging / name, f"{MOUNT}/{name}"))
    text = render_instructions(tree)
    if text:
        (staging / instructions).write_text(text)
        uploads.append((staging / instructions, f"{WORKDIR}/{instructions}"))
    return uploads


def visible(tree: Path) -> str:
    """SHA-256 over everything of a tree the agent sees: its instruction text and its mounted files. Two harnesses with
    the same visible digest give the agent the same run; with a procedure, they differ only in the delivered model."""
    out = hashlib.sha256(render_instructions(tree).encode() + b"\0")
    for name, _ in DIRECTORIES:
        for path in _shown(tree, name):
            out.update(path.relative_to(tree).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return out.hexdigest()


def delivery(tree: Path, staging: Path) -> Path | None:
    """Stage what the harness uploads to DELIVERY_MOUNT once the agent has finished: the packed procedure and the
    wrapper that fits and writes its model. None for a harness without a procedure."""
    if not packaged(tree) or (planned(tree) and json.loads((tree / PLAN_DIR / "plan.json").read_text()).get("deliver") is False):
        return None  # without a procedure, or one that only plans (plan.json "deliver": false): the agent's model stands
    target = staging / "delivery"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    (target / PACKAGE).write_bytes(pack(tree))
    shutil.copyfile(WRAPPER, target / WRAPPER.name)
    return target


def pack(tree: Path) -> bytes:
    """The procedure under scripts/model/ as a gzipped tar with the root ``model/``, byte for byte reproducible."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as package:
        root = tree / "scripts" / "model"
        for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            info = tarfile.TarInfo(f"model/{path.relative_to(root).as_posix()}")
            data = path.read_bytes()
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            package.addfile(info, io.BytesIO(data))
    return gzip.compress(buffer.getvalue(), mtime=0)


def copy(source: Path, target: Path) -> Path:
    shutil.copytree(source, target, symlinks=True)
    return target


def files(tree: Path) -> list[str]:
    return sorted(path.relative_to(tree).as_posix() for path in tree.rglob("*") if path.is_file())


def digest(tree: Path, part: str = "all") -> str:
    """SHA-256 over a tree's files, paths and contents: all of them, or only its ``modeling`` or ``planning``
    files (planning: everything outside MODELING). Two harnesses with the same planning digest give the agent
    the same instructions for everything but the delivered model."""
    keep = {"all": lambda name: True, "modeling": is_modeling, "planning": lambda name: not is_modeling(name)}[part]
    out = hashlib.sha256()
    for name in files(tree):
        if keep(name):
            out.update(name.encode() + b"\0" + (tree / name).read_bytes() + b"\0")
    return out.hexdigest()


def graft(planning: Path, modeling: Path, target: Path) -> Path:
    """A new tree with ``planning``'s plan and shared prompts and ``modeling``'s modeling files."""
    copy(planning, target)
    for name in files(target):
        if is_modeling(name):
            (target / name).unlink()
    for name in files(modeling):
        if is_modeling(name):
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(modeling / name, target / name)
    return complete(target)


def revert(current: Path, tree: Path, keep) -> list[str]:
    """Put back every file of ``tree`` outside its editor's scope (``keep(name)`` false) as it is in ``current``;
    return the files put back."""
    reverted = []
    for name in sorted(set(files(current)) | set(files(tree))):
        if keep(name):
            continue
        before, after = current / name, tree / name
        if before.is_file() and after.is_file() and before.read_bytes() == after.read_bytes():
            continue
        reverted.append(name)
        if before.is_file():
            after.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(before, after)
        else:
            after.unlink(missing_ok=True)
    return reverted


def diff(parent: Path, child: Path) -> str:
    """Unified diff over every file of two trees."""
    chunks: list[str] = []
    for name in sorted(set(files(parent)) | set(files(child))):
        before, after = _lines(parent / name), _lines(child / name)
        if before != after:
            chunks.extend(difflib.unified_diff(before, after, f"a/{name}", f"b/{name}"))
    return "".join(chunks)


def _content(directory: Path) -> list[Path]:
    """Files in a component directory other than its placeholder."""
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file() and not is_placeholder(path))


def _lines(path: Path) -> list[str]:
    return path.read_text(errors="replace").splitlines(keepends=True) if path.is_file() else []
