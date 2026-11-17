"""Checks a harness tree must pass before it runs.

``scan`` keeps answers out of the harness: no design identifier, sequence or
composition from the campaign's files, no URL or DOI, and outside the modeling procedure
(``scripts/model/``, which the procedure audit covers) no prebuilt JavaScript predictor and
no table of decimal values, the forms a model fitted on earlier runs' measurements would take. ``validate`` keeps it
runnable: every text component is present, skills and subagents declare a name
and description, scripts compile, and the tree is small UTF-8 text with no links.
"""

from __future__ import annotations

import csv
import functools
import re
import subprocess
from pathlib import Path

from .config import Campaign, Scan
from .harness import TEXT_COMPONENTS, is_placeholder

TOKEN = re.compile(r"[A-Za-z0-9_.|+\-]+")
LINK = re.compile(r"https?://|\bwww\.|\bdoi\s*:|\b10\.\d{4,9}/", re.IGNORECASE)
FRONT_MATTER = re.compile(r"---\n(.*?)\n---\n", re.DOTALL)
MIN_TOKEN = 4
MAX_BYTES = 2 * 1024 * 1024
PREDICTOR = re.compile(r"function\s+predict\s*\(")
DECIMAL = re.compile(r"(?<![\w.])-?\d+\.\d+(?:[eE][-+]?\d+)?")
MAX_DECIMALS = 200  # room for general constants (element or residue properties), not for a fitted model
PROCEDURE = "scripts/model/"  # the modeling procedure, which the shuffled-target audit checks instead


def check(tree: Path, campaign: Campaign) -> list[str]:
    return validate(tree) + scan(tree, campaign)


def report(problems: list[str]) -> str:
    """GUARD.md for a session's retry."""
    lines = "\n".join(f"- {problem}" for problem in problems)
    return f"# The harness was rejected\n\nFix each problem below, then finish as before.\n\n{lines}\n"


def scan(tree: Path, campaign: Campaign) -> list[str]:
    forbidden = _forbidden(campaign.scan)
    problems = []
    for path in _files(tree):
        text = path.read_text(errors="replace")
        name = path.relative_to(tree).as_posix()
        hits = sorted({token.strip(".-|+") for token in TOKEN.findall(text)} & forbidden)
        if hits:
            problems.append(f"{name}: names {len(hits)} design(s) from this campaign, such as {hits[0]}")
        if LINK.search(text):
            problems.append(f"{name}: contains a URL or DOI")
        if name.startswith(PROCEDURE):
            continue
        if path.suffix == ".js" and PREDICTOR.search(text):
            problems.append(f"{name}: a prebuilt predictor; the delivered model must be fitted during the run, "
                            f"from that run's own data")
        decimals = len(DECIMAL.findall(text))
        if decimals > MAX_DECIMALS:
            problems.append(f"{name}: carries {decimals} decimal values, a table of measured or fitted values; the "
                            f"harness may hold general scientific knowledge, but the values a model learns must come "
                            f"from the run's own data")
    return problems


def validate(tree: Path) -> list[str]:
    problems = []
    files = _files(tree)
    problems += [f"{_rel(path, tree)}: links are not allowed" for path in tree.rglob("*") if path.is_symlink()]
    if sum(path.stat().st_size for path in files) > MAX_BYTES:
        problems.append(f"the harness is larger than {MAX_BYTES // 1024**2} MB")
    for name, _, _ in TEXT_COMPONENTS:
        if not (tree / name).is_file():
            problems.append(f"{name}: missing; keep every component file, with its placeholder if unused")
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{_rel(path, tree)}: not UTF-8 text")
            continue
        if path.suffix == ".py":
            try:
                compile(text, str(path), "exec")
            except SyntaxError as error:
                problems.append(f"{_rel(path, tree)}: does not compile ({error.msg}, line {error.lineno})")
        elif path.suffix == ".sh" and subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode:
            problems.append(f"{_rel(path, tree)}: shell syntax error")
    skills = tree / "skills"
    for skill in sorted(path for path in skills.iterdir() if path.is_dir()) if skills.is_dir() else []:
        problems += _front_matter(skill / "SKILL.md", tree)
    for agent in sorted((tree / "agents").glob("*.md")):
        if not is_placeholder(agent):
            problems += _front_matter(agent, tree)
    return problems


@functools.cache
def _forbidden(scans: tuple[Scan, ...]) -> frozenset[str]:
    tokens: set[str] = set()
    for item in scans:
        with item.file.open(newline="") as handle:
            for row in csv.DictReader(handle):
                tokens.update(row[column] for column in item.columns)
    return frozenset(token for token in tokens if len(token) >= MIN_TOKEN and not _number(token))


def _front_matter(path: Path, tree: Path) -> list[str]:
    if not path.is_file():
        return [f"{_rel(path, tree)}: missing"]
    match = FRONT_MATTER.match(path.read_text(errors="replace"))
    fields = {}
    for line in match.group(1).splitlines() if match else []:
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    missing = [key for key in ("name", "description") if not fields.get(key)]
    return [f"{_rel(path, tree)}: front matter needs {' and '.join(missing)}"] if missing else []


def _files(tree: Path) -> list[Path]:
    return sorted(path for path in tree.rglob("*") if path.is_file() and not path.is_symlink())


def _rel(path: Path, tree: Path) -> str:
    return path.relative_to(tree).as_posix()


def _number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True
