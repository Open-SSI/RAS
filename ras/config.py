"""Study configuration: the frozen benchmark, the protocol, the executors and the campaigns.

Everything lives in ``config/*.toml``. Campaign paths are relative to the
campaign's task directory in the Autonomous Science Bench checkout.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


@dataclass(frozen=True)
class Protocol:
    steps: int
    recursions: int
    base_runs: int
    eval_runs: int
    probe_runs: int
    composed_runs: int
    confirm_runs: int = 0  # fresh runs of an uncertain winner, before it is kept
    base_eval_runs: int | None = None  # fresh stock-harness runs beside the evaluation; None: eval_runs; 0: Base is the opening runs
    selection: str = "rule"  # "rule": feedback.choose on two-run means; "sequential": screen, confirm, decide
    editors: str = "bottleneck"  # which editors a sequential step runs: "bottleneck" (_editors), "modeling", or "none"
    planner: bool = False  # add a model-guided planning branch wherever a validated procedure exists


@dataclass(frozen=True)
class Executor:
    name: str
    agent: str                 # Harbor import path of the harnessed agent
    model: str
    kwargs: dict
    env: dict
    improver: tuple[str, ...]  # CLI command for improver sessions, run in the sandbox
    improver_env: dict = field(default_factory=dict)  # secrets for the sandbox, as ${VAR} templates


@dataclass(frozen=True)
class Objective:
    field: str
    higher_is_better: bool = True
    zero_above: tuple[str, float] | None = None  # the objective is 0 when this field exceeds the limit


@dataclass(frozen=True)
class SeedFile:
    file: Path
    input: str  # the column holding what predict() receives


@dataclass(frozen=True)
class Fit:
    featurizer: str
    group: tuple[str, ...] = ()  # columns whose groups each get their own target scale


@dataclass(frozen=True)
class Scan:
    file: Path
    columns: tuple[str, ...]


@dataclass(frozen=True)
class Contract:
    """What the verifier does with a delivered model: scores ``designs`` inputs, each twice in a fresh worker,
    within ``timeout_s``, from a source of at most ``max_bytes``; ``inputs`` names the public evaluation list
    where the campaign publishes one."""
    designs: int
    timeout_s: float
    max_bytes: int
    inputs: Path | None = None
    column: str | None = None


@dataclass(frozen=True)
class Campaign:
    name: str
    task: Path
    catalog: Path
    id: str
    input: str
    groups: tuple[str, ...]
    objective: Objective
    discovery: tuple[str, ...]
    gates: dict[str, float]
    hit: tuple[str, float] | None   # ("at_least" | "above", threshold)
    random_hit_rate: float | None
    timeout_multiplier: float
    scan: tuple[Scan, ...]
    seed: tuple[SeedFile, ...] = ()
    learning_gate: tuple[str, float] | None = None  # the verifier's learning metric and its threshold
    model_ms: float | None = None  # scoring time per design a delivered model may take, with margin
    fit: Fit | None = None  # None: no modeling procedure for learning feedback yet
    ranking: dict | None = None  # the learning requirement's ranking metric, as the instruction defines it (ranking.py)
    contract: Contract | None = None  # the verifier's limits on a delivered model (contract.py)
    plan: dict | None = None  # the planner's configuration: the instruction's counting rules, restated (planner/)


@dataclass(frozen=True)
class Study:
    benchmark: Path
    commit: str
    harbor_env: str
    protocol: Protocol
    executors: dict[str, Executor]
    campaigns: dict[str, Campaign]

    @property
    def runtime(self) -> Path:
        """The benchmark's shared runtime: the verifier helpers and the JS sandbox."""
        return self.benchmark / "runtime"


def load(config_dir: Path = CONFIG_DIR) -> Study:
    study = _toml(config_dir / "study.toml")
    benchmark = (config_dir.parent / study["benchmark"]["root"]).resolve()
    executors = {
        name: Executor(name, spec["agent"], spec["model"], dict(spec.get("kwargs", {})),
                       {key: str(value) for key, value in spec.get("env", {}).items()},
                       tuple(spec["improver"]), dict(spec.get("improver_env", {})))
        for name, spec in _toml(config_dir / "executors.toml")["executors"].items()
    }
    campaigns = {
        name: _campaign(name, spec, benchmark)
        for name, spec in _toml(config_dir / "campaigns.toml")["campaigns"].items()
    }
    return Study(benchmark, study["benchmark"]["commit"], study["harbor"]["env"],
                 Protocol(**study["protocol"]), executors, campaigns)


def _campaign(name: str, spec: dict, benchmark: Path) -> Campaign:
    task = benchmark / spec["task"]
    objective = spec["objective"]
    zero = objective.get("zero_above")
    gates = {key: float(value) for key, value in spec["gates"].items()}
    if not set(gates) <= set(spec["discovery"]):
        raise ValueError(f"{name}: every discovery gate must be a discovery field")
    hit = next(((kind, float(spec[f"hit_{kind}"])) for kind in ("at_least", "above")
                if f"hit_{kind}" in spec), None)
    return Campaign(
        name=name,
        task=task,
        catalog=task / spec["catalog"],
        id=spec["id"],
        input=spec["input"],
        groups=tuple(spec.get("groups", ())),
        objective=Objective(objective["field"], objective.get("higher_is_better", True),
                            (zero[0], float(zero[1])) if zero else None),
        discovery=tuple(spec["discovery"]),
        gates=gates,
        hit=hit,
        random_hit_rate=spec.get("random_hit_rate"),
        timeout_multiplier=float(spec.get("timeout_multiplier", 0.25)),
        scan=tuple(Scan(task / item["file"], tuple(item["columns"])) for item in spec["scan"]),
        seed=tuple(SeedFile(task / item["file"], item["input"]) for item in spec.get("seed", ())),
        fit=Fit(spec["fit"]["featurizer"], tuple(spec["fit"].get("group", ()))) if "fit" in spec else None,
        learning_gate=next(((k, float(v)) for k, v in spec.get("learning_gate", {}).items()), None),
        model_ms=float(spec["model_ms"]) if "model_ms" in spec else None,
        ranking=dict(spec["ranking"]) if "ranking" in spec else None,
        contract=_contract(spec["contract"], task) if "contract" in spec else None,
        plan=dict(spec["plan"]) if "plan" in spec else None,
    )


def _contract(spec: dict, task: Path) -> Contract:
    return Contract(int(spec["designs"]), float(spec["timeout_s"]), int(spec["max_bytes"]),
                    task / spec["inputs"] if "inputs" in spec else None, spec.get("column"))


def environment(path: Path = ROOT / ".env") -> dict[str, str]:
    """The process environment, with values from .env filled in where a variable is unset."""
    values = dict(os.environ)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                if value:
                    values.setdefault(key.removeprefix("export ").strip(), value)
    return values


def resolve(templates: dict[str, str], environ: dict[str, str]) -> dict[str, str]:
    """Fill ${VAR} templates from ``environ``; a variable that is unset or empty is an error."""
    def value(match: re.Match) -> str:
        if not environ.get(match.group(1)):
            raise SystemExit(f"{match.group(1)} is not set; add it to .env")
        return environ[match.group(1)]
    return {key: re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value, template)
            for key, template in templates.items()}


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())
