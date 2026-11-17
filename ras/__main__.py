"""RAS command line.

  uv run python -m ras check                     # the benchmark is at its frozen commit
  uv run python -m ras h0 DIR                    # write the stock harness tree
  uv run python -m ras guard TREE --campaign biosensor-active-learning
  uv run python -m ras smoke --executor luna              # one real trial with a marker harness
  uv run python -m ras run --executor luna --campaign coercivity-peak-search --recursion 1
  uv run python -m ras status
  uv run python -m ras preflight --env docker    # every campaign and executor, before a study

`run` resumes: each phase of a recursion is skipped once its output exists.
Without --campaign or --recursion it runs every campaign and recursion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, guard, harness, preflight, smoke
from .improver import Improver
from .recursion import MODES, RUNS, Recursion, protocol_digest, spent, status
from .runner import DryRun, HarborRunner


def check(study: config.Study) -> list[str]:
    """Problems that would make runs answer a different question from the frozen benchmark."""
    git = ["git", "-C", str(study.benchmark)]
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    problems = []
    if head != study.commit:
        problems.append(f"the benchmark is at {head[:12] or 'no commit'}, frozen at {study.commit[:12]}")
    tasks = [str(campaign.task) for campaign in study.campaigns.values()]
    changed = subprocess.run([*git, "status", "--porcelain", "--", *tasks],
                             capture_output=True, text=True).stdout.strip()
    if changed:
        problems.append(f"campaign files differ from the frozen commit:\n{changed}")
    return problems


def run(args: argparse.Namespace, study: config.Study) -> int:
    problems = check(study)
    if problems and not args.dry_run:
        print("\n".join(problems), file=sys.stderr)
        return 1
    if args.executor not in study.executors:
        raise SystemExit(f"unknown executor {args.executor}; known: {sorted(study.executors)}")
    unknown = sorted(set(args.campaign) - set(study.campaigns))
    if unknown:
        raise SystemExit(f"unknown campaigns {unknown}; known: {sorted(study.campaigns)}")
    executor = study.executors[args.executor]
    if args.editors or args.planner:
        study = replace(study, protocol=replace(study.protocol, editors=args.editors or study.protocol.editors,
                                                planner=args.planner or study.protocol.planner))
    runner = HarborRunner(args.runs / "jobs", args.env or study.harbor_env, dry_run=args.dry_run)
    improver = Improver(executor, dry_run=args.dry_run)
    recursions = [
        Recursion(study, study.campaigns[name], executor, index, runner, improver, args.runs,
                  start=args.start_from, steps=args.steps, mode=mode,
                  allow_protocol_change=args.allow_protocol_change, ceiling=args.ceiling)
        for name in args.campaign or study.campaigns
        for index in args.recursion or range(1, study.protocol.recursions + 1)
        for mode in args.mode or ["recursive"]
    ]
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(recursion.run) for recursion in recursions]
        for recursion, future in zip(recursions, futures):
            try:
                print(f"{recursion.dir.relative_to(args.runs)}: selected harness {future.result().name}")
            except DryRun as stop:
                print(f"{recursion.dir.relative_to(args.runs)}: dry run, {stop}")
    print(f"spent so far under {args.runs}: ${spent(args.runs):.2f}")
    return 0


def run_preflight(args: argparse.Namespace, study: config.Study) -> int:
    """The model-free checks on every campaign, then an improver session and a smoke trial per executor."""
    problems = check(study)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    names = args.campaign or list(study.campaigns)
    executors = [] if args.campaigns_only else args.executor or list(study.executors)
    for name, known in [*((n, study.campaigns) for n in names), *((n, study.executors) for n in executors)]:
        if name not in known:
            raise SystemExit(f"unknown name {name}; known: {sorted(known)}")
    env = args.env or study.harbor_env
    runner = HarborRunner(args.runs / "jobs", env)
    report = {"protocol": protocol_digest(), "campaigns": {}, "executors": {}}
    if not (args.runs / "h0").exists():  # once, before the campaigns' checks share it
        harness.write_stock(args.runs / "h0")
    with ThreadPoolExecutor(PREFLIGHT_AT_ONCE) as pool:
        for name, result in zip(names, pool.map(lambda n: preflight.campaign(study, study.campaigns[n], runner, args.runs), names)):
            report["campaigns"][name] = result
            print(f"{'ok' if result['ok'] else 'FAILED':6s}  {name}: {json.dumps({k: v for k, v in result.items() if k != 'ok'}, default=str)}")
    for name in executors:
        executor = study.executors[name]
        report["executors"][name] = {"improver": preflight.improver(executor, args.runs),
                                     "trial": preflight.trial(study, executor, args.runs, env)}
        for part, result in report["executors"][name].items():
            print(f"{'ok' if result['ok'] else 'FAILED':6s}  {name} {part}: {json.dumps({k: v for k, v in result.items() if k != 'ok'}, default=str)}")
    preflight.write(report, args.runs / "preflight.json")
    passed = all(r["ok"] for r in report["campaigns"].values()) and \
        all(part["ok"] for r in report["executors"].values() for part in r.values())
    print(f"preflight {'passed' if passed else 'FAILED'} for protocol {report['protocol']}; report in {args.runs / 'preflight.json'}")
    return 0 if passed else 1


PREFLIGHT_AT_ONCE = 3  # campaigns checked side by side, each with 3 random-order trials
def run_smoke(args: argparse.Namespace, study: config.Study) -> int:
    problems = check(study)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    for name, known in ((args.executor, study.executors), (args.campaign, study.campaigns)):
        if name not in known:
            raise SystemExit(f"unknown name {name}; known: {sorted(known)}")
    report = smoke.run(study, study.executors[args.executor], study.campaigns[args.campaign],
                       args.runs, args.env or study.harbor_env)
    for check_name, passed in report["checks"].items():
        print(f"{'ok' if passed else 'FAILED'}  {check_name}")
    stats = report["stats"]
    print(f"trial    {report['trial']}")
    print(f"run      {stats['agent_minutes'] or 0:.1f} min, ${stats['cost_usd'] or 0:.2f}, "
          f"{stats['steps']} steps, exception: {stats['exception']}")
    print(f"improver would see  {report['discovery']}")
    print(f"verifier (sealed)   {report['verifier']}")
    return 0 if all(report["checks"].values()) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="the benchmark is at its frozen commit")
    stock = commands.add_parser("h0", help="write the stock harness tree")
    stock.add_argument("path", type=Path)
    scan = commands.add_parser("guard", help="scan and validate a harness tree")
    scan.add_argument("tree", type=Path)
    scan.add_argument("--campaign", required=True)
    go = commands.add_parser("run", help="run or resume recursions")
    go.add_argument("--executor", required=True)
    go.add_argument("--campaign", action="append", default=[], help="campaign name (repeatable)")
    go.add_argument("--recursion", action="append", type=int, default=[], help="recursion index (repeatable)")
    go.add_argument("--env", default=None, help="Harbor environment (default: config/study.toml)")
    go.add_argument("-j", "--jobs", type=int, default=1, help="recursions to run side by side")
    go.add_argument("--steps", type=int, default=None, help="K for these recursions (default: config/study.toml)")
    go.add_argument("--start-from", type=Path, default=None,
                    help="a harness tree to start from instead of H0; H0 still runs as Base")
    go.add_argument("--mode", action="append", choices=MODES, default=[],
                    help="recursive (RAS, the default) or flat (the non-recursive control); repeatable")
    go.add_argument("--allow-protocol-change", action="store_true",
                    help="resume recursions whose recorded protocol differs from the current one")
    go.add_argument("--dry-run", action="store_true", help="write the first job configs and stop")
    go.add_argument("--editors", choices=("bottleneck", "modeling", "none"), default=None,
                    help="which editors a sequential step runs (default: config/study.toml)")
    go.add_argument("--planner", action="store_true",
                    help="add a model-guided planning branch wherever a validated procedure exists (on in config/study.toml)")
    go.add_argument("--ceiling", type=float, default=None,
                    help="USD spent under --runs after which no recursion starts another step; evaluations still run")
    flight = commands.add_parser("preflight", help="check every campaign and executor before a study")
    flight.add_argument("--campaign", action="append", default=[], help="campaign name (repeatable; default: all)")
    flight.add_argument("--executor", action="append", default=[], help="executor name (repeatable; default: all)")
    flight.add_argument("--campaigns-only", action="store_true", help="skip the improver sessions and smoke trials")
    flight.add_argument("--env", default=None, help="Harbor environment (default: config/study.toml)")
    flight.add_argument("--runs", type=Path, default=RUNS / "preflight")
    check_mount = commands.add_parser("smoke", help="one real trial with a marker harness, to check the mount")
    check_mount.add_argument("--executor", required=True)
    check_mount.add_argument("--campaign", default="biosensor-active-learning")
    check_mount.add_argument("--env", default=None, help="Harbor environment (default: config/study.toml)")
    for sub in (go, check_mount, commands.add_parser("status", help="progress of every recursion")):
        sub.add_argument("--runs", type=Path, default=RUNS)
    args = parser.parse_args(argv)
    study = config.load()

    if args.command == "check":
        problems = check(study)
        print("\n".join(problems) if problems else f"frozen at {study.commit[:12]}: ok")
        return 1 if problems else 0
    if args.command == "h0":
        print(harness.write_stock(args.path))
        return 0
    if args.command == "guard":
        if args.campaign not in study.campaigns:
            raise SystemExit(f"unknown campaign {args.campaign}; known: {sorted(study.campaigns)}")
        problems = guard.check(args.tree, study.campaigns[args.campaign])
        print("\n".join(problems) if problems else "ok")
        return 1 if problems else 0
    if args.command == "smoke":
        return run_smoke(args, study)
    if args.command == "preflight":
        return run_preflight(args, study)
    if args.command == "status":
        rows = status(args.runs)
        columns = ["executor", "campaign", "recursion", "base", "step", "rollbacks", "selected", "eval", "eval_base", "fallbacks"]
        print("\t".join(columns))
        for row in rows:
            print("\t".join(str(row[column]) for column in columns))
        return 0
    return run(args, study)


if __name__ == "__main__":
    raise SystemExit(main())
