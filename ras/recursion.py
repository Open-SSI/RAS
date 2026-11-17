"""One recursion: base runs, K improvement steps with rollback, then the evaluation:
fresh runs of the selected harness beside as many fresh runs of the stock harness.

Layout under ``runs/recursions/<executor>/<campaign>/r<n>/``:

    harness/<name>/           h0, start, k1-planning, k1-modeling, k1, ...
    harness/<name>.notes.md   the session's RATIONALE.md or DECISIONS.md
    records/<run>/            run records with their feedback: k1-planning-1, k1-1, k1-3 (a repeat), eval-k1-1, ...
    sessions/<name>/          improver working directories and their logs
    selected.txt              the harness the evaluation runs used
    events.jsonl              phases, selections, repeats, rollbacks, guard problems and fallbacks, in order

Harnesses are optimized against the benchmark's own verifier, on its one fixed evaluation set, with
its official discovery and learning metrics and thresholds. A harness's D and L are its runs' mean
shares of their thresholds, and S = min(D, L). Each step runs the editors the bottleneck calls for
(``_editors``): modeling alone while learning fails and discovery passes in every run (it may then
change what is measured where learning needs it), planning alone in the reverse case, both otherwise;
it composes only when both ran. Selection (``_select``) then picks among the step's branches, its
probes and the composed harness, the one that beats the best
harness so far by the most on S under ``feedback.choose``'s safeguards, and keeps it only if its gain
clears one standard error of the run-to-run spread, after repeating it once if it does not yet;
otherwise the best verified harness stays. Identical runs are one observation, and a harness whose
runs all agree is not rerun to estimate a spread it does not have. The improvers see every run's
verifier aggregates and failure diagnostics (``verifier.json``), never the hidden labels, and a
guard keeps measured or held-out answers out of every harness.

Each step starts from the best harness so far, so an edit that made things worse is rolled back,
and every rejected harness is kept for analysis. A composed harness identical to the best so far
or to an earlier composed harness is not run again. The evaluation runs are fresh, never shown to
an improver, and never helped choose the harness they evaluate; when the stock harness is selected,
its evaluation is the Base evaluation itself.

A recursion may start from an existing harness instead of H0 (``start``): the stock
harness still runs as Base, and the starting harness is the best so far until a step
beats it. Every phase is skipped once its output exists, so running a recursion again
resumes it, and with more steps it continues where it stopped.

The non-recursive control (``mode="flat"``, under ``flat<n>/``) spends the same budget
without recursion: every step edits the starting harness, seeing only its runs and the
step's own probes, and the best of the K composed harnesses is selected and evaluated.

``protocol.json`` records a digest of the code, prompts and configuration a recursion
started with; resuming it under a different protocol is refused, so a frozen study
cannot drift.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import math
import re
import shutil
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from . import contract, evidence, feedback, guard, harness, procedure
from .replay import Replayer
from .simulate import Simulator
from .config import ROOT, Campaign, Executor, Study
from .improver import Improver, notes_path, prepare
from .runner import HarborRunner, Job

RUNS = ROOT / "runs"
SCOPES = ("planning", "modeling")  # the editors of the two measured stages: experiments (discovery) and model (learning)
TARGETS = {"planning": "discovery", "modeling": "learning"}
MODES = ("recursive", "flat")
OWNS = {"modeling": harness.is_modeling, "planning": lambda name: not harness.is_modeling(name)}  # each editor's files
CHANGE = "CHANGE.json"  # an editor's statement: the failure mechanism, the change, its expected effect on each gate
DIRECTIONS = ("raise", "keep", "lower")
SIGMA = 0.15  # the run-to-run spread of a run's utility assumed before any harness has two distinct runs
SIM_SOURCES = 5  # real runs a planning branch's simulations branch from, at most
CONFIRM_BRANCHES = 2  # promising branches a step may confirm, the most promising first
FOLLOWED = 0.9  # share of a proposed round the agent must have ordered for a planner run to count as its plan
LINKAGE_SLACK = 0.05  # how far a plan-keeping branch's share of hits lost to linked families may exceed the incumbent's worst
SESSION_USD = 0.3  # an improver session's cost assumed before any has run
EVAL_BASE = "eval-base"  # the stock harness's evaluation runs, beside the selected harness's eval-<name> runs
STOCK = "stock"  # an index's stock-harness runs, opening and evaluation, shared by its RAS recursion and its control
PRIOR_LIMIT = 0.2  # Spearman a procedure may reach on the evidence after fitting shuffled targets
SURE = 1.0  # standard errors a gain must exceed to be kept without repeating the candidate
TIMED = 100  # catalog designs the audit times a procedure's model on, against an empty model
EMPTY = "function predict(design) { return 0; }\n"
PROTOCOL = ("ras/**/*.py", "ras/prompts/*.md", "ras/toolkit/*", "config/*.toml")
RULE = ("Harnesses are compared on the benchmark's own verifier, on its fixed evaluation set, with its official "
        "discovery and learning metrics and thresholds. Each harness's discovery D and learning L are its runs' mean "
        "shares of their thresholds, and S = min(D, L); a run passes when both shares are at least 1. A step's "
        "branches are its probes and, where both editors ran, the composed harness; the best branch replaces the "
        "best harness so far only if it raises S (once both harnesses have S >= 1: the share of passing runs, or that "
        "share held and S), loses no gate the incumbent passes, and, where the incumbent fails both gates, does not "
        "lower its other gate beyond the runs' spread. The gain must exceed one standard error of the run-to-run "
        "spread; a smaller one is repeated with fresh runs, unless the branch's runs were identical, and decided again "
        "with every run counted. Otherwise the best harness so far stays. Identical runs count once.")
RULE_SEQUENTIAL = (
    "Harnesses are compared on the benchmark's own verifier, on its fixed evaluation set, with its official discovery "
    "and learning metrics and thresholds. A run's utility is its own bottleneck, min(D, L) as shares of the thresholds, "
    f"capped at {feedback.CAP}; it passes when both shares are at least 1. A branch that changes only the modeling files "
    "and owns scripts/model/fit.py is first replayed: refitted on the train.csv of every run the current harness's plan "
    "produced and scored by the verifier on that run's own ledger, paired with the model the run delivered. Any other "
    "branch runs twice. A branch whose mean utility is clearly lower (by more than one standard error, or at all on "
    "paired trajectories), or that fails a gate the current harness passes in every run, is rejected. The most "
    "promising branch then gets two fresh runs, and replaces the current harness only if its gain exceeds one standard "
    "error (paired where the two share trajectories, otherwise with the run-to-run spread pooled over the campaign's "
    "harnesses), it passes no smaller share of runs, its fresh runs on their own reach the current harness's mean "
    "utility, it fails no gate the current harness passes in every run, and, if it owns a procedure, every fresh run "
    "delivered that procedure's output. A branch that keeps the current plan must also reproduce its discovery in its "
    "own runs, at screening and after confirmation: a mean discovery share no lower by more than one standard error, "
    "and, where families count, no run losing a larger share of its hits to linked families than the current "
    "harness's worst by more than 5 points. Otherwise the current harness stays. Identical runs count once.")


class Recursion:
    def __init__(self, study: Study, campaign: Campaign, executor: Executor, index: int,
                 runner: HarborRunner, improver: Improver, root: Path = RUNS,
                 validator: feedback.Validator = procedure.validate, start: Path | None = None,
                 steps: int | None = None, mode: str = "recursive", allow_protocol_change: bool = False,
                 auditor=procedure.audit, scorer=None, checker: feedback.Checker | None = None,
                 ceiling: float | None = None, replayer: Replayer | None = None, simulator: Simulator | None = None):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}; known: {MODES}")
        self.study, self.campaign, self.executor, self.index = study, campaign, executor, index
        self.runner, self.improver, self.start = runner, improver, start
        # The procedure's own cross-validation (feedback.json, for editors only) uses the campaign's ranking metric.
        self.validator = functools.partial(validator, metric=campaign.ranking) if validator is procedure.validate else validator
        self.steps, self.mode, self.auditor = steps or study.protocol.steps, mode, auditor
        self.allow_protocol_change, self.ceiling, self.root = allow_protocol_change, ceiling, root
        run_model = contract.benchmark_runner(study.runtime)
        self.scorer = scorer or (lambda source, inputs: run_model(source, inputs, 300))  # the audit's timing of a procedure
        self.checker = checker or (lambda model: contract.check(model, campaign, run_model))
        name = {"recursive": "r", "flat": "flat"}[mode] + str(index)
        self.dir = root / "recursions" / executor.name / campaign.name / name
        self.stock_runs = StockRuns(campaign, executor, runner, root, index)
        self.replayer = replayer or Replayer(campaign)
        self.simulator = simulator or Simulator(campaign)
        # Sequential selection is RAS's; the flat control keeps the rule it was defined with.
        self.sequential = study.protocol.selection == "sequential" and mode != "flat"

    def run(self) -> Path:
        """Run or resume the recursion; return the selected harness."""
        self._freeze()
        protocol = self.study.protocol
        best = stock = self._stock()
        self.stock_runs.ensure("base", stock, protocol.base_runs, self.validator, self.checker)
        best_label = "base"
        if self.start:
            best, best_label = self._adopt(self.start), "start"
            self._run({"start": best}, protocol.composed_runs)
        origin, origin_label = best, best_label
        for k in range(1, self.steps + 1):
            if self.sequential:
                if not self._affordable(k):
                    break
                chosen = self._step(k, best, best_label)
                if chosen:
                    best, best_label = self._tree(chosen), chosen
                continue
            if self.ceiling is not None and not (self.dir / "harness" / f"k{k}").exists() and spent(self.root) >= self.ceiling:
                self._log("ceiling", step=k, spent=round(spent(self.root), 2), ceiling=self.ceiling)
                break
            # Recursion edits the best harness so far with all evidence; the control always edits
            # the starting harness and sees only its runs.
            current, seen = (best, None) if self.mode != "flat" else (origin, {"base", origin_label})
            current_label = best_label if self.mode != "flat" else origin_label
            scopes = self._editors(k, current_label, seen)
            candidates = {scope: self._edit(scope, f"k{k}-{scope}", current, seen=seen,
                                            diagnosis=self._diagnosis(scope, current, current_label, seen))
                          for scope in scopes}
            # An editor whose harness came back unchanged (it changed nothing, or fell back) adds no branch.
            branches = {f"k{k}-{scope}": tree for scope, tree in candidates.items() if _digest(tree) != _digest(current)}
            self._run(branches, protocol.probe_runs)
            for scope, tree in candidates.items():
                if f"k{k}-{scope}" in branches:
                    self._credit(f"k{k}-{scope}", tree, current, current_label, k, editor=scope)
            if len(branches) == len(SCOPES):
                # Compose only when both editors produced a branch.
                probes = None if seen is None else seen | set(branches)
                composed = self._edit("composer", f"k{k}", current, candidates, seen=probes,
                                      diagnosis=self._diagnosis("composer", current, current_label, probes, candidates))
                duplicate = self._duplicate(composed, best_label, k)
                if duplicate:
                    self._log("duplicate", step=k, of=duplicate)
                elif _digest(composed) not in {_digest(tree) for tree in branches.values()}:
                    self._run({f"k{k}": composed}, protocol.composed_runs)
                    self._credit(f"k{k}", composed, current, current_label, k, editor="composer")
                    branches[f"k{k}"] = composed
            chosen = self._select(k, branches, best_label)
            if chosen:
                best, best_label = branches[chosen], chosen
        (self.dir / "selected.txt").write_text(f"{best.name}\n")
        self._evaluate(best, best_label, stock)
        return best

    def _evaluate(self, best: Path, label: str, stock: Path) -> None:
        """Fresh runs of the selected harness, campaign notes and all, with the same executor and budget; no
        improver ever sees them. Base is the stock harness's fresh runs beside them (``base_eval_runs``), or,
        with none, its opening runs, which no selection touched. When the stock harness is selected, Base is
        its evaluation. A deterministic selected harness runs twice first, and its remaining evaluation runs
        only if those two differ from what it did before."""
        protocol = self.study.protocol
        runs = protocol.eval_runs
        fresh_base = runs if protocol.base_eval_runs is None else protocol.base_eval_runs
        with ThreadPoolExecutor(2) as pool:
            base = pool.submit(self.stock_runs.ensure, EVAL_BASE, stock, fresh_base, self.validator, self.checker)
            if best != stock:
                first = min(runs, 2) if self.deterministic(label) else runs
                self._run({f"eval-{best.name}": best}, first)
                if first < runs and not self._repeats(label, f"eval-{best.name}"):
                    self._more(f"eval-{best.name}", best, runs - first, after=first)
            base.result()

    def records(self, seen: set[str] | None = None, evaluation: bool = False) -> dict[str, Path]:
        """Every complete run record so far, the index's shared stock runs included, by name; with
        ``seen``, only those harness labels. Evaluation runs are left out unless asked for: no improver
        and no selection ever sees one, whichever recursion of the index made it."""
        out = _records(self.stock_runs.dir / "records", seen) | _records(self.dir / "records", seen)
        return out if evaluation else {n: p for n, p in out.items() if not _label(n).startswith("eval-")}

    def estimates(self, seen: set[str] | None = None) -> dict[str, dict]:
        """The official objective per harness label over its distinct runs (feedback.objective), with how many
        runs it has, how many differ, and how many delivered exactly their harness's procedure's output."""
        groups: dict[str, list[Path]] = defaultdict(list)
        for name, path in self.records(seen).items():
            groups[_label(name)].append(path)
        out = {}
        for label, paths in groups.items():
            distinct = self._distinct(label, paths)
            cvs = [feedback.estimate(path)[1] for path in paths]
            summary = feedback.objective([feedback.official(path) for path in distinct])
            out[label] = {key: (round(value, 4) if isinstance(value, float) else value) for key, value in summary.items()
                          if key not in ("capped", "raw")} | {
                "runs": len(paths), "distinct": len(distinct), "verified": sum(cv is not None for cv in cvs),
                "per_run": {path.name: {k: feedback.official(path).get(k) for k in ("discovery", "learning", "bottleneck", "passed")}
                            for path in paths}}
        return out

    def _select(self, step: int, branches: Mapping[str, Path], incumbent: str) -> str | None:
        """Among the step's branches (its probes and, where it composed, the composed harness), the one that beats the
        incumbent by the most on S, if any (``feedback.choose``). A gain within one standard error of the run-to-run
        spread is not yet meaningful: that branch is repeated, unless its runs were identical, and kept only if its
        gain then clears the spread with every run counted. Logged once."""
        logged = next((e for e in _events(self.dir / "events.jsonl", "accept", "rollback") if e.get("step") == step), None)
        if logged:
            return logged["best"] if logged["event"] == "accept" else None
        tests = {label: self._choose(label, incumbent) for label in branches}
        ranked = sorted((label for label, t in tests.items() if t["accept"]),
                        key=lambda label: (tests[label]["candidate"]["S"], tests[label]["candidate"]["passed"]), reverse=True)
        chosen = None
        if ranked:
            label = ranked[0]
            if not tests[label]["sure"] and self.study.protocol.confirm_runs and not self.deterministic(label):
                runs = len([n for n in self.records() if _label(n) == label])
                self._more(label, branches[label], self.study.protocol.confirm_runs, after=runs)
                tests[label] = self._choose(label, incumbent) | {"repeated": self.study.protocol.confirm_runs,
                                                                   "before_repeat": tests[label]}
            if tests[label]["accept"] and tests[label]["sure"]:
                chosen = label
        shown = chosen or (ranked[0] if ranked else max(tests, key=lambda label: tests[label]["candidate"]["S"], default=None))
        self._log("accept" if chosen else "rollback", step=step, best=chosen or incumbent,
                  test=tests.get(shown), branch=shown, options=tests)
        return chosen

    def _editors(self, step: int, label: str, seen: set[str] | None) -> tuple[str, ...]:
        """The editors a step runs, from the current harness's runs: modeling alone while learning fails and discovery
        passes in every run, planning alone in the reverse case, both while both fail or a gate passes only in some
        runs; once both pass in every run, the editor of the gate with the smaller margin."""
        runs = (list(self._results(label).values()) if self.sequential else
                [feedback.official(p) for n, p in self.records(seen).items() if _label(n) == label])
        steady = {gate: bool(runs) and all(r[gate] >= 1 for r in runs) for gate in feedback.GATES}
        means = {gate: statistics.fmean(r[gate] for r in runs) if runs else 0.0 for gate in feedback.GATES}
        if steady["discovery"] and steady["learning"] and self.sequential:
            scopes, reason = ("modeling",), "both gates pass in every run: the plan stays, the model's margin is raised"
        elif steady["discovery"] and steady["learning"]:
            scopes = ("planning",) if means["discovery"] <= means["learning"] else ("modeling",)
            reason = "both gates pass in every run: the gate with the smaller margin"
        elif steady["discovery"]:
            scopes, reason = ("modeling",), "discovery passes in every run and learning does not"
        elif steady["learning"]:
            scopes, reason = ("planning",), "learning passes in every run and discovery does not"
        else:
            scopes, reason = SCOPES, "neither gate passes in every run"
        if not any(e.get("step") == step for e in _events(self.dir / "events.jsonl", "editors")):
            self._log("editors", step=step, editors=list(scopes), reason=reason,
                      shares={gate: round(means[gate], 4) for gate in feedback.GATES})
        return scopes

    def _choose(self, candidate: str, incumbent: str) -> dict:
        return feedback.choose([feedback.official(r) for r in self._distinct(candidate)],
                               [feedback.official(r) for r in self._distinct(incumbent)], SURE)

    def deterministic(self, label: str) -> bool:
        """Every run of the harness so far was identical: the same designs measured and the same model delivered."""
        runs = [p for n, p in self.records().items() if _label(n) == label]
        return len(runs) >= 2 and len(self._distinct(label)) == 1

    def _repeats(self, label: str, fresh: str) -> bool:
        """The ``fresh`` runs are all identical to the runs ``label`` already had."""
        old = {_identity(r) for n, r in self.records().items() if _label(n) == label}
        new = {_identity(r) for n, r in self.records(evaluation=True).items() if _label(n) == fresh}
        return len(old) == 1 and new == old

    def _distinct(self, label: str, paths: list[Path] | None = None) -> list[Path]:
        """A label's runs, each set of identical runs (same designs measured, same model delivered) once."""
        out = {}
        for record in paths if paths is not None else [p for n, p in self.records().items() if _label(n) == label]:
            out.setdefault(_identity(record), record)
        return list(out.values())

    def _diagnosis(self, stage: str, current: Path, label: str, seen: set[str] | None,
                   candidates: Mapping[str, Path] | None = None) -> str:
        """The recursion's diagnosis of the current harness for one session: each gate's share of its threshold and
        its deficit, the bottleneck, failures, and the session's target; for the composer, each candidate's effect on
        both gates."""
        estimates = self.estimates(seen)
        mine = estimates.get(label)
        if not mine:
            return ""
        runs = [p for n, p in self.records(seen).items() if _label(n) == label]
        results = [feedback.official(p) for p in runs]
        gate = lambda share: f"{share:.3f}" + (" (passes)" if share >= 1 else f" ({1 - share:.3f} below its threshold)")
        lines = ["## Diagnosis", "",
                 f"The current harness, `{current.name}`, over its {mine['runs']} runs ({mine['distinct']} distinct), by the official verifier:",
                 "", "| gate | mean share of threshold | per run |", "|---|---|---|"]
        for name in feedback.GATES:
            lines.append(f"| {name} | {gate(mine[name])} | {', '.join(f'{r[name]:.3f}' for r in results)} |")
        values = defaultdict(list)
        for r in results:
            for key, g in (r.get("gates") or {}).items():
                values[key].append((g.get("value") or 0.0, g["required"]))
        detail = "; ".join(f"`{key}` {statistics.fmean(v for v, _ in pairs):.4g} on average, threshold {pairs[0][1]:g}"
                           for key, pairs in values.items())
        bottleneck = "discovery" if mine["discovery"] <= mine["learning"] else "learning"
        passed = sum(bool(r.get("passed")) for r in results)
        lines += ["", f"S = min(D, L) = {mine['S']:.3f}. The bottleneck is **{bottleneck}**. {passed} of {len(results)} runs "
                  f"passed both gates. Requirement values: {detail}."]
        errors = sorted({r["error"] for r in results if r.get("error")})
        if errors:
            lines.append(f"The verifier reported errors on some runs: {'; '.join(e[:160] for e in errors)}.")
        steady = {gate: all(r[gate] >= 1 for r in results) for gate in feedback.GATES}
        if stage in TARGETS:
            own = TARGETS[stage]
            other = "learning" if own == "discovery" else "discovery"
            if mine[own] < 1 and own == bottleneck:
                task = f"Your gate, {own}, is the bottleneck: raise it, without letting {other} fall."
            elif mine[own] < 1:
                task = (f"Your gate, {own}, fails too, though {bottleneck} is lower: raise {own}, and keep in mind that S "
                        f"rises only when the lower gate does.")
            else:
                task = (f"Your gate, {own}, passes: keep it passing. The failing gate is {bottleneck}; use your levers "
                        f"to help it where they can." if mine["S"] < 1 else
                        f"Both gates pass on average: keep {own} passing in every run and raise its margin.")
            other = "learning" if own == "discovery" else "discovery"
            if steady[other] and other == "discovery":
                task += (" Discovery passes in every run, so this step runs no planning editor. Keep what discovery relies on "
                         "and add no rigid quotas or schedules. The model learns only from what the run measures: where learning "
                         "needs data the plan does not collect, change what is measured too, keeping discovery above its threshold.")
            elif steady[other]:
                task += (" Learning passes in every run, so this step runs no modeling editor: leave how the model is built as "
                         "it is, and change it only where discovery needs it and learning stays above its threshold.")
            lines += ["", task, "", "Make only the changes this diagnosis supports. Leave every other part of the harness as it is: "
                      "an edit that does not serve the bottleneck costs runs and adds noise."]
        if candidates:
            lines += ["", "The planning editor changed the experiments (what discovery measures), the modeling editor the "
                      "predictive model (what learning measures). What each candidate did against the current harness "
                      "(official verifier, mean over its runs):", "",
                      "| candidate | editor's gate | discovery | learning | S | runs passed | components changed |", "|---|---|---|---|---|---|---|"]
            for scope, tree in candidates.items():
                theirs = estimates.get(_label(f"{tree.name}-1")) or estimates.get(tree.name)
                if not theirs:
                    continue
                changed = ", ".join(_changed(current, tree)) or "nothing"
                lines.append(f"| `{tree.name}` | {TARGETS.get(scope, '-')} | {theirs['discovery']:.3f} ({theirs['discovery'] - mine['discovery']:+.3f}) | "
                             f"{theirs['learning']:.3f} ({theirs['learning'] - mine['learning']:+.3f}) | {theirs['S']:.3f} ({theirs['S'] - mine['S']:+.3f}) | "
                             f"{theirs['passed']:.2f} | {changed} |")
        return "\n".join(lines) + "\n"

    # Sequential selection on matched trajectories ------------------------------------------------------------

    def _step(self, k: int, best: Path, best_label: str) -> str | None:
        """One step. The editors the bottleneck calls for (``_editors``) each propose one branch, within their own
        files. A branch that changes only modeling files and owns a procedure is replayed on every trajectory of
        its plan, the incumbent's own runs among them, at no run cost (replay.py); any other branch runs
        ``probe_runs`` times. Where both editors produced a branch and the modeling one owns a procedure, the
        planning branch's plan with the modeling branch's model (a graft) is replayed on the planning branch's runs:
        composition, mechanical and optional. With ``protocol.planner``, a model-guided planning branch (``_planner``)
        is simulated from the incumbent's own runs. Every branch is screened against the incumbent (``feedback.screen``);
        a clear regression stops there. The most promising branch gets ``confirm_runs`` fresh runs (``_decide``), and,
        if it fails, the next one; a branch replaces the incumbent only if ``feedback.accept`` and its realization
        checks hold. Logged once; a resumed step returns its decision."""
        logged = next((e for e in _events(self.dir / "events.jsonl", "accept", "rollback") if e.get("step") == k), None)
        if logged:
            return logged["best"] if logged["event"] == "accept" else None
        protocol = self.study.protocol
        trees = {}
        editors = {"none": (), "modeling": ("modeling",)}.get(protocol.editors)
        for scope in self._editors(k, best_label, None) if editors is None else editors:
            tree = self._edit(scope, f"k{k}-{scope}", best, diagnosis=self._science(scope, best, best_label))
            if harness.digest(tree) != harness.digest(best):  # an editor that changed nothing adds no branch
                trees[f"k{k}-{scope}"] = tree
        replayed = {label for label, tree in trees.items() if _owns(tree) and self._modeling_only(tree, best)}
        with ThreadPoolExecutor(2) as pool:
            probes = pool.submit(self._run, {l: t for l, t in trees.items() if l not in replayed}, protocol.probe_runs)
            for label in sorted(replayed):
                self._replay_family(label)
            probes.result()
        plan, model = f"k{k}-planning", f"k{k}-modeling"
        if plan in trees and model in replayed:
            graft = self._graft(f"k{k}-graft", trees[plan], trees[model])
            if harness.digest(graft) not in {harness.digest(t) for t in (best, *trees.values())}:
                trees[f"k{k}-graft"] = graft
                self._replay_family(f"k{k}-graft")
        planner = self._planner(k, best, best_label, trees, replayed)
        if planner is not None:
            trees[f"k{k}-planner"] = planner
        tests = {label: self._test(label, best_label) for label in trees}
        screened = next((e for e in _events(self.dir / "events.jsonl", "screen") if e.get("step") == k), None)
        if screened is None:
            self._log("screen", step=k, tests=tests)
        counts = {label: t.get("records", t["real_runs"]) for label, t in (screened or {"tests": tests})["tests"].items()}
        promising = sorted((l for l, t in tests.items() if t["verdict"] == "promising"),
                           key=lambda l: tests[l]["delta"] - tests[l]["se"], reverse=True)
        decisions = {}
        for branch in promising[:CONFIRM_BRANCHES]:  # the next promising branch only if the one before it failed
            decisions[branch] = self._decide(k, branch, trees, best, best_label, tests, counts)
            if not decisions[branch]:
                break
        accepted = [b for b in decisions if not decisions[b]]
        chosen = max(accepted, key=lambda b: tests[b]["delta"], default=None)
        branch = chosen or (promising[0] if promising else None)
        reasons = decisions.get(branch, []) if branch else []
        shown = branch or max(tests, key=lambda l: tests[l]["delta"], default=None)
        self._log("accept" if chosen else "rollback", step=k, best=chosen or best_label, branch=shown,
                  test=tests.get(shown), reasons=reasons, options=tests,
                  confirmed={b: r for b, r in decisions.items()})
        for label, tree in trees.items():
            self._credit_branch(k, label, tree, best, tests[label], decision="accepted" if label == chosen else
                                ("rejected after confirmation" if label in decisions else tests[label]["verdict"]))
        self._archive(k, chosen or best_label)
        return chosen

    def _decide(self, k: int, branch: str, trees: Mapping[str, Path], best: Path, best_label: str, tests: dict,
                counts: Mapping[str, int]) -> list[str]:
        """Confirm one promising branch on ``confirm_runs`` fresh runs (``feedback.accept``); return why it may not
        replace the incumbent, empty when it may. ``tests[branch]`` is updated with every run counted."""
        fresh = self._confirm(branch, after=counts.get(branch, 0))
        tests[branch] = self._test(branch, best_label) | {"screened": tests[branch]}
        theirs = self._results(best_label)
        pairs = [(feedback.official(p), theirs[p.name]) for p in fresh if p.name in theirs]
        # A plan-keeping procedure's fresh runs are the incumbent's own agent: their discovery tells nothing of it.
        keeps = self._plan_of(trees[branch]) == self._plan_of(best) and _owns(trees[branch])
        reasons = feedback.accept(tests[branch], tests[branch]["lost"], [feedback.official(p) for p in fresh],
                                  tests[branch]["incumbent"]["utility"], SURE, fresh_pairs=pairs,
                                  gate="learning" if keeps else None,
                                  incumbent_gate=statistics.fmean(r["learning"] for r in theirs.values()))
        reasons += [f"{p.name} did not deliver its procedure's model" for p in fresh
                    if _owns(trees[branch]) and not feedback.verified(p)]
        reasons += tests[branch]["unrealized"]
        if branch == f"k{k}-graft":  # a graft keeps the planning branch's plan, whose runs its replays used
            reasons += self._unrealized(branch, f"k{k}-planning")
        if harness.planned(trees[branch]):  # its simulations stand for the agent only if the agent ordered the proposals
            reasons += [f"{p.name}: {problem}" for p in fresh for problem in [_unfollowed(p)] if problem]
        return reasons

    def _planner(self, k: int, best: Path, best_label: str, trees: Mapping[str, Path], replayed: set[str]) -> Path | None:
        """The model-guided planning branch (protocol.planner): the best harness's plan with a validated procedure and
        the planner (ras/planner/acquire.py), configured by the campaign's restated counting rules. The procedure is the
        best harness's own, else this step's modeling branch if its replays beat the incumbent's models in learning on
        the same trajectories (``_learns_better``). Its simulated runs branch from the incumbent's plan's real runs after their first round
        (simulate.py), so each pairs with one of the incumbent's runs. None without a validated procedure."""
        if not (self.study.protocol.planner and self.campaign.plan):
            return None
        source = best if _owns(best) else None
        for label in sorted(replayed):
            if source is None and self._learns_better(label, best_label):
                source = trees[label]
        if source is None:
            return None
        out = self.dir / "harness" / f"k{k}-planner"
        if not out.exists():
            staging = self._staging(out)
            harness.install_planner(harness.graft(best, source, staging), dict(self.campaign.plan))
            procedure.complete(staging / procedure.MODEL_DIR, self.campaign)
            problems = guard.check(staging, self.campaign)
            if problems:
                raise RuntimeError(f"the planner branch fails the guard: {problems}")
            change_path(out).write_text(json.dumps({"planner": {"plan": best.name, "model": source.name,
                                                                "config": self.campaign.plan}}, indent=2) + "\n")
            staging.rename(out)
        earlier = {harness.digest(t) for t in (self.dir / "harness").glob("k*-planner")
                   if t.is_dir() and t != out and not t.name.endswith(".tmp")}
        if harness.digest(out) in earlier | {harness.digest(best)}:
            return None
        sources, starts = [], set()
        for record in self._family(best_label):  # runs that ordered the same first round give the same simulation
            start = _measured_rounds(record, 1)
            if start not in starts and len(sources) < SIM_SOURCES:
                sources.append(record)
                starts.add(start)
        for record in sources:
            self.simulator.run(out, record, self.dir / "simulations" / f"{out.name}@{record.name}")
        return out

    def _export_branches(self, target: Path, before: int) -> None:
        """Every branch of steps before ``before``, for an editor: its diff against the stock harness, its editor's
        notes and statement, and how selection judged it, with its result on each trajectory it was replayed on. A
        branch judged only by replay has no run in the evidence, so this is the only place its code reaches an editor."""
        archive = evidence.read_json(self.dir / "archive.json").get("branches", {})
        replays = self._replay_table()
        stock = self.dir / "harness" / "h0"
        for tree in sorted((self.dir / "harness").glob("k*")):
            step = tree.name[1:].split("-")[0]
            if not tree.is_dir() or tree.name.endswith(".tmp") or not step.isdigit() or int(step) >= before:
                continue
            out = target / tree.name
            out.mkdir(parents=True)
            (out / "harness.diff").write_text(harness.diff(stock, tree))
            for source, name in ((notes_path(tree), "RATIONALE.md"), (change_path(tree), CHANGE)):
                if source.is_file():
                    shutil.copyfile(source, out / name)
            decision = archive.get(tree.name, {}) | {"replays": replays.get(tree.name, {})}
            (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    def _replay_table(self) -> dict[str, dict[str, dict]]:
        """Every replay so far: by harness, its official result on each trajectory it was refitted on."""
        out: dict[str, dict[str, dict]] = defaultdict(dict)
        root = self.dir / "replays"
        for path in sorted(root.iterdir()) if root.is_dir() else []:
            if path.is_dir() and "@" in path.name and not path.name.endswith(".tmp"):
                label, trajectory = path.name.split("@", 1)
                result = evidence.read_json(path / "verifier.json")
                out[label][trajectory] = {k: result.get(k) for k in ("discovery", "learning", "bottleneck", "passed", "error")}
        return dict(out)

    def _tree(self, label: str) -> Path:
        return self.dir / "harness" / ("h0" if label == "base" else label)

    def _modeling_only(self, tree: Path, current: Path) -> bool:
        return all(harness.is_modeling(name) for name in _changed(current, tree))

    def _plan_of(self, tree: Path) -> str:
        """A harness's planning digest (harness.digest), once per tree: harness trees never change once written."""
        cache = self.__dict__.setdefault("_plans", {})
        if tree not in cache:
            cache[tree] = harness.digest(tree, "planning")
        return cache[tree]

    def _family(self, label: str) -> list[Path]:
        """The distinct real runs of every harness with ``label``'s plan."""
        mine = self._plan_of(self._tree(label))
        groups = defaultdict(list)
        for name, path in self.records().items():
            groups[_label(name)].append(path)
        out = []
        for other, paths in sorted(groups.items()):
            tree = self._tree((paths[0] / "harness.txt").read_text().strip())
            if tree.is_dir() and self._plan_of(tree) == mine:
                out += self._distinct(other, paths)
        return out

    def _replay_family(self, label: str) -> None:
        """Replay ``label``'s procedure on every run of its plan that is not its own. A trajectory two runs share (the
        same designs measured) is refitted once; the other run gets a copy, since the refit depends only on it."""
        tree = self._tree(label)
        own = {p.name for p in self._distinct(label)}
        done: dict[tuple, Path] = {}
        for record in self._family(label):
            if record.name in own:
                continue
            target, trajectory = self.dir / "replays" / f"{label}@{record.name}", _measured(record)
            if trajectory in done and not target.exists():
                shutil.copytree(done[trajectory], target)
                (target / "replayed.txt").write_text(f"{record.name}\n")
            else:
                self.replayer.replay(tree, record, target)
            done.setdefault(trajectory, target)

    def _results(self, label: str) -> dict[str, dict]:
        """``label``'s official result on each trajectory it has, by the name of the run that made the trajectory:
        its own distinct runs, then its replays, shadows and simulations."""
        out = {p.name: feedback.official(p) for p in self._distinct(label)}
        for root in (self.dir / "replays", self.dir / "simulations"):
            for path in sorted(root.glob(f"{label}@*")) if root.is_dir() else []:
                result = evidence.read_json(path / "verifier.json")
                if path.is_dir() and not path.name.endswith(".tmp") and not result.get("inconclusive"):
                    out.setdefault(path.name.split("@", 1)[1], result)
        return out

    def _sigma(self, value=feedback.run_utility) -> float:
        """The run-to-run spread of a run's utility (or of ``value``), pooled over every harness's distinct real runs."""
        groups = defaultdict(list)
        for name, path in self.records().items():
            groups[_label(name)].append(path)
        spreads = [[value(feedback.official(p)) for p in self._distinct(label, paths)] for label, paths in groups.items()]
        return feedback.pooled_sd(spreads) or SIGMA

    def _unrealized(self, label: str, incumbent: str) -> list[str]:
        """For a branch that keeps the incumbent's plan, how its own runs fail to reproduce the incumbent's discovery
        (``feedback.unrealized``): a replay assumes the plan is unchanged, and these runs are the check that it was."""
        if self._plan_of(self._tree(label)) != self._plan_of(self._tree(incumbent)):
            return []
        mine = {p.name: feedback.official(p) for p in self._distinct(label)}
        family = next((key for key in self.campaign.gates if "famil" in key), None)
        return feedback.unrealized(mine, list(self._results(incumbent).values()),
                                   self._sigma(lambda r: r["discovery"]), family if "discovery_hits" in self.campaign.gates else None,
                                   LINKAGE_SLACK)

    def _test(self, label: str, incumbent: str) -> dict:
        """``label`` against the incumbent (``feedback.compare``): paired on the trajectories they share where they
        have the same plan, with the incumbent's procedure replayed on the candidate's runs where it owns one."""
        same = self._plan_of(self._tree(label)) == self._plan_of(self._tree(incumbent))
        if same and _owns(self._tree(incumbent)):
            self._replay_family(incumbent)
        mine, theirs = self._results(label), self._results(incumbent)
        pairs = [(mine[t], theirs[t]) for t in sorted(set(mine) & set(theirs))]  # replays, shadows, simulations
        test = feedback.compare(list(mine.values()), list(theirs.values()), self._sigma(), pairs,
                                (self.deterministic(label), self.deterministic(incumbent)))
        kept = feedback.reliable(list(theirs.values()))
        lost = feedback.lost(mine, kept, theirs)  # on a shared trajectory, only where the incumbent passed the gate
        unrealized = self._unrealized(label, incumbent)
        own = len(self._distinct(label))
        return test | {"lost": lost, "reliable": kept, "unrealized": unrealized,
                       "verdict": "reject" if unrealized else feedback.screen(test, lost),
                       "real_runs": own, "replays": len(mine) - own,
                       "records": len([n for n in self.records() if _label(n) == label])}

    def _learns_better(self, label: str, incumbent: str) -> bool:
        """Whether ``label``'s model beats the incumbent's on the trajectories they share (its replays, paired with the
        models those runs delivered) by more than one paired standard error of learning: a planner needs the better
        model, whether or not discovery lets the better model show in a run's bottleneck."""
        mine, theirs = self._results(label), self._results(incumbent)
        gains = [mine[t]["learning"] - theirs[t]["learning"] for t in sorted(set(mine) & set(theirs))]
        if len(gains) < 2:
            return False
        se = max(statistics.stdev(gains) / math.sqrt(len(gains)), feedback.FLOOR)
        return statistics.fmean(gains) > se

    def _confirm(self, label: str, after: int) -> list[Path]:
        """``confirm_runs`` fresh runs of a promising branch, after the ``after`` runs it had when screened (so a
        resumed step finds them instead of adding more); returns them."""
        extra = self.study.protocol.confirm_runs
        if extra:
            self._more(label, self._tree(label), extra, after=after)
        return [self.dir / "records" / f"{label}-{i}" for i in range(after + 1, after + extra + 1)]

    def _graft(self, label: str, plan: Path, model: Path) -> Path:
        out = self.dir / "harness" / label
        if not out.exists():
            staging = self._staging(out)
            harness.graft(plan, model, staging)
            procedure.complete(staging / procedure.MODEL_DIR, self.campaign)
            problems = guard.check(staging, self.campaign)
            if problems:  # both parts passed the guard on their own
                raise RuntimeError(f"the graft {label} fails the guard: {problems}")
            statements = {part: evidence.read_json(change_path(tree)) for part, tree in (("planning", plan), ("modeling", model))}
            change_path(out).write_text(json.dumps({"graft": {"plan": plan.name, "model": model.name}, **statements},
                                                   indent=2) + "\n")
            staging.rename(out)
        return out

    def _credit_branch(self, step: int, label: str, tree: Path, current: Path, test: dict, decision: str) -> None:
        """What a branch changed, what its editor expected for each gate, and what the verifier saw."""
        if any(e.get("harness") == label for e in _events(self.dir / "events.jsonl", "credit")):
            return
        editor = label.split("-", 1)[1]
        observed = {gate: round(test["candidate"][gate] - test["incumbent"][gate], 4) for gate in feedback.GATES}
        self._log("credit", step=step, harness=label, editor=editor, edited_from=current.name,
                  targeted=TARGETS.get(editor, "both"),
                  components=_changed(current, tree), statement=evidence.read_json(change_path(tree)),
                  change=observed, gain=test["delta"], se=test["se"], paired=test["paired"], decision=decision,
                  candidate=test["candidate"], incumbent=test["incumbent"])

    def _archive(self, step: int, best: str) -> None:
        """archive.json: the best confirmed harness after each step, and every branch with its decision."""
        path = self.dir / "archive.json"
        archive = evidence.read_json(path) or {"best": [], "branches": {}}
        mine = list(self._results(best).values())
        archive["best"] = [b for b in archive["best"] if b["step"] != step] + [{
            "step": step, "harness": best, "trajectories": len(mine),
            "utility": round(statistics.fmean(feedback.run_utility(r) for r in mine), 4) if mine else None,
            "passed": round(statistics.fmean(float(r["passed"]) for r in mine), 4) if mine else None}]
        for e in _events(self.dir / "events.jsonl", "credit"):
            if e["step"] == step:
                archive["branches"][e["harness"]] = {k: e[k] for k in ("step", "editor", "decision", "gain", "se", "paired", "change")}
        path.write_text(json.dumps(archive, indent=2) + "\n")

    def _affordable(self, step: int) -> bool:
        """Whether the step's worst case and the evaluation after it fit under the ceiling: the sessions of the editors
        the protocol runs, their probed branches (a modeling branch is replayed, not probed; the planning editor's is
        probed, as may be a modeling branch without a procedure), confirmation of up to two branches, and the
        evaluation, priced at this campaign's mean trial so far. A step under way finishes."""
        if self.ceiling is None or any((self.dir / "harness").glob(f"k{step}-*")):
            return True
        protocol = self.study.protocol
        costs = [evidence.read_json(p / "stats.json").get("cost_usd") or 0.0 for p in self.records().values()]
        trial = statistics.fmean(costs) if costs else 0.0
        sessions, probed = {"none": (0, 0), "modeling": (1, 1)}.get(protocol.editors, (2, 2))
        need = sessions * SESSION_USD + (probed * protocol.probe_runs + CONFIRM_BRANCHES * protocol.confirm_runs
                                         + protocol.eval_runs) * trial
        used = spent(self.root)
        if used + need <= self.ceiling:
            return True
        self._log("ceiling", step=step, spent=round(used, 2), needed=round(need, 2), ceiling=self.ceiling)
        return False

    def _science(self, stage: str, current: Path, label: str) -> str:
        """The scientific diagnosis of the current harness for one editor: every trajectory's gates, requirement values
        and delivered model; which gates hold in every run; where the spread comes from; for several discovery
        requirements, which binds; the campaign's models by how they were built; the editor's task and scope; and how
        its branch will be judged."""
        results = self._results(label)
        if not results:
            return ""
        replays = [name for name in results if name not in {p.name for p in self._distinct(label)}]
        lines = ["## Diagnosis", "", f"The current harness, `{current.name}`, by the official verifier on {len(results)} "
                 f"trajectories: {len(results) - len(replays)} of its own runs"
                 + (f", {len(replays)} runs of harnesses with the same plan, with its procedure refitted on each"
                    if replays else "")
                 + ". Shares are of each threshold; S is the run's bottleneck.", "",
                 "| run | discovery | learning | S | passed | requirement values | delivered model |",
                 "|---|---|---|---|---|---|---|"]
        for name, r in results.items():
            values = ", ".join(f"`{key}` {g['value']:.4g}" for key, g in (r.get("gates") or {}).items()
                               if isinstance(g.get("value"), (int, float)))
            mark = " (replay)" if name in replays else ""
            lines.append(f"| {name}{mark} | {r['discovery']:.3f} | {r['learning']:.3f} | "
                         f"{min(r['discovery'], r['learning']):.3f} | {'yes' if r['passed'] else 'no'} | {values} | "
                         f"{self._provenance(label, name)} |")
        rs = list(results.values())
        held = feedback.reliable(rs)
        status = lambda gate: ("passes in every run" if gate in held else
                               f"passes in {sum(r[gate] >= 1 for r in rs)} of {len(rs)} runs")
        spread = lambda gate: f"{min(r[gate] for r in rs):.3f} to {max(r[gate] for r in rs):.3f}"
        lines += ["", f"Discovery {status('discovery')} ({spread('discovery')}); learning {status('learning')} "
                  f"({spread('learning')}). {sum(bool(r['passed']) for r in rs)} of {len(rs)} runs passed both gates. "
                  f"Between runs of one harness, a run's S varies with a standard deviation of {self._sigma():.3f}, "
                  "pooled over this campaign's harnesses."]
        lines += self._mechanisms(rs)
        errors = sorted({r["error"] for r in rs if r.get("error")})
        if errors:
            lines.append(f"The verifier reported errors: {'; '.join(e[:160] for e in errors)}.")
        if stage in TARGETS:
            lines += ["", self._task(stage, held, len(results))]
        return "\n".join(lines) + "\n"

    def _provenance(self, label: str, name: str) -> str:
        """How a trajectory's delivered model was built: refitted here, by its harness's procedure, or by the agent."""
        replay = self.dir / "replays" / f"{label}@{name}"
        if replay.is_dir():
            return f"this harness's procedure, refitted, {_size(replay / 'final_model.js')}"
        record = self.records()[name]
        delivered = evidence.read_json(record / "feedback.json").get("learning", {}).get("procedure", {}).get("delivered")
        size = _size(record / "final_model.js")
        return {"verified": f"its procedure (verified), {size}", "mismatch": f"not its procedure's output, {size}"}.get(
            delivered, f"built by the agent, {size}")

    def _mechanisms(self, results: list[dict]) -> list[str]:
        """Facts a failure mechanism follows from: which discovery requirement binds, hits lost to chained families,
        whether learning varies while discovery barely does, and the campaign's models by how they were built."""
        lines, gates = [], self.campaign.gates
        if len(gates) > 1:
            binds = defaultdict(int)
            for r in results:
                shares = {key: (r["gates"][key]["value"] or 0) / r["gates"][key]["required"] for key in gates}
                binds[min(shares, key=shares.get)] += 1
            lines.append("Discovery binds on " + ", ".join(f"`{key}` in {n} runs" for key, n in binds.items()) + ".")
            family = next((key for key in gates if "famil" in key), None)
            if family and "discovery_hits" in gates:
                merged = [int((r["gates"]["discovery_hits"]["value"] or 0) - (r["gates"][family]["value"] or 0)) for r in results]
                lines.append(f"Hits that fell into another hit's family (`discovery_hits` minus `{family}`), run by run: "
                             f"{', '.join(map(str, merged))}. Such a hit adds to the hit count but not to the family count, and "
                             "one that links two families removes one.")
                collapsed = sorted(((g["discovery_hits"]["value"] or 0, g[family]["value"] or 0, name)
                                    for name, path in self.records().items()
                                    for g in [feedback.official(path).get("gates") or {}] if family in g),
                                   key=lambda t: t[1] / t[0] if t[0] else 1)[:3]
                worst = [f"{name} ({hits:.0f} hits, {families:.0f} families)" for hits, families, name in collapsed
                         if hits and families < 0.9 * hits]
                if worst:
                    lines.append("Elsewhere in this campaign's runs, hits collapsed into far fewer families: "
                                 f"{'; '.join(worst)}. Near-copies of known hits are what chain families together.")
        d, l = [r["discovery"] for r in results], [r["learning"] for r in results]
        if len(results) >= 3 and statistics.pstdev(l) >= 0.05 and statistics.pstdev(d) < 0.5 * statistics.pstdev(l):
            lines.append(f"Learning varies from run to run (sd {statistics.pstdev(l):.3f}) more than twice as much as "
                         f"discovery (sd {statistics.pstdev(d):.3f}): runs that measured about equally well delivered models of "
                         "different quality, so the spread comes from how the model is built, not only from what was measured.")
        kinds = defaultdict(list)
        for path in self.records().values():
            delivered = evidence.read_json(path / "feedback.json").get("learning", {}).get("procedure", {}).get("delivered")
            kinds["a harness procedure (verified)" if delivered == "verified" else "the agent itself"].append(
                feedback.official(path)["learning"])
        if len(kinds) > 1:
            lines.append("Over every run of this campaign so far, models built by " + "; by ".join(
                f"{kind}: learning {statistics.fmean(v):.3f} on average over {len(v)} runs (sd {statistics.pstdev(v):.3f})"
                for kind, v in kinds.items()) + ".")
        return lines

    def _task(self, stage: str, held: list[str], trajectories: int) -> str:
        protocol = self.study.protocol
        own = TARGETS[stage]
        other = "learning" if own == "discovery" else "discovery"
        text = f"Your gate is {own}. " + ("It passes in every run: keep it so, and raise its margin. " if own in held else
                                          f"Raise it, without letting {other} fall. ")
        if other in held and stage == "modeling":
            text += ("Discovery passes in every run, so the experiment plan is frozen this step and there is no planning "
                     "editor: change only the modeling files, and nothing about which designs are ordered or how rounds "
                     "are planned. ")
        elif other in held:
            text += "Learning passes in every run: leave the modeling files as they are, and keep it passing. "
        if stage == "modeling":
            text += (f"How your branch is judged: if it changes only the modeling files and owns `scripts/model/fit.py`, "
                     f"the recursion refits that procedure on each of the {trajectories} trajectories above, from each one's "
                     "own `train.csv`, and scores the model with the official verifier, paired with the model that run "
                     "delivered. It must beat them on the same runs beyond the paired spread without passing fewer runs, "
                     f"and then {protocol.confirm_runs} fresh runs must deliver your procedure's output unmodified, reach "
                     "the current harness's level, and reproduce its discovery, which a branch that keeps the plan should "
                     "leave as it was: its level and, where families count, the share of hits lost to linked families. A "
                     "procedure that fails, or breaks the model contract, on a trajectory scores 0 there.")
        else:
            text += (f"How your branch is judged: it runs {protocol.probe_runs} times; a branch clearly below the current "
                     f"harness stops there, and the most promising branch gets {protocol.confirm_runs} fresh runs, which on "
                     "their own must reach the current harness's level. A gate that passes in every current run may not "
                     "fail in any of yours.")
        return text + (f" State the failure mechanism, your change and its expected effect on each gate in `{CHANGE}` "
                       "(see \"When you finish\"). Make only the changes this diagnosis supports.")

    def _credit(self, label: str, tree: Path, current: Path, current_label: str, step: int, editor: str) -> None:
        """Log what a harness changed against the one it was edited from, which gate it targeted, and how both moved."""
        if any(e.get("harness") == label for e in _events(self.dir / "events.jsonl", "credit")):
            return
        mine, theirs = self._distinct(label), self._distinct(current_label)
        if not mine or not theirs:
            return
        test = feedback.choose([feedback.official(r) for r in mine], [feedback.official(r) for r in theirs], SURE)
        self._log("credit", step=step, harness=label, editor=editor, edited_from=current.name,
                  targeted=TARGETS.get(editor, "both"),
                  components=_changed(current, tree), change=test["change"], gain=test["gain"],
                  candidate=test["candidate"], incumbent=test["incumbent"])

    def _duplicate(self, composed: Path, best_label: str, step: int) -> str | None:
        """The best harness so far or an earlier composed harness that ``composed`` is identical to:
        running it again would only give the same comparison more chances to clear its noise."""
        trees = {self.dir / "harness" / ("h0" if best_label == "base" else best_label): best_label}
        trees |= {self.dir / "harness" / f"k{j}": f"k{j}" for j in range(1, step)}
        mine = _digest(composed)
        return next((label for tree, label in trees.items() if tree.is_dir() and _digest(tree) == mine), None)

    def _stock(self) -> Path:
        path = self.dir / "harness" / "h0"
        if not path.exists():
            harness.write_stock(path)
        return path

    def _adopt(self, tree: Path) -> Path:
        """Copy a starting harness in, once; it must pass the guard like any other."""
        path = self.dir / "harness" / "start"
        if not path.exists():
            staging = self._staging(path)
            harness.complete(harness.copy(tree, staging))  # a tree from before a component existed gets its placeholder
            problems = guard.check(staging, self.campaign)
            if problems:
                raise ValueError(f"the starting harness fails the guard: {problems}")
            staging.rename(path)
        return path

    def _run(self, trees: Mapping[str, Path], attempts: int | Mapping[str, int]) -> None:
        """Run each harness ``attempts`` times (or as many as ``attempts`` gives its label) on the
        campaign; record the new trials with their feedback."""
        pending = []
        for label, tree in trees.items():
            count = attempts if isinstance(attempts, int) else attempts[label]
            if not count:
                continue
            targets = self._targets(label, count)
            if not all(target.exists() for target in targets):
                name = f"{self.executor.name}-{self.campaign.name}-{self.dir.name}-{label}"
                pending.append((Job(name, self.campaign, self.executor, tree, count), targets))
        if pending:
            self._log("run", jobs=[job.name for job, _ in pending])
            self._record(pending)
        self._feedback()

    def _more(self, label: str, tree: Path, extra: int, after: int) -> None:
        """``extra`` more runs of a harness that had ``after`` runs, recorded after them under its label;
        numbered from ``after``, so a resumed recursion finds them rather than adding more."""
        root = self.dir / "records"
        targets = [root / f"{label}-{i}" for i in range(after + 1, after + extra + 1)]
        if not all(target.exists() for target in targets):
            name = f"{self.executor.name}-{self.campaign.name}-{self.dir.name}-{label}-more{after}"
            job = Job(name, self.campaign, self.executor, tree, extra)
            self._log("repeat", jobs=[name], runs=extra)
            self._record([(job, targets)])
        self._feedback()

    def _record(self, pending: list[tuple[Job, list[Path]]]) -> None:
        if pending:
            trials = self.runner.run([job for job, _ in pending])
            stock = self.dir / "harness" / "h0"
            for job, targets in pending:
                # A resumed job keeps its finished trials; give only the unrecorded ones to missing records.
                recorded = {evidence.source(target) for target in targets if target.exists()}
                fresh = [trial for trial in trials[job.name] if str(trial) not in recorded]
                for trial, target in zip(fresh, [target for target in targets if not target.exists()]):
                    evidence.record(trial, target, self.campaign, harness.diff(stock, job.harness), job.harness.name)

    def _feedback(self) -> None:
        """Feedback for every record that lacks it; it depends only on the record and its harness. Then, in sequential
        mode, the shadow results of the runs that captured the agent's own model (``_shadows``)."""
        for record in self.records(evaluation=True).values():
            if not (record / "feedback.json").is_file():
                tree = self.dir / "harness" / (record / "harness.txt").read_text().strip()
                feedback.write(record, self.campaign, tree, self.validator, self.checker)
        if self.sequential:
            self._shadows()

    def _shadows(self) -> None:
        """A harness that owns a procedure delivers its model after the agent, and the record keeps the model the agent
        delivered itself (evidence.agent_model). The agent saw exactly what it sees under every harness with the same
        visible digest and no procedure (its twins), so that model, scored by the verifier on the run's own ledger, is
        a twin's run on the very same trajectory: each such run is a matched pair, one trajectory under both harnesses.
        Written as ``replays/<twin>@<run>``; evaluation runs are left out."""
        trees = {("base" if t.name == "h0" else t.name): t for t in (self.dir / "harness").iterdir()
                 if t.is_dir() and not t.name.endswith(".tmp")} if (self.dir / "harness").is_dir() else {}
        for name, record in self.records().items():
            tree = self._tree((record / "harness.txt").read_text().strip())
            if not (record / "agent_model.js").is_file() or not _owns(tree):
                continue
            seen = self._visible(tree)
            for twin, other in trees.items():
                if not _owns(other) and self._visible(other) == seen:
                    self.replayer.score(record, record / "agent_model.js", self.dir / "replays" / f"{twin}@{name}")

    def _visible(self, tree: Path) -> str:
        cache = self.__dict__.setdefault("_seen", {})
        if tree not in cache:
            cache[tree] = harness.visible(tree)
        return cache[tree]

    def _targets(self, label: str, attempts: int) -> list[Path]:
        root = self.dir / "records"
        return [root / label] if attempts == 1 else [root / f"{label}-{i}" for i in range(1, attempts + 1)]

    def _edit(self, stage: str, name: str, current: Path, candidates: Mapping[str, Path] | None = None,
              seen: set[str] | None = None, diagnosis: str = "") -> Path:
        """One improver session. Its harness is kept only if it passes the guard and the
        procedure audit, after one retry. ``seen`` limits the evidence to those harness labels."""
        out = self.dir / "harness" / name
        if out.exists():
            return out
        workspace = self.dir / "sessions" / name
        shutil.rmtree(workspace, ignore_errors=True)  # an unfinished session starts over
        prepare(workspace, stage, self.campaign, current, self.records(seen), candidates, diagnosis)
        if self.sequential:
            self._export_branches(workspace / "branches", before=int(name[1:].split("-")[0]))
        scores = {"rule": RULE_SEQUENTIAL if self.sequential else RULE, "current": current.name,
                  "harnesses": self.estimates(seen)}
        if self.sequential:
            scores["replays"] = self._replay_table()
        (workspace / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
        notes = workspace / ("DECISIONS.md" if stage == "composer" else "RATIONALE.md")
        problems: list[str] = []
        for attempt in (1, 2):
            notes.unlink(missing_ok=True)
            code = self.improver.session(workspace, attempt)
            for cache in (workspace / "harness").rglob("__pycache__"):  # bytecode from test runs, not content
                shutil.rmtree(cache, ignore_errors=True)
            for folder in sorted((workspace / "harness").rglob("*"), key=lambda p: -len(p.parts)):
                if folder.is_dir() and not any(folder.iterdir()):  # what an editor emptied, it removed
                    folder.rmdir()
            if not notes.is_file():
                # A session that could not write its notes did not run; stop rather than read
                # an untouched harness as a decision to change nothing. Rerunning retries it.
                self._log("session-failed", session=name, attempt=attempt, exit_code=code)
                raise RuntimeError(f"improver session {name} wrote no {notes.name}; see {workspace}")
            if self.sequential:
                harness.complete(workspace / "harness")
                procedure.complete(workspace / "harness" / procedure.MODEL_DIR, self.campaign)
                if stage in SCOPES:  # each editor changes only its own stage's files (harness.MODELING)
                    reverted = harness.revert(current, workspace / "harness", OWNS[stage])
                    if reverted:
                        self._log("scope", session=name, attempt=attempt, reverted=reverted)
            problems = guard.check(workspace / "harness", self.campaign)
            if not problems and self.sequential and stage != "composer":
                problems = change_problems(workspace / CHANGE)
            if not problems:
                problems = self._audit(workspace)
            self._log("session", session=name, attempt=attempt, exit_code=code, problems=problems)
            if not problems:
                break
            (workspace / "GUARD.md").write_text(guard.report(problems))
        staging = self._staging(out)
        harness.copy(current if problems else workspace / "harness", staging)
        if not problems:
            shutil.copyfile(notes, notes_path(out))
            if (workspace / CHANGE).is_file():
                shutil.copyfile(workspace / CHANGE, change_path(out))
        staging.rename(out)
        if problems:
            self._log("fallback", session=name, kept=current.name)
        return out

    def _audit(self, workspace: Path) -> list[str]:
        """The procedure audit (``check_procedure``) of a session's harness, logged."""
        model = workspace / "harness" / procedure.MODEL_DIR
        if not (model / "fit.py").is_file() or not any((workspace / "evidence").glob("*/train.csv")):
            return []
        report, problems = check_procedure(model, workspace / "evidence", self.campaign, self.auditor, self.scorer)
        self._log("audit", session=workspace.name, report=report)
        return problems

    def _freeze(self) -> None:
        """Record the protocol a recursion starts with, and refuse to resume it under another."""
        current = protocol_files()
        path = self.dir / "protocol.json"
        if path.is_file():
            recorded = json.loads(path.read_text())["files"]
            changed = sorted(f for f in set(recorded) | set(current) if recorded.get(f) != current.get(f))
            if changed and not self.allow_protocol_change:
                raise SystemExit(f"{self.dir}: the protocol changed since this recursion started ({', '.join(changed)}); "
                                 "pass --allow-protocol-change to resume it anyway")
            if not changed:
                return
            self._log("protocol-change", files=changed)
        self.dir.mkdir(parents=True, exist_ok=True)
        digest = protocol_digest(current)
        path.write_text(json.dumps({"digest": digest, "files": current}, indent=2) + "\n")

    def _staging(self, path: Path) -> Path:
        staging = path.with_name(path.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        staging.parent.mkdir(parents=True, exist_ok=True)
        return staging

    def _log(self, event: str, **fields) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **fields}
        with (self.dir / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(entry) + "\n")


def check_procedure(model: Path, evidence_dir: Path, campaign: Campaign, auditor, scorer) -> tuple[dict, list[str]]:
    """A procedure must learn from train.csv alone: fitted on the evidence with its targets
    shuffled, its model must not still rank the evidence designs; and its model must run in
    the benchmark's sandbox within the campaign's ``model_ms``. Returns the report and the problems."""
    report = auditor(model, evidence_dir)
    source = report.pop("source", None)
    if source and campaign.model_ms:
        designs = sorted(row[campaign.input] for row in feedback.catalog(campaign).values())[:TIMED]
        try:
            # Every design starts a fresh worker; what counts is the time the model adds to that.
            report["ms_per_design"] = round(_timed(scorer, source, designs) - _timed(scorer, EMPTY, designs), 1)
        except Exception as failure:  # noqa: BLE001 - any failure to run is the problem to report
            report["error"] = f"its model does not run in the benchmark's sandbox: {str(failure)[:200]}"
    if "error" in report:
        return report, [f"{procedure.MODEL_DIR}/fit.py: {report['error']}"]
    if abs(report["spearman"]) > PRIOR_LIMIT:
        return report, [f"{procedure.MODEL_DIR}/fit.py predicts measured outcomes without learning them: fitted on the "
                        f"evidence with its targets shuffled, its model still ranks those designs with Spearman "
                        f"{report['spearman']:.2f}. The model must learn only from the train.csv it is given."]
    if campaign.model_ms and report.get("ms_per_design", 0) > campaign.model_ms:
        return report, [f"{procedure.MODEL_DIR}/fit.py: its model adds {report['ms_per_design']:.0f} ms per design in the "
                        f"benchmark's sandbox, where every design is scored in a fresh worker that parses the whole model; "
                        f"the delivered model may add at most {campaign.model_ms:.0f} ms. Make it smaller: fewer trees, "
                        f"support points or features."]
    return report, []


class StockRuns:
    """Recursion ``index``'s stock-harness runs on a campaign, for one executor: the opening Base runs its
    editors start from and the Base evaluation runs. The index's RAS recursion and its non-recursive
    control share them, so the two arms start from the same evidence and are judged against the same
    Base; the first recursion to need them runs them, the other waits."""

    def __init__(self, campaign: Campaign, executor: Executor, runner: HarborRunner, root: Path, index: int):
        self.campaign, self.executor, self.runner, self.index = campaign, executor, runner, index
        self.dir = root / STOCK / executor.name / campaign.name / f"r{index}"

    def records(self, seen: set[str] | None = None) -> dict[str, Path]:
        return _records(self.dir / "records", seen)

    def ensure(self, label: str, tree: Path, count: int, validator: feedback.Validator,
               checker: feedback.Checker | None = None) -> None:
        if not count:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / f".{label}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # the index's other recursion, in this process or another, waits here
            targets = [self.dir / "records" / f"{label}-{i}" for i in range(1, count + 1)]
            missing = [target for target in targets if not target.exists()]
            if missing:
                name = f"{self.executor.name}-{self.campaign.name}-r{self.index}-{label}"
                trials = self.runner.run([Job(name, self.campaign, self.executor, tree, count)])[name]
                recorded = {evidence.source(target) for target in targets if target.exists()}
                fresh = [trial for trial in trials if str(trial) not in recorded]
                for trial, target in zip(fresh, missing):
                    evidence.record(trial, target, self.campaign, "", tree.name)
            for record in self.records().values():
                if not (record / "feedback.json").is_file():
                    feedback.write(record, self.campaign, tree, validator, checker)


def _unfollowed(record: Path) -> str | None:
    """Why a planner run's rounds are not its planner's: a planned round the agent ordered less than FOLLOWED of, or a
    round after the first in which the planner neither proposed (its proposals) nor deferred (its insufficient-evidence
    message in the agent's trajectory), so the agent never ran it there. A deferred round is the agent's by design."""
    proposals = {int(p.stem.split("_")[1]): p for p in (record / "proposals").glob("round_*.csv")} \
        if (record / "proposals").is_dir() else {}
    trajectory = (record / "trajectory.json").read_text(errors="replace") if (record / "trajectory.json").is_file() else ""
    deferred = {int(n) for n in re.findall(r"insufficient evidence for round (\d+)", trajectory)}
    jobs = {j["stage"]: set(map(str, j["designs"])) for j in evidence.read_json(record / "ledger.json").get("jobs", [])
            if j.get("status") == "completed"}
    for stage in sorted(jobs):
        if stage in proposals:
            proposed = {line.strip() for line in proposals[stage].read_text().splitlines()[1:] if line.strip()}
            if len(proposed & jobs[stage]) < FOLLOWED * len(proposed):
                return f"round {stage} ordered {len(proposed & jobs[stage])} of the planner's {len(proposed)} proposals"
        elif stage > 1 and stage not in deferred:
            return f"round {stage}: the agent did not run the planner"
    return None


def _owns(tree: Path) -> bool:
    """Whether a harness owns a modeling procedure, which a replay can refit."""
    return (tree / procedure.MODEL_DIR / "fit.py").is_file()


def change_path(tree: Path) -> Path:
    """Where a session's CHANGE.json is kept, next to the harness it produced."""
    return tree.with_name(f"{tree.name}.change.json")


def change_problems(path: Path) -> list[str]:
    """What an editor's CHANGE.json lacks: the failure mechanism the evidence shows, the change, the files it
    touched, and the expected effect on each gate, each starting with raise, keep or lower."""
    if not path.is_file():
        return [f"{CHANGE}: missing. State the failure mechanism, the change and its expected effect on each gate."]
    try:
        statement = json.loads(path.read_text())
    except ValueError as error:
        return [f"{CHANGE}: not valid JSON ({error})"]
    if not isinstance(statement, dict):
        return [f"{CHANGE}: must be one JSON object"]
    problems = [f"{CHANGE}: `{key}` is missing or empty" for key in ("mechanism", "change", "components", "expected")
                if not statement.get(key)]
    expected = statement.get("expected") if isinstance(statement.get("expected"), dict) else {}
    for gate in feedback.GATES:
        effect = expected.get(gate) if isinstance(expected.get(gate), dict) else {}
        if effect.get("direction") not in DIRECTIONS:
            problems.append(f"{CHANGE}: `expected.{gate}.direction` must be one of {', '.join(DIRECTIONS)}")
    return problems


def _size(model: Path) -> str:
    return f"{model.stat().st_size / 1000:.0f} kB" if model.is_file() else "no model"


def _records(root: Path, seen: set[str] | None = None) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    return {path.name: path for path in sorted(root.iterdir())
            if path.is_dir() and not path.name.endswith(".tmp") and (seen is None or _label(path.name) in seen)}


def _label(record: str) -> str:
    """A record's harness label: k1-2 -> k1, base-3 -> base, k1-planning -> k1-planning."""
    head, _, tail = record.rpartition("-")
    return head if tail.isdigit() and head else record


def _timed(scorer, source: str, designs: list[str]) -> float:
    """Milliseconds per design the benchmark's sandbox takes to score ``designs`` with ``source``."""
    start = time.monotonic()
    scorer(source, designs)
    return 1000 * (time.monotonic() - start) / len(designs)


def protocol_files() -> dict[str, str]:
    """SHA-256 of every file that defines the protocol: the pipeline, its prompts, toolkit and configuration."""
    files = sorted({path for pattern in PROTOCOL for path in ROOT.glob(pattern)
                    if path.is_file() and "__pycache__" not in path.parts})
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _digest(tree: Path) -> str:
    """SHA-256 over a harness tree's files, paths and contents."""
    digest = hashlib.sha256()
    for path in sorted(p for p in tree.rglob("*") if p.is_file()):
        digest.update(path.relative_to(tree).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def protocol_digest(files: dict[str, str] | None = None) -> str:
    """The digest ``protocol.json`` records: SHA-256 over every protocol file's own digest."""
    return hashlib.sha256(json.dumps(files or protocol_files(), sort_keys=True).encode()).hexdigest()


def _changed(before: Path, after: Path) -> list[str]:
    """The harness files that differ between two trees."""
    names = set(harness.files(before)) | set(harness.files(after))
    return sorted(n for n in names if not ((before / n).is_file() and (after / n).is_file()
                                            and (before / n).read_bytes() == (after / n).read_bytes()))


def _identity(record: Path) -> tuple:
    """What makes two runs the same observation: the designs measured, in order, and the model delivered."""
    model = record / "final_model.js"
    return _measured(record), model.read_bytes() if model.is_file() else b""


def _measured(record: Path) -> tuple:
    """The designs a run measured, in order: two runs with the same tuple gave the same evidence."""
    ledger = evidence.read_json(record / "ledger.json")
    return tuple(str(row) for job in ledger.get("jobs", []) for row in job.get("results", []))


def _measured_rounds(record: Path, rounds: int) -> tuple:
    """The designs a run ordered in its first ``rounds`` rounds, in order."""
    jobs = sorted((j for j in evidence.read_json(record / "ledger.json").get("jobs", []) if j.get("status") == "completed"),
                  key=lambda j: j["stage"])
    return tuple(tuple(map(str, j["designs"])) for j in jobs[:rounds])


def _events(path: Path, *kinds: str) -> list[dict]:
    if not path.is_file():
        return []
    events = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [e for e in events if e["event"] in kinds]


def spent(root: Path) -> float:
    """USD spent under a runs directory: every trial Harbor finished (set-aside ones too) and every improver session."""
    from . import config, usage
    models = {name: executor.model for name, executor in config.load().executors.items()}
    trials = sum((evidence.read_json(path).get("agent_result") or {}).get("cost_usd") or 0.0
                 for path in (root / "jobs").glob("**/result.json") if path.parent.parent.name != "jobs")
    sessions = sum(usage.session(log, models.get(log.parts[-6], "")).get("cost_usd") or 0.0
                   for log in (root / "recursions").glob("*/*/*/sessions/*/session*.log"))
    return trials + sessions


def status(root: Path = RUNS) -> list[dict]:
    """Progress of every recursion under ``root``."""
    rows = []
    for path in sorted(p for p in (root / "recursions").glob("*/*/*") if p.is_dir()):
        index = "".join(c for c in path.name if c.isdigit())
        stock = root / STOCK / path.parent.parent.name / path.parent.name / f"r{index}" / "records"
        names = [p.name for folder in (path / "records", stock) for p in folder.glob("*")
                 if p.is_dir() and not p.name.endswith(".tmp")]
        steps = [int(p.name[1:]) for p in (path / "harness").glob("k*") if p.name[1:].isdigit()]
        events = path / "events.jsonl"
        rows.append({
            "executor": path.parent.parent.name,
            "campaign": path.parent.name,
            "recursion": path.name,
            "base": sum(_label(name) == "base" for name in names),
            "step": max(steps, default=0),
            "rollbacks": len(_events(events, "rollback")),
            "selected": (path / "selected.txt").read_text().strip() if (path / "selected.txt").is_file() else "-",
            "eval": sum(_label(name).startswith("eval-") and _label(name) != EVAL_BASE for name in names),
            "eval_base": sum(_label(name) == EVAL_BASE for name in names),
            "fallbacks": len(_events(events, "fallback")),
        })
    return rows
