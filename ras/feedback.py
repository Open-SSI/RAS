"""Lab feedback for improvers, and the official objective selection uses.

``rounds`` summarizes each round's batch on the campaign objective, with the hit rate
where the instruction states a hit threshold and how often random ordering hits.

``learning`` checks a harness's own modeling procedure (``scripts/model/fit.py``), where it has
one: refitting it on the run's rows must reproduce the delivered model (``verified``), and it
is cross-validated on those rows by the campaign's ranking metric, for the editors' information
only. A missing procedure, a mismatch or a failure is reported, and nothing generic is
substituted for it (procedure.py).

Selection uses the verifier itself. ``objective`` reads each run's official result
(``verifier.json``, evidence.verifier): each gate's share of its threshold, and the harness's
S(H) = min(D / D_threshold, L / L_threshold). ``choose`` compares a candidate with the incumbent on
S, with the incumbent's lower gate as the target and its other gate as a constraint; once both
pass, on the share of passing runs and the margin.

Sequential selection (``compare``, ``screen``, ``accept``) works run by run instead: each run's utility
is its own bottleneck, min(D, L), capped at ``CAP``, and a candidate is compared with the incumbent on
the mean utility of its runs, paired run by run where both were scored on the same trajectory (a
replay, replay.py), unpaired otherwise, with the run-to-run spread pooled over every harness of the
campaign rather than read off two runs. The pass shares and the gates the incumbent passes in every
run constrain it.
"""

from __future__ import annotations

import csv
import functools
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path

from . import procedure, ranking
from .config import Campaign
from .evidence import read_json

# (harness tree, training rows, delivered model, probe inputs) -> validation report
Validator = Callable[[Path | None, list[dict], Path, list[str]], dict]
Checker = Callable[[Path], dict]  # a delivered model -> the verifier's contract report (contract.check)


def value(row: Mapping, campaign: Campaign) -> float:
    """The campaign objective in its own units, e.g. lead response gated on zinc response."""
    spec = campaign.objective
    if spec.zero_above and float(row[spec.zero_above[0]]) > spec.zero_above[1]:
        return 0.0
    return float(row[spec.field])


def utility(row: Mapping, campaign: Campaign) -> float:
    """The objective with higher always better, for modeling and ranking."""
    objective = value(row, campaign)
    return objective if campaign.objective.higher_is_better else -objective


def measured(ledger: Mapping, campaign: Campaign) -> list[tuple[int, Mapping]]:
    """(round, result row) for every design the lab returned, in ledger order."""
    return [(job["stage"], row) for job in ledger.get("jobs", []) if job.get("status") == "completed"
            for row in job.get("results", [])]


def rounds(ledger: Mapping, campaign: Campaign) -> list[dict]:
    by_round: dict[int, list[float]] = defaultdict(list)
    for stage, row in measured(ledger, campaign):
        by_round[stage].append(value(row, campaign))
    best = max if campaign.objective.higher_is_better else min
    summary = []
    for stage in sorted(by_round):
        values = by_round[stage]
        entry = {"round": stage, "designs": len(values),
                 "mean": statistics.fmean(values), "best": best(values)}
        if campaign.hit:
            kind, threshold = campaign.hit
            hits = sum(v > threshold if kind == "above" else v >= threshold for v in values)
            entry |= {"hits": hits, "hit_rate": hits / len(values),
                      "random_hit_rate": campaign.random_hit_rate}
        summary.append(entry)
    return summary


def training_rows(ledger: Mapping, campaign: Campaign) -> list[dict]:
    """The public seed and the run's measured designs, as the toolkit's train.csv rows."""
    rows = []
    for seed in campaign.seed:
        with seed.file.open(newline="") as handle:
            rows += [_row(record[seed.input], record, record, campaign) for record in csv.DictReader(handle)]
    designs = catalog(campaign)
    for _, result in measured(ledger, campaign):
        meta = designs[str(result[campaign.id])]
        rows.append(_row(meta[campaign.input], result, meta, campaign))
    return rows


def measured_rows(ledger: Mapping, campaign: Campaign) -> list[dict]:
    """The run's measured designs alone, as train.csv rows."""
    designs = catalog(campaign)
    return [_row(designs[str(r[campaign.id])][campaign.input], r, designs[str(r[campaign.id])], campaign)
            for _, r in measured(ledger, campaign)]


def _row(text: str, values: Mapping, meta: Mapping, campaign: Campaign) -> dict:
    row = {"input": text, "target": utility(values, campaign)}
    if campaign.fit and campaign.fit.group:
        row["group"] = "|".join(str(meta[column]) for column in campaign.fit.group)
    return row


def learning(ledger: Mapping, campaign: Campaign, harness_tree: Path | None, delivered: Path,
             validator: Validator) -> dict:
    rows = training_rows(ledger, campaign)
    designs = sorted(row[campaign.input] for row in catalog(campaign).values())
    probe = designs[:: max(1, len(designs) // procedure.PROBE)][: procedure.PROBE]
    return {"rows": len(rows), "procedure": validator(harness_tree, rows, delivered, probe)}


def write(record: Path, campaign: Campaign, harness_tree: Path | None, validator: Validator,
          checker: Checker | None = None) -> dict:
    """feedback.json for one run record; computed once, since it depends only on that run."""
    path = record / "feedback.json"
    if path.is_file():
        return json.loads(path.read_text())
    ledger = read_json(record / "ledger.json")
    report = {"objective": describe(campaign), "rounds": rounds(ledger, campaign),
              "learning": learning(ledger, campaign, harness_tree, record / "final_model.js", validator)}
    error = read_json(record / "verifier.json").get("error") or ""
    if checker is not None and "Timeout" in error and (record / "final_model.js").is_file():
        # The verifier ran out of time: was it the model, or a loaded host? Time the model alone.
        report["contract"] = checker(record / "final_model.js")
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def discovery(record: Path) -> float | None:
    """The weakest discovery gate as a share of its requirement."""
    gates = read_json(record / "discovery.json").get("gates", {})
    shares = [g["value"] / g["required"] for g in gates.values() if isinstance(g.get("value"), (int, float))]
    return min(shares) if shares else None


def verified(record: Path) -> bool:
    return read_json(record / "feedback.json").get("learning", {}).get("procedure", {}).get("delivered") == "verified"


def official(record: Path) -> dict:
    """A run's official verifier result (evidence.verifier)."""
    return read_json(record / "verifier.json")


GATES = ("discovery", "learning")


def objective(results: list[dict]) -> dict:
    """A harness's official scores over its distinct runs: each gate's mean share of its threshold,
    S = min(D/tD, L/tL) of those means, and the share of runs that passed both gates."""
    mean = lambda values: statistics.fmean(values) if values else 0.0
    out = {gate: mean([r[gate] for r in results]) for gate in GATES}
    return out | {"S": min(out.values()), "passed": mean([float(r["passed"]) for r in results]), "runs": len(results),
                  "per_run": {gate: [r[gate] for r in results] for gate in (*GATES, "bottleneck")}}


def choose(candidate: list[dict], incumbent: list[dict], sure: float = 1.0) -> dict:
    """Whether the candidate's runs beat the incumbent's, by S(H) = min(D/tD, L/tL). Each argument holds
    one verifier result per distinct run (identical runs are one observation).

    The incumbent's lower gate is the target. While either harness is below 1 on S, the candidate must
    raise S; once both are at least 1, it must raise the share of runs that pass both gates, or keep it
    and raise S, the margin. Either way it may not lose a gate the incumbent passes, and where the
    incumbent fails both, the other gate may not fall by more than one standard error of the runs' spread. A gain in S no larger than ``sure``
    standard errors is uncertain: the recursion repeats the candidate before keeping it."""
    c, i = objective(candidate), objective(incumbent)
    variance = lambda xs: statistics.variance(xs) / len(xs) if len(xs) > 1 else 0.0
    spread = lambda key: math.sqrt(variance(c["per_run"][key]) + variance(i["per_run"][key]))
    target = "discovery" if i["discovery"] <= i["learning"] else "learning"
    other = "learning" if target == "discovery" else "discovery"
    if c["S"] >= 1 and i["S"] >= 1:
        rule, accept = "robustness", (c["passed"], c["S"]) > (i["passed"], i["S"])
    else:
        rule, accept = "bottleneck", c["S"] > i["S"]
    lost = [gate for gate in GATES if i[gate] >= 1 > c[gate]]
    regressed = i[other] < 1 and c[other] < i[other] - spread(other)  # a passing gate need only stay passing
    accept = accept and not lost and not regressed
    gain, error = c["S"] - i["S"], spread("bottleneck")
    keep = lambda o: {k: round(v, 4) for k, v in o.items() if k not in ("per_run", "runs")} | {"runs": o["runs"]}
    return {"accept": accept, "sure": accept and gain > sure * error, "rule": rule, "target": target,
            "gain": round(gain, 4), "error": round(error, 4),
            "change": {gate: round(c[gate] - i[gate], 4) for gate in GATES},
            "lost_gate": lost, "other_gate_regressed": other if regressed else None,
            "candidate": keep(c), "incumbent": keep(i)}


def estimate(record: Path) -> tuple[float | None, float | None]:
    """(discovery share, the procedure's own cross-validation when verified): for reports."""
    validated = read_json(record / "feedback.json").get("learning", {}).get("procedure", {})
    return official(record).get("discovery"), (validated["cv"]["ranking"] if validated.get("delivered") == "verified" else None)


def describe(campaign: Campaign) -> dict:
    spec = campaign.objective
    text = spec.field + (f", set to 0 when {spec.zero_above[0]} > {spec.zero_above[1]}"
                         if spec.zero_above else "")
    out = {"value": text, "higher_is_better": spec.higher_is_better,
           "groups": list(campaign.fit.group) if campaign.fit else []}
    if campaign.ranking:
        out["learning_metric"] = ranking.describe(campaign.ranking)
    if campaign.contract:
        limit = campaign.contract
        out["model_contract"] = (f"at most {limit.max_bytes} bytes; {limit.designs} designs scored twice, each in a fresh "
                                 f"worker, within {limit.timeout_s:.0f} s; finite, deterministic, pointwise scores")
    return out


def catalog(campaign: Campaign) -> dict[str, dict]:
    return _catalog(campaign.catalog, campaign.id)


@functools.cache
def _catalog(path: Path, key: str) -> dict[str, dict]:
    with path.open(newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


CAP = 1.2  # a run's utility counts the margin above both thresholds up to 20%: a wide margin cannot hide a failed run
FLOOR = 0.01  # the smallest standard error a comparison claims, however alike its runs


def run_utility(result: Mapping) -> float:
    """One run's normalized bottleneck, min(D/tD, L/tL), capped at CAP."""
    return min(result["discovery"], result["learning"], CAP)


def pooled_sd(groups: list[list[float]]) -> float | None:
    """The run-to-run standard deviation of utility within a harness, pooled over harnesses with two or more
    distinct runs; None without any."""
    df = sum(len(g) - 1 for g in groups if len(g) > 1)
    if not df:
        return None
    return math.sqrt(sum((len(g) - 1) * statistics.variance(g) for g in groups if len(g) > 1) / df)


def compare(candidate: list[dict], incumbent: list[dict], sigma: float,
            pairs: list[tuple[dict, dict]] | None = None, fixed: tuple[bool, bool] = (False, False)) -> dict:
    """The candidate's runs against the incumbent's, on mean utility. With two or more ``pairs`` (the
    candidate's and the incumbent's result on the same trajectory), the comparison is paired and its
    standard error is the paired differences' own. Otherwise it is unpaired, with ``sigma`` the pooled
    spread; a harness whose runs were all identical (``fixed``) contributes no spread. Pass shares and
    gate means are over the same runs the difference is."""
    paired = bool(pairs) and len(pairs) >= 2
    c, i = ([a for a, _ in pairs], [b for _, b in pairs]) if paired else (candidate, incumbent)
    mean = lambda rs, f: statistics.fmean(f(r) for r in rs) if rs else 0.0
    delta = mean(c, run_utility) - mean(i, run_utility)
    if paired:
        differences = [run_utility(a) - run_utility(b) for a, b in pairs]
        se = statistics.stdev(differences) / math.sqrt(len(differences))
    else:
        se = math.sqrt(sum(0.0 if f or not rs else sigma ** 2 / len(rs) for rs, f in ((c, fixed[0]), (i, fixed[1]))))
    se = max(se, FLOOR)
    side = lambda rs: {"discovery": round(mean(rs, lambda r: r["discovery"]), 4),
                       "learning": round(mean(rs, lambda r: r["learning"]), 4),
                       "utility": round(mean(rs, run_utility), 4), "passed": round(mean(rs, lambda r: float(r["passed"])), 4),
                       "runs": len(rs)}
    return {"paired": paired, "delta": round(delta, 4), "se": round(se, 4), "z": round(delta / se, 2),
            "sigma": round(sigma, 4), "candidate": side(c), "incumbent": side(i)}


def reliable(results: list[dict]) -> list[str]:
    """The gates a harness passes in every one of its runs (two or more distinct runs)."""
    return [gate for gate in GATES if len(results) >= 2 and all(r[gate] >= 1 for r in results)]


def lost(candidate: list[dict] | Mapping[str, dict], kept: list[str], incumbent: Mapping[str, dict] | None = None) -> list[str]:
    """The gates in ``kept`` (the incumbent's reliable gates) that a candidate run fails. Given both harnesses' results by
    trajectory, a trajectory the incumbent also has counts only where the incumbent passes the gate there: a gate the
    shared trajectory itself failed was not lost by the candidate."""
    if incumbent is None:
        runs = candidate.values() if isinstance(candidate, Mapping) else candidate
        return [gate for gate in kept if any(r[gate] < 1 for r in runs)]
    return [gate for gate in kept if any(r[gate] < 1 and (t not in incumbent or incumbent[t][gate] >= 1)
                                         for t, r in candidate.items())]


def screen(test: dict, lost_gates: list[str]) -> str:
    """After the first runs: ``reject`` a clear regression (a reliable gate lost, or a mean utility lower by
    more than one standard error, or by anything at all on paired trajectories, where the spread is the
    comparison's own), ``promising`` a higher mean utility, ``even`` otherwise. Only promising candidates
    get confirmation runs."""
    if lost_gates or test["delta"] < -test["se"] or (test["paired"] and test["delta"] <= 0):
        return "reject"
    return "promising" if test["delta"] > 0 else "even"


def accept(test: dict, lost_gates: list[str], fresh: list[dict], incumbent_utility: float, sure: float = 1.0,
           fresh_pairs: list[tuple[dict, dict]] | None = None, gate: str | None = None,
           incumbent_gate: float | None = None) -> list[str]:
    """Why a confirmed candidate may not replace the incumbent; empty when it may. Its gain must exceed ``sure``
    standard errors; its runs may not pass less often than the incumbent's; it may not lose a gate the
    incumbent passes in every run; and its fresh confirmation runs, which played no part in choosing it,
    must on their own show it at least as good as the incumbent: the guard against keeping the luckiest branch.
    Where the incumbent has a result on those very trajectories (``fresh_pairs``), that is their mean paired
    difference in utility; otherwise, for a candidate that changes only ``gate`` (a plan-keeping procedure, whose
    fresh runs' discovery is the incumbent's own), the fresh runs' mean on that gate against ``incumbent_gate``;
    otherwise their mean utility against the incumbent's."""
    reasons = []
    if test["delta"] <= sure * test["se"]:
        reasons.append(f"gain {test['delta']:+.3f} within {sure:g} standard error ({test['se']:.3f})")
    if test["candidate"]["passed"] < test["incumbent"]["passed"]:
        reasons.append(f"passes {test['candidate']['passed']:.2f} of runs against {test['incumbent']['passed']:.2f}")
    if lost_gates:
        reasons.append(f"lost {', '.join(lost_gates)}, which the incumbent passes in every run")
    if fresh_pairs:
        gain = statistics.fmean(run_utility(a) - run_utility(b) for a, b in fresh_pairs)
        if gain < 0:
            reasons.append(f"on its fresh trajectories it scores {gain:+.3f} against the incumbent on the same ones")
    elif fresh and gate is not None and incumbent_gate is not None:
        mean = statistics.fmean(r[gate] for r in fresh)
        if mean < incumbent_gate:
            reasons.append(f"its fresh runs' {gate} averages {mean:.3f}, below the incumbent's {incumbent_gate:.3f}")
    elif fresh and statistics.fmean(run_utility(r) for r in fresh) < incumbent_utility:
        reasons.append(f"its fresh runs average {statistics.fmean(run_utility(r) for r in fresh):.3f}, below the "
                       f"incumbent's {incumbent_utility:.3f}")
    return reasons


def linked(result: Mapping, family: str) -> float:
    """The share of a run's hits that fell into another hit's family: hits minus families, over hits."""
    gates = result.get("gates") or {}
    hits = (gates.get("discovery_hits") or {}).get("value") or 0
    return (hits - ((gates.get(family) or {}).get("value") or 0)) / hits if hits else 0.0


def unrealized(mine: Mapping[str, dict], theirs: list[dict], sigma: float, family: str | None, slack: float) -> list[str]:
    """How a branch that keeps the incumbent's plan fails, in its own runs (``mine``, by name), to reproduce the
    incumbent's discovery (``theirs``, every trajectory it has): a mean discovery share lower by more than one standard
    error (``sigma`` the pooled run-to-run spread of discovery), or, where discovery counts chained families, a run that
    lost a share of its hits to linked families larger than any of the incumbent's by more than ``slack``."""
    if not mine or not theirs:
        return []
    reasons = []
    d_mine = statistics.fmean(r["discovery"] for r in mine.values())
    d_theirs = statistics.fmean(r["discovery"] for r in theirs)
    se = sigma * math.sqrt(1 / len(mine) + 1 / len(theirs))
    if d_mine < d_theirs - se:
        reasons.append(f"its runs' discovery averages {d_mine:.3f}, below the incumbent's {d_theirs:.3f} by more than one "
                       f"standard error ({se:.3f}), though it keeps the incumbent's plan")
    if family:
        worst = max(linked(r, family) for r in theirs)
        reasons += [f"{name} lost {linked(r, family):.0%} of its hits to linked families, against at most {worst:.0%} in the "
                    f"incumbent's runs, though it keeps the incumbent's plan"
                    for name, r in mine.items() if linked(r, family) > worst + slack]
    return reasons
