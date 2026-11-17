#!/usr/bin/env python3
"""Compare modeling procedures on the lab evidence, before any run uses them.

  python3 library/bench/bench.py harness/scripts/model
  python3 library/bench/bench.py /tmp/draft/scripts/model --against base         # vs the stock agents' own models
  python3 library/bench/bench.py /tmp/draft/scripts/model --compare harness/scripts/model
  python3 library/bench/bench.py /tmp/draft/scripts/model --runs base-1 k1-2
  python3 library/bench/bench.py harness/scripts/model --pooled
  python3 library/bench/bench.py harness/scripts/model --prior --no-cross

Each evidence run holds train.csv, the rows its model was fitted on (the public seed and every
design the run measured), measured.csv, the measured designs alone, and final_model.js, the model
its agent delivered. Every score here is the campaign's own ranking metric, as its instruction
defines the learning requirement (ranking.json beside this script).

The cross-run score is the learning estimate selection uses. For each run (all of them, or
--runs), the procedure is fitted on that run's train.csv, exactly as its agent would, and the
exported model is scored with deno on the designs the other evidence runs measured and that run
did not. The score is the mean over the runs.

--against LABEL compares the procedure with how a harness's agents actually built their models:
for each run of that harness (base-1, base-2, ... for --against base), the model the run delivered
and the procedure fitted on the same run's train.csv are scored on the same test designs, and the
report gives both means and their paired difference with its standard error over the runs.
--compare DIR does the same against another procedure. A replacement for the current modeling is
supported only where it is better on these paired splits by more than their standard error.

--pooled adds 5-fold cross-validation, drawn 3 times, on the distinct designs of all runs
together (at most --max-rows), as each run's feedback.json reports it for its own rows.

--prior checks that the procedure learns only from its train.csv: it is fitted on the pooled
rows with their targets shuffled, four times, and its models' mean Spearman with the real
targets on those rows must stay near 0; one shuffle alone varies by about 0.1. A harness whose
procedure fails this check is rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from procedure_cv import Failure, cross_validate, fit, predict, rank_score, spearman, spec, standardized  # noqa: E402

EVIDENCE = Path("evidence")
MIN_TEST = 5


def read(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pooled(evidence: Path, max_rows: int) -> list[dict]:
    rows: dict[str, dict] = {}
    for train in sorted(evidence.glob("*/train.csv")):
        for row in read(train):
            rows.setdefault(row["input"], row)
    table = list(rows.values())
    if len(table) > max_rows:
        keep = np.sort(np.random.default_rng(0).choice(len(table), max_rows, replace=False))
        table = [table[i] for i in keep]
    return table


def splits(evidence: Path, runs: list[str]) -> dict[str, list[dict]]:
    """For each run, the designs the other runs measured and it did not."""
    measured = {path.parent.name: read(path) for path in sorted(evidence.glob("*/measured.csv"))}
    out = {}
    for run in runs:
        own = {row["input"] for row in measured.get(run, [])}
        test: dict[str, dict] = {}
        for other, rows in measured.items():
            if other != run:
                for row in rows:
                    if row["input"] not in own:
                        test.setdefault(row["input"], row)
        if len(test) >= MIN_TEST:
            out[run] = list(test.values())
    if not out:
        raise Failure("too few designs measured by other runs to score on")
    return out


def cross(procedure: Path, evidence: Path, runs: list[str], metric: dict) -> dict:
    """The procedure fitted on each run's rows, scored on what the other runs measured and it did not."""
    scores = {}
    with tempfile.TemporaryDirectory() as work:
        for run, table in splits(evidence, runs).items():
            model = Path(work) / f"{run}.js"
            fit(procedure, read(evidence / run / "train.csv"), model)
            scores[run] = round(rank_score(predict(model, [row["input"] for row in table]), table, metric), 4)
    return {"ranking": round(float(np.mean(list(scores.values()))), 4), "runs": scores}


def paired(procedure: Path, evidence: Path, runs: list[str], metric: dict, rival) -> dict:
    """The procedure and ``rival(run, work)`` (a model path, or None if that run has none) on the same splits."""
    rows = {}
    with tempfile.TemporaryDirectory() as work:
        for run, table in splits(evidence, runs).items():
            other = rival(run, Path(work))
            if other is None:
                continue
            inputs = [row["input"] for row in table]
            model = Path(work) / f"{run}.js"
            fit(procedure, read(evidence / run / "train.csv"), model)
            mine = rank_score(predict(model, inputs), table, metric)
            try:
                theirs = rank_score(predict(other, inputs), table, metric)
            except Failure:
                theirs = 0.0  # a model that does not run scores 0, as the verifier scores it
            rows[run] = {"procedure": round(mine, 4), "rival": round(theirs, 4)}
    if not rows:
        raise Failure("no run to compare on: none of those runs has a rival model and enough test designs")
    differences = [r["procedure"] - r["rival"] for r in rows.values()]
    error = float(np.std(differences, ddof=1) / math.sqrt(len(differences))) if len(differences) > 1 else None
    gain = float(np.mean(differences))
    verdict = ("inconclusive" if error is None else "better" if gain > error else "worse" if gain < -error else "even")
    return {"procedure": round(float(np.mean([r["procedure"] for r in rows.values()])), 4),
            "rival": round(float(np.mean([r["rival"] for r in rows.values()])), 4),
            "gain": round(gain, 4), "error": None if error is None else round(error, 4), "verdict": verdict,
            "runs": rows}


def score(procedure: Path, rows: list[dict], metric: dict) -> dict:
    """The pipeline's cross-validation of a procedure on ``rows``."""
    with tempfile.TemporaryDirectory() as work:
        cv = cross_validate(procedure, rows, Path(work), metric)
    return {"ranking": round(cv["ranking"], 4), "spearman": round(cv["spearman"], 4), "rows": cv["rows"]}


SHUFFLES = 4


def prior(procedure: Path, rows: list[dict]) -> dict:
    """How well the procedure ranks ``rows`` after fitting them with their targets shuffled, on average."""
    truth, scores = standardized(rows), []
    with tempfile.TemporaryDirectory() as work:
        for seed in range(1, SHUFFLES + 1):
            shuffled = np.random.default_rng(seed).permutation([row["target"] for row in rows])
            model = Path(work) / f"shuffled-{seed}.js"
            fit(procedure, [row | {"target": str(target)} for row, target in zip(rows, shuffled)], model)
            scores.append(spearman(predict(model, [row["input"] for row in rows]), truth))
            if seed == 1:
                shutil.copyfile(model, Path.cwd() / "prior.js")  # a model of the procedure's size, for timing
    return {"spearman": round(float(np.mean(scores)), 4), "shuffles": [round(s, 4) for s in scores], "rows": len(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("procedure", type=Path, help="a scripts/model/ directory with fit.py")
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--runs", nargs="*", default=None, help="runs to fit on (default: every evidence run)")
    parser.add_argument("--against", default=None, help="compare with the models this harness label's runs delivered")
    parser.add_argument("--compare", type=Path, default=None, help="compare with another scripts/model/ directory")
    parser.add_argument("--pooled", action="store_true", help="add cross-validation on the pooled rows")
    parser.add_argument("--prior", action="store_true", help="check the procedure learns only from train.csv")
    parser.add_argument("--no-cross", action="store_true", help="skip the cross-run score")
    parser.add_argument("--max-rows", type=int, default=800)
    args = parser.parse_args(argv)
    procedure = args.procedure.resolve()
    if not (procedure / "fit.py").is_file():
        raise SystemExit(f"{args.procedure} has no fit.py")
    runs = args.runs if args.runs is not None else sorted(p.parent.name for p in args.evidence.glob("*/train.csv"))
    rows = pooled(args.evidence, args.max_rows)
    needs_metric = not args.no_cross or args.pooled or args.against or args.compare
    metric = spec() if needs_metric else {}
    targets = [] if args.no_cross else [("cross-run", lambda: cross(procedure, args.evidence, runs, metric))]
    if args.against:
        own = [run for run in runs if run.rsplit("-", 1)[0] == args.against or run == args.against]

        def delivered(run: str, work: Path) -> Path | None:
            source = args.evidence / run / "final_model.js"
            if not source.is_file():
                return None
            return Path(shutil.copyfile(source, work / f"{run}.delivered.js"))  # scored from a copy: evidence stays as it was
        targets += [(f"against {args.against}", lambda: paired(procedure, args.evidence, own, metric, delivered))]
    if args.compare:
        rival_dir = args.compare.resolve()

        def refit(run: str, work: Path) -> Path:
            out = work / f"{run}.rival.js"
            fit(rival_dir, read(args.evidence / run / "train.csv"), out)
            return out
        targets += [(f"compare {args.compare}", lambda: paired(procedure, args.evidence, runs, metric, refit))]
    targets += [("pooled", lambda: score(procedure, rows, metric))] if args.pooled else []
    targets += [("prior", lambda: prior(procedure, rows))] if args.prior else []
    report = {"metric": metric}
    for name, measure in targets:
        try:
            report[name] = measure()
        except Failure as failure:
            report[name] = {"error": str(failure)}
        result = report[name]
        if "error" in result:
            print(f"{name}: {result['error']}", flush=True)
        elif name == "cross-run":
            print(f"cross-run: {result['ranking']:.3f}, the campaign's ranking metric, mean over {len(result['runs'])} "
                  f"runs of the model fitted on one run and scored on what the others measured", flush=True)
        elif name.startswith(("against", "compare")):
            error = "n/a" if result["error"] is None else f"{result['error']:.3f}"
            print(f"{name}: procedure {result['procedure']:.3f} vs {result['rival']:.3f} on the same {len(result['runs'])} "
                  f"splits; paired gain {result['gain']:+.3f} ± {error}: {result['verdict']}", flush=True)
        elif name == "prior":
            print(f"prior: Spearman {result['spearman']:.3f} after fitting shuffled targets ({result['rows']} rows); "
                  f"near 0 means the procedure learns only from train.csv", flush=True)
        else:
            print(f"{name}: {result['ranking']:.3f} by the campaign's ranking metric, Spearman {result['spearman']:.3f} "
                  f"({result['rows']} rows)", flush=True)
    print(json.dumps({"procedure": str(args.procedure), "scores": report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
