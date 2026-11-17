#!/usr/bin/env python3
"""Cross-validate a harness's modeling procedure, and check that it produced the delivered model.

Runs in the sandbox, with no network, in a working directory holding:

  procedure/     the harness's scripts/model/, whose fit.py takes --train CSV --out JS
  train.csv      the run's rows, exactly as the harness's train.py builds them
  delivered.js   the model the run delivered (absent if it delivered none)
  probe.json     inputs on which the refitted and the delivered models are compared
  ranking.json   the campaign's ranking metric, as its instruction defines the learning requirement
  ranking.py     that metric's implementation

  python3 procedure_cv.py --out report.json

The procedure is refitted from a fresh copy on each fold's training rows, and its
exported JavaScript is scored on the held-out rows: by the campaign's ranking metric, and by
Spearman with targets standardized within each group when there is a group column. Five folds
are drawn three times and the scores averaged,
which steadies the estimate. The procedure is then refitted on every row, and the
delivered model is verified only if both predict the same on the probe inputs. A failure
anywhere is reported as such; nothing is substituted for it.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ranking  # noqa: E402

FOLDS = 5
REPEATS = 3
TIMEOUT = 900


class Failure(Exception):
    pass


def fit(procedure: Path, rows: list[dict], out: Path) -> None:
    """Run the procedure, from a fresh copy, on ``rows``; write its model to ``out``."""
    with tempfile.TemporaryDirectory() as work:
        copy = Path(work) / "procedure"
        shutil.copytree(procedure, copy)
        train = Path(work) / "train.csv"
        with train.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        done = subprocess.run([sys.executable, "fit.py", "--train", str(train), "--out", str(out)],
                              cwd=copy, capture_output=True, text=True, timeout=TIMEOUT)
    if done.returncode or not out.is_file():
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
        raise Failure(f"fit.py failed: {tail[0][:300]}")


def predict(model: Path, inputs: list[str]) -> np.ndarray:
    script = model.with_suffix(".run.js")
    script.write_text(model.read_text() + f"\nconsole.log(JSON.stringify({json.dumps(inputs)}.map(predict)));\n")
    done = subprocess.run(["deno", "run", "--quiet", "--no-prompt", str(script)],
                          capture_output=True, text=True, timeout=TIMEOUT)
    if done.returncode:
        raise Failure(f"{model.name} failed to run: {done.stderr.strip()[-300:]}")
    values = np.array(json.loads(done.stdout), float)
    if len(values) != len(inputs) or not np.all(np.isfinite(values)):
        raise Failure(f"{model.name} returned missing or non-finite predictions")
    return values


def standardized(rows: list[dict]) -> np.ndarray:
    y = np.array([float(row["target"]) for row in rows])
    if "group" in rows[0]:
        groups = np.array([row["group"] for row in rows])
        for g in set(groups):
            mask = groups == g
            y[mask] = (y[mask] - y[mask].mean()) / (y[mask].std() or 1.0)
    return y


def spec(directory: Path | None = None) -> dict:
    """The campaign's ranking metric, from ranking.json next to this script or in ``directory``."""
    for place in (directory, Path(__file__).resolve().parent, Path.cwd()):
        if place is not None and (place / "ranking.json").is_file():
            return json.loads((place / "ranking.json").read_text())
    raise Failure("no ranking.json: the campaign's ranking metric is missing")


def rank_score(prediction: np.ndarray, rows: list[dict], metric: dict) -> float:
    """The campaign's ranking metric of ``prediction`` on ``rows`` (each with target and maybe group)."""
    return ranking.score([float(p) for p in prediction], rows, metric)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = (np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float) for v in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() and rb.std() else 0.0


def cross_validate(procedure: Path, rows: list[dict], work: Path, metric: dict | None = None) -> dict:
    """The campaign's ranking metric and Spearman of the procedure's out-of-fold predictions, averaged over the repeats."""
    metric = metric or spec(work)
    y = standardized(rows)
    scores = []
    for repeat in range(REPEATS):
        folds = np.array_split(np.random.default_rng(repeat).permutation(len(rows)), FOLDS)
        out_of_fold = np.empty(len(rows))
        for i, test in enumerate(folds):
            held = set(test.tolist())
            model = work / f"fold-{repeat}-{i}.js"
            fit(procedure, [row for j, row in enumerate(rows) if j not in held], model)
            out_of_fold[test] = predict(model, [rows[j]["input"] for j in test])
        scores.append((rank_score(out_of_fold, rows, metric), spearman(out_of_fold, y)))
    return {"ranking": float(np.mean([s[0] for s in scores])), "spearman": float(np.mean([s[1] for s in scores])),
            "repeats": [round(s[0], 4) for s in scores], "folds": FOLDS, "rows": len(rows),
            "metric": ranking.describe(metric)}


def validate(work: Path) -> dict:
    procedure = work / "procedure"
    if not (procedure / "fit.py").is_file():
        raise Failure("the harness has no scripts/model/fit.py")
    with (work / "train.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    report = {"cv": cross_validate(procedure, rows, work)}
    full = work / "full.js"
    fit(procedure, rows, full)
    delivered = work / "delivered.js"
    if not delivered.is_file():
        return report | {"delivered": "missing"}
    probe = json.loads((work / "probe.json").read_text())
    ours, theirs = predict(full, probe), predict(delivered, probe)
    gap = float(np.max(np.abs(ours - theirs)))
    tolerance = 1e-9 * (1 + float(np.max(np.abs(ours))))
    return report | {"delivered": "verified" if gap <= tolerance else "mismatch", "max_difference": gap}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate(Path.cwd())
    except Failure as failure:
        report = {"error": str(failure)}
    except subprocess.TimeoutExpired:
        report = {"error": f"the procedure ran longer than {TIMEOUT} s"}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
