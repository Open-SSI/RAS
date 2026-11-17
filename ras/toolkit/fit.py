#!/usr/bin/env python3
"""Fit a campaign predictor from measured data and write final_model.js.

  python3 fit.py --train train.csv --out /app/final_model.js
  python3 fit.py --train train.csv --out /app/final_model.js --config config.json

train.csv columns:
  input    the exact string predict() receives (a catalog's design, composition or condition)
  target   the measured objective, larger is better (negate a lower-is-better measurement)
  group    optional: with group_normalize the target is standardized within each group,
           so scores are comparable across groups (libraries, substrates)
  region   optional number, for --protocol shift: train on the lower regions, validate on the highest

config.json, next to this script unless --config names another, holds the same options as the
flags ("featurizer", "protocol", "group_normalize"); flags override it. Its "ranking" option is the
campaign's ranking metric (ranking.py), by which candidate learners are compared; without it they
are compared by top-tail NDCG@20. An unknown option, or a
protocol the data cannot support, is an error rather than a silent fallback. Every candidate learner is validated on the measured data, the best
whose model fits the size limit is refitted on all of it, and the exported JavaScript is
checked against the Python model with deno. The report, with every candidate's
validation, goes next to the model as <out>.report.json.
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
from modeling import FEATURIZERS, export, select  # noqa: E402

MAX_BYTES = 1_000_000  # the campaigns accept a delivered model up to 1 MiB
OPTIONS = {"featurizer": "composition", "protocol": "random", "group_normalize": False, "ranking": None}
CONFIG = Path(__file__).resolve().with_name("config.json")


def load(path: Path, group_normalize: bool):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"input", "target"} <= set(rows[0]):
        raise SystemExit(f"{path} needs columns input and target")
    inputs = [row["input"] for row in rows]
    target = np.array([float(row["target"]) for row in rows])
    if group_normalize:
        if "group" not in rows[0]:
            raise SystemExit("group_normalize needs a group column")
        groups = np.array([row["group"] for row in rows])
        for g in set(groups):
            mask = groups == g
            target[mask] = (target[mask] - target[mask].mean()) / (target[mask].std() or 1.0)
    regions = np.array([float(row["region"]) for row in rows]) if "region" in rows[0] else None
    return inputs, target, regions


def ranking_scorer(path: Path, metric: dict | None):
    """score(prediction, row indices) by the campaign's ranking metric on the raw rows, or None."""
    if not metric:
        return None
    import ranking
    with path.open(newline="") as handle:
        rows = [{"target": float(row["target"]), "group": row.get("group", "")} for row in csv.DictReader(handle)]
    return lambda prediction, index: ranking.score(list(prediction), [rows[i] for i in index], metric)


def check(source: str, inputs: list[str], expected: np.ndarray) -> float:
    """Largest gap between the JavaScript and Python predictions, run with deno."""
    deno = shutil.which("deno")
    if deno is None:
        raise SystemExit("deno is needed to check the exported model")
    script = source + f"\nconsole.log(JSON.stringify({json.dumps(inputs)}.map(predict)));\n"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(script)
    out = subprocess.run([deno, "run", "--quiet", "--no-prompt", handle.name], capture_output=True, text=True)
    Path(handle.name).unlink()
    if out.returncode:
        raise SystemExit(f"the exported model failed in deno:\n{out.stderr[-2000:]}")
    return float(np.max(np.abs(np.array(json.loads(out.stdout)) - expected)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--featurizer", choices=sorted(FEATURIZERS), default=None)
    parser.add_argument("--protocol", choices=("random", "shift"), default=None)
    parser.add_argument("--group-normalize", action="store_true", default=None)
    args = parser.parse_args(argv)
    options = dict(OPTIONS)
    config = args.config or (CONFIG if CONFIG.is_file() else None)
    if config:
        options |= json.loads(config.read_text())
    options |= {key: value for key, value in (("featurizer", args.featurizer), ("protocol", args.protocol),
                                              ("group_normalize", args.group_normalize)) if value is not None}

    unknown = sorted(set(options) - set(OPTIONS))
    if unknown or options["featurizer"] not in FEATURIZERS or options["protocol"] not in ("random", "shift"):
        raise SystemExit(f"unsupported options: {unknown or options}")
    inputs, target, regions = load(args.train, options["group_normalize"])
    if options["protocol"] == "shift" and (regions is None or len(set(regions)) < 2):
        raise SystemExit("unsupported: shift validation needs a region column with at least two regions")
    featurizer = FEATURIZERS[options["featurizer"]]().fit(inputs)
    x = featurizer.transform(inputs)
    ranked = select(x, target, options["protocol"], regions, ranking_scorer(args.train, options["ranking"]))
    for learner, validation in ranked:
        learner.fit(x, target)
        source = export(featurizer, learner, f"{learner.label}, chosen by {options['protocol']} validation "
                                             f"on {len(target)} designs")
        if len(source.encode()) <= MAX_BYTES:
            break
    else:
        raise SystemExit("no candidate model fits the size limit")
    gap = check(source, inputs[:200], learner.predict(x[:200]))
    if gap > 1e-6 * (1 + float(np.abs(target).max())):
        raise SystemExit(f"the exported model disagrees with the fitted one by {gap:g}")
    args.out.write_text(source)
    summary = {"chosen": learner.label, "options": options, "designs": len(target), "validation": validation,
               "candidates": [row for _, row in ranked]}
    Path(f"{args.out}.report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{learner.label}: ranking {validation['ranking']:.3f}, Spearman {validation['spearman']:.3f} "
          f"({options['protocol']} validation, {len(target)} designs); wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
