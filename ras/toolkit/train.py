#!/usr/bin/env python3
"""Build train.csv for the harness's modeling procedure: the public seed, then every
measured design in the order the lab measured them.

  python3 train.py --out /app/model/train.csv

The columns are input (the string predict() receives), target (the campaign objective,
larger is better) and, where the campaign compares designs within groups, group.
spec.json, next to this script, names the catalog, seed files, columns and objective.
The pipeline builds the same rows from the lab's record, so a procedure fitted on this
file is the procedure the pipeline validates; do not edit this script or spec.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request
from pathlib import Path

SPEC = Path(__file__).resolve().with_name("spec.json")


def utility(row: dict, objective: dict) -> float:
    """The objective with larger always better, e.g. lead response, 0 when zinc response exceeds its limit."""
    limit = objective.get("zero_above")
    value = 0.0 if limit and float(row[limit[0]]) > float(limit[1]) else float(row[objective["field"]])
    return value if objective.get("higher_is_better", True) else -value


def lab(path: str) -> dict:
    # The campaign image sets ASB_LAB_URL to the lab sidecar.
    with urllib.request.urlopen(os.environ["ASB_LAB_URL"] + path, timeout=60) as response:
        return json.load(response)


def rows(spec: dict) -> list[dict]:
    group = spec.get("group") or []

    def row(text: str, values: dict, meta: dict) -> dict:
        out = {"input": text, "target": utility(values, spec["objective"])}
        if group:
            out["group"] = "|".join(str(meta[column]) for column in group)
        return out

    out = []
    for seed in spec.get("seed", []):
        with open(seed["file"], newline="") as handle:
            out += [row(r[seed["input"]], r, r) for r in csv.DictReader(handle)]
    with open(spec["catalog"], newline="") as handle:
        catalog = {r[spec["id"]]: r for r in csv.DictReader(handle)}
    jobs = sorted(lab("/v1/experiments")["jobs"], key=lambda job: job["job_id"])
    for job in jobs:
        if job["status"] != "completed":
            continue
        for result in lab(f"/v1/experiments/{job['job_id']}")["job"]["results"]:
            meta = catalog[str(result[spec["id"]])]
            out.append(row(meta[spec["input"]], result, meta))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=SPEC)
    args = parser.parse_args(argv)
    table = rows(json.loads(args.spec.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"wrote {len(table)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
