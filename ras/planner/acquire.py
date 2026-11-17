#!/usr/bin/env python3
"""Model-guided experiment planning: propose the next round's batch from this harness's own model.

  python3 /app/harness/scripts/plan/acquire.py --out /app/round_2.csv

Each round it rebuilds train.csv from the public seed and every measurement so far (the procedure's own train.py),
refits the procedure (fit.py) on all of it and on bootstrap resamples, and scores every design still orderable:
the full fit gives each design's predicted value, the resamples its uncertainty. Scores are standardized within the
campaign's groups where it ranks within groups. The batch then takes the designs with the highest predicted value plus
`beta` times their uncertainty (exploitation), and an `explore_share` of designs with the highest uncertainty among
those predicted better than the median (exploration), under the campaign's own counting rules: an equal share of the
batch for every group where `allocate` is set, and, where hits are counted in chained families, no design within the
family distance of a measured hit or of a design already in the batch, so that every new hit can found a family.

It proposes a batch only with evidence: the resamples' out-of-bag predictions must rank this run's own measured designs
(within groups) with a Spearman of at least `min_evidence`. Otherwise, as in round 1, it writes nothing, exits with
status 3, and the round is the agent's to choose. Every proposal is also kept in /app/plan/ for the run's record.
Configuration: plan.json beside this script; the procedure: ../model/ (fit.py, train.py, spec.json).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "model"
LOG = Path("/app/plan")


def lab(path: str) -> dict:
    with urllib.request.urlopen(os.environ["ASB_LAB_URL"] + path, timeout=60) as response:
        return json.load(response)


def measured() -> list[dict]:
    rows = []
    for job in sorted(lab("/v1/experiments")["jobs"], key=lambda job: job["job_id"]):
        if job["status"] == "completed":
            rows += lab(f"/v1/experiments/{job['job_id']}")["job"]["results"]
    return rows


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    sx, sy = statistics.pstdev(rx), statistics.pstdev(ry)
    if not sx or not sy:
        return None
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (len(rx) * sx * sy)


def fit(rows: list[dict], work: Path, name: str) -> Path:
    table, out = work / f"{name}.csv", work / f"{name}.js"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    copy = work / f"procedure-{name}"
    shutil.copytree(MODEL, copy, ignore=shutil.ignore_patterns("__pycache__"))
    done = subprocess.run([sys.executable, "fit.py", "--train", str(table), "--out", str(out)], cwd=copy,
                          capture_output=True, text=True)
    if done.returncode or not out.is_file():
        raise SystemExit(f"fit.py failed: {(done.stdout + done.stderr)[-1500:]}")
    return out


def score(model: Path, inputs: list[str], work: Path) -> list[float]:
    source, target = work / "inputs.json", work / "scores.json"
    source.write_text(json.dumps(inputs))
    done = subprocess.run(["deno", "run", "--allow-read", "--allow-write", str(HERE / "score.js"), str(model),
                           str(source), str(target)], capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"scoring failed: {done.stderr[-1500:]}")
    return json.loads(target.read_text())


def substitutions(sequence: str, reference: str) -> frozenset[str]:
    return frozenset(f"{a}{i + 1}{b}" for i, (a, b) in enumerate(zip(reference, sequence)) if a != b)


def reference_sequence(path: str) -> str:
    return "".join(line.strip() for line in Path(path).read_text().splitlines() if not line.startswith(">"))


def standardize(values: list[float], groups: list[str]) -> list[float]:
    by: dict[str, list[float]] = {}
    for v, g in zip(values, groups):
        by.setdefault(g, []).append(v)
    stats = {g: (statistics.fmean(vs), statistics.pstdev(vs) or 1.0) for g, vs in by.items()}
    return [(v - stats[g][0]) / stats[g][1] for v, g in zip(values, groups)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads((HERE / "plan.json").read_text())
    spec = json.loads((MODEL / "spec.json").read_text())
    status = lab("/v1/status")
    stage = status.get("next_stage")
    if stage is None:
        print("every round is complete")
        return 2
    card = lab("/v1/campaign")
    size = next(s["batch_size"] for s in card["stages"] if s["stage"] == stage)
    with open(spec["catalog"], newline="") as handle:
        catalog = {row[spec["id"]]: row for row in csv.DictReader(handle)}
    group_of = lambda row: "|".join(str(row[c]) for c in spec.get("group") or []) or "all"
    results = measured()
    done_ids = {str(r[spec["id"]]) for r in results}
    candidates = [key for key in catalog if key not in done_ids]
    rng = random.Random(int(config.get("seed", 0)) * 1000 + stage)
    with tempfile.TemporaryDirectory(prefix="plan-") as name:
        work = Path(name)
        table = work / "train.csv"
        build = subprocess.run([sys.executable, str(MODEL / "train.py"), "--out", str(table)], capture_output=True, text=True)
        if build.returncode:
            raise SystemExit(f"train.py failed: {build.stderr[-1500:]}")
        with table.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        own = rows[len(rows) - len(results):]  # train.py writes the seed first, then this run's measurements in order
        seed_rows = rows[: len(rows) - len(results)]
        evidence = None
        oob: list[list[float]] = [[] for _ in own]
        models = []
        for b in range(int(config.get("bootstrap", 8))):
            picks = [rng.randrange(len(own)) for _ in own] if own else []
            models.append((fit(seed_rows + [own[i] for i in picks], work, f"boot{b}"), set(picks)))
        inputs = [catalog[key][spec["input"]] for key in candidates]
        own_inputs = [row["input"] for row in own]
        spread = []
        for model, picked in models:
            scores = score(model, inputs + own_inputs, work)
            spread.append(scores[: len(inputs)])
            for i, s in enumerate(scores[len(inputs):]):
                if i not in picked:
                    oob[i].append(s)
        if len(own) >= size:
            pairs = [(statistics.fmean(p), float(row["target"]), row.get("group", "all")) for p, row in zip(oob, own) if p]
            by: dict[str, list[tuple[float, float]]] = {}
            for p, t, g in pairs:
                by.setdefault(g, []).append((p, t))
            weighted = [(len(v), spearman([a for a, _ in v], [b for _, b in v])) for v in by.values()]
            weighted = [(n, r) for n, r in weighted if r is not None]
            evidence = sum(n * r for n, r in weighted) / sum(n for n, _ in weighted) if weighted else None
        if evidence is None or evidence < float(config.get("min_evidence", 0.25)):
            print(f"insufficient evidence for round {stage} (out-of-bag Spearman on this run's measurements: "
                  f"{'none yet' if evidence is None else f'{evidence:.2f}'}): choose this round yourself")
            return 3
        mean = score(fit(rows, work, "full"), inputs, work)
    sd = [statistics.pstdev(column) for column in zip(*spread)]
    groups = [group_of(catalog[key]) for key in candidates]
    z_mean = standardize(mean, groups)
    scale = {}
    for m, g in zip(mean, groups):
        scale.setdefault(g, []).append(m)
    z_sd = [s / (statistics.pstdev(scale[g]) or 1.0) for s, g in zip(sd, groups)]
    beta, share = float(config.get("beta", 1.0)), float(config.get("explore_share", 0.25))
    ucb = [m + beta * s for m, s in zip(z_mean, z_sd)]
    median = statistics.median(z_mean)
    linkage = config.get("linkage")
    if linkage:
        reference = reference_sequence(linkage["reference"])
        subs = {key: substitutions(catalog[key][spec["input"]], reference) for key in candidates}
        hits = [substitutions(catalog[str(r[spec["id"]])][spec["input"]], reference) for r in results
                if float(r[spec["objective"]["field"]]) >= float(linkage["hit_at_least"])]
        distance = int(linkage["distance"])
    allocate = bool(config.get("allocate")) and len(set(groups)) > 1
    quota = {g: size // len(set(groups)) + (1 if i < size % len(set(groups)) else 0)
             for i, g in enumerate(sorted(set(groups)))} if allocate else None
    chosen: list[int] = []
    taken: dict[str, int] = {}

    def admissible(i: int) -> bool:
        if quota is not None and taken.get(groups[i], 0) >= quota[groups[i]]:
            return False
        if linkage:
            mine = subs[candidates[i]]
            if any(len(mine ^ other) <= distance for other in hits):
                return False
            if any(len(mine ^ subs[candidates[j]]) <= distance for j in chosen):
                return False
        return True

    def take(order: list[int], count: int) -> None:
        for i in order:
            if len(chosen) >= size or count <= 0:
                return
            if i not in chosen and admissible(i):
                chosen.append(i)
                taken[groups[i]] = taken.get(groups[i], 0) + 1
                count -= 1

    explore = max(0, min(size, round(share * size)))
    take(sorted(range(len(candidates)), key=lambda i: -ucb[i]), size - explore)
    take(sorted((i for i in range(len(candidates)) if z_mean[i] >= median), key=lambda i: -z_sd[i]), explore)
    take(sorted(range(len(candidates)), key=lambda i: -ucb[i]), size - len(chosen))  # fill any shortfall
    if len(chosen) < size:
        print(f"only {len(chosen)} admissible designs for a batch of {size}: choose this round yourself")
        return 3
    batch = [candidates[i] for i in chosen]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path in (args.out, LOG / f"round_{stage}.csv"):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as handle:
                handle.write(f"{spec['id']}\n" + "".join(f"{key}\n" for key in batch))
        except OSError:
            pass
    print(f"round {stage}: proposed {len(batch)} designs to {args.out} (out-of-bag Spearman {evidence:.2f}; "
          f"{size - explore} by predicted value plus uncertainty, {explore} by uncertainty"
          + (f"; {len(set(groups))} groups, equal shares" if allocate else "")
          + (f"; none within {distance} substitutions of a measured hit or of each other" if linkage else "")
          + "). Order exactly this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
