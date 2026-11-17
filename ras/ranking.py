"""The learning requirement's ranking metric, as each campaign's instruction defines it.

A campaign's ``ranking`` spec (config/campaigns.toml) restates its instruction: NDCG at ``k``, with
relevance either the objective above a fixed ``threshold`` (``relevance = "above"``), or a score
normalized within each group (``relevance = "group_cutoff"``): ``u = (y - cutoff) / (top - cutoff)``,
where ``cutoff`` is the group's value with ``cutoff_share`` of its designs at or above it and ``top``
its best value, or the value with ``top_share`` at or above it; ``u`` is capped at 1 when ``cap``.
The instruction fixes those shares against each group's full design set; on a sample, such as a
run's own rows, the same shares of the sample estimate them.

Targets are utilities, the objective with larger always better (feedback.utility). Ties keep the
input order, as "equal predictions retain the evaluation order". Standard library only: this file
is copied next to the procedure bench, the procedure's cross-validation and the modeling toolkit.
"""

from __future__ import annotations

import math
from collections import defaultdict


def relevance(rows: list[dict], spec: dict) -> list[float]:
    """Each row's relevance; a row is {"target": utility, "group": optional group label}."""
    targets = [float(row["target"]) for row in rows]
    if spec["relevance"] == "above":
        return [max(t - float(spec["threshold"]), 0.0) for t in targets]
    if spec["relevance"] != "group_cutoff":
        raise ValueError(f"unknown relevance {spec['relevance']!r}")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[str(row.get("group", ""))].append(i)
    out = [0.0] * len(rows)
    for members in groups.values():
        values = [targets[i] for i in members]
        cutoff = at_share(values, float(spec["cutoff_share"]))
        top = at_share(values, float(spec["top_share"])) if spec.get("top_share") else max(values)
        span = top - cutoff
        for i in members:
            u = (targets[i] - cutoff) / span if span > 0 else float(targets[i] > cutoff)
            out[i] = max(min(u, 1.0) if spec.get("cap") else u, 0.0)
    return out


def at_share(values: list[float], share: float) -> float:
    """The value with ``share`` of ``values`` at or above it (at least the best one)."""
    ordered = sorted(values, reverse=True)
    return ordered[min(len(ordered), max(1, round(share * len(ordered)))) - 1]


def ndcg(prediction: list[float], relevant: list[float], k: int) -> float:
    order = sorted(range(len(relevant)), key=lambda i: (-prediction[i], i))[:k]
    gain = sum(relevant[i] / math.log2(rank + 2) for rank, i in enumerate(order))
    best = sum(r / math.log2(rank + 2) for rank, r in enumerate(sorted(relevant, reverse=True)[:k]))
    return gain / best if best else 0.0


def score(prediction: list[float], rows: list[dict], spec: dict) -> float:
    """The campaign's NDCG of ``prediction`` on ``rows``."""
    return ndcg([float(p) for p in prediction], relevance(rows, spec), int(spec["k"]))


def describe(spec: dict) -> str:
    """The metric in one line, for improvers' feedback."""
    if spec["relevance"] == "above":
        rule = f"relevance max(target - {spec['threshold']}, 0)"
    else:
        top = f"the value with {spec['top_share']:.4f} of the group at or above it" if spec.get("top_share") else "the group's best"
        rule = (f"relevance max(u, 0){' capped at 1' if spec.get('cap') else ''}, u = (target - cutoff) / (top - cutoff) "
                f"within each group, cutoff the value with {spec['cutoff_share']:.4f} of the group at or above it, top {top}")
    return f"NDCG@{spec['k']}, {rule}"
