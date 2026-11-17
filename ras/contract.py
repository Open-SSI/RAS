"""The verifier's contract for a delivered model, checked on the lab side before anything relies on it.

The verifier reads ``final_model.js`` (at most ``max_bytes``), scores every evaluation design twice,
each time in a fresh Deno worker, and fails the run if the source is too large, if any score is
missing, non-finite or differs between the two passes, or if scoring takes longer than its
``timeout_s``. ``check`` does the same with the benchmark's own runner, on the campaign's public
evaluation list where it publishes one, else on as many catalog designs as the verifier scores, so
a model that would fail the verifier counts as no model in selection and its run's improvers see why.

Only one check runs at a time, so its timing is not inflated by the others.
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .config import Campaign

Runner = Callable[[str, list[str], float], list[float]]  # (source, inputs, timeout) -> scores, or raises
_ONE_AT_A_TIME = threading.Lock()


def inputs(campaign: Campaign) -> list[str]:
    """The designs the check scores: the published evaluation list, or catalog designs, as many as the verifier scores."""
    spec = campaign.contract
    if spec.inputs is not None:
        with spec.inputs.open(newline="") as handle:
            return [row[spec.column] for row in csv.DictReader(handle)]
    with campaign.catalog.open(newline="") as handle:
        designs = sorted({row[campaign.input] for row in csv.DictReader(handle)})
    stride = max(1, len(designs) // spec.designs)
    picked = designs[::stride][: spec.designs]
    return (picked * (spec.designs // len(picked) + 1))[: spec.designs]


def check(model: Path, campaign: Campaign, run: Runner) -> dict:
    """The verifier's contract on ``model``: {"ok": bool, ...} with what was measured, or why it fails."""
    spec = campaign.contract
    if spec is None:
        return {"ok": True, "checked": False}
    if not model.is_file():
        return {"ok": False, "error": "no final_model.js was delivered"}
    size = model.stat().st_size
    report = {"bytes": size, "max_bytes": spec.max_bytes, "designs": spec.designs, "timeout_s": spec.timeout_s}
    if size > spec.max_bytes:
        return report | {"ok": False, "error": f"the model is {size} bytes; the verifier reads at most {spec.max_bytes}"}
    designs = inputs(campaign)
    with _ONE_AT_A_TIME:
        started = time.monotonic()
        try:
            run(model.read_text(), designs, spec.timeout_s)
            error = None
        except Exception as failure:  # noqa: BLE001 - whatever stops the runner stops the verifier
            error = _reason(failure, spec.timeout_s)
        seconds = round(time.monotonic() - started, 1)
    return report | {"seconds": seconds, "ok": error is None} | ({"error": error} if error else {})


def benchmark_runner(runtime: Path) -> Runner:
    """The benchmark's sandboxed Deno runner, as the verifier calls it, in its unprivileged local mode."""
    def run(source: str, designs: list[str], timeout: float) -> list[float]:
        os.environ.setdefault("ASB_VERIFY_LOCAL", "1")  # read once, when asb_verify is imported
        if str(runtime) not in sys.path:
            sys.path.insert(0, str(runtime))
        from asb_verify.js_sandbox import run_model
        return run_model(source, designs, runtime / "sandbox" / "js_runner.ts", int(timeout))
    return run


def _reason(failure: Exception, timeout: float) -> str:
    if type(failure).__name__ == "TimeoutExpired":
        return f"scoring the evaluation-sized set took longer than the verifier's {timeout:.0f} s"
    return f"the verifier's runner rejects it: {str(failure)[:200] or type(failure).__name__}"
