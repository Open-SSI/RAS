#!/usr/bin/env python3
"""Fit the harness's modeling procedure on this run's measurements and write the model, once the last round is done.

  python3 /opt/ras-delivery/deliver_model.py --out /opt/ras-delivery/final_model.js

The harness runs this itself, as root, after the agent's run has returned (ras.agents), then delivers the model; the
agent never sees the procedure. It unpacks model.tar.gz beside it into a temporary directory, builds train.csv from the
public seed and the run's measurements (train.py), fits it (fit.py), and writes the model. It refuses while rounds
remain, so a run the agent did not finish keeps the agent's own deliverable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

PACKAGE = Path(__file__).resolve().with_name("model.tar.gz")


def status() -> dict:
    with urllib.request.urlopen(os.environ["ASB_LAB_URL"] + "/v1/status", timeout=60) as response:
        return json.load(response)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    progress = status()
    if progress["stages_completed"] < progress["stages_total"]:
        print(f"{progress['stages_completed']} of {progress['stages_total']} rounds are complete: the procedure builds "
              "the delivered model only after the last round.", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="model-") as work:
        with tarfile.open(PACKAGE) as package:
            try:
                package.extractall(work, filter="data")
            except TypeError:  # Python before 3.12
                package.extractall(work)
        model, train = Path(work) / "model", Path(work) / "train.csv"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for command in (["train.py", "--out", str(train)], ["fit.py", "--train", str(train), "--out", str(args.out.resolve())]):
            done = subprocess.run([sys.executable, *command], cwd=model, capture_output=True, text=True)
            if done.returncode:
                print(f"{command[0]} failed:\n{(done.stdout + done.stderr)[-2000:]}", file=sys.stderr)
                return 1
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
