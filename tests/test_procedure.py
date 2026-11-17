"""Harness-owned modeling: the train.csv builder, cross-validation of the exact procedure, and verification."""

import csv
import json
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from ras import feedback, procedure
from ras.config import Fit, Objective, SeedFile

pytestmark = pytest.mark.skipif(shutil.which("deno") is None, reason="needs deno")


@pytest.fixture
def alloys(tmp_path, campaign):
    """A composition campaign on two substrates, with a seed file and a ledger of 40 measurements."""
    data = tmp_path / "task" / "environment" / "data"
    rows = [(f"s{i:03d}", a, b, round(100 - a - b, 1), "AB"[i % 2])
            for i, (a, b) in enumerate((a, b) for a in range(0, 101, 10) for b in range(0, 101 - a, 10))]
    hardness = lambda a, b, c, sub: 0.05 * a - 0.02 * b + 0.01 * a * c / 100 + (1.0 if sub == "B" else 0.0)
    with (data / "catalog.csv").open("w") as handle:
        handle.write("sample_id,design,substrate\n")
        handle.writelines(f"{sid},{sub}|Co{a}Fe{b}Ni{c},{sub}\n" for sid, a, b, c, sub in rows)
    with (data / "seed.csv").open("w") as handle:
        handle.write("design,substrate,hardness\n")
        handle.writelines(f"{sub}|Co{a}Fe{b}Ni{c},{sub},{hardness(a, b, c, sub)}\n" for _, a, b, c, sub in rows[:6])
    alloy = replace(campaign, catalog=data / "catalog.csv", id="sample_id", input="design", objective=Objective("hardness"),
                    seed=(SeedFile(data / "seed.csv", "design"),), fit=Fit("composition", ("substrate",)))
    jobs = [{"job_id": f"job-{s:04d}", "stage": s, "status": "completed",
             "results": [{"sample_id": sid, "hardness": hardness(a, b, c, sub)} for sid, a, b, c, sub in rows[6 + 20 * (s - 1): 26 + 20 * (s - 1)]]}
            for s in (1, 2)]
    return alloy, {"jobs": jobs}


class Lab(BaseHTTPRequestHandler):
    """The two lab routes train.py reads."""

    ledger: dict = {}

    def do_GET(self):
        jobs = {job["job_id"]: job for job in self.ledger["jobs"]}
        body = ({"jobs": [{k: v for k, v in j.items() if k != "results"} for j in self.ledger["jobs"]]}
                if self.path == "/v1/experiments" else {"job": jobs[self.path.rsplit("/", 1)[1]]})
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_train_py_builds_exactly_the_pipeline_rows(tmp_path, alloys):
    alloy, ledger = alloys
    Lab.ledger = ledger
    server = HTTPServer(("127.0.0.1", 0), Lab)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    lib = procedure.library(alloy, tmp_path / "lib")
    spec = json.loads((lib / "spec.json").read_text())
    assert spec["catalog"] == "/app/data/catalog.csv" and spec["group"] == ["substrate"]
    local = spec | {"catalog": str(alloy.catalog), "seed": [{"file": str(alloy.seed[0].file), "input": "design"}]}
    (tmp_path / "local-spec.json").write_text(json.dumps(local))
    out = tmp_path / "train.csv"
    subprocess.run([sys.executable, str(lib / "train.py"), "--out", str(out), "--spec", str(tmp_path / "local-spec.json")],
                   check=True, capture_output=True, env={"ASB_LAB_URL": f"http://127.0.0.1:{server.server_port}"})
    server.shutdown()
    with out.open(newline="") as handle:
        built = [{k: (float(v) if k == "target" else v) for k, v in row.items()} for row in csv.DictReader(handle)]
    assert built == feedback.training_rows(ledger, alloy)


def deliver(tree: Path, rows: list[dict], out: Path) -> None:
    """What the agent does: run the harness's procedure on train.csv."""
    train = out.with_name("train.csv")
    with train.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    subprocess.run([sys.executable, "fit.py", "--train", str(train), "--out", str(out)],
                   cwd=tree / procedure.MODEL_DIR, check=True, capture_output=True)


def harness_with_procedure(tmp_path, alloy) -> Path:
    tree = tmp_path / "harness"
    procedure.library(alloy, tree / procedure.MODEL_DIR)
    return tree


def test_the_delivered_procedure_is_cross_validated_and_verified(tmp_path, alloys):
    alloy, ledger = alloys
    rows = feedback.training_rows(ledger, alloy)
    tree = harness_with_procedure(tmp_path, alloy)
    delivered = tmp_path / "final_model.js"
    deliver(tree, rows, delivered)
    probe = [r["design"] for r in feedback.catalog(alloy).values()]
    report = procedure.validate(tree, rows, delivered, probe, run=procedure.local, metric=alloy.ranking)
    assert report["delivered"] == "verified", report
    assert report["cv"]["rows"] == 46 and report["cv"]["spearman"] > 0.5


def test_a_model_the_procedure_did_not_produce_is_a_mismatch(tmp_path, alloys):
    alloy, ledger = alloys
    rows = feedback.training_rows(ledger, alloy)
    tree = harness_with_procedure(tmp_path, alloy)
    delivered = tmp_path / "final_model.js"
    delivered.write_text("function predict(design) { return design.length; }\n")
    report = procedure.validate(tree, rows, delivered, ["A|Co10Fe90", "B|Co50Ni50"], run=procedure.local, metric=alloy.ranking)
    assert report["delivered"] == "mismatch" and "cv" in report


def test_missing_and_failing_procedures_are_errors_not_substitutes(tmp_path, alloys):
    alloy, ledger = alloys
    rows = feedback.training_rows(ledger, alloy)
    assert "no modeling procedure" in procedure.validate(tmp_path / "empty", rows, tmp_path / "x.js", [], run=procedure.local)["error"]
    tree = tmp_path / "broken"
    (tree / procedure.MODEL_DIR).mkdir(parents=True)
    (tree / procedure.MODEL_DIR / "fit.py").write_text("raise SystemExit('unsupported: shift validation needs a region column')\n")
    report = procedure.validate(tree, rows, tmp_path / "x.js", [], run=procedure.local, metric=alloy.ranking)
    assert report == {"error": "fit.py failed: unsupported: shift validation needs a region column"}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bench_dir(tmp_path, metric) -> Path:
    """The bench as an improver's workspace has it: its scripts beside the campaign's ranking metric."""
    target = tmp_path / "library" / "bench"
    target.mkdir(parents=True)
    procedure.scorers(target, metric)
    return target / "bench.py"


def test_the_bench_scores_a_procedure_across_runs_and_pooled(tmp_path, alloys):
    alloy, ledger = alloys
    rows = feedback.training_rows(ledger, alloy)
    evidence = tmp_path / "evidence"
    for run, part in (("base-1", rows[6:30]), ("k1-1", rows[30:])):
        write_csv(evidence / run / "train.csv", rows[:6] + part)
        write_csv(evidence / run / "measured.csv", part)
    tree = harness_with_procedure(tmp_path, alloy)
    done = subprocess.run([sys.executable, str(bench_dir(tmp_path, alloy.ranking)), str(tree / procedure.MODEL_DIR),
                           "--evidence", str(evidence), "--pooled"], capture_output=True, text=True, check=True)
    scores = json.loads(done.stdout.strip().splitlines()[-1])["scores"]
    assert scores["metric"] == alloy.ranking  # the campaign's own ranking metric, not a generic one
    assert set(scores["cross-run"]["runs"]) == {"base-1", "k1-1"} and scores["cross-run"]["ranking"] > 0.5
    assert scores["pooled"]["rows"] == 46 and scores["pooled"]["spearman"] > 0.5  # the seed is counted once
    # The pooled score is the pipeline's own cross-validation of the procedure on those rows.
    report = procedure.validate(tree, rows, tmp_path / "none.js", [], run=procedure.local, metric=alloy.ranking)
    assert scores["pooled"]["ranking"] == pytest.approx(report["cv"]["ranking"], abs=1e-4)


def test_the_bench_compares_a_procedure_with_the_models_a_harness_s_runs_delivered(tmp_path, alloys):
    alloy, ledger = alloys
    rows = feedback.training_rows(ledger, alloy)
    evidence = tmp_path / "evidence"
    for run, part in (("base-1", rows[6:26]), ("base-2", rows[26:])):
        write_csv(evidence / run / "train.csv", rows[:6] + part)
        write_csv(evidence / run / "measured.csv", part)
        # The stock agents delivered a constant model: it ranks nothing.
        (evidence / run / "final_model.js").write_text("function predict(design) { return 0; }\n")
    tree = harness_with_procedure(tmp_path, alloy)
    done = subprocess.run([sys.executable, str(bench_dir(tmp_path, alloy.ranking)), str(tree / procedure.MODEL_DIR),
                           "--evidence", str(evidence), "--against", "base", "--no-cross"],
                          capture_output=True, text=True, check=True)
    paired = json.loads(done.stdout.strip().splitlines()[-1])["scores"]["against base"]
    assert set(paired["runs"]) == {"base-1", "base-2"} and paired["procedure"] > paired["rival"]
    assert paired["gain"] == pytest.approx(paired["procedure"] - paired["rival"], abs=1e-3) and paired["error"] is not None
    assert not (evidence / "base-1" / "final_model.run.js").exists()  # the evidence is left as it was


def evidence_of(tmp_path, rows) -> Path:
    evidence = tmp_path / "evidence"
    (evidence / "base-1").mkdir(parents=True)
    with (evidence / "base-1" / "train.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return evidence


LEAKY_FIT = '''
import argparse, pathlib
parser = argparse.ArgumentParser()
parser.add_argument("--train"); parser.add_argument("--out")
args = parser.parse_args()
# Ignores the targets: the answer is built in.
pathlib.Path(args.out).write_text("""function predict(s) {
  const m = s.match(/(A|B)\\\\|Co([0-9.]+)Fe([0-9.]+)Ni([0-9.]+)/);
  return 0.05 * +m[2] - 0.02 * +m[3] + (m[1] === "B" ? 1 : 0);
}""")
'''


def test_the_audit_passes_a_procedure_that_learns_and_rejects_a_built_in_answer(tmp_path, alloys):
    alloy, ledger = alloys
    evidence = evidence_of(tmp_path, feedback.training_rows(ledger, alloy))
    tree = harness_with_procedure(tmp_path, alloy)
    learns = procedure.audit(tree / procedure.MODEL_DIR, evidence, run=procedure.local)
    # On these 46 toy rows one of the four shuffles alone lands near -0.9, so the learning procedure's mean sits
    # near the production limit of 0.2 by chance (it did before the toolkit chose by the ranking metric too);
    # what the audit must do is separate it from a built-in answer.
    assert abs(learns["spearman"]) < 0.25 and learns["rows"] == 46, learns
    leaky = tmp_path / "leaky"
    leaky.mkdir()
    (leaky / "fit.py").write_text(LEAKY_FIT)
    answer = procedure.audit(leaky, evidence, run=procedure.local)
    assert answer["spearman"] > 0.5, answer
