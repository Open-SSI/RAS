"""The model-guided planner (ras/planner/acquire.py) against a fake lab, with the toolkit's train.py and a toy procedure."""

import csv
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from ras import harness, procedure

REFERENCE = "MKKAAAAAAAAAAAAAAAAA"  # 20 residues
FIT = '''import argparse, csv
p = argparse.ArgumentParser(); p.add_argument("--train"); p.add_argument("--out"); a = p.parse_args()
rows = list(csv.DictReader(open(a.train)))
xs = [r["input"].count("K") for r in rows]; ys = [float(r["target"]) for r in rows]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sum((x - mx) ** 2 for x in xs) or 1)
open(a.out, "w").write(f"function predict(s) {{ return {slope} * (s.split('K').length - 1) + {my - slope * mx}; }}\\n")
'''


def variant(positions: list[int]) -> str:
    return "".join("K" if i in positions else r for i, r in enumerate(REFERENCE))


@pytest.fixture
def world(tmp_path):
    catalog = {f"v{i:03d}": variant(sorted({(i * 7 + j * 3) % 17 + 3 for j in range(1 + i % 6)})) for i in range(120)}
    measured_ids = list(catalog)[:20]
    target = lambda key: min(3, catalog[key].count("K") // 2)
    data = tmp_path / "data"
    data.mkdir()
    with (data / "pool.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant_id", "sequence", "family"])
        writer.writerows([key, seq, "ab"[i % 2]] for i, (key, seq) in enumerate(catalog.items()))
    (data / "reference.fasta").write_text(">wt\n" + REFERENCE + "\n")
    jobs = [{"job_id": "job-0001", "stage": 1, "status": "completed",
             "results": [{"variant_id": key, "activity_class": target(key)} for key in measured_ids]}]
    state = {"next_stage": 2}

    class Lab(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/v1/status":
                body = {"next_stage": state["next_stage"], "stages_completed": 1, "stages_total": 3}
            elif self.path == "/v1/campaign":
                body = {"stages": [{"stage": s, "batch_size": 10} for s in (1, 2, 3)]}
            elif self.path == "/v1/experiments":
                body = {"jobs": [{"job_id": j["job_id"], "status": j["status"]} for j in jobs]}
            else:
                body = {"job": next(j for j in jobs if self.path.endswith(j["job_id"]))}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Lab)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    tree = harness.write_stock(tmp_path / "tree")
    model = tree / "scripts" / "model"
    model.mkdir(parents=True)
    shutil.copyfile(procedure.TOOLKIT / "train.py", model / "train.py")
    (model / "fit.py").write_text(FIT)
    (model / "spec.json").write_text(json.dumps({
        "catalog": str(data / "pool.csv"), "id": "variant_id", "input": "sequence",
        "objective": {"field": "activity_class", "higher_is_better": True, "zero_above": None}, "group": [], "seed": []}))
    yield {"tree": tree, "catalog": catalog, "measured": measured_ids, "jobs": jobs, "data": data, "server": server,
           "env": os.environ | {"ASB_LAB_URL": f"http://127.0.0.1:{server.server_port}",
                                "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}}
    server.shutdown()


def plan(world, tmp_path, **config):
    harness.install_planner(world["tree"], {"beta": 1.0, "explore_share": 0.3, "bootstrap": 4, "min_evidence": 0.25,
                                            "seed": 1} | config)
    out = tmp_path / "round_2.csv"
    done = subprocess.run([sys.executable, str(world["tree"] / "scripts" / "plan" / "acquire.py"), "--out", str(out)],
                          env=world["env"], capture_output=True, text=True)
    batch = out.read_text().splitlines()[1:] if out.is_file() else []
    return done, batch


@pytest.mark.skipif(shutil.which("deno") is None and not (Path(sys.executable).parent / "deno").exists(), reason="deno")
def test_the_planner_proposes_a_full_batch_of_unmeasured_designs_only_with_evidence(world, tmp_path):
    done, batch = plan(world, tmp_path)
    assert done.returncode == 0, done.stderr
    assert len(batch) == 10 == len(set(batch)) and not set(batch) & set(world["measured"])
    assert "Order exactly this file" in done.stdout
    world["jobs"][0]["results"] = world["jobs"][0]["results"][:5]  # fewer measurements than a batch: no evidence yet
    done, _ = plan(world, tmp_path / "again")
    assert done.returncode == 3 and "insufficient evidence" in done.stdout


@pytest.mark.skipif(shutil.which("deno") is None and not (Path(sys.executable).parent / "deno").exists(), reason="deno")
def test_the_planner_keeps_every_pick_out_of_measured_hit_families_and_out_of_each_other_s(world, tmp_path):
    linkage = {"reference": str(world["data"] / "reference.fasta"), "distance": 1, "hit_at_least": 2}
    done, batch = plan(world, tmp_path, linkage=linkage, explore_share=0.0)
    assert done.returncode == 0 and len(batch) == 10, done.stdout + done.stderr
    subs = lambda key: {i for i, (a, b) in enumerate(zip(REFERENCE, world["catalog"][key])) if a != b}
    hits = [r["variant_id"] for r in world["jobs"][0]["results"] if r["activity_class"] >= 2]
    for i, key in enumerate(batch):
        assert all(len(subs(key) ^ subs(h)) > 1 for h in hits)
        assert all(len(subs(key) ^ subs(other)) > 1 for other in batch[:i])


@pytest.mark.skipif(shutil.which("deno") is None and not (Path(sys.executable).parent / "deno").exists(), reason="deno")
def test_the_planner_gives_every_group_an_equal_share_where_asked(world, tmp_path):
    spec = json.loads((world["tree"] / "scripts" / "model" / "spec.json").read_text())
    (world["tree"] / "scripts" / "model" / "spec.json").write_text(json.dumps(spec | {"group": ["family"]}))
    done, batch = plan(world, tmp_path, allocate=True)
    assert done.returncode == 0, done.stderr
    with (world["data"] / "pool.csv").open(newline="") as handle:
        family = {row["variant_id"]: row["family"] for row in csv.DictReader(handle)}
    assert sorted(family[key] for key in batch) == ["a"] * 5 + ["b"] * 5


def test_a_planner_that_does_not_deliver_leaves_the_agent_s_model_in_place(tmp_path):
    tree = harness.write_stock(tmp_path / "tree")
    (tree / "scripts" / "model").mkdir()
    (tree / "scripts" / "model" / "fit.py").write_text("print('fit')\n")
    assert harness.delivery(tree, tmp_path / "a") is not None
    harness.install_planner(tree, {"deliver": False})
    assert harness.delivery(tree, tmp_path / "b") is None and harness.planned(tree)
    uploads = {remote: local for local, remote in harness.mount(tree, tmp_path / "m", "AGENTS.md")}
    assert (uploads["/app/harness/scripts"] / "model" / "fit.py").is_file()  # the planner needs the procedure in the run
