"""File-scoped editors with a stated mechanism, replay on matched trajectories, and sequential selection."""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from ras import evidence, feedback, harness
from ras.recursion import change_problems

from test_recursion import FakeRunner, checked, clean, events, validator
from ras.recursion import Recursion


def result(d, l, passed=None):
    return {"discovery": d, "learning": l, "passed": d >= 1 and l >= 1 if passed is None else passed}


def test_paired_comparison_uses_the_paired_spread_and_unpaired_the_pooled_one():
    incumbent = [result(1.7, 0.8), result(1.7, 1.2), result(1.7, 0.9)]
    candidate = [result(1.7, 1.0), result(1.7, 1.25), result(1.7, 1.05)]
    paired = feedback.compare(candidate, incumbent, 0.2, list(zip(candidate, incumbent)))
    unpaired = feedback.compare(candidate, incumbent, 0.2)
    assert paired["paired"] and not unpaired["paired"]
    assert paired["delta"] == unpaired["delta"] == pytest.approx(0.1167, abs=1e-4)  # utility capped at 1.2
    assert paired["se"] < unpaired["se"] == pytest.approx(0.2 * (2 / 3) ** 0.5, abs=1e-4)
    fixed = feedback.compare(candidate[:2], incumbent, 0.2, fixed=(True, False))  # identical runs add no spread
    assert fixed["se"] == pytest.approx(0.2 / 3 ** 0.5, abs=1e-4)


def test_screen_rejects_clear_regressions_and_lost_gates_and_accept_guards_the_luckiest_branch():
    incumbent = [result(1.7, 0.9), result(1.7, 1.1)]
    worse = feedback.compare([result(1.7, 0.6), result(1.7, 0.7)], incumbent, 0.1)
    assert feedback.screen(worse, []) == "reject"
    better = feedback.compare([result(1.7, 1.2), result(1.7, 1.2)], incumbent, 0.1)
    assert feedback.screen(better, []) == "promising"
    assert feedback.screen(better, ["discovery"]) == "reject"
    assert feedback.lost([result(0.9, 1.2)], feedback.reliable(incumbent)) == ["discovery"]
    assert feedback.accept(better, [], [result(1.7, 1.15)], better["incumbent"]["utility"], 1.0) == []
    shrunk = feedback.accept(better, [], [result(1.7, 0.95)], better["incumbent"]["utility"], 1.0)
    assert len(shrunk) == 1 and "fresh runs average 0.950" in shrunk[0]


def test_the_modeling_files_are_scoped_digested_grafted_and_put_back(tmp_path):
    plan = harness.write_stock(tmp_path / "plan")
    (plan / "system_prompt.md").write_text("Order across families.\n")
    model = harness.write_stock(tmp_path / "model")
    (model / "modeling.md").write_text("Deliver fit.py's model.\n")
    (model / "scripts" / "model").mkdir()
    (model / "scripts" / "model" / "fit.py").write_text("print('fit')\n")
    stock = harness.write_stock(tmp_path / "h0")
    assert harness.digest(model, "planning") == harness.digest(stock, "planning") != harness.digest(plan, "planning")
    graft = harness.graft(plan, model, tmp_path / "graft")
    assert harness.digest(graft, "planning") == harness.digest(plan, "planning")
    assert harness.digest(graft, "modeling") == harness.digest(model, "modeling")
    assert "Deliver fit.py's model" not in harness.render_instructions(graft)  # the harness applies its procedure itself
    edited = harness.copy(model, tmp_path / "edited")
    (edited / "system_prompt.md").write_text("Also plan the rounds.\n")
    assert harness.revert(stock, edited, harness.is_modeling) == ["system_prompt.md"]
    assert harness.is_placeholder(edited / "system_prompt.md") and (edited / "scripts" / "model" / "fit.py").is_file()


def test_a_change_statement_needs_a_mechanism_and_a_direction_for_each_gate(tmp_path):
    path = tmp_path / "CHANGE.json"
    assert "missing" in change_problems(path)[0]
    path.write_text(json.dumps({"mechanism": "m", "change": "c", "components": ["modeling.md"],
                                "expected": {"discovery": {"direction": "keep"}, "learning": {"direction": "up"}}}))
    assert change_problems(path) == ["CHANGE.json: `expected.learning.direction` must be one of raise, keep, lower"]


class ScopedImprover:
    """A modeling editor that adopts the procedure (and strays into system_prompt.md); a planning editor that edits the
    plan (and strays into modeling.md). Each writes CHANGE.json, except a session named in ``silent`` on its first try."""

    def __init__(self, silent: str = ""):
        self.silent, self.sessions = silent, []

    def session(self, workspace: Path, attempt: int = 1) -> int:
        self.sessions.append((workspace.name, attempt))
        tree = workspace / "harness"
        if workspace.name.endswith("modeling"):
            shutil.copytree(workspace / "library" / "model-procedure", tree / "scripts" / "model", dirs_exist_ok=True)
            (tree / "modeling.md").write_text("Build train.csv, run fit.py, deliver its output unmodified.\n")
            (tree / "system_prompt.md").write_text("Order the best predicted designs.\n")
        else:
            (tree / "system_prompt.md").write_text(f"Spread each batch across families. ({workspace.name})\n")
            (tree / "modeling.md").write_text("Use a forest.\n")
        (workspace / "RATIONALE.md").write_text(f"{workspace.name}\n")
        if not (workspace.name == self.silent and attempt == 1):
            gate = "learning" if workspace.name.endswith("modeling") else "discovery"
            (workspace / "CHANGE.json").write_text(json.dumps({
                "mechanism": "the delivered models vary on similar evidence", "change": "a deterministic procedure",
                "components": ["modeling.md"], "expected": {g: {"direction": "raise" if g == gate else "keep"}
                                                            for g in feedback.GATES}}))
        return 0


class FakeReplayer:
    """Stands in for replay.Replayer: discovery is the trajectory's own, learning the procedure's, set per harness label."""

    def __init__(self, learning: dict[str, float]):
        self.learning, self.calls = learning, []

    def score(self, record: Path, model: Path, target: Path) -> dict:
        own = evidence.read_json(record / "verifier.json")
        share = float(model.read_text().split("//")[1]) if "//" in model.read_text() else 0.5
        out = own | {"learning": share, "bottleneck": min(own["discovery"], share),
                     "passed": own["discovery"] >= 1 and share >= 1}
        target.mkdir(parents=True)
        (target / "verifier.json").write_text(json.dumps(out))
        return out

    def replay(self, tree: Path, record: Path, target: Path) -> dict:
        if (target / "verifier.json").is_file():
            return evidence.read_json(target / "verifier.json")
        label = target.name.split("@")[0]
        self.calls.append(target.name)
        own = evidence.read_json(record / "verifier.json")
        share = next(v for prefix, v in self.learning.items() if label.startswith(prefix))
        out = own | {"learning": share, "bottleneck": min(own["discovery"], share),
                     "passed": own["discovery"] >= 1 and share >= 1}
        target.mkdir(parents=True)
        (target / "verifier.json").write_text(json.dumps(out))
        (target / "final_model.js").write_text("function predict(x) { return 1; }\n")
        return out


@pytest.fixture
def sequential(study):
    return replace(study, protocol=replace(study.protocol, selection="sequential", base_runs=3, probe_runs=2,
                                           confirm_runs=2, eval_runs=2))


def make(study, campaign, root, improver, hits, ndcg, learning, **options):
    runner = FakeRunner(root / "jobs", hits, ndcg)
    return Recursion(study, campaign, study.executors["fake"], 1, runner, improver, root, validator=validator,
                     auditor=clean, scorer=lambda source, inputs: [0.0] * len(inputs), checker=checked,
                     replayer=FakeReplayer(learning), **options)


def test_steady_discovery_freezes_the_plan_and_a_replayed_procedure_is_confirmed_then_kept(tmp_path, sequential, campaign):
    run = make(sequential, campaign, tmp_path / "runs", ScopedImprover(), hits={"base": 3, "k1": 3},
               ndcg={"base": [0.40, 0.45, 0.42], "k1-modeling": [0.55, 0.56]}, learning={"k1-modeling": 1.1}, steps=1)
    assert run.run().name == "k1-modeling"
    assert {p.name for p in (run.dir / "sessions").iterdir()} == {"k1-modeling"}  # no planning editor
    replays = sorted(path.name for path in (run.dir / "replays").iterdir())
    assert replays == ["k1-modeling@base-1", "k1-modeling@base-2", "k1-modeling@base-3"]
    assert len(run.replayer.calls) == 1  # the three runs measured the same designs: one refit serves all three
    assert "k1-modeling-1" not in run.runner.jobs[0]  # the branch's first real runs are its confirmation runs
    (scope,) = events(run, "scope")
    assert scope["reverted"] == ["system_prompt.md"]  # the modeling editor's edit to the plan was put back
    assert harness.is_placeholder(run.dir / "harness" / "k1-modeling" / "system_prompt.md")
    (screen,) = events(run, "screen")
    assert screen["tests"]["k1-modeling"]["paired"] and screen["tests"]["k1-modeling"]["verdict"] == "promising"
    (accept,) = events(run, "accept")
    assert accept["reasons"] == [] and accept["test"]["real_runs"] == 2 and accept["test"]["replays"] == 3
    (credit,) = events(run, "credit")
    assert credit["statement"]["expected"]["learning"]["direction"] == "raise" and credit["decision"] == "accepted"
    archive = json.loads((run.dir / "archive.json").read_text())
    assert archive["best"][-1]["harness"] == "k1-modeling" and archive["branches"]["k1-modeling"]["paired"]
    stage = (run.dir / "sessions" / "k1-modeling" / "STAGE.md").read_text()
    assert "the experiment plan is frozen this step" in stage and "refits that procedure on each of the 3" in stage
    assert "Learning varies from run to run" not in stage  # discovery is constant, but learning's spread is too small
    assert "CHANGE.json" in stage and "| base-1 | 1.000 | 0.800 |" in stage


def test_a_procedure_worse_on_the_same_trajectories_is_rejected_without_a_run(tmp_path, sequential, campaign):
    run = make(sequential, campaign, tmp_path / "runs", ScopedImprover(), hits={"base": 3},
               ndcg={"base": [0.40, 0.45, 0.42]}, learning={"k1-modeling": 0.7}, steps=1)
    assert run.run().name == "h0"
    (rollback,) = events(run, "rollback")
    assert rollback["options"]["k1-modeling"]["verdict"] == "reject"
    assert not any("k1-modeling" in job for job in run.runner.jobs)  # no probe, no confirmation


def test_both_editors_run_while_both_gates_fail_and_the_graft_is_replayed_on_the_plan_s_runs(tmp_path, sequential, campaign):
    run = make(sequential, campaign, tmp_path / "runs", ScopedImprover(silent="k1-planning"),
               hits={"base": 2, "k1-planning": 3, "k1-graft": 3}, ndcg={"base": [0.40, 0.45, 0.42], "k1-planning": [0.45, 0.47]},
               learning={"k1-modeling": 0.95, "k1-graft": 1.1}, steps=1)
    run.run()
    assert ("k1-planning", 2) in run.improver.sessions  # the first try wrote no CHANGE.json
    assert harness.is_placeholder(run.dir / "harness" / "k1-planning" / "modeling.md")  # put back
    graft = run.dir / "harness" / "k1-graft"
    assert harness.digest(graft, "planning") == harness.digest(run.dir / "harness" / "k1-planning", "planning")
    assert {"k1-graft@k1-planning-1", "k1-graft@k1-planning-2"} <= {p.name for p in (run.dir / "replays").iterdir()}
    (screen,) = events(run, "screen")
    assert set(screen["tests"]) == {"k1-planning", "k1-modeling", "k1-graft"}
    assert not screen["tests"]["k1-graft"]["paired"] and screen["tests"]["k1-modeling"]["paired"]
    decision = events(run, "accept", "rollback")[0]
    assert decision["branch"] == "k1-graft"  # the most promising branch is the one confirmed
    assert json.loads((run.dir / "harness" / "k1-graft.change.json").read_text())["graft"]["plan"] == "k1-planning"


def test_a_replay_refits_the_procedure_on_the_run_s_own_rows_and_scores_it_with_the_verifier(tmp_path, campaign):
    from conftest import DESIGNS, write_trial
    from ras import procedure
    from ras.replay import Replayer
    record = evidence.record(write_trial(tmp_path / "trial", list(DESIGNS)[:6], ndcg=0.3), tmp_path / "base-1",
                             campaign, "", "h0")
    tree = harness.write_stock(tmp_path / "tree")
    procedure.library(campaign, tree / "scripts" / "model")
    seen = {}

    def fitter(root, argv):  # stands in for the sandbox: the rows it was given, and a model
        seen["rows"] = (root / "train.csv").read_text().splitlines()
        (root / "final_model.js").write_text("function predict(x) { return x.length; }\n")

    def verifier(c, ledger, model):  # stands in for the task's tests/test_outputs.py
        seen["model"] = model.read_text()
        return {"discovery_hits": 2, "ndcg_at_20": 0.6, "passed": False}

    out = Replayer(campaign, fitter, verifier).replay(tree, record, tmp_path / "replays" / "k1@base-1")
    assert len(seen["rows"]) == 1 + 6 and "x.length" in seen["model"]  # header and the six measured designs
    assert out["learning"] == pytest.approx(1.2) and out["discovery"] == pytest.approx(2 / 3)
    assert (tmp_path / "replays" / "k1@base-1" / "replayed.txt").read_text().strip() == "base-1"
    crashed = lambda root, argv: subprocess.CompletedProcess(argv, 1, "", "Traceback ... ValueError: no rows")
    failing = Replayer(campaign, crashed, verifier).replay(tree, record, tmp_path / "replays" / "k2@base-1")
    assert failing["learning"] == 0.0 and failing["discovery"] == pytest.approx(2 / 3) and "no rows" in failing["error"]
    assert failing["gates"]["ndcg_at_20"]["value"] is None


def test_a_procedure_never_reaches_the_agent_and_the_harness_delivers_its_model_afterwards(tmp_path, monkeypatch):
    import asyncio
    from harbor.agents.installed.codex import Codex
    from ras.agents import CAPTURE, DELIVER, START, HarnessedCodex
    from test_agents import FakeEnvironment
    tree = harness.write_stock(tmp_path / "tree")
    (tree / "modeling.md").write_text("Fit pair-context features; order near the best pairs.\n")
    (tree / "scripts" / "model").mkdir()
    (tree / "scripts" / "model" / "fit.py").write_text("print('fit')\n")
    (tree / "scripts" / "batch.py").write_text("print('batch')\n")
    text = harness.render_instructions(tree)
    assert "pair-context" not in text and "fit.py" not in text and "Delivered model" not in text
    assert harness.pack(tree) == harness.pack(tree)  # reproducible
    prose = harness.write_stock(tmp_path / "prose")  # modeling guidance without a procedure stays readable
    (prose / "modeling.md").write_text("Validate the final model by ranking.\n")
    assert "Validate the final model by ranking." in harness.render_instructions(prose)

    async def nothing(self, *args, **kwargs):
        return None
    monkeypatch.setattr(Codex, "setup", nothing)
    monkeypatch.setattr(Codex, "run", nothing)
    agent = HarnessedCodex(logs_dir=tmp_path / "logs", model_name="model", harness_dir=str(tree))
    environment = FakeEnvironment()
    asyncio.run(agent.setup(environment))
    assert environment.uploads["/app/harness/scripts"] == ["batch.py"]  # no model files while the agent works
    assert "/opt/ras-delivery" not in environment.uploads and any(c.endswith(f"touch {START}") for c in environment.commands)
    asyncio.run(agent.run("instruction", environment, None))
    assert environment.uploads["/opt/ras-delivery"] == ["deliver_model.py", "model.tar.gz"]
    assert CAPTURE in environment.commands[-2] and environment.downloads == [  # before the harness delivers:
        ("/opt/ras-agent", str(tmp_path / "logs" / "agent_models")),            # the agent's own files,
        ("/app/plan", str(tmp_path / "logs" / "plan"))]                          # and any planner proposals
    assert environment.commands[-1].endswith(DELIVER) and (tmp_path / "logs" / "delivery.txt").read_text().startswith("exit 0")


def test_the_delivery_wrapper_refuses_before_the_last_round_and_builds_after(tmp_path):
    import http.server
    import os
    import sys
    import threading
    progress = {"stages_completed": 1, "stages_total": 2}

    class Lab(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(progress).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Lab)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    tree = harness.write_stock(tmp_path / "tree")
    (tree / "scripts" / "model").mkdir()
    (tree / "scripts" / "model" / "train.py").write_text(
        "import sys; open(sys.argv[2], 'w').write('input,target\\nA,1\\n')\n")
    (tree / "scripts" / "model" / "fit.py").write_text(
        "import sys; rows = open(sys.argv[2]).read().count('\\n') - 1\n"
        "open(sys.argv[4], 'w').write(f'function predict(x) {{ return {rows}; }}\\n')\n")
    script, out = harness.delivery(tree, tmp_path / "staging") / "deliver_model.py", tmp_path / "final_model.js"
    env = os.environ | {"ASB_LAB_URL": f"http://127.0.0.1:{server.server_port}"}
    early = subprocess.run([sys.executable, str(script), "--out", str(out)], env=env, capture_output=True, text=True)
    assert early.returncode == 2 and "only after the last round" in early.stderr and not out.exists()
    progress["stages_completed"] = 2
    done = subprocess.run([sys.executable, str(script), "--out", str(out)], env=env, capture_output=True, text=True)
    server.shutdown()
    assert done.returncode == 0, done.stderr
    assert out.read_text() == "function predict(x) { return 1; }\n"


def test_a_plan_keeping_branch_must_reproduce_the_incumbent_s_discovery():
    def run(d, hits, families):
        return {"discovery": d, "learning": 1.1, "passed": d >= 1,
                "gates": {"discovery_hits": {"value": hits}, "distinct_hit_families": {"value": families}}}
    incumbent = [run(0.96, 183, 183), run(1.01, 192, 184), run(1.07, 204, 199)]
    kept = {"a-1": run(1.05, 199, 199), "a-2": run(1.0, 195, 190)}
    assert feedback.unrealized(kept, incumbent, 0.1, "distinct_hit_families", 0.05) == []
    chained = {"a-1": run(1.05, 199, 199), "a-2": run(0.97, 233, 169)}  # hits chained into fewer families
    (reason,) = feedback.unrealized(chained, incumbent, 0.1, "distinct_hit_families", 0.05)
    assert reason.startswith("a-2 lost 27% of its hits") and "at most 4%" in reason
    lower = {"a-1": run(0.8, 152, 152), "a-2": run(0.82, 156, 156)}
    assert "below the incumbent's 1.013" in feedback.unrealized(lower, incumbent, 0.1, None, 0.05)[0]


def test_later_editors_see_every_earlier_branch_with_its_code_and_replays(tmp_path, sequential, campaign):
    run = make(sequential, campaign, tmp_path / "runs", ScopedImprover(), hits={"base": 3},
               ndcg={"base": [0.40, 0.45, 0.42]}, learning={"k1-modeling": 0.7, "k2-modeling": 0.7}, steps=2)
    run.run()
    shown = run.dir / "sessions" / "k2-modeling" / "branches"
    assert [p.name for p in shown.iterdir()] == ["k1-modeling"]  # no branch of its own step
    assert "scripts/model/fit.py" in (shown / "k1-modeling" / "harness.diff").read_text()
    decision = json.loads((shown / "k1-modeling" / "decision.json").read_text())
    assert decision["decision"] == "reject" and decision["replays"]["base-1"]["learning"] == 0.7
    assert json.loads((shown / "k1-modeling" / "CHANGE.json").read_text())["expected"]["learning"]["direction"] == "raise"
    assert not (run.dir / "sessions" / "k1-modeling" / "branches").exists()


def test_the_record_keeps_the_model_the_agent_itself_delivered(tmp_path):
    import hashlib
    trial = tmp_path / "trial"
    (trial / "agent" / "agent_models").mkdir(parents=True)
    (trial / "artifacts" / "state").mkdir(parents=True)
    own, other = b"function predict(x) { return 1; }\n", b"function predict(x) { return 2; }\n"
    for body in (own, other):
        (trial / "agent" / "agent_models" / f"{hashlib.sha256(body).hexdigest()}.js").write_bytes(body)
    ledger = {"deliverables": [{"seq": 1, "sha256": hashlib.sha256(own).hexdigest()}, {"seq": 2, "sha256": "f" * 64}]}
    (trial / "artifacts" / "state" / "ledger.json").write_text(json.dumps(ledger))
    assert evidence.agent_model(trial) is None  # without the harness's delivery, the agent's model is the delivered one
    (trial / "agent" / "delivery.txt").write_text("exit 0\nwrote ...\n")
    assert evidence.agent_model(trial).read_bytes() == own


def test_fresh_runs_are_compared_on_their_own_trajectories_or_on_the_gate_the_branch_changes():
    test = {"delta": 0.05, "se": 0.03, "candidate": {"passed": 0.6}, "incumbent": {"passed": 0.0}}
    worse, better = (result(1.5, 1.05), result(1.5, 1.1)), (result(1.5, 1.1), result(1.5, 1.0))
    assert "scores -0.050" in feedback.accept(test, [], [result(1.5, 1.05)], 0.96, fresh_pairs=[worse])[0]
    assert feedback.accept(test, [], [result(0.9, 1.1)], 0.96, fresh_pairs=[better]) == []
    assert feedback.accept(test, [], [result(0.92, 1.1)], 0.96, gate="learning", incumbent_gate=0.97) == []
    assert "learning averages 0.900" in feedback.accept(test, [], [result(0.92, 0.9)], 0.96, gate="learning",
                                                        incumbent_gate=0.97)[0]
    assert "fresh runs average 0.920" in feedback.accept(test, [], [result(0.92, 1.1)], 0.96)[0]


def test_a_captured_agent_model_makes_each_fresh_run_a_matched_pair(tmp_path, sequential, campaign):
    run = make(sequential, campaign, tmp_path / "runs", ScopedImprover(), hits={"base": 3, "k1": 3},
               ndcg={"base": [0.40, 0.45, 0.42], "k1-modeling": [0.55, 0.56]}, learning={"k1-modeling": 1.1}, steps=1)
    run.run()
    for i, share in ((1, 0.9), (2, 0.95)):  # as evidence.record keeps it: the agent's own model, delivered first
        (run.dir / "records" / f"k1-modeling-{i}" / "agent_model.js").write_text(f"function predict(x) {{ return 0; }} //{share}")
    run._shadows()
    assert {p.name for p in (run.dir / "replays").iterdir()} >= {"base@k1-modeling-1", "base@k1-modeling-2"}
    base = run._results("base")
    assert base["k1-modeling-1"]["learning"] == 0.9 and len(base) == 5  # three own runs, two shadow runs
    test = run._test("k1-modeling", "base")
    assert test["paired"] and test["candidate"]["runs"] == 5  # every trajectory either harness has, under both


class FakeSimulator:
    """Stands in for simulate.Simulator: the planner's run from a source keeps the source's learning and adds discovery."""

    def __init__(self, discovery: float):
        self.discovery, self.calls = discovery, []

    def run(self, tree, source, target, branch_after=1):
        self.calls.append(target.name)
        own = evidence.read_json(source / "verifier.json")
        out = own | {"discovery": self.discovery, "learning": 1.1, "passed": self.discovery >= 1}
        target.mkdir(parents=True)
        (target / "verifier.json").write_text(json.dumps(out))
        return out


def test_a_validated_procedure_brings_a_planner_branch_simulated_from_the_incumbent_s_own_runs(tmp_path, sequential, campaign):
    study = replace(sequential, protocol=replace(sequential.protocol, planner=True, editors="modeling"))
    campaign = replace(campaign, plan={"beta": 1.0, "explore_share": 0.25, "bootstrap": 2, "min_evidence": 0.25})
    run = make(study, campaign, tmp_path / "runs", ScopedImprover(), hits={"base": 2, "k1": 3},
               ndcg={"base": [0.40, 0.45, 0.42], "k1-modeling": [0.55, 0.56], "k1-planner": [0.55, 0.56]},
               learning={"k1-modeling": 1.1}, steps=1, simulator=FakeSimulator(1.3))
    run.run()
    planner = run.dir / "harness" / "k1-planner"
    assert harness.planned(planner) and harness.packaged(planner)  # the incumbent's plan, the procedure, the planner
    assert json.loads((planner / "scripts" / "plan" / "plan.json").read_text())["bootstrap"] == 2
    assert "## Experiment planning" in harness.render_instructions(planner)
    # the fake runs all ordered the same first round, so one simulation stands for all three
    assert run.simulator.calls == ["k1-planner@base-1"]
    (screen,) = events(run, "screen")
    assert screen["tests"]["k1-planner"]["verdict"] == "promising"
    decision = events(run, "accept", "rollback")[0]
    # the fake runs carry no proposals: the agent never ran the planner, so its confirmation cannot stand for it
    assert any("did not run the planner" in r for r in decision["confirmed"].get("k1-planner", []))


def test_a_gate_a_shared_trajectory_already_failed_is_not_lost_by_the_candidate():
    incumbent = {"a": result(1.2, 0.9), "b": result(0.9, 1.0)}  # trajectory b failed discovery under both
    candidate = {"a": result(1.2, 1.1), "b": result(0.9, 1.1)}
    assert feedback.lost(candidate, ["discovery"], incumbent) == []
    assert feedback.lost(candidate | {"own": result(0.8, 1.1)}, ["discovery"], incumbent) == ["discovery"]
    assert feedback.lost({"a": result(0.95, 1.1)}, ["discovery"], incumbent) == ["discovery"]


def test_every_procedure_gets_ras_s_own_train_py_and_spec_so_it_can_be_delivered(tmp_path, campaign):
    from ras import procedure
    model = tmp_path / "scripts" / "model"
    model.mkdir(parents=True)
    (model / "fit.py").write_text("print('fit')\n")
    procedure.complete(model, campaign)
    assert (model / "train.py").read_bytes() == (procedure.TOOLKIT / "train.py").read_bytes()
    assert json.loads((model / "spec.json").read_text())["id"] == campaign.id
