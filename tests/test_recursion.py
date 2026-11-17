import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from ras import harness, recursion as recursion_module
from ras.recursion import Recursion, _label, status

from conftest import DESIGNS, activity, write_trial

HITS = [d for d in DESIGNS if activity(d) >= 2]
MISSES = [d for d in DESIGNS if activity(d) < 2]


class FakeRunner:
    """Writes finished trials where Harbor would, with the verifier's discovery hits and learning score set per
    harness label. A label's ``ndcg`` may be a list, one value per trial in turn; then its runs differ (each model
    names its trial), otherwise a label's runs are identical, as a deterministic harness's would be."""

    def __init__(self, jobs_dir: Path, hits: dict[str, int], ndcg: dict | float = 0.5):
        self.jobs_dir, self.hits, self.ndcg, self.jobs = jobs_dir, hits, ndcg, []
        self.count: dict[str, int] = {}

    def run(self, jobs):
        trials = {}
        for job in jobs:
            self.jobs.append(job.name)
            label = re.split(r"-(?:r|flat)\d+-", job.name, maxsplit=1)[1].split("-more")[0]
            hits = next((n for prefix, n in self.hits.items() if label.startswith(prefix)), 1)
            scores = self.ndcg if isinstance(self.ndcg, float) else next(
                (v for prefix, v in self.ndcg.items() if label.startswith(prefix)), 0.5)
            trials[job.name] = []
            for i in range(job.attempts):
                n = self.count[label] = self.count.get(label, 0) + 1
                ndcg = scores[(n - 1) % len(scores)] if isinstance(scores, list) else scores
                trial = write_trial(self.jobs_dir / job.name / f"trial-{i}", HITS[:hits] + MISSES[: 6 - hits], ndcg)
                model = trial / "artifacts" / "state" / "deliverables" / "final_model.js"
                tag = f"{label} {n}" if isinstance(scores, list) else label
                model.write_text(f"// {tag}\n" + model.read_text())
                trials[job.name].append(trial)
        return trials


class FakeImprover:
    """Edits one always-on component; the modeling editor adopts the model procedure; one session may leak a design id."""

    def __init__(self, leak_in: str = ""):
        self.leak_in, self.sessions = leak_in, []

    def session(self, workspace: Path, attempt: int = 1) -> int:
        self.sessions.append((workspace.name, attempt))
        tree = workspace / "harness"
        if workspace.name == self.leak_in:
            (tree / "memory.md").write_text(f"Order {next(iter(DESIGNS))} first.\n")
        else:
            (tree / "system_prompt.md").write_text(f"Plan every round. ({workspace.name})\n")
            (tree / "scripts" / "__pycache__").mkdir(exist_ok=True)  # as an editor testing a script leaves it
            (tree / "skills" / "retired").mkdir(parents=True, exist_ok=True)  # a skill whose SKILL.md was deleted
            (tree / "scripts" / "__pycache__" / "x.cpython-313.pyc").write_bytes(b"\x00\xff")
        adopted = workspace / "candidates" / "modeling" / "harness" / "scripts" / "model"
        if workspace.name.endswith("modeling"):
            shutil.copytree(workspace / "library" / "model-procedure", tree / "scripts" / "model", dirs_exist_ok=True)
        elif adopted.is_dir():  # the composer merges the modeling editor's procedure
            shutil.copytree(adopted, tree / "scripts" / "model", dirs_exist_ok=True)
        notes = "DECISIONS.md" if (workspace / "candidates").exists() else "RATIONALE.md"
        (workspace / notes).write_text(f"{workspace.name}: kept the planning edit.\n")
        return 0


def validator(tree, rows, delivered, probe):
    """A stand-in for procedure.validate: a harness with a procedure is verified at 0.6."""
    if not (tree / "scripts" / "model" / "fit.py").is_file():
        return {"error": "the harness has no modeling procedure (scripts/model/fit.py)"}
    return {"cv": {"ranking": 0.6, "spearman": 0.5}, "delivered": "verified"}


def clean(model, evidence):
    """A stand-in for procedure.audit: the procedure learns only from its train.csv."""
    return {"spearman": 0.01, "rows": 12}


def checked(model):
    """A stand-in for contract.check."""
    return {"ok": True}


HITS_BY_LABEL = {"base": 1, "k1-": 1, "k1": 2, "k2-": 2, "k2": 3}  # probes trail; each composed harness gains discovery


def recursion(study, campaign, root, improver, hits=None, ndcg=0.4, **options):
    runner = FakeRunner(root / "jobs", hits or HITS_BY_LABEL, ndcg)
    options.setdefault("auditor", clean)
    options.setdefault("scorer", lambda source, inputs: [0.0] * len(inputs))
    options.setdefault("checker", checked)
    return Recursion(study, campaign, study.executors["fake"], 1, runner, improver, root, validator=validator, **options)


def events(run, *kinds):
    return [e for e in (json.loads(line) for line in (run.dir / "events.jsonl").read_text().splitlines())
            if not kinds or e["event"] in kinds]


def test_record_labels():
    assert [_label(n) for n in ("base-3", "k1-2", "k1-planning", "k2-modeling-1", "eval-base-2")] == \
        ["base", "k1", "k1-planning", "k2-modeling", "eval-base"]


def test_full_recursion_selects_on_the_verifier_evaluates_and_resumes(tmp_path, study, campaign):
    first = recursion(study, campaign, tmp_path / "runs", FakeImprover())
    selected = first.run()

    assert selected.name == "k2" and (first.dir / "selected.txt").read_text().strip() == "k2"
    assert set(first.records(evaluation=True)) == {"base-1", "base-2", "k1-planning", "k1-modeling", "k1-1", "k1-2",
                                                   "k2-planning", "k2-modeling", "k2-1", "k2-2", "eval-k2-1", "eval-k2-2",
                                                   "eval-base-1", "eval-base-2"}
    estimates = first.estimates()
    assert estimates["base"]["discovery"] == pytest.approx(1 / 3, abs=1e-4) and estimates["base"]["learning"] == 0.8
    assert estimates["base"]["S"] == pytest.approx(1 / 3, abs=1e-4) and estimates["k2"]["S"] == 0.8
    assert estimates["base"]["verified"] == 0 and estimates["k1"]["verified"] == 2  # the adopted procedure
    decisions = [(e["event"], e["step"], e["test"]["target"]) for e in events(first, "accept", "rollback")]
    assert decisions == [("accept", 1, "discovery"), ("accept", 2, "discovery")]

    composer = first.dir / "sessions" / "k1"
    assert (composer / "candidates" / "modeling" / "RATIONALE.md").is_file()
    stage = (composer / "STAGE.md").read_text()
    assert "## Diagnosis" in stage and "| `k1-planning` | discovery |" in stage and "## Procedure bench" in stage
    assert (composer / "library" / "bench" / "ranking.json").is_file()
    assert json.loads((composer / "evidence" / "base-1" / "verifier.json").read_text())["discovery"] == pytest.approx(1 / 3)
    assert json.loads((composer / "scores.json").read_text())["harnesses"]["base"]["runs"] == 2
    for path in (first.dir / "sessions").rglob("*"):  # nothing raw from the verifier or the task reaches a session
        assert path.name not in {"sealed", "metrics.json", "reward.txt", "secret.csv"}

    again = recursion(study, campaign, tmp_path / "runs", FakeImprover())
    assert again.run() == selected
    assert again.runner.jobs == [] and again.improver.sessions == []
    (row,) = status(tmp_path / "runs")
    assert (row["base"], row["step"], row["rollbacks"], row["selected"], row["eval"]) == (2, 2, 0, "k2", 2)


def test_each_editor_is_told_the_bottleneck_and_its_own_gate(tmp_path, study, campaign):
    run = recursion(study, campaign, tmp_path / "runs", FakeImprover(), steps=1)
    run.run()
    planning = (run.dir / "sessions" / "k1-planning" / "STAGE.md").read_text()
    modeling = (run.dir / "sessions" / "k1-modeling" / "STAGE.md").read_text()
    assert "planning editor" in planning and "modeling editor" in modeling
    assert "The bottleneck is **discovery**" in planning and "| discovery | 0.333 (0.667 below its threshold)" in planning
    assert "Your gate, discovery, is the bottleneck" in planning
    assert "Your gate, learning, fails too, though discovery is lower" in modeling


def test_a_learning_bottleneck_with_steady_discovery_runs_the_modeling_editor_alone(tmp_path, study, campaign):
    run = recursion(study, campaign, tmp_path / "runs", FakeImprover(), hits={"base": 3, "k1": 3},
                    ndcg={"base": 0.4, "k1": 0.45}, steps=1)
    assert run.run().name == "k1-modeling"  # the probe itself is the branch selection keeps
    assert {p.name for p in (run.dir / "sessions").iterdir()} == {"k1-modeling"}  # no planning editor, no composer
    (editors,) = events(run, "editors")
    assert editors["editors"] == ["modeling"] and "discovery passes in every run" in editors["reason"]
    stage = (run.dir / "sessions" / "k1-modeling" / "STAGE.md").read_text()
    assert "runs no planning editor" in stage and "change what is measured too" in stage


def test_a_probe_that_beats_the_composed_harness_is_kept_directly(tmp_path, study, campaign):
    run = recursion(study, campaign, tmp_path / "runs", FakeImprover(),
                    hits={"base": 1, "k1-planning": 3, "k1-modeling": 1, "k1": 2}, steps=1)
    assert run.run().name == "k1-planning"
    (decision,) = events(run, "accept")
    assert set(decision["options"]) == {"k1-planning", "k1-modeling", "k1"} and decision["best"] == "k1-planning"
    assert {"eval-k1-planning-1", "eval-k1-planning-2"} <= set(run.records(evaluation=True))


def test_a_gain_within_the_run_to_run_spread_is_refused_even_after_repeating(tmp_path, study, campaign):
    confirming = replace(study, protocol=replace(study.protocol, confirm_runs=2, probe_runs=2))
    run = recursion(confirming, campaign, tmp_path / "runs", FakeImprover(), hits={"base": 3, "k1": 3},
                    ndcg={"base": [0.4, 0.44], "k1": [0.42, 0.46, 0.41, 0.45]}, steps=1)
    assert run.run().name == "h0"  # +0.03 in learning, inside the runs' spread
    assert {"k1-modeling-3", "k1-modeling-4"} <= set(run.records())
    (decision,) = events(run, "rollback")
    assert decision["test"]["accept"] and not decision["test"]["sure"] and decision["test"]["repeated"] == 2


def test_every_probe_and_composed_harness_gets_credit_for_what_it_changed(tmp_path, study, campaign):
    run = recursion(study, campaign, tmp_path / "runs", FakeImprover(), hits={"base": 1, "k1-planning": 3, "k1": 2}, steps=1)
    run.run()
    credit = {e["harness"]: e for e in events(run, "credit")}
    assert set(credit) == {"k1-planning", "k1-modeling", "k1"}
    assert credit["k1-planning"]["targeted"] == "discovery" and credit["k1-modeling"]["targeted"] == "learning"
    assert credit["k1-planning"]["change"]["discovery"] == pytest.approx(2 / 3, abs=1e-4)
    assert "system_prompt.md" in credit["k1-planning"]["components"]
    assert any(c.startswith("scripts/model/") for c in credit["k1-modeling"]["components"])


def test_a_worse_step_is_rolled_back_and_the_best_harness_is_evaluated(tmp_path, study, campaign):
    run = recursion(study, campaign, tmp_path / "runs", FakeImprover(), hits={"base": 1, "k1-": 1, "k1": 3, "k2": 1})
    assert run.run().name == "k1"
    decisions = [(e["event"], e["step"], e["best"]) for e in events(run, "accept", "rollback")]
    assert decisions == [("accept", 1, "k1"), ("rollback", 2, "k1")]
    k2_editor = run.dir / "sessions" / "k2-modeling"  # k1 passes discovery in every run: modeling alone
    assert json.loads((k2_editor / "scores.json").read_text())["current"] == "k1"
    assert (run.records(evaluation=True)["eval-k1-1"] / "harness.txt").read_text().strip() == "k1"
    assert (run.records(evaluation=True)["eval-base-1"] / "harness.txt").read_text().strip() == "h0"  # Base, beside it


def test_selection_follows_the_verifier_s_learning_score(tmp_path, study, campaign):
    """The same discovery everywhere: a candidate whose models score lower on the verifier is rolled back,
    one that scores higher is kept."""
    worse = recursion(study, campaign, tmp_path / "worse", FakeImprover(), hits={"base": 3, "k1": 3},
                      ndcg={"base": 0.4, "k1": 0.3}, steps=1)
    assert worse.run().name == "h0"
    better = recursion(study, campaign, tmp_path / "better", FakeImprover(), hits={"base": 3, "k1": 3},
                       ndcg={"base": 0.4, "k1": 0.45}, steps=1)
    assert better.run().name == "k1-modeling"
    (decision,) = events(better, "accept")
    assert decision["test"]["target"] == "learning" and decision["test"]["change"]["learning"] == pytest.approx(0.1)


def test_the_stock_harness_is_compared_like_any_other_and_is_its_own_evaluation(tmp_path, study, campaign):
    run = recursion(study, campaign, tmp_path / "runs", FakeImprover(), hits={"base": 3, "k1": 3},
                    ndcg={"base": 0.6, "k1": 0.4}, steps=1)
    assert run.run().name == "h0"
    assert {"eval-base-1", "eval-base-2"} <= set(run.records(evaluation=True))
    assert not [n for n in run.records(evaluation=True) if n.startswith("eval-h0")]  # H0 selected: Base is its evaluation


def test_a_gain_that_loses_a_passing_gate_is_refused(tmp_path, study, campaign):
    run = recursion(study, campaign, tmp_path / "runs", FakeImprover(), hits={"base": 3, "k1": 2},
                    ndcg={"base": 0.3, "k1": 0.9}, steps=1)
    assert run.run().name == "h0"  # S rises from 0.6 to 0.67, but discovery falls from passing to 2/3
    (decision,) = events(run, "rollback")
    assert decision["test"]["lost_gate"] == ["discovery"]


def test_an_uncertain_gain_is_repeated_and_a_deterministic_one_is_not(tmp_path, study, campaign):
    confirming = replace(study, protocol=replace(study.protocol, confirm_runs=2, probe_runs=2))
    noisy = recursion(confirming, campaign, tmp_path / "noisy", FakeImprover(), hits={"base": 3, "k1": 3},
                      ndcg={"base": [0.4, 0.42], "k1": [0.6, 0.3, 0.55, 0.5]}, steps=1)
    assert noisy.run().name == "k1-modeling"
    assert {"k1-modeling-3", "k1-modeling-4"} <= set(noisy.records())
    (repeat,) = events(noisy, "repeat")
    assert repeat["runs"] == 2
    (decision,) = events(noisy, "accept")
    assert decision["test"]["repeated"] == 2 and not decision["test"]["before_repeat"]["sure"]
    steady = recursion(confirming, campaign, tmp_path / "steady", FakeImprover(), hits={"base": 3, "k1": 3},
                       ndcg={"base": [0.4, 0.42], "k1": 0.45}, steps=1)
    assert steady.run().name == "k1-modeling" and steady.deterministic("k1-modeling")
    assert not events(steady, "repeat") and "k1-modeling-3" not in steady.records()  # identical runs: nothing to repeat


def test_identical_runs_count_once(tmp_path, study, campaign):
    run = recursion(study, campaign, tmp_path / "runs", FakeImprover(), steps=1)
    run.run()
    assert len([n for n in run.records() if _label(n) == "base"]) == 2 and len(run._distinct("base")) == 1  # identical fakes
    assert run.estimates()["base"]["distinct"] == 1 and run.deterministic("base")


def test_a_recursion_can_start_from_an_earlier_harness_and_continue(tmp_path, study, campaign):
    earlier = harness.write_stock(tmp_path / "earlier")
    (earlier / "system_prompt.md").write_text("Protect a slot quota for the weaker basin.\n")
    hits = {"base": 1, "start": 2, "k1": 2, "k2-": 2, "k2": 3}
    run = recursion(study, campaign, tmp_path / "runs", FakeImprover(), hits=hits, start=earlier, steps=1)
    assert run.run().name == "start"  # k1 is no better: the incumbent stays
    assert {"base-1", "start-1", "start-2"} <= set(run.records())
    continued = recursion(study, campaign, tmp_path / "runs", FakeImprover(), hits=hits, start=earlier, steps=2)
    assert continued.run().name == "k2"  # step 1 is resumed as decided; k2 then gains discovery
    assert {"eval-k2-1", "eval-k2-2"} <= set(continued.records(evaluation=True))


def test_a_leaking_session_is_retried_then_falls_back(tmp_path, study, campaign):
    improver = FakeImprover(leak_in="k1-planning")
    run = recursion(study, campaign, tmp_path / "runs", improver)
    run.run()
    assert ("k1-planning", 2) in improver.sessions
    assert harness.diff(run.dir / "harness" / "h0", run.dir / "harness" / "k1-planning") == ""
    assert (run.dir / "sessions" / "k1-planning" / "GUARD.md").is_file()
    assert len(events(run, "fallback")) == 1


class SilentImprover(FakeImprover):
    """A session that could not write anything, like a CLI whose own sandbox failed to start."""

    def session(self, workspace: Path, attempt: int = 1) -> int:
        self.sessions.append((workspace.name, attempt))
        return 0


def test_a_session_that_writes_no_notes_stops_the_recursion(tmp_path, study, campaign):
    run = recursion(study, campaign, tmp_path / "runs", SilentImprover())
    with pytest.raises(RuntimeError, match="wrote no RATIONALE.md"):
        run.run()
    assert not (run.dir / "harness" / "k1-planning").exists()  # rerunning retries the session


def test_the_flat_control_edits_the_start_with_only_its_evidence_and_keeps_the_best(tmp_path, study, campaign):
    run = recursion(study, campaign, tmp_path / "runs", FakeImprover(), hits={"base": 1, "k1-": 1, "k1": 3, "k2-": 1, "k2": 2},
                    mode="flat")
    assert run.run().name == "k1" and run.dir.name == "flat1"
    editor, composer = run.dir / "sessions" / "k2-planning", run.dir / "sessions" / "k2"
    assert json.loads((editor / "scores.json").read_text())["current"] == "h0"
    assert {p.name for p in (editor / "evidence").iterdir()} == {"base-1", "base-2"}
    assert {p.name for p in (composer / "evidence").iterdir()} == {"base-1", "base-2", "k2-planning", "k2-modeling"}


def test_a_recursion_refuses_to_resume_under_a_changed_protocol(tmp_path, study, campaign, monkeypatch):
    run = recursion(study, campaign, tmp_path / "runs", FakeImprover(), steps=1)
    run.run()
    recorded = json.loads((run.dir / "protocol.json").read_text())["files"]
    assert "ras/recursion.py" in recorded and "ras/prompts/editor.md" in recorded
    monkeypatch.setattr(recursion_module, "protocol_files", lambda: recorded | {"ras/prompts/editor.md": "edited"})
    with pytest.raises(SystemExit, match="ras/prompts/editor.md"):
        recursion(study, campaign, tmp_path / "runs", FakeImprover(), steps=1).run()
    allowed = recursion(study, campaign, tmp_path / "runs", FakeImprover(), steps=1, allow_protocol_change=True)
    assert allowed.run().name == "k1"
    assert len(events(run, "protocol-change")) == 1


def test_a_procedure_with_a_built_in_answer_is_rejected(tmp_path, study, campaign):
    def leaky(model, evidence):
        return {"spearman": 0.8, "rows": 12}

    run = recursion(study, campaign, tmp_path / "runs", FakeImprover(), steps=1, auditor=leaky)
    run.run()
    assert "without learning them" in (run.dir / "sessions" / "k1-modeling" / "GUARD.md").read_text()
    assert not (run.dir / "harness" / "k1-modeling" / "scripts" / "model").exists()  # fell back to the current harness


def test_a_procedure_whose_model_is_too_slow_for_the_verifier_is_rejected(tmp_path, study, campaign):
    import time

    slow_campaign = replace(campaign, model_ms=5.0)
    slow = "function predict(s) { /* a forest too large to parse per design */ return 0; }\n"

    def auditor(model, evidence):
        return {"spearman": 0.0, "rows": 12, "source": slow}

    def scorer(source, inputs):
        if source == slow:
            time.sleep(0.02 * len(inputs))  # 20 ms per design more than an empty model
        return [0.0] * len(inputs)

    run = recursion(study, slow_campaign, tmp_path / "runs", FakeImprover(), steps=1, auditor=auditor, scorer=scorer)
    run.run()
    guard = (run.dir / "sessions" / "k1-modeling" / "GUARD.md").read_text()
    assert "ms per design" in guard and "at most 5 ms" in guard


def test_a_composed_harness_identical_to_the_best_is_not_run_again(tmp_path, study, campaign):
    class Repeating(FakeImprover):
        def session(self, workspace: Path, attempt: int = 1) -> int:
            if workspace.name != "k2":
                return super().session(workspace, attempt)
            self.sessions.append((workspace.name, attempt))
            (workspace / "DECISIONS.md").write_text("k2: nothing the probes showed beats k1.\n")
            return 0

    run = recursion(study, campaign, tmp_path / "runs", Repeating(), hits={"base": 1, "k1-": 1, "k1": 2, "k2-": 2})
    assert run.run().name == "k1"
    assert not [name for name in run.records() if name.startswith("k2-") and name[3:].isdigit()]
    (duplicate,) = events(run, "duplicate")
    (step2,) = events(run, "rollback")
    assert (duplicate["step"], duplicate["of"], step2["best"]) == (2, "k1", "k1")


def test_an_index_s_ras_and_control_share_their_stock_runs(tmp_path, study, campaign):
    runner = FakeRunner(tmp_path / "runs" / "jobs", HITS_BY_LABEL, 0.4)
    arms = [Recursion(study, campaign, study.executors["fake"], 1, runner, FakeImprover(), tmp_path / "runs",
                      validator=validator, auditor=clean, checker=checked, mode=mode) for mode in ("recursive", "flat")]
    for arm in arms:
        arm.run()
    stock_jobs = [job for job in runner.jobs if job.endswith(("-base", "-eval-base"))]
    assert stock_jobs == [f"fake-{campaign.name}-r1-base", f"fake-{campaign.name}-r1-eval-base"]  # made once
    for arm in arms:
        assert {"base-1", "base-2", "eval-base-1", "eval-base-2"} <= set(arm.records(evaluation=True))
        assert not (arm.dir / "records" / "base-1").exists()  # they live with the index, not the arm


def test_no_editor_ever_sees_an_evaluation_run(tmp_path, study, campaign):
    runner = FakeRunner(tmp_path / "runs" / "jobs", HITS_BY_LABEL, 0.4)
    make = lambda mode: Recursion(study, campaign, study.executors["fake"], 1, runner, FakeImprover(), tmp_path / "runs",
                                  validator=validator, auditor=clean, checker=checked, mode=mode)
    make("flat").run()
    ras = make("recursive")
    ras.run()
    for workspace in (ras.dir / "sessions").iterdir():
        assert not [p for p in (workspace / "evidence").iterdir() if p.name.startswith("eval-")]
        assert not [label for label in json.loads((workspace / "scores.json").read_text())["harnesses"] if label.startswith("eval-")]
    assert not [n for n in ras.records() if n.startswith("eval-")] and "eval-base" not in ras.estimates()


def test_a_cost_ceiling_stops_new_steps_but_not_the_evaluation(tmp_path, study, campaign):
    run = recursion(study, campaign, tmp_path / "runs", FakeImprover(), ceiling=1.0)  # each fake trial reports $0.30
    assert run.run().name == "k1"  # base (2 trials) and step 1 (4) ran; step 2 found $1.80 spent
    (stop,) = events(run, "ceiling")
    assert stop["step"] == 2 and {"eval-k1-1", "eval-k1-2"} <= set(run.records(evaluation=True))


def test_base_can_be_the_opening_runs_instead_of_fresh_stock_runs(tmp_path, study, campaign):
    reuse = replace(study, protocol=replace(study.protocol, base_eval_runs=0))
    run = recursion(reuse, campaign, tmp_path / "runs", FakeImprover())
    assert run.run().name == "k2"
    evaluation = set(run.records(evaluation=True))
    assert {"eval-k2-1", "eval-k2-2"} <= evaluation and not [n for n in evaluation if n.startswith("eval-base")]
    assert not [job for job in run.runner.jobs if job.endswith("-eval-base")]
