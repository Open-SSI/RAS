import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ras import config, contract

TOOLKIT = Path(__file__).resolve().parents[1] / "ras" / "toolkit"
sys.path.insert(0, str(TOOLKIT))
from modeling import Composition, Learner, Mutations, Tokens, candidates, export, folds  # noqa: E402

RUNTIME = config.load().runtime
needs_sandbox = pytest.mark.skipif(shutil.which("deno") is None or not (RUNTIME / "sandbox").is_dir(),
                                   reason="needs deno and the Autonomous Science Bench runtime")


def run_js(source, inputs):
    """The benchmark's own sandboxed runner, as the verifier calls it."""
    return contract.benchmark_runner(RUNTIME)(source, inputs, 60)


def oer_like(n: int, seed: int = 0) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Compositions on two libraries with an additive, library-dependent quality."""
    rng = np.random.default_rng(seed)
    inputs, target, regions = [], [], []
    for _ in range(n):
        library = rng.choice(["Mn-Fe-Co-Ni-La-Ce", "Ca-Mn-Co-Ni-Sn-Sb"])
        elements = rng.choice(["Mn", "Co", "Ni"], size=rng.integers(1, 4), replace=False)
        parts = rng.dirichlet(np.ones(len(elements)))
        inputs.append(f"{library}|" + "".join(f"{e}{p:.2f}" for e, p in zip(elements, parts)))
        target.append(sum(p * {"Mn": 1.0, "Co": 2.0, "Ni": -1.0}[e] for e, p in zip(elements, parts))
                      + (0.5 if library.startswith("Ca") else 0.0))
        regions.append(len(elements))
    return inputs, np.array(target), np.array(regions, float)


def test_featurizers():
    composition = Composition().fit(["Lib-A|Mn0.5Co0.5", "Co6.7Fe60.3Ni33.0", "Cu-0.98-In-0.02"])
    assert composition.elements == ["Co", "Cu", "Fe", "In", "Mn", "Ni"] and composition.prefixes == ["Lib-A"]
    row = composition.transform(["Cu-0.98-In-0.02"])[0]
    assert row[1] == pytest.approx(0.98) and row[3] == pytest.approx(0.02) and row[-1] == 0.0
    tokens = Tokens().fit(["a|x|y", "b|x|z"])
    assert tokens.transform(["b|x|y"]).tolist() == [[0, 1, 1, 1, 0]]


def variants(n: int, seed: int = 0) -> tuple[list[str], np.ndarray]:
    """Aligned sequences with up to three substitutions each and an additive fitness."""
    rng = np.random.default_rng(seed)
    parent, effects = "MKTAYIAKQR", {(2, "W"): 2.0, (5, "P"): -1.5, (7, "L"): 1.0, (8, "E"): 0.5}
    inputs, target = [], []
    for _ in range(n):
        sequence = list(parent)
        for i in rng.choice(10, size=rng.integers(0, 4), replace=False):
            sequence[i] = rng.choice(list("WPLEGS"))
        inputs.append("".join(sequence))
        target.append(sum(v for (i, c), v in effects.items() if sequence[i] == c))
    return inputs, np.array(target)


def test_mutations_count_substitutions_from_the_consensus():
    mutations = Mutations().fit(["MKTA", "MKTA", "MWTA", "MKTP"])
    assert mutations.consensus == "MKTA" and mutations.mutations == [[1, "W"], [3, "P"]]
    assert mutations.transform(["MWTP", "MKTA", "GKTA"]).tolist() == [[1, 1, 2], [0, 0, 0], [0, 0, 1]]


def test_large_data_leaves_out_models_that_could_not_be_exported():
    kinds = {learner.kind for learner in candidates(3000, 401)}
    assert kinds == {"ridge", "forest", "boosting"}
    assert len(candidates()) == 22


@needs_sandbox
@pytest.mark.parametrize("learner", [Learner("ridge"), Learner("kernel", alpha=0.1), Learner("forest", features=0.5)],
                         ids=lambda l: l.kind)
def test_an_exported_sequence_model_predicts_what_was_fitted(learner):
    inputs, target = variants(80)
    featurizer = Mutations().fit(inputs)
    learner.fit(featurizer.transform(inputs), target)
    probe, _ = variants(30, seed=1)
    js = run_js(export(featurizer, learner, "test"), probe)
    assert np.allclose(js, learner.predict(featurizer.transform(probe)), atol=1e-9)


def test_region_shift_trains_below_the_top_region():
    ((train, test),) = folds(6, "shift", np.array([1, 2, 3, 3, 2, 1.0]))
    assert sorted(test) == [2, 3] and sorted(train) == [0, 1, 4, 5]


@needs_sandbox
@pytest.mark.parametrize("learner", [Learner("ridge"), Learner("pairwise", alpha=10), Learner("kernel", alpha=0.1),
                                     Learner("knn", k=3), Learner("forest", features=0.5), Learner("boosting", depth=3)],
                         ids=lambda l: l.kind)
def test_exported_model_predicts_what_was_fitted(learner):
    inputs, target, _ = oer_like(60)
    featurizer = Composition().fit(inputs)
    x = featurizer.transform(inputs)
    learner.fit(x, target)
    probe, _, _ = oer_like(25, seed=1)
    js = run_js(export(featurizer, learner, "test"), probe)
    assert np.allclose(js, learner.predict(featurizer.transform(probe)), atol=1e-9)


@needs_sandbox
def test_fit_cli_selects_validates_and_writes_a_checked_model(tmp_path):
    inputs, target, regions = oer_like(80)
    train = tmp_path / "train.csv"
    with train.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input", "target", "group", "region"])
        writer.writerows((i, t, i.split("|")[0], r) for i, t, r in zip(inputs, target, regions))
    out = tmp_path / "final_model.js"
    subprocess.run([sys.executable, str(TOOLKIT / "fit.py"), "--train", str(train), "--out", str(out),
                    "--protocol", "shift", "--group-normalize"], check=True, capture_output=True, text=True)
    report = json.loads(Path(f"{out}.report.json").read_text())
    assert report["options"]["protocol"] == "shift" and len(report["candidates"]) == 22
    assert report["validation"]["spearman"] > 0.8  # an additive target is easy to extrapolate
    scores = run_js(out.read_text(), inputs[:5])
    assert all(np.isfinite(scores))


def test_an_adopted_toolkit_passes_the_guard_and_mounts(tmp_path, campaign):
    from ras import guard, harness

    tree = harness.write_stock(tmp_path / "tree")
    (tree / "scripts" / "toolkit").mkdir()
    for name in ("modeling.py", "fit.py"):
        shutil.copyfile(TOOLKIT / name, tree / "scripts" / "toolkit" / name)
    (tree / "skills" / "modeling-toolkit").mkdir()
    shutil.copyfile(TOOLKIT / "SKILL.md", tree / "skills" / "modeling-toolkit" / "SKILL.md")
    assert guard.check(tree, campaign) == []
    remotes = {remote for _, remote in harness.mount(tree, tmp_path / "staging", "AGENTS.md")}
    assert {"/app/harness/scripts", "/app/harness/skills"} <= remotes


def run_fit(tmp_path, config: dict | None, region: bool = False):
    procedure = tmp_path / "procedure"
    shutil.copytree(TOOLKIT, procedure, ignore=shutil.ignore_patterns("__pycache__"))
    if config is not None:
        (procedure / "config.json").write_text(json.dumps(config))
    inputs, target, regions = oer_like(40)
    train = tmp_path / "train.csv"
    with train.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input", "target", "group"] + (["region"] if region else []))
        writer.writerows([i, t, i.split("|")[0], *([r] if region else [])] for i, t, r in zip(inputs, target, regions))
    out = tmp_path / "final_model.js"
    return subprocess.run([sys.executable, "fit.py", "--train", str(train), "--out", str(out)],
                          cwd=procedure, capture_output=True, text=True), out


@needs_sandbox
def test_fit_reads_its_own_config(tmp_path):
    done, out = run_fit(tmp_path, {"group_normalize": True})
    assert done.returncode == 0, done.stderr
    assert json.loads(Path(f"{out}.report.json").read_text())["options"]["group_normalize"] is True


@pytest.mark.parametrize("config, message", [({"protocl": "random"}, "unsupported options"),
                                             ({"protocol": "shift"}, "shift validation needs a region column")])
def test_unsupported_configurations_fail_explicitly(tmp_path, config, message):
    done, out = run_fit(tmp_path, config)
    assert done.returncode != 0 and message in done.stderr and not out.exists()
