import shutil
from dataclasses import replace

import pytest

from ras import config, evidence, feedback
from ras.config import Objective

from conftest import DESIGNS, activity, write_trial

BENCHMARK_RUNTIME = config.load().runtime


def test_objective_handles_constraints_and_direction(campaign):
    gated = replace(campaign, objective=Objective("lead", zero_above=("zinc", 1.0)))
    assert feedback.value({"lead": 1.4, "zinc": 0.8}, gated) == 1.4
    assert feedback.value({"lead": 1.4, "zinc": 1.2}, gated) == 0.0
    lower = replace(campaign, objective=Objective("overpotential", higher_is_better=False))
    assert feedback.utility({"overpotential": 0.3}, lower) == -0.3


def test_rounds_report_hits_against_random(campaign):
    designs = list(DESIGNS)[:8]
    ledger = {"jobs": [{"stage": 1, "status": "completed",
                        "results": [{"variant_id": d, "activity": activity(d)} for d in designs]}]}
    (summary,) = feedback.rounds(ledger, campaign)
    assert summary["designs"] == 8 and summary["hits"] == 4 and summary["hit_rate"] == 0.5
    assert summary["random_hit_rate"] == 0.25


def test_write_records_feedback_once_through_the_validator(tmp_path, campaign):
    record = evidence.record(write_trial(tmp_path / "trial", list(DESIGNS)[:6]), tmp_path / "r", campaign, "", "h0")
    calls = []

    def validator(tree, rows, delivered, probe):
        calls.append((len(rows), delivered.name, len(probe)))
        return {"cv": {"ranking": 0.3, "spearman": 0.2}, "delivered": "verified"}

    feedback.write(record, campaign, None, validator)
    feedback.write(record, campaign, None, validator)
    assert calls == [(6, "final_model.js", 12)]
    assert feedback.estimate(record) == (pytest.approx(2 / 3), 0.3)
    assert feedback.verified(record)


@pytest.mark.parametrize("report", [{"cv": {"ranking": 0.9, "spearman": 0.9}, "delivered": "mismatch"},
                                    {"error": "the harness has no modeling procedure (scripts/model/fit.py)"}])
def test_only_a_verified_delivered_model_has_a_learning_estimate(tmp_path, campaign, report):
    record = evidence.record(write_trial(tmp_path / "trial", list(DESIGNS)[:6]), tmp_path / "r", campaign, "", "h0")
    feedback.write(record, campaign, None, lambda *args: report)
    assert feedback.estimate(record)[1] is None


@pytest.mark.skipif(shutil.which("deno") is None or not (BENCHMARK_RUNTIME / "sandbox").is_dir(),
                    reason="needs deno and the Autonomous Science Bench checkout in config/study.toml")
def test_the_contract_check_runs_a_model_as_the_verifier_does(tmp_path, campaign):
    from ras import contract
    from ras.config import Contract
    limited = replace(campaign, contract=Contract(designs=20, timeout_s=60, max_bytes=200))
    model = tmp_path / "final_model.js"
    model.write_text("function predict(s) { return s.length; }\n")
    run = contract.benchmark_runner(BENCHMARK_RUNTIME)
    assert contract.check(model, limited, run)["ok"] and len(contract.inputs(limited)) == 20
    model.write_text("function predict(s) { return NaN; }\n")
    assert "rejects" in contract.check(model, limited, run)["error"]
    model.write_text("// " + "x" * 300 + "\nfunction predict(s) { return 1; }\n")
    assert "bytes" in contract.check(model, limited, run)["error"]


def result(discovery: float, learning: float) -> dict:
    """One run's verifier summary, as evidence.verifier writes it."""
    return {"discovery": discovery, "learning": learning, "bottleneck": min(discovery, learning),
            "passed": discovery >= 1 and learning >= 1}


def test_the_failing_gate_is_the_target_and_s_must_rise():
    gain = feedback.choose([result(1.2, 0.85), result(1.1, 0.95)], [result(1.3, 0.6), result(1.2, 0.7)])
    assert gain["accept"] and gain["rule"] == "bottleneck" and gain["target"] == "learning"
    assert gain["change"] == {"discovery": -0.1, "learning": 0.25}  # a passing discovery may lose margin, not its pass
    lower = feedback.choose([result(1.2, 0.6), result(1.1, 0.6)], [result(1.3, 0.6), result(1.2, 0.7)])
    assert not lower["accept"]


def test_a_passing_gate_must_stay_passing():
    lost = feedback.choose([result(0.95, 1.4), result(0.9, 1.3)], [result(1.3, 0.6), result(1.2, 0.7)])
    assert not lost["accept"] and lost["lost_gate"] == ["discovery"]  # S rose, but discovery fell below its threshold


def test_where_both_gates_fail_the_other_may_not_fall_beyond_its_noise():
    traded = feedback.choose([result(0.6, 0.5), result(0.62, 0.5)], [result(0.5, 0.9), result(0.52, 0.92)])
    assert not traded["accept"] and traded["other_gate_regressed"] == "learning"


def test_once_both_pass_robustness_then_margin_decide():
    steadier = feedback.choose([result(1.1, 1.05), result(1.2, 1.1)], [result(1.5, 1.3), result(1.4, 0.95)])
    assert steadier["rule"] == "robustness" and steadier["accept"]  # every run passes now, though the margin shrank
    wider = feedback.choose([result(1.3, 1.2), result(1.3, 1.25)], [result(1.1, 1.05), result(1.2, 1.1)])
    assert wider["accept"] and wider["gain"] > 0


def test_a_gain_within_the_run_to_run_spread_is_uncertain():
    noisy = feedback.choose([result(1.2, 0.95), result(1.2, 0.55)], [result(1.2, 0.7), result(1.2, 0.72)])
    assert noisy["accept"] and not noisy["sure"]
    clear = feedback.choose([result(1.2, 0.95), result(1.2, 0.93)], [result(1.2, 0.7), result(1.2, 0.72)])
    assert clear["accept"] and clear["sure"]
    deterministic = feedback.choose([result(1.2, 0.75)], [result(1.2, 0.7)])  # one distinct run each: no spread
    assert deterministic["sure"]


def test_the_published_ranking_metrics():
    from ras import ranking
    rows = [{"target": t} for t in (0.0, 1.0, 2.0, 3.0)]
    assert ranking.relevance(rows, {"relevance": "above", "threshold": 1.5}) == [0.0, 0.0, 0.5, 1.5]
    assert ranking.score([0, 1, 2, 3], rows, {"k": 2, "relevance": "above", "threshold": 1.5}) == 1.0
    assert ranking.score([3, 2, 1, 0], rows, {"k": 2, "relevance": "above", "threshold": 1.5}) == 0.0
    assert ranking.score([0, 0, 0, 0], rows, {"k": 1, "relevance": "above", "threshold": 1.5}) == 0.0  # ties keep input order
    # Per-group cutoffs: u = (y - cutoff) / (best - cutoff), so every group's best counts 1 whatever its scale.
    grouped = [{"target": t, "group": g} for g, ts in (("a", (0, 10, 20, 30, 40, 50)), ("b", (0, 1, 2, 3, 4, 5))) for t in ts]
    rel = ranking.relevance(grouped, {"relevance": "group_cutoff", "cutoff_share": 0.5})
    assert rel[5] == rel[11] == 1.0 and rel[0] == rel[6] == 0.0 and rel[4] == pytest.approx(0.5)
    capped = ranking.relevance(grouped, {"relevance": "group_cutoff", "cutoff_share": 0.5, "top_share": 0.34, "cap": True})
    assert capped[5] == capped[4] == 1.0  # at or above the group's second best, capped at 1
