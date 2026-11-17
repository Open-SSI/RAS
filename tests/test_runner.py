import json
import sys
from pathlib import Path

import pytest

from ras import harness
from ras.runner import DryRun, HarborRunner, Job, infra_error, job_config

FAKE_HARBOR = (sys.executable, str(Path(__file__).with_name("fake_harbor.py")))


def job(tmp_path, study, campaign, attempts=3) -> Job:
    tree = harness.write_stock(tmp_path / "h0")
    return Job("luna-toy-r1-base", campaign, study.executors["fake"], tree, attempts)


def test_job_config_points_harbor_at_the_harnessed_agent(tmp_path, study, campaign):
    config = job_config(job(tmp_path, study, campaign), tmp_path / "jobs", "docker")
    (agent,) = config["agents"]
    assert agent["import_path"] == "ras.agents:HarnessedCodex"
    assert agent["kwargs"]["harness_dir"] == str((tmp_path / "h0").resolve())
    assert config["n_attempts"] == 3 and config["agent_timeout_multiplier"] == 0.25


def test_infrastructure_errors_are_set_aside_and_rerun(tmp_path, study, campaign):
    runner = HarborRunner(tmp_path / "jobs", "docker", command=FAKE_HARBOR, retry_delay=0)
    (trials,) = runner.run([job(tmp_path, study, campaign)]).values()
    assert len(trials) == 3 and not any(infra_error(trial) for trial in trials)
    (failed,) = (tmp_path / "jobs" / "failed" / "luna-toy-r1-base").iterdir()
    assert json.loads((failed / "result.json").read_text())["exception_info"]


def test_a_trial_the_provider_cut_short_is_rerun(tmp_path, study, campaign, monkeypatch):
    monkeypatch.setenv("FAKE_FAILURE", "ApiOverloadedError")
    runner = HarborRunner(tmp_path / "jobs", "docker", command=FAKE_HARBOR, retry_delay=0)
    (trials,) = runner.run([job(tmp_path, study, campaign)]).values()
    assert len(trials) == 3 and not any(infra_error(trial) for trial in trials)
    (failed,) = (tmp_path / "jobs" / "failed" / "luna-toy-r1-base").iterdir()
    assert json.loads((failed / "result.json").read_text())["exception_info"]["exception_type"] == "ApiOverloadedError"


@pytest.mark.parametrize("exception", ["ApiOverloadedError", "ApiRateLimitError", "CancelledError"])
def test_a_run_the_provider_or_harness_cut_short_is_an_infrastructure_error(tmp_path, exception):
    trial = tmp_path / "trial"
    (trial / "artifacts" / "state").mkdir(parents=True)
    (trial / "artifacts" / "state" / "ledger.json").write_text(json.dumps({"jobs": [{"stage": 1, "status": "completed"}]}))
    (trial / "result.json").write_text(json.dumps(
        {"exception_info": {"exception_type": exception}, "agent_result": {"cost_usd": 0.1, "n_input_tokens": 9000}}))
    assert infra_error(trial)  # it did real work, but did not get the campaign's budget


def test_a_timeout_after_real_work_is_a_run_not_an_infrastructure_error(tmp_path):
    trial = tmp_path / "trial"
    (trial / "artifacts" / "state").mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps(
        {"exception_info": {"exception_type": "AgentTimeoutError"}, "agent_result": {"cost_usd": 1.2}}))
    assert not infra_error(trial)


def test_dry_run_writes_the_config_and_stops(tmp_path, study, campaign, capsys):
    runner = HarborRunner(tmp_path / "jobs", "docker", dry_run=True)
    with pytest.raises(DryRun):
        runner.run([job(tmp_path, study, campaign)])
    assert (tmp_path / "jobs" / "configs" / "luna-toy-r1-base.json").is_file()
    assert "run -c" in capsys.readouterr().out
