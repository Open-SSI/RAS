"""Stands in for `harbor run -c CONFIG`: writes a job's missing trials, and the first trial
it ever writes fails before the agent starts, like a Docker build race, or with FAKE_FAILURE
set, ends with that exception after real work, like a run the model provider cut short."""

import json
import os
import sys
import uuid
from pathlib import Path

config = json.loads(Path(sys.argv[3]).read_text())
job = Path(config["jobs_dir"]) / config["job_name"]
job.mkdir(parents=True, exist_ok=True)
marker = job.parent / "failed-once"
done = [path for path in job.iterdir() if (path / "result.json").is_file()]
for _ in range(config["n_attempts"] - len(done)):
    trial = job / f"trial__{uuid.uuid4().hex[:7]}"
    trial.mkdir()
    if marker.exists():
        result = {"exception_info": None, "agent_result": {"cost_usd": 0.3, "n_input_tokens": 100}}
    else:
        marker.write_text("")
        failure = os.environ.get("FAKE_FAILURE")
        result = ({"exception_info": {"exception_type": failure}, "agent_result": {"cost_usd": 0.1, "n_input_tokens": 50}}
                  if failure else {"exception_info": {"exception_type": "RuntimeError"}, "agent_result": None})
    (trial / "result.json").write_text(json.dumps(result))
