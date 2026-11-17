# Guidance for coding agents

RAS (Recursive Autonomous Science through Self-Improving Agent Harnessing) runs Harness²-style recursive harness optimization on Autonomous Science Bench campaigns. Read `README.md` first.

- **The verifier is the optimization signal.** Every run is scored by the benchmark's own verifier on its one fixed evaluation set, with the official discovery and learning metrics and thresholds; `evidence.verifier` writes each run's aggregates (`verifier.json`: gate shares, the bottleneck S = min(D/τD, L/τL), pass or fail, any verifier error), and selection (`feedback.choose`) and the improvers use exactly those. No proxy datasets, validation splits or substitute metrics.
- **Aggregates only, never answers.** Improver sessions read only what `evidence.export` copies and `improver.prepare` writes. Never put a record's `sealed/` files, anything from a task's `tests/` (hidden labels, held-out designs beyond what the agent's own data lists), the lab's `campaign/` or `solution/` into a working directory.
- **Every harness passes the guard** (`guard.check`) and the procedure audit before it runs, so no harness carries measured or held-out answers, a prebuilt predictor, or a lookup table.
- **Every delivered model is compared on the same footing.** Each run counts with whatever model it delivered, the stock harness's included; identical runs are one observation, and a deterministic harness is not rerun to estimate a spread it does not have.
- **Base is the stock agent.** With the stock tree, `ras.agents` must upload nothing and behave exactly like Harbor's own agents.
- **The benchmark is frozen** at `benchmark.commit` in `config/study.toml`. Do not edit it during the study; `python -m ras check` must pass before any run.
- **Freeze before the study.** A recursion records `protocol.json` and refuses to resume under changed code, prompts or configuration; change the protocol only between studies.
- **Keep phases resumable.** A phase writes its output under a `.tmp` name and renames it when complete, and is skipped once that output exists.
- **Outputs go to `runs/`**, which is gitignored; never commit them or `.env`.
- **Validate after changes:** `uv run pytest`.
