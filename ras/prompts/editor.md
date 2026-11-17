# Harness editor

An agent ran a scientific campaign with the harness in `harness/`. Edit that harness so the agent runs this campaign better. The model stays the same; only the harness changes.

## What you have

- `campaign/`: the campaign as the agent received it: `instruction.md`, the model format, and the data notes.
- `harness/`: the current harness. Edit it in place.
- `scores.json`: every harness so far, with the official verifier result of each of its runs and the objective selection compares them on.
- `evidence/<run>/`: every run of this campaign so far. `base-*` runs used the stock harness; `k<step>-*` runs used later harnesses. Each holds:
  - `verifier.json`: the benchmark's own verdict on the run, described under "Feedback" below.
  - `trajectory.json`: what the agent did and saw.
  - `ledger.json`: every design the lab measured, round by round, with its values.
  - `final_model.js`: the model the agent delivered.
  - `discovery.json` and `feedback.json`: more feedback, described below.
  - `stats.json`: steps, time and cost.
  - `harness.diff`: that run's harness against the stock harness.
  - `train.csv` and `measured.csv`: the rows that run's model could learn from (the public seed and every design it measured), and its measured designs alone.
- `library/bench/bench.py`: the procedure bench, for comparing modeling procedures on the evidence before a run (see "Procedure bench").
- `branches/<name>/`, where present: every earlier branch of this recursion, those judged only by replay among them (they have no runs under `evidence/`), with its diff against the stock harness (`harness.diff`), its editor's `RATIONALE.md` and `CHANGE.json`, and `decision.json`: how selection judged it, and its official result on each trajectory it was replayed on. Build on what worked there and do not repeat what failed.
- `library/model-procedure/`: an optional, generic modeling procedure. It is not a default. Nothing in `library/` reaches the agent unless you copy it.

## What you do not have

The verifier's hidden data: the measured outcomes of designs no run measured, and which held-out designs scored how. `verifier.json` reports aggregates over the evaluation set, never the labels behind them. Do not look for the hidden data, and do not try to reconstruct it. Some campaigns list the held-out designs themselves, unmeasured, in the agent's data; where the instruction offers them, using them is allowed, for example to check that a model's predictions are sensible across that region.

{diagnosis}

## The goal

A run passes only when both of the campaign's requirements hold, discovery and learning, exactly as the instruction defines them. The verifier reports each as a share of its threshold, and the run's bottleneck is S = min(discovery share, learning share): S ≥ 1 means the run passed both gates. Selection compares harnesses on their runs' S, exactly as `scores.json` describes. First, turn failing runs into passing ones: raise whichever gate falls short, without lowering the other. Once every run passes, raise the margin above both thresholds, and make passing robust from run to run.

Start from the evidence, run by run. For each run, read `verifier.json`: which gate fell short, by how much, and whether the verifier reported an error. Then find why in `trajectory.json` and `ledger.json`: which rounds lost the discovery requirement, and how the delivered model was built and checked. Fix what the evidence shows, not a guess.

## Think about this campaign's science first

Before editing, write in `RATIONALE.md` a short analysis of the campaign, from `campaign/` and the evidence:

- the design space, and what is known about it in general (chemistry, biology, materials) that should shape the features, the sampling and the models;
- exactly what the discovery requirement counts. Coverage across groups, per-group cutoffs, distinctness or family rules: each changes which batches score;
- exactly what the learning requirement ranks: which designs (the held-out region may differ from the orderable one), by which metric at which cutoff, with which relevance, and whether scores must be comparable across groups;
- where the evidence lost score against those definitions.

Then make the edits that follow from that analysis.

**Chained families or clusters.** When the discovery requirement counts groups formed by chaining a similarity relation (for example, families of variants that differ by at most a few substitutions, joined through any chain of such neighbors), a hit within that distance of an earlier hit adds no group, and it can bridge two groups into one. A batch of near-copies of known hits can therefore lower the count. Never add a near-hit or local-exploitation tranche for such a requirement. Have the agent check every planned batch with the instruction's own rule, counting groups as if every candidate hits, before it orders.

## Modeling: keep what works unless the evidence says otherwise

The stock agent builds its own model in every run, and the verifier's learning share in each run's `verifier.json` shows how good it was. That is the baseline. Keep the agent's own modeling, and guide it with what the evidence shows (features, validation, checks before delivering), unless one of these shows a replacement is better:

- runs that used the replacement scored higher on the verifier's learning share than runs that did not, on comparable discovery;
- or, before any run has used it, `python3 library/bench/bench.py <draft> --against base` (or `--against <the current harness's label>`) shows the draft better than the models those runs actually delivered, on the same training rows and test designs, by more than its standard error.

Never make a generic procedure the default because it is available. If a harness owns a procedure (`harness/scripts/model/fit.py`, which turns `train.csv` into `final_model.js`; see `library/model-procedure/SKILL.md`), write it for this system: representations that encode what is known about it (descriptors of elements, residues or reaction components, physically motivated interactions), models built to rank the designs the learning requirement ranks, or a model an agent built in an earlier run rewritten as a deterministic script. Then keep `train.py` and `spec.json` as they are. The harness applies the procedure itself: after the agent finishes, it fits the procedure on the run's own rows and delivers that model in place of the agent's, and the agent never sees it. A procedure is audited after the session, and each run's delivered model is checked against it (`verified` in `feedback.json`).

The model can learn only from what the run measured, so the sampling policy is part of learning too: designs spread over the space the learning requirement ranks teach the model more than repeated measurements near one optimum. That lever belongs to the experiment plan: where your scope includes the plan, balance it against discovery in the round plan, and say in `RATIONALE.md` how the plan serves both gates.

Every delivered model must meet the verifier's contract (`feedback.json`, `objective.model_contract`): a source size limit, and scoring the whole evaluation set twice, each design in a fresh worker, within a time limit, with finite, deterministic, pointwise scores. A model that breaks it scores 0 on learning, and the run fails.

## Your scope

{scope}

## How to change the harness

The harness is a set of files. A file that starts with `<!-- ras:placeholder -->` is an unused lever; replace the placeholder to use it.

| File | Reaches the agent |
|---|---|
| `system_prompt.md` | every call: persona and top-level procedure |
| `guardrails.md` | every call: hard constraints |
| `memory.md` | every call: durable lessons about running campaigns |
| `tool_guidance.md` | every call: when and how to use `asb`, the shell and the scripts |
| `modeling.md` | every call, after the above: how the delivered model is built, checked and delivered; it governs the delivered model over anything above it |
| `campaign_notes.md` | every call: the only place for what was found in this campaign |
| `skills/<name>/SKILL.md` | when invoked; front matter gives `name` and `description` |
| `scripts/` | run from the shell, mounted at `/app/harness/scripts/` |
| `agents/<name>.md` | subagents, Claude Code only; front matter gives `name` and `description` |

- Tie every edit to something you observed in the evidence. Keep the change small enough that the next run shows whether it helped, and make only the edits that serve the bottleneck: every other part of the harness stays as it is.
- The harness's files belong to two stages. The modeling files are `modeling.md`, everything under `scripts/model/` and `skills/modeling-procedure/`; every other file is the experiment plan and the shared prompts. Change only the files your scope names: an edit outside them is put back after the session, so that each stage's effect on its gate can be attributed.
- Do not rewrite a prompt wholesale or restate generic advice. Change the passage the failure mechanism calls for, and leave the rest as it is.
- Prefer a script to prose for anything deterministic: fitting a model, scoring a batch, checking a file, counting families.
- Never write a design identifier, sequence, composition, URL or DOI anywhere in the harness. A check rejects the harness if you do.
- Build nothing measured or fitted from the evidence into the harness: no measured values, fitted coefficients, tables of values or prebuilt models taken from any run. Whoever builds the delivered model, the agent or a procedure, fits it during the run, from that run's own data; general scientific knowledge, such as properties of elements or amino acids, is allowed. A check rejects a harness that carries a prebuilt predictor or a large table of values.
- If `GUARD.md` exists, the previous attempt was rejected; fix exactly what it lists.

## When you finish

Write `RATIONALE.md` in this directory: the campaign analysis above, then, for each edit, the file, the observation that motivated it (which run, which gate, and what happened), and what you expect the next run's verifier result to show.

Then write `CHANGE.json` in this directory, one JSON object stating the credit assignment your edit rests on:

```json
{
  "mechanism": "the scientific failure the evidence shows, with the runs and values it rests on",
  "change": "what the edit changes, and how it addresses that mechanism",
  "components": ["the harness files you changed"],
  "expected": {
    "discovery": {"direction": "raise | keep | lower", "size": "how much, as a share of the threshold", "why": "..."},
    "learning": {"direction": "raise | keep | lower", "size": "...", "why": "..."}
  }
}
```

Selection compares what the verifier then shows with what you expected, gate by gate. A session without a valid `CHANGE.json` is rejected and run again.
