# Harness composer

Two editors each proposed a change to the harness in `harness/`, and each candidate ran the campaign. Compose the next harness from the evidence.

## What you have

- `campaign/`: the campaign as the agent received it.
- `harness/`: the current harness. Edit it in place into the next one.
- `candidates/<editor>/`: each candidate's `harness/` and its editor's `RATIONALE.md`.
- `scores.json`: every harness so far, with the official verifier result of each of its runs and the objective selection compares them on.
- `evidence/<run>/`: every run so far, with the same files the editors saw, `verifier.json` included. The candidates' runs are `k<step>-<editor>*`. The current harness's runs are the reference: those of the harness `scores.json` names as current, or the `base-*` runs at the first step.
- `library/bench/bench.py` and `library/model-procedure/`: as the editors had them.

## What you do not have

The verifier's hidden data: the measured outcomes of designs no run measured, and which held-out designs scored how. `verifier.json` reports aggregates, never the labels behind them. Do not look for the hidden data. Some campaigns list the held-out designs themselves, unmeasured, in the agent's data; where the instruction offers them, using them is allowed.

{diagnosis}

## The goal

A run passes only when both of the campaign's requirements hold, discovery and learning, exactly as the instruction defines them. The verifier reports each as a share of its threshold, and the run's bottleneck is S = min(discovery share, learning share): S ≥ 1 means the run passed both gates. Your harness is one branch beside the two candidates: selection keeps whichever branch beats the current harness by the most on the rule in `scores.json`, if any, so compose only what adds to the better candidate; adopting it unchanged is not needed, since it is already a branch. First turn failing runs into passing ones; once every run passes, raise the margin and make passing robust.

## How to compose

- Compare each candidate's runs with the reference runs and with each other, gate by gate, on `verifier.json`, and explain the differences from the trajectories and ledgers.
- An edit may enter the next harness only if the runs of the candidate that carried it did not fall below the reference on either gate, or you can show from the trajectories that the fall had another cause. Never import an edit from a candidate whose runs show the failure that edit invites; for example, a near-hit tranche from a candidate whose runs lost families.
- Then merge the edits the evidence supports and revise edits that conflict. Each candidate is already a branch selection can keep, so a composed harness is worth running only if it combines what both candidates did well; if one candidate simply beat the other, hand back the current harness unchanged and let selection keep that candidate.
- Changes are complementary when each raised a different gate and neither lowered the other: for example, an edit to the experiments whose runs raised discovery and an edit to the model whose runs raised learning. Merge complementary changes when the evidence supports both; say in `DECISIONS.md` which gate each merged change is for.
- Each candidate ran twice, but runs still vary. Prefer edits whose effect shows in both runs and in the trajectory, not only in one number.
- Keep the modeling the evidence supports: the agent's own modeling stays unless runs with a replacement scored higher on the verifier's learning share, or the bench's paired comparison (`bench.py <procedure> --against <label>`) favors the replacement beyond its error.
- If the discovery requirement counts groups formed by chaining a similarity relation, the composed harness must not order near-copies of known hits; it should have the agent count the groups a planned batch would form, by the instruction's own rule, before ordering.
- Keeping the current harness is a valid decision when neither candidate beat it on the bottleneck.
- The editing rules still hold: tie edits to evidence; never write a design identifier, sequence, composition, URL or DOI; build nothing measured or fitted from the evidence into the harness (no measured values, fitted coefficients, tables of values or prebuilt models); keep what was found in this campaign in `campaign_notes.md`; front matter for skills and subagents. If `GUARD.md` exists, fix exactly what it lists.

## When you finish

Write `DECISIONS.md` in this directory. For each edit you kept, revised or dropped, give the observation behind the decision, with the runs' verifier results it rests on.
