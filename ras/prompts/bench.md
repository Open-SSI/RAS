
## Procedure bench

`library/bench/bench.py` compares modeling procedures on the evidence before any run uses one. It scores by the learning requirement's own ranking metric (`library/bench/ranking.json`), but on designs the evidence runs measured, not on the verifier's evaluation set: use it to choose among drafts, and let the verifier's learning share judge the result.

```bash
python3 library/bench/bench.py /tmp/draft/scripts/model --against base          # the draft vs the stock agents' own models
python3 library/bench/bench.py /tmp/draft/scripts/model --compare harness/scripts/model
python3 library/bench/bench.py harness/scripts/model                            # the current procedure, cross-run
python3 library/bench/bench.py /tmp/draft/scripts/model --runs k1-1 k1-2        # fitted on these runs only
```

`cross-run` fits the procedure on one run's `train.csv`, as that run's agent would, scores the model on the designs the other runs measured and that run did not, and averages over the runs. `--against <label>` puts the procedure beside how that harness's runs actually built their models: for each of its runs, the model the run delivered and the procedure fitted on the same `train.csv` are scored on the same designs, with the paired difference and its standard error. `--compare <dir>` does the same against another procedure. `--pooled` adds cross-validation on all runs' rows together. Treat differences within the standard error as no difference; a procedure tuned until one score rises while another falls has learned the evidence, not the system.

`fit.py` must learn only from the `train.csv` it is given. Build nothing from the evidence into the procedure: no coefficients, fitted parameters, tables of values or lists of designs taken from any run's measurements. Choices such as features, learners and their settings may follow from what the bench and the verifier show, and general scientific knowledge, such as properties of elements or amino acids, may shape the features; the values a model learns must come from the run's own data.

Every harness's procedure is checked for this after the session, as `bench.py <procedure> --prior --no-cross` shows: it is fitted on the evidence with the targets shuffled, four times, and a procedure whose models still rank the evidence designs (mean |Spearman| above 0.2) is rejected.
