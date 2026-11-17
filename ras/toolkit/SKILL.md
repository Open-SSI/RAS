---
name: modeling-procedure
description: Build the delivered final_model.js with this harness's modeling procedure. Use it for the final model of every campaign, after the last round's results are in.
---

# Modeling procedure

This harness owns the procedure that turns measured data into the delivered model. It lives in `/app/harness/scripts/model/`, and the delivered model must be its unmodified output:

```bash
python3 /app/harness/scripts/model/train.py --out /app/model/train.csv
python3 /app/harness/scripts/model/fit.py --train /app/model/train.csv --out /app/final_model.js
asb deliver /app/final_model.js --validate-only
asb deliver /app/final_model.js
```

- `train.py` writes one row per design: the public seed first, then every measured design in the order the lab measured them, with columns `input` (what `predict()` receives), `target` (the objective, larger is better) and, where scores must be comparable across groups, `group`. Do not edit it or `spec.json`: the same rows are rebuilt from the lab's record to check the model.
- `fit.py` is the procedure, and the harness may change it freely: features, learners, ensembles, any domain knowledge. It must be deterministic, take `--train` and `--out`, and write a `final_model.js` that defines `predict(input)`.
- The model is checked after the campaign by refitting `fit.py` on the same rows. If the delivered file is anything else, a hand-edited copy or a model built another way, the run gets no learning credit.

The `fit.py` provided here is a general toolkit. It chooses among ridge, pairwise ridge, RBF kernel ridge, nearest neighbours, random forest and gradient boosting by validation on the rows, and exports the best model under 1 MB with an exact JavaScript twin, checked with deno. `config.json` next to it sets its options: `featurizer` (`composition` for element fractions with an optional `library|` prefix, `tokens` for `|`-separated fields), `group_normalize` (standardize the target within each group), and `protocol` (`random` 5-fold validation, or `shift`, which needs a `region` column). Unknown options are an error. Its report, with every candidate's validation, is written next to the model as `<out>.report.json`.
