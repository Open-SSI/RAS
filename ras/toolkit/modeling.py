"""Modeling toolkit: featurize designs, choose a learner by validation on measured data, export final_model.js.

Every featurizer and learner has an exact JavaScript twin, so the delivered model
predicts what the validated model predicted. numpy throughout, plus scikit-learn to
grow the tree ensembles; tree comparisons run in float32 in both languages, as
scikit-learn's do.

  featurizers  composition   element fractions, with an optional "category|" prefix one-hot encoded
               tokens        "a|b|c" fields, one-hot encoded by position
               mutations     aligned sequences: which residues differ from the training consensus, and how many
  learners     ridge, ridge with pairwise terms, RBF kernel ridge (a GP's mean), k nearest neighbours,
               random forest, gradient boosting
  validation   random k-fold, or region shift: train on lower regions, test on the highest
  metric       top-tail NDCG@20 (relevance above the data's upper quartile), then Spearman
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

AMOUNT = re.compile(r"([A-Z][a-z]?)-?([0-9]*\.?[0-9]+)")
JS_AMOUNT = r"/([A-Z][a-z]?)-?([0-9]*\.?[0-9]+)/g"


class Composition:
    """Element fractions of the part after the first "|", and a one-hot of the part before it."""

    name = "composition"

    def fit(self, inputs: list[str]) -> Composition:
        parsed = [self._parse(text) for text in inputs]
        self.elements = sorted({element for _, amounts in parsed for element in amounts})
        self.prefixes = sorted({prefix for prefix, _ in parsed if prefix})
        return self

    def transform(self, inputs: list[str]) -> np.ndarray:
        rows = []
        for prefix, amounts in map(self._parse, inputs):
            total = sum(amounts.values()) or 1.0
            rows.append([amounts.get(e, 0.0) / total for e in self.elements] + [float(prefix == p) for p in self.prefixes])
        return np.array(rows, float)

    def params(self) -> dict:
        return {"elements": self.elements, "prefixes": self.prefixes}

    @staticmethod
    def _parse(text: str) -> tuple[str, dict[str, float]]:
        prefix, _, body = str(text).partition("|") if "|" in str(text) else ("", "", str(text))
        amounts: dict[str, float] = {}
        for element, value in AMOUNT.findall(body):
            amounts[element] = amounts.get(element, 0.0) + float(value)
        return prefix, amounts

    JS = f"""function rasFeatures(input) {{
  const text = String(input);
  const cut = text.indexOf("|");
  const prefix = cut >= 0 ? text.slice(0, cut) : "";
  const body = cut >= 0 ? text.slice(cut + 1) : text;
  const amounts = {{}};
  let total = 0;
  for (const [, element, value] of body.matchAll({JS_AMOUNT})) {{
    amounts[element] = (amounts[element] || 0) + Number(value);
    total += Number(value);
  }}
  const x = RAS.featurizer.elements.map((e) => (amounts[e] || 0) / (total || 1));
  for (const p of RAS.featurizer.prefixes) x.push(prefix === p ? 1 : 0);
  return x;
}}"""


class Tokens:
    """One-hot of each "|"-separated field, by position."""

    name = "tokens"

    def fit(self, inputs: list[str]) -> Tokens:
        self.vocabulary = sorted({f"{i}={t}" for text in inputs for i, t in enumerate(str(text).split("|"))})
        return self

    def transform(self, inputs: list[str]) -> np.ndarray:
        index = {token: i for i, token in enumerate(self.vocabulary)}
        rows = np.zeros((len(inputs), len(self.vocabulary)))
        for row, text in enumerate(inputs):
            for i, t in enumerate(str(text).split("|")):
                if f"{i}={t}" in index:
                    rows[row, index[f"{i}={t}"]] = 1.0
        return rows

    def params(self) -> dict:
        return {"vocabulary": self.vocabulary}

    JS = """function rasFeatures(input) {
  const present = new Set(String(input).split("|").map((t, i) => `${i}=${t}`));
  return RAS.featurizer.vocabulary.map((token) => (present.has(token) ? 1 : 0));
}"""


class Mutations:
    """Aligned sequences as mutations from the consensus of the training inputs: one indicator per
    residue at a position (the most frequent, up to a cap), and the number of positions that differ."""

    name = "mutations"
    LIMIT = 400

    def fit(self, inputs: list[str]) -> Mutations:
        texts = [str(text) for text in inputs]
        length = max(map(len, texts))
        self.consensus = "".join(min(Counter(t[i] for t in texts if i < len(t)).items(), key=lambda kv: (-kv[1], kv[0]))[0]
                                 for i in range(length))
        counts = Counter((i, c) for t in texts for i, c in enumerate(t) if i >= length or c != self.consensus[i])
        self.mutations = [[i, c] for (i, c), _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: self.LIMIT]]
        return self

    def transform(self, inputs: list[str]) -> np.ndarray:
        rows = np.zeros((len(inputs), len(self.mutations) + 1))
        for row, text in enumerate(map(str, inputs)):
            for j, (i, c) in enumerate(self.mutations):
                rows[row, j] = float(i < len(text) and text[i] == c)
            rows[row, -1] = sum(i >= len(self.consensus) or c != self.consensus[i] for i, c in enumerate(text))
        return rows

    def params(self) -> dict:
        return {"consensus": self.consensus, "mutations": self.mutations}

    JS = """function rasFeatures(input) {
  const text = String(input);
  const f = RAS.featurizer;
  const x = f.mutations.map(([i, c]) => (i < text.length && text[i] === c ? 1 : 0));
  let differ = 0;
  for (let i = 0; i < text.length; i++) if (i >= f.consensus.length || text[i] !== f.consensus[i]) differ++;
  x.push(differ);
  return x;
}"""


FEATURIZERS = {cls.name: cls for cls in (Composition, Tokens, Mutations)}
MAX_STORED = 60_000  # numbers a kernel or neighbour model may store, to keep final_model.js under 1 MB
MAX_PAIRWISE = 30    # features above which pairwise terms are not tried


@dataclass
class Learner:
    """One learner and its hyperparameters, fitted on standardized features."""

    kind: str                       # ridge | pairwise | kernel | knn | forest | boosting
    alpha: float = 1.0              # ridge penalty, or kernel ridge's noise term
    scale: float = 1.0              # kernel length scale, in multiples of the median training distance
    k: int = 5                      # neighbours
    features: float = 1.0           # forest: share of features tried at each split
    depth: int = 3                  # boosting: tree depth
    state: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return {"ridge": f"ridge(alpha={self.alpha:g})", "pairwise": f"ridge-pairwise(alpha={self.alpha:g})",
                "kernel": f"kernel-ridge(scale={self.scale:g}, alpha={self.alpha:g})", "knn": f"knn(k={self.k})",
                "forest": f"random-forest(features={self.features:g})",
                "boosting": f"gradient-boosting(depth={self.depth})"}[self.kind]

    def copy(self) -> Learner:
        return Learner(self.kind, self.alpha, self.scale, self.k, self.features, self.depth)

    def fit(self, x: np.ndarray, y: np.ndarray) -> Learner:
        mean, std = x.mean(0), x.std(0)
        std[std == 0] = 1.0
        z = (x - mean) / std
        offset = float(y.mean())
        state = {"mean": mean.tolist(), "std": std.tolist(), "offset": offset}
        if self.kind in ("ridge", "pairwise"):
            phi = _pairwise(z) if self.kind == "pairwise" else z
            state["w"] = np.linalg.solve(phi.T @ phi + self.alpha * np.eye(phi.shape[1]), phi.T @ (y - offset)).tolist()
        elif self.kind == "kernel":
            distances = np.sqrt(_squared_distances(z, z))
            length = self.scale * (float(np.median(distances[np.triu_indices(len(z), 1)])) or 1.0)
            kernel = np.exp(-(distances ** 2) / (2 * length ** 2))
            state |= {"z": z.tolist(), "length": length,
                      "a": np.linalg.solve(kernel + self.alpha * np.eye(len(z)), y - offset).tolist()}
        elif self.kind == "knn":
            state |= {"z": z.tolist(), "y": y.tolist()}
        else:
            self._model = _ensemble(self).fit(z, y)
            trees = [e[0] if self.kind == "boosting" else e for e in self._model.estimators_]
            state["trees"] = [_tree(e.tree_) for e in trees]
            if self.kind == "boosting":
                state |= {"init": float(self._model.init_.constant_.ravel()[0]), "rate": self._model.learning_rate}
        self.state = state
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        s = self.state
        z = (x - np.array(s["mean"])) / np.array(s["std"])
        if self.kind in ("ridge", "pairwise"):
            phi = _pairwise(z) if self.kind == "pairwise" else z
            return s["offset"] + phi @ np.array(s["w"])
        if self.kind in ("forest", "boosting"):
            return self._model.predict(z)
        d2 = _squared_distances(z, np.array(s["z"]))
        if self.kind == "kernel":
            return s["offset"] + np.exp(-d2 / (2 * s["length"] ** 2)) @ np.array(s["a"])
        d = np.sqrt(d2)
        out = []
        for row in d:
            nearest = sorted(range(len(row)), key=lambda i: (row[i], i))[: self.k]
            weights = np.array([1 / (row[i] + 1e-9) for i in nearest])
            out.append(float(weights @ np.array(s["y"])[nearest] / weights.sum()))
        return np.array(out)

    JS = """function rasLearn(x) {
  const m = RAS.learner;
  const z = x.map((v, i) => (v - m.mean[i]) / m.std[i]);
  if (m.kind === "ridge" || m.kind === "pairwise") {
    const phi = m.kind === "pairwise" ? rasPairwise(z) : z;
    return phi.reduce((sum, v, i) => sum + v * m.w[i], m.offset);
  }
  if (m.kind === "forest") return m.trees.reduce((sum, t) => sum + rasTree(t, z), 0) / m.trees.length;
  if (m.kind === "boosting") return m.trees.reduce((sum, t) => sum + m.rate * rasTree(t, z), m.init);
  const d2 = m.z.map((row) => row.reduce((sum, v, i) => sum + (z[i] - v) ** 2, 0));
  if (m.kind === "kernel") {
    return d2.reduce((sum, d, i) => sum + m.a[i] * Math.exp(-d / (2 * m.length ** 2)), m.offset);
  }
  const order = d2.map((d, i) => [Math.sqrt(d), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, m.k);
  let num = 0, den = 0;
  for (const [d, i] of order) { const w = 1 / (d + 1e-9); num += w * m.y[i]; den += w; }
  return num / den;
}
function rasTree(t, z) {
  let n = 0;
  while (t.l[n] !== -1) n = Math.fround(z[t.f[n]]) <= t.t[n] ? t.l[n] : t.r[n];
  return t.v[n];
}
function rasPairwise(z) {
  const out = z.slice();
  for (let i = 0; i < z.length; i++) for (let j = i + 1; j < z.length; j++) out.push(z[i] * z[j]);
  return out;
}"""


def _ensemble(learner: Learner):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    if learner.kind == "forest":
        return RandomForestRegressor(n_estimators=50, max_features=learner.features, max_leaf_nodes=64,
                                     min_samples_leaf=3, random_state=0)
    return GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=learner.depth, random_state=0)


def _tree(tree) -> dict:
    """One fitted scikit-learn tree as flat arrays: children, split feature and threshold, leaf value."""
    return {"l": tree.children_left.tolist(), "r": tree.children_right.tolist(), "f": tree.feature.tolist(),
            "t": tree.threshold.tolist(), "v": tree.value[:, 0, 0].tolist()}


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared distances, row by row so memory stays at len(a) x len(b)."""
    return np.array([((b - row) ** 2).sum(1) for row in a]).reshape(len(a), len(b))


def _pairwise(z: np.ndarray) -> np.ndarray:
    pairs = [z[:, i] * z[:, j] for i in range(z.shape[1]) for j in range(i + 1, z.shape[1])]
    return np.column_stack([z, *pairs]) if pairs else z


def candidates(rows: int = 0, features: int = 0) -> list[Learner]:
    """Every learner to validate; for large data, those whose model would not fit in final_model.js
    (kernel and neighbours store every training row) or whose pairwise terms would explode are left out."""
    stores = rows * features <= MAX_STORED
    return ([Learner("ridge", alpha=a) for a in (0.1, 1, 10)]
            + ([Learner("pairwise", alpha=a) for a in (1, 10, 100)] if features <= MAX_PAIRWISE else [])
            + ([Learner("kernel", alpha=a, scale=s) for s in (0.5, 1, 2) for a in (0.01, 0.1, 1)] if stores else [])
            + ([Learner("knn", k=k) for k in (3, 5, 10)] if stores else [])
            + [Learner("forest", features=f) for f in (1.0, 0.5)]
            + [Learner("boosting", depth=d) for d in (2, 3)])


def tail(y: np.ndarray) -> float:
    """Relevance starts above the upper quartile, or for discrete targets topped by it, above the next value down."""
    threshold = float(np.quantile(y, 0.75))
    below = y[y < y.max()]
    return threshold if y.max() > threshold or not len(below) else float(below.max())


def ndcg(prediction: np.ndarray, truth: np.ndarray, threshold: float, k: int = 20) -> float:
    k = min(k, len(truth))
    order = np.lexsort((np.arange(len(truth)), -prediction))[:k]
    relevance = np.maximum(truth - threshold, 0.0)
    discount = 1 / np.log2(np.arange(k) + 2)
    best = (np.sort(relevance)[::-1][:k] * discount).sum()
    return float((relevance[order] * discount).sum() / best) if best else 0.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = (np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float) for v in (a, b))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def folds(n: int, protocol: str, regions: np.ndarray | None = None, k: int = 5, seed: int = 0):
    if protocol == "shift":
        if regions is None:
            raise ValueError("region-shift validation needs a region column")
        top = regions.max()
        return [(np.where(regions < top)[0], np.where(regions == top)[0])]
    order = np.random.default_rng(seed).permutation(n)
    return [(np.setdiff1d(order, part), part) for part in np.array_split(order, k)]


def validate(learner: Learner, x: np.ndarray, y: np.ndarray, splits, score=None) -> tuple[float, float]:
    """The ranking score and Spearman of out-of-fold predictions: ``score(prediction, row indices)`` where
    given (the campaign's own ranking metric), else top-tail NDCG@20."""
    predicted, truth, index = [], [], []
    for train, test in splits:
        predicted.append(learner.copy().fit(x[train], y[train]).predict(x[test]))
        truth.append(y[test])
        index.append(test)
    p, t, i = np.concatenate(predicted), np.concatenate(truth), np.concatenate(index)
    ranked = score(p, i) if score is not None else ndcg(p, t, tail(y))
    return ranked, spearman(p, t)


def select(x: np.ndarray, y: np.ndarray, protocol: str, regions: np.ndarray | None = None,
           score=None) -> list[tuple[Learner, dict]]:
    """Every candidate with its validation, best first."""
    splits = folds(len(y), protocol, regions)
    table = []
    for learner in candidates(*x.shape):
        ranked, rho = validate(learner, x, y, splits, score)
        table.append((learner, {"learner": learner.label, "ranking": ranked, "spearman": rho}))
    return sorted(table, key=lambda item: (-item[1]["ranking"], -item[1]["spearman"]))


def export(featurizer, learner: Learner, note: str) -> str:
    """final_model.js: the featurizer and learner, as one deterministic, pointwise predict()."""
    model = {"featurizer": featurizer.params(), "learner": {"kind": learner.kind, "k": learner.k, **learner.state}}
    return (f"// Generated by the RAS modeling toolkit: {note}\n"
            f"const RAS = {json.dumps(model, separators=(',', ':'))};\n"
            f"{featurizer.JS}\n{Learner.JS}\n"
            "function predict(input) {\n  return rasLearn(rasFeatures(input));\n}\n")
