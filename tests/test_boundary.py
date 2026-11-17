"""The evidence boundary in the code itself: improvers and selection see the verifier's aggregates
(``verifier.json``), never a record's raw sealed output or anything from a task's hidden test data."""

import ast
from pathlib import Path

RAS = Path(__file__).resolve().parents[1] / "ras"
FORBIDDEN = ("sealed", "metrics.json", "reward.txt", "tests/data", "expected.json")
# Modules that may name them, and why.
ALLOWED = {
    "evidence.py": "writes the sealed output, and summarizes its aggregates into verifier.json",
    "replay.py": "runs the task's own verifier on a refitted model, keeps its output sealed, and passes on only "
                 "evidence.verifier's aggregates",
    "simulate.py": "runs a planner's rounds through the task's own lab and verifier, keeps the verifier's output "
                   "sealed, and passes on only evidence.verifier's aggregates",
    "smoke.py": "prints the smoke trial's verifier output for a person to read",
    "__main__.py": "the smoke command's printout",
    "runner.py": "checks only whether a verifier wrote reward.txt at all, to tell a verifier that never ran",
}


def names(tree: ast.AST) -> set[str]:
    """Identifiers, attributes and string constants in the code, docstrings left out."""
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                  and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            out.add(node.value)
    return out


def test_only_the_record_reads_the_verifier_s_raw_output():
    offenders = {}
    for path in sorted(RAS.rglob("*.py")):
        if path.name in ALLOWED:
            continue
        used = {term for term in FORBIDDEN for name in names(ast.parse(path.read_text()))
                if term.lower() in name.lower()}
        if used:
            offenders[path.relative_to(RAS).as_posix()] = sorted(used)
    assert offenders == {}


def test_exported_evidence_carries_verifier_aggregates_and_nothing_raw(tmp_path, campaign):
    from conftest import write_trial
    from ras import evidence
    record = evidence.record(write_trial(tmp_path / "trial", ["variant_0002", "variant_0003"], ndcg=0.77),
                             tmp_path / "record", campaign, "", "h0")
    exported = evidence.export(record, tmp_path / "workspace" / "evidence" / "run")
    assert not (exported / "sealed").exists() and not (exported / "metrics.json").exists()
    verdict = evidence.read_json(exported / "verifier.json")
    assert verdict["learning"] == 0.77 / 0.5 and verdict["gates"]["ndcg_at_20"]["value"] == 0.77
    assert verdict["discovery"] == 2 / 3 and verdict["bottleneck"] == 2 / 3 and verdict["passed"] is False
    assert all(isinstance(v, (int, float)) for v in verdict["metrics"].values())  # aggregates only, no per-design data
    text = "".join(p.read_text() for p in exported.rglob("*") if p.is_file())
    assert "never copied" not in text  # nothing from the task's tests/ (conftest's secret.csv)
