from pathlib import Path

from ras import guard, harness

from conftest import DESIGNS


def stock(tmp_path: Path) -> Path:
    return harness.write_stock(tmp_path / "tree")


def test_stock_tree_passes(tmp_path, campaign):
    assert guard.check(stock(tmp_path), campaign) == []


def test_design_ids_sequences_and_links_are_rejected(tmp_path, campaign):
    tree = stock(tmp_path)
    design, sequence = next(iter(DESIGNS.items()))
    (tree / "memory.md").write_text(f"Order {design}. Batch sizes stay at 96 and 1000.\n")
    (tree / "scripts" / "top.py").write_text(f"SEQ = '{sequence}'\n")
    (tree / "tool_guidance.md").write_text("See https://example.org for the data.\n")
    problems = guard.scan(tree, campaign)
    assert any(p.startswith("memory.md: names 1 design") for p in problems)
    assert any(p.startswith("scripts/top.py: names 1 design") for p in problems)
    assert "tool_guidance.md: contains a URL or DOI" in problems
    assert len(problems) == 3  # plain numbers are not designs


def test_a_prebuilt_model_or_a_table_of_values_is_rejected_outside_the_procedure(tmp_path, campaign):
    tree = stock(tmp_path)
    (tree / "scripts" / "prebuilt.js").write_text("function predict(design) { return 0.3; }\n")
    (tree / "campaign_notes.md").write_text("Fitted weights: " + ", ".join(f"{i / 7:.4f}" for i in range(250)) + "\n")
    problems = guard.scan(tree, campaign)
    assert any(p.startswith("scripts/prebuilt.js: a prebuilt predictor") for p in problems)
    assert any(p.startswith("campaign_notes.md: carries 250 decimal values") for p in problems)
    (tree / "scripts" / "prebuilt.js").unlink()
    (tree / "campaign_notes.md").write_text("Round 1: 12 per library; weigh Pauling electronegativity 1.91 and 1.88.\n")
    model = tree / "scripts" / "model"
    model.mkdir()
    (model / "fit.py").write_text("TEMPLATE = 'function predict(x) { return %s; }'\n")  # the audit covers procedures
    (model / "export.js").write_text("function predict(x) { return 0; }\n")
    assert guard.scan(tree, campaign) == []


def test_validators(tmp_path, campaign):
    tree = stock(tmp_path)
    (tree / "skills" / "batch").mkdir()
    (tree / "skills" / "batch" / "SKILL.md").write_text("Pick a batch.\n")
    (tree / "agents" / "critic.md").write_text("---\nname: critic\n---\nReview the batch.\n")
    (tree / "scripts" / "broken.py").write_text("def fit(:\n")
    (tree / "guardrails.md").unlink()
    problems = guard.validate(tree)
    assert "skills/batch/SKILL.md: front matter needs name and description" in problems
    assert "agents/critic.md: front matter needs description" in problems
    assert any(p.startswith("scripts/broken.py: does not compile") for p in problems)
    assert any(p.startswith("guardrails.md: missing") for p in problems)
