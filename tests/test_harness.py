from pathlib import Path

from ras import harness


def test_stock_tree_renders_and_mounts_nothing(tmp_path: Path):
    tree = harness.write_stock(tmp_path / "h0")
    assert harness.render_instructions(tree) == ""
    assert harness.mount(tree, tmp_path / "staging", "AGENTS.md") == []


def test_real_components_render_in_order_and_mount(tmp_path: Path):
    tree = harness.write_stock(tmp_path / "h1")
    (tree / "guardrails.md").write_text("Never order a design twice.\n")
    (tree / "system_prompt.md").write_text("Plan all rounds before the first order.\n")
    (tree / "scripts" / "fit.py").write_text("print('fit')\n")
    (tree / "skills" / "batch").mkdir()
    (tree / "skills" / "batch" / "SKILL.md").write_text("---\nname: batch\ndescription: pick a batch\n---\n")

    text = harness.render_instructions(tree)
    assert text.index("## System prompt") < text.index("## Guardrails") < text.index("## Harness scripts")
    assert "Memory" not in text

    uploads = dict((remote, local) for local, remote in harness.mount(tree, tmp_path / "staging", "CLAUDE.md"))
    assert set(uploads) == {"/app/harness/skills", "/app/harness/scripts", "/app/CLAUDE.md"}
    assert not (uploads["/app/harness/skills"] / "README.md").exists()  # placeholders stay behind


def test_diff(tmp_path: Path):
    parent = harness.write_stock(tmp_path / "a")
    child = harness.copy(parent, tmp_path / "b")
    assert harness.diff(parent, child) == ""
    (child / "campaign_notes.md").write_text("Round 1 mostly missed.\n")
    assert "+Round 1 mostly missed." in harness.diff(parent, child)
