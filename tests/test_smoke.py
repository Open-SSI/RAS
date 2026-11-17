from ras import guard, harness, smoke


def test_marker_harness_passes_the_guard_and_mounts_every_kind(tmp_path, campaign):
    tree = smoke.marker_harness(tmp_path / "smoke")
    assert guard.check(tree, campaign) == []
    assert smoke.CODE not in harness.render_instructions(tree)  # only the mounted script holds the code
    remotes = {remote for _, remote in harness.mount(tree, tmp_path / "staging", "CLAUDE.md")}
    assert remotes == {"/app/harness/skills", "/app/harness/scripts", "/app/harness/agents", "/app/CLAUDE.md"}
