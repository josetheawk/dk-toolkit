from tools.dk.platforms.gc_mwcc.gencmd import stretches


def test_stretches_group_by_gap_between_declared_units():
    declared = [(0x100, 0x200), (0x300, 0x400)]
    found = [(0x250, "b"), (0x10, "a"), (0x210, "c"), (0x500, "d")]
    got = {(lo, hi): names for lo, hi, names in stretches(found, declared)}
    assert got == {(0x200, 0x300): ["c", "b"], (0, 0x100): ["a"], (0x400, 0): ["d"]}


def test_quoted_includes_follow_new_headers(tmp_path):
    from types import SimpleNamespace
    from tools.dk.platforms.gc_mwcc.matching import quoted_includes
    (tmp_path / "include" / "a").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "include" / "a" / "x.h").write_text('#include "a/y.h"\n')
    (tmp_path / "include" / "a" / "y.h").write_text('#include <stdio.h>\n')
    (tmp_path / "src" / "u.cpp").write_text('#include "a/x.h"\n#include "missing.h"\n')
    got = sorted(quoted_includes(SimpleNamespace(root=tmp_path), tmp_path / "src" / "u.cpp"))
    assert got == ["include/a/x.h", "include/a/y.h"]


def test_base_path_follows_the_offset_zero_chain():
    from types import SimpleNamespace
    from tools.dk.platforms.gc_mwcc.gencmd import Context
    recs = [{"name": "Root", "rtti": 1, "bases": []},
            {"name": "Mid", "rtti": 2, "bases": [[1, 0]]},
            {"name": "Other", "rtti": 3, "bases": []},
            {"name": "Leaf", "rtti": 4, "bases": [[3, 64], [1, 0], [2, 0]]}]
    ctx = SimpleNamespace(class_records=recs, classes={c["name"]: c for c in recs})
    assert Context.base_path(ctx, "Leaf", "Root") == ["Root", "Mid", "Leaf"]
    assert Context.base_path(ctx, "Leaf", "Mid") == ["Mid", "Leaf"]
    assert Context.base_path(ctx, "Leaf", "Other") is None        # not at offset 0
    assert Context.base_path(ctx, "Mid", "Leaf") is None


def test_ranged_templates_cover_library_code_past_code_end():
    from types import SimpleNamespace
    from tools.dk.platforms.gc_mwcc.gencmd import Context
    ctx = SimpleNamespace(templates=[(0x900, 0xA00, "gui/G.cpp")], template="game/T.cpp", code_end=0x800)
    assert Context.template_for(ctx, 0x100) == "game/T.cpp"
    assert Context.template_for(ctx, 0x950) == "gui/G.cpp"
    assert Context.template_for(ctx, 0x850) is None              # other library code
