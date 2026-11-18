"""dk brief: a generated agent brief from a synthetic report, a template and notes."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import make_project
from tools.dk.core import briefcmd
from tools.dk.core.config import load


def _fn(name, addr, size, pct):
    return {"name": name, "size": str(size), "fuzzy_match_percent": pct, "metadata": {"virtual_address": str(addr)}}


def _unit(name, src, fns, complete=False, auto=False):
    total = sum(int(f["size"]) for f in fns)
    matched = sum(int(f["size"]) for f in fns if f["fuzzy_match_percent"] == 100.0)
    meta = {"complete": complete, "auto_generated": auto}
    if src:
        meta["source_path"] = src
    return {"name": name, "metadata": meta, "functions": fns,
            "measures": {"total_code": str(total), "matched_code": str(matched), "total_functions": len(fns),
                         "matched_functions": sum(f["fuzzy_match_percent"] == 100.0 for f in fns)}}


REPORT = {"units": [
    _unit("main/a/Small", "src/a/Small.cpp", [_fn("s1", 0x80001000, 0x40, 100.0), _fn("s2", 0x80001040, 0x20, 50.0)]),
    _unit("main/a/Big", "src/a/Big.cpp", [_fn("b1", 0x80002000, 0x400, 90.0), _fn("b2", 0x80002400, 0x100, 99.0),
                                         _fn("b3", 0x80002500, 0x10, 100.0)]),
    _unit("main/a/Done", "src/a/Done.cpp", [_fn("d1", 0x80003000, 0x80, 100.0)], complete=True),
    _unit("main/auto_80004000_text", "", [_fn("fn_80004000", 0x80004000, 0x200, 0.0)], auto=True),
    _unit("main/b/Outside", "src/b/Outside.cpp", [_fn("o1", 0x80009000, 0x80, 10.0)]),
]}

TEMPLATE = """# {{role}} {{slice}}
{{scope}}
{{#sweeper}}
SWEEP
{{undeclared}}
{{units}}
{{/sweeper}}
{{#closer}}
CLOSE
{{near}}
{{/closer}}
NOTES {{notes}}
"""


class BriefTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        make_project(self.root, files={"build/report.json": json.dumps(REPORT), "playbook/agents.md": "# A\n## Loop\n## Roles\n",
                                       "playbook/README.md": "# R\n", "AGENTS.md": "# Agents\n", "rules.md": TEMPLATE})
        (self.root / "dk.toml").write_text((self.root / "dk.toml").read_text() + '\n[brief]\ntemplate = "rules.md"\n')
        self.cfg = load(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def brief(self, *scope, role="sweeper", **kw):
        args = argparse.Namespace(scope=list(scope), role=role, slice=None, max_units=25, functions=6,
                                  no_score=False, out=None, **kw)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch("tools.dk.platforms.gc_mwcc.brief_near",
                                                         lambda cfg, names: ["near link (1 ...):", f"  {names}"]):
            code = briefcmd.run(self.cfg, args)
        return code, out.getvalue()

    def test_a_slice_lists_its_units_biggest_open_first_with_their_open_functions(self):
        code, out = self.brief("0x80000000-0x80008000")
        self.assertEqual(code, 0, out)
        self.assertIn("# sweeper 80000000-80008000", out)
        self.assertNotIn("Outside", out)                       # its .text starts past the slice
        self.assertNotIn("CLOSE", out)
        units = [l for l in out.splitlines() if l.startswith("- `")]
        self.assertEqual(units, ["- `a/Big.cpp`: 1280 / 1296 bytes open, 1/3 exact",
                                 "- `a/Small.cpp`: 32 / 96 bytes open, 1/2 exact"])
        fns = [l.split()[-1] for l in out.splitlines() if l.startswith("    - ")]
        self.assertEqual(fns, ["`b1`", "`b2`", "`s2`"])          # biggest open function first, exact ones left out
        self.assertIn("3 declared unit(s), 1 linked", out)
        self.assertIn("`./dk gen --harvest 0x80000000 0x80008000`", out)
        self.assertIn("0.5 KB in 1 stretch(es)", out)
        self.assertIn("NOTES (none yet)", out)

    def test_closer_role_named_units_and_notes(self):
        notes = self.root / "build" / "dk" / "notes" / "Big.md"
        notes.parent.mkdir(parents=True)
        notes.write_text("b1: try the loop as a do-while\n")
        code, out = self.brief("a/Big.cpp", role="closer")
        self.assertEqual(code, 0, out)
        self.assertIn("# closer Big", out)
        self.assertIn("CLOSE", out)
        self.assertNotIn("SWEEP", out)
        self.assertIn("  ['main/a/Big']", out)                # scored: only the WIP units in scope
        self.assertIn("NOTES b1: try the loop as a do-while", out)

    def test_units_by_name_source_or_stem_and_unknown_ones(self):
        name, found, _, span, unknown = briefcmd.select(self.cfg, REPORT, ["main/a/Big", "src/a/Small.cpp", "a/Done", "x.cpp"])
        self.assertEqual([u["name"] for u in found], ["main/a/Big", "main/a/Small", "main/a/Done"])
        self.assertEqual((name, span, unknown), ("Big-and-2", None, ["x.cpp"]))
        self.assertEqual(self.brief("x.cpp")[0], 2)

    def test_the_output_is_deterministic_and_the_template_is_configurable(self):
        self.assertEqual(self.brief("80000000-80008000"), self.brief("80000000-80008000"))
        code, out = self.brief("80000000-80008000")
        self.assertTrue(out.startswith("# sweeper 80000000-80008000\nSlice 0x80000000-0x80008000"))
        (self.root / "rules.md").write_text("{{nosuch}}\n")
        code, out = self.brief("80000000-80008000")
        self.assertEqual(code, 2)
        self.assertIn("unknown placeholder(s): nosuch", out)

    def test_the_shipped_template_renders_for_both_roles(self):
        (self.root / "dk.toml").write_text((self.root / "dk.toml").read_text().replace('[brief]\ntemplate = "rules.md"\n', ""))
        self.cfg = load(self.root)
        for role in briefcmd.ROLES:
            code, out = self.brief("80000000-80008000", role=role)
            self.assertEqual(code, 0, out)
            self.assertIn("## Rules (every agent)", out)
            self.assertIn("`./dk worktree add 80000000-80008000`", out)
            self.assertNotIn("{{", out)

    def test_render_keeps_only_the_roles_blocks(self):
        text = briefcmd.render("a\n{{#closer}}C {{x}}\n{{/closer}}{{#sweeper}}S {{x}}\n{{/sweeper}}z\n", "sweeper", {"x": "1"})
        self.assertEqual(text, "a\nS 1\nz\n")
        with self.assertRaises(KeyError):
            briefcmd.render("{{y}}", "closer", {})


if __name__ == "__main__":
    unittest.main()
