import json
import tempfile
import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc import blockers as bl
from tools.dk.platforms.gc_mwcc import route

BULLET = """\
# opus grind
- fn_80636DC8/E98/7058 (unk/80636410, ~210B, same blocker) start 10%: GAVE UP 99.5% ~8 min. reach: temp-temp rule; 9 shapes.
- fn_80637538 (unk/80636410, 188B) start 10%: LANDED 4a2a3fb1b, ~4 min. cross-tier local result before fe-temp.
- vf13__12SwitchButtonFiii / SwitchButton_EnsureTwoPlayers__FP3Wnd (fe/SwitchButton) start 10%: GAVE UP 99.96/99.95% ~4 min. temp-temp inside inline.
- fn_804DB518 (ai/X, 1B) start 99.1%: looked only (1 min): offsets. Not attempted.
  -> unit now 10/10: ACCEPTED
## Summary (16:14)
"""
PIPE = """\
fn_805A13EC + fn_805A149C | 10% | gave up | 4 | 'r = f(h) != 0' srwi r0 vs r31
fn_8042F928__12CameraTarget | 10% | skipped | 1 | register-numbering only
fn_804FB1B4__16LightsReflection | 10% | skipped | 3 | exact only via byte-offset cast (invented cast, not landed)
SUMMARY | start 10% | 4 landed | x | y
"""


class ParseTest(unittest.TestCase):
    def test_names(self):
        self.assertEqual(bl.names_of("fn_80636DC8/E98/7058 (unk/x, 210B)"), ["fn_80636DC8", "fn_80636E98", "fn_80637058"])
        self.assertEqual(bl.names_of("a_b / c_d + e_f, g_h"), ["a_b", "c_d", "e_f", "g_h"])
        self.assertEqual(bl.names_of("fn_8052D818/fn_8052D8AC"), ["fn_8052D818", "fn_8052D8AC"])

    def test_bullet_and_pipe_logs(self):
        rows = bl.parse_log(BULLET, "v1", "T/x.md", "2026-11-06")
        got = {r["function"]: (r["blocker"], r["outcome"]) for r in rows}
        self.assertEqual(got["fn_80636E98"], ("temp-temp", "gave-up"))
        self.assertEqual(got["fn_80637538"], ("cross-tier", "landed"))
        self.assertEqual(got["SwitchButton_EnsureTwoPlayers__FP3Wnd"][0], "temp-temp")
        self.assertEqual(got["fn_804DB518"][1], "skipped")
        self.assertEqual(len(rows), 7)
        self.assertEqual(next(r for r in rows if r["function"] == "fn_80636DC8")["best"], 99.5)
        rows = bl.parse_log(PIPE, "v1", "T/y.md", "2026-11-06")
        got = {r["function"]: (r["blocker"], r["outcome"]) for r in rows}
        self.assertEqual(got["fn_805A149C"], ("compare-form", "gave-up"))
        self.assertEqual(got["fn_8042F928__12CameraTarget"], ("register-only", "skipped"))
        self.assertEqual(got["fn_804FB1B4__16LightsReflection"][0], "device-only")
        self.assertEqual(len(rows), 4)                       # SUMMARY line left out

    def test_backfill_is_idempotent_and_latest_row_wins(self):
        with tempfile.TemporaryDirectory() as d:
            ddir = Path(d)
            log = ddir / "log.md"
            log.write_text(BULLET)
            self.assertEqual(bl.backfill(ddir, [log], "v1", "2026-11-06"), (7, 0))
            self.assertEqual(bl.backfill(ddir, [log], "v1", "2026-11-06"), (0, 7))
            bl.append(ddir, [bl.make_row("fn_80636E98", "shape", "v2", "2026-11-07", "gave-up")])
            self.assertEqual(bl.load(ddir)["fn_80636E98"]["playbook_version"], "v2")
            with self.assertRaises(ValueError):
                bl.append(ddir, [bl.make_row("f", "Bad Tag", "v", "d", "gave-up")])
            with self.assertRaises(ValueError):
                bl.append(ddir, [bl.make_row("f", "shape", "v", "d", "nope")])
            (ddir / bl.FILE).write_text("not json\n" + json.dumps({"function": "f", "blocker": "shape"}) + "\n")
            self.assertEqual(list(bl.load(ddir)), ["f"])

    def test_frozen_functions(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "bench.json").write_text(json.dumps({"functions": [{"function": "a"}]}))
            (Path(d) / "heldout.json").write_text(json.dumps({"functions": [{"function": "b"}]}))
            self.assertEqual(bl.frozen_functions(Path(d)), {"a", "b"})
            self.assertEqual(bl.frozen_functions(Path(d) / "none"), set())


def row(**kw):
    r = {"function": "f", "size": 0x100, "fuzzy": 95.0, "kind": "shape", "class": "near-shape", "blocker": None,
         "playbook_version": None, "attempts": 0}
    r.update(kw)
    return r


class FreshTest(unittest.TestCase):
    def test_pool_rules(self):
        f = bl.is_fresh
        self.assertTrue(f(row(), "v2", set()))
        self.assertTrue(f(row(size=2048, kind="register", **{"class": "giant"}), "v2", set()))
        self.assertFalse(f(row(size=2049), "v2", set()))
        self.assertFalse(f(row(fuzzy=79.0), "v2", set()))
        self.assertFalse(f(row(kind="scheduling"), "v2", set()))
        self.assertFalse(f(row(kind="data", **{"class": "unit-data"}), "v2", set()))
        self.assertFalse(f(row(), "v2", {"f"}))                                  # frozen
        self.assertFalse(f(row(blocker="temp-temp", playbook_version="v2"), "v2", set()))   # tried under this playbook
        self.assertTrue(f(row(blocker="temp-temp", playbook_version="v1"), "v2", set()))    # a new rule since
        self.assertTrue(f(row(attempts=3), "v2", set()))                         # local-model attempts do not count...
        self.assertFalse(f(row(attempts=3), "v2", set(), strict=True))           # ...unless strict
        self.assertEqual(bl.depletion(300, 150), 2.0)

    def test_route_rows_carry_blocker_version_and_fresh(self):
        fns = [{"unit": "u", "source": "src/u.cpp", "function": n, "address": i, "size": 0x40, "fuzzy": 95.0}
               for i, n in enumerate(("a", "b", "c", "d"))]
        blk = {"a": {"blocker": "temp-temp", "playbook_version": "v1"}, "b": {"blocker": "cross-tier", "playbook_version": "v2"}}
        rows = {r["function"]: r for r in route.route_rows(fns, {}, {}, {}, blockers=blk, version="v2", frozen_fns={"c"},
                                                                  frozen={("u", "d"): "gate"})}
        self.assertEqual((rows["a"]["blocker"], rows["a"]["playbook_version"], rows["a"]["fresh"]), ("temp-temp", "v1", False))
        self.assertEqual(rows["b"]["fresh"], False)
        self.assertEqual(rows["c"]["blocker"], None)
        self.assertEqual((rows["c"]["fresh"], rows["d"]["fresh"], rows["d"]["worker"]), (False, False, "frozen"))


class VersionTest(unittest.TestCase):
    def test_version_follows_matching_notes_and_rules(self):
        from tests.helpers import make_project
        with tempfile.TemporaryDirectory() as d:
            cfg = make_project(Path(d))
            (Path(d) / "playbook").mkdir(exist_ok=True)
            m = Path(d) / "playbook" / "matching.md"
            m.write_text("one")
            v1 = bl.playbook_version(cfg)
            self.assertEqual(v1, bl.playbook_version(cfg))
            m.write_text("two")
            self.assertNotEqual(v1, bl.playbook_version(cfg))


if __name__ == "__main__":
    unittest.main()
