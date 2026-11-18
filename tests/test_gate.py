"""The frozen gate sets (gate.py): grind-log parsing, seeded picks, the grind-queue bar and the leakage hashes.
Synthetic fixtures only."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.dk.platforms.gc_mwcc import gate, llmdata, llmgrind, route

DEF = "int fn_80001000(Car* c) {\n    int a = c->m4;\n    return a + 1;\n}"
RENAMED = DEF.replace("fn_80001000", "Car::Foo")


def rec(unit, fn, **kw):
    return {"unit": unit, "function": fn, "address": 0x80001000, "size": 0x40, "fuzzy": 99.0, "asm_shape": "S_" + fn, **kw}


class DataDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.environ.get("DK_LLM_DATA")
        os.environ["DK_LLM_DATA"] = self.tmp.name
        self.cfg = SimpleNamespace(root=Path(self.tmp.name))
        self.d = Path(self.tmp.name)
        (self.d / gate.GATE_NAME).mkdir()

    def tearDown(self):
        if self.old is None:
            os.environ.pop("DK_LLM_DATA", None)
        else:
            os.environ["DK_LLM_DATA"] = self.old
        self.tmp.cleanup()

    def write_gate(self):
        doc = {"functions": [rec("u/a", "fn_A", before=DEF, hashes=sorted(gate.code_hashes(DEF)))]}
        (self.d / gate.GATE_NAME / "gate.json").write_text(json.dumps(doc))
        (self.d / "near-sample-60.json").write_text(json.dumps({"functions": [rec("u/b", "fn_B")]}))


class ParseTest(unittest.TestCase):
    def test_giveups_only_and_slash_lists(self):
        text = ("fn_804D2758 | 10% | LANDED 3e6 | 3 | note\n"
                "fn_8050EE74 | 10% | gave up | 6 | tried it\n"
                "- fn_80636DC8/E98/7058 (unk/80636410, ~210B) start 10%: GAVE UP 99.5% ~8 min.\n"
                "- vf13__12SwitchButtonFiii (fe/SwitchButton) start 10%: LANDED 1\n")
        self.assertEqual(gate.giveup_names(text), ["fn_8050EE74", "fn_80636DC8", "fn_80636E98", "fn_80637058"])

    def test_resolve_by_name_then_address_prefix(self):
        by_name = {"fn_80001000__Q23Foo3BarFv": {}, "plain": {}}
        by_addr = {0x80001000: "fn_80001000__Q23Foo3BarFv"}
        self.assertEqual(gate.resolve(["plain", "fn_80001000", "fn_80009999", "zzz"], by_name, by_addr),
                         ["plain", "fn_80001000__Q23Foo3BarFv"])

    def test_pick_is_seeded_and_sorted(self):
        pool = [f"f{i}" for i in range(50)]
        a = gate.pick(pool, 10, 7)
        self.assertEqual(a, gate.pick(list(reversed(pool)), 10, 7))
        self.assertEqual(a, sorted(a))
        self.assertNotEqual(a, gate.pick(pool, 10, 8))
        self.assertEqual(len(gate.pick(pool[:3], 10, 7)), 3)


class HashTest(unittest.TestCase):
    def test_renamed_copy_and_comments_hash_alike(self):
        self.assertTrue(gate.code_hashes(DEF) & gate.code_hashes(RENAMED))
        self.assertEqual(gate.agnostic_hash(DEF), gate.agnostic_hash(RENAMED + " // x"))
        self.assertNotEqual(gate.agnostic_hash(DEF), gate.agnostic_hash(DEF.replace("+ 1", "+ 2")))

    def test_variants_of_a_definition(self):
        src = "void g(int* p, int n) {\n    if (n > 0) {\n        *p = n;\n    }\n}\n"
        vs = gate.variants(src, "g", (0, len(src)), True)
        self.assertTrue(all(v.rstrip().endswith("}") for v in vs))
        self.assertNotIn(src, vs)


class FrozenTest(DataDir):
    def test_keys_functions_shapes_hashes(self):
        self.write_gate()
        keys = gate.frozen_keys(self.cfg)
        self.assertEqual(dict(keys), {"fn_A": "gate-lora2", "fn_B": "near-sample-60"})
        self.assertEqual(keys.get(("stale/unit", "fn_A")), "gate-lora2")     # the unit never hides a function
        fns, shapes = gate.never(self.cfg)
        self.assertEqual(fns, {"fn_A": "gate-lora2", "fn_B": "near-sample-60"})
        self.assertEqual(shapes["S_fn_A"], "gate-lora2")
        self.assertIn(gate.agnostic_hash(RENAMED), gate.hashes(self.cfg))

    def test_rehash_file_feeds_the_filter(self):
        self.write_gate()
        self.assertNotIn("feedbeef", gate.hashes(self.cfg))
        (self.d / gate.GATE_NAME / gate.REHASH_FILE).write_text(json.dumps({"sets": {"gate-lora2": {"fn_A": ["feedbeef"]}}}))
        self.assertEqual(gate.hashes(self.cfg)["feedbeef"], "gate-lora2")

    def test_rehash_keeps_earlier_hashes(self):
        sets = {"gate-lora2": {"fn_A": {"aa", "bb"}}}
        old = {"sets": {"gate-lora2": {"fn_A": ["bb", "cc"]}, "near-sample-60": {"fn_B": ["dd"]}}}
        self.assertEqual(gate.keep_earlier(sets, old), 2)
        self.assertEqual(sets, {"gate-lora2": {"fn_A": {"aa", "bb", "cc"}}, "near-sample-60": {"fn_B": {"dd"}}})
        self.assertEqual(gate.keep_earlier(sets, {}), 0)

    def test_nothing_frozen_without_files(self):
        self.assertEqual(gate.frozen_keys(self.cfg), {})

    def test_route_bars_the_frozen_row(self):
        fns = [{"unit": "u/a", "source": "src/a.cpp", "function": "fn_A", "address": 1, "size": 0x40, "fuzzy": 99.0},
               {"unit": "u/a", "source": "src/a.cpp", "function": "fn_Z", "address": 2, "size": 0x40, "fuzzy": 99.0}]
        rows = {r["function"]: r for r in route.route_rows(fns, {}, {}, {}, frozen={("u/a", "fn_A"): "gate-lora2"})}
        self.assertEqual((rows["fn_A"]["worker"], rows["fn_A"]["frozen"]), ("frozen", "gate-lora2"))
        self.assertIn("FROZEN", rows["fn_A"]["why"])
        self.assertNotEqual(rows["fn_Z"]["worker"], "frozen")
        self.assertEqual(route.select(list(rows.values()), "frozen")[0]["function"], "fn_A")
        self.assertEqual(route.summary(list(rows.values()))["workers"]["frozen"]["functions"], 1)

    def test_grind_queue_skips_frozen(self):
        fns = [{"unit": "u/a", "source": "src/a.cpp", "function": n, "address": i, "size": 0x40, "fuzzy": 0.0}
               for i, n in enumerate(("fn_A", "fn_Z"))]
        items = llmgrind.queue(fns, set(), set(), {}, frozen={("u/a", "fn_A"): "gate-lora2"})
        self.assertEqual([i["function"] for i in items], ["fn_Z"])

    def test_leak_filter_checks_fenced_blocks_by_hash(self):
        self.write_gate()
        llmdata.never_definitions(self.cfg, set())            # loads the frozen hashes
        try:
            def msg(code):
                return {"messages": [{"content": "sys"}, {"content": "current:\n```cpp\n" + code + "\n```"}],
                        "tags": {"unit": "u/x", "function": "fn_other"}}
            self.assertEqual(llmdata.leaks(msg(RENAMED), set(), set(), set(), []), "frozen definition hash")
            self.assertIsNone(llmdata.leaks(msg(DEF.replace("+ 1", "+ 9")), set(), set(), set(), []))
        finally:
            llmdata.GATE_HASHES.clear()


if __name__ == "__main__":
    unittest.main()


class RefreshTest(DataDir):
    def row(self, fn, addr, fuzzy=99.0, cls="near-register", attempts=0, size=0x40):
        return {"unit": "u/x", "function": fn, "address": addr, "size": size, "fuzzy": fuzzy, "worker": "cpu",
                "class": cls, "attempts": attempts, "source": ""}

    def test_retires_exact_and_redraws_by_half_with_next_seeds(self):
        council = self.d / "logs"
        for t in ("run23", "run30"):
            (council / t).mkdir(parents=True)
        (council / "run23" / "a.md").write_text("fn_80000010 | 10% | gave up | 1 | x\n")
        (council / "run30" / "b.md").write_text("fn_80000020 | 10% | gave up | 1 | x\nfn_80000040 tried\n")
        (self.d / "near-sample-60.json").write_text(json.dumps({"functions": [rec("u/x", "fn_80000050")]}))
        gate_recs = [rec("u/x", "fn_80000010", why="gave-up", address=0x80000010, commit="old"),
                     rec("u/x", "fn_80000030", why="never-tried", address=0x80000030, commit="old"),
                     rec("u/x", "fn_80000060", why="never-tried", address=0x80000060, commit="old")]
        doc = {"seed": 100, "functions": gate_recs}
        rows = [self.row("fn_80000010__3FooFv", 0x80000010),           # renamed, still open: kept
                self.row("fn_80000020", 0x80000020, attempts=2),        # run30 give-up: the gave-up replacement
                self.row("fn_80000040", 0x80000040),                    # named in run30: not never-tried
                self.row("fn_80000050", 0x80000050),                    # near-sample: never
                self.row("fn_80000070", 0x80000070),                    # the never-tried pool
                self.row("fn_80000080", 0x80000080),
                self.row("fn_80000090", 0x80000090, size=4096)]         # too big
        doc["functions"][0]["why"] = "never-tried"                      # 10 stays, so 30 and 60 (both never-tried) go
        new = gate.redraw(self.cfg, doc, council, rows, "c" * 40, {})
        self.assertEqual([r["function"] for r in new["functions"]][:1], ["fn_80000010"])
        self.assertEqual(sorted(r["function"] for r in new["retired"]), ["fn_80000030", "fn_80000060"])
        added = new["refreshes"][0]["added"]
        self.assertEqual(sorted(added), ["fn_80000070", "fn_80000080"])
        self.assertEqual(new["refreshes"][0]["seeds"], [102, 103])
        self.assertTrue(all(r["commit"] == "c" * 40 and r["why"] == "never-tried" for r in new["functions"][1:]))
        self.assertEqual(doc["functions"][1]["function"], "fn_80000030")    # input untouched
        # a second refresh with nothing exact changes nothing; retired functions are never drawn again
        rows2 = rows + [self.row("fn_80000030", 0x80000030)]
        again = gate.redraw(self.cfg, new, council, rows2, "d" * 40, {})
        self.assertEqual(again["refreshes"], new["refreshes"])

    def test_retire_names_and_trained_exports_stay_out_of_draws(self):
        council = self.d / "logs"
        (council / "run30").mkdir(parents=True)
        (council / "run30" / "b.md").write_text("".join(f"fn_800000{n}0 | 10% | gave up | 1 | x\n" for n in (2, 3, 4, 5)))
        rec_dir = self.d / "dataset" / "records"
        rec_dir.mkdir(parents=True)
        # trained: 20 by function field, 30 by an asm header in a chat row (its call to fn_80000040 is not a row),
        # 40 by the hash of its definition today (under another name)
        (rec_dir / "base.jsonl").write_text(json.dumps({"function": "fn_80000020", "exact": "int a() { return 1; }"}) + "\n")
        msg = {"messages": [{"role": "system", "content": "s"},
                            {"role": "user", "content": ".fn fn_80000030__3FooFv, global\n bl fn_80000040\n"}]}
        (self.d / "dataset" / "train.jsonl").write_text(json.dumps(msg) + "\n")
        (rec_dir / "attempts.jsonl").write_text(json.dumps({"function": "zz", "wrong": "```cpp\n" + DEF + "\n```"}) + "\n")
        src = self.d / "u.cpp"
        src.write_text(DEF.replace("fn_80001000", "fn_80000040") + "\n")
        doc = {"seed": 1, "functions": [rec("u/x", "fn_80000010", why="gave-up", address=0x80000010),
                                        rec("u/x", "fn_80000090", why="gave-up", address=0x80000090)]}
        rows = [self.row(f"fn_800000{n}0", 0x80000000 + n * 16, attempts=1) for n in (1, 2, 3, 4, 5)]
        rows[3]["source"] = "u.cpp"
        rows.append(self.row("fn_80000090", 0x80000090, attempts=1))
        cfg = SimpleNamespace(root=self.d)
        bar = gate.trained_bar(cfg)
        self.assertEqual([r["function"] for r in rows if bar(r)], ["fn_80000020", "fn_80000030", "fn_80000040"])
        new = gate.redraw(cfg, doc, council, rows, "c" * 40, {}, retire=("fn_80000010",), bar=bar)
        self.assertEqual(sorted(r["function"] for r in new["functions"]), ["fn_80000050", "fn_80000090"])
        self.assertEqual(new["retired"][0]["retired_why"], "trained earlier")
        self.assertEqual(new["refreshes"][0]["trained_earlier"], ["fn_80000010"])
        text = gate.card({**new, "frozen": "d", "commit": "c" * 40,
                          "stats": {"gave_up_pool": 1, "gave_up_logged": 1, "fresh_pool": 0}}, {"gate-lora2": new["functions"]})
        self.assertIn("fn_80000010 (trained earlier)", text)

    def test_gave_up_half_takes_t30_names(self):
        council = self.d / "logs"
        (council / "run30").mkdir(parents=True)
        (council / "run30" / "b.md").write_text("fn_80000020 | 10% | gave up | 1 | x\n")
        doc = {"seed": 1, "functions": [rec("u/x", "fn_80000010", why="gave-up", address=0x80000010)]}
        rows = [self.row("fn_80000020", 0x80000020, attempts=3)]
        new = gate.redraw(self.cfg, doc, council, rows, "c" * 40, {})
        self.assertEqual([(r["function"], r["why"]) for r in new["functions"]], [("fn_80000020", "gave-up")])
        text = gate.card({**new, "frozen": "d", "commit": "c" * 40,
                          "stats": {"gave_up_pool": 1, "gave_up_logged": 1, "fresh_pool": 0}}, {"gate-lora2": new["functions"]})
        self.assertIn("refresh 1", text)
        self.assertIn("retired fn_80000010 (exact); added fn_80000020", text)


class ReleaseFileTest(DataDir):
    def test_the_release_file_beside_the_eval_sets_is_not_read_as_a_set(self):
        self.write_gate()
        (self.d / "eval").mkdir()
        (self.d / "eval" / "bench.json").write_text(json.dumps({"functions": [rec("u/c", "fn_C")]}))
        (self.d / "eval" / "released.json").write_text(json.dumps({"fn_C": {"date": "2000-01-01", "reason": "x"}}))
        files = gate.set_files(self.cfg)
        self.assertIn("eval:bench", files)
        self.assertNotIn("eval:released", files)
        self.assertIn("eval", gate.load_sets(self.cfg))
