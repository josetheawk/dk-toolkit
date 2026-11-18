import json
import unittest
from types import SimpleNamespace

from tools.dk.platforms.gc_mwcc import closedata, unlock
from tools.dk.platforms.gc_mwcc.datagen import Draft
from tools.dk.platforms.gc_mwcc.matching import FunctionScore, UnitScore


def us(data, fns=((".f", 100.0),)):
    return UnitScore("u", [FunctionScore(n, 0x40, p) for n, p in fns], dict(data))


class OutcomeTest(unittest.TestCase):
    def test_closed_raised_and_kept(self):
        before = us({".data": 40.0, ".sbss": 0.0, ".bss": 100.0})
        after = us({".data": 70.0, ".sbss": 100.0, ".bss": 100.0})
        o = closedata.outcome(before, after)
        self.assertEqual((o["closed"], o["raised"], o["worse"], o["keep"], o["exact"]), ([".sbss"], [".data"], [], True, False))
        self.assertEqual(o["before"], {".data": 40.0, ".sbss": 0.0})

    def test_anything_lower_is_not_kept(self):
        before = us({".sbss": 0.0, ".bss": 100.0})
        after = us({".sbss": 100.0, ".bss": 80.83})
        o = closedata.outcome(before, after)
        self.assertEqual(o["closed"], [".sbss"])
        self.assertFalse(o["keep"])
        self.assertIn(".bss 100.00% before, now 80.83%", o["worse"][0])
        worse_fn = us({".sbss": 100.0}, fns=((".f", 99.0),))
        self.assertFalse(closedata.outcome(before, worse_fn)["keep"])

    def test_nothing_rose_or_no_compile_is_not_kept(self):
        before = us({".data": 40.0})
        self.assertFalse(closedata.outcome(before, us({".data": 40.0}))["keep"])
        o = closedata.outcome(before, None)
        self.assertEqual((o["compiles"], o["keep"]), (False, False))
        self.assertTrue(closedata.outcome(before, us({".data": 100.0}))["exact"])


class RecordTest(unittest.TestCase):
    def test_record_holds_target_bytes_relocs_and_the_definition(self):
        ud = SimpleNamespace(unit={"name": "main/x"}, rel="x.cpp",
                             attrs={"tbl": {"type": "object", "size": "0x8", "scope": "global", "data": "4byte"}},
                             original=lambda name, size: (bytes.fromhex("0000000100000000"), {4: "fn_80001000"}))
        d = Draft("tbl", ".data", 0x80250000, 8, "verified", text="int tbl[2] = {1, (int)fn_80001000};",
                  needs=['extern "C" void fn_80001000();'], evidence="dtk type 4byte")
        r = closedata.record(ud, d)
        self.assertEqual(r["target"], {"bytes": "0000000100000000", "relocs": {"0x4": "fn_80001000"}})
        self.assertEqual((r["name"], r["section"], r["address"], r["size"]), ("tbl", ".data", "0x80250000", 8))
        self.assertEqual(r["symbol"]["data"], "4byte")
        self.assertEqual(r["definition"], d.text)
        self.assertFalse(r["replaces_extern"])
        json.dumps(r)                                                  # one JSON line
        z = closedata.record(ud, Draft("z", ".bss", 0x80300000, 16, "verified", text="int z[4];"))
        self.assertEqual(z["target"], {"bytes": None, "relocs": {}})   # zero-filled: nothing to compare but size


class CensusTest(unittest.TestCase):
    @staticmethod
    def unit(name, complete, *sections):
        return {"name": name, "metadata": {"complete": complete},
                "sections": [{"name": n, "fuzzy_match_percent": p} for n, p in sections]}

    def setUp(self):
        self.UNITS = [self.unit("a", False, (".data", 26.8), (".text", 99.9), (".sbss", 50.0)),
                      self.unit("b", False, (".sdata2", 85.71)), self.unit("c", True, (".data", 10.0)),
                      self.unit("d", False, (".bss", 0.0))]
        self.scores = {"a": us({".data": 100.0, ".sbss": 100.0}, fns=((".f", 99.0),)),
                       "b": us({".sdata2": 85.71}), "d": None}
        self.rows = closedata.census_rows(self.UNITS, self.scores)

    def test_rows_keep_dk_and_raw_apart_and_skip_linked_units(self):
        self.assertEqual(sorted(self.rows), ["a", "b", "d"])
        self.assertEqual(self.rows["a"], {"dk": {}, "raw": {".data": 26.8, ".sbss": 50.0}, "open_functions": 1})
        self.assertEqual(self.rows["b"], {"dk": {".sdata2": 85.71}, "raw": {".sdata2": 85.71}, "open_functions": 0})
        self.assertIsNone(self.rows["d"]["dk"])
        text = "\n".join(closedata.census_summary(self.rows))
        self.assertIn("data open under dk's scoring: 1; under the raw report: 2; raw-only", text)
        self.assertIn("every function exact (data is the last blocker): 1", text)
        self.assertEqual(json.loads(json.dumps(self.rows, sort_keys=True)), self.rows)

    def test_unlock_prefers_the_census_over_the_raw_report(self):
        unit = {"name": "a", "sections": [{"name": ".data", "size": "3472", "fuzzy_match_percent": 26.8},
                                          {"name": ".sbss", "size": "8", "fuzzy_match_percent": 50.0}]}
        self.assertEqual([s["name"] for s in unlock.data_open(unit)], [".data", ".sbss"])
        self.assertEqual(unlock.data_open(unit, self.rows), [])
        b = {"name": "b", "sections": [{"name": ".sdata2", "size": "56", "fuzzy_match_percent": 85.71}]}
        self.assertEqual(unlock.data_open(b, self.rows), [{"name": ".sdata2", "size": 56, "fuzzy": 85.71}])
        d = {"name": "d", "sections": [{"name": ".bss", "size": "16", "fuzzy_match_percent": 0.0}]}
        self.assertEqual([s["name"] for s in unlock.data_open(d, self.rows)], [".bss"])   # unscored: the raw report
        report = {"measures": {"total_code": "1000"}, "units": [
            {**unit, "measures": {"total_code": "100"}, "metadata": {"complete": False, "source_path": "src/a.cpp"}, "functions": []}]}
        plain = unlock.rank(report, [])
        with_census = unlock.rank(report, [], census=self.rows)
        self.assertEqual(plain["units"][0]["data_open"], [".data", ".sbss"])
        self.assertEqual((with_census["units"][0]["data_open"], with_census["units"][0]["blockers"][0]["kind"]), ([], "link"))


if __name__ == "__main__":
    unittest.main()
