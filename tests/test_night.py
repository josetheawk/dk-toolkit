"""dk night's helpers: the register-order-only list from residue censuses, and the night's candidates."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc.nightcmd import candidates_of, first_list


class NightTest(unittest.TestCase):
    def test_first_list_keeps_forced_exact_only(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, out = Path(d) / "a.json", Path(d) / "b.json", Path(d) / "first.json"
            a.write_text(json.dumps([{"unit": "u", "function": "f", "forced": 100.0},
                                     {"unit": "u", "function": "g", "forced": 99.5},
                                     {"unit": "u", "function": "h", "error": "no oracle values"}]))
            b.write_text("not json")
            self.assertEqual(first_list([a, b, Path(d) / "missing.json"], out), 1)
            self.assertEqual(json.loads(out.read_text()), [["u", "f"]])

    def test_candidates_skip_non_records(self):
        with tempfile.TemporaryDirectory() as d:
            c = Path(d) / "candidates" / "2026-11-16"
            c.mkdir(parents=True)
            (c / "abc.json").write_text(json.dumps({"id": "abc", "unit": "u", "function": "f"}))
            (c / "labels.json").write_text(json.dumps({"abc": "approved"}))
            self.assertEqual([r["id"] for r in candidates_of(Path(d), "2026-11-16")], ["abc"])
            self.assertEqual(candidates_of(Path(d), "2000-01-01"), [])


if __name__ == "__main__":
    unittest.main()
