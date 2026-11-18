"""dk card: batch picking from residue censuses and the numbered source window."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.dk.platforms.gc_mwcc import cardcmd


class CardTest(unittest.TestCase):
    def test_source_window_numbers_and_gaps(self):
        text = "\n".join(f"line{k}" for k in range(1, 21))
        out = cardcmd.source_window(text, [3, 15])
        self.assertEqual(out[0], "     1| line1")
        self.assertIn("      ...", out)
        self.assertEqual(out[-1], "    17| line17")

    def test_pick_fewest_rows_not_order_only_not_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            census = Path(d) / "residue-near-shape.json"
            row = lambda n: [{"class": "OTHER", "line": 1, "target": "a", "ours": "b"}] * n
            census.write_text(json.dumps([
                {"unit": "u", "function": "three", "forced": 99.0, "rows": row(3)},
                {"unit": "u", "function": "one", "forced": 99.5, "rows": row(1)},
                {"unit": "u", "function": "order", "forced": 100.0, "rows": []},
                {"unit": "u", "function": "linked", "forced": 99.9, "forced_linked": True, "rows": row(1)},
                {"unit": "u", "function": "skipped", "forced": 99.9, "rows": row(1)},
                {"unit": "u", "function": "err", "error": "no oracle values"}]))
            with mock.patch.object(cardcmd, "census_files", lambda cfg: [census]):
                got = cardcmd.pick(SimpleNamespace(root=Path(d)), 5, {("u", "skipped")})
        self.assertEqual([r["function"] for r in got], ["one", "three"])


if __name__ == "__main__":
    unittest.main()
