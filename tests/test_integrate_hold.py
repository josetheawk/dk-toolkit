import unittest
from unittest import mock

from tools.dk.core import committools, floor


class FloorUnitsTest(unittest.TestCase):
    def test_the_floor_file_carries_per_unit_counts_without_changing_the_floor(self):
        text = floor.render({"matched_functions": 12, "complete_code": 120, "complete_units": 2},
                            {"main/b": 3, "main/a": 9})
        self.assertEqual(floor.parse(text), {"matched_functions": 12, "complete_code": 120, "complete_units": 2})
        self.assertEqual(floor.units(text), {"main/a": 9, "main/b": 3})
        self.assertEqual(floor.units(floor.render({"matched_functions": 1, "complete_code": 1, "complete_units": 1})), {})

    def test_the_gate_records_the_builds_per_unit_counts(self):
        committed = {"matched_functions": 10, "complete_code": 100, "complete_units": 2}
        with mock.patch.object(floor, "committed", return_value=committed):
            below, text = floor.gate(None, {"matched_functions": 11, "complete_code": 100, "complete_units": 2,
                                            "units": {"main/a": 11}})
        self.assertEqual(below, [])
        self.assertEqual(floor.units(text), {"main/a": 11})


class HoldBackTest(unittest.TestCase):
    def test_units_below_their_head_count_fall_largest_first(self):
        self.assertEqual(committools.fallen({"a": 5, "b": 1, "c": 7}, {"a": 6, "b": 4, "c": 7, "d": 2}),
                         [("b", 1, 4), ("a", 5, 6)])

    def test_a_unit_missing_from_the_build_moved_and_did_not_fall(self):
        # a split or rename: the old unit's functions now count in another unit
        self.assertEqual(committools.moved({"a": 5, "new": 45}, {"a": 5, "old": 45, "empty": 0}), ["old"])
        self.assertEqual(committools.fallen({"a": 5, "new": 45}, {"a": 5, "old": 45}), [])

    def test_the_floor_compares_a_builds_units_with_its_table(self):
        below, above, gone = floor.unit_differences({"a": 2, "b": 5, "new": 4}, {"a": 3, "b": 5, "old": 4})
        self.assertEqual(below, [("a", 2, 3)])
        self.assertEqual(above, [("new", 4, 0)])
        self.assertEqual(gone, [("old", 0, 4)])
        text = floor.render({"matched_functions": 1, "complete_code": 1, "complete_units": 1}, {"a": 1})
        self.assertIn("`dk check` does not count it", text)

    def test_a_changed_source_is_held_back_and_an_unchanged_one_is_named(self):
        sources = {"main/a": "src/a.cpp", "main/b": "src/b.cpp"}
        overlay = {"src/a.cpp": b"new", "include/x/b.h": b"new"}
        includes = lambda src: ["x/b.h"] if src == "src/b.cpp" else []
        hold, notes = committools.hold_back_plan([("main/a", 1, 3), ("main/b", 2, 4)], sources, overlay, includes)
        self.assertEqual(hold, [("src/a.cpp", 1, 3)])
        self.assertEqual(len(notes), 1)
        self.assertIn("main/b fell to 2 exact (4 at HEAD) with its source unchanged; changed headers it includes: "
                      "include/x/b.h", notes[0])

    def test_a_source_already_held_back_is_not_held_again(self):
        hold, notes = committools.hold_back_plan([("main/a", 1, 3)], {"main/a": "src/a.cpp"}, {"src/a.cpp": b"head"},
                                                 lambda src: [], {"src/a.cpp"})
        self.assertEqual(hold, [])
        self.assertIn("held back already", notes[0])


if __name__ == "__main__":
    unittest.main()
