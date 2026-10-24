import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_project
from tools.dk.core.nextcmd import queue
from tools.dk.platforms.gc_mwcc.ctxcmd import callers_of, class_slots, index_asm

ASM = """\
.include "macros.inc"
.text
.fn fn_A, global
/* 80003100 */	bl fn_B
/* 80003104 */	lwz r3, lbl_X@sda21(r0)
/* 80003108 */	blr
.endfn fn_A

.fn fn_B, local
/* 8000310C */	blr
.endfn fn_B

.fn fn_C, global
/* 80003110 */	bl fn_B
/* 80003114 */	bl fn_Bx
/* 80003118 */	blr
.endfn fn_C
"""


class QueueTest(unittest.TestCase):
    def test_open_functions_then_candidates(self):
        report = {"units": [
            {"name": "main/b", "metadata": {"source_path": "src/b.c"}, "functions": [
                {"name": "g", "size": "8", "fuzzy_match_percent": 50.0, "metadata": {"virtual_address": "2147500000"}},
                {"name": "f", "size": "8", "fuzzy_match_percent": 100.0, "metadata": {"virtual_address": "2147499000"}},
                {"name": "h", "size": "8", "metadata": {"virtual_address": "2147490000"}},
            ]},
            {"name": "main/auto_big", "metadata": {"auto_generated": True}, "measures": {"total_code": "400"},
             "functions": [{"name": "fn_1", "metadata": {"virtual_address": "10"}}]},
            {"name": "main/auto_lib", "metadata": {"auto_generated": True}, "measures": {"total_code": "900"},
             "functions": [{"name": "OSInit", "metadata": {"virtual_address": "20"}}]},
            {"name": "main/auto_small", "metadata": {"auto_generated": True}, "measures": {"total_code": "4"},
             "functions": [{"name": "fn_2", "metadata": {"virtual_address": "30"}}]},
        ]}
        open_functions, candidates = queue(report)
        self.assertEqual([f["function"] for f in open_functions], ["h", "g"])     # link order, 100% excluded
        self.assertEqual(open_functions[0]["fuzzy"], 0.0)                          # omitted percent = 0
        self.assertEqual([c["unit"] for c in candidates], ["main/auto_lib", "main/auto_small", "main/auto_big"])


class AsmIndexTest(unittest.TestCase):
    def test_index_and_callers(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "u.s").write_text(ASM)
            fns = index_asm(Path(tmp))
        self.assertEqual(sorted(fns), ["fn_A", "fn_B", "fn_C"])
        self.assertIn("lbl_X@sda21", fns["fn_A"].text)
        self.assertTrue(fns["fn_B"].text.endswith(".endfn fn_B"))
        self.assertEqual(callers_of("fn_B", fns), ["fn_A", "fn_C"])               # fn_Bx is not fn_B


class OwnerTest(unittest.TestCase):
    def test_most_base_class_owns_inherited_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_project(Path(tmp))
            rows = [
                {"name": "Base", "rtti": 1, "bases": [], "vtables": [{"address": 100, "offset": 0, "slots": [7, 8]}]},
                {"name": "Mid", "rtti": 2, "bases": [[1, 0]], "vtables": [{"address": 200, "offset": 0, "slots": [7, 9]}]},
                {"name": "Leaf", "rtti": 3, "bases": [[1, 0], [2, 0]], "vtables": [{"address": 300, "offset": 0, "slots": [7, 9]}]},
                {"name": "Other", "rtti": 4, "bases": [], "vtables": [{"address": 400, "offset": 0, "slots": [5]}]},
            ]
            path = Path(tmp) / "build" / "dk" / "rtti.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(rows))
            slots, owner = class_slots(cfg, 7)
            self.assertEqual(owner, "Base")
            self.assertEqual(len(slots), 3)
            self.assertEqual(class_slots(cfg, 9)[1], "Mid")
            self.assertEqual(class_slots(cfg, 12345), ([], None))


if __name__ == "__main__":
    unittest.main()
