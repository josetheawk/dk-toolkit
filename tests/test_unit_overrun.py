import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.test_zero_filled_kept import LOCAL, make_object
from tools.dk.platforms.gc_mwcc.unitcmd import section_overruns


class SectionOverrunTest(unittest.TestCase):
    def test_a_label_reaching_past_its_section_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.o").write_bytes(make_object({".data": bytes(8)}, [("gap_07_80001004_data", ".data", 4, 8, LOCAL)]))
            (root / "b.o").write_bytes(make_object({".data": bytes(8)}, [("lbl_80002000", ".data", 0, 8, LOCAL)]))
            (root / "objdiff.json").write_text(json.dumps({"units": [{"name": "main/a", "target_path": "a.o"},
                                                                     {"name": "main/b", "target_path": "b.o"}]}))
            self.assertEqual(section_overruns(SimpleNamespace(root=root)),
                             ["main/a: gap_07_80001004_data ends at .data+0xC, past its 0x8 bytes"])


if __name__ == "__main__":
    unittest.main()
