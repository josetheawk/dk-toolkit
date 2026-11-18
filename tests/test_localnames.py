"""localnames.py: a search's new locals typed from the callee's return type and named after it."""

import tempfile
import unittest
from pathlib import Path

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc.judge import PLACEHOLDER_LOCAL
from tools.dk.platforms.gc_mwcc.localnames import call_return_types, callee_of, name_placeholders, natural_name

SRC = """int atoi(const char* s);
extern "C" int CARDUnmount(int chan);
void f(const char* format, int chan) {
    int width = 0;
    __typeof__(atoi(format)) t0 = atoi(format);
    width = t0;
    int t1 = CARDUnmount(chan);
    width += t1;
}
"""


def body(src):
    span = next(f for f in cbody.functions(src) if f.name == "f")
    return cbody.parse_body(src, span.open, cbody.type_names(src))


class LocalNamesTest(unittest.TestCase):
    def test_callee_of(self):
        self.assertEqual(callee_of("atoi(format)"), "atoi")
        self.assertEqual(callee_of("p->fn_80483DA0(shape, index)"), "fn_80483DA0")
        self.assertIsNone(callee_of("(int)x"))
        self.assertIsNone(callee_of("a + b"))

    def test_natural_name(self):
        self.assertEqual(natural_name("CARDUnmount", set()), "unmount")
        self.assertEqual(natural_name("atoi", set()), "result")
        self.assertEqual(natural_name("fn_80483DA0", {"value"}), "value2")
        self.assertEqual(natural_name("GetFontHeight", set()), "height")

    def test_return_type_from_the_unit_or_headers(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "include").mkdir()
            self.assertEqual(call_return_types(Path(d), SRC, body(SRC), ["t0"]), {"t0": "int"})

    def test_placeholders_named_inside_the_body(self):
        new, renames = name_placeholders(SRC, body(SRC), {"width"}, PLACEHOLDER_LOCAL)
        self.assertEqual(renames, {"t0": "result", "t1": "unmount"})
        self.assertIn("width = result;", new)
        self.assertIn("int unmount = CARDUnmount(chan);", new)
        self.assertNotIn("t0", new.split("void f")[1])


if __name__ == "__main__":
    unittest.main()
