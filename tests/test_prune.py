import hashlib
import re
import unittest

from tools.dk.platforms.gc_mwcc import prunecmd
from tools.dk.platforms.gc_mwcc.prunecmd import Pruner, items, removable

SRC = '''#include "a.h"

// Unused helper type.
struct Dead {
    int a;
};
struct Live {
    int b;
};
extern "C" {
extern int lbl_unused;
extern int lbl_used;
}
typedef int Dead2;
static inline int unused_inline(int x) { return x + 1; }
int use(Live* l) {
    return l->b + lbl_used;
}
'''


class ItemsTest(unittest.TestCase):
    def test_top_level_items_with_their_comments(self):
        got = [(it.kind, it.label) for it in items(SRC)]
        self.assertEqual(got, [("pp", '#include "a.h"'), ("decl", "struct Dead {"), ("decl", "struct Live {"),
                               ("decl", "extern int lbl_unused;"), ("decl", "extern int lbl_used;"),
                               ("decl", "typedef int Dead2;"), ("inline", "static inline int unused_inline(int x) { return x + 1; }"),
                               ("function", "int use(Live* l) {")])
        dead = next(it for it in items(SRC) if it.label == "struct Dead {")
        self.assertTrue(SRC[dead.start:dead.end].startswith("// Unused helper type.\nstruct Dead {"))
        self.assertEqual([it.kind for it in items(SRC) if removable(it)], ["decl"] * 5 + ["inline"])


class FakePruner(Pruner):
    """The object is what the function uses: the names in its body that some item declares."""

    def __init__(self):
        import tempfile
        from pathlib import Path
        self.jobs, self.log, self.calls = 2, (lambda s: None), 0
        self.work = Path(tempfile.mkdtemp())

    def digest(self, text, slot):
        self.calls += 1
        if "struct Live" not in text or "extern int lbl_used;" not in text:
            return None                               # use() would not compile
        return hashlib.sha1(b"same").hexdigest()


class PruneTest(unittest.TestCase):
    def test_everything_the_object_does_not_need_goes(self):
        new, removed, compiles = FakePruner().run(SRC)
        self.assertEqual(sorted(it.label for it in removed),
                         ["extern int lbl_unused;", "static inline int unused_inline(int x) { return x + 1; }",
                          "struct Dead {", "typedef int Dead2;"])
        self.assertNotIn("Unused helper type", new)
        self.assertIn("struct Live {\n    int b;\n};\n", new)
        self.assertIn('extern "C" {\nextern int lbl_used;\n}\n', new)
        self.assertIn('#include "a.h"', new)

    def test_items_that_only_pass_alone_are_proved_one_by_one(self):
        class OneOf(FakePruner):
            def digest(self, text, slot):          # either of two typedefs may go, not both
                if not re.search(r"typedef int (A|B);", text):
                    return None
                return hashlib.sha1(b"same").hexdigest()
        new, removed, _ = OneOf().run("typedef int A;\ntypedef int B;\n")
        self.assertEqual(len(removed), 1)
        self.assertEqual(len(re.findall("typedef", new)), 1)


if __name__ == "__main__":
    unittest.main()
