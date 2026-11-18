"""dk asset: embedded game data declared in units.toml, extracted from the original at build time."""

import argparse
import ast
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import List

from tests.helpers import make_dol
from tests.test_zero_filled_kept import GLOBAL, make_object
from tools.dk.core.config import load
from tools.dk.platforms.gc_mwcc import assets
from tools.dk.platforms.gc_mwcc.unitcommit import compose_units
from tools.dk.platforms.gc_mwcc.units import read_units_file, render_units_file

REPO = Path(__file__).resolve().parents[1]
BLOB = bytes(range(256)) * 2                     # 0x200 bytes of "texture" at 0x80001100
DATA = b"\x11" * 0x100 + BLOB + b"\x22" * 0x100   # D0 at 0x80001000

DK_TOML = """\
[project]
name = "demo"
game_id = "DEMO01"
platform = "gc_mwcc"

[target]
original = "orig/main.dol"
built = "build/main.dol"
sha1 = "{sha1}"

[platform]
symbols = "config/symbols.txt"
splits = "config/splits.txt"
units = "config/units.toml"
"""
UNITS = """\
# units
[[lib]]
name = "game"
objects = [
    { path = "game/a.cpp", status = "NonMatching", cflags = "game" },
    { path = "game/b.cpp", status = "NonMatching" },
]
"""
SPLITS = """\
Sections:
\t.text       type:code align:4
\t.data       type:data align:32

game/a.cpp:
\t.text       start:0x80000100 end:0x80000200
\t.data       start:0x80001000 end:0x80001300
"""
SYMBOLS = """\
lbl_80001000 = .data:0x80001000; // type:object size:0x100
lbl_80001100 = .data:0x80001100; // type:object size:0x200
lbl_80001300 = .data:0x80001300; // type:object size:0x100
"""


def project(root: Path, units: str = UNITS):
    dol = make_dol([("T0", 0x80000100, b"\x60\0\0\0" * 0x40), ("D0", 0x80001000, DATA)])
    for rel, text in (("config/units.toml", units), ("config/splits.txt", SPLITS), ("config/symbols.txt", SYMBOLS)):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    (root / "orig").mkdir()
    (root / "orig" / "main.dol").write_bytes(dol)
    (root / "dk.toml").write_text(DK_TOML.format(sha1=hashlib.sha1(dol).hexdigest()))
    return load(root)


def add_args(unit, name, address, size, width=1):
    return argparse.Namespace(action="add", unit=unit, name=name, address=address, size=size, width=width)


def quiet(fn, *args):
    out = io.StringIO()
    with redirect_stdout(out):
        code = fn(*args)
    return code, out.getvalue()


class RenderTest(unittest.TestCase):
    def test_bytes_halfwords_and_words_are_big_endian_with_a_comma_each(self):
        a = assets.Asset("u.cpp", "x", 0x80001000, 4)
        lines = assets.render(a, b"\x01\x02\x03\x04", "orig/main.dol").splitlines()
        self.assertTrue(lines[0].startswith("/* x: 0x4 bytes at 0x80001000 of orig/main.dol"))
        self.assertEqual(lines[1:], ["0x01, 0x02, 0x03, 0x04,"])
        two = assets.render(assets.Asset("u.cpp", "x", 0, 4, 2), b"\x01\x02\x03\x04", "o").splitlines()[1:]
        self.assertEqual(two, ["0x0102, 0x0304,"])
        four = assets.render(assets.Asset("u.cpp", "x", 0, 4, 4), b"\x01\x02\x03\x04", "o").splitlines()[1:]
        self.assertEqual(four, ["0x01020304,"])

    def test_sixteen_bytes_per_line(self):
        text = assets.render(assets.Asset("u.cpp", "x", 0, 0x21), bytes(0x21), "o")
        self.assertEqual([len(l.split()) for l in text.splitlines()[1:]], [16, 16, 1])


class DeclaredTest(unittest.TestCase):
    def units(self, entry: str) -> str:
        return UNITS.replace('cflags = "game" }', f'cflags = "game", assets = [{entry}] }}')

    def test_reads_the_object_entries(self):
        got = assets.declared(self.units('{ name = "tex/fail", address = 0x80001100, size = 0x200 }'))
        self.assertEqual(got, [assets.Asset("game/a.cpp", "tex/fail", 0x80001100, 0x200)])
        self.assertEqual(got[0].include, "assets/tex/fail.inc")

    def test_rejects_bad_declarations(self):
        for entry, why in (('{ name = "../x", address = 0x0, size = 0x4 }', "relative path"),
                           ('{ name = "x", address = 0x0 }', "missing 'size'"),
                           ('{ name = "x", address = 0x0, size = 0x6, width = 4 }', "multiples"),
                           ('{ name = "x", address = 0x0, size = 0x4, width = 3 }', "width"),
                           ('{ name = "x", address = 0x0, size = 0x4, type = "u8" }', "unknown key"),
                           ('{ name = "x", address = 0x0, size = 0x4 }, { name = "x", address = 0x8, size = 0x4 }',
                            "declared twice")):
            with self.subTest(entry=entry), self.assertRaisesRegex(assets.AssetError, why):
                assets.declared(self.units(entry))

    def test_units_file_round_trip_writes_hex(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "units.toml")
            path.write_text(self.units('{ name = "x", address = 0x80001100, size = 0x200, width = 2 }'))
            text = render_units_file(read_units_file(path))
            self.assertIn('assets = [{ name = "x", address = 0x80001100, size = 0x200, width = 2 }]', text)
            path.write_text(text)
            self.assertEqual(render_units_file(read_units_file(path)), text)

    def test_a_value_the_renderer_does_not_know_says_to_sync(self):
        from tools.dk.platforms.gc_mwcc.units import _value
        with self.assertRaisesRegex(TypeError, "run ./dk worktree sync"):
            _value(1.5)

    def test_a_units_save_carries_its_own_assets_only(self):
        head = UNITS
        work = UNITS.replace('cflags = "game" }', 'cflags = "game", assets = [{ name = "a", address = 0x10, size = 0x4 }] }')
        work = work.replace('status = "NonMatching" }', 'status = "NonMatching", assets = [{ name = "b", address = 0x20, size = 0x4 }] }')
        got = assets.declared(compose_units(head, work, "game/a.cpp"))
        self.assertEqual([a.name for a in got], ["a"])


class ExtractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = project(self.root)
        self.asset = assets.Asset("game/a.cpp", "tex/fail", 0x80001100, 0x200)

    def tearDown(self):
        self.tmp.cleanup()

    def inc(self) -> Path:
        return self.root / "build" / "DEMO01" / "include" / "assets" / "tex" / "fail.inc"

    def test_writes_the_original_bytes_once(self):
        written, removed = assets.extract(self.cfg, [self.asset])
        self.assertEqual((written, removed), (["build/DEMO01/include/assets/tex/fail.inc"], []))
        body = "".join(self.inc().read_text().splitlines(True)[1:])
        self.assertEqual(bytes(int(v.rstrip(","), 16) for v in body.split()), BLOB)
        stamp = self.inc().stat().st_mtime_ns
        self.assertEqual(assets.extract(self.cfg, [self.asset]), ([], []))      # unchanged: not rewritten
        self.assertEqual(self.inc().stat().st_mtime_ns, stamp)

    def test_an_undeclared_inc_is_removed(self):
        assets.extract(self.cfg, [self.asset])
        self.assertEqual(assets.extract(self.cfg, []), ([], ["build/DEMO01/include/assets/tex/fail.inc"]))
        self.assertFalse(self.inc().exists())

    def test_refuses_a_wrong_original_or_a_range_outside_one_section(self):
        with self.assertRaisesRegex(assets.AssetError, "not inside one section"):
            assets.extract(self.cfg, [assets.Asset("game/a.cpp", "x", 0x80001300, 0x200)])
        (self.root / "orig" / "main.dol").write_bytes(b"\0" * 0x200)
        with self.assertRaisesRegex(assets.AssetError, "SHA-1"):
            assets.extract(self.cfg, [self.asset])

    def test_ensure_extracts_a_missing_asset_for_private_compiles(self):
        units = UNITS.replace('cflags = "game" }',
                              'cflags = "game", assets = [{ name = "tex/fail", address = 0x80001100, size = 0x200 }] }')
        (self.root / "config" / "units.toml").write_text(units)
        assets._ensured.discard(self.cfg.root)
        assets.ensure(self.cfg)
        self.assertTrue(self.inc().is_file())


class AddRemoveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = project(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_declares_extracts_and_names_the_symbol(self):
        code, out = quiet(assets.run, self.cfg, add_args("src/game/a.cpp", "fail", 0x80001100, 0x200))
        self.assertEqual(code, 0, out)
        self.assertIn("covers lbl_80001100 exactly", out)
        self.assertIn('#include "assets/fail.inc"', out)
        self.assertEqual(assets.load(self.cfg), [assets.Asset("game/a.cpp", "fail", 0x80001100, 0x200)])
        self.assertTrue((assets.out_dir(self.cfg) / "fail.inc").is_file())
        code, out = quiet(assets.run, self.cfg, argparse.Namespace(action="remove", unit="game/a.cpp", name="fail"))
        self.assertEqual(code, 0, out)
        self.assertEqual(assets.load(self.cfg), [])
        self.assertFalse((assets.out_dir(self.cfg) / "fail.inc").exists())
        self.assertNotIn("assets", (self.root / "config" / "units.toml").read_text())

    def test_add_refuses(self):
        quiet(assets.run, self.cfg, add_args("game/a.cpp", "fail", 0x80001100, 0x200))
        for args, why in ((add_args("game/c.cpp", "x", 0x80001100, 0x10), "not in"),
                          (add_args("game/b.cpp", "fail", 0x80001000, 0x10), "already declared"),
                          (add_args("game/a.cpp", "y", 0x800011F0, 0x20), "overlaps"),
                          (add_args("game/a.cpp", "y", 0x80001300, 0x10), "not inside one of"),
                          (add_args("game/a.cpp", "y", 0x80000100, 0x10), "code or zero-fill")):
            with self.subTest(why=why):
                code, out = quiet(assets.run, self.cfg, args)
                self.assertNotEqual(code, 0)
                self.assertIn(why, out)

    def test_add_refuses_a_range_holding_pointers(self):
        target = self.root / "build" / "DEMO01" / "obj" / "game" / "a.o"
        target.parent.mkdir(parents=True)
        target.write_bytes(make_object({".data": DATA[:0x300]}, [("lbl_80001100", ".data", 0x100, 0x200, GLOBAL)],
                                       [(".data", 0x180, "lbl_80001100")]))
        (self.root / "objdiff.json").write_text(json.dumps({"units": [
            {"name": "main/game/a", "target_path": "build/DEMO01/obj/game/a.o",
             "metadata": {"source_path": "src/game/a.cpp"}}]}))
        code, out = quiet(assets.run, self.cfg, add_args("game/a.cpp", "fail", 0x80001100, 0x200))
        self.assertEqual(code, 1)
        self.assertIn("1 relocation(s)", out)
        code, out = quiet(assets.run, self.cfg, add_args("game/a.cpp", "fail", 0x80001100, 0x80))
        self.assertEqual(code, 0, out)


def asset_outputs():
    """configure.py's asset_outputs, without running configure.py itself."""
    import tomllib
    tree = ast.parse((REPO / "configure.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "asset_outputs")
    ns = {"Path": Path, "List": List, "tomllib": tomllib}
    exec(compile(ast.Module([fn], []), "configure.py", "exec"), ns)
    return ns["asset_outputs"]


@unittest.skipUnless((REPO / "configure.py").is_file(), "needs the project's configure.py (game projects only)")
class ConfigureTest(unittest.TestCase):
    def test_the_build_writes_every_declared_inc_before_compiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "units.toml")
            path.write_text(UNITS.replace('cflags = "game" }',
                                          'cflags = "game", assets = [{ name = "t/x", address = 0x0, size = 0x4 }] }'))
            got = asset_outputs()(path, Path("build/DEMO01/include"))
            self.assertEqual(got, [Path("build/DEMO01/include/assets/t/x.inc")])
            self.assertEqual(asset_outputs()(Path(tmp, "missing.toml"), Path("x")), [])
        text = (REPO / "configure.py").read_text()
        self.assertIn('"$python -m tools.dk asset extract --quiet"', text)
        self.assertIn('"pre-compile"', text)


if __name__ == "__main__":
    unittest.main()
