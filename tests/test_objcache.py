"""The shared compile/objdiff cache (objcache.py): keys, round trips, the spot check, the objdiff memo."""

import os
import shutil
import stat
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.dk.platforms.gc_mwcc import objcache


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "repo"
        (self.root / "include").mkdir(parents=True)
        (self.root / "src").mkdir()
        (self.root / "include" / "a.h").write_text("int a;\n")
        self.cfg = SimpleNamespace(root=self.root)
        self.out = self.tmp / "out"
        self.out.mkdir()
        p = mock.patch.object(objcache, "main_root", lambda cfg: self.tmp / "main")
        p.start()
        self.addCleanup(p.stop)
        t = mock.patch.object(objcache, "FINGERPRINT_TTL", 0.0)
        t.start()
        self.addCleanup(t.stop)
        d = mock.patch.object(objcache, "_disabled", False)
        d.start()
        self.addCleanup(d.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def argv(self):
        return ["wibo", "mwcceppc.exe", "-O4,p", "-i", "include", "-c", "src/x.cpp", "-o", str(self.out)]

    def test_key_follows_source_flags_and_headers(self):
        k = objcache.compile_key(self.cfg, self.argv(), self.out, "int f();")
        self.assertEqual(k, objcache.compile_key(self.cfg, self.argv(), self.out, "int f();"))
        self.assertNotEqual(k, objcache.compile_key(self.cfg, self.argv(), self.out, "int g();"))
        other = self.argv()
        other[2] = "-O4,s"
        self.assertNotEqual(k, objcache.compile_key(self.cfg, other, self.out, "int f();"))
        (self.root / "include" / "a.h").write_text("int b;\n")
        self.assertNotEqual(k, objcache.compile_key(self.cfg, self.argv(), self.out, "int f();"))

    def test_key_ignores_the_output_dir_and_worktree_path(self):
        k = objcache.compile_key(self.cfg, self.argv(), self.out, "x")
        other_out = self.tmp / "elsewhere"
        argv = self.argv()
        argv[-1] = str(other_out)
        self.assertEqual(k, objcache.compile_key(self.cfg, argv, other_out, "x"))
        copy = self.tmp / "worktree"
        shutil.copytree(self.root, copy)
        argv = [a.replace(str(self.root), str(copy)) for a in self.argv()]
        self.assertEqual(k, objcache.compile_key(SimpleNamespace(root=copy), argv, self.out, "x"))

    def test_round_trip_restores_the_object(self):
        (self.out / "x.o").write_bytes(b"\x7fELF obj")
        objcache.save(self.cfg, "ab" * 20, self.out, True, "warn")
        (self.out / "x.o").unlink()
        self.assertEqual(objcache.lookup(self.cfg, "ab" * 20, self.out), (True, "warn"))
        self.assertEqual((self.out / "x.o").read_bytes(), b"\x7fELF obj")
        self.assertIsNone(objcache.lookup(self.cfg, "cd" * 20, self.out))

    def test_failed_compiles_are_cached_without_objects(self):
        objcache.save(self.cfg, "ef" * 20, self.out, False, "error: x")
        self.assertEqual(objcache.lookup(self.cfg, "ef" * 20, self.out), (False, "error: x"))

    def test_spot_check_mismatch_turns_the_cache_off(self):
        (self.out / "x.o").write_bytes(b"one")
        objcache.save(self.cfg, "12" * 20, self.out, True, "")
        (self.out / "x.o").write_bytes(b"two")
        with mock.patch("sys.stderr"):
            objcache.verify(self.cfg, "12" * 20, self.out, True, "")
        self.assertTrue(objcache.stats()["disabled"])
        self.assertIsNone(objcache.lookup(self.cfg, "12" * 20, self.out))
        self.assertIn("differs", (objcache.store(self.cfg) / "MISMATCH.log").read_text())

    def test_prune_drops_old_entries_only(self):
        (self.out / "x.o").write_bytes(b"o")
        objcache.save(self.cfg, "aa" * 20, self.out, True, "")
        objcache.save(self.cfg, "bb" * 20, self.out, True, "")
        os.utime(objcache._entry(self.cfg, "aa" * 20), (0, 0))
        self.assertEqual(objcache.prune(self.cfg), 1)
        self.assertIsNone(objcache.lookup(self.cfg, "aa" * 20, self.out))
        self.assertIsNotNone(objcache.lookup(self.cfg, "bb" * 20, self.out))

    def test_prune_keeps_the_store_under_its_size_cap(self):
        (self.out / "x.o").write_bytes(b"o" * 1000)
        for k, key in enumerate(("aa", "bb", "cc")):
            objcache.save(self.cfg, key * 20, self.out, True, "")
            os.utime(objcache._entry(self.cfg, key * 20), (time.time() - 100 + k, time.time() - 100 + k))
        self.assertEqual(objcache.prune(self.cfg, max_bytes=2100), 1)
        self.assertIsNone(objcache.lookup(self.cfg, "aa" * 20, self.out))      # the oldest went first
        self.assertIsNotNone(objcache.lookup(self.cfg, "cc" * 20, self.out))


class ObjdiffTest(unittest.TestCase):
    def test_same_objects_run_objdiff_once(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        tools = tmp / "build" / "tools"
        tools.mkdir(parents=True)
        calls = tmp / "calls"
        script = tools / "objdiff-cli"
        script.write_text(f'#!/bin/sh\necho x >> {calls}\necho \'{{"left": {{"symbols": []}}}}\'\n')
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        (tmp / "t.o").write_bytes(b"t")
        (tmp / "o.o").write_bytes(b"o")
        cfg = SimpleNamespace(root=tmp)
        for _ in range(3):
            doc, err = objcache.objdiff_json(cfg, "t.o", "o.o")
            self.assertEqual((doc, err), ({"left": {"symbols": []}}, ""))
        self.assertEqual(calls.read_text().count("x"), 1)
        (tmp / "o.o").write_bytes(b"changed")
        objcache.objdiff_json(cfg, "t.o", "o.o")
        self.assertEqual(calls.read_text().count("x"), 2)

    def test_function_pair_is_kept_on_disk(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        tools = tmp / "build" / "tools"
        tools.mkdir(parents=True)
        calls = tmp / "calls"
        script = tools / "objdiff-cli"
        doc = '{"left": {"symbols": [{"name": "f", "instructions": [1]}]}, "right": {"symbols": [{"name": "f", "instructions": [2]}]}}'
        script.write_text(f"#!/bin/sh\necho x >> {calls}\necho '{doc}'\n")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        (tmp / "t.o").write_bytes(b"t2")
        (tmp / "o.o").write_bytes(b"o2")
        cfg = SimpleNamespace(root=tmp)
        with mock.patch.object(objcache, "main_root", lambda cfg: tmp), mock.patch.object(objcache, "_disabled", False):
            first = objcache.function_pair(cfg, "t.o", "o.o", "f")
            objcache._DIFFS.clear()                       # a new process: only the disk entry is left
            self.assertEqual(objcache.function_pair(cfg, "t.o", "o.o", "f"), first)
            self.assertIsNone(objcache.function_pair(cfg, "t.o", "o.o", "g"))
        self.assertEqual(first, ({"name": "f", "instructions": [1]}, {"name": "f", "instructions": [2]}))
        self.assertEqual(calls.read_text().count("x"), 2)  # f once, g once

    def test_symbol_pair(self):
        doc = {"left": {"symbols": [{"name": "f", "instructions": [1]}]},
               "right": {"symbols": [{"name": "f", "instructions": [2]}, {"name": "g"}]}}
        self.assertEqual(objcache.symbol_pair(doc, "f"), ({"name": "f", "instructions": [1]},
                                                          {"name": "f", "instructions": [2]}))
        self.assertIsNone(objcache.symbol_pair(doc, "g"))


if __name__ == "__main__":
    unittest.main()
