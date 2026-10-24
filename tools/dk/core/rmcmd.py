"""dk rm <path>...: retire obsolete source/header files safely.

Refuses while anything still #includes a file, deletes it, proves the build still matches,
and commits exactly the deletion. Agents never need raw `git rm`.
"""

from __future__ import annotations

import argparse
import re

from .command import MISMATCH, OK, USAGE, Command, hint
from .config import Config
from .gitops import tracked
from .isolated import commit_overlay, verify_overlay
from .util import run_ninja

SOURCE_SUFFIXES = (".c", ".cpp", ".cp", ".h", ".hpp", ".inc")


def includers(cfg: Config, rel: str) -> list[str]:
    name = rel.rsplit("/", 1)[-1]
    pattern = re.compile(r'#\s*include\s*[<"]([^>"]*/)?' + re.escape(name) + r'[>"]')
    hits = []
    for root in (cfg.root / "src", cfg.root / "include"):
        for f in sorted(root.rglob("*")) if root.is_dir() else []:
            if f.suffix in SOURCE_SUFFIXES and f.is_file() and cfg.rel(f) != rel and pattern.search(f.read_text(errors="replace")):
                hits.append(cfg.rel(f))
    return hits


def run(cfg: Config, args: argparse.Namespace) -> int:
    rels = [p.removeprefix("./") for p in args.paths]
    for rel in rels:
        if not rel.startswith(("src/", "include/")):
            print(f"error: {rel}: only files under src/ or include/ can be removed this way")
            return USAGE
        if not (cfg.root / rel).is_file() and not tracked(cfg.root, rel):
            print(f"error: {rel} does not exist")
            return USAGE
        users = includers(cfg, rel)
        if users:
            print(f"REJECTED: {rel} is still included by: {', '.join(users)}")
            return MISMATCH
    saved = {rel: (cfg.root / rel).read_bytes() for rel in rels if (cfg.root / rel).is_file()}
    for rel in saved:
        (cfg.root / rel).unlink()
    if run_ninja(cfg.root).returncode != 0:
        for rel, data in saved.items():
            (cfg.root / rel).write_bytes(data)
        run_ninja(cfg.root)
        print("REJECTED: the build no longer matches without these files; restored")
        hint("build --verbose")
        return MISMATCH
    message = "Remove obsolete " + ", ".join(rels) + "\n"
    if args.reason:
        message += f"\n{args.reason}\n"
    if args.trailer:
        message += "\n" + "\n".join(args.trailer) + "\n"
    overlay = {rel: None for rel in rels}
    ok, detail = verify_overlay(cfg, overlay)
    if ok:
        ok, detail = commit_overlay(cfg, overlay, message)
    else:
        detail = "HEAD without these files does not build: " + detail
    print(f"removed and committed ({detail})" if ok else f"removed; note: not committed ({detail})")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("paths", nargs="+", help="files under src/ or include/")
    p.add_argument("--reason", help="why it is obsolete (goes in the commit message)")
    p.add_argument("--trailer", action="append", default=[], help="commit trailer line")


COMMAND = Command("rm", "matching", "retire obsolete files: refuse if still included, prove the hash, commit", run,
                  _configure, locked=True)
