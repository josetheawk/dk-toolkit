"""The progress floor: committed numbers that may only go up.

accept/save/commit-tools build HEAD plus the change in a clean worktree, refuse it when its
report is below the committed config/<id>/floor.toml, and commit the floor raised to it. A refinement that silently breaks a match therefore
cannot be committed, and progress is monotonic by construction.
"""

from __future__ import annotations

import argparse
import subprocess
import tomllib
from pathlib import Path

from .command import MISMATCH, OK, USAGE, Command
from .config import Config
from .progress import load_report

KEYS = ("matched_functions", "complete_code", "complete_units")


def path(cfg: Config) -> Path:
    return cfg.root / cfg.quality.get("floor", f"config/{cfg.game_id}/floor.toml")


def current(cfg: Config) -> dict[str, int]:
    measures = (load_report(cfg) or {}).get("measures", {})
    return {k: int(measures.get(k, 0) or 0) for k in KEYS}


def read(cfg: Config) -> dict[str, int]:
    p = path(cfg)
    if not p.is_file():
        return {k: 0 for k in KEYS}
    data = tomllib.loads(p.read_text())
    return {k: int(data.get(k, 0)) for k in KEYS}


def parse(text: str) -> dict[str, int]:
    data = tomllib.loads(text)
    return {k: int(data.get(k, 0)) for k in KEYS}


def committed(cfg: Config) -> dict[str, int]:
    """The floor as committed at HEAD (the shared working copy may be ahead or behind)."""
    proc = subprocess.run(["git", "show", f"HEAD:{cfg.rel(path(cfg))}"], cwd=cfg.root, capture_output=True, text=True)
    return parse(proc.stdout) if proc.returncode == 0 else {k: 0 for k in KEYS}


def gate(cfg: Config, measures: dict) -> tuple[list[str], str]:
    """For a commit whose clean-tree build produced `measures`: its regressions against the
    committed floor (empty = OK), and the floor file it commits (raised to what it reaches)."""
    now = {k: int(measures.get(k, 0) or 0) for k in KEYS}
    floor = committed(cfg)
    below = [f"{k} {now[k]} < floor {floor[k]}" for k in KEYS if now[k] < floor[k]]
    return below, render({k: max(now[k], floor[k]) for k in KEYS})


def render(values: dict[str, int]) -> str:
    return ("# Progress floor: may only go up. Written by ./dk accept and ./dk save.\n"
            + "".join(f"{k} = {values[k]}\n" for k in KEYS))


def check(cfg: Config) -> list[str]:
    """Regressions against the floor, as messages (empty = OK)."""
    now, floor = current(cfg), read(cfg)
    return [f"{k} {now[k]} < floor {floor[k]}" for k in KEYS if now[k] < floor[k]]


def raise_floor(cfg: Config) -> bool:
    now, floor = current(cfg), read(cfg)
    new = {k: max(now[k], floor[k]) for k in KEYS}
    if new == floor and path(cfg).is_file():
        return False
    path(cfg).write_text(render(new))
    return True


def run(cfg: Config, args: argparse.Namespace) -> int:
    if args.lower:
        if not args.reason:
            print("error: --lower needs --reason (it is recorded in the floor file)")
            return USAGE
        # to what HEAD itself builds (a clean checkout), committed with the reason
        from . import isolated
        measures: dict = {}
        ok, detail = isolated.verify_overlay(cfg, {}, measures=measures)
        if not ok or not measures:
            print(f"REFUSED: HEAD does not build and match on its own:\n{detail}")
            return MISMATCH
        body = render({k: int(measures.get(k, 0) or 0) for k in KEYS})
        rel = cfg.rel(path(cfg))
        ok, detail = isolated.commit_overlay(cfg, {rel: body.encode()}, f"Lower the progress floor to HEAD's clean build\n\n{args.reason}\n")
        if not ok:
            print(f"REFUSED: {detail}")
            return MISMATCH
        path(cfg).write_text(body)
        print(f"floor set to HEAD's clean build and committed ({detail})")
        return OK
    now, floor = current(cfg), read(cfg)
    for k in KEYS:
        print(f"  {k:<18} now {now[k]:>8}   floor {floor[k]:>8}{'   BELOW' if now[k] < floor[k] else ''}")
    return MISMATCH if check(cfg) else OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lower", action="store_true", help="reset the floor to what HEAD builds in a clean checkout, committed (needs --reason; not for agents)")
    p.add_argument("--reason", help="why the floor is being lowered")


COMMAND = Command("floor", "build", "progress floor (matched/linked) that accept/save refuse to go below", run, _configure,
                  locked=True)
