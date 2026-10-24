"""The progress floor: committed numbers that may only go up.

accept/save compare the fresh objdiff report with config/<id>/floor.toml and refuse any
state below it, then raise the floor. A refinement that silently breaks a match therefore
cannot be committed, and progress is monotonic by construction.
"""

from __future__ import annotations

import argparse
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


def check(cfg: Config) -> list[str]:
    """Regressions against the floor, as messages (empty = OK)."""
    now, floor = current(cfg), read(cfg)
    return [f"{k} {now[k]} < floor {floor[k]}" for k in KEYS if now[k] < floor[k]]


def raise_floor(cfg: Config) -> bool:
    now, floor = current(cfg), read(cfg)
    new = {k: max(now[k], floor[k]) for k in KEYS}
    if new == floor and path(cfg).is_file():
        return False
    body = "# Progress floor: may only go up. Written by ./dk accept and ./dk save.\n"
    body += "".join(f"{k} = {new[k]}\n" for k in KEYS)
    path(cfg).write_text(body)
    return True


def run(cfg: Config, args: argparse.Namespace) -> int:
    if args.lower:
        if not args.reason:
            print("error: --lower needs --reason (it is recorded in the floor file)")
            return USAGE
        now = current(cfg)
        body = f"# Progress floor: may only go up. Lowered once: {args.reason}\n"
        body += "".join(f"{k} = {now[k]}\n" for k in KEYS)
        path(cfg).write_text(body)
        print("floor lowered to the current state")
        return OK
    now, floor = current(cfg), read(cfg)
    for k in KEYS:
        print(f"  {k:<18} now {now[k]:>8}   floor {floor[k]:>8}{'   BELOW' if now[k] < floor[k] else ''}")
    return MISMATCH if check(cfg) else OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lower", action="store_true", help="reset the floor to the current state (needs --reason; not for agents)")
    p.add_argument("--reason", help="why the floor is being lowered")


COMMAND = Command("floor", "build", "progress floor (matched/linked) that accept/save refuse to go below", run, _configure,
                  locked=True)
