"""The progress floor: committed numbers that may only go up.

accept/save/commit-tools build HEAD plus the change in a clean worktree, refuse it when its
report is below the committed config/<id>/floor.toml, and commit the floor raised to it. A refinement that silently breaks a match therefore
cannot be committed, and progress is monotonic by construction.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from .command import MISMATCH, OK, USAGE, Command, hint
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


def units(text: str) -> dict[str, int]:
    """The floor file's per-unit table: exact functions per unit at the commit that wrote it."""
    return {k: int(v) for k, v in tomllib.loads(text).get("units", {}).items()}


def committed_units(cfg: Config) -> dict[str, int]:
    """HEAD's per-unit exact-function counts, from its floor file (empty before the first
    commit that records them)."""
    from . import isolated
    data = isolated.show(cfg, cfg.rel(path(cfg)))
    return units(data.decode()) if data is not None else {}


def committed(cfg: Config) -> dict[str, int]:
    """The floor as committed at the verified (pinned) HEAD; the shared working copy may be
    ahead or behind. A newer floor lands with a newer HEAD, and then the commit is retried."""
    from . import isolated
    data = isolated.show(cfg, cfg.rel(path(cfg)))
    return parse(data.decode()) if data is not None else {k: 0 for k in KEYS}


def gate(cfg: Config, measures: dict) -> tuple[list[str], str]:
    """For a commit whose clean-tree build produced `measures`: its regressions against the
    committed floor (empty = OK), and the floor file it commits (raised to what it reaches)."""
    now = {k: int(measures.get(k, 0) or 0) for k in KEYS}
    floor = committed(cfg)
    below = [f"{k} {now[k]} < floor {floor[k]}" for k in KEYS if now[k] < floor[k]]
    return below, render({k: max(now[k], floor[k]) for k in KEYS}, measures.get("units"))


def render(values: dict[str, int], per_unit: dict[str, int] | None = None) -> str:
    text = ("# Progress floor: may only go up. Written by ./dk accept and ./dk save.\n"
            + "".join(f"{k} = {values[k]}\n" for k in KEYS))
    if per_unit:
        text += ("\n# Exact functions per unit at this commit, as objdiff's report counts them: a function whose\n"
                 "# only difference is where a relocation points (literal or string order) counts here, while\n"
                 "# `dk check` does not count it, so check can show fewer. Not a floor: an integration holds\n"
                 "# back a unit whose count fell below it.\n[units]\n"
                 + "".join(f'"{name}" = {count}\n' for name, count in sorted(per_unit.items())))
    return text


def check(cfg: Config) -> list[str]:
    """Regressions against the floor, as messages (empty = OK)."""
    now, floor = current(cfg), read(cfg)
    return [f"{k} {now[k]} < floor {floor[k]}" for k in KEYS if now[k] < floor[k]]


def run(cfg: Config, args: argparse.Namespace) -> int:
    if args.lower:
        if not args.reason:
            print("error: --lower needs --reason (it is recorded in the floor file)")
            return USAGE
        # to what HEAD itself builds (a clean checkout), committed with the reason
        from . import isolated
        measures: dict = {}
        try:
            isolated.pin_head(cfg, isolated.UNLOCKED_TRIES)     # one attempt, holding the land lock
            ok, detail = isolated.verify_overlay(cfg, {}, measures=measures)
        except isolated.LandError as e:
            ok, detail = False, str(e)
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
        hint("status")
        return OK
    now, floor = current(cfg), read(cfg)
    for k in KEYS:
        print(f"  {k:<18} now {now[k]:>8}   floor {floor[k]:>8}{'   BELOW' if now[k] < floor[k] else ''}")
    if args.units:
        print_units(cfg)
    below = check(cfg)
    hint("check --all" if below else "status")
    return MISMATCH if below else OK


def unit_differences(now: dict[str, int], table: dict[str, int]) -> tuple[list, list, list]:
    """(units below the table, units above it, units the table has that the build does not), each
    as (unit, now, table) sorted by name. A unit missing from the build was renamed, split or merged:
    its functions count where they live now."""
    below = sorted((u, now[u], n) for u, n in table.items() if u in now and now[u] < n)
    above = sorted((u, c, table.get(u, 0)) for u, c in now.items() if c > table.get(u, 0))
    gone = sorted((u, 0, n) for u, n in table.items() if u not in now and n)
    return below, above, gone


def print_units(cfg: Config) -> None:
    """This build's per-unit counts against the table in the floor file."""
    report = load_report(cfg) or {}
    now = {u["name"]: int(u.get("measures", {}).get("matched_functions", 0) or 0)
           for u in report.get("units", []) if u.get("name")}
    p = path(cfg)
    table = units(p.read_text()) if p.is_file() else {}
    if not table:
        print("  per unit: the floor file has no per-unit table yet")
        return
    below, above, gone = unit_differences(now, table)
    print(f"  per unit: {len(table)} in the floor file's table, {len(now)} in this build's report; "
          f"{len(below)} below, {len(above)} above, {len(gone)} not in this build")
    print("    (both count objdiff's matched functions; `dk check` also checks where relocations point, so it can "
          "show fewer than either. A difference here means this build is not the commit's clean build: "
          "an edited or stale tree, or functions that moved between units)")
    for label, rows in (("below", below), ("above", above), ("not in this build (renamed, split or merged)", gone)):
        for u, c, n in rows[:20]:
            print(f"    {label}: {u} {c} (table {n})")
        if len(rows) > 20:
            print(f"    {label}: ... {len(rows) - 20} more")


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lower", action="store_true", help="reset the floor to what HEAD builds in a clean checkout, committed (needs --reason; not for agents)")
    p.add_argument("--reason", help="why the floor is being lowered")
    p.add_argument("--units", action="store_true", help="also compare this build's per-unit counts with the floor file's table")


COMMAND = Command("floor", "build", "progress floor (matched/linked) that accept/save refuse to go below", run, _configure,
                  locked=True)
