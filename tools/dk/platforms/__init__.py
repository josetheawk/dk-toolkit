"""Platform packages. A project names one in dk.toml `project.platform`.

A platform is what the toolkit knows about one kind of target. `raw` works on any binary
(layout, signatures, carving, disassembly). A specialised platform sets `BASE` to build on
another one: `spi_nor` is `raw` plus flash geometry. `dotnet` reads .NET assemblies.
`gc_mwcc` (GameCube/Wii + CodeWarrior) is a matching platform with no base.
Generic logic stays in tools/dk/core/.

Each one exposes:

    COMMANDS: list[Command]               extra platform-specific commands
    BASE: str                             optional: the platform this one builds on; its commands and
                                          hooks apply unless this one defines the same name
    SKIP_CORE: tuple[str, ...]            optional: core commands that do not apply (an analysis-only
                                          platform has no ninja build, report or progress floor)
    first_diff(cfg, built, original, limit) -> list[str]
                                          human summary of where two binaries differ
    prebuild(cfg)                         optional: called before `dk build` runs ninja
    after_config_sync(cfg) -> bool        optional: `dk worktree sync` brought config/ changes (re-derive build inputs)
    link_queue(cfg, limit) -> list[dict]  optional: `dk next --link`: near-link units and their blockers
    declared_sources(cfg, read) -> set    optional: source paths the unit list declares (`dk rm` keeps them)
    WORKTREE_SETUP: tuple of argv tuples   optional: dk commands `dk worktree add` runs after the first build
    brief_near(cfg, unit_names) -> list   optional: `dk brief` lines on units nearly exact (scored now)
    frozen_functions(cfg) -> dict         optional: the one frozen union {function: set} (core.frozen.FrozenMap: also
                                          looked up by (unit, function)); every queue skips them, `dk next`/`dk brief` mark them
    frozen_definitions(cfg) -> (dict, str) optional: {function: [(source name, param count)]} and the release file;
                                          commits that change a frozen function's source text are refused (core.frozen.guard)
    status_checks(cfg) -> list            optional: `dk status` rows (passed, text, next command)
    status_ready_hint(cfg) -> str         optional: `dk status` next command when every row passes
    usage_fields(cfg, args) -> dict       optional: usage-log fields read from the parsed arguments (core/usage.py)
    USAGE_SEARCH: {label: (cmd, flags)}   optional: search tools `dk usage report` rates by later landings
    USAGE_LANDING: tuple[str, ...]        optional: commands whose committed run notes the functions it `landed`
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


def load(name: str) -> ModuleType:
    return importlib.import_module(f"{__name__}.{name}")


def chain(name: str) -> list[ModuleType]:
    """The platform, then its BASE, and so on."""
    out: list[ModuleType] = []
    seen: set[str] = set()
    while name and name not in seen:
        seen.add(name)
        module = load(name)
        out.append(module)
        name = getattr(module, "BASE", "")
    return out


def hook(name: str, attr: str) -> Any:
    """The first definition of `attr` along the platform's chain, or None."""
    for module in chain(name):
        if hasattr(module, attr):
            return getattr(module, attr)
    return None


def commands(name: str) -> list:
    """The chain's COMMANDS; a platform's own command replaces a base's command of the same name."""
    seen: set[str] = set()
    out = []
    for module in chain(name):
        for cmd in getattr(module, "COMMANDS", []):
            if cmd.name not in seen:
                seen.add(cmd.name)
                out.append(cmd)
    return out
