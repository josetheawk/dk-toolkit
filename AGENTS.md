# AGENTS.md

This is dk-toolkit, a deterministic toolkit for decompilation and reverse engineering of any
binary. A generic core plus one platform per kind of target: `raw` (any binary), `spi_nor`
(flash dumps, on `raw`), `dotnet` (.NET assemblies), `gc_mwcc` (GameCube/Wii + CodeWarrior
matching, layered on decomp-toolkit, objdiff and dtk-template). It is designed to be driven
by AI coding agents, alone or many at once. Read this first.

## If you were asked to start or work on a decomp

Don't work in this repo. The toolkit is vendored into each decomp project:

1. Follow [NEW_PROJECT.md](NEW_PROJECT.md) to set up the project and copy the toolkit in.
2. In the project, its `AGENTS.md` (from [templates/AGENTS.md](templates/AGENTS.md) or [templates/AGENTS-raw.md](templates/AGENTS-raw.md)) is your map: task → command.
3. Read [playbook/README.md](playbook/README.md). [playbook/agents.md](playbook/agents.md) has the matching loop, the guard rails, and how to run many agents at once. [playbook/matching.md](playbook/matching.md) and [playbook/platforms/gamecube-mwcc.md](playbook/platforms/gamecube-mwcc.md) map asm symptoms to the source idioms that fixed them.

Ground rules that make it work:
- **Only byte-exact counts.**
- **On matching platforms, commit only through `./dk`** (`save`, `accept`, `commit-tools`). Each commit is verified in a clean worktree and refused below the progress floor.
- **No matching devices.** That means no dummy functions or fake padding, and no pragma or `volatile` added only to force a match. Record any exception in the project's decision log.
- **Branch on exit codes,** not on output text: `0` ok, `1` mismatch, `2` usage, `3` build error. Every command ends with a `next:` hint.

## Platforms

`project.platform` in dk.toml names one package under `tools/dk/platforms/`. A platform may
set `BASE` to build on another: it gets the base's commands and hooks unless it defines its
own, and `SKIP_CORE` hides core commands that do not apply.

- `raw` — any binary. `map`, `carve`, `dis`. Read [playbook/platforms/raw.md](playbook/platforms/raw.md).
- `spi_nor` — SPI NOR dumps, on `raw`: erased 0xFF sectors, geometry from the length. Read [playbook/platforms/spi-nor.md](playbook/platforms/spi-nor.md).
- `dotnet` — .NET (CLI) assemblies: metadata, IL, strings, resources. Read [playbook/platforms/dotnet.md](playbook/platforms/dotnet.md).
- `gc_mwcc` — GameCube/Wii, CodeWarrior. Matching commands (`check`, `gen`, `accept`, …).

To specialise, add a platform with `BASE` set; do not change another platform's behavior to
serve a new target. Generic logic goes in `tools/dk/core/`, which a live decomp shares: keep
core changes small and list them in the commit.

## If you are changing the toolkit itself

- Make the change inside a project that vendors the toolkit, where a real binary exercises it.
  Commit it there, run `./dk toolkit export`, then commit here. Only `dk`, `tools/dk/`, `tests/`
  and `playbook/` are synced; this file, the README, NEW_PROJECT.md and `templates/` are edited here.
- **Every command has a test.** Run `python3 -m pytest -q tests`.
- **Output is deterministic:** sorted, no timestamps.
- **Python is stdlib-first.**
- **Generic logic goes in `tools/dk/core/`; platform logic in `tools/dk/platforms/<platform>/`.** Project specifics come from the project's `dk.toml`.
- **Nothing here names a game.** Synthetic fixtures and addresses only; a project's golden tests live in its
  own `[test] paths` folder. `./dk toolkit export` refuses on a scan hit (`--check` runs the scan alone).
- **Playbook entries** follow the format rule → why → how to apply, use no project names, stay within the budgets, and are each verified by an exact match. `./dk doc-lint` checks all of this.
