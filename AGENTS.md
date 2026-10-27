# AGENTS.md

This is dk-toolkit, a deterministic toolkit for matching decompilation of GameCube/Wii games
built with CodeWarrior, layered on decomp-toolkit, objdiff and dtk-template. It is designed
to be driven by AI coding agents, alone or many at once. Read this first.

## If you were asked to start or work on a decomp

Don't work in this repo. The toolkit is vendored into each decomp project:

1. Follow [NEW_PROJECT.md](NEW_PROJECT.md) to set up the project and copy the toolkit in.
2. In the project, [templates/AGENTS.md](templates/AGENTS.md) (filled in) is your map: task → command.
3. Read [playbook/README.md](playbook/README.md). [playbook/agents.md](playbook/agents.md) has the matching loop, the guard rails, and how to run many agents at once. [playbook/matching.md](playbook/matching.md) and [playbook/platforms/gamecube-mwcc.md](playbook/platforms/gamecube-mwcc.md) map asm symptoms to the source idioms that fixed them.

Ground rules that make it work:
- **Only byte-exact counts.**
- **Commit only through `./dk`** (`save`, `accept`, `commit-tools`). Each commit is verified in a clean worktree and refused below the progress floor.
- **No matching devices.** That means no dummy functions or fake padding, and no pragma or `volatile` added only to force a match. Record any exception in the project's decision log.
- **Branch on exit codes,** not on output text: `0` ok, `1` mismatch, `2` usage, `3` build error. Every command ends with a `next:` hint.

## If you are changing the toolkit itself

- Make the change inside a project that vendors the toolkit, where a real binary exercises it.
  Commit it there, run `./dk toolkit export`, then commit here. Only `dk`, `tools/dk/`, `tests/`
  and `playbook/` are synced; this file, the README, NEW_PROJECT.md and `templates/` are edited here.
- **Every command has a test.** Run `python3 -m pytest -q tests`.
- **Output is deterministic:** sorted, no timestamps.
- **Python is stdlib-first.**
- **Generic logic goes in `tools/dk/core/`; platform logic in `tools/dk/platforms/<platform>/`.** Project specifics come from the project's `dk.toml`.
- **Playbook entries** follow the format rule → why → how to apply, use no project names, stay within the budgets, and are each verified by an exact match. `./dk doc-lint` checks all of this.
