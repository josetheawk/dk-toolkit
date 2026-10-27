# dk-toolkit

A deterministic toolkit for matching decompilation: one entry point (`./dk`), tested
commands with fixed exit codes, ratchets that stop a project from getting worse, and a
portable playbook of verified lessons.

Each decomp project vendors this toolkit at the same paths. Use `./dk toolkit export` and
`./dk toolkit import` inside a project to sync it with this repo.

## Layout

```
dk                       launcher: python3 -m tools.dk "$@"
tools/dk/core/           generic commands: build, verify-head, progress, floor, quality, docs, sync
tools/dk/platforms/      one package per platform; gc_mwcc = GameCube/Wii + CodeWarrior via decomp-toolkit
tests/                   toolkit tests (synthetic fixtures; golden tests skip without a project)
playbook/                portable lessons (no project names; ./dk doc-lint enforces it)
templates/               dk.toml and AGENTS.md for a new project
NEW_PROJECT.md           how to start a decomp with this toolkit
```

## What the gc_mwcc platform does

- **Analysis:**
  - `analyze rtti` gives class names and vtables.
  - `analyze lib` and `port` find and port SDK, MSL and runtime code from pinned reference decomps.
  - `unit propose` finds translation-unit boundaries, including the `__sinit` tails.
- **Matching:**
  - `check` scores a unit's functions, data and layout, modelling the original link's data dead-stripping.
  - `check --all` checks the whole project.
  - `diff` shows target against ours, side by side.
  - `compiler-search` tries compiler versions and flags.
- **Generation:** `gen` writes C++ drafts that are verified exact. It handles leaf functions, stack frames, C and C++ calls (with a CodeWarrior demangler), if/else, virtual calls, int↔float conversions, stack locals, and virtual destructors and constructors rebuilt from RTTI.
  - `gen --survey` measures coverage.
  - `gen --regress REV` ensures a change loses nothing.
- **Commits:**
  - `accept` and `save` commit HEAD plus exactly one unit's delta, verified in a clean worktree under a project lock (safe for parallel agents).
  - `verify-head` proves every commit builds on its own.

## Developing the toolkit

1. Change it inside a project, where the real binary exercises it.
2. Commit there.
3. Run `./dk toolkit export`, then commit here.
4. Here, run `python3 -m pytest -q tests`.

New lessons go in `playbook/`, in its format: rule → why → how to apply. Keep it within the
budgets.

## License

MIT, see [LICENSE](LICENSE). A copy also sits in `tools/dk/LICENSE`, so it travels with every
project's vendored toolkit.
