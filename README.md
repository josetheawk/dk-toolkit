# dk-toolkit

A deterministic toolkit for matching decompilation: one entry point (`./dk`), tested
commands with fixed exit codes, ratchets that stop a project from getting worse, and a
portable playbook of verified lessons.

Each decomp project vendors this toolkit at the same paths. Use `./dk toolkit export` and
`./dk toolkit import` inside a project to sync it with this repo.

It targets GameCube/Wii games built with CodeWarrior, on top of
[decomp-toolkit](https://github.com/encounter/decomp-toolkit),
[objdiff](https://github.com/encounter/objdiff) and
[dtk-template](https://github.com/encounter/dtk-template). It is built to be driven by AI
coding agents, alone or as a team; see [Using it with AI coding agents](#using-it-with-ai-coding-agents).

## Using it with AI coding agents

The toolkit was developed with Claude Code running up to ~16 agents in parallel on one
real GameCube decomp. In one session that team took the game's matched code from 41% to 49%,
with every commit verified. What makes that safe:

- **One entry point and machine-readable outcomes.** Every command is `./dk <command>` with `-h`. Exit codes are fixed: `0` ok, `1` mismatch, `2` usage, `3` build error. Every command ends with a `next:` hint, so an agent branches on codes instead of parsing prose.
- **Only byte-exact counts.** `check` scores functions, data and layout against the original. `accept` links a unit only if the whole binary's hash still matches.
- **Agents can't make a project worse.** `save`, `accept` and `commit-tools` commit HEAD plus exactly their own delta, verified in a clean worktree under a lock, and refuse anything below the committed progress floor. So many agents can share one working tree.
- **Integration that tolerates work in progress.** The coordinator's `commit-tools --integrate` lands cross-unit changes (header retypes, symbol renames) together. It holds back units an agent is still editing.
- **Verified knowledge, not guesses.** `playbook/` maps asm symptoms to the source idioms that fixed them. It also covers the compiler's register-allocation and emission-order rules. Each entry was proven by an exact match.

To point an agent at it:
- [AGENTS.md](AGENTS.md) (also read as `CLAUDE.md`) orients an agent in this repo.
- [NEW_PROJECT.md](NEW_PROJECT.md) sets up a decomp.
- [templates/AGENTS.md](templates/AGENTS.md) becomes the project's map.
- [playbook/agents.md](playbook/agents.md) covers the matching loop, the guard rails and running many agents at once: roles, briefs, integration, context hand-offs, wind-down.

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
