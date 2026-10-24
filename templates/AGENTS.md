# AGENTS.md

Matching decompilation of <GAME> (<platform>, `<GAMEID>`): C/C++ that rebuilds a
byte-identical `main.dol`. Read this file first; it is kept under 100 lines on purpose.

## Start here

1. `./dk status`: current state and the next command to run.
2. [playbook/README.md](playbook/README.md): portable rules (process, agents, matching).
3. [docs/<project>/overview.md](docs/<project>/overview.md): facts about this game and toolchain.

## Task → command

| Task | Command |
|---|---|
| What now? | `./dk status` |
| Fresh clone / regenerate build files | `./dk setup` |
| Build and verify the hash | `./dk build` |
| Prove a commit builds on its own (clean worktree) | `./dk verify-head [rev]` |
| Progress numbers | `./dk progress` |
| What to work on next (open functions, then units to split) | `./dk next` |
| Context pack: unit, class/vtable owner, calls, callers, asm, m2c draft | `./dk ctx <function>` |
| Verified exact C++ drafts (frames, calls, if/else, virtuals, ctors/dtors; prints only; `--survey` dry run, `--regress REV` before changing the generator) | `./dk gen <fn…> \| --unit U [--max-insns 40]` |
| Classes/vtables from RTTI → `docs/generated/classes.md` (`--apply` renames, `--check` for CI) | `./dk analyze rtti` |
| Find known library code (SDK…) via pinned reference decomps → `docs/generated/libraries.md` | `./dk analyze lib [--apply]` |
| Port located library objects to linked units (failures → `docs/generated/ports.md`) | `./dk port [ref:path…]` |
| Propose a unit's full split (text + data, externs) from a text range or function; fixes dtk's `__sinit` boundaries (`--raw` skips that) | `./dk unit propose <fn \| START END>` |
| Declare a translation unit (splits + units.toml + empty source) | `./dk unit add <src path> --lib L SECTION:0xA-0xB…` |
| Fix a unit's ranges (`--rename OLD NEW` in the same step, e.g. widening a linked unit) / trim data padding / undeclare it | `./dk unit set …` / `./dk unit fit <unit>` / `./dk unit remove <unit>` |
| Rename symbols to match source (mangled) names; `--local` for a static whose name another TU also uses | `./dk rename OLD NEW … [--local]` |
| Split / resize (absorbing auto labels) / add dtk's data labels and asm labels | `./dk symbol split ADDR` / `size NAME SIZE` / `add NAME SEC:ADDR [SIZE] [--label]` |
| List units with status and match % | `./dk unit list` |
| Score a unit or one function (functions, data, layout) | `./dk check <unit\|function> [--function F]` |
| Re-score every unit: linked must stay exact (exit 1 = regression), WIP ranked by % | `./dk check --all` |
| Side-by-side target vs ours for non-exact functions and data chunks | `./dk diff <unit> [--function F]` |
| Which compiler/flags reproduce a unit? | `./dk compiler-search <unit> [--flags '…']` |
| Link an exact unit, verify the hash, commit | `./dk accept <unit>` |
| Commit partial progress on a unit (not linked; build must match) | `./dk save <unit>` |
| Retire an obsolete header/source (refused while still included) | `./dk rm <path> --reason "…"` |
| Source quality: forbidden patterns (must be 0) + placeholder debt | `./dk quality [paths]` |
| Progress floor (accept/save refuse to go below it) | `./dk floor` |
| Run toolkit tests | `./dk test` |
| Check docs (budgets, links, portability) | `./dk doc-lint` |
| End of a phase | `./dk retro` |
| Inspect one unit's diff by hand | objdiff GUI, project dir = repo root |

Exit codes for every command: `0` ok/match, `1` mismatch, `2` usage/environment, `3` build error.

## Rules

- **This repo is decomp only.** No mods, features, fixes or ports. Those live in a separate repo that tracks this one as upstream. `--non-matching` / `Equivalent` exists only for functionally identical code.
- **The hash must match after every commit:** `./dk build` exits 0.
- **Improvements may never make things worse.** The progress floor, zero forbidden patterns and the tests gate every commit. Do refinement passes on your own earlier units (see [playbook/agents.md](playbook/agents.md)).
- **Never commit game data:** `orig/` and `build/` are git-ignored. No asm, no extracted bytes.
- **`config/<GAMEID>/{splits,symbols}.txt` change only in analysis work,** and `./dk build` runs right after. Agents matching functions edit only `src/` and `include/`.
- **Agent output follows [playbook/agents.md](playbook/agents.md):** exact match only, review checklist, one commit per match.
- **Commit only through `./dk accept`, `./dk save` or `./dk rm`.** Each commits HEAD plus exactly that unit's own delta, verified to build and match in a clean worktree first. Never run raw git (`add`, `commit`, `rm`, `reset`, `stash`).
- **A new repeatable step becomes a `./dk` command** in `tools/dk/` with a test in `tests/`. Don't leave one-off scripts lying around.
- **Docs go where they belong:**
  - generic lesson → `playbook/` (no project names)
  - game fact → `docs/<project>/`
  - tool output → `docs/generated/` (never hand-edited)
- **End each phase with `./dk retro`,** and keep docs within budget (`./dk doc-lint`).

## Layout

```
dk, dk.toml           toolkit entry point + its only project config
configure.py          dtk-template build config (tool pins, compiler flags, units)
config/<GAMEID>/        dtk config.yml, splits.txt, symbols.txt, build.sha1
orig/<GAMEID>/          original game files (ignored)
src/, include/        decompiled code
tools/dk/core/        generic toolkit commands
tools/dk/platforms/   platform code (gc_mwcc: DOL, dtk files, first-diff)
tools/*.py            dtk-template helpers (leave as upstream)
tests/                toolkit tests (synthetic fixtures + golden tests that skip without orig/)
playbook/             portable knowledge
docs/<project>/             this game: overview, decisions, formats
docs/generated/       dk output
docs/upstream/        vendored upstream docs (reference only)
```
