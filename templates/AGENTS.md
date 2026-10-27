# AGENTS.md

Matching decompilation of <GAME> (<platform>, `<GAMEID>`): C/C++ that rebuilds a
byte-identical `main.dol`. Read this file first; it is kept under 100 lines on purpose.

## Start here

1. `./dk status`: current state and the next command to run.
2. [playbook/README.md](playbook/README.md): portable rules (process, agents, matching).
3. [docs/<project>/overview.md](docs/<project>/overview.md): facts about this game and toolchain.

## Task → command

This table is the one index of the toolkit. Every command explains its options with `-h`.

| Task | Command |
|---|---|
| What now? / progress numbers / what to work on next (open functions, then units to split) | `./dk status` / `./dk progress` / `./dk next` |
| Fresh clone, regenerate build files | `./dk setup` |
| Build and verify the hash / prove HEAD builds on its own in a clean worktree | `./dk build` / `./dk verify-head` |
| Context pack: unit, class and vtable owner, calls, callers, asm, m2c draft | `./dk ctx <fn>` |
| Verified exact C++ drafts; `--unverified` adds the closest non-exact ones; `--asm` writes code the original wrote in assembly as a verified `asm` function; run `--regress REV` before changing the generator | `./dk gen <fn…> \| --unit U [--max-insns N] [--asm]` |
| **Harvest first:** undeclared stretches ranked by how many functions have exact drafts; `--wip` ranks WIP units by exact drafts for their open functions | `./dk gen --harvest [START END] [--wip]` |
| Paste a unit's exact drafts into its source, each re-verified there (`--write` into an empty source, `--merge` into one with code); then `rename --unit U` and save | `./dk gen --unit U --write\|--merge [--into PATH]` |
| Classes/vtables from RTTI → `docs/generated/classes.md` (`--apply` renames, `--check` for CI) | `./dk analyze rtti` |
| Known library code via pinned reference decomps → `docs/generated/libraries.md` | `./dk analyze lib [--apply]` |
| Port located library objects to linked units (failures → `docs/generated/ports.md`) | `./dk port [ref:path…]` |
| Propose a unit's full split (text, data, `.ctors` slot, externs) from a function, address or range; on a declared unit it prints the one `unit set` that corrects it | `./dk unit propose <fn \| ADDR \| START END>` |
| Declare a translation unit (splits + units.toml + empty source) | `./dk unit add <src path> --lib L SECTION:0xA-0xB…` |
| Replace a unit's ranges (`--rename OLD NEW`, `--drop SECTION`) / trim data padding / undeclare it | `./dk unit set …` / `unit fit <unit>` / `unit remove <unit>` |
| Per-unit compiler / flag set / extra flags | `./dk unit flags <unit> [--mw-version V] [--cflags SET] [--extra-cflags "…"]` |
| List units with status and match % | `./dk unit list` |
| Rename symbols to source (mangled) names; OLD may be an address. `--unit U` names every fn_ADDR the unit defines. Refuses while other sources use OLD as a C name; `--keep-c` skips those | `./dk rename OLD NEW … \| --unit U [--local] [--keep-c] [--dry-run]` |
| Stale `extern "C"` names and vtable slots whose header and symbol names disagree (`--fix` the mechanical ones) | `./dk stale [--fix]` |
| Split / resize / add a data or asm label / keep a function the original kept unreferenced | `./dk symbol split ADDR` / `size NAME SIZE` / `add NAME SEC:ADDR [SIZE] [--label]` / `force-active NAME…` |
| Score a unit or function: functions, data, layout, and a fix hint for one-relocation misses. `--function F` also hand-links F. `--private`: no project lock (iterate with it) | `./dk check <unit\|fn> [--function F] [--private]` |
| Re-score every unit: linked must stay exact (exit 1 = regression), WIP ranked by %; `--near` lists WIP units ≤3 functions from exact with each blocker | `./dk check --all [--near]` |
| Target vs ours for non-exact functions and data; `--literals` lists the pool objects in the target's order | `./dk diff <unit> [--function F] [--literals]` |
| Register-numbering near miss: each variable's register ours vs target, and the declaration order that fixes it; `--try` compiles variants privately, `--apply` keeps an exact one | `./dk regs <unit> <fn> [--try] [--apply]` |
| Sort a unit's definitions into target address order; kept only if nothing scores lower | `./dk reorder <unit> [--dry-run]` |
| Which compiler/flags reproduce a unit? | `./dk compiler-search <unit> [--flags '…']` |
| Link an exact unit, verify the hash, commit | `./dk accept <unit>` |
| Commit partial progress (not linked), verified in a clean tree; several units commit together. Takes 1-5 min | `./dk save <unit…> --note "what's left"` |
| Retire an obsolete header/source | `./dk rm <path> --reason "…"` |
| Source quality: forbidden patterns (must be 0) + placeholder debt | `./dk quality [paths]` |
| Progress floor (accept/save refuse to go below it) | `./dk floor` |
| Toolkit tests / docs check / end of a phase | `./dk test` / `./dk doc-lint` / `./dk retro` |
| Commit toolkit/docs changes, or a cross-unit source change (`check --all` first) | `./dk commit-tools PATH… -m MSG` |
| Coordinator only: commit the agents' whole `src/`, `include/`, `config/` state at once; units that fell below HEAD are held back | `./dk commit-tools --integrate [--exclude PATH] -m MSG` |
| Sync the shared toolkit repo with this copy | `./dk toolkit diff\|export\|import` |
| Inspect one unit's diff by hand | objdiff GUI, project dir = repo root |

Exit codes for every command: `0` ok/match, `1` mismatch, `2` usage/environment, `3` build error.

## Rules

- **This repo is decomp only.** No mods, features, fixes or ports; those live in a separate repo that tracks this one. `--non-matching` / `Equivalent` exists only for functionally identical code.
- **The hash must match after every commit:** `./dk build` exits 0.
- **Improvements may never make things worse.** The progress floor, zero forbidden patterns and the tests gate every commit. Do refinement passes on your own earlier units (see [playbook/agents.md](playbook/agents.md)).
- **Never commit game data:** `orig/` and `build/` are git-ignored. No asm, no extracted bytes.
- **`config/<GAMEID>/{splits,symbols}.txt` change only through `dk` commands.** Agents matching functions edit only `src/` and `include/`.
- **Agent output follows [playbook/agents.md](playbook/agents.md):** exact match only, the review checklist, and its rules for running many agents at once.
- **Commit only through `./dk`:** `accept`, `save` or `rm` for a unit, `commit-tools` for everything else. Never run raw git (`add`, `commit`, `rm`, `reset`, `stash`).
- **A new repeatable step becomes a `./dk` command** with a test. Don't leave one-off scripts lying around.
- **Docs go where they belong:** generic lesson → `playbook/` (no project names); game fact or decision → `docs/<project>/`; tool output → `docs/generated/` (never hand-edited).
- **End each phase with `./dk retro`,** and keep docs within budget (`./dk doc-lint`).

## Layout

```
dk, dk.toml           toolkit entry point + its only project config
configure.py          dtk-template build config (units from units.toml)
config/<GAMEID>/      dtk config.yml, splits.txt, symbols.txt, units.toml, floor.toml, build.sha1
orig/<GAMEID>/        original game files (ignored)
src/, include/        decompiled code
tools/dk/             the toolkit (vendored; sync with ./dk toolkit)
tools/*.py            dtk-template helpers (leave as upstream)
tests/                toolkit tests (synthetic fixtures + golden tests that skip without orig/)
playbook/             portable knowledge
docs/<project>/       this game: overview, decisions, types, formats
docs/generated/       dk output
docs/upstream/        vendored upstream docs (reference only)
```
