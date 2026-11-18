# Starting a new project

Pick the platform for the target (dk.toml `project.platform`):

- **`raw`**: any binary you don't have a platform for yet (firmware, ROM, flash or disk dump,
  unknown file). Always a valid start.
- **`spi_nor`**: a SPI NOR flash dump. `raw` plus erased-flash rules.
- **`gc_mwcc`**: a GameCube/Wii game built with CodeWarrior, for a byte-matching decomp.

## Any binary (`raw`, `spi_nor`)

1. **Repo.** A fresh git repo with the image inside it (for example `orig/image.bin`).
   Git-ignore the image and `build/`: never commit the target's bytes.
2. **Toolkit.** Copy this repo's `dk`, `tools/dk/`, `tests/` and `playbook/` into the project.
   After that, `./dk toolkit import` keeps them current.
3. **Config.** Copy `templates/dk-raw.toml` (or `templates/dk-spi.toml`) to `dk.toml` and fill in
   the image path, its SHA-1, the toolkit repo path and the banned playbook terms. Copy
   `templates/AGENTS-raw.md` to `AGENTS.md` and fill in the first paragraph. Commit.
4. **Triage.** `./dk status`, then `./dk map` (regions, entry points, containers, names),
   `./dk carve` (writes them out with a manifest; `--check` verifies) and `./dk dis`.
5. **Record.** Facts go in the project's `docs/`, each verified against the bytes. Lessons that
   would help on any target go to `playbook/`, then `./dk toolkit export`.

When the target needs knowledge `raw` lacks (a medium, a container, a CPU, a compiler), add a
platform with `BASE = "raw"` (see below) instead of editing `raw` for one target.

## GameCube/Wii (`gc_mwcc`)

1. **Project skeleton.** Copy [dtk-template](https://github.com/encounter/dtk-template) at a
   pinned commit into a fresh repo. Rename the `GAMEID` directories, and put the disc image
   or `sys/main.dol` in `orig/<ID>/` (git-ignored).
2. **Toolkit.** Copy this repo's `dk`, `tools/dk/`, `tests/` and `playbook/` into the
   project. After that, `./dk toolkit import` keeps them current.
3. **Config.**
   - Copy `templates/dk.toml` to the project root and fill in the game id, SHA-1, paths and banned playbook terms.
   - Add a `[toolkit] repo = "<path to this repo>"` entry.
   - Copy `templates/AGENTS.md` and fill in the first paragraph.
4. **First build.** `./dk setup && ./dk build` should report the original hash with no source at all. Commit.
5. **Analysis.** Run `./dk analyze rtti --apply`, then `./dk analyze lib --apply` and `./dk port` (reference decomps are pinned in dk.toml `[[platform.references]]`). Each step keeps the hash matching.
6. **Compiler.** Run `./dk compiler-search` on a few small units per library group. Record the result in `configure.py`'s CFLAGS and in the project's decision log.
7. **Matching loop.**
   1. `./dk next` and `./dk unit propose`
   2. `./dk unit add`
   3. `./dk gen --unit`, which gives verified drafts
   4. `./dk ctx` for the rest
   5. `./dk check` and `./dk diff`
   6. `./dk accept`, or `./dk save`

   For parallel agents, give each an address slice, plus one agent that closes WIP units.
   `playbook/agents.md` has the rules.
8. **Every phase:** run `./dk retro`. Generic lessons go to `playbook/`, then `./dk toolkit export`.

## A new platform

A package under `tools/dk/platforms/<name>/` that exposes `COMMANDS` and reuses
`tools/dk/core/`. Set `BASE` to build on another platform: its commands and hooks apply
unless this one defines the same name (`spi_nor` overrides only `profile`). A platform with
no ninja build lists the core commands that do not apply in `SKIP_CORE`. The hooks are listed
in `tools/dk/platforms/__init__.py`. Every command gets a test; lessons go to
`playbook/platforms/<name>.md`.
