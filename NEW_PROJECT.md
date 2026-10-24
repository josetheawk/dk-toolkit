# Starting a new decomp (GameCube/Wii, CodeWarrior)

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

A new platform (another compiler or console) is a package under `tools/dk/platforms/`. It
exposes `COMMANDS` and can reuse everything in `tools/dk/core/`.
