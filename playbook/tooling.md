# Tooling

## Choices

| Need | Tool | Notes |
|---|---|---|
| GC/Wii split + analysis | decomp-toolkit (`dtk`) | Built into dtk-template. Reads disc images (ISO/GCM/CISO/RVZ/WBFS/GCZ/NFS/TGC) directly. |
| N64/PSX/PS2 split | splat + spimdisasm | The MIPS-era equivalent of the dtk flow. |
| Per-function diff, progress report | objdiff (GUI + `objdiff-cli`) | Reads the `objdiff.json` the build writes. |
| Windows compilers on Linux | wibo | Downloaded by the template; fall back to wine. |
| Asm → C first draft | m2c | Supports MIPS and PowerPC. |
| Near-miss brute force | decomp-permuter | Supports mwcc. |
| Shared scratch / compiler presets | decomp.me | Good for asking others. Local `dk` commands are faster for search. |
| Browsing | Ghidra (+ platform loader) | Import dtk symbols for names. |

## Rules

- **Pin every tool version in the build config.** Why: builds must be reproducible offline. Apply: the dtk-template `configure.py` pins dtk/objdiff/wibo/binutils/compilers. Change pins only in a dedicated commit.
- **Keep template files untouched where possible.** Why: upstream fixes can then be pulled cleanly. Apply: add project tooling in its own package (`tools/dk/`), and override settings in `configure.py` only.
- **Vendored upstream docs go under `docs/upstream/<name>/` with their source commit.** Why: offline reference without mixing them into budgeted docs.
- **Parallel agents share one tree safely only through the tools.**
  - Use one project lock for builds and shared-file edits.
  - Commit each unit as **HEAD plus exactly its own delta**: compose the shared files from HEAD plus only the unit's entries (its units entry, its splits block, the symbols in its ranges or named in its files).
  - **Build that overlay in a clean worktree** before committing.
  - Write the commit through a private index, and land it with a compare-and-swap on HEAD.
  - Why: committing shared files whole once landed another agent's rename without the header it needed, and HEAD stopped building. Agents never run raw git.
- **Isolated builds pass explicit tool paths** (`configure.py --dtk … --wrapper … --compilers …`) rather than symlinking the tools directory. Why: a fresh checkout makes the download script look newer, and ninja then re-downloads *through* the symlink into the shared tools while other builds are running them.
- **Never chain `git add <paths> && git commit` in tools.** Why: when one explicit path is git-ignored, `git add` exits non-zero but **still stages the other paths**, which then leak into the next commit. Apply: tools check `git diff --cached --quiet` before staging, and refuse if the index isn't clean.

## Verified source generators

- **Generate drafts, then prove them with the real compiler.** Lift the asm symbolically (registers → expressions, stores/calls → statements, branches → if/else up to the immediate post-dominator), render several source variants, compile each in its own scratch unit with the unit's exact flags, and keep only a variant whose function matches. Why: nothing is guessed. Every draft is either proven or refused with a reason.
- **A function-level match means, per instruction:**
  - identical bits outside relocated fields
  - the same relocation kind, symbol and **addend**
  - a compiler literal holding the same bytes **in the same section** (the base register is part of the linked word); the address of a literal compares the whole object
  - one relocation per word
  The unit's literal-pool layout is checked separately by the unit score and the hash.
- **Before committing a generator change, run old and new on the same function list.** Apply: no previously exact draft may be lost; report "lost / gained".
- **Get the verifier reviewed independently.** Why: a second reader found five gaps the tests didn't cover (addends, literal section, address-of size, duplicate relocations, aliasing reorders).
- **Byte-identical isn't fidelity.** `bl f; blr` is both `f();` and `return f();`, and a pass-through argument is invisible to a virtual call. Apply: decide return vs void from the callers (do they read the result register after the call?), and print a caveat where the asm can't decide.
- **Rebuild compiler-shaped functions from metadata instead of lifting them.** Virtual destructors and constructors are mostly compiler output. Build the class chain from RTTI and vtable stores, compile `C::~C() {}`, and verify. Name an unnamed callee only on structural evidence (its own asm stores that base's vtable), and emit the rename for the agent.
- **Enumerate source variants instead of hand-tweaking.** Why: agents matched faster with scripts that compiled every combination (declaration order, if/else vs ternary, helper vs inline) than by editing one guess at a time. Apply: a generator renders each variant, compiles all of them, and keeps the exact one.

## Shared working trees

- **Judge ratchets on the clean build a commit is made from, never on the shared tree.** Why: the shared tree also holds other agents' half-edited units, so its numbers come and go; a floor raised from them crept above what HEAD builds and then refused honest commits. Apply: build HEAD + the change in a clean worktree, compare that report with the committed floor, and commit the floor raised to it.
- **Shared config files change entry by entry.** A unit's commit composes HEAD's copy plus only that unit's entries; committing the whole file would land other agents' unfinished entries.

## GC/Wii template quick start

1. Copy dtk-template, rename `GAMEID` dirs, put the disc image or extracted `sys/main.dol` in `orig/<ID>/`.
2. In `config.yml`: `object`, `hash`, `symbols`, `splits`, `detect_objects`, `detect_strings`, `write_asm`. Drop `modules` if there are no RELs.
3. Write `build.sha1` pointing at `build/<ID>/main.dol`, set `VERSIONS` in `configure.py`, and set `config.libs = []` until splitting starts.
4. `python configure.py && ninja`: the first run analyses, splits and links. Expect `main.dol: OK` with no source at all.
