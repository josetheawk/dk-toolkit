# Tooling

## Choices

| Need | Tool | Notes |
|---|---|---|
| GC/Wii split + analysis | decomp-toolkit (`dtk`) | Built into dtk-template. Reads disc images (ISO/GCM/CISO/RVZ/WBFS/GCZ/NFS/TGC) directly. |
| N64/PSX/PS2 split | splat + spimdisasm | The MIPS-era equivalent of the dtk flow. |
| Per-function diff, progress report | objdiff (GUI + `objdiff-cli`) | Reads the `objdiff.json` the build writes. |
| Windows compilers on Linux | wibo | Downloaded by the template; fall back to wine. |
| Asm → C first draft | m2c | Supports MIPS and PowerPC. |
| Near-miss brute force | decomp-permuter | Supports mwcc. Try it last: random rewrites closed no mwcc near miss in three sessions of parallel agents, while register-numbering diffs follow the allocation rule in the platform notes. |
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
- **A public toolkit gets a machine scan before every export, never a reviewer's memory.** Why: a toolkit grown inside one project carried its game names, real symbol addresses, its asm as test fixtures and its constants (eval salts, log folders) into the public repo, and only a history rewrite took them out. Apply: every game-specific value is a config key with a neutral default; tests that need the original live in a project-owned folder outside the toolkit's; the export scans the files it would copy for the project's names (config list), any address in the symbols file and any 3 instruction-like words that follow each other in the original binary, and refuses on a hit (`dk toolkit export --check`).
- **Never chain `git add <paths> && git commit` in tools.** Why: when one explicit path is git-ignored, `git add` exits non-zero but **still stages the other paths**, which then leak into the next commit. Apply: tools check `git diff --cached --quiet` before staging, and refuse if the index isn't clean.
- **A LoRA for the matching model trains on exactly the inference prompt, and its path is proven on a tiny model of the same architecture first.**
  - Render the prompt with the model's own chat template plus its generation prompt and the same reply prefill. Put the loss on the answer only.
  - Leave out records over the length cap rather than truncating them. Take the eval split by whole units.
  - For a mixture-of-experts base:
    - Train a bf16 LoRA; 4-bit QLoRA can't quantize fused experts.
    - Merge in bf16, then convert and quantize. Never merge into a quantized model.
    - Run train → merge → GGUF → serve on a random-weight copy of the architecture before renting a GPU.
  - Why: a cloud run is paid by the hour. A template mismatch or an unconvertible checkpoint, found after training, wastes the whole run.

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
- **Find a big function's inlined helpers by matching values, not text.** Why: an inlined copy is rescheduled into its neighbours, renames registers, sits at a stack address (every displacement shifts), shares constants, and loses the null checks of a known `this` and the branches on known arguments. Register-renamed contiguous matching found almost none of 44 known copies; value numbering found all 44. Apply: number values as (root, offset) through copies and address arithmetic, strip the frame, seed on the template's rarest calls and symbols, bind parameters with one offset per base, and score matched over unfolded weight. Also demand most of the template's calls and half its weight, or look-alike functions that share the helper's idioms win the overlaps (`dk inlines`).
- **An inlined body is a function in disguise: cut it out and plan it as one.** Why: constructors inlined into a new-expression and member destructors inlined into their container's destructor stayed undraftable while the generator only lifted whole functions; cutting the region out (from the null check to the join), rebasing the object's register to `this` and planning it with the ordinary constructor or destructor model drafted 48 more functions in one pass. Apply: verify the container with the piece defined `inline`, and require every copy of one class's inlined body to render alike.
- **Enumerate source variants instead of hand-tweaking.** Why: agents matched faster with scripts that compiled every combination (declaration order, if/else vs ternary, helper vs inline) than by editing one guess at a time. Apply: a generator renders each variant, compiles all of them, and keeps the exact one.

## Shared working trees

- **Judge ratchets on the clean build a commit is made from, never on the shared tree.** Why: the shared tree also holds other agents' half-edited units, so its numbers come and go; a floor raised from them crept above what HEAD builds and then refused honest commits. Apply: build HEAD + the change in a clean worktree, compare that report with the committed floor, and commit the floor raised to it.
- **Never wait while holding a shared lock.** Why: a retry loop (waiting for another agent's broken source) inside a locked command made every queued command wait in turn; one broken file stalled 16 agents for 20 minutes. Apply: fail fast, name the culprit, and let the caller retry.
- **Renaming a placeholder to a member's mangled name breaks every unit that still uses it as a C name** (`extern "C"` declarations or blocks, asm `bl`, C files): their link fails with `undefined`, and a hash proof that links only matched units does not notice. Apply: find those uses before renaming and refuse (`dk rename` does).
- **A unit's commit carries the headers its source includes, and with them other agents' half-done edits.** Why: one agent's in-progress shared-header edit broke or floor-dropped everyone's saves. Apply: when the clean verify fails, retry with each header that differs from HEAD left at HEAD, and keep an edit only if the units compile and score worse without it.
- **Keep in-repo paths unresolved, and never let a throwaway tree touch a path outside itself.** Why: a config loader that resolved symlinks turned a worktree's linked original binary into an absolute path, so the clean-verify step "replaced" the original by unlinking the main checkout's real file. Apply: normalise paths without following links, check that every path a build tree deletes lies inside it, and copy (not link) the original into each worktree.
- **A commit composed entry by entry keeps the committed file's order.** Why: re-sorting a shared config file on every commit rewrote the whole file each time and left the committer's copy differing from what it committed, so the next whole-state commit put the old order back. Apply: replace changed entries in place; put new ones next to their neighbour in the working copy.
- **An integration commit (the whole working state) carries only the decomp's sources and config.** Why: integrations swept toolsmiths' half-finished tool files into HEAD, and one in-progress test blocked every integration. Toolkit and doc changes are committed by their authors when done.

## GC/Wii template quick start

1. Copy dtk-template, rename `GAMEID` dirs, put the disc image or extracted `sys/main.dol` in `orig/<ID>/`.
2. In `config.yml`: `object`, `hash`, `symbols`, `splits`, `detect_objects`, `detect_strings`, `write_asm`. Drop `modules` if there are no RELs.
3. Write `build.sha1` pointing at `build/<ID>/main.dol`, set `VERSIONS` in `configure.py`, and set `config.libs = []` until splitting starts.
4. `python configure.py && ninja`: the first run analyses, splits and links. Expect `main.dol: OK` with no source at all.
