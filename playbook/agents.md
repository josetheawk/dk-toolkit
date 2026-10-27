# Agents

AI agents draft code. The byte-exact gate and a human review decide what stays.

## Loop

`next` (pick work) → `ctx` (gather context) → edit source → `check` (exact diff?) → `accept` (full hash, commit).
Run one agent per translation unit at a time, so edits never collide.

In `dk`: `./dk next` picks the work and `./dk ctx <fn>` prints the context pack. `./dk unit add` declares a unit (splits + units file + source stub) and proves the re-split build still matches. `./dk check <unit|function>` exits 0 only when functions, data and in-unit layout are exact. `./dk accept <unit>` links it, re-verifies the whole hash, refuses any drop in matched functions, and commits.

Context pack per function: target asm, current diff, machine draft (m2c), owning class and vtable,
already-matched neighbours in the unit, and the relevant headers.

## Ratchets: improvement may never make things worse

- **Progress floor.** Matched functions, linked bytes and linked units are recorded in a committed floor file. Commits below it are refused, and successful commits raise it. Lowering the floor is a human decision with a recorded reason.
- **Forbidden patterns stay at zero** (pointer-plus-offset field access, workaround pragmas). They're checked on the unit's files at every commit.
- **Placeholder debt is measured** (unnamed functions and globals, `Unk` types, offset-named members, slot-named virtuals, pad arrays). Refinement passes pay it down. The floor and the forbidden list keep those gains.
- **Renames update the sources** that reference the old placeholder, so naming work can't silently break a match.
- **Never prove against stale outputs.** Force the split or analysis step to re-run whenever its inputs changed, independent of the build system's dependency records, and periodically build HEAD in a clean worktree (`dk verify-head`) to prove every commit is self-contained.

## Running many agents at once

Verified on a run of about 16 parallel agents against one shared working tree.

- **Roles.**
  - A coordinator assigns work and integrates; it matches nothing itself.
  - Sweepers each own one address slice. A unit belongs to the slice its `.text` starts in.
  - Closers take nearly-exact WIP units (about 88% of functions exact or more) to linked.
  - Toolsmiths turn repeated manual work into tested commands.
  - Shared header families (a UI framework, core classes) get one owner. Every layout change goes through that owner, in batches, compared on all includers.
- **Progress counts code bytes.** Point sweepers at big functions they can get exact, not at register-only near misses.
- **Every brief carries the same rules.**
  - Commit only through the tool.
  - Iterate on private copies only. Writing another unit's file or a shared header in a search loop breaks everyone else's build.
  - After any edit to a header, a linked unit or a symbol name, run the full link, not just the unit check.
  - Additive header edits are fine: re-read the header just before editing, and commit promptly.
  - Keep going until told to wind down.
- **Cross-unit changes go through the coordinator.** Renames of linked symbols and header retypes are sent as "ready to integrate: files, renames". The coordinator commits the shared working state in one integration (`commit-tools --integrate`). The integration holds back units that fell below their last commit, so agents mid-edit don't block it. Never re-run an integration just to read its error: it commits.
- **Agents run out of context.** Replace one with a fresh agent that carries the old one's notes, and keep a roster of agent ids. Messaging a finished agent restarts it, so send every agent the current roster, and have toolsmiths report "unit X needs Y" to the coordinator instead of to owners.
- **Decide devices once, for everyone.** When an agent asks whether a pragma, `volatile` or dummy layout is acceptable, the answer goes into the project's decision log. Then every brief can cite it.
- **Wind down in order.**
  1. Each agent finishes its current function, commits, and reports what is committed and what is open.
  2. The coordinator runs a final integration and stops any periodic integrator.
  3. The coordinator exports the toolkit and records the open leads for the next session.

## Refinement passes

Every few units, revisit your own earlier units and replace placeholders that new evidence explains. Evidence includes:
- Shared headers other agents created.
- RTTI and strings.
- Callers in newly matched code.

Merge duplicate struct definitions into shared headers, and delete obsolete workarounds. Every refinement is re-checked and committed like new work, so it can only keep or raise the match.

## Guard rails

- **Only exact counts.** Why: "close" code hides wrong types and wrong control flow. Apply: instructions and relocations must be identical, then the whole-binary hash must still match.
- **Agents edit only source and headers.** Why: split, symbol and build files define the target, and editing them can fake a match. Apply: those files change only in analysis work, through `dk` commands.
- **Regressions are rejected.** Why: a "fix" that breaks an earlier match is a net loss. Apply: `accept` re-checks the whole build and refuses any drop.
- **Re-score everything after touching a shared header.** Why: the hash covers only linked code, so an edit can quietly break other agents' unlinked work-in-progress or a unit's data score. Apply: a whole-project re-score (`dk check --all`) must report every linked unit exact, and no work-in-progress unit should drop.
- **One commit per accepted match.** Why: easy review and easy revert.

## Naming placeholders

- Unknown virtual methods are `vf<slot>`, because an override must share its base's name. Other unknown functions are `fn_<address>`, and unknown globals keep dtk's `lbl_<address>`.
- Unknown struct members are `m<offset>` (e.g. `m34`), and unknown types are `Unk<Context>` with explicit `pad` arrays. Those are placeholders, not hard-coded offsets; the naming pass replaces them.
- Keep `symbols.txt` in sync with mangled source names using `./dk rename OLD NEW …`, which re-proves the hash.

## Review checklist (reject if any apply)

- Hard-coded struct offsets or casts to `char*` + offset where a field belongs.
- Register-forcing tricks, dummy variables, dummy functions placed to force a layout, or reordered statements with no source-level reason.
- Assembly for code the compiler produced. Assembly is right only where the original was hand-written assembly (instructions or register use the compiler never emits; the platform notes list them), written as a whole `asm` function the way the original was.
- Invented names presented as real. Use placeholders unless there's evidence.
- Magic constants that have a symbolic meaning available (enum, define, sizeof).
- Copy-paste duplicates of an existing inline or helper.
