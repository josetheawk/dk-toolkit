# Process

Phases run in order. Each has an exit check, and each ends with `./dk retro`.

## Principles

- **The build always matches.** Why: every later step is judged against the hash, and a broken baseline hides regressions. Apply: build every split from the original first, then swap units to source one at a time. `./dk build` exits 0 after every commit.
- **No original bytes in git.** Why: the repo can then be shared. Apply: originals live in a git-ignored `orig/`, and assets and asm are regenerated from the user's own copy.
- **Generate, don't hand-maintain.** Why: hand-kept tables drift. Apply: anything derived from the binary (class lists, unit maps) is written by a `dk` command and verified with `--check`.
- **Free symbols before guessing names.** Why: a map file, debug ELF, DWARF dump or sibling build can name thousands of functions at once. Apply: search the disc, prototypes, demos and ports on other platforms of the same codebase before naming by hand.
- **One target version first.** Why: multi-version support multiplies config work before anything matches. Apply: add other regions or revisions once the first is well along.
- **Match first, document later.** Why: mixing them slows both. Apply: placeholder names by address (`fn_8xxxxxxx`) during matching, and a separate naming pass after.

## 0. Recon and scaffold

- Identify the platform, compiler family, SDK version, size (function count) and any symbol sources, with evidence for each claim.
- Scaffold from the platform's template and reproduce the target hash with nothing decompiled.
- Exit: a facts table with evidence in the project docs, plus a matching all-split build.

## 1. Split and name

- Turn the binary into translation units (TUs) and named symbols. Use static-initializer tables, RTTI/vtables, assert filename strings and per-TU constant pools as boundary signals.
- Exit: every section is covered by named units, and the build still matches.

## 2. Compiler and flags

- Find the exact compiler build and flags per library by compiling small leaf functions against every candidate.
- Exit: at least one unit per library group compiles exactly and is linked.

## 3. Libraries first

- SDK, runtime and C library code usually exists in other decomps. Port it and adjust it to the revision.
- Exit: those libraries are linked, and the shared headers exist.

## 4. Match loop

- Small leaf functions and class methods first (they firm up structs), then unit by unit so files can be linked.
- Exit per function: an exact diff. Exit per unit: linked, with the hash intact.

## 5. Document and publish

- Naming pass, then CI (build + tests + doc-lint), then a progress site.

## 6. Assets and versions

- Descriptor-driven, byte-exact round-trip tools for data formats, plus extra regions or revisions.
