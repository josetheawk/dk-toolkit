# Matching cookbook

One entry per lesson: **asm symptom → source idiom that fixes it**. Why. Which compilers it applies to.
Group entries under a heading per compiler family. Add an entry only after it produced an exact match.

## CodeWarrior (mwcc)

- **Every function is 100% but the unit's function order differs → `-inline deferred` emits functions in reverse source order.** Check a matching reference decomp of the same library before inventing a source order: Metrowerks' runtime and MSL were built with `-inline deferred,auto`, and their canonical sources reproduce the binary exactly that way. Keep one flag set per library. Detect: `./dk check` reports `layout: MISMATCH` with swapped offsets, which objdiff's per-function score alone misses.
- **Judge a compiler with whole units, not single trivial functions.** Why: simple code compiles identically on GC 1.3 through 2.7, so it can't discriminate. Apply: run `./dk compiler-search` on units with loops, floats, switches and inlining, and treat a version as pinned only when it's the sole exact match.
- **objdiff's default scoring ignores what a relocation points at.** `m = 0.0f` and `m = 0.1f` both score 100% when the literal lives in a pool. Why: the default `functionRelocDiffs=none` is lenient. Apply: score with `-c functionRelocDiffs=data_value` (`dk check` does) and treat any data-section mismatch as a real bug, not noise.
- **C++ (mwcc): each class's out-of-line destructor follows its constructor.** The base's inline destructor gets its out-of-line copy right before the first derived destructor, and inline virtuals referenced only by vtables go to the end of the unit. Symptom: all functions 100% but `layout: MISMATCH`, with destructors bunched at the end. Fix: define the destructors out of line in source order (`X::~X() {}`), rather than inline in the class.
- **Pure virtual slots are `0` in mwcc vtables,** not a stub. A `0` slot means `= 0` in the base declaration.
- **Data padding belongs outside the unit.** Trailing zero words before the next unit's aligned data aren't part of the object, and objdiff counts them as a mismatch. End data splits at the real end, word-aligned.
- **Weak data copies are expected, not mismatches.** A base class with inline virtuals makes every TU emit a weak copy of its vtable, and the linker keeps the first. A correct unit therefore carries data its target lacks. Apply: discount base-only bytes inside weak symbols the target doesn't define (`dk check` does), and let the linked hash prove it.
- **A TU-scope `const int` written in `__sinit` (`li` + `stw`) came from a non-constant initializer.** Reproduce it as `static const int x = f(0.25f);` with `inline int f(float s) { return s * 60.0f; }`, or from another call, rather than as a literal. A literal folds away and leaves no `__sinit` code.
- **Virtual-base thunks** are named `@<this-adj>@<vbase-off>@<method>` and are emitted after the TU's `__sinit`, together with its factory functions. When a class has a virtual base, name its own unknown virtuals `vf<slot>_<Class>` so they don't clash with the base's `vf<slot>`.
- **Find `-fp_contract` from the binary, not by trial.** Count fused multiply-add instructions (`fmadds`, `fmsubs`, `fnmadds`, `fnmsubs`) per unit. If game code uses separate `fmuls` + `fadds` everywhere and FMAs only appear in a few library units, game code was built with `-fp_contract off`.
- **Zero blocks the size of a vtable in the middle of `.data` are discarded weak copies.** The original linker zero-filled a duplicate weak object in mid-section and removed trailing ones. To reproduce it, make the unit *use* that class the way the original did, so the compiler emits the RTTI name and the weak vtable at that position. Don't pad with a zero array. `dk check` counts a target zero block of a weak copy's size as explained, and the link proves it.
- **dtk folds literals a function reaches as base+offset into one auto label.** Split it at the real object boundaries (`dk symbol split ADDR`) so each source-level object pairs with its own target symbol.

### Codegen idioms (mwcc, C++), each verified by an exact match

| asm symptom | source idiom |
|---|---|
| Stores in reverse order (z, y, x) | Chained assignment `a = b = c = K`, which stores the last one first |
| `fcmpu f, 0` vs `fcmpu 0, f` | `if (f)` vs `f != 0.0f` |
| `x * 0.5f` with `x` loaded before the constant | `x / 2.0f` (a literal `x * 0.5f` loads the constant first) |
| An add of 0.0 that survives | A variable: `float d = 0.0f; x += d;` (a literal 0.0f is folded away) |
| Stack slots | Aggregate locals sit above scalars; within a group, the earlier declaration gets the higher offset |
| A loop differing only in register allocation (`li rX,0; mr rY,rX`) | The loop was an inlined helper; write it as an inline member function |
| An inline virtual emitted right after one function | That function calls it through the vtable; the weak copy lands there |
| Functions after `__sinit` | Inline virtuals of classes constructed only in the static initializer, and thunks |
| `name$N` plus an `init$N` guard in `.sbss` | A function-local static; size the guard so `.sbss` pairs |
| `strcmp(...) == 0` gives `srwi` vs `extrwi 8,19` | Returned as `bool` vs `int`/`unsigned char` |
| `cmplwi` against 0 on a count | The field is unsigned |
| `lfs`/`stfs` per field vs `lwz`/`stw` words | Assigning an all-float struct vs initializing a by-value local |
| A pooled float one ulp off | Write the exact literal (e.g. `0.025000002f`) |
| `fmuls` operand order: constant first vs variable first | `q * 2.0f` puts the literal first; `x *= K` puts the variable first |
| A global re-read after every compare or store | The global is `volatile` |
| `if (!a \|\| !b) return;` shape: `bne next; b exit` | Exactly that `\|\|` early return (not two separate `if`s) |
| `bge; b; fmr` vs `cror` for a min/max | A ternary `t < 0 ? t : 0` gives the branches; a plain `if` gives `cror` |
| `mulli; addi; lfsx` on an array element | The element was read through a reference: `T& e = a[i];` |
| Saved registers differ across consecutive loops | Declare one shared `int i` for all the loops |
| Callee-saved register numbers swapped between two locals | Locals get registers from r31 downward in declaration order; an inline accessor's result gets a lower one |
| Literals in a different pool order | Pool order is creation order: reorder the expressions that first use them |
| Stack locals of an inlined function in reverse order | Inlined functions' locals are laid out in reverse; reorder their declarations there |
| An implicit `operator=` emitted at the end of the unit | Give the struct an inline user `operator=` (or copy fields), so no implicit one is generated |
| A 4-byte empty `__sinit` | A global of a class with an empty inline constructor |
| A vtable in an unexpected unit | The unit that defines the class's first non-inline virtual (its key function) emits it |

