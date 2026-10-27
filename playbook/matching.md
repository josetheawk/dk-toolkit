# Matching cookbook

One entry per lesson: **asm symptom → source idiom that fixes it**. Why. Which compilers it applies to.
Group entries under a heading per compiler family. Add an entry only after it produced an exact match.

## CodeWarrior (mwcc)

- **Every function is 100% but the unit's function order differs → `-inline deferred` emits functions in reverse source order.** Check a matching reference decomp of the same library before inventing a source order: Metrowerks' runtime and MSL were built with `-inline deferred,auto`, and their canonical sources reproduce the binary exactly that way. Keep one flag set per library. Detect: `./dk check` reports `layout: MISMATCH` with swapped offsets, which objdiff's per-function score alone misses.
- **Judge a compiler with whole units, not single trivial functions.** Why: simple code compiles identically on GC 1.3 through 2.7, so it can't discriminate. Apply: run `./dk compiler-search` on units with loops, floats, switches and inlining, and treat a version as pinned only when it's the sole exact match.
- **objdiff's default scoring ignores what a relocation points at.** `m = 0.0f` and `m = 0.1f` both score 100% when the literal lives in a pool. Why: the default `functionRelocDiffs=none` is lenient. Apply: score with `-c functionRelocDiffs=data_value` (`dk check` does) and treat any data-section mismatch as a real bug, not noise.
- **C++ (mwcc): each class's out-of-line destructor follows its constructor.** The base's inline destructor gets its out-of-line copy right before the first derived destructor, and inline virtuals referenced only by vtables go to the end of the unit. Symptom: all functions 100% but `layout: MISMATCH`, with destructors bunched at the end. Fix: define the destructors out of line in source order (`X::~X() {}`), rather than inline in the class.
- **Pure virtual slots are `0` in mwcc vtables,** not a stub. A `0` slot means `= 0` in the base declaration.
- **objdiff scores a vtable by its bytes, which are zero where relocations go,** so an override declared in the wrong slot scores 100% and only the link catches it. Compare which function each slot points at (`dk check` does).
- **objdiff counts our relocation against a section symbol (`...data.0+0xA0`) or an `@N` literal as different from dtk's label at the same address.** Treat them as equal when both are the same offset of the same section of the unit with the same bytes there (`dk check` does; each such function hand-linked byte-identical to the original).
- **Trailing zero padding in a data split is not a mismatch** (`dk check` allows under 32 bytes), but where a split may end is a dtk rule: see the platform notes on `gap_` symbols.
- **Weak data copies are expected, not mismatches.** A base class with inline virtuals makes every TU emit a weak copy of its vtable, and the linker keeps the first. A correct unit therefore carries data its target lacks. Apply: discount base-only bytes inside weak symbols the target doesn't define (`dk check` does), and let the linked hash prove it.
- **A TU-scope `const int` written in `__sinit` (`li` + `stw`) came from a non-constant initializer.** Reproduce it as `static const int x = f(0.25f);` with `inline int f(float s) { return s * 60.0f; }`, or from another call, rather than as a literal. A literal folds away and leaves no `__sinit` code.
- **Two pointer loops over adjacent array members that read a fresh `.sdata` word each iteration (`lwz r0, lbl_sdata; stw r0, 0(p)`), the second storing a register zero, came from by-reference fill helpers.** A `fill(first, last, const T& value)` template called with a literal binds the literal to a hidden writable static, so the value is reloaded every pass; a second call whose value is a typed null pointer keeps it in a register. Two inlined copies get two such statics. Apply: `Fill(m3C, m3C + 6, 0); Fill(m54, m54 + 6, (T*)0);` reproduced the ctor, the `.sdata` pair and the shared end pointer exactly.
- **`addi r3, r31, 0x64` string arguments off a shared data base score below 100% even though the bytes are identical, when the target's pooled string block is one dtk object (floats, then strings).** Why: objdiff compares the symbol each such immediate points into (data content from that address on), and separate `@N` literals differ from the target's one object. Apply: write the strings as one `static char names[N] = "a\0\0\0b\0..."` with the exact padding and put it in source order after the floats, then pass `names + offset`. If another unit externs the object's first member (`lbl_...[4]`), define that member as a global with the link name.
- **Virtual-base thunks** are named `@<this-adj>@<vbase-off>@<method>` and are emitted after the TU's `__sinit`, together with its factory functions. When a class has a virtual base, name its own unknown virtuals `vf<slot>_<Class>` so they don't clash with the base's `vf<slot>`.
- **Find `-fp_contract` from the binary, not by trial.** Count fused multiply-add instructions (`fmadds`, `fmsubs`, `fnmadds`, `fnmsubs`) per unit. If game code uses separate `fmuls` + `fadds` everywhere and FMAs only appear in a few library units, game code was built with `-fp_contract off`.
- **Zero blocks the size of a vtable in the middle of `.data` are discarded weak copies.** The original linker zero-filled a duplicate weak object in mid-section and removed trailing ones. To reproduce it, make the unit *use* that class the way the original did, so the compiler emits the RTTI name and the weak vtable at that position. Don't pad with a zero array. `dk check` counts a target zero block of a weak copy's size as explained, and the link proves it.
- **The target keeps a TU's unreferenced data (RTTI of a weak vtable, class-name strings of discarded inline copies) → `#pragma force_active on` after the includes, left on to the end of the file.** Why: the pragma sets the force-active flag (in `.comment`) on every object of the TU, compiler-generated ones included; the linker still drops weak duplicates but keeps everything else, unreferenced functions too. `force_active reset` before the end keeps the strings but not the RTTI emitted at end of TU, which matches TUs that keep only the strings. Detect: code and data exact, the link is short in `.sdata`/`.data`, and the target holds objects your compile has that nothing references (often just past the unit's split end).
- **A TU whose only content is a header static's `__sinit` links short by one `.ctors` word → `#pragma force_active on`.** Why: nothing references the TU, so the linker drops its static initializer and the `.ctors` entry with it, though the original kept them. Detect: code and data exact, accept fails with every later section shifted by 4 bytes after `.ctors`.
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
| Callee-saved register numbers swapped between two locals | Locals rank by declaration order; temporaries and inline results rank above them; see gamecube-mwcc.md Register allocation |
| A local set on two paths takes r31 ahead of locals defined before it | It came from `c ? a : b` or an inline with two returns; `if (c) x = a; else x = b;` in the caller keeps definition order |
| `this` and another saved local swapped | Reach the data through an inline member (`NumActors()`, `Settings()->Profile()`) instead of a free helper or a plain load |
| A field the function already loaded is loaded again inside later branches | Those uses go through the class's inline accessor (`r.fn_Right()` for `m8 + mC`), not the field |
| A first argument loaded before a later argument's nested call | It was an inline call itself (`f(Settings(), 13, g(t))`); a plain load is evaluated after the nested call |
| Array-of-objects static init adds the element size to a base address (`addi r3, r30, 0x24; addi r3, r3, 0x24`) | GC 1.3.2 or later (1.3 folds the offset); if 1.3.2 then inlines a constructor the target calls, use `-inline noauto` with `static inline` helpers |
| Literals in a different pool order | Pool order is creation order: reorder the expressions that first use them |
| Stack locals of an inlined function in reverse order | Inlined functions' locals are laid out in reverse; reorder their declarations there |
| An implicit `operator=` emitted at the end of the unit | Give the struct an inline user `operator=` (or copy fields), so no implicit one is generated |
| A 4-byte empty `__sinit` | A global of a class with an empty inline constructor |
| A global loaded into r3 and copied (`lwz r3, g; cmplwi r3, 0; mr r31, r3`) where the target loads it straight into r31 | Assign in the condition, `if ((p = g) != 0)`, or read it through the class's inline accessor; `T* p = g; if (p != 0)` makes the copy |
| `cmpwi r3, 0; li r3, K; beq end; li r3, 0` where ours is branchless (`cntlzw`/`neg`/`andc`) | Name the call's result and set the return value before the test: `int ok = f(); int r = K; if (ok) r = 0; return r;` (`return f() ? 0 : K` and early returns both fold) |
| A clamp that loads the upper bound first and keeps the result in the lower bound's register | Write the clamp with its own compares (`if (lo > hi) lo = hi; if (v < lo) return lo; if (v > hi) return hi; return v;`), not nested `min()`/`clamp()` inlines |
| An inline function's weak copy at its definition point instead of after its first user | It was declared in the class and defined later in the unit: the copy is emitted where the definition appears |
| An inline's result tested in r3 and only then copied to a saved register (`cmplwi r3, 0; mr r30, r3`) where ours keeps it in the saved register | A cast on the inline call, even a redundant one: `T* p = (T*)cast<T>(w);` |
| A call the target keeps (`bl f`) where ours auto-inlines `f`, defined earlier in the unit | The original `f` was bigger before inlining: write its own inline helper calls out by hand (one expanded `Clear()` was enough) |
| A vtable in an unexpected unit | The unit that defines the class's first non-inline virtual (its key function) emits it |
| `lbzx base, idx` with `idx = (skip + K) + i` vs a folded `lbz K(base + i + skip)` | Index the result of an inline accessor: `inline T* Keys(H* h) { return (T*)h + Skip(h) + K; }`, then `Keys(h)[i]` |
| `mullw` operands swapped (`conv, n` vs `n, conv`) | Put the converted operand in a named local (`int q = (int)(t / n); q * n`); swapping the source operands does nothing |

