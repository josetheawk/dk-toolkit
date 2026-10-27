# Matching cookbook

One entry per lesson: **asm symptom → source idiom that fixes it**. Why. Which compilers it applies to.
Group entries under a heading per compiler family. Add an entry only after it produced an exact match.
How to identify the compiler and flags is in the platform notes.

## CodeWarrior (mwcc)

### What a per-function score misses

- **Every function is 100% but the unit's function order differs → `-inline deferred` emits functions in reverse source order.** Check a matching reference decomp of the same library before inventing a source order: Metrowerks' runtime and MSL were built with `-inline deferred,auto`, and their canonical sources reproduce the binary exactly that way. Keep one flag set per library. Detect: `./dk check` reports `layout: MISMATCH` with swapped offsets, which objdiff's per-function score alone misses.
- **objdiff's default scoring ignores what a relocation points at.** `m = 0.0f` and `m = 0.1f` both score 100% when the literal lives in a pool. Why: the default `functionRelocDiffs=none` is lenient. Apply: score with `-c functionRelocDiffs=data_value` (`dk check` does) and treat any data-section mismatch as a real bug, not noise.
- **objdiff scores a vtable by its bytes, which are zero where relocations go,** so an override declared in the wrong slot scores 100% and only the link catches it. Compare which function each slot points at (`dk check` does). Pure virtual slots are `0` (not a stub): a `0` slot means `= 0` in the base declaration.
- **objdiff counts our relocation against a section symbol (`...data.0+0xA0`) or an `@N` literal as different from dtk's label at the same address.** Treat them as equal when both are the same offset of the same section of the unit with the same bytes there (`dk check` does; each such function hand-linked byte-identical to the original).
- **Trailing zero padding in a data split is not a mismatch** (`dk check` allows under 32 bytes), but where a split may end is a dtk rule: see the platform notes on `gap_` symbols.
- **Weak copies explain data on both sides.** A base class with inline virtuals makes every TU emit a weak copy of its vtable, and the linker keeps the first. So a correct unit carries data its target lacks, and where the original linker discarded a duplicate in mid-section it left a zero block of the object's size (trailing ones it removed). Apply: discount base-only bytes inside weak symbols the target doesn't define, and count a target zero block of a weak copy's size as explained (`dk check` does both; the link proves it). To reproduce such a block, make the unit *use* that class the way the original did, so the compiler emits the weak vtable at that position. Never pad with a zero array.
- **dtk folds literals a function reaches as base+offset into one auto label.** Split it at the real object boundaries (`dk symbol split ADDR`) so each source-level object pairs with its own target symbol.

### Data and static initializers

- **A TU-scope `const int` written in `__sinit` (`li` + `stw`) came from a non-constant initializer.** Reproduce it as `static const int x = f(0.25f);` with `inline int f(float s) { return s * 60.0f; }`, or from another call, rather than as a literal. A literal folds away and leaves no `__sinit` code.
- **Two pointer loops over adjacent array members that read a fresh `.sdata` word each iteration (`lwz r0, lbl_sdata; stw r0, 0(p)`), the second storing a register zero, came from by-reference fill helpers.** A `fill(first, last, const T& value)` template called with a literal binds the literal to a hidden writable static, so the value is reloaded every pass; a second call whose value is a typed null pointer keeps it in a register. Two inlined copies get two such statics. Apply: `Fill(m3C, m3C + 6, 0); Fill(m54, m54 + 6, (T*)0);` reproduced the ctor, the `.sdata` pair and the shared end pointer exactly.
- **`addi r3, r31, 0x64` string arguments off a shared data base score below 100% even though the bytes are identical, when the target's pooled string block is one dtk object (floats, then strings).** Why: objdiff compares the symbol each such immediate points into (data content from that address on), and separate `@N` literals differ from the target's one object. Apply: write the strings as one `static char names[N] = "a\0\0\0b\0..."` with the exact padding and put it in source order after the floats, then pass `names + offset`. If another unit externs the object's first member (`lbl_...[4]`), define that member as a global with the link name.
- **The target keeps a TU's unreferenced data (RTTI of a weak vtable, class-name strings of discarded inline copies) → `#pragma force_active on` after the includes, left on to the end of the file.** Why: the pragma sets the force-active flag (in `.comment`) on every object of the TU, compiler-generated ones included; the linker still drops weak duplicates but keeps everything else, unreferenced functions too. `force_active reset` before the end keeps the strings but not the RTTI emitted at end of TU, which matches TUs that keep only the strings. Detect: code and data exact, the link is short in `.sdata`/`.data`, and the target holds objects your compile has that nothing references (often just past the unit's split end). A unit that is only a `__sinit` is a different case: see the platform notes. When the link comes out longer instead, a weak copy of ours that the target lacks survives because the first copy, in an earlier not-yet-matched object, still has an auto name (`fn_ADDR`): to the linker ours is the only definition and force_active keeps it. Give that copy its mangled name (`dk rename`) and ours becomes a dropped duplicate.

### Classes

- **Each class's out-of-line destructor follows its constructor.** The base's inline destructor gets its out-of-line copy right before the first derived destructor, and inline virtuals referenced only by vtables go to the end of the unit. Symptom: all functions 100% but `layout: MISMATCH`, with destructors bunched at the end. Fix: define the destructors out of line in source order (`X::~X() {}`), rather than inline in the class.
- **Virtual-base thunks** are named `@<this-adj>@<vbase-off>@<method>`. When a class has a virtual base, name its own unknown virtuals `vf<slot>_<Class>` so they don't clash with the base's `vf<slot>`.

### Codegen idioms (mwcc, C++), each verified by an exact match

| asm symptom | source idiom |
|---|---|
| Stores in reverse order (z, y, x) | Chained assignment `a = b = c = K`, which stores the last one first |
| `fcmpu f, 0` vs `fcmpu 0, f` | `if (f)` vs `f != 0.0f` |
| `x * 0.5f` with `x` loaded before the constant | `x / 2.0f` (a literal `x * 0.5f` loads the constant first) |
| An add of 0.0 that survives | A variable: `float d = 0.0f; x += d;` (a literal 0.0f is folded away) |
| Stack slots | Aggregate locals sit above scalars; within a group, the earlier declaration gets the higher offset. An inlined function's locals are laid out in reverse: reorder their declarations there |
| A loop differing only in register allocation (`li rX,0; mr rY,rX`) | The loop was an inlined helper; write it as an inline member function |
| An inline's weak copy (or a function) in an unexpected place | It lands right after its first user (a call through the vtable counts); at its definition when it is declared in the class and defined later in the unit; after `__sinit` when only the static initializer constructs its class (the TU's thunks and factory functions go there too) |
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
| Callee-saved register numbers swapped between locals (or with `this`) | The rank rule (gamecube-mwcc.md, Register allocation): named locals rank by declaration order, temporaries and inline results above them. A value made by `c ? a : b`, an inline with two returns or an inline member (`NumActors()`, `Settings()->Profile()`) takes r31 ahead of locals declared before it; `if (c) x = a; else x = b;`, a free helper or a plain load keeps declaration order. `dk regs` shows each variable's register |
| A field the function already loaded is loaded again inside later branches | Those uses go through the class's inline accessor (`r.fn_Right()` for `m8 + mC`), not the field |
| A first argument loaded before a later argument's nested call | It was an inline call itself (`f(Settings(), 13, g(t))`); a plain load is evaluated after the nested call |
| Array-of-objects static init adds the element size to a base address (`addi r3, r30, 0x24; addi r3, r3, 0x24`) | GC 1.3.2 or later (1.3 folds the offset); if 1.3.2 then inlines a constructor the target calls, use `-inline noauto` with `static inline` helpers |
| Literals in a different pool order | Pool order is creation order: reorder the expressions that first use them (`dk diff --literals` lists the target's order and first users) |
| A function's literals start the pool ahead of an earlier function's, or one block's literals lead its function's | That block was an ordinary helper defined before the user and auto-inlined (its own copy dead-stripped): its literals are created at its definition. Float literals are created in source order, int-to-float conversion constants after the function's other literals |
| An implicit `operator=` emitted at the end of the unit | Give the struct an inline user `operator=` (or copy fields), so no implicit one is generated |
| A 4-byte empty `__sinit` | A global of a class with an empty inline constructor |
| A global loaded into r3 and copied (`lwz r3, g; cmplwi r3, 0; mr r31, r3`) where the target loads it straight into r31 | Assign in the condition, `if ((p = g) != 0)`, or read it through the class's inline accessor; `T* p = g; if (p != 0)` makes the copy |
| An inline's result tested in r3 and only then copied to a saved register (`cmplwi r3, 0; mr r30, r3`) where ours keeps it in the saved register | A cast on the inline call, even a redundant one: `T* p = (T*)cast<T>(w);` |
| `cmpwi r3, 0; li r3, K; beq end; li r3, 0` where ours is branchless (`cntlzw`/`neg`/`andc`) | Name the call's result and set the return value before the test: `int ok = f(); int r = K; if (ok) r = 0; return r;` (`return f() ? 0 : K` and early returns both fold) |
| A clamp that loads the upper bound first and keeps the result in the lower bound's register | Write the clamp with its own compares (`if (lo > hi) lo = hi; if (v < lo) return lo; if (v > hi) return hi; return v;`), not nested `min()`/`clamp()` inlines |
| A call the target keeps (`bl f`) where ours auto-inlines `f`, defined earlier in the unit | Try GC 1.3 for the unit first: it auto-inlines less than 1.3.2 (three units matched that way). Check for a TU boundary between them too: a class change, weak inline copies or factories after a `__sinit` mean the unit is two TUs, and splitting it makes the call a `bl` with no source change. Otherwise the original `f` was bigger before inlining: write its own inline helper calls out by hand (one expanded `Clear()` was enough) |
| A call's struct-valued result passed straight to a callee, in a frame smaller than ours by the struct's size | The parameter is `const T&` bound to the temporary: `g(f(x))`, not a named local plus a pointer. `T r(f(x));` matches a target that copies the result; `const T& r = f(x);` matches one that reads fields from the return slot |
| `cmpwi; beq case; b default` where ours gives `bne` | A `switch` with one `case`, not an `if` |
| A float compared, then returned as is: `fcmpu; bne next; b end` with `f1` already loaded | `if (x == v) return v;` (an `if` with an empty branch gives `beq`) |
| `srwi; subfc; subfe; andc` computing `n - 1` floored at 0 | `int k = n < 1 ? 0 : n - 1;` (`n > 0` and `n >= 1` give other sequences) |
| `mulhwu` vs `mulhw` dividing a 16-bit table value by a constant | `(unsigned)t[i] / 12` vs plain `t[i] / 12` |
| `lbz; cmpwi` vs `lbz; cmplwi` against a small constant | `(int)c == 3` vs `c == 3` on an `unsigned char` |
| The address of a stack object hoisted into a callee-saved register | A local pointer to it: `T* p = &local;` |
| An empty counted loop unrolled by 8, or a `subic.` countdown | That loop as written (`for (v = 0x7F; v >= 0; v--)` gives `subic.`) |
| A global object copied (inline default ctor, then `operator=`) before it is passed to a `const T&` parameter | The argument was an explicit temporary: `f(T(global))` |
| A caller narrows with `clrlwi` into an `unsigned char` parameter where the target doesn't | The parameter is `int` and the callee tests `(unsigned char)flag` itself |
| `addi; add` where ours reassociates to `add; addi` | Plain `a + b + K`, not `a + (b + K)` |
| The same float or double constant twice in one `.sdata2` stretch | A TU boundary between their first users: each TU pools its own literals, so split the unit there |
| A vtable in an unexpected unit | The unit that defines the class's first non-inline virtual (its key function) emits it |
| `lbzx base, idx` with `idx = (skip + K) + i` vs a folded `lbz K(base + i + skip)` | Index the result of an inline accessor: `inline T* Keys(H* h) { return (T*)h + Skip(h) + K; }`, then `Keys(h)[i]` |
| `mullw` operands swapped (`conv, n` vs `n, conv`) | Put the converted operand in a named local (`int q = (int)(t / n); q * n`); swapping the source operands does nothing |
| Branchless signed compare `xor; srawi 1; and; subf; srwi 31` whose `and` takes the other operand, flagged only as a register difference | The compare runs the other way (`a > b`, not `a < b`): a real semantic bug, so read the `and` operand before trusting the source |
| Every stack offset above one local struct is shifted | The struct's size is wrong: size it from what its callee writes (a 0x10 out-struct declared 0x18 moved all later slots) |
| A short loop's element offset left in a pointer (`addi rP, rBase, K` then `0x44(rP)`) where the target uses fixed displacements | Start the counter at 0 and index `a[c + 1]` instead of counting from 1 |
| Loads scheduled in another order than the target's (`lhz mC`, `lhz *data`, `lwz m8`) | Declare the locals in the target's load order; their initializing loads follow declaration order |
| A `clrlwi r, r, 16` re-mask of a value already loaded with `lhz`, at each use | The value was stored to an `unsigned short` local from an `int`-returning accessor (`unsigned short n = Count(p);`), or passed to an inline's `unsigned short` parameter |
| A loop counter and its offset temporary swapped in callee-saved registers, inside an otherwise exact function | The loop was an inline helper (its counter ranks as a temporary); a pointer bumped in the body may also go in the `for` increment (`i += 4, map++`) to order the increments |
| Four copies of the same stores, scheduled differently from the target and in a bigger frame | A short constant-count `for` loop that the compiler unrolled: write the loop, not the copies |
| A store the target keeps ahead of a global's load, where ours stores after it | Move that statement before the one that reads the global: a store through a pointer is never scheduled past a load that may alias it |
| The target's `bool` flag starts in the register of the block's other zero stores (one `li r30, 0`), ours loads its own zero | `bool b = a \|\| c;`, not `bool b = false; if (a \|\| c) b = true;` |
| `extrwi` + `xori 1` passed on as a `bool` with no 0/1 normalisation (`neg; or; srwi`) | A comparison, `(m & 2) == 0`, not `((m >> 1) & 1) ^ 1` |
| A `const` aggregate in `.rodata` that ours puts in `.data` | Define it before its first use; defined after it (behind an `extern const` declaration) it goes to `.data` |
| `divwu` by a constant with its own `li`, then a shift loop unrolled by the counter with no zero-trip test | A subtract loop the optimizer turned into a counted loop: `while (c >= K) { c -= K; s <<= 1; }` (a `/ K` written out gives `mulhwu`) |
| A float struct copied with `lwz`/`stw` pairs in a function that also does float math, where ours copies with `lfs`/`stfs` | The struct holds an array (`float m[3][3]`), not named float members; named members copy word-wise only in functions without float math |
| A `switch` whose `default` body comes before the case bodies | Write `default:` first in the source; bodies are laid out in source order |
| One `lis`/`addi` base reaching several arrays by displacement (`lbz 0x124(rB)`) where a single-array function uses each array's own symbol | The arrays are separate objects pooled from the start of the TU's `.data`: a function touching three or more of a TU's data objects addresses them all from the section start. That start marks the TU's first data object, so check the unit split before rewriting the data as one struct |
