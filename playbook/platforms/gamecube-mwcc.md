# GameCube / Wii with Metrowerks CodeWarrior

## Identify the toolchain

- **CodeWarrior, not ProDG (SN GCC):** look for the string `Metrowerks Target Resident Kernel for PowerPC` at the start of `.init` (the TRK vector table), and for `extab` + `extabindex` data sections, which only CodeWarrior's linker emits. Why: the compiler family decides the template, the compiler set and whether sibling projects' code can be reused. Code built with ProDG never matches under mwcc.
- **Typical CodeWarrior DOL order:** `.init`, `extab`, `extabindex`, `.text`, `.ctors`, `.dtors`, `.rodata`, `.data`, `.sdata`, `.sdata2`, then bss (`.bss`, `.sbss`, `.sbss2`).
- **C++ exceptions are off** when `extabindex` lists only a handful of functions, all inside the runtime library. Each entry is a triple (function, size, extab pointer).
- **RTTI is on** when class-name strings are referenced by 8-byte `{name*, bases*}` records, and vtables start with `{rtti*, 0, fn*, …}`. The template's default cflags include `-RTTI off`, so override it per library. `./dk analyze rtti` recovers classes, bases and vtables, and names primary vtables `__vt__<mangled>`.
  - Base lists are `{rtti*, offset}` pairs ending in one 0 word, and they list every base, indirect ones included.
  - Secondary vtables (multiple inheritance) carry a negative offset and sit inside the primary vtable's object.
- **RTTI records are emitted once per TU that needs them** (local copies). Each `.sdata` cluster of records (bases first, then the defining class) belongs to one TU, and the clusters run in link order. Why: it gives TU boundaries in `.sdata` and ties each cluster to the TU that owns the vtable in `.data`. Apply: never give duplicate copies global names before splitting.
- **Prologue order dates the compiler:**
  - GC ≤ 1.2.5 emits `mflr r0; stw r0,4(r1); stwu r1,-N(r1)`.
  - GC 1.3, 1.3.2 and 2.0 emit `stwu r1,-N(r1); mflr r0; stw r0,N+4(r1)`.
  - Apply: count both patterns over `.text` to tell old-compiler libraries (usually the SDK) from game code. Telling the 1.3+ versions apart needs real matched functions.

## Size and split signals

- Function count: dtk's analysis is the reliable number. A `blr` word count over `.text` is a quick recon estimate. It overcounts functions with several returns and misses ones that end in a tail-call `b`.
- Non-null `.ctors` entries ≈ the number of TUs with static initializers, a lower bound on C++ TU count. They're in link order.
- Assert and `__FILE__` strings name source files directly.

## Free wins from dtk

- On first analysis, dtk names runtime, MSL, TRK and parts of the Dolphin SDK by signature (hundreds of symbols) with no map file.
- The SDK OS banner (`Dolphin OS $Revision: N $`) plus a nearby build-date string dates the SDK. Pick SDK decomp sources from the same period.

## Known-source matching (`./dk analyze lib`)

- **Compile reference decomps and find their functions in the DOL, before decompiling anything by hand.** Why: the SDK, runtime and MSL make up a large share of a GC binary and already have matching source elsewhere. Apply: pin references in `dk.toml`, run `./dk analyze lib --apply`, and read `docs/generated/libraries.md`.
  - Compare instruction words with relocated fields masked: REL24 `0x03FFFFFC`, ADDR16 `0xFFFF`, SDA21 `0x1FFFFF`.
  - **The linker dead-strips unused functions,** so an object's surviving functions keep their order but not their offsets. Locate objects by chaining matches in order with bounded gaps, never by fixed offsets.
  - Several files can contain the same small static helper, so give each address to the strongest object.
- **Names flow along the call graph from exact matches.** A byte-identical function's `bl foo` and its `lis/addi` or SDA21 data references land on the real `foo` in the DOL. So every named relocation target of a matched function names a DOL address. Keep only names that are consistent both ways. Why: objdiff scores `bl fn_801D1DD8` vs `bl __CARDGetControlBlock` as a mismatch, so identical code shows up as 99.x% until the callees are named. Apply: `dk analyze lib --apply` does it, and re-running a port afterwards turns those near misses exact.
- **Reject weak evidence:** a one-function chain with no relocations (empty varargs stubs, `blr` helpers) matches unrelated code. Found as a false "dsp_debug.c" inside game code.
- **dtk's auto-label sizes are guesses.** A real split boundary often lands inside one (`Split … ends within symbol 'lbl_…'`). Trim the auto label to the boundary and re-split. Never trim a named symbol, because that indicates a real layout difference.
- **End a data split at its last object's end when that end is word-aligned, otherwise at the next symbol.** dtk fills the bytes after a split's last object with a `gap_` symbol that reaches the next symbol, and objdiff rejects one that leaves its section (`gap_… size out of section bounds`). Zero padding left after an aligned end becomes a harmless auto split.
- **A unit whose only code is a static initializer is dropped whole by the linker** (text and `.ctors` entry): nothing outside references its object. Detect: code and data exact, but the link has `.ctors` one entry short and every later section shifted. A lone `__sinit` after a TU without one of its own is that TU's tail: merge the two (`dk unit propose` prints the steps). Keep a genuinely standalone one (say, a TU holding only a header static) with `#pragma force_active on` after the includes.
- mwcc include flags: `-I<dir>` must be attached, while `-i <dir>` and `-ir <dir>` (recursive) take a separate argument. SDK sources include sibling private headers (`__os.h`), so they need `-ir src`.
- **Renaming functions to their real names wakes up dtk's signatures,** which then add known data symbols. Those can overlap labels dtk invented earlier, and the split fails with `Symbol X overlaps with symbol lbl_…`. Apply: drop the stale `lbl_` and re-split (`dk analyze lib --apply` does this automatically).

## Template mechanics worth knowing

- **Keep the unit list in a data file** (`config/<ID>/units.toml`) that `configure.py` loads, and keep only the named flag sets in Python. Why: tools can then add units and flip status without editing Python.
- **The template's compile line writes to an output directory** (`-o build/<ID>/src/<dir>`) and chains a dependency step after ` && `. Apply: to recompile with a different compiler, take `ninja -t commands <obj>`, cut at ` && `, and swap the compiler path and `-o` dir (`dk compiler-search` does this).
- When a lib uses `-inline deferred`, the template marks the unit `reverse_fn_order` in `objdiff.json`.
- The template's `.gitignore` ignores `/*.txt` at the root, so add `!/requirements.txt` if the repo has one.
- `objdiff.json` and dtk's split objects are regenerated only when **ninja** runs the split step, not by `configure.py` alone. Apply: after editing `splits.txt`, run the build before diffing (`dk unit add` does).
- `objdiff-cli diff -1 <target.o> -2 <base.o> -o - --format json` gives a one-shot, scriptable score (`left` = target side).
- **dtk's static-initializer auto units start with the previous TU's tail.** A TU's factory functions and virtual-base thunks follow its `__sinit`, so the real boundary is a few functions after each auto-unit start. dtk also gives each `__sinit` its own object.
  - Detect: move a leading function back to the previous TU only when that TU ends in a `__sinit`, and only if it is a thunk into it (2 instructions: `addi r3, r3, N` then `b` into the TU) or references it by at least 2 more calls/objects than its own unit. Move the next `__sinit` in when it touches this unit (one reference is enough), or when it touches only data no other function references (a header's static, initialised in every TU that includes it: `li r0, K; stw r0, x`), then pull later functions in the same way.
  - Verify against units you have already matched: this rule reproduced every accepted boundary, while a plain affinity count (no `__sinit` gate, margin 1) got 48 of 53.
- **The linker dead-strips data objects, not just functions.** Literals and statics used only by stripped functions, and globals nothing uses, vanish from the middle of `.sdata2`/`.bss`, and the rest pack together at their alignment. A byte diff can't line that up. Detect: code exact, data DIFF, and a dead-stripped function list. Apply: score by removing those objects, re-packing the kept ones in order, and comparing; the accept-time hash stays the final proof.
- **wibo puts itself in its own process group.** Killing a hung build means killing ninja's whole session, not just its group, or hung `mwldeppc` processes keep spinning for hours.
- **dtk never deletes asm for splits that no longer exist.** Tools that read `build/<ID>/asm` must only use files for units listed in `build/<ID>/config.json`, otherwise stale copies shadow current ones.
- **Proposing a unit's data from its code** (`./dk unit propose`): an object is the unit's when only the unit (or its own data) references it, when it is a vtable with a slot pointing at a unit function (an inlined constructor elsewhere may reference it too), or when the unit's static initializer constructs it (`.bss`). Then:
  - Keep each range between the declared neighbours in link order (an overlap breaks the split, a crossing makes a link-order cycle).
  - Split at runs of three or more objects that only other units reach; one or two are globals the unit defines for others.
  - Extend forward over what the unit uses (weak vtables, their RTTI copies) and over unreferenced objects: strings and `.bss` reached at an offset from a referenced base. The next unit's first object is its base, so it is referenced.
  - Leave a leading weak copy that an earlier unit also reaches to that unit, because the link keeps the first copy.
  - Repeat until stable, because chosen objects make their referents the unit's.
  - Verified on 349 linked units: 757 of 821 data ranges reproduced, against 619 with locality alone.

## Emission order

- **A unit's functions, and the strings and literals they introduce, come out in definition order.** Why: a unit whose functions all match still fails layout and literal offsets when the source has them in another order. Apply: sort the source by target address before chasing literal offsets (`./dk reorder <unit>` compiles before and after and keeps nothing that scores lower). A static table keeps its strings' place only if it moves with the function after it.
- **MWCC 1.3 to 2.7 holds back an `extern "C"` function that assigns a struct, or that needs an inline function's out-of-line copy, like a weak copy.** It comes out at the end of the next top-level declaration with C++ linkage (ahead of its own copies), most recently held first, and at the end of the unit when no such declaration follows. A following `extern "C"` function does not release it. Only in C++ mode: C files, C++ linkage, copy-initialisation (`S s = *p;`) and GC 1.0, 1.2.5 and 3.0 are unaffected. Why: verified in toys (a C++ function, a static or `struct Fwd;` after it releases it), in two units where such functions sat at the end, and in one where it came out after the next `extern "C"` function until a static table stood between them. Detect: the main `.text` ends in functions whose source lines run backwards, or one function sits after its successor. Apply: define them in reverse target order, or put the C++ declaration the target implies right after the function.
- **An inline function's out-of-line (weak) copy comes out where it is first needed, not where it is defined.**
  - The compiler queues each copy that a function needs: a call it does not inline, a virtual call, or a call in a branch that an inlined helper folds away. Code under `if (0)` queues nothing. A copy queues once per unit.
  - The queue is emitted at the end of the next top-level declaration that has C++ linkage, most recently queued first. After a C++ function, that is right after the function itself. After an `extern "C"` function, the copies wait for the next C++ declaration of any kind: a function, variable, typedef, struct, enum or static. An `extern "C"` declaration or a `#pragma` does not count.
  - Inline virtuals that only a vtable needs come at the end of the unit, after the last function. The classes come in reverse order of their key functions' definitions, and each class's virtuals in declaration order: the order the class itself declares them, so a derived class that declares its overrides out of slot order (`vf0, vf1, vf4, vf2…`) emits them that way while its slots still follow the base.
  - Why: verified in toys and in linked units. Of 933 weak copies there, 832 follow a C++ function, 67 follow `__sinit`, and the 34 that follow an `extern "C"` function either end the unit, follow a C++ declaration, or belong to a function emitted last. In a unit where a copy landed one function late, `struct Fwd;` after the `extern "C"` function moved it into the target's place.
  - Apply: to place a copy, reference it from the function the target puts it after; when that function is `extern "C"`, follow it with any C++ declaration.
  - Detect from a normal build: `-sym on` moves copies defined in headers into sections of their own.
- **Vtables and their RTTI (class names, base lists) come at the end of the unit's data, in reverse order of first need.**
  - A class's own vtable is needed where its key function is defined. The key function is the first non-inline virtual the class *declares*.
  - A weak vtable (a class without a key function) is needed where a function first stores it, through an inlined constructor or destructor.
  - So whether the destructor or another virtual is declared first decides whether the class's RTTI and vtable come before or after the weak copies of its bases' and members' vtables.
  - The class-name strings are created with the RTTI. An inline function compiled later that returns the same name reuses that string (`-str reuse`).
  - Why: verified in toys. In one unit, `.data` went from 37% to exact by declaring an overridden virtual before the destructor.
  - Apply: match the target's data order by choosing which virtual each class declares first.

## C++ object model

- **CodeWarrior class layout:** the vtable pointer sits where the first virtual function is declared (offset 0 when virtuals come first, after the preceding data otherwise); slot N is at vtable + 8 + 4N (`{rtti, 0, fn…}`). With a second base, its vtable (RTTI, offset, then its slots) is embedded right after the primary base's slots, and the virtuals the derived class adds come after that embedded part, so their slot numbers skip it. A virtual call is `lwz r12, voff(obj); lwz r12, 8+4N(r12); mtctr r12; bctrl`.
- **CodeWarrior virtual destructor shape:** `mr. this; beq`, vtable store(s), base destructor (`li r4, 0; bl __dt__B`, or inlined: another vtable store and another null check), `extsh. flags; ble`, then `operator delete`. A class-specific `operator delete` appears as `__dl__<Class>FPv`. RTTI base lists hold every ancestor, root first, and point at per-unit RTTI copies.
- **CodeWarrior mangling for calls:** `name__F<params>`, `name__<len><Class>[C]F<params>`, `Q<n>` for nested names. Return types aren't mangled, so a declaration may choose one freely.

## Register allocation (callee-saved r14-r31, f14-f31)

- **The callee-saved number of a value is set by its rank, and declaration order sets a named local's rank.** Values get registers one at a time in rank order. Each takes the lowest-numbered register already handed out that no overlapping value holds, and only otherwise a new one (r31 first, then down). Rank, highest first:
  1. Compiler temporaries, newest first: a call result held while the rest of an expression is evaluated, an inline function's locals and result, CSE values, loop-invariant addresses and strength-reduced loop pointers.
  2. Named locals, in declaration order. Where the first definition or first use sits does not matter, and nested-block locals count where they are declared.
  3. Parameters, last one first.

  A local or parameter that merges with a temporary takes the temporary's rank. That happens when its value feeds an operation (arithmetic, a compare, a store) straight from the call or expression that made it, or when a loop pointer is strength-reduced from it. A local that is only passed on as an argument or returned keeps its declaration rank. Use count, loop depth and `register` have no effect, and GC 1.3 through 2.7 behave alike. Why: the rule predicted every register in 360 generated test functions, where first-use order, live-range length and "highest free register" did not. Apply:
  - Compile with `-sym on` to get DWARF that names every variable's register; the code stays byte-identical. Compare that per variable with the target. `dk regs <unit> <fn>` does this.
  - Move the declarations into the target's order. Split `T x = e;` into `T x;` plus `x = e;` where needed. `dk regs --try` compiles those variants.
  - When a local ranks wrongly although its declaration order is right, change how its value is made: a `?:` initializer versus if/else assignments, or a value fed straight into an operation versus first passed on as an argument.
- **An inline function's parameters are temporaries, in volatile registers too.** A pointer that a caller keeps in a named local for repeated element access (`a = p + j; o[0] = a[0] + t * (b[0] - a[0]); …`) ranks below the CSE'd offset temporary. Passed as a parameter of a small `static inline` helper, it ranks above that temporary. Why: that turned three volatile-register near misses exact, and the helper's parameter order mattered in one of them. Apply: where the target gives the offset a later register than the pointers, move the repeated block into an inline helper that takes the pointers, and try its parameter orders.
- **Between plain operands, source order sets a commutative instruction's operand order. Compound operands the compiler orders itself.** `a*b + c` and `c + a*b` compile alike, but `x + y` and `y + x` do not (x, y variables or loads). Naming the compound part in a local makes it plain again (`int t = i * w; r = t + y;`), so the source order then decides. Why: verified in toys, and it turned two volatile-register near misses exact. Apply: `dk regs --try` maps the differing instructions to source lines (DWARF `.line`) and on those lines swaps plain operands and names compound ones, in both orders.
- **A loop index that is a signed counter converted to unsigned is strength-reduced to an offset, not a pointer.** `a[(unsigned int)i]`, or an accessor with an `unsigned int` index parameter called with an `int` counter, gives `addi r0, rOff, K; lwzx rD, rBase, r0` with rOff stepped by the element size. A plain `a[i]`, or an unsigned counter, gives `lwz rD, K(rPtr)` with the pointer stepped. The conversion also changes callee-saved ranks: the counter and the offset temporary rank differently than the counter and a pointer do. Why: verified in toys and on 7 real functions. Apply: `dk regs` reports which form each side uses, and `--try` compiles the cast and counter-type variants.

## Hand-written assembly

- **What the original wrote in assembly stays assembly, as a whole `asm` function.** Why: no C source compiles to it, and the SDK's own sources do the same. Detect: instructions or register use the compiler never emits (`mftb`, `mtspr`, cache ops, paired-single math outside the frame, `fmadds` under fp_contract off, an `mflr`-first prologue in a 1.3-era unit). An `mftb` inside otherwise compiled code is an inline `asm { }` block in a C function instead.
- **CodeWarrior's assembler syntax differs from a disassembler's.** `x@sda21(r0)` is `x(r13)` (or `x(r2)` for `.sdata2`/`.sbss2`), and `li rD, x@sda21` is `la rD, x(r13)`. `psq` takes a GQR number (`0`, not `qr0`), CR logic takes bit numbers (`crclr 6`, not `cr1eq`), and SPRs take numbers (`GQR0` = 912). It takes no qualified C++ names, so declare a callee `extern "C"` under its mangled name. Apply: `./dk gen --asm` converts and verifies.

## Compilers

- **Judge a compiler with whole units, not single trivial functions.** Why: simple code compiles identically on GC 1.3 through 2.7, so it can't discriminate. Apply: run `./dk compiler-search` on units with loops, floats, switches and inlining, and treat a version as pinned only when it's the sole exact match.
- **Find `-fp_contract` from the binary, not by trial.** Count fused multiply-add instructions (`fmadds`, `fmsubs`, `fnmadds`, `fnmsubs`) per unit. If game code uses separate `fmuls` + `fadds` everywhere and FMAs only appear in a few library units, game code was built with `-fp_contract off`.
- **`-char signed` with GC 1.3:** a char compare emitting `cmplwi` where the target has `extsb.` means the unit was built with `-char signed`. Set it per unit (`./dk unit flags`).
- **GC 1.3 vs 1.3.2 in a realigned frame:** `psq_st fN, …, 1, qr7` is GC 1.3; GC 1.3.2 emits `0, qr0`. GC 1.3 also auto-inlines less, so treat the compiler as per unit.
- **An asm function with interior global labels can hang `mwldeppc`** when its unit is linked on its own. Declare the unit linked instead of linking it in isolation.
- The template's compilers bundle has GC 1.0 through 3.0a5.2, Wii versions, and ProDG 3.5–3.9.3. `build/compilers/info.txt` maps each label to its mwcc build number.
- `mw_comment_version` in `config.yml`: 8 = GC 1.0+, 10 = GC 1.3.2+, 11 = GC 2.7+, 14 = GC 3.0a3+.
- SDK libraries were prebuilt by Nintendo with an older compiler than most games used, so expect per-library compiler versions.
