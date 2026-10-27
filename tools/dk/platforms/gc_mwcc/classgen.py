"""Drafts for compiler-shaped class functions: virtual destructors.

A CodeWarrior virtual destructor `C::~C() {}` is all compiler output: null check, the
vtable store(s), the base destructor (called, or inlined with its own vtable store), and
`operator delete` when the deleting flag is set. So instead of lifting it, the class chain
is rebuilt from metadata and compiled:

  class C          from the first vtable store (__vt__<C>)
  bases            from RTTI (single inheritance only, root first)
  inlined bases    one more vtable store each, in chain order
  called base      the remaining call (its own asm must store that base's vtable)
  operator delete  __dl__FPv (global) or __dl__<Q>FPv (declared in Q, a class of the chain)

The draft is verified like any other; an unnamed base destructor (fn_X) comes with the
rename the agent must apply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import gen
from .demangle import _Parser
from .dtk_symbols import is_auto_name


@dataclass
class ClassDraft:
    source: str               # the scratch TU (also the draft shown)
    function: str             # our symbol for the drafted function
    renames: dict             # our symbol -> target auto name, applied in the comparison
    advice: list              # "./dk rename ..." lines for the agent
    notes: list


def _class_of_vtable(sym: str) -> tuple | None:
    if not sym.startswith("__vt__"):
        return None
    p = _Parser(sym[6:])
    try:
        q = p.qual()
    except (ValueError, IndexError):
        return None
    return q if p.i == len(p.s) else None


def _mangle(q: tuple) -> str:
    parts = "".join(f"{len(x)}{x}" for x in q)
    return parts if len(q) == 1 else f"Q{len(q)}{parts}"


def vtable_stores(insns: list) -> list[tuple[tuple, int]]:
    """(class, offset) for each `lis/addi __vt__X; stw rY, off(rZ)`, in order."""
    held: dict[str, str] = {}
    out = []
    for insn in insns:
        r = insn.reloc()
        if r and r[0] == "l" and insn.op == "addi" and r[1].startswith("__vt__"):
            held[insn.args[0]] = r[1]
        elif insn.op == "stw" and insn.args[0] in held:
            m = gen._MEM.match(insn.args[1])
            q = _class_of_vtable(held.pop(insn.args[0]))
            if m and q and re.fullmatch(r"-?(?:0x[0-9A-Fa-f]+|\d+)", m["disp"]):    # not into a global
                out.append((q, int(m["disp"], 0)))
        elif insn.args and insn.args[0] in held and insn.op not in ("stw",):
            held.pop(insn.args[0], None)
    return out


def chain_of(classes: dict, name: str, records: list | None = None) -> list[str] | None:
    """[C, parent, ..., root] from RTTI, or None for multiple inheritance. `records` is every
    RTTI record: classes have per-unit copies, and base lists point at any of them."""
    by_rtti = {c["rtti"]: c["name"] for c in (records or classes.values())}
    c = classes.get(name)
    if c is None:
        return None
    bases = [by_rtti.get(b) for b, off in c["bases"]]
    if None in bases or any(off for _, off in c["bases"]):
        return None
    for i, b in enumerate(bases):                        # each base's bases: the ones before it
        if [by_rtti.get(x) for x, _ in classes[b]["bases"]] != bases[:i]:
            return None
    return [name] + bases[::-1]


def solve_dtor(insns: list, classes: dict, asm_of, records: list | None = None) -> ClassDraft | gen.Unsupported | None:
    """A draft for a virtual destructor, Unsupported with a reason, or None if not one."""
    calls = [i.args[0] for i in insns if i.op == "bl"]
    deletes = [c for c in calls if c.startswith("__dl__")]
    if not deletes or not any(i.op == "extsh." for i in insns):
        return None
    stores = vtable_stores(insns)
    if not stores:
        return None
    others = [c for c in calls if not c.startswith("__dl__")]
    if len(deletes) != 1 or len(others) > 1:
        return gen.Unsupported("destructor with member destructors (not modelled yet)")
    voffs = {off for _, off in stores}
    if len(voffs) != 1:
        return gen.Unsupported("destructor storing vtables at several offsets")
    voff = voffs.pop()
    cname = "::".join(stores[0][0])
    chain = chain_of(classes, cname, records)
    if chain is None:
        return gen.Unsupported(f"{cname}: no single-inheritance chain in RTTI")
    stored = ["::".join(q) for q, _ in stores]
    if stored != chain[:len(stored)]:
        return gen.Unsupported("vtable stores do not follow the RTTI base chain")
    k = len(stored)                                      # chain[1:k] are inlined
    renames, advice = {}, []
    if others:
        if k >= len(chain):
            return gen.Unsupported("calls a base destructor but RTTI has no further base")
        base = chain[k]
        ours = f"__dt__{_mangle(tuple(base.split('::')))}Fv"
        callee = others[0]
        if callee != ours:
            if not is_auto_name(callee):
                return gen.Unsupported(f"base destructor call to {callee}, expected {ours}")
            if not any(q == tuple(base.split("::")) for q, _ in vtable_stores(gen.parse(asm_of(callee) or ""))):
                return gen.Unsupported(f"{callee} does not store {base}'s vtable: not its destructor")
            renames[ours] = callee
            advice.append(f"./dk rename {callee} {ours}")
    elif k < len(chain):
        return gen.Unsupported("RTTI has a base whose destructor is neither called nor inlined")
    delete = deletes[0]
    delete_in = None
    if delete != "__dl__FPv":
        m = re.fullmatch(r"__dl__(.+)FPv", delete)
        owner = None
        if m:
            p = _Parser(m[1])
            try:
                owner = p.qual()
            except (ValueError, IndexError):
                owner = None
        if owner is None or "::".join(owner) not in chain:
            return gen.Unsupported(f"operator delete {delete} not from a class of the chain")
        delete_in = "::".join(owner)

    # the classes, root first; inlined destructors have bodies, the rest are declared
    skeletons = {}
    for name in chain:
        q = name.split("::")
        if len(q) > 2:
            return gen.Unsupported("classes nested three deep")
        if len(q) == 2:
            skeletons.setdefault(q[0], []).append(q[1])
    out = []
    for outer, inner in sorted(skeletons.items()):
        out.append(f"struct {outer} {{ {' '.join(f'struct {x};' for x in inner)} }};")
    for i, name in enumerate(reversed(chain)):
        depth = len(chain) - 1 - i                       # index in chain
        parent = chain[depth + 1] if depth + 1 < len(chain) else None
        head = f"struct {name}" + (f" : {parent}" if parent else "")
        short = name.split("::")[-1]
        members = []
        if parent is None and voff:
            members.append(f"char _pad0[0x{voff:X}];")
        inline = 0 < depth < k
        members.append(f"virtual ~{short}()" + (" {}" if inline else ";"))
        if name == delete_in:
            members.append("static void operator delete(void*);")
        out.append(head + " { " + " ".join(members) + " };")
    cshort = cname.split("::")[-1]
    out.append(f"{cname}::~{cshort}() {{}}")
    ours = f"__dt__{_mangle(tuple(cname.split('::')))}Fv"
    notes = [f"class chain (root last): {' : '.join(chain)}"]
    if voff:
        notes.append(f"vtable pointer at +0x{voff:X}")
    return ClassDraft("\n".join(out) + "\n", ours, renames, advice, notes)


# --- destructors as a class model: members, member arrays, inlined bases, a body ----------------

@dataclass
class DtorModel:              # what gen.render_tu needs to write C::~C() around a lifted body
    cls: str                  # C (qualified)
    decl: str                 # its destructor's declaration: `virtual ~C();` or `~C();`
    dtor_at: int | None       # C declares the first virtual function (its destructor) here: the vtable pointer
    start: int                # where C's own members start (the size of its base)
    objects: dict             # offset -> (class, count, element size): the members destroyed here
    classes: list             # [(class, parent, member lines)]: every other class, bases first


@dataclass
class DtorPlan:               # a destructor lifted into a model: render each variant, then verify
    lifted: object
    model: DtorModel
    function: str             # our symbol for it
    renames: dict             # our symbol -> target auto name, applied in the comparison
    advice: list
    notes: list


def dtor_frame(body: list) -> tuple[str, list] | None:
    """CodeWarrior's destructor frame `if (this) { ...; if (flag > 0) operator delete(this); }
    return this;` -> (operator delete, the code in between as a function of `this` in r3
    ending in blr), or None when `body` (frame stripped) is not shaped like one."""
    beq = next((i for i, x in enumerate(body[:4]) if x.op == "beq"), None)
    if beq is None:
        return None
    this = flag = None
    for x in body[:beq]:
        if x.op == "mr." and x.args[1] == "r3":
            this = x.args[0]
        elif x.op == "mr" and x.args[1] == "r4":
            flag = x.args[0]
        else:
            return None
    if this is None or len(body) < beq + 6 or body[-1].op != "blr" or body[-2].op != "mr" or body[-2].args != ("r3", this):
        return None
    j = len(body) - 3
    if not (body[j].op == "bl" and body[j].args[0].startswith("__dl__")):
        return None
    delete, end = body[j].args[0], range(body[j].address + 4, body[-2].address + 1)   # (epilogue removed)
    j -= 1
    if body[j].op == "mr" and body[j].args == ("r3", this):
        j -= 1
    elif flag is not None:
        return None
    if body[j].op != "ble" or gen._target(body[j]) not in end or gen._target(body[beq]) not in end:
        return None
    ble = j
    # the flag test; the body's last instructions may be scheduled between it and the branch
    j = next((i for i in range(ble - 1, max(beq, ble - 4), -1) if body[i].op == "extsh."), None)
    if j is None or body[j].args[1] != (flag or "r4") or \
            any(x.op.endswith(".") or x.op.startswith(("cmp", "b")) or "r0" in x.args[:1] for x in body[j + 1:ble]):
        return None
    middle = body[beq + 1:j] + body[j + 1:ble]
    if flag is None and any(x.op == "bl" for x in middle):
        return None
    synth = [gen.Insn(body[0].address, 0x7C000378, "mr", (this, "r3"))] + middle \
        + [gen.Insn(body[j].address, 0x4E800020, "blr", ())]
    return delete, synth


def looks_like_dtor(text: str | None) -> bool:
    """A function's asm starts like a destructor (for auto-named callees): `if (this)`, then
    the deleting flag (r4, or the register it was saved in) tested with extsh. Its delete may
    be a class's inline operator delete, so only the start is checked."""
    if not text:
        return False
    try:
        body = gen.strip_frame(gen.parse(text))
    except gen.Unsupported:
        return False
    beq = next((i for i, x in enumerate(body[:4]) if x.op == "beq"), None)
    if beq is None:
        return False
    head = {x.op: x.args for x in body[:beq]}
    flag = head["mr"][0] if head.get("mr", ("", ""))[1:] == ("r4",) else "r4"
    if head.get("mr.", ("", ""))[1:] != ("r3",) or len(head) != beq:
        return False
    return any(x.op == "extsh." and x.args[1] == flag for x in body[beq + 1:])


def _is_this(cond) -> bool:
    """`this != 0`, the null check an inlined destructor starts with."""
    return isinstance(cond, gen.Op) and cond.op == "!=" and cond.args == (gen.Param("r3"), gen.Const(0))


def _object_class(callee: str, asm_of) -> tuple[str, bool]:
    """(class, auto-named) of a destructor: its mangled name, else the vtable its code stores,
    else Unk<address>."""
    m = re.fullmatch(r"__dt__(.+)Fv", callee)
    if m:
        p = _Parser(m[1])
        try:
            q = p.qual()
            if p.i == len(p.s):
                return "::".join(q), False
        except (ValueError, IndexError):
            pass
        raise gen.Unsupported(f"destructor name {callee}")
    if not is_auto_name(callee):
        raise gen.Unsupported(f"destructor {callee} is not named like one")
    stores = vtable_stores(gen.parse(asm_of(callee) or ""))
    if stores:
        return "::".join(stores[0][0]), True
    return f"Unk{callee.removeprefix('fn_')[:8].upper()}", True


def plan_dtor(name: str, address: int, insns: list, classes: dict, asm_of, records: list, literal, signature,
              callers=None) -> DtorPlan | gen.Unsupported | None:
    """A destructor as a class model around a lifted body: vtable store(s), the body, member
    destructors (named, or auto-named functions shaped like destructors), member arrays
    (__destroy_arr), then inlined base destructors (each `if (this)`, its vtable store) and
    the called base destructor. None when `insns` is not a destructor."""
    try:
        frame = dtor_frame(gen.strip_frame(insns))
    except gen.Unsupported:
        return None
    if frame is None:
        return None
    delete, synth = frame
    try:
        lifted = gen.lift(synth, literal, True, signature, None, callers=callers,
                          dtor_of=lambda c: looks_like_dtor(asm_of(c)))
    except gen.Unsupported as e:
        return gen.Unsupported(f"destructor: {e}")
    tree = list(lifted.tree)
    if tree and isinstance(tree[-1], gen.Return):
        tree.pop()
    lead = []
    while tree and isinstance(tree[0], gen.VtStore):
        lead.append(tree.pop(0))
    # the tail, from the end: inlined bases (if (this) { vtable store; ... }), the called base,
    # then the members, last declared first
    inlined, base_call = [], None
    if tree and isinstance(tree[-1], gen.If) and _is_this(tree[-1].cond) and not tree[-1].other:
        level = tree.pop()
        while True:
            arm = list(level.then)
            if not arm or not isinstance(arm[0], gen.VtStore):
                return gen.Unsupported("destructor: an inlined base destructor with code of its own")
            inlined.append(arm.pop(0))
            if not arm:
                break
            if len(arm) == 1 and isinstance(arm[0], gen.If) and _is_this(arm[0].cond) and not arm[0].other:
                level = arm[0]
                continue
            if len(arm) == 1 and isinstance(arm[0], gen.DtorCall) and arm[0].flag == 0 and arm[0].offset == 0 \
                    and not arm[0].count:
                base_call = arm[0]
                break
            return gen.Unsupported("destructor: an inlined base destructor with code of its own")
    elif tree and isinstance(tree[-1], gen.DtorCall) and tree[-1].flag == 0:
        base_call = tree.pop()
        if base_call.offset or base_call.count:
            return gen.Unsupported("destructor of a base class at an offset (multiple inheritance)")
    members = []
    while tree and isinstance(tree[-1], gen.DtorCall):
        members.insert(0, tree.pop())
    body = tree
    for x in gen._flatten(body):
        if isinstance(x, (gen.VtStore, gen.DtorCall)):
            return gen.Unsupported("destructor: an object destroyed (or a vtable stored) inside its body")
    if any(m.flag != -1 for m in members):
        return gen.Unsupported("destructor of a base class at an offset (multiple inheritance)")
    offs = [m.offset for m in members]
    if offs != sorted(set(offs), reverse=True):
        return gen.Unsupported("member destructors out of declaration order")

    # the class and its bases
    stored = lead + inlined
    if len({v.offset for v in stored}) > 1:
        return gen.Unsupported("destructor storing vtables at several offsets")
    renames, advice, notes = {}, [], []
    own = re.fullmatch(r"__dt__(.+)Fv", name)
    named = None
    if own:
        p = _Parser(own[1])
        try:
            q = p.qual()
            named = "::".join(q) if p.i == len(p.s) else None
        except (ValueError, IndexError):
            named = None
    voff = stored[0].offset if stored else None
    if stored:
        if not lead:
            return gen.Unsupported("destructor: an inlined base destructor but no vtable store of its own")
        qs = [_class_of_vtable(v.vtable) for v in stored]
        if None in qs:
            return gen.Unsupported(f"vtable symbol {stored[0].vtable}")
        names_ = ["::".join(q) for q in qs]
        cname = names_[0]
        chain = chain_of(classes, cname, records)
        if chain is None:
            return gen.Unsupported(f"{cname}: no single-inheritance chain in RTTI")
        if names_ != chain[:len(names_)]:
            return gen.Unsupported("vtable stores do not follow the RTTI base chain")
        k = len(names_)
        if len(lead) > 1 and inlined:
            return gen.Unsupported("destructor: inlined bases both with and without a null check")
    else:
        if named is None:
            if not is_auto_name(name):
                return gen.Unsupported(f"destructor {name} without a class")
            cname = f"Unk{address:08X}"
        else:
            cname = named
        chain, k = [cname], 1
    if named is not None and named != cname:
        return gen.Unsupported(f"{name} stores {cname}'s vtable")
    if base_call is not None:
        if stored and k >= len(chain):
            return gen.Unsupported("calls a base destructor but RTTI has no further base")
        try:
            base, auto = _object_class(base_call.callee, asm_of)
        except gen.Unsupported as e:
            return e
        if stored:
            if auto and base != chain[k]:
                return gen.Unsupported(f"{base_call.callee} does not store {chain[k]}'s vtable: not its destructor")
            if not auto and base != chain[k]:
                return gen.Unsupported(f"base destructor call to {base_call.callee}, expected {chain[k]}'s")
            base = chain[k]
        else:
            chain = [cname, base]
        ours = f"__dt__{_mangle(tuple(base.split('::')))}Fv"
        if base_call.callee != ours:
            renames[ours] = base_call.callee
            advice.append(f"./dk rename {base_call.callee} {ours}")
    elif stored and k < len(chain):
        return gen.Unsupported("RTTI has a base whose destructor is neither called nor inlined")
    delete_in = None
    if delete != "__dl__FPv":
        m = re.fullmatch(r"__dl__(.+)FPv", delete)
        owner = None
        if m:
            p = _Parser(m[1])
            try:
                owner = p.qual()
            except (ValueError, IndexError):
                owner = None
        if owner is None or "::".join(owner) not in chain:
            return gen.Unsupported(f"operator delete {delete} not from a class of the chain")
        delete_in = "::".join(owner)

    # member objects: one class per destructor, sized by its arrays (else one byte)
    objects, sizes, member_classes = {}, {}, []
    for mcall in members:
        try:
            mcls, auto = _object_class(mcall.callee, asm_of)
        except gen.Unsupported as e:
            return e
        if mcls in chain:
            return gen.Unsupported("a member of its container's own class (base class or composition?)")
        ours = f"__dt__{_mangle(tuple(mcls.split('::')))}Fv"
        if mcall.callee != ours:
            if renames.get(ours, mcall.callee) != mcall.callee:
                return gen.Unsupported(f"two destructors for {mcls}")
            renames[ours] = mcall.callee
            line = f"./dk rename {mcall.callee} {ours}"
            if line not in advice:
                advice.append(line)
        if mcall.count:
            if sizes.get(mcls, mcall.size) != mcall.size:
                return gen.Unsupported(f"{mcls} arrays with two element sizes")
            sizes[mcls] = mcall.size
        objects[mcall.offset] = (mcls, mcall.count, mcall.size)
        if mcls not in member_classes:
            member_classes.append(mcls)
    for off, (mcls, count, size) in objects.items():
        objects[off] = (mcls, count, sizes.get(mcls, 1))
    ends = sorted(objects)
    for a, b in zip(ends, ends[1:]):
        if a + objects[a][2] * max(objects[a][1], 1) > b:
            return gen.Unsupported("member objects overlap")

    # the class declarations: the chain root first; inlined bases have empty inline destructors
    cshort = cname.split("::")[-1]
    polymorphic = bool(stored)
    model_classes = []
    for depth in range(len(chain) - 1, 0, -1):
        cls = chain[depth]
        parent = chain[depth + 1] if depth + 1 < len(chain) else None
        short = cls.split("::")[-1]
        lines = []
        if parent is None and polymorphic and voff:
            lines.append(f"char _pad0[0x{voff:X}];")
        virtual = "virtual " if polymorphic else ""
        lines.append(f"{virtual}~{short}()" + (" {}" if depth < k else ";"))
        if parent is None and not polymorphic:
            lines.insert(0, "char _pad0[0x1];")
        if cls == delete_in:
            lines.append("static void operator delete(void*);")
        model_classes.append((cls, parent, lines))
    for mcls in member_classes:
        if mcls in chain:
            return gen.Unsupported("a member of its container's own class (base class or composition?)")
        model_classes.append((mcls, None, [f"char _pad0[0x{sizes.get(mcls, 1):X}];", f"~{mcls.split('::')[-1]}();"]))
    parent = chain[1] if len(chain) > 1 else None
    model_classes.append((cname, parent, []))
    start = (voff + 4 if polymorphic else 1) if parent else 0
    decl = f"{'virtual ' if polymorphic else ''}~{cshort}();"
    extra = ["static void operator delete(void*);"] if delete_in == cname else []
    if extra:
        decl = decl + " " + extra[0]
    model = DtorModel(cname, decl, voff if polymorphic and parent is None else None, start, objects, model_classes)
    bound = lifted.bound.get(("this",)) or lifted.bound.get(gen.Param("r3"))
    if bound and "::".join(bound) not in chain:
        return gen.Unsupported(f"calls {'::'.join(bound)} methods on this ({cname})")
    lifted.tree = body
    lifted.ret, lifted.ret_fp = None, False
    if any(isinstance(x, gen.Return) for x in gen._flatten(body)):
        return gen.Unsupported("a destructor body that returns early")
    ours = f"__dt__{_mangle(tuple(cname.split('::')))}Fv"
    notes.append(f"class chain (root last): {' : '.join(chain)}")
    if objects:
        notes.append("members destroyed: " + ", ".join(f"+0x{o:X} {c}" + (f"[{n}]" if n else "")
                                                       for o, (c, n, _) in sorted(objects.items())))
    if polymorphic and voff:
        notes.append(f"vtable pointer at +0x{voff:X}")
    return DtorPlan(lifted, model, ours, renames, advice, notes)


# --- constructors ----------------------------------------------------------------------------

def _value_text(v, params: dict) -> str | None:
    if isinstance(v, gen.Const):
        return gen._int_text(v.value)
    if isinstance(v, gen.Literal):
        return gen.float_literal(v.raw)
    if isinstance(v, gen.Param) and v.reg != "r3":
        return params.setdefault(v.reg, f"arg{int(v.reg[1:]) - 3}" if v.reg[0] == "r" else f"farg{int(v.reg[1:]) - 1}")
    return None


def _fields(fields: dict, start: int) -> list[str]:
    out, cur = [], start
    for off in sorted(fields):
        ctype = fields[off]
        width = gen.SIZES.get(ctype, 4)
        if off < cur or off % width:
            raise gen.Unsupported(f"field +0x{off:X} does not fit after the base")
        if off > cur:
            out.append(f"char _pad{cur:X}[0x{off - cur:X}];")
        out.append(f"{ctype} m{off:X};")
        cur = off + width
    return out


def solve_ctor(insns: list, classes: dict, asm_of, records: list, literal, signature):
    """Drafts (one per field attribution) for a constructor, Unsupported, or None."""
    calls = [i.args[0] for i in insns if i.op == "bl"]
    if any(c.startswith("__dl__") for c in calls) or not vtable_stores(insns):
        return None
    try:
        lifted = gen.lift(insns, literal, True, signature, None, ctor_mode=True)
    except gen.Unsupported as e:
        return gen.Unsupported(f"constructor: {e}")
    if gen.has_flow(lifted) and any(isinstance(x, gen.If) for x in lifted.tree):
        return gen.Unsupported("constructor with control flow")
    tree = lifted.tree
    if any(isinstance(x, gen.MemberCtor) for x in tree):
        return gen.Unsupported("constructor with member objects")
    vts = [x for x in tree if isinstance(x, gen.VtStore)]
    if len({v.offset for v in vts}) != 1:
        return gen.Unsupported("constructor storing vtables at several offsets")
    voff = vts[0].offset
    stored = []
    for v in vts:
        q = _class_of_vtable(v.vtable)
        if q is None:
            return gen.Unsupported(f"vtable symbol {v.vtable}")
        stored.append("::".join(q))
    cname = stored[-1]
    chain = chain_of(classes, cname, records)
    if chain is None:
        return gen.Unsupported(f"{cname}: no single-inheritance chain in RTTI")
    if stored[::-1] != chain[:len(stored)]:
        return gen.Unsupported("vtable stores do not follow the RTTI base chain")
    k = len(stored)
    base_call = lifted.base_call
    others = [c for c in lifted.calls if c is not base_call]
    if others:
        return gen.Unsupported("constructor calls more than its base constructor")
    if base_call is None and k < len(chain):
        return gen.Unsupported("RTTI has a base that is neither constructed nor inlined")
    if base_call is not None and k >= len(chain):
        return gen.Unsupported("calls a base constructor but RTTI has no further base")
    inits = [x for x in tree if isinstance(x, gen.Store)]
    for st in inits:
        if not (isinstance(st.loc, gen.Field) and st.loc.base == gen.Param("r3")):
            return gen.Unsupported("constructor writes something other than its own fields")
    renames, advice = {}, []
    base_params = []
    if base_call is not None:
        base = chain[k]
        if base_call.cxx is not None:
            base_params = [t.text for t in base_call.cxx.params]
        else:
            base_params = ["int"] * (base_call.n_gpr - 1)
        callee = base_call.callee
        if gen.is_auto(callee):
            target = gen.parse(asm_of(callee) or "")
            if not any(q == tuple(base.split("::")) for q, _ in vtable_stores(target)) or \
                    any(i.op == "bl" and i.args[0].startswith("__dl__") for i in target):
                return gen.Unsupported(f"{callee} does not look like {base}'s constructor")
        elif base_call.cxx is None or base_call.cxx.quals != tuple(base.split("::")):
            return gen.Unsupported(f"base constructor call to {callee}, expected one of {base}")
    params: dict = {}
    base_args = []
    if base_call is not None:
        for a in base_call.args[1:]:
            text = _value_text(a, params)
            if text is None:
                return gen.Unsupported("base constructor argument the draft cannot express")
            base_args.append(text)
    values = []
    for st in inits:
        text = _value_text(st.value, params)
        if text is None:
            return gen.Unsupported("field initialised from a value the draft cannot express")
        values.append(text)
    param_types = {}
    for st, text in zip(inits, values):
        if isinstance(st.value, gen.Param):
            param_types[st.value.reg] = st.ctype
    if base_call is not None:
        for a, t in zip(base_call.args[1:], base_params):
            if isinstance(a, gen.Param):
                param_types.setdefault(a.reg, t)
    regs_r = sorted(int(r[1:]) for r in params if r[0] == "r")
    regs_f = sorted(int(r[1:]) for r in params if r[0] == "f")
    sig = [(param_types.get(f"r{n}", "int"), params.get(f"r{n}", "")) for n in range(4, max(regs_r, default=3) + 1)]
    sig += [(param_types.get(f"f{n}", "float"), params.get(f"f{n}", "")) for n in range(1, max(regs_f, default=0) + 1)]
    owner_of_store = []
    current = None
    for x in tree:                                   # positional: after which vtable store
        if isinstance(x, gen.VtStore):
            current = "::".join(_class_of_vtable(x.vtable))
        elif isinstance(x, gen.Store):
            owner_of_store.append(current or stored[0])
    drafts = []
    for attribution in ("derived", "positional"):
        owner = [cname] * len(inits) if attribution == "derived" else owner_of_store
        if attribution == "positional" and owner == [cname] * len(inits):
            continue
        if any(o != cname and o not in stored for o in owner):
            continue
        try:
            source = _ctor_source(chain, k, cname, voff, inits, values, owner, sig, base_params, base_args, base_call is not None)
        except gen.Unsupported as e:
            if not drafts and attribution == "derived":
                last = e
            continue
        drafts.append(ClassDraft(source, f"__ct__{_mangle(tuple(cname.split('::')))}F", {}, [],
                                 [f"class chain (root last): {' : '.join(chain)}"]))
    if not drafts:
        return locals().get("last") or gen.Unsupported("no constructor layout fits")
    for d in drafts:
        if base_call is not None and gen.is_auto(base_call.callee):
            d.renames = {"@base_ctor": base_call.callee}
    return drafts


@dataclass
class CtorModel:              # what gen.render_tu needs to write C::C(...) : inits { body }
    cls: str                  # C (qualified)
    dtor_at: int | None       # C is the polymorphic root: `virtual ~C();` declared at its vtable pointer
    start: int                # where C's own members start (the size of its base)
    objects: dict             # offset -> (class, count, element size): member objects
    classes: list             # [(class, parent, member lines)]: every other class, bases first
    base: str | None          # the called base constructor's class
    base_call: object         # its gen.Call (None: no base, or an inlined one)
    inits: list               # [(offset, gen.Store | gen.MemberCtor)] in declaration order


@dataclass
class CtorPlan:
    lifted: object
    model: CtorModel
    function: str
    renames: dict
    advice: list
    notes: list


def plan_ctor(name: str, address: int, insns: list, classes: dict, asm_of, records: list, literal, signature,
              callers=None) -> CtorPlan | gen.Unsupported | None:
    """A constructor (by its mangled name) as a class model around a lifted body: the base
    constructor call, the vtable store(s), member initializers (stores to fields and member
    constructors, __construct_array arrays, in declaration order), then the body. None when
    `name` is not a constructor's."""
    from .demangle import demangle
    own = demangle(name) if name.startswith("__ct__") else None
    if own is None or own.special != "ctor" or not own.quals:
        return None
    cname = "::".join(own.quals)
    try:
        lifted = gen.lift(insns, literal, True, signature, None, ctor_mode=True, callers=callers, own=own)
    except gen.Unsupported as e:
        return gen.Unsupported(f"constructor: {e}")
    tree = list(lifted.tree)
    if tree and isinstance(tree[-1], gen.Return):
        if tree[-1].value not in (gen.Param("r3"), None):
            return gen.Unsupported("a constructor that returns something other than this")
        tree.pop()
    base_call = lifted.base_call
    if base_call is not None:
        if not tree or tree[0] is not base_call:
            return gen.Unsupported("constructor: code before the base constructor")
        tree.pop(0)
    stored = []
    while tree and isinstance(tree[0], gen.VtStore):
        stored.append(tree.pop(0))
    last = max((i for i, x in enumerate(tree) if isinstance(x, gen.MemberCtor)), default=-1)
    inits, body = tree[:last + 1], tree[last + 1:]
    for x in gen._flatten(body):
        if isinstance(x, (gen.VtStore, gen.MemberCtor)):
            return gen.Unsupported("constructor: a vtable stored or an object constructed in its body")
    if any(isinstance(x, gen.Return) for x in gen._flatten(body)):
        return gen.Unsupported("a constructor that returns early")
    for x in inits:
        if isinstance(x, gen.Store) and not (isinstance(x.loc, gen.Field) and x.loc.base == gen.Param("r3")):
            return gen.Unsupported("constructor: an initializer that is not one of its fields")
        if not isinstance(x, (gen.Store, gen.MemberCtor)):
            return gen.Unsupported("constructor: code among the member initializers")
    offs = [x.loc.offset if isinstance(x, gen.Store) else x.offset for x in inits]
    if offs != sorted(set(offs)):
        return gen.Unsupported("member initializers out of declaration order")

    # the class and its bases
    renames, advice = {}, []
    if len({v.offset for v in stored}) > 1:
        return gen.Unsupported("constructor storing vtables at several offsets")
    voff = stored[0].offset if stored else None
    if stored:
        qs = [_class_of_vtable(v.vtable) for v in stored]
        if None in qs:
            return gen.Unsupported(f"vtable symbol {stored[0].vtable}")
        names_ = ["::".join(q) for q in qs]
        if names_[-1] != cname:
            return gen.Unsupported(f"{name} stores {names_[-1]}'s vtable last")
        chain = chain_of(classes, cname, records)
        if chain is None:
            return gen.Unsupported(f"{cname}: no single-inheritance chain in RTTI")
        if names_[::-1] != chain[:len(names_)]:
            return gen.Unsupported("vtable stores do not follow the RTTI base chain")
        k = len(names_)
        if base_call is None and k < len(chain):
            return gen.Unsupported("RTTI has a base that is neither constructed nor inlined")
        if base_call is not None and k >= len(chain):
            return gen.Unsupported("calls a base constructor but RTTI has no further base")
    else:
        chain, k = [cname], 1
    base = None
    if base_call is not None:
        if base_call.cxx is not None:
            if base_call.cxx.special != "ctor":
                return gen.Unsupported(f"{base_call.callee} called first is not a constructor")
            base = "::".join(base_call.cxx.quals)
            if stored and base != chain[k]:
                return gen.Unsupported(f"base constructor call to {base_call.callee}, expected one of {chain[k]}")
        elif stored and gen.is_auto(base_call.callee):
            target = gen.parse(asm_of(base_call.callee) or "")
            if not any(q == tuple(chain[k].split("::")) for q, _ in vtable_stores(target)):
                return gen.Unsupported(f"{base_call.callee} does not look like {chain[k]}'s constructor")
            base = chain[k]
        else:
            return gen.Unsupported(f"constructor: the first call, {base_call.callee}, is not a known constructor")
        if not stored:
            chain = [cname, base]
    # member objects
    objects, member_classes, sizes = {}, {}, {}
    for x in inits:
        if not isinstance(x, gen.MemberCtor):
            continue
        if x.cxx is None or x.cxx.special != "ctor":
            return gen.Unsupported(f"member constructor {x.callee} is not named like one")
        mcls = "::".join(x.cxx.quals)
        if mcls in chain:
            return gen.Unsupported("a member of its container's own class (base class or composition?)")
        if x.count:
            if x.cxx.params:
                return gen.Unsupported("an array constructed with arguments")
            if sizes.setdefault(mcls, x.size) != x.size:
                return gen.Unsupported(f"{mcls} arrays with two element sizes")
            if x.dtor and x.dtor != f"__dt__{_mangle(x.cxx.quals)}Fv":
                return gen.Unsupported(f"{mcls}'s array destroyed by {x.dtor}")
        decl = f"{x.cxx.quals[-1]}({', '.join(t.text for t in x.cxx.params)});"
        lines = member_classes.setdefault(mcls, [])
        if decl not in lines:
            lines.append(decl)
        if x.dtor and f"~{x.cxx.quals[-1]}();" not in lines:
            lines.append(f"~{x.cxx.quals[-1]}();")
        objects[x.offset] = (mcls, x.count, 0)
    for off, (mcls, count, _) in objects.items():
        objects[off] = (mcls, count, sizes.get(mcls, 1))
    ends = sorted(objects)
    for a, b in zip(ends, ends[1:]):
        if a + objects[a][2] * max(objects[a][1], 1) > b:
            return gen.Unsupported("member objects overlap")

    polymorphic = bool(stored)
    model_classes = []
    for depth in range(len(chain) - 1, 0, -1):
        cls = chain[depth]
        parent = chain[depth + 1] if depth + 1 < len(chain) else None
        short = cls.split("::")[-1]
        lines = []
        if parent is None and polymorphic:
            if voff:
                lines.append(f"char _pad0[0x{voff:X}];")
            lines.append(f"virtual ~{short}();")
        elif parent is None:
            lines.append("char _pad0[0x1];")
        if depth < k:
            lines.append(f"{short}() {{}}")                      # an inlined base constructor
        elif depth == k and base_call is not None:
            params = [t.text for t in base_call.cxx.params] if base_call.cxx is not None else ["int"] * (base_call.n_gpr - 1)
            lines.append(f"{short}({', '.join(params)});")
        model_classes.append((cls, parent, lines))
    for mcls, lines in member_classes.items():
        model_classes.append((mcls, None, [f"char _pad0[0x{sizes.get(mcls, 1):X}];"] + lines))
    parent = chain[1] if len(chain) > 1 else None
    model_classes.append((cname, parent, []))
    start = (voff + 4 if polymorphic else 1) if parent else 0
    model = CtorModel(cname, voff if polymorphic and parent is None else None, start, objects, model_classes,
                      base, base_call, [(o, x) for o, x in zip(offs, inits)])
    # the body is what the lifted tree keeps; initializers and the base call are the model's
    init_ids = {id(x) for x in inits}
    lifted.tree = body
    lifted.stores = [x for x in lifted.stores if id(x) not in init_ids]
    lifted.init_stores = [x for x in inits if isinstance(x, gen.Store)]
    lifted.init_args = list(base_call.args[1:] if base_call is not None else ()) + \
        [a for x in inits if isinstance(x, gen.MemberCtor) for a in x.args]
    lifted.calls = [c for c in lifted.calls if c is not base_call]
    if base_call is not None and gen.is_auto(base_call.callee):
        renames["@base_ctor"] = base_call.callee
    lifted.ret, lifted.ret_fp = None, False
    notes = [f"class chain (root last): {' : '.join(chain)}"]
    if objects:
        notes.append("members constructed: " + ", ".join(f"+0x{o:X} {c}" + (f"[{n}]" if n else "")
                                                         for o, (c, n, _) in sorted(objects.items())))
    if polymorphic and voff:
        notes.append(f"vtable pointer at +0x{voff:X}")
    return CtorPlan(lifted, model, name, renames, advice, notes)


def _ctor_source(chain, k, cname, voff, inits, values, owner, sig, base_params, base_args, calls_base) -> str:
    q_parts = {}
    for name in chain:
        q = name.split("::")
        if len(q) > 2:
            raise gen.Unsupported("classes nested three deep")
        if len(q) == 2:
            q_parts.setdefault(q[0], []).append(q[1])
    out = [f"struct {o} {{ {' '.join(f'struct {x};' for x in inner)} }};" for o, inner in sorted(q_parts.items())]
    fields_of: dict = {}
    body_of: dict = {}
    for st, text, o in zip(inits, values, owner):
        fields_of.setdefault(o, {})[st.loc.offset] = st.ctype
        body_of.setdefault(o, []).append(f"m{st.loc.offset:X} = {text};")
    cur = voff + 4
    for i, name in enumerate(reversed(chain)):
        depth = len(chain) - 1 - i
        parent = chain[depth + 1] if depth + 1 < len(chain) else None
        short = name.split("::")[-1]
        members = []
        if parent is None and voff:
            members.append(f"char _pad0[0x{voff:X}];")
        if parent is None:
            members.append(f"virtual ~{short}();")
        if name == cname:
            members.append(f"{short}({', '.join(t for t, _ in sig)});")
        elif depth < k:                                  # inlined base constructor
            members.append(f"{short}() {{ {' '.join(body_of.get(name, []))} }}")
        elif depth == k and calls_base:                  # the called base constructor
            members.append(f"{short}({', '.join(base_params)});")
        fields = fields_of.get(name, {})
        members += _fields(fields, cur) if fields else []
        if fields:
            last = max(fields)
            cur = last + gen.SIZES.get(fields[last], 4)
            cur = (cur + 3) & ~3
        head = f"struct {name}" + (f" : {parent}" if parent else "")
        out.append(head + " { " + " ".join(members) + " };")
    short = cname.split("::")[-1]
    init = f" : {chain[k]}({', '.join(base_args)})" if calls_base and base_args else ""
    body = " ".join(body_of.get(cname, []))
    params = ", ".join(f"{t} {n}".rstrip() for t, n in sig)
    out.append(f"{cname}::{short}({params}){init} {{ {body} }}".replace("{  }", "{}"))
    return "\n".join(out) + "\n"
