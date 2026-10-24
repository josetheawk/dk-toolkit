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
    lines, skeletons = [], {}
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
    base_text, base_params = "", []
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
