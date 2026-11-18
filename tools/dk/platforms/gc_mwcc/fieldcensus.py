"""dk analyze fields (step 0): which struct fields each function reads and writes through its arguments, and
which class each argument is, from the original asm alone (no compiles).

Facts: every load/store whose address is an argument register at entry plus a displacement, with its width,
kind (int/float/double) and sign (only lha/lhax/lhau, or extsb/extsh of the loaded register right after; never a
bare lbz: u8, bool and char load alike). Roots, i.e. the class of an argument register: the demangled name (`this`
in r3 for a method, class-pointer parameters by position), vtable slots (build/dk/rtti.json: a slot function's r3
is that class, `fn_` slots too), the callee rule (the caller's own argument passed unchanged in r3 to a demangled
method `C::m` is a C*) and one pass of caller agreement (every caller passing a typed value in that register agrees).
Conflicting evidence is reported, never voted. `kind_pairs` counts the near misses where target and ours load the
same displacement with another kind or width (lwz/lfs, stw/stfs, lbz/lha, lhz/lha) or extend it on one side only.
"""

from __future__ import annotations

import json
import re

from .asmflow import flow_of

WIDTH = {"b": 1, "h": 2, "w": 4, "d": 8}
_MEM = re.compile(r"^(?P<m>(?:l|st)(?:b|h|w|f[sd])(?:a|z)?(?:u)?)$")
PAIRS = {frozenset(p) for p in (("lwz", "lfs"), ("stw", "stfs"), ("lbz", "lha"), ("lhz", "lha"))}
EXT = ("extsb", "extsh", "clrlwi")


def access(mnem: str) -> tuple[int, str, str | None, bool] | None:
    """(width, kind, sign, store) of a plain load/store mnemonic (update and indexed forms reduced), else None."""
    m = mnem.rstrip(".").removesuffix("x")
    m = m[:-1] if m.endswith("u") and m not in ("lu",) else m
    if not _MEM.match(m) or m.startswith(("lmw", "stmw")):
        return None
    store = m.startswith("st")
    core = m[2:] if store else m[1:]
    if core in ("fs", "fd"):
        return (4 if core == "fs" else 8), ("float" if core == "fs" else "double"), None, store
    w = WIDTH.get(core[0])
    if w is None:
        return None
    return w, "int", ("signed" if core.endswith("a") else None), store


def facts(name: str, text: str) -> tuple[list[dict], list[dict]]:
    """(field accesses through entry argument registers, calls with their argument values)."""
    flow = flow_of(name, text)
    out, calls = [], []
    nodes = flow.nodes
    for n in nodes:
        if n.kind == "call":
            target = n.insn.ops[0] if n.insn.ops else ""
            args = {r: v for r, v in (n.regs or {}).items()
                    if isinstance(v, tuple) and v and isinstance(v[0], tuple) and v[0][:1] == ("arg",)}
            calls.append({"target": target, "args": {r: [v[0][1], v[1]] for r, v in sorted(args.items())}})
            continue
        if n.mem is None:
            continue
        a = access(n.mnem)
        root, off = n.mem
        if a is None or not (isinstance(root, tuple) and root[:1] == ("arg",)) or not str(root[1]).startswith("r"):
            continue
        width, kind, sign, store = a
        if sign is None and not store and kind == "int" and n.i + 1 < len(nodes):
            nxt = nodes[n.i + 1].insn
            dest = n.insn.ops[0] if n.insn.ops else None
            if nxt.mnem in ("extsb", "extsh") and len(nxt.ops) > 1 and nxt.ops[1] == dest:
                sign = "signed"
        out.append({"reg": root[1], "offset": off, "width": width, "kind": kind, "sign": sign, "store": store})
    return out, calls


def demangled_roots(symbol: str) -> dict[str, str]:
    """reg -> class from the mangled name: `this` in r3 for a method, class-pointer parameters by position."""
    from .demangle import demangle
    fn = demangle(symbol)
    if fn is None:
        return {}
    out: dict[str, str] = {}
    gpr = 3
    if fn.quals:
        out["r3"] = "::".join(fn.quals)
        gpr = 4
    for t in fn.params:
        if t.reg != "gpr":
            continue
        if t.classes and t.text.rstrip().endswith(("*", "&")) and t.text.count("*") + t.text.count("&") == 1:
            out[f"r{gpr}"] = "::".join(t.classes[0])
        gpr += 1
    return out


def census(entries: dict, slots: dict[str, str]) -> dict:
    """{function: {facts, calls, roots: {reg: {class, sources}}, conflicts}} over every function entry.
    slots: function name -> class whose vtable holds it (primary vtables)."""
    from .demangle import demangle
    rows: dict[str, dict] = {}
    for name in sorted(entries):
        e = entries[name]
        if getattr(e, "kind", "fn") != "fn":
            continue
        try:
            fs, calls = facts(name, e.text)
        except Exception:                                    # an unparsable body: no facts, never a guess
            fs, calls = [], []
        roots: dict[str, dict] = {}
        for reg, cls in demangled_roots(name).items():
            roots.setdefault(reg, {"classes": {}, "sources": []})["classes"].setdefault(cls, []).append("name")
        if name in slots:
            roots.setdefault("r3", {"classes": {}, "sources": []})["classes"].setdefault(slots[name], []).append("vtable")
        for c in calls:                                      # the callee rule
            fn = demangle(c["target"])
            arg = c["args"].get("r3")
            if fn is not None and fn.quals and arg and arg[1] == 0:
                roots.setdefault(arg[0], {"classes": {}, "sources": []})["classes"] \
                    .setdefault("::".join(fn.quals), []).append("callee")
        rows[name] = {"facts": fs, "calls": calls, "roots": roots}
    votes: dict[str, dict[str, set]] = {}                    # caller agreement, one pass
    for name, r in rows.items():
        for c in r["calls"]:
            if c["target"] not in rows:
                continue
            for reg, (src, off) in c["args"].items():
                known = r["roots"].get(src)
                if off == 0 and known and len(known["classes"]) == 1:
                    votes.setdefault(c["target"], {}).setdefault(reg, set()).add(next(iter(known["classes"])))
    for callee, regs in votes.items():
        for reg, classes in regs.items():
            if len(classes) == 1 and reg not in rows[callee]["roots"]:
                rows[callee]["roots"][reg] = {"classes": {next(iter(classes)): ["callers"]}, "sources": []}
    for r in rows.values():
        r["conflicts"] = sorted(reg for reg, v in r["roots"].items() if len(v["classes"]) > 1)
        for v in r["roots"].values():
            v["sources"] = sorted({s for ss in v["classes"].values() for s in ss})
            v["classes"] = {k: sorted(set(s)) for k, s in sorted(v["classes"].items())}
    return rows


def typed_args(row: dict) -> list[str]:
    """Argument registers with exactly one class."""
    return sorted(reg for reg, v in row["roots"].items() if len(v["classes"]) == 1)


def _mem(text: str) -> tuple[str, str] | None:
    parts = text.split(None, 1)
    if len(parts) < 2:
        return None
    m = re.search(r",\s*(-?(?:0x[0-9a-fA-F]+|\d+))\((r\d+)\)$", parts[1])
    return (parts[0], m.group(1)) if m else None


def kind_pairs(left: list[dict], right: list[dict], window: int = 3) -> list[str]:
    """Rows where target and ours touch the same displacement with another kind/width pair (ours searched up to
    `window` rows away among flagged rows, since objdiff may align a replaced load as a delete plus an insert), or
    extend a load of the same displacement on one side only."""
    def text(row):
        ins = row.get("instruction")
        return ins.get("formatted") if ins else None
    hits, used = [], set()
    for k, t in enumerate(left):
        a = text(t)
        ma = _mem(a) if a else None
        if not ma:
            continue
        b = text(right[k]) if k < len(right) else None
        mb = _mem(b) if b else None
        if mb and ma == mb and ma[0] in ("lbz", "lhz", "lha") and k + 1 < len(left):
            na, nb = text(left[k + 1]) or "", text(right[k + 1]) or ""
            ea, eb = na.split(" ")[0] in EXT, nb.split(" ")[0] in EXT
            if ea != eb:
                hits.append(f"{ma[0]}+{(na if ea else nb).split(' ')[0]} one side @{ma[1]}")
            continue
        for j in sorted(range(max(0, k - window), min(len(right), k + window + 1)), key=lambda j: abs(j - k)):
            o = right[j]
            if j in used or not (t.get("diff_kind") or o.get("diff_kind")):
                continue
            mo = _mem(text(o) or "")
            if mo and mo[1] == ma[1] and frozenset((ma[0], mo[0])) in PAIRS:
                hits.append(f"{ma[0]}<->{mo[0]} @{ma[1]}")
                used.add(j)
                break
    return hits


def rtti_slots(doc: list[dict], names_by_addr: dict[int, str]) -> dict[str, str]:
    """function -> class from primary vtable slots; a slot several classes share (an inherited virtual) goes to
    the one that is a base of all the others, else it is left out (a conflict, not a vote)."""
    by_rtti = {c["rtti"]: c for c in doc}

    def ancestors(c: dict) -> set[str]:
        out, todo = set(), [b[0] for b in c.get("bases", [])]
        while todo:
            b = by_rtti.get(todo.pop())
            if b and b["name"] not in out:
                out.add(b["name"])
                todo += [x[0] for x in b.get("bases", [])]
        return out

    anc = {c["name"]: ancestors(c) for c in doc}
    seen: dict[str, set[str]] = {}
    for c in doc:
        for v in c.get("vtables", []):
            if v.get("offset") == 0:
                for a in v.get("slots", []):
                    if names_by_addr.get(a):
                        seen.setdefault(names_by_addr[a], set()).add(c["name"])
    out: dict[str, str] = {}
    for fn, classes in sorted(seen.items()):
        base = [c for c in sorted(classes) if all(c == o or c in anc.get(o, set()) for o in classes)]
        if len(base) == 1:
            out[fn] = base[0]
    return out


def counts(rows: dict, route_rows: list[dict], diffs: dict) -> dict:
    """C1 (typing reach on the fn_ fresh functions) and C2 (kind-pair near misses), giants apart."""
    fresh = [r for r in route_rows if r["class"] == "fresh" and r["function"].startswith("fn_")]
    typed = [r["function"] for r in fresh if r["function"] in rows and typed_args(rows[r["function"]])]
    c2: dict[str, list] = {}
    for r in route_rows:
        if r["class"] not in ("near-shape", "far", "giant"):
            continue
        pair = diffs.get(r["unit"], ({}, []))[0].get(r["function"])
        hits = kind_pairs(*pair) if pair else []
        if hits:
            c2[f"{r['unit']}::{r['function']}"] = {"class": r["class"], "hits": hits}
    pool = [r for r in route_rows if r["class"] in ("near-shape", "far")]
    return {"c1_fresh_fn": len(fresh), "c1_typed": len(typed), "c1_share": round(len(typed) / len(fresh), 4) if fresh else 0.0,
            "c1_functions": typed, "c2_pool": len(pool), "c2_hits": sum(1 for v in c2.values() if v["class"] != "giant"),
            "c2_giants": sum(1 for v in c2.values() if v["class"] == "giant"), "c2_rows": c2,
            "functions": len(rows), "with_facts": sum(1 for r in rows.values() if r["facts"]),
            "typed_any": sum(1 for r in rows.values() if typed_args(r)),
            "conflicts": sum(1 for r in rows.values() if r["conflicts"]),
            "sources": {s: sum(1 for r in rows.values() for v in r["roots"].values() if len(v["classes"]) == 1 and s in v["sources"])
                        for s in ("name", "vtable", "callee", "callers")}}


def run(cfg, args) -> int:
    """dk analyze fields: the census into build/dk/fields.json and the step-0 counts."""
    from . import route
    from .dtk_symbols import read_symbols
    from .llmtry import asm_entries
    from .matching import load_objdiff_units
    entries = asm_entries(cfg)
    names = {s.address: s.name for s in read_symbols(cfg.root / cfg.platform_options["symbols"])}
    try:
        doc = json.loads((cfg.root / "build" / "dk" / "rtti.json").read_text())
    except (OSError, ValueError):
        print("note: build/dk/rtti.json missing (./dk analyze rtti): no vtable roots")
        doc = []
    rows = census(entries, rtti_slots(doc, names))
    out = cfg.root / "build" / "dk" / "fields.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, sort_keys=True) + "\n")
    rrows = route.table(cfg) or []
    wanted = {r["unit"] for r in rrows if r["class"] in ("near-shape", "far", "giant")}
    diffs = {u["name"]: route.unit_diff(cfg, u) for u in load_objdiff_units(cfg) if u["name"] in wanted and u.get("base_path")}
    c = counts(rows, rrows, diffs)
    if args.json:
        print(json.dumps(c, indent=1, sort_keys=True))
    else:
        print(f"{c['functions']} function(s), {c['with_facts']} with argument field accesses, {c['typed_any']} with a typed "
              f"argument ({c['conflicts']} with conflicting evidence); roots by source: {c['sources']}")
        print(f"C1: {c['c1_typed']}/{c['c1_fresh_fn']} fn_ fresh functions have a typed argument ({100 * c['c1_share']:.1f} %)")
        print(f"C2: {c['c2_hits']}/{c['c2_pool']} near-shape/far functions show a kind/width pair or one-sided extension "
              f"(giants: {c['c2_giants']})")
        print(f"wrote {cfg.rel(out)}")
    return 0
