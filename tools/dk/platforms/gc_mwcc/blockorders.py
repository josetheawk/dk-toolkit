"""Per-block declaration order: write a wanted order of named locals into the blocks that declare them.

The order solver's declaration-order lever (`oracle.wanted_orders`) gives the locals to declare first-to-last,
each with the source line of its value. Names may repeat (`i` in two blocks) and `for (int i = ...)` counts
as a declaration, so every entry is mapped to the innermost block that declares that name around its line.
Per block with at least two wanted items, three kinds of variant:

1. the wanted items declared bare at the block's top in the wanted order (initialisers stay behind as
   assignments, a `for` keeps `i = 0`);
2. the declaration statements in the wanted order at the first one's place, each keeping its initialiser
   when that is legal (a pure initialiser, nothing in between reads or writes what it touches), else split;
3. single moves: one item declared before another (bare), a declaration with a pure initialiser moved
   after a later one.
And every block at once (each block's kind 1 or 2 variant), first, when two or more blocks change.

Each yields (kind, label, new source); nothing here compiles."""

from __future__ import annotations

import itertools
import re

from ...core import cbody
from . import rewrites


def _emit(site, edits) -> str | None:
    if not edits:
        return None
    try:
        new = cbody.apply_edits(site.src, edits)
    except ValueError:
        return None
    return new if new != site.src else None


def items(site, blk) -> list[tuple]:
    """A block's declaration items: (kind, stmt, decl, declarator), kind `decl` for a one-declarator
    declaration and `for` for a for-statement whose init declares one variable. Static, array, reference,
    const-initialised and constructor-initialised declarations are left out (moving them is not neutral)."""
    out = []
    for s in blk.kids:
        if s.kind == "decl" and s.decl is not None and len(s.decl.declarators) == 1:
            typ = site.src[s.decl.type_start:s.decl.type_end]
            d = s.decl.declarators[0]
            if "static" in typ or d.array or "&" in d.ptr or (d.init is not None and d.init_kind != "="):
                continue
            if d.init is not None and "const" in typ.split():
                continue
            out.append(("decl", s, s.decl, d))
        elif s.kind == "for" and s.init is not None and s.init.decl is not None and len(s.init.decl.declarators) == 1:
            d = s.init.decl.declarators[0]
            if d.init is not None and d.init_kind == "=" and not d.array:
                out.append(("for", s, s.init.decl, d))
    return out


def _bare(site, decl, d) -> str:
    return site.src[decl.type_start:d.name_end].strip() + ";"


def _split_edit(site, kind, s, decl, d):
    """The edit that leaves an item's initialiser behind (or removes a bare declaration)."""
    if kind == "for":
        return (decl.type_start, d.name_start, "")
    if d.init is not None:
        return (s.start, s.end, f"{d.name} = {site.text(d.init)};")
    blk, k = site.parents[id(s)]
    nxt = blk.kids[k + 1].start if k + 1 < len(blk.kids) else s.end
    return (s.start, max(nxt, s.end), "")


def _uses_between(site, name: str, a: int, b: int) -> bool:
    return re.search(r"\b%s\b" % re.escape(name), site.src[a:b]) is not None


def block_orders(site, wanted: list[tuple[str, int | None]], names: set[str] | frozenset = frozenset()) -> list[tuple[str, str, str]]:
    """(kind, label, source) variants that declare `wanted` [(name, 1-based line)...] in that order, block by
    block; `names` limits the single moves to pairs that involve one of them (empty: all pairs)."""
    blocks = [b for b in cbody.statements(site.body) if b.kind == "block"]
    out: list[tuple[str, str, str]] = []
    per_block, per_block_split = [], []
    for blk in blocks:
        its = items(site, blk)
        if len(its) < 2:
            continue
        lo, hi = site.line(blk.start) + 1, site.line(blk.end) + 1
        inner = [b for b in blocks if b is not blk and blk.start < b.start and b.end <= blk.end]
        by_name: dict[str, list] = {}
        for it in its:
            by_name.setdefault(it[3].name, []).append(it)
        order = []
        for n, ln in wanted:
            if n not in by_name or ln is None or not (lo <= ln <= hi):
                continue
            if any(site.line(b.start) + 1 <= ln <= site.line(b.end) + 1 and any(x[3].name == n for x in items(site, b))
                   for b in inner):
                continue                          # a deeper block declares this name around that line
            before = [x for x in by_name[n] if site.line(x[1].start) + 1 <= ln]   # the declaration in force at that line
            it = before[-1] if before else by_name[n][0]
            if it not in order:
                order.append(it)
        if len(order) >= 2:
            cur = [it for it in its if it in order]
            if cur != order:
                first = blk.kids[0]
                ind = cbody.indent_at(site.src, first.start)
                split = [(first.start, first.start, "".join(_bare(site, it[2], it[3]) + "\n" + ind for it in order))]
                split += [_split_edit(site, *it) for it in order]
                new = _emit(site, split)
                if new:
                    out.append(("block-decl", f"block@{lo}: declare {', '.join(it[3].name for it in order)} at the top (split)", new))
                anchor = cur[0][1]
                ind = cbody.indent_at(site.src, anchor.start)
                txt, edits = [], []
                for it in order:
                    kind, st, decl, d = it
                    keep = kind == "decl"
                    if keep and d.init is not None:
                        a, b = sorted((anchor.start, st.start))
                        refs = [d.name] + cbody.names(d.init)
                        keep = rewrites.pure(d.init) and not any(_uses_between(site, x, a, b) for x in refs if x)
                    if keep:
                        txt.append(site.src[st.start:st.end])
                        if st is not anchor:
                            blk_, k_ = site.parents[id(st)]
                            nxt = blk_.kids[k_ + 1].start if k_ + 1 < len(blk_.kids) else st.end
                            edits.append((st.start, max(nxt, st.end), ""))
                        else:
                            edits.append((st.start, st.end, ""))
                    else:
                        txt.append(_bare(site, decl, d))
                        e = _split_edit(site, *it)
                        if st is anchor:
                            e = (e[0], min(e[1], st.end), e[2])
                        edits.append(e)
                edits.append((anchor.start, anchor.start, ("\n" + ind).join(txt) + "\n" + ind))
                new = _emit(site, edits)
                if new:
                    out.append(("block-decl", f"block@{lo}: declarations in order {', '.join(it[3].name for it in order)} "
                                              f"(initialisers kept where legal)", new))
                    per_block.append(edits)
                    per_block_split.append(split)
        for j in range(len(its)):
            for k in range(j + 1, len(its)):
                a_it, b_it = its[j], its[k]
                if names and a_it[3].name not in names and b_it[3].name not in names:
                    continue
                ind = cbody.indent_at(site.src, a_it[1].start)
                new = _emit(site, [(a_it[1].start, a_it[1].start, _bare(site, b_it[2], b_it[3]) + "\n" + ind),
                                   _split_edit(site, *b_it)])
                if new:
                    out.append(("block-decl", f"{b_it[3].name} declared before {a_it[3].name} (block@{lo})", new))
                kind, s, decl, d = a_it
                if kind == "decl" and (d.init is None or rewrites.pure(d.init)) and b_it[0] == "decl":
                    between = site.src[s.end:b_it[1].end]
                    refs = [d.name] + (cbody.names(d.init) if d.init is not None else [])
                    if not any(re.search(r"\b%s\b" % re.escape(x), between) for x in refs if x):
                        new = _emit(site, [(s.start, s.end, ""),
                                           (b_it[1].end, b_it[1].end, "\n" + cbody.indent_at(site.src, b_it[1].start)
                                            + site.src[s.start:s.end])])
                        if new:
                            out.append(("block-decl", f"{d.name} declared after {b_it[3].name} (block@{lo})", new))
    if len(per_block) >= 2:
        for combo in itertools.islice(itertools.product(*[(x, y) for x, y in zip(per_block, per_block_split)]), 8):
            new = _emit(site, [e for es in combo for e in es])
            if new:
                out.insert(0, ("block-decl", "every block in the solver's declaration order", new))
    return out
