"""The rank solver: which declaration order (and which temporary merges) the register-allocation
rule says produce the target's registers.

The rule (playbook, gamecube-mwcc.md "Register allocation"): values get registers one at a time in
rank order: compiler temporaries newest first, then named locals in declaration order, then
parameters last first; a local that merges with a temporary takes its rank. A callee-saved value
takes r31 (f31) down. In volatile registers named locals were seen taking ascending registers in
declaration order after the temporaries (toy compiles; the solver tries that order and `dk close`
reports when it closes a function).

For each register bank (callee-saved or volatile, GPR or FPR) the solver
1. fits the tiers to our registers: every named local's allocation position, the longest run that
   follows declaration order (plain locals), the rest ranked as merged with a temporary;
2. fits them to the target's registers (objdiff's alignment gives each of our registers its
   target register) the same way;
3. reads off the declaration order that sorts the locals by their target position, the locals
   whose merge differs between the two, and what the rule does not explain (the missing rule).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from ...core import cbody
from . import rewrites

BANKS = (("r", True), ("f", True), ("r", False), ("f", False))


def saved(reg: str) -> bool:
    return int(reg[1:]) >= 14


def position(reg: str, callee_saved: bool) -> int:
    """Allocation order within a bank: r31 first for callee-saved, the lowest first for volatile."""
    n = int(reg[1:])
    return 31 - n if callee_saved else n


@dataclass
class Local:
    name: str
    decl: int                  # declaration index (debug info order)
    ours: str
    target: str | None         # None: ours maps to several target registers


@dataclass
class Fit:
    cls: str
    callee_saved: bool
    locals: list[Local] = field(default_factory=list)
    merged_ours: list[str] = field(default_factory=list)
    merged_target: list[str] = field(default_factory=list)
    order: list[str] = field(default_factory=list)          # declaration order the target's registers want
    current: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def bank(self) -> str:
        return f"{'callee-saved' if self.callee_saved else 'volatile'} {'GPR' if self.cls == 'r' else 'FPR'}s"


def longest_run(seq: list[int]) -> set[int]:
    """Indexes of a longest strictly increasing subsequence (the earliest one on ties)."""
    if not seq:
        return set()
    best_len = [1] * len(seq)
    prev = [-1] * len(seq)
    for i in range(len(seq)):
        for j in range(i):
            if seq[j] < seq[i] and best_len[j] + 1 > best_len[i]:
                best_len[i], prev[i] = best_len[j] + 1, j
    i = max(range(len(seq)), key=lambda k: (best_len[k], -k))
    out = set()
    while i >= 0:
        out.add(i)
        i = prev[i]
    return out


def fit(dwarf, pairs: dict, cls: str, callee_saved: bool) -> Fit | None:
    """The bank's named locals with their positions ours and target, and what the rule reads."""
    f = Fit(cls, callee_saved)
    if dwarf is None:
        return None
    decl = [v for v in dwarf.vars if v.kind == "local"]
    names = [v.name for v in decl]
    in_bank = lambda r: bool(r) and r[0] == cls and r[1:].isdigit() and saved(r) == callee_saved and r not in ("r1", "r2", "r13")
    for k, v in enumerate(decl):
        if not in_bank(v.reg) or names.count(v.name) != 1 or v.reg not in pairs:
            continue
        top = pairs[v.reg].most_common()
        target = top[0][0] if len(top) == 1 or top[0][1] > 2 * top[1][1] else None
        f.locals.append(Local(v.name, k, v.reg, target if target and in_bank(target) else None))
    if len(f.locals) < 1:
        return f
    f.current = [x.name for x in f.locals]
    ours = [position(x.ours, callee_saved) for x in f.locals]
    plain = longest_run(ours)
    f.merged_ours = [x.name for i, x in enumerate(f.locals) if i not in plain]
    known = [x for x in f.locals if x.target is not None]
    tpos = [position(x.target, callee_saved) for x in known]
    tplain = longest_run(tpos)
    f.merged_target = [x.name for i, x in enumerate(known) if i not in tplain]
    f.order = [x.name for x in sorted(known, key=lambda x: (position(x.target, callee_saved), x.decl))]
    f.order += [x.name for x in f.locals if x.target is None]
    temps = sorted((r for r in pairs if in_bank(r) and not any(v.reg == r for v in dwarf.vars)),
                   key=lambda r: position(r, callee_saved))
    if all(x.target == x.ours for x in known) and all(pairs[r].most_common(1)[0][0] == r for r in temps):
        f.order, f.merged_ours, f.merged_target = list(f.current), [], []
        return f                                   # this bank already matches
    # what the rule explains, in words
    if [x.name for x in known] != f.order[:len(known)]:
        f.notes.append("declare in this order (first gets " + ("the highest" if callee_saved else "the lowest")
                       + " register): " + ", ".join(f"{x} ({next(l.target for l in known if l.name == x)})" for x in f.order[:len(known)]))
    for n in f.merged_ours:
        if n not in f.merged_target:
            f.notes.append(f"{n} ranks as a temporary in ours but not in the target: its value feeds an operation straight "
                           "from the call or expression that made it; assign it on its own line, or pass it on first")
    for n in f.merged_target:
        if n not in f.merged_ours:
            f.notes.append(f"{n} ranks as a temporary in the target: there its value feeds an operation straight from the "
                           "call or expression that made it (or it is declared first)")
    for r in temps:
        t = pairs[r].most_common(1)[0][0]
        if t != r and in_bank(t):
            below = [x.name for x in known if position(x.target, callee_saved) < position(t, callee_saved)]
            if below:
                f.notes.append(f"missing rule: the temporary in {r} sits after named locals in the target ({t}, after "
                               f"{', '.join(below)}): there it is not a temporary, or the locals rank above it")
    ours_regs = {x.ours for x in f.locals} | set(temps)
    target_regs = {t for r in ours_regs for t in pairs.get(r, {}) if in_bank(t)}
    if len(target_regs) > len(ours_regs):
        f.notes.append(f"missing rule: the target uses {len(target_regs)} registers here, ours {len(ours_regs)}: it keeps a "
                       "value ours folds or recomputes")
    return f


def _target(pairs: dict, reg: str) -> str | None:
    top = pairs.get(reg, {}) and pairs[reg].most_common()
    if not top:
        return None
    return top[0][0] if len(top) == 1 or top[0][1] > 2 * top[1][1] else None


@dataclass
class Below:
    name: str | None           # the local, or None for a temporary
    ours: str
    target: str
    param: str                 # the lowest parameter in the target
    param_reg: str


def below_parameters(dwarf, pairs: dict) -> list[Below]:
    """Callee-saved values the target ranks below every parameter (a register numbered under all
    of theirs, so handed out after them) while ours ranks them above one. The rule ranks locals
    and temporaries above parameters, except inside an inlined helper: there its locals and
    temporaries rank below the caller's parameters. So each one was a helper's local."""
    if dwarf is None:
        return []
    out = []
    for cls in ("r", "f"):
        bank = lambda r: bool(r) and r[0] == cls and r[1:].isdigit() and saved(r) and r not in ("r1", "r2", "r13")
        entry = getattr(pairs, "entry", {})
        params = [(v, entry.get(v.reg) or _target(pairs, v.reg)) for v in dwarf.vars
                  if v.kind == "param" and bank(v.reg) and v.reg in pairs]
        params = [(v, t) for v, t in params if t is not None and bank(t)]
        if not params:
            continue
        low_v, low_t = min(params, key=lambda vt: int(vt[1][1:]))
        low_ours = min(int(v.reg[1:]) for v, _ in params)
        held = {v.reg for v in dwarf.vars if v.kind == "param"}
        named = {}
        for v in dwarf.vars:
            if v.kind == "local" and bank(v.reg) and v.reg not in held:
                named.setdefault(v.reg, v.name)
        for reg in sorted((r for r in pairs if bank(r) and r not in held), key=lambda r: -int(r[1:])):
            t = pairs[reg].most_common(1)[0][0] if pairs[reg] else None     # a reused register maps to several
            if t is None or not bank(t) or int(t[1:]) >= int(low_t[1:]) or int(reg[1:]) <= low_ours:
                continue
            if any(v.reg == reg for v in dwarf.vars if v.kind == "local") and reg not in named:
                continue
            out.append(Below(named.get(reg), reg, t, low_v.name, low_t))
    return out


def inline_notes(dwarf, pairs: dict) -> list[str]:
    """The inline-helper rule's diagnosis, in words."""
    found = below_parameters(dwarf, pairs)
    out = []
    for b in (x for x in found if x.name):
        out.append(f"inline-helper rule: {b.name} ({b.ours}) ranks below every parameter in the target ({b.target}, "
                   f"under {b.param} in {b.param_reg}); inside an inlined helper the locals rank below the caller's "
                   f"parameters, so the loop or block using {b.name} was a `static inline` helper (`dk close` moves it)")
    temps = [x for x in found if not x.name]
    if temps:
        out.append("inline-helper rule: the temporaries in " + ", ".join(f"{x.ours} (target {x.target})" for x in temps)
                   + f" rank below every parameter in the target (under {temps[0].param} in {temps[0].param_reg}): "
                   "there they are an inlined helper's locals. If ours are an inline's already, its shape differs: "
                   "which block it holds, what it takes, or a caller local declared before the call")
    return out


def solve(dwarf, pairs: dict) -> list[Fit]:
    return [x for x in (fit(dwarf, pairs, cls, cs) for cls, cs in BANKS) if x is not None and x.locals]


def report(fits: list[Fit]) -> list[str]:
    out = []
    for f in fits:
        if f.notes:
            out.append(f"{f.bank}:")
            out += [f"  {n}" for n in f.notes]
    return out


# ---- candidates for dk close ---------------------------------------------------------------------

def single_decls(site: rewrites.Site) -> dict[str, cbody.Stmt]:
    found: dict[str, list[cbody.Stmt]] = {}
    for s in cbody.statements(site.body):
        if s.kind == "decl" and s.decl is not None:
            for d in s.decl.declarators:
                found.setdefault(d.name, []).append(s)
    return {n: v[0] for n, v in found.items() if len(v) == 1 and len(v[0].decl.declarators) == 1 and site.in_block(v[0])
            and "static" not in site.src[v[0].decl.type_start:v[0].decl.type_end]}


def reorder_edits(site: rewrites.Site, order: list[str], hoist: bool = False) -> list[tuple[int, int, str]] | None:
    """Edits that make `order` the declaration order of those locals: minimal moves (each local that
    must come earlier goes just before the one it has to precede), or all of them hoisted to the
    top of the function. An initializer stays behind as an assignment."""
    decls = single_decls(site)
    if any(n not in decls for n in order) or len(order) < 2:
        return None
    pos = {n: decls[n].start for n in order}
    anchor = {n: decls[n] for n in order}
    moved: list[str] = []
    if hoist:
        top = site.body.kids[0] if site.body.kids else None
        if top is None:
            return None
        moved = list(order)
        anchor = {n: top for n in order}
    else:
        for k in range(len(order) - 2, -1, -1):
            x, nxt = order[k], order[k + 1]
            if pos[x] < pos[nxt]:
                continue
            moved.append(x)
            pos[x] = pos[nxt] - 0.5 ** (len(order) - k)
            anchor[x] = anchor[nxt]
    if not moved:
        return None
    edits = []
    by_anchor: dict[int, list[str]] = {}
    for n in order:
        if n not in moved:
            continue
        a = anchor[n]
        block = site.parents[id(a)][0]
        if not (block.start <= decls[n].start and decls[n].end <= block.end):
            return None                       # a declaration only moves out to an enclosing block
        parts = rewrites.decl_parts(site, decls[n])
        if parts is None:
            return None
        by_anchor.setdefault(id(a), []).append(parts[0])
        edits.append(parts[1])
    for n in order:
        a = anchor[n]
        if id(a) in by_anchor:
            ind = cbody.indent_at(site.src, a.start)
            edits.append((a.start, a.start, "".join(f"{h}\n{ind}" for h in by_anchor.pop(id(a)))))
    return edits


def tie_orders(order: list[str], f: Fit, limit: int = 4) -> list[list[str]]:
    """The order, plus permutations of locals the target puts in one register (no order between them)."""
    target = {x.name: x.target for x in f.locals}
    groups: list[list[str]] = []
    for n in order:
        if groups and target.get(groups[-1][0]) is not None and target.get(groups[-1][0]) == target.get(n):
            groups[-1].append(n)
        else:
            groups.append([n])
    out = [order]
    for gi, g in enumerate(groups):
        if len(g) < 2:
            continue
        for perm in itertools.permutations(g):
            cand = [x for h in groups[:gi] for x in h] + list(perm) + [x for h in groups[gi + 1:] for x in h]
            if cand not in out:
                out.append(cand)
            if len(out) >= limit:
                return out
    return out


def solver_candidates(site: rewrites.Site, dwarf, pairs: dict):
    """(label, source) the solver proposes: each bank's target declaration order (minimal moves, then
    hoisted, ties permuted), every bank's order at once, and a local the target ranks as a temporary
    declared first."""
    fits = solve(dwarf, pairs)
    seen = {site.src}

    def emit(label, edits):
        if edits:
            try:
                new = cbody.apply_edits(site.src, edits)
            except ValueError:
                return None
            if new not in seen:
                seen.add(new)
                return label, new
        return None

    for f in fits:
        if f.order == f.current or len(f.order) < 2:
            continue
        for order in tie_orders(f.order, f):
            for hoist in (False, True):
                got = emit(f"{f.bank} in target order: {', '.join(order)}{' (hoisted)' if hoist else ''}",
                           reorder_edits(site, order, hoist))
                if got:
                    yield got
    decl_order = [v.name for v in (dwarf.vars if dwarf else []) if v.kind == "local"]
    for f in fits:
        for n in f.merged_target:
            if n in f.merged_ours or n not in decl_order:
                continue
            rest = [x for x in decl_order if x != n and x in single_decls(site)]
            got = emit(f"{n} declared first (the target ranks it as a temporary)", reorder_edits(site, [n] + rest[:1]))
            if got:
                yield got
