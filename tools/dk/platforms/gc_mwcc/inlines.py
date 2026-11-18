"""Find inlined copies of known functions inside a big function, on value-numbered code.

An inlined copy is not a verbatim stretch: the compiler reschedules and interleaves it with its
neighbours, renames registers, puts `this` at a stack address (every displacement shifts), shares
constants, and folds away the null checks of a `this` it can see and branches on arguments it
knows. So a template (a function's code, frame stripped) is matched node by node on values:

1. Seed: pair one of the template's rarest anchors (calls, symbol references; the top 3, since
   one may be folded away) with an occurrence in the target.
2. Grow greedily inside a window of about 2n+16 instructions: nodes whose inputs are already
   known are paired first (their images must be equal); otherwise a node may bind a parameter
   register (or the template's stack) to a target value, one offset per base (this = sp+0x26c).
   Memory operations on one address, and calls, keep their relative order.
3. Score: matched weight / unfolded weight. Folded: null checks of a `this` bound to the stack
   or a global, compares on unbound parameters or constants, the ops feeding them and the
   branches on them; code those branches control is optional (counted only when found).

Overlapping matches: the most matched weight wins.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from .asmflow import ARG_FPRS, ARG_GPRS, ARG_REGS, COMMUTATIVE, K, MASK, SP, Flow, Value, consumers, frame_closure, is_reg
from .demangle import demangle

FIXED = frozenset({"k", "sym", "ha", "lit", "str", "tok"})
PARAMS = frozenset(ARG_REGS)
WEIGHTED = ("op", "call", "branch")
MIN_WEIGHT = 6           # smaller templates (getters, one store) match anywhere
MAX_WEIGHT = 100         # bigger ones are look-alike code, not inlined helpers
MIN_CONF = 0.75          # matched / unfolded weight
MIN_COVER = 0.5          # matched / the template's whole weight: a part of a big function is not a copy of it
MIN_CALLS = 0.6          # share of the template's live calls found: calls are the strongest evidence


def _bindable(root: tuple) -> bool:
    k = root[0]
    return k == "sp" or k == "phi" or (k == "arg" and root[1] in PARAMS)


def _arith(node) -> bool:
    """A register-only operation (no memory, no call): extsh, rlwinm, add, ..."""
    return node.kind == "op" and node.mem is None and not node.store and not _cmp_like(node.key) \
        and not node.key.startswith(("l", "st", "psq", "mt", "mf", "bctr", "cr"))


def _cmp_like(key: str) -> bool:
    return key.startswith(("cmp", "fcmp")) or key == "mr."


# ---------------------------------------------------------------- prepared functions

@dataclass
class Prepared:
    """A flow plus the indexes the matcher needs (built once per function)."""
    flow: Flow
    frame: set[int]
    weighted: list[int]                       # op/call/branch nodes outside the frame
    pure: list[int]                           # pure nodes outside the frame
    feeds: dict[int, set[int]]                # pure node -> weighted nodes it feeds (through copies/pure)
    feeders: dict[int, set[int]]              # weighted node -> pure/copy nodes feeding it
    by_key: dict[str, list[int]]
    anchors: dict[tuple, list[int]]           # anchor key -> weighted nodes carrying it
    weak: dict[int, frozenset[str]]           # call -> argument registers that may be incidental
    by_sig: dict[tuple, list[int]] = field(default_factory=dict)   # (key, inputs) -> nodes
    static_weight: int = 0


def anchor_keys(node) -> list[tuple]:
    if node.kind == "call" and node.key != "bctrl" and node.key != "blrl":
        return [(node.key,)]
    out = []
    for v in node.ins:
        if v[0][0] in ("sym", "lit", "str"):
            out.append((node.key, v[0]))
    return out


def callee_regs(key: str) -> tuple[str, ...] | None:
    """Argument registers of a call whose target's mangled name gives its parameters (member
    functions take this in r3; operator new/delete do not). None when unknown."""
    if not key.startswith("bl "):
        return None
    f = demangle(key[3:])
    if f is None:
        return None
    gpr, fpr = list(ARG_GPRS), list(ARG_FPRS)
    out = []
    if f.quals and f.name not in ("operator new", "operator delete", "operator new[]", "operator delete[]"):
        out.append(gpr.pop(0))
    for p in f.params:
        pool = fpr if p.reg == "fpr" else gpr if p.reg == "gpr" else None
        if pool is None or not pool:
            break                      # passed in memory, variadic, or out of registers
        out.append(pool.pop(0))
    return tuple(out)


def prepare(flow: Flow) -> Prepared:
    frame = frame_closure(flow)
    for n in flow.nodes:                # a known callee says exactly which registers are arguments
        regs = callee_regs(n.key) if n.kind == "call" and n.regs else None
        if regs is not None:
            n.args = tuple((r, n.regs[r]) for r in regs)
    cons = consumers(flow)
    nodes = flow.nodes
    weighted = [n.i for n in nodes if n.kind in WEIGHTED and n.i not in frame]
    pure = [n.i for n in nodes if n.kind == "pure" and n.i not in frame]
    feeds: dict[int, set[int]] = {}
    for n in reversed(nodes):
        if n.kind in ("pure", "copy") and n.i not in frame:
            s: set[int] = set()
            for c in cons[n.i]:
                if nodes[c].kind in WEIGHTED:
                    s.add(c)
                elif c in feeds:
                    s |= feeds[c]
            feeds[n.i] = s
    feeders: dict[int, set[int]] = {}
    for p, ws in feeds.items():
        for w in ws:
            feeders.setdefault(w, set()).add(p)
    by_key: dict[str, list[int]] = {}
    by_sig: dict[tuple, list[int]] = {}
    anchors: dict[tuple, list[int]] = {}
    for i in weighted:
        n = nodes[i]
        by_key.setdefault(n.key, []).append(i)
        by_sig.setdefault((n.key, n.ins), []).append(i)
        for a in anchor_keys(n):
            anchors.setdefault(a, []).append(i)
    weak: dict[int, frozenset[str]] = {}
    for i in weighted:
        n = nodes[i]
        if n.kind != "call" or not n.args or callee_regs(n.key) is not None:
            continue
        w = set()
        for reg, _ in n.args:
            # the producer of this register also feeds something else: maybe not an argument
            prod = _producer(flow, i, reg)
            if prod is None or any(c != i and prod < c < i for c in cons[prod]):
                w.add(reg)
        weak[i] = frozenset(w)
    p = Prepared(flow, frame, weighted, pure, feeds, feeders, by_key, anchors, weak, by_sig)
    p.static_weight = len(weighted) - sum(1 for i in weighted if nodes[i].kind == "jump") + \
        sum(1 for i in pure if feeds.get(i))
    return p


def _producer(flow: Flow, call: int, reg: str) -> int | None:
    """Last node at or before `call` that wrote `reg` (from the call's deps)."""
    for d in sorted(flow.nodes[call].deps, reverse=True):
        if reg in flow.nodes[d].outs:
            return d
    return None


# ---------------------------------------------------------------- matching

@dataclass
class Match:
    template: str
    seed: tuple[int, int]
    pairs: dict[int, int]
    bind: dict
    matched: int
    total: int
    folded: dict[int, str] = field(default_factory=dict)
    claims: frozenset[int] = frozenset()
    calls: tuple[int, int] = (0, 0)          # (live calls found, live calls)

    @property
    def confidence(self) -> float:
        return self.matched / self.total if self.total else 0.0

    @property
    def span(self) -> tuple[int, int]:
        g = self.pairs.values()
        return min(g), max(g)


class _Matcher:
    def __init__(self, t: Prepared, g: Prepared, lo: int, hi: int):
        self.t, self.g = t, g
        self.tn, self.gn = t.flow.nodes, g.flow.nodes
        self.lo, self.hi = lo, hi
        self.bind: dict = {}
        self.pair: dict[int, int] = {}
        self.rpair: dict[int, int] = {}
        self.journal: list[tuple] = []
        self.soft = 0

    # -- journal
    def _mark(self) -> int:
        return len(self.journal)

    def _undo(self, mark: int) -> None:
        while len(self.journal) > mark:
            kind, a = self.journal.pop()
            if kind == "bind":
                del self.bind[a]
            else:
                g = self.pair.pop(a)
                del self.rpair[g]

    # -- values
    def image(self, v: Value):
        root, off = v
        k = root[0]
        if k in FIXED:
            return v
        if k == "n":
            g = self.pair.get(root[1])
            return (("n", g, root[2]), off) if g is not None else None
        if _bindable(root):
            b = self.bind.get(root)
            if b is None:
                return None
            return (K, (off + b[1]) & MASK) if b[0] == K else (b[0], off + b[1])
        return False            # never matchable (clobbered, lr, junk entry registers)

    def unify(self, vt: Value, vg: Value, strong: bool = True):
        """True: equal under the binding (maybe after binding/pairing). None: unknown and weak."""
        root, off = vt
        k = root[0]
        if k in FIXED:
            return vt == vg
        groot, goff = vg
        if groot[0] == "x":
            return False
        if k == "n":
            g = self.pair.get(root[1])
            if g is not None:
                return groot == ("n", g, root[2]) and goff == off
            if not strong:
                return None
            if groot[0] != "n" or groot[2] != root[2] or goff != off:
                return False
            return self.pair_nodes(root[1], groot[1], pulled=True)
        if _bindable(root):
            b = self.bind.get(root)
            if b is not None:
                if b[0] == K:
                    return groot == K and goff == (off + b[1]) & MASK
                return groot == b[0] and goff == off + b[1]
            if not strong:
                return None
            if k == "sp" and groot != SP:
                return False
            if k == "phi" and groot[0] != "phi":
                return False
            self.bind[root] = (groot, goff - off)
            self.journal.append(("bind", root))
            return True
        return False

    # -- nodes
    def _ordered(self, t: int, g: int) -> bool:
        tn = self.tn[t]
        t_call = tn.kind == "call"
        t_mem = tn.mem
        for tp, gp in self.pair.items():
            if tp == t:
                continue
            p = self.tn[tp]
            if t_call and (p.kind == "call" or p.mem is not None) or \
                    t_mem is not None and (p.kind == "call" or p.mem == t_mem and (tn.store or p.store)):
                if (t < tp) != (g < gp):
                    return False
        return True

    def pair_nodes(self, t: int, g: int, pulled: bool = False, all_strong: bool = False) -> bool:
        if t in self.pair:
            return self.pair[t] == g
        if g in self.rpair:
            return False
        tn, gn = self.tn[t], self.gn[g]
        if tn.key != gn.key or tn.kind != gn.kind or len(tn.ins) != len(gn.ins) or g in self.g.frame:
            return False
        if not pulled and not (self.lo <= g <= self.hi):
            return False
        mark = self._mark()
        self.pair[t] = g
        self.rpair[g] = t
        self.journal.append(("pair", t))
        if not self._inputs(tn, gn, all_strong) or not self._ordered(t, g):
            self._undo(mark)
            return False
        return True

    def _inputs(self, tn, gn, all_strong: bool) -> bool:
        orders = [tn.ins]
        if tn.key in COMMUTATIVE and len(tn.ins) >= 2:
            orders.append((tn.ins[1], tn.ins[0]) + tuple(tn.ins[2:]))
        for ins in orders:
            mark = self._mark()
            if all(self.unify(a, b) for a, b in zip(ins, gn.ins)):
                break
            self._undo(mark)
        else:
            return False
        if tn.kind == "call":
            weak = frozenset() if all_strong else self.t.weak.get(tn.i, frozenset())
            for reg, v in tn.args:
                gv = gn.regs.get(reg) if gn.regs else None
                if gv is None:
                    if reg in weak:
                        self.soft += 1
                        continue
                    return False
                r = self.unify(v, gv, strong=reg not in weak)
                if r is False:
                    if reg not in weak:
                        return False
                    self.soft += 1
        return True

    def candidates(self, t: int) -> list[int]:
        """Unused same-key nodes in the window whose inputs equal every image already known,
        nearest to where the template's neighbours landed first."""
        tn = self.tn[t]
        imgs = [self.image(v) for v in tn.ins]
        if any(x is False for x in imgs):
            return []
        if all(imgs):           # every input known: exact signature lookup
            lst = self.g.by_sig.get((tn.key, tuple(imgs)), [])
            if tn.key in COMMUTATIVE and len(imgs) >= 2:
                sw = (imgs[1], imgs[0]) + tuple(imgs[2:])
                lst = sorted(set(lst) | set(self.g.by_sig.get((tn.key, sw), [])))
        else:
            lst = self.g.by_key.get(tn.key, [])
        a, b = bisect.bisect_left(lst, self.lo), bisect.bisect_right(lst, self.hi)
        known = [(k, im) for k, im in enumerate(imgs) if im is not None]
        swapped = None
        if tn.key in COMMUTATIVE and len(imgs) >= 2:
            sw = [imgs[1], imgs[0]] + imgs[2:]
            swapped = [(k, im) for k, im in enumerate(sw) if im is not None]
        strong = []
        if tn.kind == "call":
            weak = self.t.weak.get(t, frozenset())
            for reg, v in tn.args:
                im = self.image(v)
                if im is False and reg not in weak:
                    return []
                if im and reg not in weak:
                    strong.append((reg, im))
        gn, rpair = self.gn, self.rpair
        out = []
        for g in lst[a:b]:
            if g in rpair:
                continue
            ins = gn[g].ins
            if not all(ins[k] == im for k, im in known) and \
                    (swapped is None or not all(ins[k] == im for k, im in swapped)):
                continue
            if strong:
                regs = gn[g].regs or {}
                if not all(regs.get(r) == im for r, im in strong):
                    continue
            out.append(g)
        if len(out) > 1:
            exp = self.expected(t)
            out.sort(key=lambda g: (abs(g - exp), g))
        return out

    def state(self, t: int) -> str:
        """known: every input has an image; dead: one never can; open: something is unbound."""
        tn = self.tn[t]
        res = "known"
        for v in tn.ins + tuple(v for _, v in tn.args):
            im = self.image(v)
            if im is False:
                return "dead"
            if im is None:
                res = "open"
        return res

    def expected(self, t: int) -> int:
        best = None
        for tp, gp in self.pair.items():
            d = abs(tp - t)
            if best is None or d < best[0] or d == best[0] and tp < best[1]:
                best = (d, tp, gp)
        return best[2] + (t - best[1]) if best else (self.lo + self.hi) // 2

    def try_best(self, t: int) -> bool:
        best = None
        for g in self.candidates(t):
            mark, soft = self._mark(), self.soft
            ok = self.pair_nodes(t, g)
            gained = self.soft - soft
            self.soft = soft
            if ok:
                if gained == 0:
                    self.soft += 0
                    return True
                self._undo(mark)
                if best is None or gained < best[0]:
                    best = (gained, g)
            else:
                self._undo(mark)
        if best is not None:
            soft = self.soft
            if self.pair_nodes(t, best[1]):
                return True
            self.soft = soft
        return False

    def grow(self, floor: int = 0) -> None:
        """Pair what can be paired; stop early once fewer than `floor` nodes could still pair."""
        dead: set[int] = set()
        tried: set[int] = set()
        budget = len(self.t.weighted) - floor
        while True:
            if len(dead) > budget:
                return
            progress = False
            for t in self.t.weighted:
                if t in self.pair or t in dead or self.tn[t].kind == "jump":
                    continue
                st = self.state(t)
                if st == "open":
                    continue
                if st == "dead":
                    dead.add(t)
                    continue
                if self.try_best(t):
                    progress = True
                else:
                    dead.add(t)
            if progress:
                continue
            # nothing determined is left: let one node bind something new (anchors first)
            order = sorted((t for t in self.t.weighted if t not in self.pair and t not in dead and t not in tried
                            and self.tn[t].kind != "jump"),
                           key=lambda t: (not anchor_keys(self.tn[t]), t))
            for t in order:
                tried.add(t)
                if self.try_best(t):
                    progress = True
                    break
            if not progress:
                return


def window(t: Prepared) -> int:
    """Half-width of the search window around a seed: about 2n+16 instructions in all."""
    return len(t.weighted) + len(t.pure) + 8


def match_at(t: Prepared, g: Prepared, seed_t: int, seed_g: int, name: str = "", floor: int = 0) -> Match | None:
    w = window(t)
    m = _Matcher(t, g, max(0, seed_g - w), min(len(g.flow.nodes) - 1, seed_g + w))
    if not m.pair_nodes(seed_t, seed_g, all_strong=True):
        return None
    m.grow(floor)
    return score(m, name, (seed_t, seed_g))


# ---------------------------------------------------------------- folding and score

def score(m: _Matcher, name: str, seed: tuple[int, int]) -> Match:
    t = m.t
    nodes = t.flow.nodes
    folded: dict[int, str] = {}
    memo: dict[int, bool] = {}

    def unbound(v: Value) -> bool:
        """Derived only through arithmetic from a parameter the copy never bound (the compiler
        knew its value). Loads, calls, loop values and the template's stack are not parameters."""
        root = v[0]
        if root[0] == "arg" and root[1] in PARAMS:
            return root not in m.bind
        if root[0] == "n" and root[1] not in m.pair and _arith(nodes[root[1]]):
            j = root[1]
            if j not in memo:
                memo[j] = False
                memo[j] = any(unbound(x) for x in nodes[j].ins)
            return memo[j]
        return False

    def constlike(v: Value) -> bool:
        root = v[0]
        if root[0] in FIXED:
            return True
        if _bindable(root):
            b = m.bind.get(root)
            return b is not None and b[0][0] in ("k", "sym", "lit", "str")
        if root[0] == "n" and root[1] not in m.pair:
            return folded.get(root[1]) == "const"
        return False

    def addresslike(v: Value) -> bool:
        img = m.image(v)
        return bool(img) and img[0][0] in ("sp", "sym", "lit", "str")

    changed = True
    while changed:
        changed = False
        for i in t.weighted:
            if i in m.pair or i in folded:
                continue
            n = nodes[i]
            reason = None
            if n.kind == "op" and _cmp_like(n.key):
                vals = list(n.ins)
                zero = [v for v in vals if v == (K, 0)]
                rest = [v for v in vals if v != (K, 0)]
                if (zero or n.key == "mr.") and len(rest) == 1 and addresslike(rest[0]):
                    reason = "null"
                elif any(unbound(v) for v in vals):
                    reason = "param"
                elif vals and all(constlike(v) for v in vals):
                    reason = "const"
            elif n.kind == "op" and n.mem is None and not n.store and n.key != "bctr" \
                    and not n.key.startswith(("mt", "mf", "st", "l", "psq")):
                if any(unbound(v) for v in n.ins):
                    reason = "param"
                elif n.ins and all(constlike(v) for v in n.ins):
                    reason = "const"
                elif n.key.startswith("cr") and all(v[0][0] == "n" and v[0][1] in folded for v in n.ins) and n.ins:
                    reason = folded[n.ins[0][0][1]]
            elif n.kind == "branch" and n.ins:
                src = n.ins[0][0]
                if src[0] == "n" and src[1] in folded:
                    reason = folded[src[1]]
            if reason:
                folded[i] = reason
                changed = True

    # Code a folded branch controls is folded with it only when none of it was found: the
    # compiler dropped the whole arm (the delete in a destructor called on a local).
    optional: set[int] = set()
    labels = t.flow.labels
    for i, reason in folded.items():
        n = nodes[i]
        if n.kind != "branch" or n.target not in labels:
            continue
        j = labels[n.target]
        if j <= i:
            continue
        arms = [set(range(i + 1, j))]
        last = nodes[j - 1]
        if last.kind == "jump" and last.target in labels and labels[last.target] > j:
            arms.append(set(range(j, labels[last.target])))
        if reason == "null" and n.key == "beq":
            arms = arms[1:]            # the then arm always runs; only the null path can go
        for arm in arms:
            if not any(x in m.pair for x in arm):
                optional |= arm

    live = [i for i in t.weighted if nodes[i].kind != "jump" and i not in folded
            and not (i in optional and i not in m.pair)]
    live_set = set(live)
    matched = sum(1 for i in live if i in m.pair)
    total = len(live)
    paired = set(m.pair)
    for p in t.pure:
        targets = t.feeds.get(p, set())
        if targets & live_set:
            total += 1
            if targets & paired:
                matched += 1
    claims = set(m.pair.values())
    for g in list(claims):
        claims |= m.g.feeders.get(g, set())
    calls = [i for i in live if nodes[i].kind == "call"]
    found = sum(1 for i in calls if i in m.pair)
    return Match(name, seed, dict(m.pair), dict(m.bind), matched, total, folded, frozenset(claims),
                 (found, len(calls)))


def accepted(m: Match, t: Prepared, min_weight: int = MIN_WEIGHT, min_conf: float = MIN_CONF,
             min_cover: float = MIN_COVER, min_calls: float = MIN_CALLS) -> bool:
    found, calls = m.calls
    return (m.matched >= min_weight and m.confidence >= min_conf and m.matched >= min_cover * t.static_weight
            and found >= min_calls * calls)


# ---------------------------------------------------------------- search

def find_copies(templates: dict[str, Prepared], g: Prepared, exclude: str = "", top: int = 3,
                min_weight: int = MIN_WEIGHT, min_conf: float = MIN_CONF, min_cover: float = MIN_COVER,
                max_weight: int = MAX_WEIGHT, overlaps: bool = False, rank=None) -> list[Match]:
    """Every template's copies inside g, overlaps resolved (most matched weight wins)."""
    found: list[Match] = []
    for name in sorted(templates):
        if name == exclude:
            continue
        t = templates[name]
        if not (min_weight <= t.static_weight <= max_weight):
            continue
        present = sorted((len(g.anchors[a]), a) for a in t.anchors if a in g.anchors)
        if not present:
            continue
        need = sum(len(v) for v in t.anchors.values())
        w = window(t)
        seen: set[int] = set()
        for _, a in present[:top]:
            for sg in g.anchors[a]:
                if sg in seen:
                    continue
                # at least half of the template's anchors must occur near the seed
                have = 0
                for ta, tnodes in t.anchors.items():
                    occ = g.anchors.get(ta)
                    if occ:
                        have += min(len(tnodes), bisect.bisect_right(occ, sg + w) - bisect.bisect_left(occ, sg - w))
                if 2 * have < need:
                    continue
                for st in t.anchors[a]:
                    m = match_at(t, g, st, sg, name, int(min_cover * t.static_weight) - len(t.pure))
                    if m is None:
                        continue
                    if accepted(m, t, min_weight, min_conf, min_cover):
                        found.append(m)
                        seen |= set(m.pairs.values())
                        break
    return found if overlaps else resolve(found, rank)


def resolve(matches: list[Match], rank=None) -> list[Match]:
    """Keep non-overlapping matches: most matched weight first, then confidence, compactness, and
    `rank(template)` (lower is better: exact before open, named before fn_)."""
    rank = rank or (lambda name: 0)
    order = sorted(matches, key=lambda m: (-m.matched, -m.confidence, m.span[1] - m.span[0], rank(m.template),
                                           m.template, m.span[0]))
    taken: set[int] = set()
    kept = []
    for m in order:
        nodes = set(m.pairs.values())
        if nodes & taken:
            continue
        taken |= nodes
        kept.append(m)
    return sorted(kept, key=lambda m: (m.span[0], m.template))


# ---------------------------------------------------------------- literals

def canonicalize(flow: Flow, canon) -> Flow:
    """Rewrite symbol values through `canon(value) -> value` (literal-pool and string symbols
    become their contents, so copies compiled in another unit compare equal)."""
    cache: dict = {}

    def c(v: Value) -> Value:
        if v is None or v[0][0] != "sym":
            return v
        if v not in cache:
            cache[v] = canon(v)
        return cache[v]

    for n in flow.nodes:
        if any(v[0][0] == "sym" for v in n.ins):
            n.ins = tuple(c(v) for v in n.ins)
        if n.mem is not None:
            n.mem = c(n.mem)
        if n.args:
            n.args = tuple((r, c(v)) for r, v in n.args)
        if n.regs:
            n.regs = {r: c(v) for r, v in n.regs.items()}
    return flow


def literal_canon(symbols: dict, read):
    """canon() for canonicalize: `symbols` maps name -> (section, address, size), `read(addr, n)`
    returns original bytes. Auto-named .sdata2/.rodata objects compare by bytes; a pointer into
    an auto-named .data/.rodata string compares by the string. Named globals keep their names."""
    def canon(v: Value) -> Value:
        name, off = v[0][1], v[1]
        sym = symbols.get(name)
        if sym is None or not name.startswith(("lbl_", "@")):
            return v
        section, address, size = sym
        try:
            if section in (".data", ".rodata"):
                raw = read(address + off, 64)
                end = raw.find(b"\0")
                text = raw[:end] if end > 0 else b""
                if text and all(32 <= ch < 127 or ch in (9, 10) for ch in text):
                    return (("str", text.decode("latin-1")), 0)
            if section in (".sdata2", ".rodata") and 0 < size <= 64:
                return (("lit", read(address, size)), off)
        except ValueError:
            pass
        return v
    return canon


# ---------------------------------------------------------------- repeated blocks

KGRAM = 8                # shortest repeated block worth reporting (normalised instructions)
BLOCK_CHECK = 0.75       # share of a block's instructions whose values agree with its partner


def token(node) -> str:
    """An instruction with registers and stack displacements abstracted (field offsets,
    immediates and symbols kept): equal tokens are the same operation on another register."""
    ops = []
    for op in node.insn.ops:
        if op == "r1":
            ops.append("sp")
        elif is_reg(op):
            ops.append("F" if op.startswith("f") else "R")
        elif op.endswith("(r1)"):
            ops.append("sp")
        elif op.endswith(")") and "(" in op:
            ops.append(op[:op.index("(")] + "(R)")
        elif op.startswith(".L_") or op.startswith('".L_'):
            ops.append("L")
        else:
            ops.append(op)
    if node.mnem in ("addi", "subi") and len(node.insn.ops) == 3 and node.insn.ops[1] == "r1":
        ops[2] = "*"                               # addi rX, r1, K: a stack address
    return node.mnem.rstrip("+-") + " " + ",".join(ops)


def block_sequence(p: Prepared) -> list[int]:
    """The nodes repeated-block search looks at: no frame, copies or no-ops."""
    return [n.i for n in p.flow.nodes if n.i not in p.frame and n.kind not in ("copy", "skip", "frame")]


@dataclass
class Segment:
    """One suggested helper: a piece of a block between strong cuts."""
    start: int
    end: int
    partners: list                  # (function, displacement) of copies covering all of it
    verified: float = 0.0           # share of instructions whose values agree with the first partner
    bind: dict = field(default_factory=dict)


@dataclass
class Block:
    """A stretch of one function that also occurs elsewhere (one or more partner copies)."""
    function: str
    start: int                      # positions in block_sequence(function), end exclusive
    end: int
    runs: list                      # (start, end, (partner function, displacement)) shared stretches
    segments: list = field(default_factory=list)

    @property
    def partners(self) -> list:
        return sorted({r[2] for r in self.runs})


def kgram_index(seqs: dict[str, list[str]], k: int = KGRAM) -> dict[tuple, list[tuple[str, int]]]:
    index: dict[tuple, list[tuple[str, int]]] = {}
    for name in sorted(seqs):
        toks = seqs[name]
        for i in range(len(toks) - k + 1):
            gram = tuple(toks[i:i + k])
            if len(set(gram)) >= 3:                # not a run of identical stores
                index.setdefault(gram, []).append((name, i))
    return index


def repeated_blocks(name: str, tokens: dict[str, list[str]], index: dict, k: int = KGRAM) -> list[Block]:
    """Stretches of `name` that occur elsewhere. A run is a maximal stretch shared with one partner
    (another function or place, at one displacement); overlapping runs form a block, and the
    places inside a block where runs start or end are where one inlined helper gives way to the
    next (the suggested helper boundaries)."""
    toks = tokens[name]
    active: dict[tuple, int] = {}
    runs: list[tuple[int, int, tuple]] = []
    last = len(toks) - k
    for p in range(last + 2):
        here = set()
        if p <= last:
            for n2, q in index.get(tuple(toks[p:p + k]), []):
                if not (n2 == name and abs(q - p) < k):
                    here.add((n2, q - p))
        for al in sorted(set(active) - here):
            runs.append((active.pop(al), p - 1 + k, al))
        for al in here:
            active.setdefault(al, p)
    runs.sort()
    blocks: list[Block] = []
    for r in runs:
        if blocks and r[0] < blocks[-1].end:
            blocks[-1].end = max(blocks[-1].end, r[1])
            blocks[-1].runs.append(r)
        else:
            blocks.append(Block(name, r[0], r[1], [r]))
    for b in blocks:
        b.segments = _segments(b)
    return blocks


def _segments(b: Block, shortest: int = 4) -> list[Segment]:
    """Cut a block where many shared stretches start or end together (at least 2, and a quarter
    of those around): copies of one helper begin and end at the same place."""
    starts: dict[int, int] = {}
    ends: dict[int, int] = {}
    cover = [0] * (b.end - b.start)
    for r in b.runs:
        starts[r[0]] = starts.get(r[0], 0) + 1
        ends[r[1]] = ends.get(r[1], 0) + 1
        for x in range(r[0], r[1]):
            cover[x - b.start] += 1
    cuts = [b.start]
    for x in range(b.start + 1, b.end):
        strength = starts.get(x, 0) + ends.get(x, 0)
        around = max(cover[x - b.start - 1], cover[x - b.start])
        if strength >= 2 and 4 * strength >= around and x - cuts[-1] >= shortest and b.end - x >= shortest:
            cuts.append(x)
    cuts.append(b.end)
    out = []
    for s0, e0 in zip(cuts, cuts[1:]):
        full = sorted({r[2] for r in b.runs if r[0] <= s0 and r[1] >= e0})
        if not full:      # no single copy spans it: the ones covering most of it
            best = max(min(r[1], e0) - max(r[0], s0) for r in b.runs)
            full = sorted({r[2] for r in b.runs if min(r[1], e0) - max(r[0], s0) == best})
        out.append(Segment(s0, e0, full))
    return out


def same_block(fa: Flow, a: list[int], fb: Flow, b: list[int]) -> tuple[float, dict]:
    """Pair a[i] with b[i]: values made inside the block must correspond, values from outside
    bind with one offset per base (the stack only to the stack). Share of agreeing nodes."""
    pos = {x: i for i, x in enumerate(a)}
    bind: dict = {}
    ok = 0

    def agree(va: Value, vb: Value) -> bool:
        ra, oa = va
        rb, ob = vb
        if ra[0] == "n" and ra[1] in pos:
            return rb[0] == "n" and rb[1] == b[pos[ra[1]]] and rb[2] == ra[2] and oa == ob
        if ra[0] in FIXED:
            return va == vb
        if ra[0] == "x" or rb[0] == "x":
            return False
        if ra in bind:
            return bind[ra] == (rb, ob - oa)
        if (ra[0] == "sp") != (rb[0] == "sp"):
            return False
        bind[ra] = (rb, ob - oa)
        return True

    evidence = 0
    for x, y in zip(a, b):
        na, nb = fa.nodes[x], fb.nodes[y]
        if na.kind == "pure":                  # its value shows up in the nodes that use it
            continue
        evidence += 1
        good = na.key == nb.key and len(na.ins) == len(nb.ins) and all(agree(u, v) for u, v in zip(na.ins, nb.ins))
        if good and na.kind == "call":
            good = all(nb.regs and r in nb.regs and agree(v, nb.regs[r]) for r, v in na.args)
        ok += good
    return (ok / evidence if evidence else 0.0), bind


def check_blocks(blocks: list[Block], preps: dict[str, Prepared], seqs: dict[str, list[int]]) -> list[Block]:
    """Verify every segment against its first partner with same_block."""
    for blk in blocks:
        for seg in blk.segments:
            other, d = seg.partners[0]
            lo, hi = max(seg.start, -d), min(seg.end, len(seqs[other]) - d)
            a = seqs[blk.function][lo:hi]
            b = seqs[other][lo + d:hi + d]
            seg.verified, seg.bind = same_block(preps[blk.function].flow, a, preps[other].flow, b)
            seg.verified *= (hi - lo) / (seg.end - seg.start)
    return blocks
