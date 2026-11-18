"""dk try with line-level edits (atoms), their combinations, and the oracle's verdict per variant.

  ./dk try <unit> <fn> --replace OLD NEW [--replace OLD NEW …] [--grid] [--oracle]
  ./dk try <unit> <fn> --atoms EDITS.txt --grid --oracle
  ./dk try <unit> <fn> VARIANTS.cpp --oracle

An atom replaces one piece of the unit's source (OLD must occur once: in the function's definition when it
is there, else once in the file; whitespace may differ) by NEW. EDITS.txt holds atoms separated by lines
`// ---`, each `OLD` lines, a line `// =>`, then `NEW` lines (an optional first line `// label: …`).

Without --grid each atom is one variant. --grid compiles every combination of 1 to K atoms (K = 3, or 2
for a function over 2 KB, `--grid-max`): the single atoms first, then the pairs and triples of the
best-ranked singles first and those of atoms that alone change nothing in the code last, never two atoms
that touch the same text, at most `--max-compiles`. A
combination whose code a batched compile (batch.py, free functions under 2 KB) shows to repeat an earlier
one is not compiled again; otherwise only identical sources are skipped.

--oracle compiles each variant under the compiler oracle (DK_ORACLE, contract v2) and reports, next to the
fuzzy %: the values coloured as the target (N/M), the solver's rules still to satisfy on the variant's own
graph (K, with their values), and close v2's ranking key (exact, rows that differ in more than registers,
rules of the starting source satisfied, wrong values, fuzzy %). Variants are ranked by that key, so one
that puts every value in the right register but moves an instruction is not shown as a plain loss.

Guards: an atom that declares a new name that is a placeholder (t0, tmp, temp1, __typeof__ …) is refused:
new locals need real names. `--apply` keeps an exact variant only when the judge (judge.py: hand-linked
identical, nothing else worse, no device) accepts it.
"""

from __future__ import annotations

import itertools
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ...core.command import BUILD_ERROR, MISMATCH, OK, USAGE, hint
from .jobs import resolve

ARROW = "// =>"
PLACEHOLDER = re.compile(r"(?:t|tmp|temp|var|v|unk|val)\d*|_\w*|\w*__\w*")
KEYWORDS = {"return", "if", "else", "for", "while", "do", "switch", "case", "break", "continue", "const", "static",
            "int", "float", "double", "char", "short", "long", "unsigned", "signed", "void", "bool", "true", "false",
            "sizeof", "struct", "class", "union", "enum", "inline", "volatile", "goto", "default", "this", "new",
            "delete", "operator", "typedef", "extern", "register", "auto", "u8", "u16", "u32", "s8", "s16", "s32",
            "f32", "f64"}


@dataclass
class Atom:
    old: str
    new: str
    label: str = ""

    def name(self, k: int) -> str:
        return self.label or f"a{k}"


@dataclass
class Row:
    """One compiled variant."""
    atoms: tuple[int, ...]
    text: str
    percent: float = 0.0
    error: str = ""
    worse: list[str] = field(default_factory=list)
    e: object = None                  # closecmd.Eval with --oracle
    right: int | None = None
    known: int | None = None
    rules_left: list[str] = field(default_factory=list)     # the solver's flipped rules on this variant
    reachable: object = None
    sat: int = 0
    key: tuple = ()
    code: str = ""                    # hash of the function's code and relocations (same code: same verdict)


def parse_atoms(text: str) -> list[Atom]:
    """Atoms from a file: blocks separated by `// ---`, OLD and NEW separated by `// =>`."""
    from .trycmd import SEPARATOR
    out, block = [], []
    for line in text.splitlines() + [SEPARATOR]:
        if line.strip() == SEPARATOR:
            if any(x.strip() for x in block):
                label = ""
                if block and block[0].strip().startswith("// label:"):
                    label = block.pop(0).strip()[len("// label:"):].strip()
                arrows = [i for i, x in enumerate(block) if x.strip() == ARROW]
                if len(arrows) != 1:
                    raise ValueError(f"an atom needs one line `{ARROW}` between OLD and NEW: {' '.join(block)[:80]}")
                i = arrows[0]
                out.append(Atom("\n".join(block[:i]).strip("\n"), "\n".join(block[i + 1:]).strip("\n"), label))
            block = []
        else:
            block.append(line)
    return out


def _folded(old: str) -> re.Pattern:
    parts = [re.escape(p) for p in re.findall(r"\w+|\S", old)]
    return re.compile(r"\s*".join(parts))


def locate(src: str, old: str, span: tuple[int, int]) -> tuple[int, int] | None:
    """Where OLD is: its one occurrence in the function's definition, else its one occurrence in the file;
    exact text first, then with whitespace folded. None when it is not there once."""
    if not old.strip():
        return None
    for finder in (lambda: [(m.start(), m.start() + len(old)) for m in re.finditer(re.escape(old), src)],
                   lambda: [(m.start(), m.end()) for m in _folded(old).finditer(src)]):
        hits = finder()
        inside = [h for h in hits if span[0] <= h[0] and h[1] <= span[1]]
        if len(inside) == 1:
            return inside[0]
        if not inside and len(hits) == 1:
            return hits[0]
        if hits:
            return None
    return None


def compose(src: str, places: list[tuple[int, int] | None], atoms: list[Atom], pick: tuple[int, ...]) -> str | None:
    """The source with the picked atoms applied together; None when one is not found or two overlap."""
    edits = []
    for k in pick:
        if places[k] is None:
            return None
        edits.append((places[k][0], places[k][1], atoms[k].new))
    edits.sort()
    for (a0, a1, _), (b0, _b1, _) in zip(edits, edits[1:]):
        if b0 < a1:
            return None
    out = src
    for lo, hi, new in reversed(edits):
        out = out[:lo] + new + out[hi:]
    return out


def placeholder_names(atom: Atom, src: str) -> list[str]:
    """Names the atom brings in that nothing had before and that are placeholders, not real names."""
    if "__typeof__" in atom.new and "__typeof__" not in atom.old:
        return ["__typeof__"]
    had = set(re.findall(r"[A-Za-z_]\w*", src))
    fresh = set(re.findall(r"[A-Za-z_]\w*", atom.new)) - set(re.findall(r"[A-Za-z_]\w*", atom.old)) - had - KEYWORDS
    return sorted(n for n in fresh if PLACEHOLDER.fullmatch(n))


def grid(n: int, kmax: int, rank: list[int] | None = None, neutral: set[int] | frozenset = frozenset()):
    """Atom combinations of size 2..kmax: fewest neutral atoms first (an atom alone compiles to the current
    code), then those of the best-ranked singles (rank: atom -> place)."""
    rank = rank or list(range(n))
    for k in range(2, kmax + 1):
        yield from sorted(itertools.combinations(range(n), k),
                          key=lambda c: (sum(i in neutral for i in c), sum(rank[i] for i in c), c))


def rule_text(r: dict) -> str:
    def v(x):
        who = x.get("name") or x.get("kind", "?")
        return f"{who} (line {x['line']})" if x.get("line") else who
    return f"{r['cls']}: {v(r['before'])} before {v(r['after'])}"


def describe(row: Row, atoms: list[Atom]) -> str:
    return " + ".join(atoms[k].name(k) for k in row.atoms) if row.atoms else "(variant)"


# ---- scoring -------------------------------------------------------------------------------------

class Scorer:
    """Compiles variants of one unit's source: plain (fuzzy %, worse functions) or under the oracle."""

    def __init__(self, cfg, unit: dict, symbol: str, src: str, oracle: bool, jobs: int, log=print):
        self.cfg, self.unit, self.symbol, self.src, self.oracle, self.log = cfg, unit, symbol, src, oracle, log
        self.jobs = resolve(jobs, cfg)
        self.work = cfg.root / "build" / "dk" / "try-atoms"
        self.closer = None
        self.before: dict[str, float] = {}
        self.base: Row | None = None

    def start(self) -> str:
        """Compile the current source; "" or why not."""
        from .closecmd import Closer, Options
        if self.oracle:
            from . import oracle
            if oracle.runner(self.cfg) is None:
                return oracle.UNAVAILABLE
            self.closer = Closer(self.cfg, self.unit, self.symbol, Options(moves="oracle", jobs=self.jobs),
                                 log=lambda s: None)
            if not self.closer.v2:
                return "the oracle runner has no order solver (contract v2: `--extra`, `--force`, `--solve`)"
        rows = self.score([Row((), self.src)])
        self.base = rows[0]
        if self.base.error:
            return "the unit's current source does not compile: " + self.base.error
        if self.oracle and self.base.e is not None:
            self.closer.diagnose(self.base.e)
            self.base.sat = self.base.e.sat
            self.base.key = self.base.e.key
        return ""

    def score(self, rows: list[Row]) -> list[Row]:
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            list(pool.map(self._one, rows, range(len(rows))))
        return rows

    def _one(self, row: Row, slot: int) -> None:
        if self.oracle:
            return self._oracle(row, slot)
        from .matching import MatchError, score
        from .reorder import compile_variant
        work = self.work / f"v{slot}"
        try:
            obj, err = compile_variant(self.cfg, self.unit, row.text, work)
            if obj is None:
                row.error = " / ".join((err.strip().splitlines() or ["compile error"])[-2:])[:160]
                return
            s = score(self.cfg, self.unit["name"], self.cfg.root / self.unit["target_path"], obj)
            row.code = code_of(obj, self.symbol)
            got = {f.name: f.percent for f in s.functions}
            row.percent = got.get(self.symbol, 0.0)
            if not self.before:
                self.before = got
            row.worse = [n for n, p in self.before.items() if n != self.symbol and got.get(n, -1.0) < p]
            row.key = (-(row.percent == 100.0), len(row.worse) > 0, -row.percent)
        except MatchError as e:
            row.error = str(e)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _oracle(self, row: Row, slot: int) -> None:
        from . import oracle
        e = self.closer.evaluate(row.text, f"t{slot}")
        if e is None:
            row.error = "does not compile (or the function is missing)"
            return
        row.e, row.percent, row.sat, row.key, row.code = e, e.percent, e.sat, e.key, e.code
        if e.values is None:
            return
        row.right, row.known = oracle.registers_right(e.values)
        an = oracle.solve(self.closer.oracle_cmd, e.record, e.values, self.closer.work / f"solve{slot}")
        if "error" not in an:
            row.reachable = an.get("reachable")
            row.rules_left = [rule_text(r) for r in oracle.flat_rules(an) if r["flip"]]

    def close(self) -> None:
        if self.closer is not None:
            shutil.rmtree(self.closer.work, ignore_errors=True)
        shutil.rmtree(self.work, ignore_errors=True)


def code_of(obj: Path, symbol: str) -> str:
    import hashlib
    from .elf import Elf
    fn = Elf.parse(Path(obj).read_bytes()).functions().get(symbol)
    if fn is None:
        return ""
    return hashlib.sha1(fn.code + repr([(r.offset, r.type, r.symbol) for r in fn.relocs]).encode()).hexdigest()


def judge_exact(cfg, unit: dict, symbol: str, src: str, text: str):
    """The judge's verdict on an exact variant (closecmd.Closer.final_check): (ok, reasons, final text)."""
    from .closecmd import Closer, Options
    closer = Closer(cfg, unit, symbol, Options(moves="table", jobs=1), log=lambda s: None)
    try:
        e = closer.evaluate(text, "judge")
        before, _proof, err = closer.unit_score(src, "before")
        if e is None or before is None:
            return False, ["does not compile: " + err[-200:]], text
        verdict, final = closer.final_check(e, src, before)
        return verdict.ok, verdict.reasons, final
    finally:
        shutil.rmtree(closer.work, ignore_errors=True)


# ---- the command ---------------------------------------------------------------------------------

def size_of(cfg, unit: dict, symbol: str) -> int | None:
    from .elf import Elf
    try:
        fn = Elf.parse((cfg.root / unit["target_path"]).read_bytes()).functions().get(symbol)
    except OSError:
        return None
    return len(fn.code) if fn is not None else None


def batch_filter(cfg, unit: dict, symbol: str, src: str, rows: list[Row], jobs: int, seen: set) -> tuple[list[Row], int]:
    """Rows whose code is new (batch.py's hashes; the base's code counts as seen). (rows, skipped)."""
    from .batch import Batcher, compiler_for
    work = cfg.root / "build" / "dk" / "try-atoms" / "batch"
    try:
        b = Batcher(compiler_for(cfg, unit, work), src, symbol, src_is_c(unit), size_of(cfg, unit, symbol), 64, jobs)
        if not b.ready:
            return rows, 0
        hashes = b.hashes([src] + [r.text for r in rows])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if hashes[0]:
        seen.add(hashes[0])
    keep, skipped = [], 0
    for r, h in zip(rows, hashes[1:]):
        if h is not None and h in seen:
            skipped += 1
            continue
        if h is not None:
            seen.add(h)
        keep.append(r)
    return keep, skipped


def src_is_c(unit: dict) -> bool:
    return unit["metadata"]["source_path"].endswith(".c")


def run_atoms(cfg, args, unit: dict, src: str, span: tuple[int, int], variant_texts: list[str]) -> int:
    atoms: list[Atom] = [Atom(o, n) for o, n in (args.replace or [])]
    if args.atoms:
        try:
            atoms += parse_atoms(Path(args.atoms).read_text())
        except (OSError, ValueError) as e:
            print(f"error: {e}")
            return USAGE
    places = [locate(src, a.old, span) for a in atoms]
    usable = []
    for k, (a, p) in enumerate(zip(atoms, places)):
        bad = placeholder_names(a, src)
        if p is None:
            print(f"  atom {a.name(k)}: OLD is not in the source once: {' '.join(a.old.split())[:70]}")
        elif bad:
            print(f"  atom {a.name(k)}: refused, new locals need real names ({', '.join(bad)})")
            places[k] = None
        else:
            usable.append(k)
    if not usable and not variant_texts:
        print("error: no usable atom or variant")
        return USAGE
    scorer = Scorer(cfg, unit, args.function, src, args.oracle, args.jobs)
    try:
        why = scorer.start()
        if why:
            print(f"error: {why}")
            return USAGE
        base = scorer.base
        print(f"{args.function}: now {base.percent:.2f}%" + (_verdict(base) if args.oracle else ""))
        if args.oracle and scorer.closer.rules:
            print(f"  the solver's rules on the current source: {len(scorer.closer.rules)} "
                  f"({sum(1 for r in scorer.closer.rules if r['flip'])} flipped)")
        size = size_of(cfg, unit, args.function) or 0
        kmax = args.grid_max or (2 if size > 2048 else 3)
        cap = args.max_compiles
        seen_text = {src}
        seen_code: set = set()
        rows: list[Row] = []
        skipped = 0

        def batch(cands: list[Row]) -> list[Row]:
            nonlocal skipped
            fresh = []
            for r in cands:
                if r.text not in seen_text:
                    seen_text.add(r.text)
                    fresh.append(r)
            if args.grid and fresh:
                fresh, n = batch_filter(cfg, unit, args.function, src, fresh, resolve(args.jobs, cfg), seen_code)
                skipped += n
            fresh = fresh[:max(0, cap - len(rows))]
            rows.extend(scorer.score(fresh))
            return fresh

        singles = [Row((), t) for t in variant_texts] + [Row((k,), compose(src, places, atoms, (k,))) for k in usable]
        batch([r for r in singles if r.text is not None])
        if args.grid and len(usable) > 1:
            done = [r for r in rows if len(r.atoms) == 1 and not r.error]
            order = sorted(done, key=lambda r: r.key)
            rank = {r.atoms[0]: i for i, r in enumerate(order)}
            ranks = [rank.get(k, len(usable)) for k in range(len(atoms))]
            neutral = {r.atoms[0] for r in done if r.code and r.code == base.code}
            if neutral:
                print(f"  {len(neutral)} atom(s) alone compile to the current code: their combinations go last")
            pool = [c for c in grid(len(atoms), kmax, ranks, neutral) if all(k in usable for k in c)]
            while pool and len(rows) < cap:
                chunk, pool = pool[:max(1, cap - len(rows)) * 2], pool[max(1, cap - len(rows)) * 2:]
                cands = [Row(c, t) for c in chunk if (t := compose(src, places, atoms, c)) is not None]
                batch(cands)
        ranked = sorted((r for r in rows if not r.error), key=lambda r: r.key)
        report(rows, ranked, atoms, base, args.oracle, args.top)
        if args.json:
            import json
            Path(args.json).write_text(json.dumps(to_json(rows, atoms, base), indent=1) + "\n")
        best_single = next((r for r in ranked if len(r.atoms) <= 1), None)
        best = ranked[0] if ranked else None
        if args.grid and best is not None and best_single is not None and len(best.atoms) > 1 and best.key < best_single.key:
            print(f"  a combination beats every single edit: {describe(best, atoms)}")
        print(f"  {len(rows)} compiles" + (f", {skipped} skipped as repeated code" if skipped else ""))
        if getattr(args, "show", -1) >= 1 and ranked:
            from .trycmd import show_variant
            k = min(args.show, len(ranked))
            print(f"variant {k} ({describe(ranked[k - 1], atoms)}), target vs ours:")
            show_variant(cfg, unit, ranked[k - 1].text, args.function, cfg.root / "build" / "dk" / "try-atoms" / "show", getattr(args, "show_rows", 80))
        if best is None:
            return MISMATCH
        if best.percent == 100.0:
            ok, reasons, final = judge_exact(cfg, unit, args.function, src, best.text)
            if not ok:
                print(f"  the exact variant is refused by the judge: {'; '.join(reasons)[:300]}")
                return MISMATCH
            print(f"  exact and judged faithful: {describe(best, atoms)}")
            if args.apply:
                (cfg.root / unit["metadata"]["source_path"]).write_text(final)
                print(f"  applied to {unit['metadata']['source_path']}")
                hint(f"check {args.unit} --function {args.function} --private")
            return OK
        return MISMATCH
    finally:
        scorer.close()


def _verdict(r: Row) -> str:
    if r.e is None:
        return ""
    vals = f"values {r.right}/{r.known}" if r.right is not None else "values: not aligned"
    rules = f"rules left {len(r.rules_left)}" + (f" ({'; '.join(r.rules_left[:3])}"
                                                + (" …" if len(r.rules_left) > 3 else "") + ")" if r.rules_left else "")
    if r.reachable is False:
        rules += ", not reachable by order"
    k = r.key
    return f"  {vals}  {rules}  key (shape {k[1]}, sat {-k[2]}, wrong {k[3]})"


def register_better(r: Row, base: Row) -> bool:
    """Fewer values in another register than the base and no more rules left (a register gain the shape hides)."""
    if r.right is None or base.right is None or r.reachable is False:
        return False
    return (r.known - r.right) < (base.known - base.right) and len(r.rules_left) <= len(base.rules_left)


def to_json(rows: list[Row], atoms: list[Atom], base: Row) -> dict:
    def one(r: Row) -> dict:
        return {"atoms": [atoms[k].name(k) for k in r.atoms], "percent": r.percent, "error": r.error,
                "values_right": r.right, "values_known": r.known, "rules_left": r.rules_left,
                "reachable": r.reachable, "key": list(r.key[:5]) if r.key else None, "worse": r.worse,
                "register_better": register_better(r, base)}
    return {"base": one(base), "variants": [one(r) for r in rows]}


def report(rows: list[Row], ranked: list[Row], atoms: list[Atom], base: Row, oracle: bool, top: int) -> None:
    for r in [r for r in rows if r.error][:5]:
        print(f"  {describe(r, atoms)}: {r.error}")
    print(f"  ranked by {'close v2 key (exact, shape rows, rules satisfied, wrong values, fuzzy)' if oracle else 'fuzzy %'}"
          + ("; + better, = same, - worse, R worse by that key but fewer values in the wrong register and no more rules"
             if oracle else "") + ":")
    for r in ranked[:top]:
        mark = "+" if r.key < base.key else ("=" if r.key[:-1] == base.key[:-1] else "-")
        if oracle and mark == "-" and register_better(r, base):
            mark = "R"
        worse = f"  (worse: {', '.join(r.worse[:3])})" if r.worse else ""
        print(f"  {mark} {r.percent:6.2f}%{_verdict(r) if oracle else ''}{worse}   {describe(r, atoms)}")
    if len(ranked) > top:
        print(f"    … {len(ranked) - top} more")
