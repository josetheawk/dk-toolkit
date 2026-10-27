"""Find reference-source objects inside the target binary (known-source signature matching).

Compile every file of a reference decomp (SDK, runtime, MSL, ...), then look for each
object's functions in the DOL. Instruction words are compared with relocated fields
masked, so code matches regardless of where it was linked. An object is located by
voting: every function that matches uniquely proposes where the object's .text starts,
and the winning start is then verified function by function.
"""

from __future__ import annotations

import bisect
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .dol import Dol
from .elf import Elf

# Relocation type -> bits of the 32-bit instruction word that the linker owns.
_FIELD_MASK = {
    4: 0x0000FFFF,     # R_PPC_ADDR16_LO
    5: 0x0000FFFF,     # R_PPC_ADDR16_HI
    6: 0x0000FFFF,     # R_PPC_ADDR16_HA
    10: 0x03FFFFFC,    # R_PPC_REL24
    11: 0x0000FFFC,    # R_PPC_REL14
    109: 0x001FFFFF,   # R_PPC_EMB_SDA21 (rA + d16)
}


def masks_for(code_len: int, relocs) -> list[int]:
    """Per-word "keep" masks for a function's code (1 bits are compared)."""
    keep = [0xFFFFFFFF] * (code_len // 4)
    for r in relocs:
        i = r.offset // 4
        if 0 <= i < len(keep):
            keep[i] &= ~_FIELD_MASK.get(r.type, 0xFFFFFFFF) & 0xFFFFFFFF
    return keep


def masked_equal(words_a: tuple[int, ...], words_b: tuple[int, ...], keep: list[int]) -> bool:
    return len(words_a) == len(words_b) and all((a ^ b) & k == 0 for a, b, k in zip(words_a, words_b, keep))


@dataclass
class RefFunction:
    name: str
    offset: int          # within the object's .text
    words: tuple[int, ...]
    keep: list[int]
    relocs: int = 0      # relocation count; functions without any (stubs) are weak evidence

    @property
    def size(self) -> int:
        return len(self.words) * 4


@dataclass
class ObjectMatch:
    source: str                       # path relative to the reference src dir
    variant: str                      # compiler / flag label that produced the object
    text_size: int
    ref_functions: int = 0
    matched: list[tuple[str, int, int]] = field(default_factory=list)   # (name, DOL address, size), in order
    unmatched: list[str] = field(default_factory=list)                  # ref functions not found (stripped or differing)
    gap_functions: list[int] = field(default_factory=list)              # DOL functions inside the span that matched nothing
    object: str = ""                  # compiled reference object that produced this match
    compiler: str = ""
    flags: tuple[str, ...] = ()       # full flag list used for that compile

    @property
    def start(self) -> int | None:
        return self.matched[0][1] if self.matched else None

    @property
    def end(self) -> int | None:
        return self.matched[-1][1] + self.matched[-1][2] if self.matched else None

    @property
    def clean(self) -> bool:
        """Every DOL function in the span is explained by the reference (missing ones were dead-stripped)."""
        return bool(self.matched) and not self.gap_functions


def text_functions(obj: Path) -> tuple[list[RefFunction], int]:
    elf = Elf.parse(obj.read_bytes())
    funcs = []
    text_size = 0
    for fn in elf.functions().values():
        if fn.section != ".text" or not fn.code:
            continue
        sym = next(s for s in elf.symbols if s.name == fn.name and s.type == 2)
        words = tuple(int.from_bytes(fn.code[i:i + 4], "big") for i in range(0, len(fn.code) // 4 * 4, 4))
        funcs.append(RefFunction(fn.name, sym.value, words, masks_for(len(fn.code), fn.relocs), len(fn.relocs)))
    for s in elf.sections:
        if s.name == ".text":
            text_size = s.size
    funcs.sort(key=lambda f: f.offset)
    return funcs, text_size


class DolIndex:
    """Words of every known function in the DOL, bucketed by size."""

    def __init__(self, dol: Dol, functions: list[tuple[int, int]]):
        self.dol = dol
        self.by_size: dict[int, list[int]] = defaultdict(list)
        self.words: dict[int, tuple[int, ...]] = {}
        for address, size in functions:
            if size <= 0:
                continue
            try:
                raw = dol.read(address, size)
            except ValueError:
                continue
            self.words[address] = tuple(int.from_bytes(raw[i:i + 4], "big") for i in range(0, size // 4 * 4, 4))
            self.by_size[size // 4 * 4].append(address)

        self.sorted_addresses = sorted(self.words)

    def functions_between(self, lo: int, hi: int) -> int:
        """Number of known functions starting in [lo, hi)."""
        return bisect.bisect_left(self.sorted_addresses, hi) - bisect.bisect_left(self.sorted_addresses, lo)

    def candidates(self, fn: RefFunction) -> list[int]:
        return [a for a in self.by_size.get(fn.size, []) if masked_equal(self.words[a], fn.words, fn.keep)]


MAX_CANDIDATES = 40     # ignore functions that match this many places (stubs, `blr`)
MAX_GAP_FUNCTIONS = 2   # unexplained DOL functions allowed between consecutive matches
MIN_SINGLE_SIZE = 64    # a chain must cover at least this many bytes of code to count


def locate(index: DolIndex, source: str, variant: str, obj: Path) -> ObjectMatch:
    """Longest chain of matches increasing in both reference order and DOL address.

    Linkers dead-strip unused functions, so an object's surviving functions keep their order
    but not their offsets. Chaining finds them anyway; gaps bound the chain to one object.
    """
    funcs, text_size = text_functions(obj)
    return locate_functions(index, source, variant, funcs, text_size)


def locate_functions(index: DolIndex, source: str, variant: str, funcs: list[RefFunction], text_size: int) -> ObjectMatch:
    result = ObjectMatch(source, variant, text_size, len(funcs))
    pairs = []
    for i, fn in enumerate(funcs):
        hits = index.candidates(fn)
        if 0 < len(hits) <= MAX_CANDIDATES:
            pairs += [(a, i) for a in hits]
    pairs.sort()
    # best[k] = (length, -total_gap, prev index) for a chain ending at pairs[k]
    best: list[tuple[int, int, int]] = []
    for k, (a, i) in enumerate(pairs):
        choice = (1, 0, -1)
        for j in range(k):
            pa, pi = pairs[j]
            if pi >= i or pa >= a:
                continue
            gap = a - (pa + funcs[pi].size)
            if gap < 0 or index.functions_between(pa + funcs[pi].size, a) > MAX_GAP_FUNCTIONS:
                continue
            cand = (best[j][0] + 1, best[j][1] - gap, j)
            if cand[:2] > choice[:2]:
                choice = cand
        best.append(choice)
    if not best:
        result.unmatched = [f.name for f in funcs]
        return result
    k = max(range(len(best)), key=lambda x: (best[x][0], best[x][1], -pairs[x][0]))
    chain = []
    while k != -1:
        chain.append(pairs[k])
        k = best[k][2]
    chain.reverse()
    # A single tiny match proves nothing.
    # Too little evidence: a short chain, or one function with no calls/data references at all
    # (empty varargs stubs and the like are identical in unrelated code).
    if sum(funcs[i].size for _, i in chain) < MIN_SINGLE_SIZE or (len(chain) == 1 and not funcs[chain[0][1]].relocs):
        result.unmatched = [f.name for f in funcs]
        return result
    in_chain = {i for _, i in chain}
    result.matched = [(funcs[i].name, a, funcs[i].size) for a, i in chain]
    result.unmatched = [f.name for i, f in enumerate(funcs) if i not in in_chain]
    matched_addrs = {a for a, _ in chain}
    lo, hi = result.start, result.end
    result.gap_functions = sorted(a for a in index.words if lo <= a < hi and a not in matched_addrs)
    return result


@dataclass(frozen=True)
class Variant:
    label: str
    compiler: str                # e.g. GC/1.2.5n
    flags: tuple[str, ...]


def compile_all(root: Path, wibo: Path, compilers: Path, ref_root: Path, sources: list[Path],
                variant: Variant, includes: list[tuple[str, str]], out_dir: Path, jobs: int = 0) -> dict[Path, Path]:
    """Compile each source with one variant. Returns {source: object} for successful builds.

    includes: (flag, dir) pairs exactly as the reference build passes them, e.g. ("-I", "include"),
    ("-ir", "src"); dirs are relative to ref_root. "-I-" is always passed first.
    """
    exe = compilers / variant.compiler / "mwcceppc.exe"
    # mwcc wants "-I<dir>" attached, but "-i <dir>" / "-ir <dir>" as two arguments.
    inc = [a for flag, d in includes
           for a in ((f"{flag}{ref_root / d}",) if flag == "-I" else (flag, str(ref_root / d)))]

    def one(src: Path) -> tuple[Path, Path | None]:
        rel = src.relative_to(ref_root)
        dest = out_dir / variant.label / rel.parent
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [str(wibo), str(exe), *variant.flags, "-I-", *inc, "-c", str(src), "-o", str(dest)]
        proc = subprocess.run(cmd, cwd=root, capture_output=True)
        obj = dest / (src.stem + ".o")
        return src, obj if proc.returncode == 0 and obj.is_file() else None

    results: dict[Path, Path] = {}
    with ThreadPoolExecutor(max_workers=jobs or os.cpu_count() or 4) as pool:
        for src, obj in pool.map(one, sources):
            if obj is not None:
                results[src] = obj
    return results


def resolve_conflicts(matches: list[ObjectMatch]) -> list[ObjectMatch]:
    """Give each DOL address to the strongest object claiming it; drop chains left too weak."""
    claimed: set[int] = set()
    kept = []
    for m in sorted(matches, key=lambda m: (-len(m.matched), not m.clean, m.source)):
        if not m.matched:
            continue
        own = [x for x in m.matched if x[1] not in claimed]
        if len(own) != len(m.matched):
            lost = [x[0] for x in m.matched if x[1] in claimed]
            m.unmatched = sorted(set(m.unmatched) | set(lost))
            m.matched = own
        if not m.matched or (len(m.matched) == 1 and m.matched[0][2] < MIN_SINGLE_SIZE):
            continue
        claimed.update(x[1] for x in m.matched)
        kept.append(m)
    return sorted(kept, key=lambda m: m.start)
