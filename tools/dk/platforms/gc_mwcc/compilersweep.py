"""dk compiler-search --sweep: every WIP unit compiled privately with the other compiler (GC/1.3
vs 1.3.2 by default) and, where its code is C (no mangled names in the target), as C instead of
C++. A variant is reported when it gains (more exact functions, or a higher code score) and loses
no exact function; nothing is applied: the report prints the `dk unit flags` line to run (a
language change also renames the source). Linked units are never swept.

Why: units that matched only on another compiler (1.3 auto-inlines less; 1.3 and 1.3.2 lay out
weak copies differently) or as C were found by hand, one by one, in earlier runs.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .matching import MatchError, UnitScore, compile_command, retarget, score

CACHE = "build/dk/sweep.json"       # the last full `--sweep`, read by `dk route`
COMPILER = re.compile(r"compilers/([^/]+/[^/]+)/mwcceppc\.exe$")
MANGLED = re.compile(r"__(?:F|C?F|Q\d|\d)|^@\d+@|^__(?:ct|dt|vt|sinit)")


@dataclass
class Variant:
    compiler: str
    lang: str                      # "c" or "c++"
    score: UnitScore | None = None
    error: str = ""
    inline: str | None = None      # an -inline mode in place of the unit's own (None: keep the unit's)

    @property
    def label(self) -> str:
        return f"{self.compiler} {'C' if self.lang == 'c' else 'C++'}" + (f" -inline {self.inline}" if self.inline else "")


def with_inline(argv: list[str], mode: str | None) -> list[str]:
    """The command with every `-inline X` replaced by one `-inline mode` (None: unchanged)."""
    if mode is None:
        return argv
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "-inline":
            skip = True
            continue
        out.append(a)
    k = out.index("-c") if "-c" in out else len(out)
    return out[:k] + ["-inline", mode] + out[k:]


@dataclass
class Sweep:
    unit: dict
    base: Variant
    variants: list[Variant] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.unit["metadata"].get("source_path", self.unit["name"]).removeprefix("src/")


def exact_names(s: UnitScore | None) -> set[str]:
    return {f.name for f in s.functions if f.percent == 100.0} if s else set()


def compare(base: UnitScore, other: UnitScore) -> tuple[list[str], list[str], float]:
    """(functions exact only in other, exact functions other loses, code points gained)."""
    b, o = exact_names(base), exact_names(other)
    return sorted(o - b), sorted(b - o), other.code_percent - base.code_percent


def gains(base: UnitScore | None, other: UnitScore | None) -> bool:
    if base is None or other is None:
        return False
    won, lost, delta = compare(base, other)
    return not lost and delta > -0.005 and (bool(won) or delta > 0.005)     # and the unit's code not worse


def to_json(sweeps: list) -> list[dict]:
    """The sweep as data: per unit, what it scores now and each variant (gains, functions won and lost)."""
    return [{
        "unit": s.name, "now": s.base.label, "exact": len(exact_names(s.base.score)),
        "code": s.base.score.code_percent if s.base.score else None,
        "variants": [{"try": v.label, "error": v.error,
                      "exact": len(exact_names(v.score)) if v.score else None,
                      "code": v.score.code_percent if v.score else None,
                      "gains": gains(s.base.score, v.score),
                      "won": compare(s.base.score, v.score)[0] if v.score and s.base.score else [],
                      "lost": compare(s.base.score, v.score)[1] if v.score and s.base.score else []}
                     for v in s.variants]} for s in sweeps]


def c_code(target_functions: list[str]) -> bool:
    """Does the target's code look like C: no function name is a C++ mangled one?"""
    return bool(target_functions) and not any(MANGLED.search(n) for n in target_functions)


def with_lang(argv: list[str], lang: str) -> list[str]:
    out = [a for a in argv if not a.startswith("-lang")]
    k = out.index("-c") if "-c" in out else len(out)
    return out[:k] + [f"-lang={lang}"] + out[k:]


def current(argv_text: str, source: str) -> tuple[str, str]:
    m = COMPILER.search(argv_text.split(" -", 1)[0]) or COMPILER.search(argv_text)
    lang = "c" if source.endswith(".c") or "-lang=c " in argv_text + " " else "c++"
    if "-lang=c++" in argv_text:
        lang = "c++"
    return (m.group(1) if m else "?"), lang


EXTERN_C_BLOCK = re.compile(r'\bextern\s+"C"\s*\{')
EXTERN_C = re.compile(r'\bextern\s+"C"\s+')


def as_c(text: str) -> str:
    """A C++ source written as C: `extern "C"` (a no-op in C) dropped, and its `{ }` blocks opened."""
    from ...core.cbody import matching_brace
    while True:
        m = EXTERN_C_BLOCK.search(text)
        if m is None:
            break
        close = matching_brace(text, m.end() - 1)
        if close is None:
            break
        text = text[:m.start()] + text[m.end():close] + text[close + 1:]
    return EXTERN_C.sub("", text)


def _compile(cfg, unit: dict, command: str, v: Variant, base_lang: str = "c++") -> Variant:
    from .reorder import mirror
    work = Path(tempfile.mkdtemp(prefix="dk-sweep-", dir=cfg.root / "build" / "dk"))
    tree = None
    try:
        argv = with_inline(with_lang(retarget(command, work, v.compiler), v.lang), v.inline)
        cwd = cfg.root
        if v.lang == "c" and base_lang != "c":       # the source as C, in a private mirror of the tree
            source = unit["metadata"]["source_path"]
            tree = Path(tempfile.mkdtemp(prefix="dk-sweep-tree-"))
            mirror(cfg, source, as_c((cfg.root / source).read_text()), tree)
            cwd = tree
        import subprocess
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
        ok, text = proc.returncode == 0, proc.stdout + proc.stderr
        objs = list(work.glob("*.o"))
        if not ok or not objs:
            v.error = (text.strip().splitlines() or ["compile error"])[-1][:120]
            return v
        try:
            v.score = score(cfg, unit["name"], cfg.root / unit["target_path"], objs[0])
        except MatchError as e:
            v.error = str(e)[:120]
        return v
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if tree is not None:
            shutil.rmtree(tree, ignore_errors=True)


def sweep_unit(cfg, unit: dict, versions: list[str], languages: bool) -> Sweep | None:
    try:
        command = compile_command(cfg, unit["base_path"], locked=False)
    except MatchError:
        return None
    source = unit["metadata"].get("source_path", "")
    compiler, lang = current(command, source)
    base = _compile(cfg, unit, command, Variant(compiler, lang))
    out = Sweep(unit, base)
    if base.score is None:
        return out
    tries = [Variant(f"GC/{v}", lang) for v in versions if f"GC/{v}" != compiler]
    if languages and lang == "c++" and c_code([f.name for f in base.score.functions]):
        tries += [Variant(compiler, "c")] + [Variant(f"GC/{v}", "c") for v in versions if f"GC/{v}" != compiler]
    for v in tries:
        out.variants.append(_compile(cfg, unit, command, v, lang))
    return out


def run_sweep(cfg, units: list[dict], versions: list[str], languages: bool, jobs: int) -> list[Sweep]:
    (cfg.root / "build" / "dk").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        found = list(pool.map(lambda u: sweep_unit(cfg, u, versions, languages), units))
    return [s for s in found if s is not None]


def unit_flags_line(s: Sweep, v: Variant) -> str:
    path = s.name
    if v.lang != s.base.lang:
        new = re.sub(r"\.cpp$", ".c", path) if v.lang == "c" else re.sub(r"\.c$", ".cpp", path)
        tail = f"  # and compile as {'C' if v.lang == 'c' else 'C++'}: rename src/{path} to src/{new}"
    else:
        tail = ""
    flags = f"./dk unit flags {path} --mw-version {v.compiler}" if v.compiler != s.base.compiler else f"# {path}"
    return flags + tail


def render(sweeps: list[Sweep], every: bool = False) -> tuple[list[str], int]:
    """The report lines (gains first, best first) and how many units gain."""
    out = []
    winners = []
    for s in sweeps:
        best = None
        for v in s.variants:
            if gains(s.base.score, v.score):
                key = compare(s.base.score, v.score)
                if best is None or (len(key[0]), key[2]) > (len(compare(s.base.score, best.score)[0]),
                                                           compare(s.base.score, best.score)[2]):
                    best = v
        if best is not None:
            winners.append((s, best))
    for s, v in sorted(winners, key=lambda sv: -len(compare(sv[0].base.score, sv[1].score)[0])):
        won, _, delta = compare(s.base.score, v.score)
        b, o = len(exact_names(s.base.score)), len(exact_names(v.score))
        data = [sec for sec, p in s.base.score.data.items() if v.score.data.get(sec, 0.0) < p]
        out.append(f"  GAIN  {s.name:<44} {s.base.label:<12} -> {v.label:<12} exact {b} -> {o}, code "
                   f"{s.base.score.code_percent:.2f}% -> {v.score.code_percent:.2f}% ({delta:+.2f})"
                   + (f", data worse in {', '.join(data)}" if data else ""))
        if won:
            out.append(f"        newly exact: {', '.join(won[:6])}{' …' if len(won) > 6 else ''}")
        out.append(f"        {unit_flags_line(s, v)}")
    if every:
        for s in sweeps:
            if any(s is w for w, _ in winners):
                continue
            for v in s.variants:
                if v.score is None:
                    out.append(f"  ----  {s.name:<44} {v.label:<12} {v.error or 'no score'}")
                    continue
                won, lost, delta = compare(s.base.score, v.score)
                out.append(f"  ----  {s.name:<44} {v.label:<12} +{len(won)} exact, -{len(lost)} exact, code {delta:+.2f}")
    return out, len(winners)


# ---- one compiler for every unit? ------------------------------------------------------------------

GRID = (("GC/1.3", "auto"), ("GC/1.3", "noauto"), ("GC/1.3.2", "auto"), ("GC/1.3.2", "noauto"))


@dataclass
class Uniform:
    unit: dict
    lib: str
    linked: bool
    base: Variant
    cells: list[Variant] = field(default_factory=list)


def losses(base: UnitScore | None, other: UnitScore | None) -> list[str]:
    """What a setting loses against the unit's own: exact functions, code, data, layout."""
    if base is None:
        return []
    if other is None:
        return ["does not compile"]
    won, lost, delta = compare(base, other)
    out = [f"{len(lost)} exact lost"] if lost else []
    if delta < -0.005:
        out.append(f"code {delta:+.2f}")
    out += [f"{sec} {other.data.get(sec, 0.0) - p:+.2f}" for sec, p in sorted(base.data.items())
            if other.data.get(sec, 0.0) < p - 0.005]
    new = [p for p in other.layout_problems if p not in base.layout_problems]
    if new:
        out.append("layout")
    return out


def uniform_unit(cfg, unit: dict, lib: str, grid) -> Uniform | None:
    try:
        command = compile_command(cfg, unit["base_path"], locked=False)
    except MatchError:
        return None
    compiler, lang = current(command, unit["metadata"].get("source_path", ""))
    out = Uniform(unit, lib, bool(unit["metadata"].get("complete")), _compile(cfg, unit, command, Variant(compiler, lang)))
    if out.base.score is not None:
        out.cells = [_compile(cfg, unit, command, Variant(c, lang, inline=mode), lang) for c, mode in grid]
    return out


def run_uniform(cfg, units: list[dict], libs: dict[str, str], grid, jobs: int) -> list[Uniform]:
    (cfg.root / "build" / "dk").mkdir(parents=True, exist_ok=True)
    lib_of = lambda u: libs.get(u["metadata"].get("source_path", "").removeprefix("src/"), "?")
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        found = list(pool.map(lambda u: uniform_unit(cfg, u, lib_of(u), grid), units))
    return [x for x in found if x is not None and x.base.score is not None]


def render_uniform(rows: list[Uniform], grid, limit: int = 12) -> list[str]:
    """Per setting: how many units (linked, WIP) lose nothing on it, and which lose, by library;
    then the linked units grouped by the settings that keep them exact."""
    out = []
    labels = [f"{c} -inline {m}" for c, m in grid]
    for k, label in enumerate(labels):
        bad = [(r, losses(r.base.score, r.cells[k].score)) for r in rows if k < len(r.cells)]
        bad = [(r, why) for r, why in bad if why]
        linked = sum(1 for r in rows if r.linked)
        lost_linked = sum(1 for r, _ in bad if r.linked)
        out.append(f"{label:<24} loses nothing in {len(rows) - len(bad)}/{len(rows)} units "
                   f"(linked {linked - lost_linked}/{linked})")
        by_lib: dict[str, list] = {}
        for r, why in bad:
            by_lib.setdefault(r.lib, []).append((r, why))
        for lib, items in sorted(by_lib.items(), key=lambda kv: -len(kv[1])):
            names = ", ".join(f"{r.unit['metadata'].get('source_path', '').removeprefix('src/')}"
                              f"{'*' if r.linked else ''} ({'; '.join(why)})" for r, why in items[:limit])
            out.append(f"    {lib}: {len(items)}: {names}{' …' if len(items) > limit else ''}")
    groups: dict[tuple, list[Uniform]] = {}
    for r in rows:
        if r.linked:
            keep = tuple(label for k, label in enumerate(labels) if k < len(r.cells) and not losses(r.base.score, r.cells[k].score))
            groups.setdefault(keep, []).append(r)
    out.append("linked units by the settings that keep them exact:")
    for keep, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        libs = {}
        for r in items:
            libs[r.lib] = libs.get(r.lib, 0) + 1
        out.append(f"  {len(items):4} {' | '.join(keep) or '(none of them)'}   by library: "
                   + ", ".join(f"{lib} {n}" for lib, n in sorted(libs.items(), key=lambda kv: -kv[1])))
    return out
