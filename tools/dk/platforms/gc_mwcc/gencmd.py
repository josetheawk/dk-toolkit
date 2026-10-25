"""dk gen: verified drafts for small functions (see gen.py for the method).

  ./dk gen <function>...        exact drafts, ready to paste (exit 1 if any has none)
  ./dk gen --unit <unit>        every function of a declared unit
  ./dk gen --survey             dry run over all functions outside linked units: counts only

Drafts are verified in isolation with the unit's own compile command (or, outside declared
units, the template unit's from dk.toml [platform.gen]); `dk check` still judges the unit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from . import asmgen, asmindex, classgen, gen
from .ctxcmd import class_slots
from .dol import Dol
from .dtk_symbols import is_auto_name, read_symbols, read_units
from .elf import Elf
from .matching import MatchError, compile_command, load_objdiff_units, resolve_unit, retarget

MAX_INSNS = 16
LITERAL_SECTIONS = (".sdata2", ".rodata")


@dataclass
class Result:
    name: str
    address: int
    status: str                 # exact | differs | unsupported | thunk | error | skipped
    detail: str = ""
    source: str = ""            # display draft (exact only)
    notes: list[str] = field(default_factory=list)
    compiler: str = ""
    closest: str = ""           # differs: how far the closest variant matched


class Context:
    """Everything one run needs, loaded once."""

    def __init__(self, cfg: Config):
        opts = cfg.platform_options
        self.cfg = cfg
        self.symbols = read_symbols(cfg.root / opts["symbols"])
        self.by_name = {s.name: s for s in self.symbols}
        self.by_address = {s.address: s for s in self.symbols if s.type == "function"}
        self.dol = Dol.parse(cfg.original.read_bytes())
        self.entries = {n: e for n, e in asmindex.index(asmindex.current_asm_files(cfg)).items() if e.kind == "fn"}
        self.splits = read_units(cfg.root / opts["splits"])
        units = [u for u in _objdiff_units(cfg) if u.get("base_path")]
        self.objdiff = {u["metadata"]["source_path"].removeprefix(opts.get("src", "src") + "/"): u for u in units}
        settings = opts.get("gen", {})
        self.code_end = int(settings.get("code_end", 0))
        self.template = settings.get("template")
        self._commands: dict[str, str] = {}
        self._command_lock = threading.Lock()      # compile_command runs ninja under a process lock
        self._signatures: dict[str, tuple[int, int] | None] = {}
        self._signature_lock = threading.RLock()
        self.max_insns = MAX_INSNS
        self._callers: dict | None = None
        rtti = cfg.root / "build" / "dk" / "rtti.json"
        self.class_records = json.loads(rtti.read_text()) if rtti.is_file() else []
        self.classes = {c["name"]: c for c in self.class_records}

    def vsignature(self, owner: str | None):
        """slot -> parameter counts of what `owner`'s primary vtable holds there."""
        cls = self.classes.get(owner or "")
        vt = next((v for v in cls["vtables"] if v["offset"] == 0), None) if cls else None
        if vt is None:
            return None

        def sig(slot: int):
            if slot >= len(vt["slots"]) or not vt["slots"][slot]:
                return None
            target = self.by_address.get(vt["slots"][slot])
            return self.signature(target.name) if target else None
        return sig

    def signature(self, name: str, depth: int = 0) -> tuple[int, int] | None:
        """A function's (GPR, FPR) parameter counts, inferred from its own asm. Serialised:
        the recursion guard below must never be seen by another worker thread."""
        with self._signature_lock:
            if name in self._signatures:
                return self._signatures[name]
            entry = self.entries.get(name)
            if entry is None or depth > 4:
                return None
            self._signatures[name] = None              # recursion guard
            sig = gen.infer_signature(gen.parse(entry.text), lambda n: self.signature(n, depth + 1))
            self._signatures[name] = sig
            return sig

    def literal(self, sym: str, size: int, offset: int = 0) -> bytes | None:
        s = self.by_name.get(sym)
        if s is None or s.section not in LITERAL_SECTIONS or not (is_auto_name(sym) or sym.startswith("@")):
            return None
        try:
            return self.dol.read(s.address + offset, size)
        except ValueError:
            return None

    def literal_info(self, sym: str) -> tuple[str, int] | None:
        s = self.by_name.get(sym)
        if s is None or s.section not in LITERAL_SECTIONS or not (is_auto_name(sym) or sym.startswith("@")):
            return None
        return s.section, s.size

    def result_used(self, name: str) -> bool | None:
        """Do callers read r3 after calling `name`? True/False from the callers' asm, None
        when nothing calls it directly (virtual functions)."""
        with self._signature_lock:
            if self._callers is None:
                self._build_callers()
        return self._result_used(name)

    def _build_callers(self) -> None:
        callers: dict = {}
        for entry in self.entries.values():
            insns = gen.parse(entry.text)
            for i, insn in enumerate(insns):
                if insn.op == "bl" and insn.args:
                    callers.setdefault(insn.args[0], []).append(insns[i + 1:])
        self._callers = callers

    def _result_used(self, name: str) -> bool | None:
        sites = self._callers.get(name)
        if not sites:
            return None
        for after in sites:
            for insn in after:
                reads, writes = gen._reads_writes(insn, None)
                if "r3" in reads or "f1" in reads:
                    return True
                if "r3" in writes or insn.op in ("bl", "b", "blr", "bctrl") or insn.op.startswith("b"):
                    break
        return False

    def unit_of(self, address: int) -> str | None:
        return next((u.name for u in self.splits
                     if any(sec == ".text" and a <= address < b for sec, a, b in u.splits)), None)

    def linked(self, unit: str | None) -> bool:
        u = self.objdiff.get(unit or "")
        return bool(u and u["metadata"].get("complete"))

    def command(self, address: int) -> tuple[str | None, str]:
        """(compile command, why not) for a function at `address`."""
        unit = self.unit_of(address)
        if unit is None:
            if not self.template or address >= self.code_end:
                return None, "library code outside a declared unit (use the reference decomps)"
            unit = self.template
        if not unit.endswith((".cpp", ".cp")):
            return None, f"{unit} is C (drafts are C++)"
        with self._command_lock:
            if unit not in self._commands:
                u = self.objdiff.get(unit)
                if u is None:
                    return None, f"{unit} has no objdiff unit"
                try:
                    self._commands[unit] = compile_command(self.cfg, u["base_path"])
                except MatchError as e:
                    return None, str(e)
            return self._commands[unit], ""


def _objdiff_units(cfg: Config) -> list[dict]:
    """objdiff.json, retried: a parallel agent's build may be rewriting it right now."""
    for attempt in range(5):
        try:
            return load_objdiff_units(cfg)
        except json.JSONDecodeError:
            time.sleep(1 + attempt)
    return load_objdiff_units(cfg)


def _display_names(ctx: Context, name: str, address: int) -> tuple[str, str, bool]:
    """(class for display, method name, r3 is `this` of a known class)."""
    m = re.match(r"^(?P<m>[A-Za-z_]\w*?)__(?P<n>\d+)(?P<rest>\w+)$", name)
    if m and len(m["rest"]) >= int(m["n"]):
        return m["rest"][:int(m["n"])], m["m"], True
    _, owner = class_slots(ctx.cfg, address)
    if owner:
        return owner, name, True
    return f"Unk{address:08X}", name, False


def _compile(ctx: Context, command: str, source: str, work: Path, stem: str) -> tuple[Elf | None, str]:
    src = work / f"{stem}.cpp"
    src.write_text(source)
    args = retarget(command, work)
    i = args.index("-c")
    args[i + 1] = str(src)
    proc = subprocess.run(args, cwd=ctx.cfg.root, capture_output=True, text=True, timeout=60)
    obj = work / f"{stem}.o"
    if proc.returncode != 0 or not obj.is_file():
        text = [l for l in (proc.stdout + proc.stderr).splitlines()
                if l.strip() and "Usage Warning" not in l and "Specified directory" not in l]
        return None, next((l.strip() for l in text if "rror" in l), text[-1].strip() if text else "compile failed")
    return Elf.parse(obj.read_bytes()), ""


def solve(ctx: Context, name: str, work: Path) -> Result:
    sym = ctx.by_name.get(name)
    if sym is None or name not in ctx.entries:
        return Result(name, 0, "skipped", "no such function in the asm")
    insns = gen.parse(ctx.entries[name].text)
    result = Result(name, sym.address, "unsupported")
    kind = gen.thunk_kind(insns)
    if kind:
        result.status, result.detail = "thunk", f"compiler-generated {kind}: write the class, not this function"
        return result
    try:
        body = gen.strip_frame(insns)
    except gen.Unsupported as e:
        result.detail = str(e)
        return result
    if len(body) > ctx.max_insns:
        result.detail = f"longer than {ctx.max_insns} instructions"
        return result
    command, why = ctx.command(sym.address)
    if command is None:
        result.status, result.detail = "skipped", why
        return result
    asm_of = lambda n: ctx.entries[n].text if n in ctx.entries else None
    ctors = classgen.solve_ctor(insns, ctx.classes, asm_of, ctx.class_records, ctx.literal, ctx.signature)
    if isinstance(ctors, gen.Unsupported):
        result.detail = str(ctors)
        return result
    if ctors:
        for draft in ctors:
            result = _verify_class_draft(ctx, command, insns, draft, work, Result(name, sym.address, "unsupported"))
            if result.status == "exact":
                return result
        return result
    draft = classgen.solve_dtor(insns, ctx.classes, asm_of, ctx.class_records)
    if isinstance(draft, gen.Unsupported):
        result.detail = str(draft)
        return result
    if draft is not None:
        return _verify_class_draft(ctx, command, insns, draft, work, result)
    this_type, method, known_class = _display_names(ctx, name, sym.address)
    try:
        lifted = gen.lift(insns, ctx.literal, known_class, ctx.signature, ctx.vsignature(this_type if known_class else None))
        member = gen.Param("r3") in lifted.nodes or (known_class and "r3" not in lifted.int_params)
        # a member call on this names this's class: the scratch class must carry that name
        this_q = lifted.bound.get(("this",)) or lifted.bound.get(gen.Param("r3"))
        if this_q and known_class and "::".join(this_q) != this_type:
            raise gen.Unsupported(f"calls {'::'.join(this_q)} methods on this ({this_type})")
        if this_q:
            this_type, member = "::".join(this_q), True
        verify_type = this_type if this_q else "G"
        used = ctx.result_used(name)
        if used is not None and gen.return_value(lifted) is not None:
            lifted.void_first = not used          # the callers say whether a value comes back
        sizes = {g: ctx.by_name[g].size for g in lifted.global_objs if g in ctx.by_name}
        # a variant that cannot be written down (e.g. returning a branch-local value) is skipped
        chosen, first_error = [], None
        for v in gen.variants(lifted):
            try:
                gen.render_tu(lifted, member, verify_type, "v", [v], sizes)
                chosen.append(v)
            except gen.Unsupported as e:
                first_error = first_error or e
        if not chosen:
            raise first_error or gen.Unsupported("no variant can be written")
        source, _ = gen.render_tu(lifted, member, verify_type, "v", chosen, sizes)
    except gen.Unsupported as e:
        result.detail = str(e)
        return result
    result.compiler = next((p.split("/")[-2] for p in shlex.split(command) if p.endswith("mwcceppc.exe")), "")
    stem = f"f{sym.address:08X}"
    obj, error = _compile(ctx, command, source, work, stem)
    objects = [obj] * len(chosen)
    if obj is None:
        # one variant the renderer got wrong must not hide the others: compile each alone
        objects = []
        for i, v in enumerate(chosen):
            try:
                one, _ = gen.render_tu(lifted, member, verify_type, "v", [v], sizes)
            except gen.Unsupported:
                objects.append(None)
                continue
            objects.append(_compile(ctx, command, one, work, f"{stem}_{i}")[0])
        if not any(objects):
            result.status, result.detail = "error", f"generator bug (draft does not compile): {error}"
            return result
    first_diff, best = "", (-1, None)
    for i, v in enumerate(chosen):
        obj = objects[i]
        if obj is None:
            continue
        index = i if len(set(map(id, objects))) == 1 else 0
        sym_name = next((f for f in obj.functions() if re.match(rf"^v{index}(?:__|$)", f)), f"v{index}")
        diff, matched = gen.compare_detail(insns, obj, sym_name, ctx.literal, literal_info=ctx.literal_info)
        if diff is not None and matched > best[0]:
            best = (matched, v, diff)
        if diff is None:
            try:
                result.source, result.notes = gen.render_tu(lifted, member, this_type, method, [v], sizes, display=True)
            except gen.Unsupported as e:
                result.status, result.detail = "error", f"generator bug (display): {e}"
                return result
            result.status = "exact"
            result.notes += lifted.notes
            return result
        first_diff = first_diff or diff
    result.status, result.detail = "differs", f"no variant matched ({len(chosen)} tried): {first_diff}"
    if best[1] is not None:                     # the closest variant, for --unverified
        try:
            result.source, result.notes = gen.render_tu(lifted, member, this_type, method, [best[1]], sizes, display=True)
            result.notes += lifted.notes
            result.closest = f"{best[0]}/{len(insns)} instructions match, then: {best[2]}"
        except gen.Unsupported:
            result.source = ""
    return result


def _verify_class_draft(ctx: Context, command: str, insns, draft, work: Path, result: Result) -> Result:
    result.compiler = next((p.split("/")[-2] for p in shlex.split(command) if p.endswith("mwcceppc.exe")), "")
    obj, error = _compile(ctx, command, draft.source, work, f"c{result.address:08X}")
    if obj is None:
        result.status, result.detail = "error", f"generator bug (class draft does not compile): {error}"
        return result
    fn_symbol = draft.function
    if fn_symbol not in obj.functions():                  # a constructor: the name ends in its parameters
        fn_symbol = next((f for f in obj.functions() if f.startswith(draft.function)), draft.function)
    renames = dict(draft.renames)
    if "@base_ctor" in renames:                           # our symbol for the base constructor call
        callee = renames.pop("@base_ctor")
        index = next(i for i, insn in enumerate(insns) if insn.op == "bl" and insn.args[0] == callee)
        fn = obj.functions().get(fn_symbol)
        ours = next((r.symbol for r in (fn.relocs if fn else []) if r.offset & ~3 == 4 * index), None)
        if ours:
            renames[ours] = callee
            draft.advice.append(f"./dk rename {callee} {ours}")
    draft.function = fn_symbol
    diff = gen.compare(insns, obj, fn_symbol, ctx.literal, renames, ctx.literal_info)
    if diff is not None:
        result.status, result.detail = "differs", f"class draft: {diff}"
        return result
    result.status = "exact"
    result.source = draft.source
    result.notes = draft.notes + [f"then: {a}" for a in draft.advice + [f"./dk rename {result.name} {draft.function}"]
                                  if result.name != draft.function and gen.is_auto(result.name)] \
        if result.name != draft.function else draft.notes + [f"then: {a}" for a in draft.advice]
    return result


def solve_asm(ctx: Context, name: str, work: Path) -> Result:
    """A function the original wrote in assembly, as a CodeWarrior asm function, verified."""
    sym = ctx.by_name.get(name)
    if sym is None or name not in ctx.entries:
        return Result(name, 0, "skipped", "no such function in the asm")
    result = Result(name, sym.address, "unsupported")
    command, why = ctx.command(sym.address)
    if command is None:
        result.status, result.detail = "skipped", why
        return result
    section_of = lambda s: ctx.by_name[s].section if s in ctx.by_name else None
    try:
        source, defined = asmgen.asm_function(name, ctx.entries[name].text, section_of, ctx.literal)
    except gen.Unsupported as e:
        result.detail = str(e)
        return result
    result.compiler = next((p.split("/")[-2] for p in shlex.split(command) if p.endswith("mwcceppc.exe")), "")
    obj, err = _compile(ctx, command, source, work, re.sub(r"\W", "_", name)[:60] + "_asm")
    if obj is None:
        result.status, result.detail = "error", f"generator bug (assembly does not compile): {err}"
        return result
    # literals the draft defines must hold the original bytes; everything else it references by
    # its own name (the unit's data defines it), so the names must be equal
    own = lambda s, size, addend=0: ctx.literal(s, size, addend) if s in defined else None
    diff, matched = gen.compare_detail(gen.parse(ctx.entries[name].text), obj, name, own,
                                       literal_info=lambda s: ctx.literal_info(s) if s in defined else None)
    if diff:
        result.status, result.detail = "differs", diff
        result.closest = f"{matched} instructions match, then: {diff}"
        result.source = source
        return result
    result.status, result.source = "exact", source
    result.notes = ["hand-written assembly: use this only where the original was assembly"]
    return result


def solve_many(ctx: Context, names: list[str], jobs: int, asm: bool = False) -> list[Result]:
    root = ctx.cfg.root / "build" / "dk" / "gen"
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{os.getpid()}-", dir=root))
    solver = solve_asm if asm else solve
    try:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            return list(pool.map(lambda n: solver(ctx, n, work), names))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def format_draft(r: Result) -> str:
    if r.status != "exact":
        lines = [f"// dk gen: UNVERIFIED draft for {r.name} (closest variant, NOT exact: {r.closest})",
                 "// start from it, then fix the differences with ./dk check and ./dk diff"]
    else:
        lines = [f"// dk gen: exact draft for {r.name} (verified alone with GC/{r.compiler} and its unit's flags)"]
    if r.notes:
        lines.append("// layout it relies on (offsets exact; names and int/word types are placeholders):")
        lines += [f"//   {n}" for n in r.notes]
    return "\n".join(lines) + "\n" + r.source.rstrip()


def run(cfg: Config, args: argparse.Namespace) -> int:
    try:
        ctx = Context(cfg)
    except (OSError, MatchError, ValueError) as e:
        print(f"error: {e}")
        return USAGE
    ctx.max_insns = args.max_insns
    if args.regress:
        return _regress(ctx, args)
    if args.survey:
        return _survey(ctx, args)
    names = list(args.functions)
    if args.unit:
        try:
            unit = resolve_unit(cfg, args.unit)
        except MatchError as e:
            print(f"error: {e}")
            return USAGE
        rel = unit["metadata"]["source_path"].removeprefix(cfg.platform_options.get("src", "src") + "/")
        ranges = [(a, b) for u in ctx.splits if u.name == rel for sec, a, b in u.splits if sec == ".text"]
        names += [s.name for s in ctx.symbols if s.type == "function" and any(a <= s.address < b for a, b in ranges)]
    if not names:
        print("error: give functions, --unit or --survey")
        return USAGE
    results = solve_many(ctx, names, args.jobs, asm=args.asm)
    for r in results:
        if r.status == "exact" or args.unverified and r.status == "differs" and r.source:
            print(format_draft(r) + "\n")
        else:
            print(f"// {r.name}: no draft ({r.status}): {r.detail}\n")
    exact = sum(r.status == "exact" for r in results)
    print(f"{exact}/{len(results)} function(s) have an exact draft")
    if exact:
        hint("check <unit> after pasting")
    return OK if exact == len(results) else MISMATCH


_OLD_RUNNER = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools.dk.core import config
from tools.dk.platforms.gc_mwcc import gencmd
ctx = gencmd.Context(config.load(Path(sys.argv[2])))
ctx.max_insns = int(sys.argv[3])
names = json.loads(Path(sys.argv[4]).read_text())
print(json.dumps({r.name: r.status for r in gencmd.solve_many(ctx, names, int(sys.argv[5]))}))
"""


def _regress(ctx: Context, args: argparse.Namespace) -> int:
    """The generator at REV and the working tree's, on the same functions: nothing that was
    exact may stop being exact (functions that left the asm meanwhile are not counted).
    Named functions, or every survey function."""
    names = [n for n in args.functions if n in ctx.entries] if args.functions else _survey_names(ctx, args)
    root = ctx.cfg.root / "build" / "dk" / "gen"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        tree = Path(tmp)
        archive = subprocess.run(["git", "archive", args.regress, "tools"], cwd=ctx.cfg.root, capture_output=True)
        if archive.returncode != 0:
            print(f"error: git archive {args.regress}: {archive.stderr.decode().strip()}")
            return USAGE
        subprocess.run(["tar", "-x", "-C", str(tree)], input=archive.stdout, check=True)
        (tree / "names.json").write_text(json.dumps(names))
        proc = subprocess.run([sys.executable, "-c", _OLD_RUNNER, str(tree), str(ctx.cfg.root), str(ctx.max_insns),
                               str(tree / "names.json"), str(args.jobs)], cwd=ctx.cfg.root, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"error: the generator at {args.regress} failed:\n{proc.stderr[-2000:]}")
            return USAGE
        old = json.loads(proc.stdout.strip().splitlines()[-1])
    new = {r.name: r for r in solve_many(ctx, names, args.jobs)}
    lost = sorted(n for n, s in old.items() if s == "exact" and new[n].status not in ("exact", "skipped"))
    gained = sorted(n for n, r in new.items() if r.status == "exact" and old.get(n) != "exact")
    print(f"dk gen regress vs {args.regress}: {len(names)} functions, exact {sum(s == 'exact' for s in old.values())}"
          f" -> {sum(r.status == 'exact' for r in new.values())}; lost {len(lost)}, gained {len(gained)}")
    for n in lost:
        print(f"  LOST {n}: {new[n].status}: {new[n].detail}")
    return MISMATCH if lost else OK


def _survey_names(ctx: Context, args: argparse.Namespace) -> list[str]:
    names = []
    for name, entry in sorted(ctx.entries.items(), key=lambda kv: ctx.by_name[kv[0]].address if kv[0] in ctx.by_name else 0):
        sym = ctx.by_name.get(name)
        if sym is None or sym.type != "function":
            continue
        if args.include_linked != ctx.linked(ctx.unit_of(sym.address)):
            continue
        try:
            n = len(gen.strip_frame(gen.parse(entry.text)))
        except gen.Unsupported:
            n = len(gen.parse(entry.text))              # judged by its raw length
        if n <= args.max_insns:
            names.append(name)
    return names


def _survey(ctx: Context, args: argparse.Namespace) -> int:
    names = []
    for name, entry in sorted(ctx.entries.items(), key=lambda kv: ctx.by_name[kv[0]].address if kv[0] in ctx.by_name else 0):
        sym = ctx.by_name.get(name)
        if sym is None or sym.type != "function":
            continue
        if args.include_linked != ctx.linked(ctx.unit_of(sym.address)):
            continue
        try:
            n = len(gen.strip_frame(gen.parse(entry.text)))
        except gen.Unsupported:
            n = len(gen.parse(entry.text))              # judged by its raw length
        if n <= args.max_insns:
            names.append(name)
    results = solve_many(ctx, names, args.jobs)
    by_status = Counter(r.status for r in results)
    scope = "functions in linked units (self-test)" if args.include_linked else "functions outside linked units"
    print(f"dk gen survey: {len(results)} {scope} with <= {args.max_insns} instructions")
    for status in ("exact", "differs", "unsupported", "thunk", "skipped", "error"):
        print(f"  {status:<12} {by_status.get(status, 0)}")
    for status in ("unsupported", "differs", "skipped", "error"):
        reasons = Counter(re.sub(r"0x[0-9A-F]+|\b\d+\b|\b\w+_[0-9A-F]{8}\b", "#", r.detail)
                          for r in results if r.status == status)
        if reasons:
            print(f"{status} reasons:")
            for reason, n in reasons.most_common(args.reasons):
                print(f"  {n:5}  {reason}")
    if args.show:
        for r in results:
            if r.status == args.show:
                print(f"\n// {r.name}: {r.detail}" if r.status != "exact" else "\n" + format_draft(r))
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("functions", nargs="*", help="function symbols")
    p.add_argument("--unit", help="every function of this declared unit")
    p.add_argument("--survey", action="store_true", help="dry run over every small function outside linked units")
    p.add_argument("--include-linked", action="store_true", help="survey linked units instead (self-test)")
    p.add_argument("--max-insns", type=int, default=MAX_INSNS, help="instruction limit (body, without the frame)")
    p.add_argument("--reasons", type=int, default=12, help="survey: how many reasons to list per status")
    p.add_argument("--show", choices=("exact", "differs", "unsupported", "error", "skipped", "thunk"),
                   help="survey: also print each result with this status")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="parallel compiles")
    p.add_argument("--unverified", action="store_true",
                   help="also print the closest NOT-exact draft of a function that lifts but does not match")
    p.add_argument("--asm", action="store_true",
                   help="the function as a CodeWarrior asm function (only for code the original wrote in assembly)")
    p.add_argument("--regress", metavar="REV", help="compare with the generator at REV on the same functions (exit 1 if a draft is lost)")


COMMAND = Command("gen", "matching", "verified C++ drafts for small functions (exact-only; --survey for a dry run)",
                  run, _configure)
