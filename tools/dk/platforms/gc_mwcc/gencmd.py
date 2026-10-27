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
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ...core.command import BUILD_ERROR, MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from . import asmgen, asmindex, classgen, gen, genmerge
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
    standalone: str = ""        # exact: a self-contained TU under the function's own names (--write)
    symbol: str = ""            # the standalone TU's symbol for the function, when known
    renames: dict = field(default_factory=dict)     # applied when comparing (class drafts)
    caller_evidence: bool = False                   # a call took more arguments from other callers' setup
    alternatives: list = field(default_factory=list)    # standalone TUs of the other exact variants


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
        # library ranges built with a known linked unit's flags: [(start, end, unit)]
        self.templates = [(int(t["start"]), int(t["end"]), t["unit"]) for t in settings.get("templates", [])]
        self._commands: dict[str, str] = {}
        self._command_lock = threading.Lock()      # compile_command runs ninja under a process lock
        self._signatures: dict[str, tuple[int, int] | None] = {}
        self._signature_lock = threading.RLock()
        self.max_insns = MAX_INSNS
        self._callers: dict | None = None
        self._caller_args: dict = {}
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

    def base_path(self, derived: str, base: str) -> list[str] | None:
        """[base, ..., derived]: `derived`'s classes from `base` on, along the bases RTTI records
        at offset 0 (so `this` passes unchanged), or None when `base` is not one of them. The
        classes in between come only from a provable single chain, else [base, derived]."""
        by_rtti = {c["rtti"]: c["name"] for c in self.class_records}
        zero = lambda name: [by_rtti.get(b) for b, off in self.classes[name]["bases"] if off == 0] \
            if name in self.classes else None
        mine = zero(derived)
        if not mine or base not in mine or None in mine:
            return None
        # a single chain: the i-th class (root first) has exactly the i classes before it as bases
        chain = sorted(mine, key=lambda n: len(zero(n) or ()))
        if any(set(zero(n) or ()) != set(chain[:i]) for i, n in enumerate(chain)):
            return [base, derived]
        return chain[chain.index(base):] + [derived]

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

    def caller_args(self, name: str) -> tuple[int, int] | None:
        """The most (GPR, FPR) argument registers any direct call to `name` sets up: written
        on the path into the bl and read by nothing first (varargs calls aside)."""
        with self._signature_lock:
            if self._callers is None:
                self._build_callers()
        return self._caller_args.get(name)

    def _build_callers(self) -> None:
        callers: dict = {}
        args: dict = {}
        for entry in self.entries.values():
            insns = gen.parse(entry.text)
            targets = None
            for i, insn in enumerate(insns):
                if insn.op not in ("bl", "b") or not insn.args or insn.args[0].startswith(".L"):
                    continue
                if insn.op == "bl":
                    callers.setdefault(insn.args[0], []).append(insns[i + 1:])
                if i and insns[i - 1].op in ("crclr", "crset"):
                    continue                            # varargs: each call passes its own count
                targets = targets if targets is not None else {t for t in map(gen._target, insns) if t is not None}
                g, f = gen.set_before(insns, i, "r", targets) - 2, gen.set_before(insns, i, "f", targets)
                old = args.get(insn.args[0], (0, 0))
                args[insn.args[0]] = (max(old[0], g), max(old[1], f))
        self._callers, self._caller_args = callers, args

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

    def template_for(self, address: int) -> str | None:
        """The linked unit whose compile command verifies an undeclared function here."""
        ranged = next((u for a, b, u in self.templates if a <= address < b), None)
        return ranged or (self.template if address < self.code_end else None)

    def command(self, address: int) -> tuple[str | None, str]:
        """(compile command, why not) for a function at `address`."""
        unit = self.unit_of(address)
        if unit is None:
            unit = self.template_for(address)
            if unit is None:
                return None, "library code outside a declared unit (use the reference decomps)"
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
    fn = _demangled(name)
    if fn is not None and fn.quals:                 # a nested class (Q2AI7CarBase): its member
        return "::".join(fn.quals), _mangled_prefix(name), True
    _, owner = class_slots(ctx.cfg, address)
    if owner:
        return owner, name, True
    return f"Unk{address:08X}", name, False


def _own_params(ctx: Context, name: str) -> tuple[int, int] | None:
    """(GPR, FPR) parameters a function has: from its mangled name (with `this`), else the
    most its direct callers pass, else None."""
    from .demangle import demangle
    fn = demangle(name) if "__" in name.lstrip("_") else None
    if fn is not None and not any(t.reg == "none" for t in fn.params):
        gpr = sum(t.reg == "gpr" for t in fn.params) + bool(fn.quals)
        return gpr, sum(t.reg == "fpr" for t in fn.params)
    return ctx.caller_args(name)


def _compile(ctx: Context, command: str, source: str, work: Path, stem: str, include_dir: Path | None = None,
             preprocess: bool = False, overlay: Path | None = None) -> tuple[Elf | str | None, str]:
    """(object, "") or (None, the first error). `include_dir`: where the real source sits (its
    quoted includes); `preprocess`: (the preprocessed text, "") instead of an object."""
    src = work / f"{stem}.cpp"
    src.write_text(source)
    args = retarget(command, work)
    i = args.index("-c")
    args[i + 1] = str(src)
    if include_dir is not None:
        args[i:i] = ["-i", str(include_dir)]
        i += 2
    if overlay is not None and overlay.is_dir():        # searched before every other include path
        first = args.index("-i")
        args[first:first] = ["-i", str(overlay)]
        i += 2
    if preprocess:
        args[i] = "-E"
        proc = subprocess.run(args, cwd=ctx.cfg.root, capture_output=True, text=True, timeout=60)
        return (proc.stdout, "") if proc.returncode == 0 else (None, (proc.stdout + proc.stderr).strip()[-300:])
    proc = subprocess.run(args, cwd=ctx.cfg.root, capture_output=True, text=True, timeout=60)
    obj = work / f"{stem}.o"
    if proc.returncode != 0 or not obj.is_file():
        text = [l for l in (proc.stdout + proc.stderr).splitlines()
                if l.strip() and "Usage Warning" not in l and "Specified directory" not in l]
        # mwcc: "#   Error:   ^" under the source line, then the message on the next line
        at = next((i for i, l in enumerate(text) if "rror" in l), None)
        if at is None:
            return None, text[-1].strip() if text else "compile failed"
        message = ""
        if text[at].rstrip().endswith("^"):                 # the message, and its continuation line
            more = [l.lstrip("# ").strip() for l in text[at + 1:at + 3] if l.startswith("#   ") and not l.startswith("#    ")]
            message = " ".join(more[:1] + [m for m in more[1:2] if not re.match(r"\d+:", m)])
        return None, f"{text[at].strip().rstrip('^ ')} {message}".strip()
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
    plan = classgen.plan_ctor(name, sym.address, insns, ctx.classes, asm_of, ctx.class_records, ctx.literal,
                              ctx.signature, ctx.caller_args)
    modelled = None
    if isinstance(plan, classgen.CtorPlan):       # the class model first; the older field-only one after
        modelled = _verify_plan(ctx, command, insns, plan, work, Result(name, sym.address, "unsupported"))
        if modelled.status == "exact":
            return modelled
    elif isinstance(plan, gen.Unsupported):
        modelled = Result(name, sym.address, "unsupported", str(plan))
    ctors = classgen.solve_ctor(insns, ctx.classes, asm_of, ctx.class_records, ctx.literal, ctx.signature)
    if isinstance(ctors, gen.Unsupported):
        result.detail = str(ctors)
        return modelled if modelled is not None and modelled.status != "unsupported" else result
    if ctors:
        for draft in ctors:
            result = _verify_class_draft(ctx, command, insns, draft, work, Result(name, sym.address, "unsupported"))
            if result.status == "exact":
                return result
        return modelled if modelled is not None and modelled.status != "unsupported" else result
    draft = classgen.solve_dtor(insns, ctx.classes, asm_of, ctx.class_records)
    plan = classgen.plan_dtor(name, sym.address, insns, ctx.classes, asm_of, ctx.class_records, ctx.literal,
                              ctx.signature, ctx.caller_args)
    if plan is not None:                                 # a destructor: the class model, else the vtable-only one
        if isinstance(plan, classgen.DtorPlan):
            result = _verify_plan(ctx, command, insns, plan, work, result)
            if result.status == "exact":
                return result
        else:
            result.detail = str(plan)
        if isinstance(draft, classgen.ClassDraft):
            old = _verify_class_draft(ctx, command, insns, draft, work, Result(name, sym.address, "unsupported"))
            if old.status == "exact" or result.status == "unsupported":
                return old
        return result
    if isinstance(draft, gen.Unsupported):
        result.detail = str(draft)
        return result
    if draft is not None:
        return _verify_class_draft(ctx, command, insns, draft, work, result)
    own = _demangled(name)
    if own is not None and own.params:
        # the parameter types of the mangled name first: the draft then has the function's own
        # symbol, and its class pointers bound to their classes (as the unit's headers have them)
        typed = _solve_lifted(ctx, name, sym, insns, command, work, Result(name, sym.address, "unsupported"),
                              ctx.caller_args, own=own)
        if typed.status == "exact":
            return typed
    first = _solve_lifted(ctx, name, sym, insns, command, work, Result(name, sym.address, "unsupported"), ctx.caller_args)
    if first.status == "exact":
        return first
    if first.caller_evidence:
        # other callers' argument registers made this call longer: the shorter reading may match
        second = _solve_lifted(ctx, name, sym, insns, command, work, Result(name, sym.address, "unsupported"), None)
        if second.status == "exact":
            return second
    if first.status == "differs" and "(stwu)" in first.detail:
        # the frame size differs: the struct locals may be smaller than the space they span
        tight = _solve_lifted(ctx, name, sym, insns, command, work, Result(name, sym.address, "unsupported"),
                              ctx.caller_args, tight=True)
        if tight.status == "exact":
            return tight
    return first


def _global_sizes(ctx: Context, lifted) -> dict:
    """The size of each global a draft uses (a small one without a size in the symbols: its
    string's, read from the binary)."""
    out = {}
    for g in set(lifted.global_objs) | set(lifted.small_globals):
        s = ctx.by_name.get(g)
        if s is None:
            continue
        out[g] = s.size
        if (not s.size or s.size > 8) and g in lifted.small_globals:       # (a symbol may hold several strings)
            try:
                data = ctx.dol.read(s.address, 8)
            except ValueError:
                continue
            if b"\0" in data and all(32 <= b < 127 for b in data[:data.index(b"\0")]):
                out[g] = data.index(b"\0") + 1
    return out


def _mangled_prefix(name: str) -> str:
    """The function-name part of a mangled name (`fn_X` of fn_X__Q22AI7CarBaseFv, `__ct`)."""
    return name[:name.index("__", 2 if name.startswith("__") else 0)]


def _cxx_free(name: str) -> str | None:
    """A free C++ function's name (fn_X of fn_X__FPCc), else None."""
    fn = _demangled(name)
    return _mangled_prefix(name) if fn is not None and not fn.quals and not fn.special else None


def _demangled(name: str):
    """The demangled function (None for a C name or one the demangler does not read)."""
    from .demangle import demangle
    return demangle(name) if "__" in name.lstrip("_") else None


def _solve_lifted(ctx: Context, name: str, sym, insns, command: str, work: Path, result: Result, callers,
                  tight: bool = False, own=None) -> Result:
    """The lifted (not class-model) draft of one function, verified; `callers` gives other
    call sites' argument counts (None: only this function's own evidence); `tight`: struct
    locals as small as their fields; `own`: its demangled name, whose parameter types it keeps."""
    this_type, method, known_class = _display_names(ctx, name, sym.address)
    free = _cxx_free(name)
    if free is not None:
        if own is None:
            # as extern "C" it would carry the mangled name as a C name: a duplicate of the real one
            result.detail = "a C++ function: drafted only with its mangled parameter types"
            return result
        method = free
    if method == "__dt":
        result.detail = "a destructor without a vtable (member destructors need the class model)"
        return result
    try:
        lifted = gen.lift(insns, ctx.literal, known_class, ctx.signature, ctx.vsignature(this_type if known_class else None),
                          callers=callers, params=_own_params(ctx, name), own=own)
        result.caller_evidence = lifted.caller_evidence
        lifted.polymorphic = {c.new.cxx.quals for c in lifted.calls if c.new is not None
                              and "::".join(c.new.cxx.quals) in ctx.classes}
        if tight:
            sized = gen.tight_locals(lifted)
            if sized == lifted.locals:
                raise gen.Unsupported("no smaller local layout")
            lifted.locals = sized
        member = gen.Param("r3") in lifted.nodes or (known_class and "r3" not in lifted.int_params)
        # a member call on this names this's class: the scratch class must carry that name
        this_q = lifted.bound.get(("this",)) or lifted.bound.get(gen.Param("r3"))
        if this_q and not known_class and gen.Param("r3") not in lifted.receivers and not lifted.nodes.get(gen.Param("r3")) \
                and not lifted.vptrs.get(gen.Param("r3")):
            this_q, member = None, False          # only passed on as a class pointer: a parameter, not this
        chain = None                              # [base, ..., this_type]: inherited methods called on this
        if lifted.this_classes:                   # several classes' methods on this: all of this_type's chain
            if not known_class or not this_q:
                raise gen.Unsupported("one object used as two classes")
            names = {"::".join(q) for q in lifted.this_classes | {this_q}} - {this_type}
            paths = {n: ctx.base_path(this_type, n) for n in names}
            if any(p is None for p in paths.values()):
                raise gen.Unsupported(f"calls {', '.join(sorted(n for n, p in paths.items() if p is None))} "
                                      f"methods on this ({this_type})")
            root = max(paths, key=lambda n: len(paths[n]))
            if any(n not in paths[root] for n in names):
                raise gen.Unsupported("classes called on this that are not one inheritance chain")
            this_q = tuple(root.split("::"))
        if this_q and known_class and "::".join(this_q) != this_type:
            chain = ctx.base_path(this_type, "::".join(this_q))
            if chain is None:
                raise gen.Unsupported(f"calls {'::'.join(this_q)} methods on this ({this_type})")
        if this_q:
            this_type, member = "::".join(this_q), True
        verify_type = this_type if this_q else "G"
        used = ctx.result_used(name)
        if used is not None and gen.return_value(lifted) is not None:
            lifted.void_first = not used          # the callers say whether a value comes back
        sizes = _global_sizes(ctx, lifted)
        # a variant that cannot be written down (e.g. returning a branch-local value) is skipped
        chosen, first_error = [], None
        for v in gen.variants(lifted):
            try:
                gen.render_tu(lifted, member, verify_type, "v", [v], sizes, chain=chain, cxx_free=free is not None)
                chosen.append(v)
            except gen.Unsupported as e:
                first_error = first_error or e
        if not chosen:
            raise first_error or gen.Unsupported("no variant can be written")
        source, _ = gen.render_tu(lifted, member, verify_type, "v", chosen, sizes, chain=chain, cxx_free=free is not None)
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
                one, _ = gen.render_tu(lifted, member, verify_type, "v", [v], sizes, chain=chain, cxx_free=free is not None)
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
                result.source, result.notes = gen.render_tu(lifted, member, this_type, method, [v], sizes, display=True, chain=chain, cxx_free=free is not None)
            except gen.Unsupported as e:
                result.status, result.detail = ("unsupported", str(e)) if method == "__ct" else \
                    ("error", f"generator bug (display): {e}")
                return result
            try:
                result.standalone, _ = gen.render_tu(lifted, member, this_type, method, [v], sizes, standalone=True, chain=chain, cxx_free=free is not None)
            except gen.Unsupported:
                pass                            # --write leaves it out
            if method == "__ct":                 # verify the constructor form under its own symbol
                obj_c, error = _compile(ctx, command, result.standalone, work, f"k{sym.address:08X}")
                diff_c = error if obj_c is None else \
                    gen.compare(insns, obj_c, name, ctx.literal, literal_info=ctx.literal_info)
                if diff_c is not None:
                    result.status, result.detail = "differs", f"exact only as a plain function; as a constructor: {diff_c}"
                    return result
            result.status = "exact"
            result.notes += lifted.notes
            # the other exact variants (another return type, statement order...): --merge falls
            # back on them when the unit's headers declare the function differently
            for j in range(i + 1, len(chosen)):
                obj_j = objects[j]
                if obj_j is None:
                    continue
                index = j if len(set(map(id, objects))) == 1 else 0
                sym_j = next((f for f in obj_j.functions() if re.match(rf"^v{index}(?:__|$)", f)), f"v{index}")
                if gen.compare(insns, obj_j, sym_j, ctx.literal, literal_info=ctx.literal_info) is None:
                    try:
                        alt, _ = gen.render_tu(lifted, member, this_type, method, [chosen[j]], sizes, standalone=True,
                                               chain=chain, cxx_free=free is not None)
                    except gen.Unsupported:
                        continue
                    if alt != result.standalone and alt not in result.alternatives:
                        result.alternatives.append(alt)
            return result
        first_diff = first_diff or diff
    result.status, result.detail = "differs", f"no variant matched ({len(chosen)} tried): {first_diff}"
    if best[1] is not None:                     # the closest variant, for --unverified
        try:
            result.source, result.notes = gen.render_tu(lifted, member, this_type, method, [best[1]], sizes, display=True, chain=chain, cxx_free=free is not None)
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
    result.source = result.standalone = draft.source
    result.symbol, result.renames = fn_symbol, renames
    result.notes = draft.notes + [f"then: {a}" for a in draft.advice + [f"./dk rename {result.name} {draft.function}"]
                                  if result.name != draft.function and gen.is_auto(result.name)] \
        if result.name != draft.function else draft.notes + [f"then: {a}" for a in draft.advice]
    return result


def _verify_plan(ctx: Context, command: str, insns, plan, work: Path, result: Result) -> Result:
    """Each (void) variant of a constructor's or destructor's body in its class model, until
    one is exact."""
    lifted = plan.lifted
    sizes = _global_sizes(ctx, lifted)
    sources, first_error = [], None
    for v in gen.variants(lifted):
        if v.ret_mode != "void":
            continue
        try:
            if isinstance(plan, classgen.CtorPlan):
                source, _ = gen.render_tu(lifted, True, plan.model.cls, "__ct", [v], sizes, standalone=True,
                                          ctor_model=plan.model)
            else:
                source, _ = gen.render_tu(lifted, True, plan.model.cls, "__dt", [v], sizes, standalone=True, dtor=plan.model)
        except gen.Unsupported as e:
            first_error = first_error or e
            continue
        if source not in sources:
            sources.append(source)
    if not sources:
        what = "constructor" if isinstance(plan, classgen.CtorPlan) else "destructor"
        result.status, result.detail = "unsupported", f"{what}: {first_error or 'no variant can be written'}"
        return result
    first = None
    for source in sources:
        draft = classgen.ClassDraft(source, plan.function, dict(plan.renames), list(plan.advice), list(plan.notes))
        r = _verify_class_draft(ctx, command, insns, draft, work, Result(result.name, result.address, "unsupported"))
        if r.status == "exact":
            r.notes += lifted.notes
            return r
        first = first or r
    return first


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
    result.standalone, result.symbol = source, name
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
    if args.harvest is not None:
        return _harvest(ctx, args)
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
    if args.write:
        if not (args.unit or args.into) or args.asm:
            print("error: --write needs --unit or --into (and no --asm)")
            return USAGE
        return _write(ctx, args, results, cfg.root / unit["metadata"]["source_path"] if args.unit else Path(args.into),
                      unit if args.unit else None)
    if args.merge:
        if not args.unit or args.asm:
            print("error: --merge needs --unit (and no --asm)")
            return USAGE
        return _merge(ctx, args, results, unit)
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
results = gencmd.solve_many(ctx, names, int(sys.argv[5]))
Path(sys.argv[4]).with_suffix(".drafts.json").write_text(json.dumps(
    {r.name: getattr(r, "standalone", "") for r in results if r.status == "exact"}))
print(json.dumps({r.name: r.status for r in results}))
"""


def _symbol_in(obj: Elf, ctx: Context, r: Result) -> str | None:
    """Our symbol for a draft's function: given, or its name (mangled as a member)."""
    if r.symbol:
        return r.symbol
    _, method, _ = _display_names(ctx, r.name, r.address)
    found = [f for f in obj.functions() if f == method or f.startswith(method + "__")]
    return found[0] if len(found) == 1 else None


def _merged_failures(ctx: Context, command: str, unit: genmerge.Unit, drafts: list, work: Path) -> tuple[str, dict]:
    """(merged source, draft name -> why it is not exact in it) for the drafts in `unit`."""
    source = unit.render([])
    obj, error = _compile(ctx, command, source, work, "merged")
    if obj is None:
        return source, {r.name: f"merged source does not compile: {error}" for r, _ in drafts}
    failures = {}
    for r, sym in drafts:
        diff = gen.compare(gen.parse(ctx.entries[r.name].text), obj, sym, ctx.literal, r.renames or None, ctx.literal_info)
        if diff:
            failures[r.name] = diff
    return source, failures


def _has_code(path: Path) -> bool:
    return path.is_file() and any(l.strip() and not l.startswith(("//", "#include")) for l in path.read_text().splitlines())


def _write(ctx: Context, args: argparse.Namespace, results: list[Result], unit_source: Path, unit: dict | None = None) -> int:
    """--write: the unit's exact drafts merged into one source, every function verified again
    in it (see genmerge.py). Writes the unit's source when it is still empty, else --into.
    Classes a header under include/ defines come from that header (#include), as --merge
    does with a source that has code."""
    target = Path(args.into) if args.into else unit_source
    if not args.into and _has_code(target):
        print(f"error: {ctx.cfg.rel(target)} already has code: --into PATH writes the merged drafts elsewhere")
        return USAGE
    for r in results:
        if r.status == "exact" and r.name.startswith("__sinit_"):        # (see _merge)
            print(f"  left out {r.name}: compiler-generated (write the globals it initializes, see the draft: ./dk gen {r.name})")
    results = [r for r in results if not r.name.startswith("__sinit_")]
    drafts = [r for r in results if r.status == "exact" and r.standalone]
    headers = _class_headers(ctx, drafts) if unit is not None else {}
    if headers:
        own = [l for l in (unit_source.read_text().splitlines() if unit_source.is_file() else []) if l.startswith("#include")]
        rel = ctx.cfg.rel(unit_source)
        includes = list(dict.fromkeys(own + [f'#include "{h}"' for h in sorted(set(headers.values()))]))
        start = "\n".join([f"// {rel}: exact drafts merged by ./dk gen --write (placeholder types and names)", ""]
                          + includes + [""]) + "\n"
        print(f"classes from headers: {', '.join(f'{n} ({h})' for n, h in sorted(headers.items()))}")
        return _merge(ctx, args, results, unit, start_source=start, out=target)
    if not drafts:
        print("no exact drafts to write")
        return MISMATCH
    command, _ = ctx.command(drafts[0].address)
    root = ctx.cfg.root / "build" / "dk" / "gen"
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{os.getpid()}-w-", dir=root))
    dropped = [(r.name, "its draft cannot be written under its own names")
               for r in results if r.status == "exact" and not r.standalone]

    def alone(r: Result):
        """Each draft under its own names must still be exact on its own."""
        obj, error = _compile(ctx, command, r.standalone, work, f"s{r.address:08X}")
        if obj is None:
            return r, None, f"does not compile under its own names: {error}"
        sym = _symbol_in(obj, ctx, r)
        if sym is None:
            return r, None, "its function is not found under its own name"
        diff = gen.compare(gen.parse(ctx.entries[r.name].text), obj, sym, ctx.literal, r.renames or None, ctx.literal_info)
        return r, sym, diff
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            checked = list(pool.map(alone, sorted(drafts, key=lambda r: r.address)))
        ok = []
        for r, sym, why in checked:
            (dropped.append((r.name, why)) if why else ok.append((r, sym)))
        # all at once first; on any failure, greedily in address order
        unit, kept = genmerge.Unit(), []
        for r, sym in ok:
            try:
                unit = unit.add(r.standalone, r.address)
                kept.append((r, sym))
            except genmerge.Conflict as e:
                dropped.append((r.name, f"conflicts with an earlier draft: {e}"))
        source, failures = _merged_failures(ctx, command, unit, kept, work)
        if failures:
            unit, greedy = genmerge.Unit(), []
            for r, sym in kept:
                try:
                    trial = unit.add(r.standalone, r.address)
                except genmerge.Conflict as e:
                    dropped.append((r.name, f"conflicts with an earlier draft: {e}"))
                    continue
                _, fails = _merged_failures(ctx, command, trial, greedy + [(r, sym)], work)
                if fails:
                    dropped.append((r.name, "not exact once merged: " + "; ".join(f"{n}: {d}" for n, d in fails.items())))
                    continue
                unit, greedy = trial, greedy + [(r, sym)]
            kept = greedy
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rel = ctx.cfg.rel(unit_source)
    header = [f"// {rel}: {len(kept)} exact drafts merged by ./dk gen --write (placeholder types and names)", ""]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(unit.render(header))
    print(f"wrote {ctx.cfg.rel(target)}: "
          f"{len(kept)}/{len(drafts)} exact drafts, each verified exact in the merged source")
    for name, why in dropped:
        print(f"  left out {name}: {why}")
    shadowed = _header_classes(ctx, unit)
    for name, path in shadowed:
        print(f"  note: {name} is also defined in {path}: use that class instead of the placeholder")
    if args.unit:
        hint(f"rename --unit {args.unit}, then check {args.unit}")
    return OK


def _merge(ctx: Context, args: argparse.Namespace, results: list[Result], unit: dict,
           start_source: str | None = None, out: Path | None = None) -> int:
    """--merge: the exact drafts of functions the unit's source does not define yet, inserted
    into it (genmerge.Paste). All at once first; on any failure greedily in address order. A
    draft stays only if the merged source compiles, it is exact there, and no function of
    the unit scores lower than before (objdiff). Members a draft needs from a class of the
    unit's own header (same stem) are added to it, inside its padding, if that lets more
    drafts in. `start_source`/`out`: --write starts from the headers it includes."""
    path = ctx.cfg.root / unit["metadata"]["source_path"]
    source = start_source if start_source is not None else path.read_text() if path.is_file() else ""
    # a static initializer is compiler-generated: written by hand it has no .ctors entry (and a
    # unit with initialized globals already emits it), so its draft is only shown
    sinits = [r for r in results if r.status == "exact" and r.name.startswith("__sinit_") and start_source is None]
    results = [r for r in results if not r.name.startswith("__sinit_")]
    drafts = sorted((r for r in results if r.status == "exact" and r.standalone), key=lambda r: r.address)
    for r in sinits:
        print(f"  left out {r.name}: compiler-generated (write the globals it initializes, see the draft: ./dk gen {r.name})")
    if not drafts:
        print("no exact drafts to merge")
        return MISMATCH
    command, why = ctx.command(drafts[0].address)
    if command is None:
        print(f"error: {why}")
        return USAGE
    root = ctx.cfg.root / "build" / "dk" / "gen"
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{os.getpid()}-m-", dir=root))
    overlay = work / "overlay"                      # grown headers, searched before include/
    out = out or (Path(args.into) if args.into else path)
    grown: dict = {}
    try:
        first = _paste_attempt(ctx, args, unit, command, source, drafts, work, overlay)
        if isinstance(first, int):
            return first
        best = first
        base, plan = _grow_headers(ctx, path, first.missing, first.probe)
        if plan:
            for rel, text in plan.items():
                (overlay / rel).parent.mkdir(parents=True, exist_ok=True)
                (overlay / rel).write_text(text)
            second = _paste_attempt(ctx, args, unit, command, source, drafts, work, overlay)
            if not isinstance(second, int) and len(second.put) > len(first.put):
                best, grown = second, plan
        if best.put:
            if out == path and start_source is None and (path.read_text() if path.is_file() else "") != source:
                print(f"error: {ctx.cfg.rel(path)} changed while merging: run again")
                return USAGE
            if out == path and start_source is not None and _has_code(path):
                print(f"error: {ctx.cfg.rel(path)} got code while writing: run --merge instead")
                return USAGE
            for rel, text in grown.items():
                dest = (out.parent / f"{out.name}.include" if args.into else ctx.cfg.root / "include") / rel
                if not args.into and dest.read_text() != base[rel]:
                    print(f"error: include/{rel} changed while merging: run again")
                    return USAGE
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(text)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(best.text)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"{ctx.cfg.rel(out)}: merged {len(best.put)}/{len(best.ok)} exact drafts into {ctx.cfg.rel(path)}, each exact "
          f"in the merged unit (no function scores lower)")
    for rel in grown:
        where = f"{ctx.cfg.rel(out)}.include/{rel}" if args.into else f"include/{rel}"
        print(f"  grew {where}: members the drafts need, inside its padding (layout unchanged)")
    for r, _ in best.put:
        print(f"  put in {r.name}")
    for name, why in best.dropped:
        print(f"  left out {name}: {why}")
    if best.put and not args.into:
        hint(f"rename --unit {args.unit}, then check {args.unit}")
    return OK


@dataclass
class _Pasted:
    ok: list                    # (Result, symbol) exact alone, not in the source yet
    put: list                   # (Result, symbol) exact in the merged text
    dropped: list               # (name, why)
    missing: list               # genmerge.Missing raised by the drafts left out
    text: str
    probe: object


def _paste_attempt(ctx: Context, args: argparse.Namespace, unit: dict, command: str, source: str, drafts: list,
                   work: Path, overlay: Path):
    """One --merge pass against the headers as they are (overlay first): _Pasted, or an exit code."""
    from .matching import score
    path = ctx.cfg.root / unit["metadata"]["source_path"]
    target = ctx.cfg.root / unit["target_path"]
    counter = iter(range(1 << 30))
    dropped, missing = [], []

    def build(text: str):
        """(object path, Elf) of the unit's source as `text`, or (None, error)."""
        stem = f"u{next(counter)}"
        obj, error = _compile(ctx, command, text, work, stem, include_dir=path.parent, overlay=overlay)
        return (work / f"{stem}.o", obj) if obj is not None else (None, error)
    base_path, base = build(source)
    if base_path is None:
        print(f"error: {ctx.cfg.rel(path)} does not compile as it is: {base}")
        return BUILD_ERROR
    before = {f.name: f.percent for f in score(ctx.cfg, unit["name"], target, base_path).functions}
    preprocessed, error = _compile(ctx, command, source, work, "pre", include_dir=path.parent, overlay=overlay,
                                   preprocess=True)
    if preprocessed is None:
        print(f"error: cannot preprocess {ctx.cfg.rel(path)}: {error}")
        return BUILD_ERROR
    have = set(base.functions())

    def alone(r: Result):
        obj, error = _compile(ctx, command, r.standalone, work, f"s{r.address:08X}")
        if obj is None:
            return r, None, f"does not compile under its own names: {error}"
        sym = _symbol_in(obj, ctx, r)
        if sym is None:
            return r, None, "its function is not found under its own name"
        if sym in have or r.name in have:
            return r, None, "already in the source"
        diff = gen.compare(gen.parse(ctx.entries[r.name].text), obj, sym, ctx.literal, r.renames or None, ctx.literal_info)
        return r, sym, diff
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        checked = list(pool.map(alone, drafts))
    ok = []
    for r, sym, why in checked:
        if why == "already in the source":
            continue
        (dropped.append((r.name, why)) if why else ok.append((r, sym)))
    probe = _Probe(ctx, command, source, path, work, overlay)
    if not ok:
        return _Pasted(ok, [], dropped, missing, source, probe)
    lines = source.splitlines()
    addresses = {f: _address_of(ctx, f) for f in have}
    keys = {genmerge.definition_key(f): a for f, a in addresses.items() if a is not None and genmerge.definition_key(f)}
    start = genmerge.Paste.of(source, preprocessed, genmerge.function_starts(lines, keys), probe.members)

    def verify(paste: genmerge.Paste, put: list) -> tuple[str, dict]:
        """(merged text, draft or "@unit" -> why it fails there)."""
        text = paste.render()
        obj_path, obj = build(text)
        if obj_path is None:
            return text, {"@unit": f"the merged source does not compile: {obj}"}
        fails, defined = {}, obj.functions()
        for r, sym in put:
            sym = r.name if r.name in defined else sym      # (a head the header declares names it fully)
            diff = gen.compare(gen.parse(ctx.entries[r.name].text), obj, sym, ctx.literal, r.renames or None, ctx.literal_info)
            if diff:
                fails[r.name] = f"not exact once merged: {diff}"
        after = {f.name: f.percent for f in score(ctx.cfg, unit["name"], target, obj_path).functions}
        lower = [f"{f} {p:.1f}% -> {after.get(f, 0.0):.1f}%" for f, p in before.items() if after.get(f, 0.0) < p]
        if lower:
            fails["@unit"] = "lowers " + ", ".join(lower[:3])
        return text, fails

    def conflict(r: Result, e: genmerge.Conflict) -> None:
        dropped.append((r.name, f"conflicts: {e}"))
        if isinstance(e, genmerge.Missing):
            missing.append(e)
    alone_dropped = list(dropped)
    paste, put = start, []
    for r, sym in ok:
        try:
            paste = paste.add(r.standalone, r.address)
            put.append((r, sym))
        except genmerge.Conflict as e:
            conflict(r, e)
    text, fails = verify(paste, put)
    if fails or any(r.alternatives for r, s_ in ok if (r, s_) not in put):
        # one by one, in address order; a draft that fails goes in as another exact variant
        # (say the void one, when a header declares the function void) if one fits
        dropped[:], missing[:] = alone_dropped, []
        paste, kept, text = start, [], source
        for r, sym in ok:
            first = None
            for candidate in [r.standalone] + r.alternatives[:3]:
                try:
                    trial = paste.add(candidate, r.address)
                except genmerge.Conflict as e:
                    first = first or e
                    continue
                trial_text, trial_fails = verify(trial, kept + [(r, sym)])
                if trial_fails:
                    why = "; ".join(trial_fails.values())
                    if re.search(r"' redefined", why):
                        why += " (a header defines it already: an inline function is emitted where the unit uses it)"
                    elif re.search(r"' redeclared", why):
                        why += " (a header declares it with another signature)"
                    first = first or why
                    continue
                paste, kept, text = trial, kept + [(r, sym)], trial_text
                break
            else:
                conflict(r, first) if isinstance(first, genmerge.Conflict) else dropped.append((r.name, first))
        put = kept
    return _Pasted(ok, put, dropped, missing, text, probe)


def _grow_headers(ctx: Context, unit_source: Path, missing: list, probe) -> tuple[dict, dict]:
    """(include path -> its text now, include path -> its new text) for the unit's own headers
    (same stem as its source), with the members the drafts lacked added inside the class's
    padding (genmerge.grow_class)."""
    wanted: dict = {}                           # class -> (fields by offset, methods)
    for m in missing:
        fields, methods = wanted.setdefault(m.cls, ({}, []))
        for off, size, text in m.fields:
            fields.setdefault(off, (off, size, text))
        methods += [x for x in m.methods if x not in methods]
    headers = _class_headers_named(ctx, {c.split("::")[-1] for c in wanted})
    base, plan = {}, {}
    for cls, (fields, methods) in wanted.items():
        rel = headers.get(cls.split("::")[-1])
        if rel is None or Path(rel).stem != unit_source.stem:
            continue                                # not this unit's header: the reason stays
        base.setdefault(rel, (ctx.cfg.root / "include" / rel).read_text())
        text = plan.get(rel, base[rel])
        body = re.search(rf"\b(?:class|struct)\s+{re.escape(cls.split('::')[-1])}\b[^;{{()]*\{{(.*?)^\}};", text, re.S | re.M)
        names = re.findall(r"\bchar\s+(\w+)\s*\[", body[1]) if body else []
        values = probe.values([f"(int)&(({cls}*)0)->{n}" for n in names] + [f"(int)sizeof((({cls}*)0)->{n})" for n in names])
        pads = {n: (values[i], values[len(names) + i]) for i, n in enumerate(names)
                if values[i] is not None and values[len(names) + i] is not None}
        grown = genmerge.grow_class(text, cls, list(fields.values()), methods, pads)
        if grown is not None:
            plan[rel] = grown
    return base, plan


def _class_headers_named(ctx: Context, names: set) -> dict[str, str]:
    """Class name -> the one header under include/ that defines it (as an #include path)."""
    if not names:
        return {}
    pattern = re.compile(r"^\s*(?:struct|class)\s+(" + "|".join(map(re.escape, sorted(names))) + r")\b[^;{()]*\{", re.M)
    found: dict = {}
    root = ctx.cfg.root / "include"
    for path in sorted(root.rglob("*.h")):
        for m in pattern.finditer(path.read_text(errors="replace")):
            found.setdefault(m[1], set()).add(str(path.relative_to(root)))
    return {n: next(iter(hs)) for n, hs in found.items() if len(hs) == 1}


class _Probe:
    """Constant expressions about the unit's types, evaluated by compiling them after its
    source: all failures reported in one compile (-maxerrors), then the values read back."""

    def __init__(self, ctx: Context, command: str, source: str, path: Path, work: Path, overlay: Path):
        self.ctx, self.command, self.path, self.work, self.overlay = ctx, command, path, work, overlay
        self.lines = source.rstrip("\n").splitlines()
        self.cache: dict = {}

    def _run(self, text: str, stem: str):
        (self.work / f"{stem}.cpp").write_text(text)
        args = retarget(self.command, self.work)
        i = args.index("-c")
        args[i + 1] = str(self.work / f"{stem}.cpp")
        args[i:i] = ["-maxerrors", "1000", "-i", str(self.path.parent)]
        if self.overlay.is_dir():
            first = args.index("-i")
            args[first:first] = ["-i", str(self.overlay)]
        proc = subprocess.run(args, cwd=self.ctx.cfg.root, capture_output=True, text=True, timeout=120)
        obj = self.work / f"{stem}.o"
        if proc.returncode == 0 and obj.is_file():
            return Elf.parse(obj.read_bytes()), set()
        return None, {int(m[1]) for m in re.finditer(r"^#\s+(\d+):", proc.stdout + proc.stderr, re.M)}

    def values(self, exprs: list[str]) -> list:
        """Each expression's int value, or None where it does not compile."""
        todo = [e for e in dict.fromkeys(exprs) if e not in self.cache]
        if todo:
            checks = [f"static const int __dk_probe{k} = {e};" for k, e in enumerate(todo)]
            _, bad = self._run("\n".join(self.lines + checks) + "\n", "probe")
            good = [e for k, e in enumerate(todo) if len(self.lines) + 1 + k not in bad]
            got = {}
            if good:
                array = 'extern "C" const int __dk_probe[] = { ' + ", ".join(good) + " };"
                obj, _ = self._run("\n".join(self.lines + [array]) + "\n", "probe2")
                sym = next((x for x in obj.symbols if x.name == "__dk_probe"), None) if obj else None
                if sym is not None:
                    data = obj.sections[sym.shndx].data[sym.value:sym.value + 4 * len(good)]
                    got = dict(zip(good, struct.unpack(f">{len(good)}i", data)))
            for e in todo:
                self.cache[e] = got.get(e)
        return [self.cache[e] for e in exprs]

    def members(self, cls: str, names: list[str]) -> dict:
        """(class, [member]) -> {member: its offset, or None when the class has no such member}."""
        return dict(zip(names, self.values([f"(int)&(({cls}*)0)->{n}" for n in names])))


def _address_of(ctx: Context, symbol: str) -> int | None:
    """Target address of one of our function symbols: its own name, or fn_ADDR in it."""
    if symbol in ctx.by_name:
        return ctx.by_name[symbol].address
    m = re.match(r"(?:fn|lbl)_([0-9A-F]{8})(?:__|$)", symbol)
    return int(m[1], 16) if m else None


def _class_headers(ctx: Context, drafts: list[Result]) -> dict[str, str]:
    """Named classes the drafts define (not generator placeholders) -> the one header under
    include/ that defines each, as an #include path."""
    names = set()
    for r in drafts:
        try:
            names |= {n.split("::")[-1] for n in genmerge.Unit().add(r.standalone, r.address).structs}
        except genmerge.Conflict:
            continue
    return _class_headers_named(ctx, {n for n in names if not re.match(r"Unk|G_|\w+_(?:m|arg|l|ret|target|elem|vf)[0-9A-F_]", n)})


def _header_classes(ctx: Context, unit: genmerge.Unit) -> list[tuple[str, str]]:
    """Placeholder structs whose name a header under include/ already defines."""
    names = {n.split("::")[-1]: n for n in unit.structs if not re.match(r"Unk|G_|\w+_(m|arg|l)[0-9A-F]", n)}
    found = []
    if not names:
        return found
    pattern = re.compile(r"^\s*(?:struct|class)\s+(" + "|".join(map(re.escape, names)) + r")\b\s*[:{]", re.M)
    for path in sorted((ctx.cfg.root / "include").rglob("*.h")):
        for m in pattern.finditer(path.read_text(errors="replace")):
            found.append((names[m[1]], ctx.cfg.rel(path)))
    return found


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
        old_drafts = json.loads((tree / "names.drafts.json").read_text()) if (tree / "names.drafts.json").is_file() else {}
        new = {r.name: r for r in solve_many(ctx, names, args.jobs)}
        dropped = sorted(n for n, s in old.items() if s == "exact" and new[n].status not in ("exact", "skipped"))
        # a draft that could not be written under its own name (say `C* C::__ct()`) was never
        # usable: retiring it is not a loss
        retired = [n for n in dropped if old_drafts and not _writable(ctx, n, old_drafts.get(n, ""), tree)]
    lost = [n for n in dropped if n not in retired]
    gained = sorted(n for n, r in new.items() if r.status == "exact" and old.get(n) != "exact")
    print(f"dk gen regress vs {args.regress}: {len(names)} functions, exact {sum(s == 'exact' for s in old.values())}"
          f" -> {sum(r.status == 'exact' for r in new.values())}; lost {len(lost)}, gained {len(gained)}"
          + (f", retired {len(retired)} drafts not writable under their own names" if retired else ""))
    for n in lost:
        print(f"  LOST {n}: {new[n].status}: {new[n].detail}")
    for n in retired:
        print(f"  retired {n}: now {new[n].status}: {new[n].detail}")
    return MISMATCH if lost else OK


_C_DEFINITION = re.compile(r'^extern "C" [^;{]*?\b(\w+)\s*\([^;]*\)\s*\{', re.M)


def _writable(ctx: Context, name: str, standalone: str, work: Path) -> bool:
    """Does this draft compile under its own names and stay exact there? (Not when it defines
    a mangled C++ name as an extern "C" function: that duplicates the real function.)"""
    sym = ctx.by_name.get(name)
    command = ctx.command(sym.address)[0] if sym else None
    if not standalone or command is None:
        return False
    if any(_demangled(m[1]) is not None for m in _C_DEFINITION.finditer(standalone)):
        return False
    obj, _ = _compile(ctx, command, standalone, work, f"w{sym.address:08X}")
    if obj is None:
        return False
    r = Result(name, sym.address, "exact", standalone=standalone)
    # a special member (__ct__, __dt__, __as__...) must come out under exactly its symbol: a
    # member literally named __ct is a different function
    found = name if name in obj.functions() else None if name.startswith("__") else _symbol_in(obj, ctx, r)
    if found != name and _demangled(name) is not None:
        return False                    # a mangled name must come out as exactly that symbol
    return found is not None and gen.compare(gen.parse(ctx.entries[name].text), obj, found, ctx.literal,
                                             literal_info=ctx.literal_info) is None


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


def stretches(found: list[tuple[int, str]], declared: list[tuple[int, int]]) -> list[tuple[int, int, list[str]]]:
    """Group (address, name) of functions outside every unit into the undeclared stretches of
    .text between declared ranges: [(start, end, names)], end exclusive (0 = open-ended)."""
    bounds = sorted(declared)
    out: dict[tuple[int, int], list[str]] = {}
    for addr, name in sorted(found):
        lo = max((b for a, b in bounds if b <= addr), default=0)
        hi = min((a for a, b in bounds if a > addr), default=0)
        out.setdefault((lo, hi), []).append(name)
    return [(lo, hi, names) for (lo, hi), names in out.items()]


def _harvest(ctx: Context, args: argparse.Namespace) -> int:
    """Undeclared stretches ranked by how many of their functions already have exact drafts:
    declaring that unit (unit propose / unit add) turns them into progress at once."""
    lo, hi = (int(x, 16) for x in args.harvest) if len(args.harvest) == 2 else (0, 1 << 32)
    if args.wip:
        return _harvest_wip(ctx, args, lo, hi)
    names = []
    for name, entry in ctx.entries.items():
        sym = ctx.by_name.get(name)
        if sym is None or sym.type != "function" or not lo <= sym.address < hi or ctx.unit_of(sym.address):
            continue
        if ctx.template_for(sym.address) is None:
            continue
        try:
            n = len(gen.strip_frame(gen.parse(entry.text)))
        except gen.Unsupported:
            n = len(gen.parse(entry.text))
        if n <= args.max_insns:
            names.append(name)
    exact = [(ctx.by_name[r.name].address, r.name) for r in solve_many(ctx, names, args.jobs) if r.status == "exact"]
    declared = [(a, b) for u in ctx.splits for sec, a, b in u.splits if sec == ".text"]
    ranked = sorted(stretches(exact, declared), key=lambda t: (-len(t[2]), t[0]))
    print(f"dk gen harvest: {len(exact)} of {len(names)} undeclared functions (<= {args.max_insns} instructions) "
          f"have exact drafts, in {len(ranked)} undeclared stretches")
    for start, end, fns in ranked:
        span = f"0x{start:08X}-0x{end:08X}" if end else f"0x{start:08X}-"
        print(f"  {len(fns):4} exact  {span}  e.g. {fns[0]}")
    if ranked:
        hint(f"unit propose {ranked[0][2][0]}")
    return OK


def _harvest_wip(ctx: Context, args: argparse.Namespace, lo: int, hi: int) -> int:
    """--harvest --wip: declared units that are not linked, ranked by how many of their
    not-yet-exact functions (per the last build's objdiff report) have exact drafts."""
    if not ctx.cfg.report.is_file():
        print(f"error: no report at {ctx.cfg.rel(ctx.cfg.report)} (./dk build writes it)")
        return USAGE
    report = json.loads(ctx.cfg.report.read_text())
    open_at = {int(f["metadata"]["virtual_address"]) for u in report.get("units", []) for f in u.get("functions", [])
               if f.get("fuzzy_match_percent", 0) < 100 and "virtual_address" in f.get("metadata", {})}
    names = []
    for address in sorted(open_at):
        sym = ctx.by_address.get(address)
        unit = ctx.unit_of(address)
        if sym is None or sym.name not in ctx.entries or not unit or ctx.linked(unit) or not lo <= address < hi:
            continue
        try:
            n = len(gen.strip_frame(gen.parse(ctx.entries[sym.name].text)))
        except gen.Unsupported:
            n = len(gen.parse(ctx.entries[sym.name].text))
        if n <= args.max_insns:
            names.append(sym.name)
    by_unit: dict[str, list[str]] = {}
    for r in solve_many(ctx, names, args.jobs):
        if r.status == "exact":
            by_unit.setdefault(ctx.unit_of(r.address), []).append(r.name)
    ranked = sorted(by_unit.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    print(f"dk gen harvest --wip: {sum(map(len, by_unit.values()))} of {len(names)} not-yet-exact functions "
          f"(<= {args.max_insns} instructions) in WIP units have exact drafts, in {len(ranked)} units")
    for unit, fns in ranked:
        print(f"  {len(fns):4} exact  {unit}  e.g. {fns[0]}")
    if ranked:
        hint(f"gen --unit {ranked[0][0]} --max-insns {args.max_insns}")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("functions", nargs="*", help="function symbols")
    p.add_argument("--unit", help="every function of this declared unit")
    p.add_argument("--survey", action="store_true", help="dry run over every small function outside linked units")
    p.add_argument("--harvest", nargs="*", metavar="HEX",
                   help="rank undeclared stretches by exact drafts (optionally only START END) to split first")
    p.add_argument("--wip", action="store_true",
                   help="with --harvest: WIP units ranked by exact drafts for their not-yet-exact functions")
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
    p.add_argument("--write", action="store_true",
                   help="with --unit: merge the exact drafts into one source (the unit's, while it is empty), each verified again")
    p.add_argument("--into", metavar="PATH", help="with --write or --merge: write the merged source here instead")
    p.add_argument("--merge", action="store_true",
                   help="with --unit: insert the exact drafts of functions the unit's source lacks into it (by address), "
                        "reusing its and its headers' definitions; keeps a draft only if it is exact there and no "
                        "other function scores lower")
    p.add_argument("--regress", metavar="REV", help="compare with the generator at REV on the same functions (exit 1 if a draft is lost)")


COMMAND = Command("gen", "matching", "verified C++ drafts for small functions (exact-only; --survey for a dry run)",
                  run, _configure)
