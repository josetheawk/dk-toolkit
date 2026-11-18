"""dk close: close a near-miss function with faithful source rewrites, found by a parallel beam
search and kept only when the judge accepts them.

  ./dk close <unit> <fn> [--apply]            one function
  ./dk close --near [--min 90] [--sample N]   every WIP function at or above --min %, or a sample
  ./dk close --near --from sample.json        a fixed list of [unit, fn] pairs
  ./dk close --near … --baseline              the old `regs --try` moves only (declaration order,
                                              loop indexing, operand swaps), greedy: the comparison

How it works:
- each candidate source compiles privately (a mirror of the tree; the real file is untouched) with
  `-sym on`, which leaves the code identical and adds debug info: objdiff then maps every
  differing instruction to its source line, and the debug info names each local's register;
- the symptom classifier (rewrites.py) reads the aligned instructions, the rank solver
  (ranksolve.py) reads the registers; the rewrite table proposes edits at the lines the mismatches
  come from (rules ordered by how much of the symptom they answer), the solver proposes the
  declaration orders that fit the target's registers;
- a beam of the best few sources (match %, then fewest differing rows) is expanded level by level,
  every level's candidates compiled in parallel; output is deterministic (candidates are ordered
  by generation, never by finishing time);
- a 100% candidate gets its `__typeof__` locals typed from the debug info, is compiled with the
  unit's real flags and judged (judge.py): exact, hand-linked identical, nothing else in the unit
  worse, and no device added. Only then is it reported exact, and `--apply` writes it.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ...core import cbody
from ...core.command import BUILD_ERROR, MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from . import rewrites
from .dwarftypes import class_members, local_types
from .elf import Elf
from .jobs import resolve
from .judge import Verdict, forbidden_rules, judge, replace_typeof
from .linkproof import function_proof
from .matching import MatchError, UnitScore, load_objdiff_units, resolve_unit, score
from .regs import debug_refs, find_function, parse_dwarf1, register_map, _diff
from .reorder import compile_variant

BASELINE_RULES = ("decl-order", "unsigned-index", "operand-swap")


@dataclass
class Eval:
    text: str
    percent: float
    rows: int                       # differing instruction rows
    sym: rewrites.Symptoms
    first_line: int | None          # source line (1-based) of the function's first instruction
    types: dict[str, str] = field(default_factory=dict)      # local -> type (when __typeof__ is used)
    dwarf: object = None            # regs.DwarfFunction: each variable's register
    pairs: dict = field(default_factory=dict)                # our register -> Counter of target registers
    labels: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    index: int = 0                  # generation order (ties)
    defs: dict = field(default_factory=dict)                 # our register -> line that first writes it
    values: list | None = None      # --moves oracle: oracle.Value per coloured value (None: no table)
    right: int | None = None        # --moves oracle: values coloured as the target (registers-right)
    wrong: int = 0                  # --moves oracle: values coloured otherwise
    shape_rows: int = 0             # differing rows that are not only register operands
    record: dict | None = None      # --moves oracle: the oracle's record of the function (colouring order with v2)
    sat: int = 0                    # --moves oracle v2: the solver's rules this compile colours in the rule's order
    code: str = ""                  # hash of the function's code (same code, same registers)
    rejected: bool = False          # a 100% candidate the judge refused: never expanded
    view: list = field(default_factory=list)                 # relative_rows (Closer.keep_rows: llm-agent's state)
    parent: object = None           # the beam state this one was made from (the trajectory, --trace)

    @property
    def key(self) -> tuple:
        if self.right is not None:             # same shape first, then rules satisfied, fewest wrong values, fuzzy %
            return (-(self.percent == 100.0), self.shape_rows, -self.sat, self.wrong, -self.percent, self.rows, self.index)
        return (-self.percent, self.rows, self.index)


@dataclass
class Options:
    width: int = 4
    depth: int = 4
    budget: int = 600
    per_state: int = 80
    per_rule: int = 10
    jobs: int = 0
    rules: tuple[str, ...] | None = None
    solver: bool = True
    groups: bool = True             # also try a rewrite at every site of one shape at once
    skip: tuple[str, ...] = ()      # rules left out (to measure what one adds)
    moves: str = "table"            # "truth": the register explainer's edits first at every state (regexplain.py);
                                    # "oracle": the compiler oracle's blamed statements first, scored by registers-right
    memo: bool = True               # resume from build/dk/close/<fn>.json (explored candidates, best source)
    batch: int = 0                  # compile up to N candidates in one TU first (renamed copies; batch.py): same
                                    # code as a sibling or the parent is not compiled again; 0 = off
    trace: str = ""                 # append the exact result's verified trajectory (JSONL) to this file


def baseline_options(budget: int = 600, jobs: int = 0) -> Options:
    """Only the moves `regs --try` makes (declaration order, loop indexing, operand swaps), greedily."""
    return Options(width=1, depth=3, budget=budget, per_state=40, per_rule=40, jobs=jobs,
                   rules=BASELINE_RULES, solver=False, groups=False)


@dataclass
class Result:
    unit: str
    function: str
    start: float
    best: float
    exact: bool = False             # judged exact and faithful
    compiles: int = 0
    secs: float = 0.0
    labels: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)      # why a 100% candidate was rejected
    text: str | None = None         # the accepted source
    error: str = ""
    diagnosis: list[str] = field(default_factory=list)     # the rank solver on the best source
    rules: list[str] = field(default_factory=list)         # the rule of each label
    batched: int = 0                # candidates the batch showed to repeat code already compiled (not compiled again)
    review: tuple[str, list[str]] | None = None    # the exact one's review() (category, flags): clean or needs-judgment


class Closer:
    def __init__(self, cfg: Config, unit: dict, symbol: str, opts: Options, log=print):
        self.cfg, self.unit, self.symbol, self.opts, self.log = cfg, unit, symbol, opts, log
        self.source = cfg.root / unit["metadata"]["source_path"]
        self.c = self.source.suffix == ".c"
        tag = hashlib.sha1(f"{unit['name']}:{symbol}".encode()).hexdigest()[:8]
        self.work = cfg.root / "build" / "dk" / "close" / f"{os.getpid()}_{tag}"
        self.counter = 0
        self.members: frozenset | None = None     # `this`'s data members (debug info of the first compile)
        self.accessors: dict | None = None        # indexing accessors of the headers (shaperules.accessor)
        self.text: str | None = None              # start from this source instead of the unit's file
        self.explored: set[str] = set()           # candidates a past run compiled (the memo), by hash
        from .objcache import main_root            # one memo for every worktree (another agent's run counts)
        self.memo_path = main_root(cfg) / "build" / "dk" / "close" / f"{memo_name(symbol)}.json"
        self.oracle_cmd = None
        self.v2 = False                           # the runner has contract v2 (colouring order, solver, write mode)
        self.temps = False                        # ask the runner for temp origins (contract v3; regs --why-temp)
        self.rules: list[dict] = []               # v2: the solver's rules on the starting source (values of `base`)
        self.base_record: dict | None = None
        self.analysis: dict | None = None
        self.wanted: list = []                    # v2: the declaration-order lever per class [(name, line)...]
        self.rule_lines: set[int] = set()         # v2: source lines (1-based) of the flipped rules' values
        self.rule_names: set[str] = set()
        self.batcher = None
        self.batched = 0                          # candidates the batch answered without a compile of their own
        self.keep_rows = False                    # keep the target | ours rows on each Eval (llm-agent)
        self.diag = None                          # --moves directed: regdirect.Diagnosis of the start
        if opts.moves in ("oracle", "directed"):
            from . import oracle
            self.oracle_cmd = oracle.runner(cfg)
            self.temps = opts.moves == "directed"
            if self.oracle_cmd is not None:
                self.v2 = oracle.supports_solve(self.oracle_cmd, self.work / "probe")

    # -- compiling and scoring
    def evaluate(self, text: str, slot, force: dict | None = None) -> Eval | None:
        work = self.work / f"s{slot}"
        try:
            wrap, table = None, None
            if self.opts.moves in ("oracle", "directed") and self.oracle_cmd:
                from . import oracle
                table = work.resolve() / "vregs.json"
                work.mkdir(parents=True, exist_ok=True)
                spec = None
                if force is not None:
                    spec = work.resolve() / "force.json"
                    spec.write_text(json.dumps(force))
                wrap = oracle.wrap(self.oracle_cmd, table, self.symbol, extra=self.v2, force=spec, temps=self.temps)
            obj, _err = compile_variant(self.cfg, self.unit, text, work, ("-sym", "on"), wrap=wrap)
            if obj is None:
                return None
            pair = _diff(self.cfg, self.unit, obj, self.symbol)
            if pair is None:
                return None
            left, right = pair
            lt, rt = left["instructions"], right["instructions"]
            rows = sum(1 for t, o in zip(lt, rt) if t.get("diff_kind") or o.get("diff_kind"))
            first = next((o["instruction"]["line_number"] for o in rt if o.get("instruction", {}).get("line_number")), None)
            elf = Elf.parse(obj.read_bytes())
            dbg = elf.section(".debug")
            dwarf = find_function(parse_dwarf1(dbg.data, debug_refs(elf)), self.symbol) if dbg else None
            types = local_types(elf, self.symbol, self.c) if "__typeof__" in text else {}
            if slot == 0 and self.members is None:
                self.members = class_members(elf, self.symbol)
            from .regexplain import def_lines
            e = Eval(text, float(left.get("match_percent", 0.0)), rows, rewrites.symptoms(lt, rt), first, types,
                     dwarf, register_map(lt, rt), defs=def_lines(rt))
            if self.keep_rows:
                from .llmtry import relative_rows
                e.view = relative_rows(left, right)
            fn = elf.functions().get(self.symbol)
            e.code = hashlib.sha1(fn.code + repr([(r.offset, r.type, r.symbol) for r in fn.relocs]).encode()).hexdigest() \
                if fn is not None else ""
            e.shape_rows = sum(1 for t, o in zip(lt, rt)       # pure objdiff (the RL tiers read it without the oracle)
                               if (t.get("diff_kind") or o.get("diff_kind")) not in (None, "DIFF_ARG_MISMATCH"))
            if table is not None:
                e.record, e.values = oracle_values(table, elf, self.symbol, lt, rt)
                right, known = oracle.registers_right(e.values) if e.values is not None else (-1, 0)
                e.right, e.wrong = right, (known - right if e.values is not None else 1 << 20)
                if self.rules and self.base_record is not None and e.record is not None:
                    e.sat = sum(1 for x in oracle.rule_status(self.rules, self.base_record, e.record) if x)
            return e
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def evaluate_all(self, texts: list[str]) -> list[Eval | None]:
        jobs = resolve(self.opts.jobs, self.cfg)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            return list(pool.map(self.evaluate, texts, range(len(texts))))

    def unit_score(self, text: str, slot: str) -> tuple[UnitScore | None, object, str]:
        work = self.work / slot
        try:
            obj, err = compile_variant(self.cfg, self.unit, text, work)
            if obj is None:
                return None, None, err
            s = score(self.cfg, self.unit["name"], self.cfg.root / self.unit["target_path"], obj)
            return s, function_proof(self.cfg, self.unit, obj, self.symbol), ""
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # -- where the function is
    def body(self, text: str, first_line: int | None) -> cbody.Stmt | None:
        """The function's body in the unit's source: found by the line of its first instruction,
        checked by name (an inline member defined in a header has header line numbers)."""
        plain = plain_name(self.symbol)
        span = cbody.function_at_line(text, first_line - 1) if first_line else None
        if span is not None and plain is not None and span.name.split("::")[-1] != plain:
            span = None
        if span is None and plain is not None:
            from .trycmd import candidates                # by class and parameters when names repeat
            hits = candidates(text, self.symbol)
            span = hits[0] if len(hits) == 1 else None
        if span is None:
            return None
        try:
            return cbody.parse_body(text, span.open, cbody.type_names(text))
        except cbody.ParseError:
            return None

    def site(self, e: Eval) -> rewrites.Site | None:
        body = self.body(e.text, e.first_line)
        if body is None:
            return None
        lo, hi = cbody.line_of(e.text, body.start), cbody.line_of(e.text, body.end)
        focus = {ln - 1 + d for ln in e.sym.lines for d in (-1, 0, 1)} & set(range(lo, hi + 1))
        from .ranksolve import below_parameters
        prefer = frozenset(b.name for b in below_parameters(e.dwarf, e.pairs) if b.name)
        if self.accessors is None:
            from .shaperules import project_accessors
            self.accessors = project_accessors(self.cfg.root, e.text)
        return rewrites.Site(e.text, body, focus or None, not self.c, members=self.members, prefer=prefer,
                             accessors=self.accessors)

    def proposals(self, e: Eval, seen: set[str]) -> list[tuple[str, str, str]]:
        """(rule, label, source) to try from one beam state: the rank solver's orders first, then each
        rule (most relevant first) up to its share."""
        site = self.site(e)
        if site is None:
            return []
        out: list[tuple[str, str, str]] = []
        if self.opts.moves == "directed":
            from . import regdirect
            if self.diag is None or self.diag.kind == "none":
                return []
            diff_lines = {ln - 1 for ln in e.sym.lines}
            found = regdirect.candidates(site, self.diag, e.values, self.wanted, self.rule_names, diff_lines)
            for kind, label, new in found:
                if new not in seen and _sha(new) not in self.explored:
                    seen.add(new)
                    out.append((kind, label, new))
            return out[:self.opts.per_state]
        if self.v2 and self.opts.rules is None and self.wanted:
            from .blockorders import block_orders
            for wanted in self.wanted:
                try:
                    found = block_orders(site, wanted, self.rule_names)
                except (ValueError, KeyError, IndexError, AttributeError):
                    found = []
                for kind, label, new in found:
                    if new not in seen and _sha(new) not in self.explored:
                        seen.add(new)
                        out.append((f"oracle:{kind}", label, new))
        if self.opts.moves == "oracle" and self.opts.rules is None and e.values is not None:
            from . import oracle
            for kind, label, new in oracle.moves(site, e.values):
                if new not in seen and _sha(new) not in self.explored:
                    seen.add(new)
                    out.append((f"oracle:{kind}", label, new))
        if self.opts.moves == "truth" and self.opts.rules is None:
            from .regexplain import explain
            for m in explain(site, e.dwarf, e.pairs, defs=e.defs).moves:
                if m.source not in seen and _sha(m.source) not in self.explored:
                    seen.add(m.source)
                    out.append((f"truth:{m.kind}", f"{m.label} [{m.var}]", m.source))
        if self.opts.solver and self.opts.rules is None:
            from .ranksolve import solver_candidates
            for label, new in solver_candidates(site, e.dwarf, e.pairs):
                if new not in seen and _sha(new) not in self.explored:
                    seen.add(new)
                    out.append(("rank-solver", label, new))
        rules = rewrites.ranked(e.sym)
        if self.opts.rules is not None:
            rules = [r for r in rules if r.name in self.opts.rules]
        rules = [r for r in rules if r.name not in self.opts.skip]
        for k, rule in enumerate(rules):
            share = self.opts.per_rule * (2 if k == 0 else 1)
            n = 0
            for name, label, new in rewrites.candidates(site, [rule], groups=self.opts.groups):
                if new in seen or _sha(new) in self.explored:
                    continue
                seen.add(new)
                out.append((name, label, new))
                n += 1
                if n >= share:
                    break
        if self.v2 and self.rules:
            out = rule_directed(e.text, out, self.open_rule_lines(e), self.rule_names)
        return out[:self.opts.per_state]

    # -- contract v2: the order solver's rules
    def diagnose(self, start: Eval) -> None:
        """Ask the runner's solver for the minimal rules on the starting source; they direct and score the search."""
        from . import oracle
        if not self.v2 or start.record is None or start.values is None:
            if self.opts.moves == "directed":
                from . import regdirect
                self.diag = regdirect.Diagnosis("none")
                self.log("  directed: no contract-v2 oracle record for the start: nothing to direct")
            return
        an = oracle.solve(self.oracle_cmd, start.record, start.values, self.work / "solve")
        if self.opts.moves == "directed":
            from . import regdirect
            self.diag = regdirect.diagnose(None if "error" in an else an, start.record, start.values)
            for line in regdirect.render(self.diag):
                self.log(line)
        if "error" in an:
            self.log(f"  order solver: {an['error']}")
            return
        self.analysis, self.base_record = an, start.record
        self.rules = oracle.flat_rules(an)
        self.wanted = oracle.wanted_orders(an)
        flipped = [r for r in self.rules if r["flip"]]
        self.rule_names = {x["name"] for r in flipped for x in (r["before"], r["after"])
                           if x.get("name") and x.get("kind") == "local"}
        start.sat = sum(1 for x in oracle.rule_status(self.rules, start.record, start.record) if x)
        reach = {True: "reachable by order", False: "not reachable by order", None: "reachability unknown"}[an.get("reachable")]
        self.log(f"  order solver: {reach}, {len(self.rules)} rule(s), {len(flipped)} flipped"
                 + (f"; declaration order {' / '.join(', '.join(n for n, _ in w) for w in self.wanted)}" if self.wanted else ""))

    def open_rule_lines(self, e: Eval) -> set[int]:
        """Source lines (1-based, in e's source) of the values of the rules e does not satisfy yet."""
        from . import oracle
        if e.record is None or self.base_record is None:
            return set()
        status = oracle.rule_status(self.rules, self.base_record, e.record)
        want = {(r["cls"], r[k]) for r, ok in zip(self.rules, status) if not ok for k in ("a", "b")}
        base_lines = {(v.get("cls"), v["index"]): v.get("line") for v in self.base_record.get("vregs", [])}
        if e.record is self.base_record:
            return {base_lines.get(k) for k in want if base_lines.get(k)}
        mapped = oracle.map_values(self.base_record, e.record)
        lines = {(v.get("cls"), v["index"]): v.get("line") for v in e.record.get("vregs", [])}
        return {lines.get((c, mapped.get((c, i)))) for c, i in want if lines.get((c, mapped.get((c, i))))}

    # -- the search
    def run(self) -> Result:
        t0 = time.time()
        text = self.text if self.text is not None else self.source.read_text()
        res = Result(self.unit["name"], self.symbol, 0.0, 0.0)
        try:
            start = self.evaluate(text, 0)
            if start is None:
                res.error = "the unit does not compile or the function is missing"
                return res
            res.start = res.best = start.percent
            res.compiles = 1
            if start.percent == 100.0:
                res.error = "already exact"
                return res
            self.diagnose(start)
            before, _proof, err = self.unit_score(text, "before")
            if before is None:
                res.error = "the unit does not compile: " + err[-300:]
                return res
            beam, seen, best, index = [start], {text}, start, 1
            rejected_codes: set[str] = set()            # code the judge refused: the same code is not judged again
            batch_seen: set[str] = set()                # batch code hashes already compiled for real
            if self.opts.batch:
                from .batch import Batcher, compiler_for
                size = None
                try:
                    tfn = Elf.parse((self.cfg.root / self.unit["target_path"]).read_bytes()).functions().get(self.symbol)
                    size = len(tfn.code) if tfn is not None else None
                except OSError:
                    pass
                self.batcher = Batcher(compiler_for(self.cfg, self.unit, self.work / "batch"), text, self.symbol, self.c,
                                       size, self.opts.batch, resolve(self.opts.jobs, self.cfg))
                if not self.batcher.ready:
                    self.log(f"  batch: off ({self.batcher.reason})")
            memo = self.load_memo(text)
            if memo:
                self.log(f"  memo: {len(self.explored)} candidates already explored, best {memo.get('best', 0.0):.2f}%"
                         + (f" with {' + '.join(memo.get('labels', []))}" if memo.get("labels") else ""))
                prior = memo.get("text")
                if prior and memo.get("best", 0.0) > start.percent:
                    again = self.evaluate(prior, 0)
                    res.compiles += 1
                    if again is not None:
                        again.labels, again.rules, again.index = list(memo.get("labels", [])), ["memo"], index
                        index += 1
                        seen.add(prior)
                        beam, best = [again, start], again
            compiled: list[str] = []
            for depth in range(1, self.opts.depth + 1):
                cands = []
                for parent in beam:
                    for rule, label, new in self.proposals(parent, seen):
                        cands.append((parent, rule, label, new))
                if self.batcher is not None and self.batcher.ready and cands:
                    cands = self.batch_filter(beam, cands, batch_seen, compiled)
                cands = cands[:max(0, self.opts.budget - res.compiles)]
                if not cands:
                    break
                results = self.evaluate_all([c[3] for c in cands])
                res.compiles += len(cands)
                compiled += [_sha(c[3]) for c in cands]
                children = []
                for (parent, rule, label, _new), r in zip(cands, results):
                    if r is None:
                        continue
                    r.labels, r.rules, r.index = parent.labels + [label], parent.rules + [rule], index
                    r.parent = parent
                    index += 1
                    children.append((parent, r))
                for _parent, r in sorted(children, key=lambda pr: pr[1].key):
                    if r.percent == 100.0 and r.code and r.code in rejected_codes:
                        r.rejected = True
                        continue
                    if r.percent == 100.0:
                        verdict, final = self.final_check(r, text, before)
                        if verdict.ok:
                            res.best, res.exact, res.labels, res.text = 100.0, True, r.labels, final
                            res.review = verdict.review
                            res.rules = r.rules
                            res.secs = time.time() - t0
                            self.log(f"  depth {depth}: {len(cands)} variants: EXACT with {' + '.join(r.labels)}")
                            self.save_memo(text, compiled, r)
                            if self.opts.trace:
                                write_trace(Path(self.opts.trace), self, r, final, res.compiles)
                            return res
                        res.reasons = verdict.reasons
                        r.rejected = True                           # a judged device is never expanded
                        if r.code:
                            rejected_codes.add(r.code)
                n = (4 if self.rules else 3) if self.opts.moves in ("oracle", "directed") else 2   # oracle: shape rows, rules, wrong values
                keep = sorted((r for p, r in children if r.key[:n] <= p.key[:n] and not r.rejected), key=lambda r: r.key)
                ranked_children = sorted((r for _, r in children), key=lambda r: r.key)
                if ranked_children and ranked_children[0].key < best.key:
                    best = ranked_children[0]
                self.log(f"  depth {depth}: {len(cands)} variants, best {best.percent:.2f}%"
                         + (f" with {' + '.join(best.labels)}" if best.labels else ""))
                beam = keep[:self.opts.width]
                if not beam:
                    break
            res.best, res.labels, res.rules = best.percent, best.labels, best.rules
            if best is not start:
                res.text = best.text
            self.save_memo(text, compiled, best)
            from .ranksolve import inline_notes, report, solve
            res.diagnosis = report(solve(best.dwarf, best.pairs)) + inline_notes(best.dwarf, best.pairs)
            return res
        finally:
            res.secs = time.time() - t0
            res.batched = self.batched
            shutil.rmtree(self.work, ignore_errors=True)

    def batch_filter(self, beam: list[Eval], cands: list, seen_codes: set[str], compiled: list[str]) -> list:
        """Candidates worth a compile of their own: one per distinct code (batch.py), none with its parent's code
        except the first per parent (a neutral step may open new edits). The rest are recorded as explored."""
        hashes = self.batcher.hashes([p.text for p in beam] + [c[3] for c in cands])
        parent_code = {id(p): h for p, h in zip(beam, hashes)}
        seen_codes.update(h for h in parent_code.values() if h)
        neutral: set[int] = set()
        keep = []
        for c, h in zip(cands, hashes[len(beam):]):
            if h is None:
                keep.append(c)
                continue
            if h == parent_code.get(id(c[0])):
                if id(c[0]) in neutral:
                    self.batched += 1
                    compiled.append(_sha(c[3]))
                    continue
                neutral.add(id(c[0]))
            elif h in seen_codes:
                self.batched += 1
                compiled.append(_sha(c[3]))
                continue
            seen_codes.add(h)
            keep.append(c)
        return keep

    # -- the frontier memo
    def load_memo(self, text: str) -> dict | None:
        """The memo of past runs on this source (by hash): fills `explored`; None when there is none."""
        if not self.opts.memo or not self.memo_path.is_file():
            return None
        try:
            memo = json.loads(self.memo_path.read_text())
        except (OSError, ValueError):
            return None
        if memo.get("source") != _sha(text) or memo.get("moves") != self.opts.moves:
            return None
        self.explored = set(memo.get("explored", []))
        return memo

    def save_memo(self, text: str, compiled: list[str], best: Eval) -> None:
        if not self.opts.memo:
            return
        old = self.load_memo(text) or {}
        keep = old.get("best", 0.0) > best.percent
        memo = {"unit": self.unit["name"], "function": self.symbol, "source": _sha(text), "moves": self.opts.moves,
                "runs": old.get("runs", 0) + 1, "explored": sorted(set(old.get("explored", [])) | set(compiled)),
                "best": old["best"] if keep else best.percent,
                "labels": old.get("labels", []) if keep else best.labels,
                "text": old.get("text") if keep else (best.text if best.text != text else None)}
        self.memo_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.memo_path.with_name(f".{self.memo_path.name}.{os.getpid()}")
        tmp.write_text(json.dumps(memo) + "\n")
        os.replace(tmp, self.memo_path)

    def probe_types(self, text: str, body: cbody.Stmt, names: list[str], first_line: int | None) -> dict[str, str]:
        """Types of locals the compiler optimized out of the debug info: a probe compile (never kept)
        passes each one's address to an external function right after its declaration, so it gets
        a stack slot and a debug entry."""
        edits = [(0, 0, ('extern "C" ' if not self.c else "") + "void dk_type_probe(const void*);\n")]
        refs = {}
        for s in cbody.statements(body):
            if s.kind == "decl" and s.decl.declarators[0].name in names:
                d = s.decl.declarators[0]
                if "&" in d.ptr:              # a reference has no entry of its own: probe a pointer to it
                    refs[f"dk_ref_{d.name}"] = d.name
                    edits.append((s.end, s.end, f" __typeof__(&{d.name}) dk_ref_{d.name} = &{d.name};"
                                                f" dk_type_probe(&dk_ref_{d.name});"))
                else:
                    edits.append((s.end, s.end, f" dk_type_probe(&{d.name});"))
        if len(edits) == 1:
            return {}
        probe = cbody.apply_edits(text, edits)
        work = self.work / "probe"
        try:
            obj, _err = compile_variant(self.cfg, self.unit, probe, work, ("-sym", "on"))
            if obj is None:
                return {}
            found = local_types(Elf.parse(obj.read_bytes()), self.symbol, self.c)
            out = {k: v for k, v in found.items() if k in names}
            out.update({refs[k]: v[:-1] for k, v in found.items() if k in refs and v.endswith("*")})
            return out
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def final_check(self, e: Eval, before_text: str, before: UnitScore) -> tuple[Verdict, str]:
        text = e.text
        if "__typeof__" in text:
            body = self.body(text, e.first_line)
            if body is None:
                return Verdict(["the function's body could not be read to write its local types"]), text
            types = dict(e.types)
            _, missing = replace_typeof(text, body, types)
            if missing:
                types.update(self.probe_types(text, body, missing, e.first_line))
                _, missing = replace_typeof(text, body, types)
            if missing:                                 # a call's result: the callee's declared return type
                from .localnames import call_return_types
                types.update(call_return_types(self.cfg.root, text, body, missing))
            text, missing = replace_typeof(text, body, types)
            if missing:
                return Verdict([f"no type in the debug info for {', '.join(missing)}"]), text
        text = self.name_placeholders(before_text, text, e.first_line)
        after, proof, err = self.unit_score(text, "final")
        if after is None:
            return Verdict(["does not compile with the real types written: " + err[-200:]]), text
        b_body = self.body(before_text, e.first_line)
        a_body = self.body(text, e.first_line)
        probe = self.cast_probe()
        v = judge(before_text, text, b_body, a_body, before, after, self.symbol, forbidden_rules(self.cfg), proof, probe)
        if v.ok:                                    # the exact flag consults review() too (RL1: an UnkLoop helper)
            from .judge import exact_review
            blocking, category, flags = exact_review(before_text, text, self.symbol, forbidden_rules(self.cfg), probe,
                                                     placeholders_ok=True)
            v.reasons += blocking
            v.review = (category, flags)
        return v, text

    def name_placeholders(self, before_text: str, text: str, first_line: int | None) -> str:
        """New placeholder locals (`t0`) that hold a call's result, named after the callee (localnames.py)."""
        from .judge import PLACEHOLDER_LOCAL, declared
        from .localnames import name_placeholders
        b_body, a_body = self.body(before_text, first_line), self.body(text, first_line)
        if b_body is None or a_body is None:
            return text
        new, renames = name_placeholders(text, a_body, {n for n, _, _ in declared(b_body)}, PLACEHOLDER_LOCAL)
        if renames:
            self.log("  named " + ", ".join(f"{a} -> {b}" for a, b in renames.items()))
        return new

    def cast_probe(self):
        """The judge's compiler probe (same-type casts that steer the code are devices): the same one
        llm-try, llm-grind and the audit pass, so every path applies one rule."""
        from .judge import unit_probe
        return unit_probe(self.cfg, self.unit, self.symbol, self.work / "casts")


def oracle_values(table: Path, elf: Elf, symbol: str, lt: list[dict], rt: list[dict]):
    """(record, values): the oracle's record for `symbol` and its values lined up with objdiff's rows; None
    for either when there is no table, no record or no alignment."""
    from . import oracle
    doc = oracle.load(table)
    if "error" in doc:
        return None, None
    funcs = sorted((s for s in elf.symbols if s.type == 2 and s.name and oracle.oracle_name(s.name) == oracle.oracle_name(symbol)),
                   key=lambda s: (s.shndx, s.value))
    me = next((s for s in funcs if s.name == symbol), None)
    size = me.size if me else None
    ordinal = [s.name for s in funcs if s.size == size].index(symbol) if me else 0
    rec = oracle.find_record(doc, symbol, size, ordinal)
    return rec, (oracle.values(rec, lt, rt) if rec else None)


def changed_lines(old: str, new: str) -> tuple[int, int]:
    """The 1-based line range of `old` an edit touches (common prefix and suffix removed)."""
    n = min(len(old), len(new))
    i = 0
    while i < n and old[i] == new[i]:
        i += 1
    j = 0
    while j < n - i and old[len(old) - 1 - j] == new[len(new) - 1 - j]:
        j += 1
    return old.count("\n", 0, i) + 1, old.count("\n", 0, max(i, len(old) - j)) + 1


def write_trace(path: Path, closer: "Closer", exact: Eval, final: str, compiles: int) -> None:
    """Append the exact state's verified trajectory: the start source, each applied move with the compiled
    state's score, and the judged final source (tool output only; every state was compiled)."""
    chain: list[Eval] = []
    e = exact
    while e is not None:
        chain.append(e)
        e = e.parent
    chain.reverse()
    steps = [{"label": s.labels[-1] if s.labels else "", "rule": s.rules[-1] if s.rules else "", "percent": s.percent,
              "rows": s.rows, "shape_rows": s.shape_rows, "right": s.right, "wrong": s.wrong, "sat": s.sat,
              "source_sha": _sha(s.text), "definition": definition_text(s.text, closer.symbol)} for s in chain[1:]]
    rec = {"schema": "close-trace.v1", "unit": closer.unit["name"], "function": closer.symbol, "moves": closer.opts.moves,
           "compiles": compiles, "start": {"percent": chain[0].percent, "rows": chain[0].rows, "source_sha": _sha(chain[0].text),
                                            "definition": definition_text(chain[0].text, closer.symbol)},
           "diagnosis": {"kind": closer.diag.kind, "cells": sorted({r.cell for r in closer.diag.rules}),
                         "stop": list(closer.diag.stop)} if getattr(closer, "diag", None) is not None else None,
           "steps": steps, "final_sha": _sha(final), "final_definition": definition_text(final, closer.symbol)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def trace_card(folder: Path) -> int:
    """Write DIR/CARD.md and DIR/SHA256SUMS for DIR/traces.jsonl (close --trace records): counts per diagnosis
    and move, compiles, functions; tool output only."""
    import hashlib as _h
    from collections import Counter
    src = folder / "traces.jsonl"
    if not src.exists():
        print(f"error: {src} not found")
        return USAGE
    recs = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    moves, kinds, fns = Counter(), Counter(), []
    for r in recs:
        fns.append((r["unit"], r["function"], r["compiles"], len(r["steps"])))
        kinds[(r.get("diagnosis") or {}).get("kind", "none")] += 1
        for st in r["steps"]:
            moves[st.get("rule", "")] += 1
    lines = ["# close trajectories (schema close-trace.v1)", "",
             f"records: {len(recs)}; every state compiled and scored by `dk close --moves {recs[0]['moves'] if recs else '?'}`; "
             "the final definition judged exact and faithful. Tool output only.", "",
             "| unit | function | compiles | moves |", "|---|---|---|---|"]
    lines += [f"| {u} | {f} | {c} | {n} |" for u, f, c, n in fns]
    lines += ["", "diagnosis kinds: " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())),
              "moves: " + ", ".join(f"{k} {n}" for k, n in moves.most_common()), ""]
    (folder / "CARD.md").write_text("\n".join(lines))
    sums = []
    for name in ("traces.jsonl", "CARD.md"):
        sums.append(f"{_h.sha256((folder / name).read_bytes()).hexdigest()}  {name}")
    (folder / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"{folder}: {len(recs)} records, CARD.md and SHA256SUMS written")
    return OK


def definition_text(text: str, symbol: str) -> str | None:
    """The function's definition in a unit source (None when it cannot be found)."""
    plain = plain_name(symbol)
    for span in cbody.functions(text):
        if plain is not None and span.name.split("::")[-1] == plain:
            return text[span.head_start:span.close + 1]
    return None


def rule_directed(text: str, cands: list[tuple[str, str, str]], lines: set[int], names: set[str]) -> list[tuple[str, str, str]]:
    """Candidates whose edit touches a line of an unsatisfied rule's value (or names one of its locals) first,
    the rest after, each group in its own order."""
    def hits(item) -> bool:
        a, b = changed_lines(text, item[2])
        return any(a - 1 <= x <= b + 1 for x in lines) or any(re.search(r"\b%s\b" % re.escape(n), item[1]) for n in names)
    first = [c for c in cands if hits(c)]
    return first + [c for c in cands if not hits(c)]


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def memo_name(symbol: str) -> str:
    """A file name for a function's memo: the symbol, shortened with a hash when long or odd."""
    safe = re.sub(r"[^A-Za-z0-9_.]", "_", symbol)
    return safe if len(safe) <= 80 and safe == symbol else f"{safe[:64]}_{hashlib.sha1(symbol.encode()).hexdigest()[:8]}"


OPERATORS = {"or": "operator|", "and": "operator&", "pl": "operator+", "mi": "operator-", "ml": "operator*",
             "dv": "operator/", "eq": "operator==", "ne": "operator!=", "as": "operator=", "lt": "operator<",
             "gt": "operator>", "vc": "operator[]", "cl": "operator()", "apl": "operator+=", "ami": "operator-=",
             "amu": "operator*=", "adv": "operator/=", "aor": "operator|=", "aad": "operator&="}


def plain_name(symbol: str) -> str | None:
    """The name a definition is written with: `fn_80001234__5ClassFv` -> fn_80001234, `__ct__…` -> the
    class, `__or__…` -> operator|; an extern "C" name as is. None for compiler-made symbols."""
    if symbol.startswith("@"):
        return None
    if symbol.startswith("__"):
        rest = symbol[2:]
        op, _, tail = rest.partition("__")
        if op in ("ct", "dt"):
            parts = 1
            if tail.startswith("Q") and tail[1:2].isdigit():
                parts, tail = int(tail[1]), tail[2:]
            name = None
            for _ in range(parts):
                digits = len(tail) - len(tail.lstrip("0123456789"))
                if not digits:
                    return None
                n = int(tail[:digits])
                name, tail = tail[digits:digits + n], tail[digits + n:]
            return name if op == "ct" else "~" + name
        return OPERATORS.get(op, symbol if not tail else None)
    return symbol.split("__")[0] if "__" in symbol else symbol


# ---- reporting ---------------------------------------------------------------------------------

def source_diff(before: str, after: str, path: str) -> list[str]:
    return list(difflib.unified_diff(before.splitlines(), after.splitlines(), path, path + " (closed)", n=1, lineterm=""))


def close_one(cfg: Config, unit: dict, symbol: str, opts: Options, apply: bool, log=print,
              source: Path | None = None) -> Result:
    closer = Closer(cfg, unit, symbol, opts, log)
    src = cfg.root / unit["metadata"]["source_path"]
    if source is not None:
        closer.text, src = source.read_text(), source
    res = closer.run()
    if res.exact and res.text is not None:
        log(f"EXACT and judged faithful: {symbol} {res.start:.2f}% -> 100% ({res.compiles} compiles, {res.secs:.1f} s)")
        for line in source_diff(src.read_text(), res.text, cfg.rel(src)):
            log("  " + line)
        if apply:
            src.write_text(res.text)
            log(f"  applied to {cfg.rel(src)}")
    elif res.error:
        log(f"{symbol}: {res.error}")
    else:
        log(f"{symbol}: best {res.best:.2f}% (was {res.start:.2f}%, {res.compiles} compiles"
            + (f", {res.batched} skipped as repeated code" if res.batched else "") + f", {res.secs:.1f} s)"
            + (f" with {' + '.join(res.labels)}" if res.labels else ""))
        if res.reasons:
            log("  a 100% candidate was rejected by the judge: " + "; ".join(res.reasons[:4]))
        if res.diagnosis:
            log("  rank solver (registers ours vs target):")
            for line in res.diagnosis[:8]:
                log("    " + line)
        if res.text is not None and res.best > res.start:
            out = cfg.root / "build" / "dk" / "close" / f"{src.stem[:24]}_{hashlib.sha1(symbol.encode()).hexdigest()[:8]}{src.suffix}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(res.text)
            log(f"  best source written to {cfg.rel(out)}")
    return res



def held_set(frozen: dict, unit: str, fn: str, thaw: tuple | None = None) -> str | None:
    """The frozen set that still holds (unit, fn), or None: sets in `thaw` (default unlock.THAW, the
    gate and near-samples the owner released on 2026-11-08) no longer block sweeps or --apply;
    training data keeps excluding them (gate.never / llmdata leaks are untouched)."""
    if thaw is None:
        from .unlock import THAW
        thaw = THAW
    held = frozen.get((unit, fn))
    return None if not held or held in thaw else held

def near_misses(cfg: Config, minimum: float, jobs: int = 0) -> list[tuple[str, str, float]]:
    """(unit, function, %) of every WIP function at or above `minimum` and below 100, from the
    project's current objects (`ninja` first)."""
    from . import freshness as fresh
    units = [u for u in load_objdiff_units(cfg) if u.get("base_path") and not u["metadata"].get("complete")]
    fresh.build(cfg, ["-k", "0", "-j", str(resolve(jobs, cfg)), *(u["base_path"] for u in units)])
    from .gate import frozen_keys
    frozen = frozen_keys(cfg)                                # unreleased frozen sets are never swept (held_set)

    def one(u: dict) -> list[tuple[str, str, float]]:
        base = cfg.root / u["base_path"]
        if not base.is_file():
            return []
        try:
            s = score(cfg, u["name"], cfg.root / u["target_path"], base)
        except MatchError:
            return []
        return [(u["name"], f.name, f.percent) for f in s.functions
                if minimum <= f.percent < 100.0 and not held_set(frozen, u["name"], f.name)]

    with ThreadPoolExecutor(max_workers=resolve(jobs, cfg)) as pool:
        return sorted(x for rows in pool.map(one, units) for x in rows)


def sample(items: list, n: int) -> list:
    """A deterministic spread: the n items whose names hash lowest."""
    return sorted(items, key=lambda x: hashlib.sha1(f"{x[0]}:{x[1]}".encode()).hexdigest())[:n]


def read_pairs(doc) -> list[tuple[str, str]]:
    """[unit, fn] pairs from a `--from` file: a list of pairs, a list of {unit, fn|function} records
    (a `--json` result), or a frozen set {"functions": [...]} (`llm-bench freeze`)."""
    if isinstance(doc, dict):
        doc = doc.get("functions", [])
    out = []
    for p in doc:
        if isinstance(p, dict):
            out.append((p["unit"], p.get("fn") or p["function"]))
        else:
            out.append(tuple(p[:2]))
    return out


def run_near(cfg: Config, args: argparse.Namespace, opts: Options) -> int:
    if args.from_:
        pairs = read_pairs(json.loads(Path(args.from_).read_text()))
    else:
        found = near_misses(cfg, args.min, args.jobs)
        print(f"{len(found)} WIP function(s) at {args.min:g}% or more")
        pairs = [(u, f) for u, f, _ in (sample(found, args.sample) if args.sample else found)]
    from .gate import frozen_keys
    frozen = frozen_keys(cfg)
    results = []
    for k, (u, f) in enumerate(pairs, 1):
        try:
            unit = resolve_unit(cfg, u)
        except MatchError as e:
            print(f"[{k}/{len(pairs)}] {u} {f}: {e}")
            continue
        print(f"[{k}/{len(pairs)}] {u} {f}", flush=True)          # flushed: a sweep's log shows its progress
        held = held_set(frozen, u, f)                            # an explicit --from list may measure a frozen function, never apply
        if held and args.apply:
            print(f"  frozen ({held}): measured only, not applied")
        results.append(close_one(cfg, unit, f, opts, args.apply and not held, log=lambda s: print("  " + s, flush=True)))
    done = [r for r in results if not r.error]
    exact = [r for r in done if r.exact]
    gains = [r.best - r.start for r in done]
    print(f"\n{'baseline (regs --try moves, greedy)' if args.baseline else 'closer'}: {len(exact)}/{len(done)} exact and judged "
          f"faithful, median gain {statistics.median(gains) if gains else 0.0:.2f} points, "
          f"{sum(r.compiles for r in done)} compiles, {sum(r.secs for r in done):.0f} s")
    rejected = [r for r in done if r.reasons and not r.exact]
    if rejected:
        print(f"  {len(rejected)} reached 100% only with a candidate the judge rejected")
    for r in exact:
        print(f"  exact: {r.unit} {r.function} ({' + '.join(r.labels)})")
    if args.diffs:
        out = Path(args.diffs)
        out.mkdir(parents=True, exist_ok=True)
        for r in exact:
            src = cfg.root / resolve_unit(cfg, r.unit)["metadata"]["source_path"]
            (out / f"{memo_name(r.function)}.diff").write_text(
                "\n".join(source_diff(src.read_text(), r.text or "", cfg.rel(src))) + "\n")
        print(f"  {len(exact)} diff(s) written to {out} (nothing applied)")
    if args.json:
        Path(args.json).write_text(json.dumps([{"unit": r.unit, "fn": r.function, "start": r.start, "end": r.best,
                                                "exact": r.exact, "compiles": r.compiles, "batched": r.batched,
                                                "secs": round(r.secs, 1),
                                                "labels": r.labels, "rules": r.rules, "rejected": r.reasons, "error": r.error}
                                               for r in results], indent=1) + "\n")
        print(f"  results written to {args.json}")
    hint("close <unit> <fn> --apply", "check <unit> --private")
    return OK if exact else MISMATCH


def fix_source(cfg: Config, rec: dict) -> str | None:
    """A shape-fix record's BEFORE definition put back into the unit source of its own landing commit
    (the after file with only this function's definition reverted); None when it cannot be placed."""
    from .llmfixes import git
    from .trycmd import definition, spliced
    after = git(cfg, "show", f"{rec['commit']}:{rec['path']}")
    span = definition(after, rec["function"]) if after else None
    if span is None or after[span[0]:span[1]].strip() != rec["after"].strip():
        return None
    return spliced(after, span, rec["before"])


def tally(rows: list[dict], key: str) -> dict[str, list[int]]:
    """{kind: [exact, records]} over rows whose `key` is a kind or a list of kinds."""
    out: dict[str, list[int]] = {}
    for r in rows:
        kinds = r.get(key) or ["-"]
        for k in ([kinds] if isinstance(kinds, str) else kinds):
            t = out.setdefault(k, [0, 0])
            t[0] += bool(r.get("exact"))
            t[1] += 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1][1], kv[0])))


def run_fixes(cfg: Config, args: argparse.Namespace, opts: Options) -> int:
    """Replay shape-fix records (`dk dataset fixes`): each record's before version, in a snapshot tree of
    its landing commit (that commit's headers and unit source, only the function's definition reverted),
    closed from scratch (no memo). Tallies judged-exact results per fix kind and diff class."""
    from .llmfixes import Snapshots, read_records
    from .llmtry import LLM_DIR
    recs = read_records(Path(args.fixes).expanduser())
    if args.sample:
        recs = recs[:args.sample]
    snaps = Snapshots(cfg, cfg.root / LLM_DIR / "fixsnaps")
    opts.memo = False
    tmp = cfg.root / "build" / "dk" / "close" / "fixes"
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, rec in enumerate(recs, 1):
        row = {"unit": rec["unit"], "fn": rec["function"], "commit": rec["commit"], "fix_kind": rec.get("fix_kind", []),
               "diff_class": rec.get("diff_class"), "recorded": rec.get("before_percent"), "exact": False}
        print(f"[{k}/{len(recs)}] {rec['unit']} {rec['function']} @ {rec['commit'][:9]}", flush=True)
        text = fix_source(cfg, rec)
        try:
            unit = resolve_unit(cfg, rec["unit"])
        except MatchError as e:
            unit, row["error"] = None, str(e)
        if text is None and unit is not None:
            row["error"] = "the before definition could not be placed in its commit's source"
        if "error" in row:
            print("  " + row["error"])
            rows.append(row)
            continue
        snap, _tree = snaps.at(rec["commit"])
        src = tmp / f"{memo_name(rec['function'])}{Path(rec['path']).suffix}"
        src.write_text(text)
        res = close_one(snap, unit, rec["function"], opts, False, log=lambda s: print("  " + s, flush=True), source=src)
        row.update({"start": res.start, "end": res.best, "exact": res.exact, "compiles": res.compiles,
                    "secs": round(res.secs, 1), "labels": res.labels, "rules": res.rules, "rejected": res.reasons,
                    "error": res.error})
        if res.exact and args.diffs:
            out = Path(args.diffs)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{memo_name(rec['function'])}.diff").write_text("\n".join(source_diff(text, res.text or "", rec["path"])) + "\n")
        rows.append(row)
    ok = [r for r in rows if not r.get("error")]
    exact = [r for r in rows if r["exact"]]
    print(f"\nfixes replayed: {len(exact)}/{len(rows)} exact and judged faithful ({len(rows) - len(ok)} not run: "
          f"error or already exact), {sum(r.get('compiles', 0) for r in rows)} compiles")
    for key in ("fix_kind", "diff_class"):
        print(f"  by {key}: " + ", ".join(f"{k} {e}/{n}" for k, (e, n) in tally(rows, key).items()))
    for r in exact:
        print(f"  exact: {r['fn']} ({' + '.join(r['labels'])})")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "by_kind": tally(rows, "fix_kind"),
                                               "by_class": tally(rows, "diff_class")}, indent=1) + "\n")
        print(f"  results written to {args.json}")
    return OK if exact else MISMATCH


def run(cfg: Config, args: argparse.Namespace) -> int:
    skip = tuple(x for x in (args.skip or "").split(",") if x)
    unknown = [x for x in skip if x not in rewrites.BY_NAME and x != "rank-solver"]
    if unknown:
        print(f"error: no rewrite rule named {', '.join(unknown)} (rules: {', '.join(rewrites.BY_NAME)}, rank-solver)")
        return USAGE
    opts = Options(width=args.width, depth=args.depth, budget=args.budget, jobs=args.jobs, skip=skip,
                   solver="rank-solver" not in skip, moves=args.moves, memo=not args.fresh, batch=args.batch,
                   trace=args.trace)
    if args.moves == "directed":
        opts.width, opts.depth, opts.per_state = min(args.width, 3), min(args.depth, 3), 16
        if args.budget == 600:
            opts.budget = 150
    if args.baseline:
        opts = baseline_options(args.budget, args.jobs)
        opts.memo = not args.fresh
    if args.moves in ("oracle", "directed"):
        from . import oracle
        if oracle.runner(cfg) is None:
            print(f"error: {oracle.UNAVAILABLE}")
            return USAGE
    if args.card:
        return trace_card(Path(args.card))
    if args.fixes:
        return run_fixes(cfg, args, opts)
    if args.queue:
        from .closequeue import run_queue
        return run_queue(cfg, args, opts)
    if args.near:
        return run_near(cfg, args, opts)
    if not args.unit or not args.function:
        print("usage: dk close <unit> <fn> [--apply], or dk close --near")
        return USAGE
    try:
        unit = resolve_unit(cfg, args.unit)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    from .gate import frozen_keys
    held = held_set(frozen_keys(cfg), unit["name"], args.function)
    from ...core.usage import note
    note(unit=unit["name"], function=args.function)
    if held and args.apply:
        note(refused_by="frozen")
        print(f"error: {args.function} is frozen ({held}): not applied (measure it without --apply)")
        return USAGE
    res = close_one(cfg, unit, args.function, opts, args.apply, source=Path(args.source) if args.source else None)
    note(score_before=round(res.start, 2), score_after=round(res.best, 2), exact=res.exact, compiles=res.compiles,
         applied=bool(args.apply and res.exact), **({"close_error": res.error[:80]} if res.error else {}))
    if res.error == "already exact":
        return OK
    if res.error:
        return BUILD_ERROR
    hint(f"check {args.unit} --private", f"diff {args.unit} --function {args.function}")
    return OK if res.exact else MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("unit", nargs="?", help="unit (source path or name)")
    p.add_argument("function", nargs="?", help="function symbol, as dk check lists it")
    p.add_argument("--apply", action="store_true", help="write a judged-exact source into the unit's file (or --source)")
    p.add_argument("--source", help="start from this file instead of the unit's source (compiled as the unit; "
                                    "--apply writes it)")
    p.add_argument("--near", action="store_true", help="close every WIP function at --min %% or more (or --sample / --from)")
    p.add_argument("--min", type=float, default=90.0, help="with --near: lowest match %% to try (default 90)")
    p.add_argument("--sample", type=int, default=0, help="with --near: a deterministic sample of N functions")
    p.add_argument("--from", dest="from_", help="with --near: a JSON file of functions to close instead: [unit, fn] pairs, "
                   "a --json result, or a frozen set {\"functions\": [...]}")
    p.add_argument("--baseline", action="store_true",
                   help="only the moves `regs --try` makes (declaration order, loop indexing, operand swaps), greedily")
    p.add_argument("--fixes", metavar="JSONL", help="replay shape-fix records (`dk dataset fixes`): each record's before "
                   "version in a snapshot of its landing commit (headers + unit source, only its definition reverted), "
                   "closed with no memo; tallies exact per fix kind (--sample N: the first N; --json, --diffs as --near)")
    p.add_argument("--queue", action="store_true", help="the CPU enumerator over `dk route`'s open queue (closequeue.py): "
                   "near misses at --min %% or more, not frozen/blocked/benchmark/held-out/busy, <= --small bytes first, "
                   "judged-exact results written as candidates to llm-data/candidates/<date>/ (never applied)")
    p.add_argument("--small", type=int, default=2048, help="with --queue: functions up to this size go first (default 2048)")
    p.add_argument("--thaw", default="", metavar="SET[,SET]", help="with --queue: frozen sets CPU tools may work "
                   "(gate-lora2 and the near-sample sets, owner decision 2026-11-08; results stay out of every dataset)")
    p.add_argument("--max-size", type=int, default=4096, help="with --queue: largest function tried (default 4096)")
    p.add_argument("--until", default="", metavar="HH:MM", help="with --queue: stop starting new functions at this time")
    p.add_argument("--resume", action="store_true", help="with --queue: continue the last run's state (llm-data/close-queue-state.json)")
    p.add_argument("--first", action="append", metavar="JSON",
                   help="with --queue: [unit, fn] pairs (or records with unit/function) to run before the rest")
    p.add_argument("--skip-list", action="append", metavar="JSON", help="with --queue: more [unit, fn] pairs to leave out "
                   "(repeatable; a frozen set or a --json result works too)")
    p.add_argument("--json", help="with --near/--fixes: write the per-function results here")
    p.add_argument("--diffs", help="with --near: write each judged-exact function's source change as DIR/<fn>.diff "
                                   "(the sources stay untouched unless --apply)")
    p.add_argument("--width", type=int, default=4, help="beam width (default 4)")
    p.add_argument("--depth", type=int, default=4, help="most rewrites stacked on one source (default 4)")
    p.add_argument("--budget", type=int, default=600, help="most compiles per function (default 600)")
    p.add_argument("--jobs", type=int, default=0, help="parallel compiles (default: dk.toml [platform] jobs, DK_JOBS, "
                                                       "else the cores the machine leaves idle)")
    p.add_argument("--trace", default="", metavar="JSONL", help="append an exact result's verified trajectory (start, each "
                   "applied move with its compiled score, the judged final definition) to this file")
    p.add_argument("--card", default="", metavar="DIR", help="write DIR/CARD.md and DIR/SHA256SUMS for DIR/traces.jsonl "
                   "(records from --trace) and exit")
    p.add_argument("--moves", choices=("table", "truth", "oracle", "directed"), default="table",
                   help="truth: at every beam state compile the register explainer's edits first (`regs --explain`: "
                        "declaration order, temporary vs named shape, statement order, inline helper, each aimed at a "
                        "variable whose register differs), then the rewrite table and the solver. oracle: every compile "
                        "also runs under the compiler oracle (DK_ORACLE), states are ranked by values coloured as the "
                        "target (fuzzy %% breaks ties) and the edits aimed at the statements that create the wrong values "
                        "come first (name or fold an operand, commutative operand order, statement order, split/join). "
                        "directed: the oracle's diagnosis of the start (reachable by order / stuck / scheduling only) "
                        "picks the moves per flipped rule's cell (regdirect.py): only those are compiled, cross-phase "
                        "temp pairs are reported as STOP; defaults width 3, depth 3, budget 150, 16 per state")
    p.add_argument("--batch", type=int, default=0, metavar="K",
                   help="compile up to K candidates of a level in one copy of the unit first (renamed copies after the "
                        "real definition, with a control copy that must equal it): candidates whose code a sibling, the "
                        "parent or an earlier level already has are not compiled again. Free functions under 2 KB only; "
                        "every score still comes from the candidate's own compile. 0 (default) = off")
    p.add_argument("--fresh", action="store_true",
                   help="ignore the frontier memo (build/dk/close/<fn>.json: candidates already compiled on this source "
                        "are skipped and the best one resumes the beam)")
    p.add_argument("--skip", help="comma-separated rules to leave out (`rank-solver` for the solver's orders), "
                                  "to measure what one adds")


COMMAND = Command("close", "matching", "close near-miss functions with judged, faithful rewrites (beam search)", run, _configure)
