"""dk dataset fixes: the "shape fix" records, from landing commits (compiler-verified, faithful by construction).
  ./dk dataset fixes [--since 2026-11-05 | --range A..B] [--resume] [--jobs 4] [--out DIR]
For every function that became exact in a commit (not exact before it, exact after it, its own definition
changed, nothing else in the record guessed) one record: the definition BEFORE and AFTER, the objdiff rows of
the BEFORE version against the target, its fuzzy score, the oracle summary (`regs --explain --oracle`, `regs
--reach`, short forms) when a runner is configured (DK_ORACLE), a fix kind derived from the two definitions, and
where the commit is (hash, never its message). Both versions are compiled with the headers of their own commit
(a snapshot tree: the commit's include/, everything else linked), so BEFORE reads exactly as the repo did.
Records hold project source, diffs and tool output only, no model text. The frozen eval sets (benchmark, held-out,
repair-eval, review-eval, near-samples, giants-99/97 when present) leave out by unit, function, asm shape and
content hash. Output (outside the repo): llm-data/datasets/shape-fixes/v0/{fixes.jsonl,CARD.md,SHA256SUMS,state.json}.
Incremental by commit range: `--resume` reads state.json and does only the commits not yet done. Re-run after every
grind day. Why: a model that fixes the shape of its own near miss learns from what really fixed the repo's ones.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from ...core.command import OK, USAGE
from ...core.config import Config

SCHEMA = 1
DEFAULT_SINCE = "2026-11-05"
EVAL_FILES = ("bench", "heldout", "repair-eval", "review-eval", "near-sample-60", "near-sample-fresh-30",
              "giants-99", "giants-97")
UNIT_SETS = ("bench", "heldout", "repair-eval", "review-eval")   # excluded by unit; the samples by function, shape, hash
SKIP_NAMES = ("@", "__sinit")


def sha(*parts: str) -> str:
    return hashlib.sha1("\0".join(parts).encode()).hexdigest()


def git(cfg: Config, *a: str, check: bool = False) -> str:
    p = subprocess.run(["git", *a], cwd=cfg.root, capture_output=True, text=True, errors="replace")
    if check and p.returncode:
        raise RuntimeError(p.stderr.strip())
    return p.stdout


# ---- fix kind (deterministic, from the two definitions) ---------------------------------------------

_DECL = re.compile(r"^(?:const\s+|unsigned\s+|signed\s+|struct\s+|register\s+)*[A-Za-z_][\w:<>]*\s*[*&]*\s+\**[A-Za-z_]\w*"
                   r"(?:\s*\[[^\]]*\])?(?:\s*=[^;]*)?;$")
_CTL = re.compile(r"\b(if|else|switch|case|for|while|do|goto|return|break|continue)\b|\?")
_CAST = re.compile(r"\(\s*(?:const\s+)?(?:unsigned\s+|signed\s+)?[A-Za-z_][\w:]*\s*\**\s*\)\s*[\w(&*-]")
_CALL = re.compile(r"\b([A-Za-z_][\w:]*)\s*\(")
_KEYWORDS = {"if", "while", "for", "switch", "return", "sizeof"}


def strip_comments(code: str) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/", " ", code, flags=re.S)


def code_lines(code: str) -> list[str]:
    return [" ".join(x.split()) for x in strip_comments(code).splitlines() if x.strip()]


def fix_kinds(before: str, after: str) -> list[str]:
    """Labels for what the edit did, from the two definitions alone (never from a message)."""
    b, a = code_lines(before), code_lines(after)
    kinds: list[str] = []
    if sorted(b) == sorted(a) and b != a:
        decl_only = [x for x in b if _DECL.match(x)] != [x for x in a if _DECL.match(x)] and \
            [x for x in b if not _DECL.match(x)] == [x for x in a if not _DECL.match(x)]
        return ["declaration-order" if decl_only else "statement-order"]
    bd, ad = [x for x in b if _DECL.match(x)], [x for x in a if _DECL.match(x)]
    if len(ad) > len(bd):
        kinds.append("local-added")
    elif len(ad) < len(bd):
        kinds.append("local-removed")
    bt, at = strip_comments(before), strip_comments(after)
    if sorted(re.findall(r"\b(?:if|else|switch|case|for|while|do|goto|break|continue)\b|\?", bt)) != \
            sorted(re.findall(r"\b(?:if|else|switch|case|for|while|do|goto|break|continue)\b|\?", at)):
        kinds.append("control-flow")
    if len(re.findall(r"\breturn\b", bt)) != len(re.findall(r"\breturn\b", at)):
        kinds.append("return-form")
    if len(_CAST.findall(bt)) != len(_CAST.findall(at)):
        kinds.append("cast")
    calls = lambda t: {c for c in _CALL.findall(t) if c not in _KEYWORDS}
    if calls(bt) != calls(at):
        kinds.append("call-or-accessor")
    nums = lambda t: sorted(re.findall(r"\b0x[0-9A-Fa-f]+\b|\b\d+(?:\.\d+)?f?\b", t))
    if nums(bt) != nums(at):
        kinds.append("literal")
    ops = lambda t: sorted(re.findall(r"<<|>>|[<>=!]=|&&|\|\||[-+*/%&|^~!<>]", t))
    if ops(bt) != ops(at):
        kinds.append("operator")
    words = lambda t: sorted(re.findall(r"\b[A-Za-z_]\w*\b", t))
    if not kinds and words(bt) != words(at):
        kinds.append("identifier")
    return kinds or ["expression-form"]


# ---- snapshot trees: a commit's headers, everything else linked ----------------------------------------

class Snapshots:
    """cfg-like roots whose include/ is a commit's own (cached by the include tree id; shared by threads)."""

    def __init__(self, cfg: Config, base: Path):
        self.cfg, self.base, self.lock, self.made = cfg, base, threading.Lock(), {}
        self.trees: dict[str, str] = {}

    def tree_id(self, rev: str) -> str:
        with self.lock:
            if rev not in self.trees:
                self.trees[rev] = git(self.cfg, "rev-parse", f"{rev}:include").strip()
            return self.trees[rev]

    def at(self, rev: str) -> tuple[Config, str]:
        tid = self.tree_id(rev)
        with self.lock:
            root = self.made.get(tid)
            if root is None:
                root = self.base / tid[:16]
                if not (root / ".done").exists():
                    shutil.rmtree(root, ignore_errors=True)
                    root.mkdir(parents=True)
                    for e in self.cfg.root.iterdir():
                        if e.name not in ("include", ".git", "worktrees", "build") and not e.name.startswith("."):
                            (root / e.name).symlink_to(e)
                    (root / "build").symlink_to(self.cfg.root / "build")
                    arc = subprocess.run(["git", "archive", rev, "include"], cwd=self.cfg.root, capture_output=True)
                    if arc.returncode:
                        raise RuntimeError(arc.stderr.decode(errors="replace"))
                    subprocess.run(["tar", "-x", "-C", str(root)], input=arc.stdout, check=True)
                    (root / ".done").write_text("")
                self.made[tid] = root
        return dataclasses.replace(self.cfg, root=root), tid


# ---- eval sets (the leakage filter) ----------------------------------------------------------------------

def norm_hash(code: str) -> str:
    from .llmdata import norm
    return hashlib.sha1(norm(code).encode()).hexdigest()


def frozen(cfg: Config) -> dict:
    """Everything the training must never see: units, function names, asm shapes and content hashes of
    the frozen eval sets that exist in llm-data/."""
    from .llmtry import asm_entries, asm_shape, data_dir
    d = data_dir(cfg)
    entries = asm_entries(cfg)
    out = {"units": {}, "functions": {}, "shapes": {}, "hashes": {}, "sets": {}}
    for name in EVAL_FILES:
        path = d / f"{name}.json"
        if not path.is_file():
            out["sets"][name] = None
            continue
        doc = json.loads(path.read_text())
        fs = doc["functions"] if isinstance(doc, dict) else doc
        out["sets"][name] = len(fs)
        if isinstance(doc, dict):
            for x in doc.get("units", []) if isinstance(doc.get("units"), list) and name in UNIT_SETS else []:
                u = x if isinstance(x, str) else (x.get("unit") or x.get("name") if isinstance(x, dict) else None)
                if u:
                    out["units"].setdefault(u, name)
        for r in fs:
            if not isinstance(r, dict):
                continue
            if r.get("unit") and name in UNIT_SETS:
                out["units"].setdefault(r["unit"], name)
            fn = r.get("function")
            if fn:
                out["functions"].setdefault(fn, name)
            shape = r.get("asm_shape") or (asm_shape(entries[fn].text) if fn in entries else None)
            if shape:
                out["shapes"].setdefault(shape, name)
            for k in ("exact", "start_code", "before_def", "after_def"):
                if isinstance(r.get(k), str) and r[k].strip():
                    out["hashes"].setdefault(norm_hash(r[k]), name)
    from . import gate                                  # gate-lora2 and the near-sample snapshots (definition + variant hashes)
    g_fns, g_shapes = gate.never(cfg)
    for k, v in g_fns.items():
        out["functions"].setdefault(k, v)
    for k, v in g_shapes.items():
        out["shapes"].setdefault(k, v)
    for k, v in gate.hashes(cfg).items():
        out["hashes"].setdefault(k, v)
    out["sets"].update({k: len(v) for k, v in gate.load_sets(cfg).items() if k == gate.GATE_NAME})
    return out


def leak(rec: dict, fz: dict) -> str | None:
    """Why a record may not be trained on, with the first frozen set that holds it (None: clean)."""
    if rec["unit"] in fz["units"]:
        return f"unit ({fz['units'][rec['unit']]})"
    if rec["function"] in fz["functions"]:
        return f"function ({fz['functions'][rec['function']]})"
    if rec["asm_shape"] in fz["shapes"]:
        return f"asm shape ({fz['shapes'][rec['asm_shape']]})"
    from .gate import agnostic_hash
    for k in ("before", "after"):
        for h in (norm_hash(rec[k]), agnostic_hash(rec[k])):
            if h in fz["hashes"]:
                return f"content hash ({fz['hashes'][h]})"
    return None


# ---- oracle summaries (short forms) -----------------------------------------------------------------------

def oracle_summary(cfg: Config, unit: dict, symbol: str, text: str) -> dict | None:
    """{"explain": ..., "reach": ...} short forms of `regs --explain --oracle` and `regs --reach`, or None when
    no oracle runner is configured or it gave nothing."""
    from . import oracle
    if oracle.runner(cfg) is None:
        return None
    from .closecmd import Closer, Options
    from .regs import _explain_oracle, reach_report
    out: dict = {}
    closer = Closer(cfg, unit, symbol, Options(moves="oracle", jobs=1), log=lambda s: None)
    try:
        e = closer.evaluate(text, 0)
        if e is not None and e.values is not None:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _explain_oracle(closer, e, text, SimpleNamespace(limit=10, unit=unit["name"], function=symbol))
            lines = [x for x in buf.getvalue().splitlines() if x.strip() and not x.startswith("next:")]
            keep, causes = [], False
            for x in lines:
                if x.strip().startswith("oracle:") or "!!" in x:
                    keep.append(x.rstrip())
                elif x.strip() == "causes:":
                    causes = True
                    keep.append(x.rstrip())
                elif causes and x.strip().startswith("- "):
                    keep.append(x.rstrip())
            if keep and not any("!!" in x for x in keep):
                keep.append("  (every coloured value already has the target's register)")
            out["explain"] = "\n".join(keep[:24])[:3000]
    except Exception as exc:                         # an unsupported build: the record goes on without it
        out["explain_error"] = type(exc).__name__
    finally:
        shutil.rmtree(closer.work, ignore_errors=True)
    try:
        lines: list[str] = []
        code, _res = reach_report(cfg, unit, symbol, text, say=lines.append, limit=8)
        lines = [x.rstrip() for x in lines if x.strip() and not x.startswith("next:")]
        if lines:
            out["reach"] = "\n".join(lines[:24])[:3000]
    except Exception as exc:
        out["reach_error"] = type(exc).__name__
    return out or None


# ---- the work ----------------------------------------------------------------------------------------------

class Cache:
    """File-level results by hash (one JSON per line), shared by threads."""

    def __init__(self, path: Path):
        self.path, self.lock, self.rows = path, threading.Lock(), {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            for line in path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.rows[r["key"]] = r
                except (ValueError, KeyError):
                    pass

    def get(self, k: str):
        return self.rows.get(k)

    def put(self, k: str, val: dict) -> None:
        with self.lock:
            self.rows[k] = {"key": k, **val}
            with self.path.open("a") as fh:
                fh.write(json.dumps(self.rows[k]) + "\n")


def compile_snap(cfg: Config, snap: Config, unit: dict, text: str, work: Path):
    """compile_variant against a snapshot root: (object or None, compiler output)."""
    from .reorder import _command, mirror, retarget
    out_dir = work.resolve() / "out"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    argv = retarget(_command(cfg, unit["base_path"]), out_dir)
    tree = work.resolve() / "tree"
    try:
        mirror(snap, unit["metadata"]["source_path"], text, tree)
        proc = subprocess.run(argv, cwd=tree, capture_output=True, text=True)
    finally:
        shutil.rmtree(tree, ignore_errors=True)
    objs = list(out_dir.glob("*.o"))
    return (objs[0] if proc.returncode == 0 and objs else None), (proc.stdout + proc.stderr)[-2000:]


def scored(cfg: Config, obj: Path, unit: dict) -> dict[str, tuple[float, int]]:
    from .matching import score
    s = score(cfg, unit["name"], cfg.root / unit["target_path"], obj)
    return {f.name: (f.percent, f.size) for f in s.functions}


def landing_commits(cfg: Config, args) -> list[dict]:
    if args.range:
        spec = [args.range]
    else:
        spec = [f"--since={args.since}", "HEAD"]
    out = git(cfg, "log", "--no-merges", "--reverse", "--format=%H\t%P\t%ct", *spec)
    commits = []
    for line in out.splitlines():
        h, parents, ct = (line.split("\t") + ["", ""])[:3]
        if parents.strip():
            commits.append({"commit": h, "parent": parents.split()[0], "time": int(ct)})
    return commits


def grind_logs(logs_dir: Path) -> list[tuple[str, str]]:
    out = []
    for p in sorted(logs_dir.glob("*.md")) if logs_dir.is_dir() else []:
        out.append((p.name, p.read_text(errors="replace")))
    return out


def logged_landing(logs: list[tuple[str, str]], fn: str, commit: str) -> str | None:
    """The council grind log that lists this function as LANDED at this commit (its name only)."""
    for name, text in logs:
        for line in text.splitlines():
            if fn.split("__")[0] in line.split("|")[0] and "LANDED" in line and commit[:8] in line:
                return name
    return None


def process(cfg: Config, args, ctx: dict, c: dict, path: str, unit: dict) -> tuple[list[dict], dict]:
    """One file of one commit: its fix records and the counts of what was left out and why."""
    from .llmtry import asm_shape, classify, compact_asm, relative_rows
    from .regs import _diff
    from .trycmd import definition
    st: dict = {}

    def bump(k: str, n: int = 1) -> None:
        st[k] = st.get(k, 0) + n
    old = git(cfg, "show", f"{c['parent']}:{path}")
    new = git(cfg, "show", f"{c['commit']}:{path}")
    if not old or not new or old == new:
        bump("file new or unchanged")
        return [], st
    snap_p, tree_p = ctx["snaps"].at(c["parent"])
    snap_c, tree_c = ctx["snaps"].at(c["commit"])
    tag = sha(c["commit"], path)[:10]
    work = ctx["work"] / tag
    got: list[dict] = []
    try:
        key_c = sha("file", unit["name"], tree_c, new)
        after = ctx["cache"].get(key_c)
        if after is None:
            obj, err = compile_snap(cfg, snap_c, unit, new, work / "a")
            after = {"compiled": obj is not None, "fns": {k: list(v) for k, v in scored(cfg, obj, unit).items()} if obj else {}}
            ctx["cache"].put(key_c, after)
            shutil.rmtree(work / "a", ignore_errors=True)
        if not after["compiled"]:
            bump("after did not compile")
            return [], st
        key_p = sha("file", unit["name"], tree_p, old)
        before = ctx["cache"].get(key_p)
        obj_b = None
        want = [f for f, (pct, size) in after["fns"].items() if pct == 100.0 and not f.startswith(SKIP_NAMES)
                and size >= args.min_size]
        if before is None or before["compiled"]:
            obj_b, err = compile_snap(cfg, snap_p, unit, old, work / "b")
            before = {"compiled": obj_b is not None,
                      "fns": {k: list(v) for k, v in scored(cfg, obj_b, unit).items()} if obj_b else {}}
            if key_p not in ctx["cache"].rows:
                ctx["cache"].put(key_p, before)
        if not before["compiled"]:
            bump("before did not compile")
            return [], st
        headers = None
        for fn in want:
            b = before["fns"].get(fn)
            if b is None:
                bump("not in the before object (a new function)")
                continue
            if b[0] >= 100.0:
                continue                                  # exact before too: no fix here
            bump("became exact in the commit")
            so, sn = definition(old, fn), definition(new, fn)
            if so is None or sn is None:
                bump("definition not found (once)")
                continue
            d_old, d_new = old[so[0]:so[1]], new[sn[0]:sn[1]]
            if re.match(r"\s*(?:extern\s+\"C\"\s+)?asm\b", d_old) or re.match(r"\s*(?:extern\s+\"C\"\s+)?asm\b", d_new):
                bump("an asm stub before or after")
                continue
            from .llmdata import norm
            if norm(d_old) == norm(d_new):
                bump("own definition unchanged (fixed elsewhere)")
                continue
            if fn not in ctx["entries"]:
                bump("no target asm under this name today")
                continue
            if fn in ctx["quarantine"]:
                bump("quarantined by the judge audit")
                continue
            entry = ctx["entries"][fn]
            row_key = sha("row", unit["name"], tree_p, fn, old)
            hit = ctx["cache"].get(row_key)
            if hit is None:
                pair = _diff(cfg, unit, obj_b, fn) if obj_b is not None else None
                if obj_b is None:                          # cached file result, object gone: compile again
                    obj_b, _e = compile_snap(cfg, snap_p, unit, old, work / "b")
                    pair = _diff(cfg, unit, obj_b, fn) if obj_b is not None else None
                if pair is None:
                    bump("no diff pair")
                    continue
                hit = {"rows": relative_rows(pair[0], pair[1]), "diff_class": classify(pair[0], pair[1])}
                if not args.no_oracle:
                    ora = oracle_summary(snap_p, unit, fn, old)
                    if ora:
                        hit["oracle"] = ora
                ctx["cache"].put(row_key, hit)
            if headers is None:
                headers = [x for x in git(cfg, "diff", "--name-only", c["parent"], c["commit"], "--", "include").split()]
            rec = {"v": SCHEMA, "source": "shape-fix", "unit": unit["name"], "function": fn, "size": int(b[1]),
                   "commit": c["commit"], "parent": c["parent"], "commit_time": c["time"], "path": path,
                   "asm_shape": asm_shape(entry.text), "before": d_old, "after": d_new,
                   "before_percent": b[0], "after_percent": 100.0, "diff_class": hit["diff_class"], "rows": hit["rows"],
                   "oracle": hit.get("oracle"), "fix_kind": fix_kinds(d_old, d_new),
                   "headers_changed": headers[:20], "grind_log": logged_landing(ctx["logs"], fn, c["commit"]),
                   "asm": compact_asm(entry.text)}
            got.append(rec)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return got, st


def landing_logs(cfg: Config) -> Path:
    """The grind logs that name landings: [platform.gate] landing_logs, relative to the data dir's parent."""
    from .gate import _gate_conf
    from .llmtry import data_dir
    return data_dir(cfg).parent / _gate_conf(cfg).get("landing_logs", "logs")


def read_records(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []


def card(out: Path, recs: list[dict], state: dict, fz: dict, stats: dict) -> str:
    kinds: dict[str, int] = {}
    for r in recs:
        for k in r["fix_kind"]:
            kinds[k] = kinds.get(k, 0) + 1
    by_class: dict[str, int] = {}
    for r in recs:
        by_class[r["diff_class"]] = by_class.get(r["diff_class"], 0) + 1
    orc = sum(1 for r in recs if r.get("oracle"))
    commits = sorted({r["commit"] for r in recs})
    times = [r["commit_time"] for r in recs]
    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
    lines = [
        "# shape-fixes v0", "",
        "Functions that became exact in a landing commit of the project: the definition before and after, the objdiff rows of",
        "the before version against the target, its fuzzy score, the oracle summary, a fix kind. Generated by",
        "`./dk dataset fixes` (tools/dk/platforms/gc_mwcc/llmfixes.py). Game-derived code: private, never commit or publish.", "",
        f"- records: **{len(recs)}** in {len(commits)} commits, {len({r['function'] for r in recs})} functions, "
        f"{len({r['unit'] for r in recs})} units",
        f"- commits covered: {fmt(min(times)) if times else '-'} .. {fmt(max(times)) if times else '-'}; "
        f"state.json lists {len(state['done'])} commits done (up to {state.get('last_commit', '')[:10]})",
        f"- diff class of the before version: " + ", ".join(f"{k} {v}" for k, v in sorted(by_class.items())),
        f"- with an oracle summary: {orc}",
        f"- fix kinds (a record can have several): " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda x: -x[1])),
        "", "## Fields (jsonl, one record per line)", "",
        "unit, function, size, commit, parent, commit_time, path, asm_shape, before, after (the function's definition),",
        "before_percent, after_percent (100), diff_class (register|shape), rows (target | ours rows around each difference),",
        "oracle ({explain, reach} short forms of `regs --explain --oracle` / `regs --reach` on the before version, or null),",
        "fix_kind, headers_changed (include/ files the commit touched), grind_log (the grind log that lists the landing, a name only),",
        "asm (the target's compact listing). `before`/`after` read as `wrong`/`exact` in `dataset` records.", "",
        "No model reasoning text: commit messages and grind-log notes are not copied (the fix kind is computed from the two",
        "definitions). Both versions are compiled with the headers of their own commit (snapshot trees), so before is what the",
        "repo held. Declarations (`decls`) are not stored: render them at export (llm-try's prepare over the before source).", "",
        "## Sources", "",
        f"git history of the project, non-merge commits {stats['range']}; per commit, every changed unit source; a function counts when",
        "its score is below 100 before and 100 after, it is defined in C both times (not an asm stub) and its own definition changed.",
        "", "## Left out (this run, by reason)", ""]
    lines.insert(len(lines) - 2, f"(functions that went from below 100 to exact in the commits read this run: "
                 f"{stats['left'].get('became exact in the commit', 0)})")
    for k, v in sorted(stats["left"].items(), key=lambda x: -x[1]):
        if k != "became exact in the commit":
            lines.append(f"- {k}: {v}")
    lines += ["", "## Leakage note", "",
              "Records are dropped when their unit (bench, heldout, repair-eval, review-eval sets only; the near-samples and giants by function, shape and hash), function name, asm shape (instruction mnemonics and operand kinds) or the content hash",
              "of the before/after definition is in a frozen eval set. Sets found in llm-data/ (functions): " +
              ", ".join(f"{k} {v}" if v is not None else f"{k} (absent)" for k, v in fz["sets"].items()) + ".",
              f"Union: {len(fz['units'])} units, {len(fz['functions'])} functions, {len(fz['shapes'])} asm shapes, {len(fz['hashes'])} definition hashes.",
              f"Dropped this run: " + (", ".join(f"{k} {v}" for k, v in sorted(stats['leaks'].items())) or "none") + ".",
              "Functions the judge audit quarantines (llm-data/judge-audit.json) are left out too. The records already in fixes.jsonl",
              "are filtered again at every run, so a newly frozen set removes its records.", "",
              "## Re-run", "",
              "Re-run after every grind day: `./dk dataset fixes --resume` (in a worktree, <= 4 jobs; the oracle needs `DK_ORACLE`). It is",
              "incremental by commit range (state.json holds the commits done; `--since DATE` or `--range A..B` choose another span,",
              "`--redo` recomputes). fixes.jsonl, CARD.md and SHA256SUMS are rewritten each time; copy the directory to v1 for a frozen version.", ""]
    return "\n".join(lines)


def run(cfg: Config, args: argparse.Namespace) -> int:
    from .judge import forbidden_rules  # noqa: F401  (import check: the toolkit's judge is present)
    from .llmdata import quarantined
    from .llmtry import asm_entries, data_dir
    from .matching import load_objdiff_units
    t0 = time.time()
    out = Path(args.out).expanduser() if args.out else data_dir(cfg) / "datasets" / "shape-fixes" / "v0"
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() and not args.redo else {"done": [], "last_commit": ""}
    done = set(state["done"])
    commits = [c for c in landing_commits(cfg, args) if c["commit"] not in done]
    if args.limit:
        commits = commits[:args.limit]
    units = {u["metadata"]["source_path"]: u for u in load_objdiff_units(cfg) if u.get("base_path")}
    fz = frozen(cfg)
    if not fz["units"]:
        print("error: no frozen eval set found in llm-data/ (bench.json, heldout.json): refusing to build")
        return USAGE
    from .llmtry import LLM_DIR
    ctx = {"snaps": Snapshots(cfg, Path(args.scratch).expanduser() if args.scratch else cfg.root / LLM_DIR / "fixsnaps"),
           "work": cfg.root / LLM_DIR / "work" / f"fixes{int(t0)}", "cache": Cache(out / "cache" / "fixes.jsonl"),
           "entries": asm_entries(cfg), "quarantine": quarantined(data_dir(cfg) / "judge-audit.json") or set(),
           "logs": grind_logs(Path(args.logs).expanduser() if args.logs else landing_logs(cfg))}
    jobs: list[tuple[dict, str, dict]] = []
    for c in commits:
        files = git(cfg, "diff", "--name-only", c["parent"], c["commit"], "--", "src").split()
        jobs += [(c, f, units[f]) for f in files if f in units]
    print(f"{len(commits)} commits to read, {len(jobs)} unit sources changed ({time.time() - t0:.0f} s)")
    stats = {"left": {}, "leaks": {}, "range": args.range or f"since {args.since}"}
    lock = threading.Lock()
    new: list[dict] = []

    def one(j):
        c, f, u = j
        try:
            recs, st = process(cfg, args, ctx, c, f, u)
        except Exception as exc:                            # one bad file never stops the run
            recs, st = [], {f"error {type(exc).__name__}": 1}
        with lock:
            for k, v in st.items():
                stats["left"][k] = stats["left"].get(k, 0) + v
            new.extend(recs)
    with ThreadPoolExecutor(max_workers=min(args.jobs, 4)) as pool:
        list(pool.map(one, jobs))
    old = read_records(out / "fixes.jsonl")
    by_key = {(r["commit"], r["function"]): r for r in old}
    by_key.update({(r["commit"], r["function"]): r for r in new})
    kept = []
    for r in by_key.values():
        why = leak(r, fz)
        if why:
            stats["leaks"][why] = stats["leaks"].get(why, 0) + 1
        else:
            kept.append(r)
    kept.sort(key=lambda r: (r["commit_time"], r["unit"], r["function"]))
    (out / "fixes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    state["done"] = sorted(done | {c["commit"] for c in commits})
    if commits:
        state["last_commit"] = commits[-1]["commit"]
    state["schema"] = SCHEMA
    state_path.write_text(json.dumps(state, indent=1) + "\n")
    (out / "CARD.md").write_text(card(out, kept, state, fz, stats))
    sums = [f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}" for n in ("fixes.jsonl", "CARD.md", "state.json")]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    shutil.rmtree(ctx["work"], ignore_errors=True)
    print(f"fixes: {len(new)} new records, {len(kept)} kept in total in {out}; leaks dropped {stats['leaks']}; "
          f"wall {time.time() - t0:.0f} s")
    for k, v in sorted(stats["left"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    return OK


def configure(sub) -> None:
    lc = sub.add_parser("leakcheck", help="the leakage filter (frozen sets, gate-lora2, definition and variant hashes) over a "
                        "seeded sample of pairs from the existing datasets: lists hits, exit 1 on any")
    lc.add_argument("files", nargs="*", help="jsonl files (default llm-data/datasets/**/*.jsonl and dataset/**/*.jsonl)")
    lc.add_argument("--sample", type=int, default=500, help="pairs to test (default 500)")
    lc.add_argument("--seed", type=int, default=20261106, help="sampling seed")
    f = sub.add_parser("fixes", help="shape-fix records from landing commits: the function before and after, the before "
                       "version's diff rows, oracle summary and fix kind -> llm-data/datasets/shape-fixes/v0 (incremental)")
    f.add_argument("--since", default=DEFAULT_SINCE, help=f"commits since this date (default {DEFAULT_SINCE})")
    f.add_argument("--range", default="", help="a git revision range A..B instead of --since")
    f.add_argument("--resume", action="store_true", help="only commits not in state.json (the default; kept for scripts)")
    f.add_argument("--redo", action="store_true", help="ignore state.json and recompute every commit in the span")
    f.add_argument("--no-oracle", action="store_true", help="no oracle summaries (no DK_ORACLE runner)")
    f.add_argument("--min-size", type=lambda x: int(x, 0), default=0x10, help="skip smaller functions (default 0x10)")
    f.add_argument("--jobs", type=int, default=4, help="parallel files (default and at most 4)")
    f.add_argument("--limit", type=int, default=0, help="first N commits only (a test)")
    f.add_argument("--out", default="", help="output directory (default llm-data/datasets/shape-fixes/v0)")
    f.add_argument("--scratch", default="", help="where the snapshot trees live (default under the repo's llm work dir)")
    f.add_argument("--logs", default="",
                   help="grind logs, to name the one that lists a landing (default: [platform.gate] landing_logs "
                        "beside the data dir)")
