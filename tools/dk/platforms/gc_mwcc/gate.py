"""The frozen gate sets: functions nothing may train on, tune on or grind.

`llm-data/gate-lora2/` holds before-version snapshots (commit hash, definition hash, the function's source then) of

  gate.json                       the LoRA-2 gate: 30 functions Claude gave up on (the give-up logs) + 30 never tried
  near-sample-60.snapshot.json    the Phase 1c sample, snapshotted the same way
  near-sample-fresh-30.snapshot.json

`frozen_keys` (unit, function) is what the grind queues skip: `dk route` (worker "frozen"), `dk next` and
`dk brief` (marked), `llm-grind`, `close --near` (swept and applied nowhere). `never` and `hashes` feed the
leakage filters (`llmtry.never_train`, `llmdata.leaks`, `dataset fixes`): function names, asm shapes, definition
hashes, and the hashes of each definition's one-step rewrite variants (the inverse generators' output), with and
without the function's own name.

  ./dk llm-bench gate-freeze [--seed N] [--force]     build the gate and the two snapshots
  ./dk llm-bench gate-freeze --refresh [--dry-run]    retire gate functions exact today, redraw as many by the same
                                                      method (the refresh logs, next seeds); the rest stays.
                                                      Draws never hold a function an existing training or review export
                                                      has (name, address or definition hash); --retire FN also drops a
                                                      still-open one, --retire-trained every such gate function
  ./dk llm-bench gate-freeze --rehash                 the variant hashes again with today's rewrite table (new rows
                                                      make new inverse variants): variant-hashes.json beside the sets,
                                                      read by `hashes`; the sets themselves are not touched
  ./dk dataset leakcheck [--sample 500]               the filter over pairs from the existing datasets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

from ...core.command import MISMATCH, OK, USAGE
from ...core.config import Config

SCHEMA = 1
GATE_NAME = "gate-lora2"
SNAPSHOT_SETS = ("near-sample-60", "near-sample-fresh-30")      # llm-data/<name>.json -> gate-lora2/<name>.snapshot.json
MAX_SIZE = 2048                                                  # the gate's functions: at most 2 KB
GIVEUP_N, FRESH_N = 30, 30
GIVEUP_LOGS = ("*",)             # the first freeze's give-up logs: [platform.gate] giveup_logs (folder globs)
REFRESH_LOGS = ("*",)            # the logs a refresh reads: [platform.gate] refresh_logs (folder globs)
FRESH_CLASSES = ("near-register", "near-shape")
VARIANT_CAP = 60                                                 # one-step rewrite variants hashed per function
REHASH_FILE = "variant-hashes.json"                              # --rehash: the variants of today's table, per set
REHASH_PER_RULE, REHASH_CAP = 6, 400                             # every row gets its share (the old cap starved the late rows)
_FENCE = re.compile(r"```[A-Za-z+]*\n(.*?)```", re.S)
_NAME = re.compile(r"([~\w:]+)\s*\(")
_IDENT = re.compile(r"[A-Za-z_]\w*")


# ---- hashes -----------------------------------------------------------------------------------------

def norm_hash(code: str) -> str:
    from .llmdata import norm
    return hashlib.sha1(norm(code).encode()).hexdigest()


def agnostic_hash(code: str) -> str:
    """The hash of a definition with its own name blanked: `dk rename` and a copy under another name hash alike."""
    from .llmdata import norm
    brace = code.find("{")
    m = _NAME.search(code[:brace] if brace >= 0 else code)
    text = re.sub(r"(?<![\w:~])" + re.escape(m[1]) + r"(?!\w)", "FN", code) if m else code
    return hashlib.sha1(norm(text).encode()).hexdigest()


def code_hashes(code: str) -> set[str]:
    return {norm_hash(code), agnostic_hash(code)}


def fenced(text: str) -> list[str]:
    return _FENCE.findall(text)


# ---- the sets on disk -------------------------------------------------------------------------------

def gate_dir(cfg: Config) -> Path:
    from .llmtry import data_dir
    return data_dir(cfg) / GATE_NAME


def set_files(cfg: Config) -> dict[str, Path]:
    """name -> file of every function-level frozen set that exists: the gate, the snapshots, and the
    original near-sample lists (their functions stay frozen even when no snapshot was made)."""
    from .llmtry import data_dir
    d = gate_dir(cfg)
    out = {GATE_NAME: d / "gate.json"}
    for name in SNAPSHOT_SETS:
        out[name] = d / f"{name}.snapshot.json"
        out[name + ":list"] = data_dir(cfg) / f"{name}.json"
    released = frozen_config(cfg)[2].resolve()
    for p in sorted((data_dir(cfg) / "eval").glob("*.json")):        # frozen eval sets (llm-bench t63-freeze)
        if p.resolve() != released:                                  # the release file is not a set
            out["eval:" + p.stem] = p
    return {k: p for k, p in out.items() if p.is_file()}


def load_sets(cfg: Config) -> dict[str, list[dict]]:
    sets: dict[str, list[dict]] = {}
    for name, path in set_files(cfg).items():
        doc = json.loads(path.read_text())
        sets.setdefault(name.split(":")[0], []).extend(doc["functions"] if isinstance(doc, dict) else doc)
    return sets


# Every frozen set, in label order (the first label holding a function names it). dk.toml
# `[platform.frozen]` overrides: `data` (dir, else llm-data beside the main checkout), `released`
# (relative to it) and `[platform.frozen.sets]` label -> [glob[#key,key]]; `*` in a label is the file stem.
DEFAULT_SETS = {
    GATE_NAME: ["gate-lora2/gate.json"],
    "near-sample-60": ["gate-lora2/near-sample-60.snapshot.json", "near-sample-60.json"],
    "near-sample-fresh-30": ["gate-lora2/near-sample-fresh-30.snapshot.json", "near-sample-fresh-30.json"],
    "eval:*": ["eval/*.json"],
    "bench": ["bench.json"],
    "heldout": ["heldout.json"],
    "heldout-split": ["heldout-split.json#dev,final"],
    "repair-eval": ["repair-eval.json"],
    "review-eval": ["review-eval.json"],
}
DEFAULT_RELEASED = "eval/released.json"


def frozen_config(cfg: Config) -> tuple[Path, dict, Path]:
    """(data dir, sets, release file) from dk.toml `[platform.frozen]`, else the defaults above."""
    from .llmtry import data_dir
    conf = (getattr(cfg, "platform_options", None) or {}).get("frozen") or {}
    d = data_dir(cfg)                                    # DK_LLM_DATA, else llm-data beside the main checkout
    if conf.get("data") and not os.environ.get("DK_LLM_DATA"):
        d = Path(conf["data"]).expanduser()
        d = d if d.is_absolute() else cfg.root / d
    return d, dict(conf.get("sets") or DEFAULT_SETS), d / conf.get("released", DEFAULT_RELEASED)


def frozen_keys(cfg: Config):
    """The one frozen union: {function: set} of every eval set, minus llm-data/eval/released.json.
    Also answers (unit, function) lookups by the function alone (core.frozen.FrozenMap), so a set file's
    stale unit name never hides a function. Every reader (brief, next, route, close, close --queue,
    llm-grind, t63, the commit guard) uses this."""
    from ...core import frozen
    d, sets, released = frozen_config(cfg)
    return frozen.union(d, sets, released)


def released_functions(cfg: Config) -> dict[str, dict]:
    from ...core import frozen
    return frozen.load_released(frozen_config(cfg)[2])


def frozen_definitions(cfg: Config) -> tuple[dict[str, list[tuple[str, int | None]]], str]:
    """For the commit guard: each frozen function -> how its definition is written in source
    ((qualified name, parameter count) from its mangled name; a plain name as itself, and its
    fn_<address> name from symbols.txt), and the release file to name in a refusal."""
    from .demangle import demangle
    names = frozen_keys(cfg)
    by_name: dict[str, int] = {}
    by_addr: dict[int, str] = {}
    try:
        from .dtk_symbols import read_symbols
        path = cfg.root / (getattr(cfg, "platform_options", None) or {}).get("symbols", "")
        for s in read_symbols(path) if path.is_file() else []:
            if s.section == ".text":
                by_name[s.name] = s.address
                by_addr.setdefault(s.address, s.name)
    except Exception:
        pass

    def forms_of(name: str) -> list[tuple[str, int | None]]:
        f = demangle(name)
        if f is None:
            return [(name, None)]
        last = f.quals[-1] if f.quals else ""
        return [("::".join((*f.quals, {"ctor": last, "dtor": "~" + last}.get(f.name, f.name))), len(f.params))]

    out: dict[str, list[tuple[str, int | None]]] = {}
    for fn in names:
        # the set's name, the address name, and today's symbol name (a set may hold either)
        m = re.fullmatch(r"fn_([0-9A-Fa-f]{8})", fn)
        address = int(m[1], 16) if m else by_name.get(fn)
        aliases = [fn] + ([f"fn_{address:08X}", by_addr.get(address, "")] if address is not None else [])
        forms: list[tuple[str, int | None]] = []
        for a in dict.fromkeys(x for x in aliases if x):
            forms += [x for x in forms_of(a) if x not in forms]
        out[fn] = forms
    return out, cfg.rel(frozen_config(cfg)[2]) if hasattr(cfg, "rel") else str(frozen_config(cfg)[2])


def never(cfg: Config) -> tuple[dict[str, str], dict[str, str]]:
    """({function: set}, {asm shape: set}) of the function-level frozen sets (units stay open to training)."""
    from .llmtry import asm_entries, asm_shape
    entries = None
    fns: dict[str, str] = {}
    shapes: dict[str, str] = {}
    for name, rows in load_sets(cfg).items():
        for r in rows:
            fn = r.get("function")
            if not fn:
                continue
            fns.setdefault(fn, name)
            shape = r.get("asm_shape")
            if not shape:
                entries = asm_entries(cfg) if entries is None else entries
                shape = asm_shape(entries[fn].text) if fn in entries else None
            if shape:
                shapes.setdefault(shape, name)
    return fns, shapes


def hashes(cfg: Config) -> dict[str, str]:
    """{content hash: set}: the stored definitions and their variants (name-keeping and name-blank), and the
    variants of the current rewrite table (`--rehash`, variant-hashes.json) when they were made."""
    out: dict[str, str] = {}
    extra = gate_dir(cfg) / REHASH_FILE
    if extra.is_file():
        for name, fns in json.loads(extra.read_text()).get("sets", {}).items():
            for hs in fns.values():
                for h in hs:
                    out.setdefault(h, name)
    for name, rows in load_sets(cfg).items():
        for r in rows:
            for h in r.get("hashes", []):
                out.setdefault(h, name)
            for k in ("before", "after"):
                if r.get(k):
                    for h in code_hashes(r[k]):
                        out.setdefault(h, name)
    return out


# ---- what earlier exports already trained on -----------------------------------------------------------

TRAINED_GLOBS = ("**/train*.jsonl", "**/dpo*.jsonl", "**/records/*.jsonl", "**/review/eval.jsonl")
TRAINED_SKIP = ("cache", "llm-train-smoke", GATE_NAME)           # prompt caches, bundle copies, the gate itself
_ADDR = re.compile(r"fn_([0-9A-Fa-f]{8})")
_ASM_FN = re.compile(r"\.fn ([^\s,\\\"]+)")                    # an asm header: the function the row is about (calls are not)
_CODE_KEYS = ("exact", "wrong", "before", "after", "chosen", "rejected")


def _line_hashes(r: dict, out: set) -> None:
    for k in _CODE_KEYS:
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            out |= code_hashes(v)
            for blk in fenced(v):
                out |= code_hashes(blk)
    for m in r.get("messages") or ():
        c = m.get("content")
        if m.get("role") != "system" and isinstance(c, str):
            for blk in fenced(c):
                out |= code_hashes(blk)


def trained_index(cfg: Config) -> tuple[set[str], set[str], set[str]]:
    """(function names, address tokens `8000ABCD`, definition hashes) every existing training or review export holds:
    llm-data/**/train*.jsonl, dpo*.jsonl, review/eval.jsonl and the structured records/*.jsonl. Names come from the
    `function` field and the `.fn NAME` asm headers (a `bl fn_ADDR` call in some other function's asm is not a row for
    it), the address tokens from those names; hashes (name-keeping and name-blank) off the code
    fields and the fenced blocks of the chat messages. Cached in build/dk/gate-trained.json by file size and mtime."""
    from .llmtry import data_dir
    base = data_dir(cfg)
    files = sorted({p for g in TRAINED_GLOBS for p in base.glob(g)
                    if p.is_file() and not any(x in p.relative_to(base).parts for x in TRAINED_SKIP)})
    stamp = {str(p.relative_to(base)): [p.stat().st_size, p.stat().st_mtime_ns] for p in files}
    cache = Path(cfg.root) / "build" / "dk" / "gate-trained.json"
    try:
        c = json.loads(cache.read_text())
        if c.get("v") == 2 and c.get("stamp") == stamp:
            return set(c["names"]), set(c["addrs"]), set(c["hashes"])
    except (OSError, ValueError, KeyError):
        pass
    names: set[str] = set()
    addrs: set[str] = set()
    hs: set[str] = set()
    for p in files:
        with p.open(errors="replace") as f:
            for line in f:
                names.update(_ASM_FN.findall(line))
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(r, dict):
                    continue
                for src in (r, r.get("tags") or {}):
                    if isinstance(src.get("function"), str):
                        names.add(src["function"])
                _line_hashes(r, hs)
    for n in names:
        addrs.update(a.upper() for a in _ADDR.findall(n))
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"v": 2, "stamp": stamp, "names": sorted(names), "addrs": sorted(addrs), "hashes": sorted(hs)}))
    except OSError:
        pass
    return names, addrs, hs


def trained_bar(cfg: Config):
    """A predicate over an open function row: True when an existing export already holds it, by name, by address token
    or by the hash of its definition today (name-keeping or name-blank). Used to keep such functions out of the pools."""
    from .trycmd import definition
    names, addrs, hs = trained_index(cfg)
    srcs: dict = {}

    def bar(row: dict) -> bool:
        fn = row["function"]
        if fn in names or ("%08X" % row["address"] if isinstance(row.get("address"), int) else "") in addrs:
            return True
        m = _ADDR.match(fn)
        if m and m[1].upper() in addrs:
            return True
        path = Path(cfg.root) / row["source"] if row.get("source") else None
        if path is not None and path.is_file():
            src = srcs.setdefault(row["source"], path.read_text(errors="replace"))
            span = definition(src, fn)
            if span is not None and code_hashes(src[span[0]:span[1]]) & hs:
                return True
        return False
    return bar


# ---- building the sets ------------------------------------------------------------------------------

def _gate_conf(cfg: Config) -> dict:
    return dict((getattr(cfg, "platform_options", None) or {}).get("gate") or {})


def council_dir(cfg: Config) -> Path:
    """The grind logs (one folder per run, `*.md` inside): [platform.gate] logs, relative to the data dir's parent
    (default `logs`)."""
    from .llmtry import data_dir
    return data_dir(cfg).parent / _gate_conf(cfg).get("logs", "logs")


def giveup_globs(cfg: Config) -> tuple:
    return tuple(_gate_conf(cfg).get("giveup_logs") or GIVEUP_LOGS)


def refresh_globs(cfg: Config) -> tuple:
    return tuple(_gate_conf(cfg).get("refresh_logs") or REFRESH_LOGS)


def _expand(line: str) -> list[str]:
    """Names on a log line, `fn_80001DC8/E98/7058` read as three functions."""
    names = _IDENT.findall(line)
    for m in re.finditer(r"\bfn_([0-9A-F]{8})((?:/[0-9A-F]{3,4})+)", line):
        base = m[1]
        for tail in m[2].strip("/").split("/"):
            names.append("fn_" + base[:8 - len(tail)] + tail)
    return names


def giveup_names(text: str) -> list[str]:
    """Functions a council log says were given up on (the names before the first ` | ` or ` (`, in order)."""
    out: list[str] = []
    for line in text.splitlines():
        if not re.search(r"gave up", line, re.I):
            continue
        body = line.lstrip("- ")
        head = body.split(" | ")[0] if " | " in body else re.split(r" \(| start ", body)[0]
        head = head.replace(" + ", " ").replace(" / ", " ")
        out += [n for n in _expand(head) if n.startswith(("fn_", "vf", "__")) or "__" in n]
    return list(dict.fromkeys(out))


def resolve(names, by_name: dict, by_addr: dict) -> list[str]:
    """Log names -> today's open function names: by name, else by the `fn_ADDR` prefix (a later `dk rename`
    gave the function a mangled name)."""
    out = []
    for n in names:
        if n in by_name:
            out.append(n)
            continue
        m = re.match(r"fn_([0-9A-F]{8})", n)
        if m and int(m[1], 16) in by_addr:
            out.append(by_addr[int(m[1], 16)])
    return list(dict.fromkeys(out))


def mentioned(text: str, by_name: dict, by_addr: dict | None = None) -> set[str]:
    return set(resolve(_expand(text), by_name, by_addr or {}))


def pick(pool: list[str], n: int, seed: int) -> list[str]:
    pool = sorted(set(pool))
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


def head_commit(cfg: Config) -> str:
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cfg.root, capture_output=True, text=True)
    return p.stdout.strip()


def variants(src: str, symbol: str, span: tuple[int, int], cplusplus: bool, cap: int = VARIANT_CAP,
             per_rule: int = 6, accessors: dict | None = None) -> list[str]:
    """One-step rewrite-table variants of a definition (what the inverse generators would make of it)."""
    from . import rewrites
    from ...core import cbody
    try:
        body = cbody.parse_body(src, src.index("{", span[0], span[1]), cbody.type_names(src))
        site = rewrites.Site(src, body, None, cplusplus, accessors=accessors)
    except (ValueError, cbody.ParseError):
        return []
    out: list[str] = []
    for rule in rewrites.RULES:
        try:
            got = list(rewrites.candidates(site, [rule], groups=False))
        except Exception:
            continue
        for _name, _label, new in got[:per_rule]:
            chunk = new[span[0]:span[1] + len(new) - len(src)]
            if chunk.rstrip().endswith("}"):
                out.append(chunk)
        if len(out) >= cap:
            break
    return out[:cap]


def snapshot(cfg: Config, row: dict, why: str, head: str, entries: dict, srcs: dict) -> dict:
    """One before-version record for an open function row (unit, function, source, address, size, fuzzy)."""
    from .llmtry import asm_shape
    from .trycmd import definition
    rec = {"unit": row["unit"], "function": row["function"], "address": row["address"], "size": row["size"],
           "fuzzy": round(row["fuzzy"], 2), "source": row.get("source", ""), "why": why, "commit": head}
    if row.get("class"):
        rec["class"] = row["class"]
    if row.get("attempts") is not None:
        rec["attempts"] = row["attempts"]
    fn = row["function"]
    if fn in entries:
        rec["asm_shape"] = asm_shape(entries[fn].text)
    path = cfg.root / rec["source"] if rec["source"] else None
    if path is not None and path.is_file():
        rec["source_blob"] = subprocess.run(["git", "hash-object", str(path)], cwd=cfg.root, capture_output=True,
                                            text=True).stdout.strip()
        src = srcs.setdefault(rec["source"], path.read_text(errors="replace"))
        span = definition(src, fn)
        if span is not None:
            text = src[span[0]:span[1]]
            rec["before"] = text
            rec["definition_hash"] = norm_hash(text)
            rec["hashes"] = sorted(code_hashes(text).union(*(code_hashes(v) for v in variants(
                src, fn, span, rec["source"].endswith((".cpp", ".cp"))))))
    return rec


def open_rows(cfg: Config) -> list[dict]:
    """`dk route`'s table (class, attempts, best) merged with the source path, free data."""
    from ...core.nextcmd import queue
    from ...core.progress import load_report
    from . import route
    report = load_report(cfg)
    if report is None:
        return []
    src = {(f["unit"], f["function"]): f.get("source", "") for f in queue(report)[0]}
    rows = route.table(cfg) or []
    return [{**r, "source": src.get((r["unit"], r["function"]), "")} for r in rows]


def pools(cfg: Config, council: Path, rows: list[dict], globs=None, skip: set = frozenset(), bar=None):
    """(gave-up pool, never-tried pool, stats) by the gate's method: the gave-up pool is the functions the give-up
    logs name, the never-tried pool the near-register/near-shape ones with 0 attempts the logs never name; both still
    open, <= 2 KB, not blocked, outside the near-samples and `skip`. The never-tried pool still holds gave-up names
    a pick may take: the caller removes its own picks. `bar(row)` true keeps a function out of both pools."""
    globs = giveup_globs(cfg) if globs is None else globs
    by_name = {r["function"]: r for r in rows}
    logs = sorted(p for g in globs for p in council.glob(g + "/*.md"))
    text = "\n".join(p.read_text() for p in logs)
    by_addr = {r["address"]: r["function"] for r in rows}
    gave_up = resolve([n for p in logs for n in giveup_names(p.read_text())], by_name, by_addr)
    tried = mentioned(text, by_name, by_addr)
    near_sets = {r["function"] for rs in (json.loads(p.read_text())["functions"] for k, p in set_files(cfg).items()
                                          if k.endswith(":list")) for r in rs}
    ok = lambda r: r["size"] <= MAX_SIZE and r["fuzzy"] < 100.0 and not r["function"].startswith(("@", "__sinit")) \
        and r["worker"] != "blocked" and r["function"] not in skip and not (bar and bar(r))
    pool_a = [n for n in gave_up if n in by_name and ok(by_name[n]) and n not in near_sets]
    pool_b = [r["function"] for r in rows if ok(r) and r["class"] in FRESH_CLASSES and r["attempts"] == 0
              and r["function"] not in tried and r["function"] not in near_sets]
    return pool_a, pool_b, {"gave_up_logged": len(set(gave_up)), "tried_in_logs": len(tried),
                            "logs": [p.parent.name + "/" + p.name if len(globs) > 1 else p.name for p in logs]}


def build(cfg: Config, seed: int, council: Path, rows: list[dict], head: str, entries: dict) -> tuple[list[dict], dict]:
    """(gate records, stats). Pure of the filesystem apart from reading the council logs and sources."""
    by_name = {r["function"]: r for r in rows}
    pool_a, pool_b, st = pools(cfg, council, rows)
    giveups = pick(pool_a, GIVEUP_N, seed)
    pool_b = [n for n in pool_b if n not in giveups]
    fresh = pick(pool_b, FRESH_N, seed + 1)
    srcs: dict = {}
    recs = [snapshot(cfg, by_name[n], "gave-up", head, entries, srcs) for n in giveups] + \
           [snapshot(cfg, by_name[n], "never-tried", head, entries, srcs) for n in fresh]
    return recs, {"gave_up_logged": st["gave_up_logged"], "gave_up_pool": len(pool_a), "fresh_pool": len(pool_b),
                  "tried_in_logs": st["tried_in_logs"], "logs": st["logs"]}


def redraw(cfg: Config, doc: dict, council: Path, rows: list[dict], head: str, entries: dict,
           retire: tuple = (), bar=None) -> dict:
    """One refresh of the gate document (returned changed, `doc` untouched): records whose function is no longer an
    open row (exact today) move to `retired`; as many new ones per half are drawn by the gate's method from the
    pools minus everything the gate ever held, with seeds seed + 2k / seed + 2k + 1 for the k-th refresh; the
    refresh is logged in `refreshes`. Records still open are kept as they are (their commit and before stay).
    `retire` names functions to drop although still open (found in an earlier export); `bar` (see `trained_bar`) keeps
    functions earlier exports hold out of the draws."""
    open_now = {r["function"] for r in rows if r["fuzzy"] < 100.0} | {r["address"] for r in rows if r["fuzzy"] < 100.0}
    still = lambda r: r["function"] not in retire and (r["function"] in open_now or r.get("address") in open_now)
    keep = [r for r in doc["functions"] if still(r)]
    gone = [r for r in doc["functions"] if not still(r)]
    out = {**doc, "functions": keep, "retired": list(doc.get("retired", [])), "refreshes": list(doc.get("refreshes", []))}
    if not gone:
        return out
    k = len(out["refreshes"]) + 1
    seed_a, seed_b = doc["seed"] + 2 * k, doc["seed"] + 2 * k + 1
    ever = {r["function"] for r in doc["functions"]} | {r["function"] for r in out["retired"]}
    pool_a, pool_b, st = pools(cfg, council, rows, refresh_globs(cfg), ever, bar)
    need = {w: sum(r["why"] == w for r in gone) for w in ("gave-up", "never-tried")}
    new_a = pick(pool_a, need["gave-up"], seed_a)
    pool_b = [n for n in pool_b if n not in new_a]
    new_b = pick(pool_b, need["never-tried"], seed_b)
    by_name = {r["function"]: r for r in rows}
    srcs: dict = {}
    added = [snapshot(cfg, by_name[n], "gave-up", head, entries, srcs) for n in new_a] + \
            [snapshot(cfg, by_name[n], "never-tried", head, entries, srcs) for n in new_b]
    day = time.strftime("%Y-%m-%d")
    out["retired"] += [{**r, "retired": day, "retired_why": "trained earlier" if r["function"] in retire else f"exact at {head[:10]}"}
                       for r in gone]
    out["functions"] = keep + added
    out["refreshes"].append({"date": day, "commit": head, "retired": [r["function"] for r in gone],
                             "added": [r["function"] for r in added], "seeds": [seed_a, seed_b],
                             "trained_earlier": [r["function"] for r in gone if r["function"] in retire],
                             "gave_up_pool": len(pool_a), "fresh_pool": len(pool_b), "logs": st["logs"]})
    return out


def history_before(cfg: Config, path: str, names: list[str], now: str, depth: int = 40) -> tuple[str, str] | None:
    """(definition, commit) of the newest version of a function that differs from `now` (its exact text today),
    walking the commits that touched its source back from HEAD: the before-version of the landing that made it exact."""
    from .llmdata import norm
    from .trycmd import definition
    log = subprocess.run(["git", "log", "--format=%H", f"-n{depth}", "--", path], cwd=cfg.root, capture_output=True,
                         text=True).stdout.split()
    target = norm(now)
    for c in log:
        src = subprocess.run(["git", "show", f"{c}:{path}"], cwd=cfg.root, capture_output=True, text=True,
                             errors="replace").stdout
        for n in names:
            span = definition(src, n)
            if span is not None:
                text = src[span[0]:span[1]]
                if norm(text) != target:
                    return text, c
                break
    return None


def snapshot_set(cfg: Config, name: str, rows: list[dict], head: str, entries: dict) -> list[dict]:
    from .llmtry import data_dir
    doc = json.loads((data_dir(cfg) / f"{name}.json").read_text())
    by_key = {(r["unit"], r["function"]): r for r in rows}
    srcs: dict = {}
    out = []
    from .matching import load_objdiff_units
    sources = {u["name"]: u["metadata"].get("source_path", "") for u in load_objdiff_units(cfg)}
    fixes = {}
    fp = data_dir(cfg) / "datasets" / "shape-fixes" / "v0" / "fixes.jsonl"
    if fp.is_file():
        for line in fp.read_text().splitlines():
            x = json.loads(line)
            fixes[x["function"]] = x
    for f in doc["functions"]:
        r = by_key.get((f["unit"], f["function"]))
        if r is None:                                   # exact since the sample was drawn: the version before its fixing commit
            r = {**f, "source": sources.get(f["unit"], ""), "class": None, "attempts": f.get("attempts")}
            rec = {**snapshot(cfg, r, name, head, entries, srcs), "exact_now": True}
            rec["after"] = rec.pop("before", None)      # what the source holds now is the exact version
            fx = fixes.get(f["function"])
            got = (fx["before"], fx["parent"]) if fx else None
            if got is None and rec.get("after") and r["source"]:
                got = history_before(cfg, r["source"], [f["function"], "fn_%08X" % f["address"]], rec["after"])
            if got:
                rec.update(before=got[0], before_commit=got[1], definition_hash=norm_hash(got[0]))
                rec["hashes"] = sorted(set(rec.get("hashes", [])) | code_hashes(got[0]))
            out.append(rec)
        else:
            out.append(snapshot(cfg, r, name, head, entries, srcs))
    return out


def card(doc: dict, sets: dict[str, list[dict]]) -> str:
    g = doc["functions"]
    lines = ["# LoRA-2 gate (frozen)", "",
             "Before-version snapshots of the functions the LoRA-2 gate is judged on. Game-derived code: private, never publish.",
             "Barred from every grind queue (`dk route` worker `frozen`, `llm-grind`, `close --near`, marked in `dk next` and `dk brief`)",
             "and from training data (`llmtry.never_train`, `llmdata.leaks`, `dataset fixes`). Frozen: " + doc["frozen"] + ".", "",
             f"- commit: `{doc['commit']}`  seed: {doc['seed']}  schema {SCHEMA}",
             f"- gave-up half: {sum(r['why'] == 'gave-up' for r in g)} (the give-up logs' 'gave up' lines, still open, <= {MAX_SIZE} B; "
             f"pool {doc['stats']['gave_up_pool']} of {doc['stats']['gave_up_logged']} logged)",
             f"- never-tried half: {sum(r['why'] == 'never-tried' for r in g)} (near-register/near-shape, 0 attempts, not named in "
             f"the logs, <= {MAX_SIZE} B; pool {doc['stats']['fresh_pool']}), random.Random(seed + 1).sample of the sorted pool",
             f"- other snapshots: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()), ""] + \
            [f"- refresh {i}, {r['date']} at `{r['commit'][:10]}` (seeds {r['seeds'][0]}/{r['seeds'][1]}, give-up logs "
             f"the refresh logs, pools {r['gave_up_pool']}/{r['fresh_pool']}, none already in a training or review export): retired "
             + ", ".join(f"{n} ({'trained earlier' if n in r.get('trained_earlier', []) else 'exact'})" for n in r["retired"]) + "; "
             f"added {', '.join(r['added'])}" for i, r in enumerate(doc.get("refreshes", []), 1)] + \
            ([""] if doc.get("refreshes") else []) + [
             "## Fields", "",
             "unit, function, address, size, fuzzy, source, why, class, attempts, commit (HEAD when frozen), asm_shape, source_blob",
             "(git blob of the unit source), before (the definition then), definition_hash (sha1 of its comment-free, whitespace-",
             "collapsed text), hashes (that hash and the name-blank hash of the definition and of up to "
             f"{VARIANT_CAP} one-step rewrite-table variants).", "",
             "## Rules", "",
             "Gate pass: LoRA 2 closes at least 3 more than the CPU enumerator on the give-up half,",
             "or reaches at least 0.3 S on the never-tried half; 8 turns for both contestants, judge-clean, 0 devices.",
             "Never edit gate.json by hand. `./dk llm-bench gate-freeze --refresh` retires functions that became exact and draws",
             "replacements (results on retired ones drop out; the rest stay comparable); `--force` makes a new gate (voids results).",
             "Replacements are never functions an existing training or review export holds (name, address or definition hash).", ""]
    return "\n".join(lines)


def write_sums(d: Path) -> None:
    names = sorted(p.name for p in d.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (d / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n" for n in names))


def table_names() -> list[str]:
    from . import rewrites
    return [r.name for r in rewrites.RULES]


def keep_earlier(sets: dict[str, dict[str, set]], old_doc: dict) -> int:
    """The leakage filter only grows: the hashes an earlier table's rehash found stay in `sets` (a changed row
    may stop offering a variant it offered before). Returns how many were kept that today's table lacks."""
    kept = 0
    for name, fns in (old_doc.get("sets") or {}).items():
        for fn, hs in fns.items():
            got = sets.setdefault(name, {}).setdefault(fn, set())
            kept += len(set(hs) - got)
            got.update(hs)
    return kept


def rehash(cfg: Config, args: argparse.Namespace) -> int:
    """Every frozen definition (before and after) hashed again with the current rewrite table's one-step variants,
    each from the source of its own commit (`git show REV:path`), into variant-hashes.json, keeping the hashes of
    earlier rehashes (the filter never shrinks); SHA256SUMS follows."""
    from .shaperules import project_accessors
    from .trycmd import definition
    from .llmdata import norm
    d = gate_dir(cfg)
    if not (d / "gate.json").is_file():
        print("error: no gate yet (./dk llm-bench gate-freeze)")
        return USAGE
    srcs: dict[tuple[str, str], str] = {}
    sets: dict[str, dict[str, set]] = {}
    stats = {"definitions": 0, "variants": 0, "located": 0, "alone": 0}
    acc = project_accessors(cfg.root)
    for name, rows in load_sets(cfg).items():
        for r in rows:
            fn, path = r.get("function"), r.get("source", "")
            for field_, rev in (("before", r.get("before_commit") or r.get("commit")), ("after", r.get("commit"))):
                text = r.get(field_)
                if not fn or not text or not rev:
                    continue
                src = ""
                if path:
                    k = (rev, path)
                    if k not in srcs:
                        srcs[k] = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=cfg.root, capture_output=True,
                                                 text=True, errors="replace").stdout
                    src = srcs[k]
                span = definition(src, fn) if src else None
                if span is not None and norm(src[span[0]:span[1]]) != norm(text):
                    at = src.find(text)
                    span = (at, at + len(text)) if at >= 0 else None
                if span is None:                         # the definition alone (its file's other text is not needed)
                    src, span = text, (0, len(text))
                    stats["alone"] += 1
                else:
                    stats["located"] += 1
                cpp = not path.endswith(".c")
                vs = variants(src, fn, span, cpp, cap=REHASH_CAP, per_rule=REHASH_PER_RULE, accessors=acc)
                got = sets.setdefault(name, {}).setdefault(fn, set())
                got |= code_hashes(text)
                for v in vs:
                    got |= code_hashes(v)
                stats["definitions"] += 1
                stats["variants"] += len(vs)
    prev = d / REHASH_FILE
    kept = keep_earlier(sets, json.loads(prev.read_text()) if prev.is_file() else {})
    stats["kept from earlier tables"] = kept
    doc = {"schema": SCHEMA, "made": time.strftime("%Y-%m-%d %H:%M"), "head": head_commit(cfg), "table": table_names(),
           "per_rule": REHASH_PER_RULE, "cap": REHASH_CAP, "stats": stats,
           "sets": {k: {f: sorted(h) for f, h in sorted(v.items())} for k, v in sorted(sets.items())}}
    old = sum(1 for _ in hashes(cfg))
    (d / REHASH_FILE).write_text(json.dumps(doc, indent=0) + "\n")
    write_sums(d)
    print(f"rehash: {stats['definitions']} definitions ({stats['located']} in their commit's source, {stats['alone']} alone), "
          f"{stats['variants']} variants of {len(doc['table'])} table rows, {kept} hashes kept from earlier tables "
          f"-> {cfg.rel(d / REHASH_FILE)}")
    print(f"  hashes in the filter: {old} before, {len(hashes(cfg))} now")
    return OK


def freeze(cfg: Config, args: argparse.Namespace) -> int:
    from .llmtry import asm_entries
    d = gate_dir(cfg)
    if getattr(args, "rehash", False):
        return rehash(cfg, args)
    if getattr(args, "refresh", False):
        return refresh(cfg, args)
    if (d / "gate.json").is_file() and not args.force:
        print(f"{cfg.rel(d / 'gate.json')} exists: the gate is frozen (--force makes a new one and voids results)")
        return USAGE
    rows = open_rows(cfg)
    if not rows:
        print("error: no objdiff report yet (./dk build)")
        return USAGE
    head = head_commit(cfg)
    entries = asm_entries(cfg)
    council = Path(args.council).expanduser() if args.council else council_dir(cfg)
    if not council.is_dir():
        print(f"error: no council logs at {council}")
        return USAGE
    d.mkdir(parents=True, exist_ok=True)
    recs, stats = build(cfg, args.seed, council, rows, head, entries)
    doc = {"schema": SCHEMA, "frozen": time.strftime("%Y-%m-%d"), "seed": args.seed, "commit": head, "stats": stats,
           "purpose": "LoRA-2 gate: do not train on, tune on or grind these", "functions": recs}
    (d / "gate.json").write_text(json.dumps(doc, indent=1) + "\n")
    sets = {GATE_NAME: recs}
    for name in SNAPSHOT_SETS:
        out = d / f"{name}.snapshot.json"
        if out.is_file() and not args.force:
            sets[name] = json.loads(out.read_text())["functions"]
            continue
        got = snapshot_set(cfg, name, rows, head, entries)
        out.write_text(json.dumps({"schema": SCHEMA, "frozen": doc["frozen"], "commit": head, "set": name, "functions": got},
                                  indent=1) + "\n")
        sets[name] = got
    (d / "CARD.md").write_text(card(doc, sets))
    write_sums(d)
    gave = sum(r["why"] == "gave-up" for r in recs)
    print(f"gate frozen in {cfg.rel(d)}: {gave} gave-up + {len(recs) - gave} never-tried; snapshots "
          + ", ".join(f"{k} {len(v)}" for k, v in sets.items() if k != GATE_NAME))
    print(f"  pools: gave-up {stats['gave_up_pool']}, never-tried {stats['fresh_pool']}")
    return OK


def refresh(cfg: Config, args: argparse.Namespace) -> int:
    """Retire the gate functions exact today and redraw as many (see `redraw`); CARD.md and SHA256SUMS follow.
    The near-sample snapshots are left alone (their functions stay frozen when exact, as before)."""
    from .llmtry import asm_entries
    d = gate_dir(cfg)
    if not (d / "gate.json").is_file():
        print("error: no gate yet (./dk llm-bench gate-freeze)")
        return USAGE
    rows = open_rows(cfg)
    if not rows:
        print("error: no objdiff report yet (./dk build)")
        return USAGE
    council = Path(args.council).expanduser() if args.council else council_dir(cfg)
    doc = json.loads((d / "gate.json").read_text())
    bar = trained_bar(cfg)
    by_name = {r["function"]: r for r in rows}
    held = [r["function"] for r in doc["functions"] if r["function"] in by_name and bar(by_name[r["function"]])]
    if held:
        print("in an existing training/review export: " + ", ".join(held))
    retire = tuple(args.retire or ()) + (tuple(held) if args.retire_trained else ())
    new = redraw(cfg, doc, council, rows, head_commit(cfg), asm_entries(cfg), retire, bar)
    if not new["refreshes"] or new["refreshes"] == doc.get("refreshes", []):
        print("gate unchanged: every gate function is still open")
        return OK
    r = new["refreshes"][-1]
    print(f"refresh {len(new['refreshes'])} (seeds {r['seeds'][0]}/{r['seeds'][1]}; pools gave-up {r['gave_up_pool']}, "
          f"never-tried {r['fresh_pool']}):")
    print("  retired: " + ", ".join(r["retired"]))
    print("  added:   " + ", ".join(r["added"]))
    if args.dry_run:
        print("dry run: nothing written")
        return OK
    (d / "gate.json").write_text(json.dumps(new, indent=1) + "\n")
    sets = {GATE_NAME: new["functions"]}
    for name in SNAPSHOT_SETS:
        p = d / f"{name}.snapshot.json"
        if p.is_file():
            sets[name] = json.loads(p.read_text())["functions"]
    (d / "CARD.md").write_text(card(new, sets))
    write_sums(d)
    return OK


# ---- leakage dry run --------------------------------------------------------------------------------

def why_which(rec: dict, fns: set, shapes: set, pats: list) -> str:
    """Which frozen function a text hit names (for the report)."""
    from .llmdata import LITERAL
    text = "\n".join(m["content"] for m in rec["messages"][1:])
    for p in pats:
        lit = LITERAL.get(p.pattern)
        if lit is not None and lit in text and p.search(text):
            return f"[{lit}]"
    return ""


def leakcheck(cfg: Config, args: argparse.Namespace) -> int:
    """The leakage filter over a seeded sample of pairs from the existing datasets (chat records with a `tags`
    block and `messages`): every pair is tested with `llmdata.leaks`; hits are listed."""
    from .llmdata import leaks, never_definitions
    from .llmtry import data_dir, never_train
    units, fns, shapes = never_train(cfg)
    pats = never_definitions(cfg, fns)
    files = [Path(p) for p in args.files] if args.files else [
        p for p in sorted(data_dir(cfg).glob("datasets/**/*.jsonl")) + sorted(data_dir(cfg).glob("dataset/**/*.jsonl"))
        if "cache" not in p.parts and p.name != "eval.jsonl"]          # caches and the never-trained eval are not training pairs
    pairs: list[tuple[str, dict]] = []
    for p in files:
        try:
            for i, line in enumerate(p.read_text(errors="replace").splitlines()):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and ("messages" in r or "exact" in r or "before" in r or "state" in r):
                    pairs.append((f"{p.name}:{i + 1}", r))
        except OSError:
            continue
    random.Random(args.seed).shuffle(pairs)
    sample = pairs[:args.sample]
    hits = []
    for where, r in sample:
        if "messages" in r:
            rec = {"messages": r["messages"], "tags": r.get("tags") or {"unit": r.get("unit", ""), "function": r.get("function", "")}}
        else:                                           # a structured record: render its code fields as one message
            st = r.get("state") if isinstance(r.get("state"), dict) else {}      # a policy example: its state's definition
            hunks = [h.get(k) for h in (r.get("patch") or []) if isinstance(h, dict) for k in ("search", "replace")]
            code = "\n".join(f"```cpp\n{t}\n```" for t in [r.get(k) for k in ("wrong", "exact", "before", "after", "window")]
                             + [st.get("definition")] + hunks if isinstance(t, str))   # stacked/window records: hunks and window too
            rec = {"messages": [{"content": ""}, {"content": code}],
                   "tags": {"unit": r.get("unit", ""), "function": r.get("function", ""), "asm_shape": r.get("asm_shape"),
                            "mutant_shape": r.get("mutant_shape", "")}}
        rec["tags"].setdefault("unit", "")
        rec["tags"].setdefault("function", "")
        why = leaks(rec, units, fns, shapes, pats)
        if why:
            hits.append((where, rec["tags"].get("function"), why + " " + why_which(rec, fns, shapes, pats)))
    print(f"{len(sample)} pair(s) sampled of {len(pairs)} in {len(files)} file(s) (seed {args.seed}); "
          f"filter: {len(units)} unit(s), {len(fns)} function(s), {len(shapes)} asm shape(s), "
          f"{len(hashes(cfg))} definition hash(es)")
    for w, f, why in hits[:30]:
        print(f"  HIT {w} {f}: {why}")
    print(f"{len(hits)} hit(s)")
    return OK if not hits else MISMATCH


def configure_freeze(sub) -> None:
    g = sub.add_parser("gate-freeze", help="freeze the LoRA-2 gate (30 gave-up + 30 never-tried open functions <= 2 KB) and "
                       "snapshot near-sample-60 / near-sample-fresh-30 as before-versions into llm-data/gate-lora2/; "
                       "every grind queue and the leakage filters then skip them")
    g.add_argument("--seed", type=int, default=20261106, help="seed of the random picks (default 20261106)")
    g.add_argument("--council", default="", help="the grind logs (default: [platform.gate] logs beside the data dir)")
    g.add_argument("--force", action="store_true", help="make the gate again (voids results measured on the old one)")
    g.add_argument("--refresh", action="store_true", help="retire the gate functions exact today and draw as many "
                   "replacements by the same method (the refresh logs, the next seeds); logged in gate.json")
    g.add_argument("--retire", action="append", metavar="FN", help="with --refresh: also retire this still-open gate "
                   "function (replaced by a draw that no training or review export holds)")
    g.add_argument("--retire-trained", action="store_true", help="with --refresh: retire every gate function an existing "
                   "training or review export holds (by name, address or definition hash)")
    g.add_argument("--dry-run", action="store_true", help="with --refresh: print the change, write nothing")
    g.add_argument("--rehash", action="store_true", help="hash every frozen definition's one-step variants again with "
                   "today's rewrite table (after new rows) into variant-hashes.json; the sets are not touched")
