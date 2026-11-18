"""dk site, part 1: gather everything the status site shows from the repo's own files (stdlib only).

Inputs, each optional so a missing one leaves its panel empty instead of failing the build:
  report.json (objdiff)            units, functions, match %
  dk.toml [platform.systems]       game system per unit/function (regex)
  `./dk route --json`              blocker class, worker and frozen label per open function
  the floor file's git history     matched functions / linked code / linked units per landing
  git log                          recent landings, commits per day
  [site] content dir               hand-written pages: tours/*.md, milestones.toml, ideas/failures/future.md
  [site] notes, library, experiments, released   notes, docs, council experiments, released functions
Paths in [site] are tried against the checkout first, then the main checkout (a worktree shares its build/).
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config

DEFAULTS: dict[str, Any] = {
    "port": 8003,
    "out": "build/dk/site",
    "shots": "build/dk/site-shots",
    "content": "docs/site",
    "notes": "build/dk/notes",
    "released": "",
    "experiments": "",
    "floor": "",
    "faithfulness": "",
    "library": {},
    "tours": {},
    # wording (the toolkit stays game-neutral; a project names itself here)
    "name": "",                     # brand in the top bar and page titles (default: [project] name)
    "headline": "Rebuilt byte for byte",
    "categories_hint": '"game" is the game\'s own code; "sdk" is the system libraries it links against.',
    "shot_unit": "",                # dk site shots: the unit page to capture (its slug, e.g. game__gfx__Renderer)
    "shot_tour": "",                # dk site shots: the tour page to capture
}


def site_options(cfg: Config) -> dict[str, Any]:
    raw = {}
    path = cfg.root / "dk.toml"
    if path.is_file():
        with path.open("rb") as f:
            raw = tomllib.load(f).get("site", {})
    out = dict(DEFAULTS)
    out.update(raw)
    if not out["floor"]:
        out["floor"] = cfg.quality.get("floor", "")
    if not out["name"]:
        out["name"] = cfg.name
    return out


def _git(root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    except OSError:
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def main_root(root: Path) -> Path:
    """The main checkout (a worktree's shared build/ and the files beside it live there)."""
    common = _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    if common:
        p = Path(common)
        if p.name == ".git":
            return p.parent
    return root


def resolve(cfg: Config, spec: str, home: Path) -> list[Path]:
    """Files for one path or glob: tried against the checkout, then the main checkout."""
    spec = os.path.expanduser(spec)
    bases = [Path("/")] if os.path.isabs(spec) else [cfg.root, home]
    for base in bases:
        hits = sorted(Path(p) for p in glob.glob(str(base / spec)) if Path(p).is_file())
        if hits:
            return [Path(os.path.normpath(h)) for h in hits]
    return []


def resolve_dir(cfg: Config, spec: str, home: Path) -> Path | None:
    if not spec:
        return None
    spec = os.path.expanduser(spec)
    for base in ([Path("/")] if os.path.isabs(spec) else [cfg.root, home]):
        p = Path(os.path.normpath(base / spec))
        if p.is_dir():
            return p
    return None


# ---- units and functions ----------------------------------------------------------------------------

def _n(m: dict, k: str) -> int:
    return int(m.get(k, 0) or 0)


@dataclass
class Function:
    name: str
    demangled: str
    size: int
    pct: float
    address: int
    route: dict | None = None
    line: int | None = None

    @property
    def status(self) -> str:
        if self.pct >= 100.0:
            return "exact"
        if self.route and self.route.get("frozen"):
            return "frozen"
        if self.pct == 0.0:
            return "unwritten"
        return "near" if self.pct >= 80.0 else "far"


@dataclass
class Unit:
    name: str
    short: str
    source: str | None
    complete: bool
    category: str
    measures: dict
    sections: list[dict]
    functions: list[Function]
    systems: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", self.short.replace("/", "__"))

    @property
    def total_code(self) -> int:
        return _n(self.measures, "total_code")

    @property
    def code_pct(self) -> float:
        return float(self.measures.get("matched_code_percent", 0.0) or 0.0) if self.total_code else 100.0

    @property
    def exact_count(self) -> int:
        return sum(1 for f in self.functions if f.pct >= 100.0)

    @property
    def open_functions(self) -> list[Function]:
        return [f for f in self.functions if f.pct < 100.0]

    @property
    def status(self) -> str:
        if self.complete:
            return "linked"
        if not self.functions:
            return "data"
        if not self.open_functions:
            return "exact"
        return "wip"

    @property
    def classes(self) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for f in self.functions:
            m = re.match(r"^([A-Za-z_][\w:]*)::~?[\w]+", f.demangled)
            if m:
                cls = m.group(1)
                counts[cls] = counts.get(cls, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def load_units(report: dict, systems: dict[str, str], route_rows: list[dict]) -> list[Unit]:
    pats = {k: re.compile(v) for k, v in systems.items()}
    by_fn = {(r.get("unit"), r.get("function")): r for r in route_rows}
    units = []
    for u in report.get("units", []):
        meta = u.get("metadata", {}) or {}
        fns = []
        for f in u.get("functions", []):
            md = f.get("metadata", {}) or {}
            fns.append(Function(
                name=f["name"], demangled=md.get("demangled_name") or f["name"], size=int(f.get("size", 0) or 0),
                pct=round(float(f.get("fuzzy_match_percent") or 0.0), 2),
                address=int(md.get("virtual_address", 0) or 0), route=by_fn.get((u["name"], f["name"]))))
        fns.sort(key=lambda f: f.address)
        name = u["name"]
        short = name.split("/", 1)[1] if "/" in name else name
        hits: dict[str, int] = {}
        for k, p in pats.items():
            n = sum(f.size for f in fns if p.search(name) or p.search(f.name))
            if n or p.search(name):
                hits[k] = n
        cats = meta.get("progress_categories") or []
        units.append(Unit(
            name=name, short=short, source=meta.get("source_path"), complete=bool(meta.get("complete")),
            category=cats[0] if cats else "", measures=u.get("measures", {}) or {}, sections=u.get("sections", []) or [],
            functions=fns, systems=sorted(hits, key=lambda k: -hits[k])))
    units.sort(key=lambda u: (u.short.startswith("auto_"), not u.short.startswith("game/"), u.short.lower()))
    return units


def system_rows(units: list[Unit], systems: dict[str, str]) -> list[dict]:
    pats = {k: re.compile(v) for k, v in systems.items()}
    rows = []
    for k, p in pats.items():
        row = {"name": k, "units": 0, "code": 0, "exact": 0, "near": 0, "open": 0, "unwritten": 0, "linked": 0}
        for u in units:
            touched = False
            for f in u.functions:
                if not (p.search(u.name) or p.search(f.name)):
                    continue
                touched = True
                row["code"] += f.size
                if f.pct >= 100.0:
                    row["exact"] += f.size
                else:
                    row["open"] += 1
                    row["near"] += f.size if f.pct >= 80.0 else 0
                    row["unwritten"] += 1 if f.pct == 0.0 else 0
            if touched:
                row["units"] += 1
                row["linked"] += 1 if u.complete else 0
        rows.append(row)
    return rows


# ---- source -> function lines -----------------------------------------------------------------------

def definition_lines(text: str, fns: list[Function]) -> None:
    """Set Function.line to the first line that looks like its definition (name followed by '(' and a body)."""
    lines = text.split("\n")
    for f in fns:
        names = []
        m = re.match(r"^(?:[\w:<>]+::)?([\w:]+)::(~?\w+)\s*\(", f.demangled)
        if m:
            names.append(re.escape(f"{m.group(1)}::{m.group(2)}"))
            last = m.group(1).rsplit("::", 1)[-1]
            if last != m.group(1):
                names.append(re.escape(f"{last}::{m.group(2)}"))     # defined inside `namespace X {`
        names.append(re.escape(f.name))
        bare = re.match(r"^(~?\w+)\s*\(", f.demangled)
        if bare and bare.group(1) != f.name:
            names.append(re.escape(bare.group(1)))
        for pat in names:
            rx = re.compile(rf"(?<![\w:.>]){pat}\s*\(")
            for i, ln in enumerate(lines):
                if not rx.search(ln) or ln.startswith((" ", "\t")) or ln.rstrip().endswith(";"):
                    continue
                if ln.lstrip().startswith(("//", "#", "*", "/*")):
                    continue
                tail = " ".join(lines[i:i + 6])
                if "{" in tail and tail.find("{") < (tail.find(";") if ";" in tail else 10**9):
                    f.line = i + 1
                    break
            if f.line:
                break


def header_comment(text: str, limit: int = 14) -> str:
    """The comment block at the top of a source (licence-free prose the authors wrote), or ''."""
    out = []
    lines = text.lstrip("﻿").split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith("/*"):
        while i < len(lines):
            out.append(re.sub(r"^\s*(/\*+|\*+/?|\*/)\s?", "", lines[i]).replace("*/", "").rstrip())
            if "*/" in lines[i]:
                break
            i += 1
    else:
        while i < len(lines) and lines[i].lstrip().startswith("//"):
            out.append(re.sub(r"^\s*//+\s?", "", lines[i]).rstrip())
            i += 1
    text = "\n".join(x for x in out).strip()
    return "\n".join(text.split("\n")[:limit])


# ---- mentions in notes and docs ---------------------------------------------------------------------

def mention_keys(u: Unit) -> list[str]:
    """Names a doc may use for the unit: its path (game/gfx/Renderer) and, when distinctive, its file name."""
    keys = [u.short]
    base = u.short.rsplit("/", 1)[-1]
    if not re.fullmatch(r"[0-9A-Fa-f]{6,8}", base) and (len(base) >= 8 or re.search(r"[A-Z0-9]", base)) \
            and not base.startswith("auto_"):
        keys.append(base)
    return keys


_WORD = re.compile(r"[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*")


def _tokens(block: str) -> set[str]:
    out = set()
    for w in _WORD.findall(block):
        parts = w.split("/")
        for i in range(len(parts)):
            out.add("/".join(parts[i:]))
    return out


def paragraphs(text: str) -> list[tuple[str, str]]:
    """(heading, block) pairs: each bullet or paragraph with the nearest heading above it."""
    out, head, cur = [], "", []
    for ln in text.split("\n"):
        if ln.startswith("#"):
            if cur:
                out.append((head, "\n".join(cur)))
                cur = []
            head = ln.lstrip("#").strip()
        elif not ln.strip():
            if cur:
                out.append((head, "\n".join(cur)))
                cur = []
        elif re.match(r"^[-*]\s|^\d+[.)]\s", ln) and cur:
            out.append((head, "\n".join(cur)))
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        out.append((head, "\n".join(cur)))
    return out


def find_mentions(units: list[Unit], docs: list[tuple[str, Path, str]], per_doc: int = 3, per_unit: int = 8) -> dict[str, list[dict]]:
    """{unit name: [{doc, path, heading, text}]} from the given (label, path, text) docs (one token index, no scans)."""
    index: dict[str, list[tuple[int, int]]] = {}
    blocks = []
    for di, (label, path, text) in enumerate(docs):
        paras = paragraphs(text)
        blocks.append((label, path, paras))
        for pi, (_, block) in enumerate(paras):
            if block.lstrip().startswith("|"):      # table rows read badly out of context
                continue
            for t in _tokens(block):
                index.setdefault(t, []).append((di, pi))
    out: dict[str, list[dict]] = {}
    for u in units:
        hits = sorted({h for k in mention_keys(u) for h in index.get(k, [])})
        found, per = [], {}
        for di, pi in hits:
            if per.get(di, 0) >= per_doc:
                continue
            per[di] = per.get(di, 0) + 1
            label, path, paras = blocks[di]
            head, block = paras[pi]
            snippet = block if len(block) <= 420 else block[:400].rsplit(" ", 1)[0] + " ..."
            found.append({"doc": label, "path": path, "heading": head, "text": snippet})
            if len(found) >= per_unit:
                break
        if found:
            out[u.name] = found
    return out


# ---- history -----------------------------------------------------------------------------------------

def floor_history(root: Path, floor_rel: str) -> list[dict]:
    """[{t, rev, subject, matched_functions, complete_code, complete_units}] oldest first, one per change."""
    if not floor_rel:
        return []
    log = _git(root, "log", "--format=@@C %h %ct %s", "-p", "--unified=0", "--", floor_rel)
    rows, cur = [], None
    for ln in log.split("\n"):
        if ln.startswith("@@C "):
            if cur and len(cur) > 3:
                rows.append(cur)
            parts = ln.split(" ", 3)
            cur = {"rev": parts[1], "t": int(parts[2]), "subject": parts[3] if len(parts) > 3 else ""}
            continue
        m = re.match(r"^\+(matched_functions|complete_code|complete_units) = (\d+)", ln)
        if m and cur is not None:
            cur[m.group(1)] = int(m.group(2))
    if cur and len(cur) > 3:
        rows.append(cur)
    rows.reverse()
    # carry values forward: a commit may change only one key
    last: dict[str, int] = {}
    for r in rows:
        for k in ("matched_functions", "complete_code", "complete_units"):
            if k in r:
                last[k] = r[k]
            elif k in last:
                r[k] = last[k]
    return [r for r in rows if all(k in r for k in ("matched_functions", "complete_code", "complete_units"))]


def commits(root: Path, n: int = 0) -> list[dict]:
    args = ["log", "--no-merges", "--format=%h\x1f%ct\x1f%s"]
    if n:
        args.append(f"-{n}")
    out = []
    for ln in _git(root, *args).split("\n"):
        parts = ln.split("\x1f")
        if len(parts) == 3:
            out.append({"rev": parts[0], "t": int(parts[1]), "subject": parts[2]})
    return out


def head_info(root: Path) -> dict:
    s = _git(root, "log", "-1", "--format=%h\x1f%ct\x1f%s").strip().split("\x1f")
    return {"rev": s[0], "t": int(s[1]), "subject": s[2]} if len(s) == 3 else {"rev": "", "t": 0, "subject": ""}


def route_json(cfg: Config) -> dict:
    """`./dk route --json` output, or {} when the platform has no route command or it fails."""
    dk = cfg.root / "dk"
    if not dk.exists():
        return {}
    try:
        proc = subprocess.run([str(dk), "route", "--json"], cwd=cfg.root, capture_output=True, text=True, timeout=300)
        return json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip().startswith("{") else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}


def load_milestones(path: Path | None) -> list[dict]:
    if not path or not path.is_file():
        return []
    try:
        with path.open("rb") as f:
            return [e for e in tomllib.load(f).get("event", []) if isinstance(e, dict) and e.get("date")]
    except (OSError, tomllib.TOMLDecodeError):
        return []
