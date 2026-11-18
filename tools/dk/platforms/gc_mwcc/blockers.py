"""Blockers: why a grind attempt on an open function stopped, in one machine-readable file.

`llm-data/blockers.jsonl` (beside the main checkout, outside the repo) holds one JSON object per line:
`{function, blocker, playbook_version, date, outcome, best?, note?, source?}`. `blocker` is a tag from
TAGS; `outcome` is gave-up, skipped, partial or landed; `playbook_version` is `playbook_version(cfg)` when the
attempt was made. `dk route` reads the file: the latest row per function gives the route row its `blocker`,
and a function whose blocker was recorded under another playbook version (the matching notes or the rewrite
table changed since: a new rule or tool) counts as fresh again. `dk route --record` appends a row;
`dk route --backfill LOG…` reads the grind logs the sessions write (the bullet and pipe formats)
and appends their rows once (a log is skipped for rows already recorded).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

from ...core.config import Config

FILE = "blockers.jsonl"
OUTCOMES = ("gave-up", "skipped", "partial", "landed")
# first match wins; the keywords are what the sessions write in their notes
TAG_KEYWORDS = (
    ("temp-temp", ("temp-temp",)),
    ("cross-tier", ("cross-tier",)),
    ("not-reachable", ("not reachable", "unreachable", "not reproduced")),
    ("device-only", ("invented", "unfaithful", "byte-offset cast", "guessing case sets")),
    ("inline-policy", ("auto-inline", "pragma", "inline limit", "inlines the", "setters")),
    ("float-reg", ("float reg", "fpr", "fp regs", "fmuls operand", "fmr")),
    ("frame", ("stack slot", "frame ", "stack bytes", "c/d stack")),
    ("scheduling", ("schedule", "scheduling", "increment order", "order of")),
    ("register-only", ("register-numbering", "register-only", "register/temp-numbering", "reg numbering", "registers right")),
    ("branch-shape", ("beq", "bne", "ble", "bge", "blt", "bgt", "branch", "bdnz", "ctr")),
    ("compare-form", ("cmplwi", "cmpwi", "extrwi", "srwi", "clrlwi", "cntlzw")),
)
TAGS = tuple(t for t, _ in TAG_KEYWORDS) + ("shape",)


def playbook_version(cfg: Config) -> str:
    """Ten hex digits over the matching notes and the rewrite table's rule names, symptoms and idioms:
    changes when the playbook learns something or `dk close` gains a rule."""
    from .rewrites import RULES
    h = hashlib.sha1()
    path = cfg.root / "playbook" / "matching.md"
    h.update(path.read_bytes() if path.is_file() else b"")
    for r in RULES:
        h.update(f"\n{r.name}\t{r.symptom}\t{r.idiom}".encode())
    return h.hexdigest()[:10]


def path_of(ddir: Path) -> Path:
    return ddir / FILE


def load(ddir: Path) -> dict[str, dict]:
    """function -> its latest row (file order), skipping lines that are not rows."""
    out: dict[str, dict] = {}
    try:
        lines = path_of(ddir).read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("function") and r.get("blocker"):
            out[r["function"]] = r
    return out


def _rows(ddir: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in path_of(ddir).read_text().splitlines() if l.strip().startswith("{")]
    except (OSError, ValueError):
        return []


def append(ddir: Path, rows: list[dict]) -> int:
    """Append rows (validated); returns how many were written."""
    for r in rows:
        if r["outcome"] not in OUTCOMES:
            raise ValueError(f"outcome {r['outcome']!r} is not one of {', '.join(OUTCOMES)}")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", r["blocker"]):
            raise ValueError(f"blocker tag {r['blocker']!r}: lowercase words joined by '-' (known: {', '.join(TAGS)})")
    if rows:
        ddir.mkdir(parents=True, exist_ok=True)
        with path_of(ddir).open("a") as f:
            f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    return len(rows)


def make_row(function: str, blocker: str, version: str, day: str, outcome: str, best: float | None = None,
             note: str = "", source: str = "") -> dict:
    r = {"function": function, "blocker": blocker, "playbook_version": version, "date": day, "outcome": outcome}
    if best is not None:
        r["best"] = best
    if note:
        r["note"] = note[:200]
    if source:
        r["source"] = source
    return r


# ---- reading the sessions' logs ---------------------------------------------------------------------

def classify_note(text: str) -> str:
    low = text.lower()
    for tag, words in TAG_KEYWORDS:
        if any(w in low for w in words):
            return tag
    return "shape"


def names_of(field: str) -> list[str]:
    """Function names in a log line's name field: `A + B`, `A / B`, `A, B`, `fn_80001DC8/E98/7058`
    (short hex tails replace the end of the first address), without any trailing `(unit, size)`."""
    field = field.split(" (")[0].strip()
    out: list[str] = []
    for part in re.split(r"\s+[+/]\s+|,\s*", field):
        part = part.strip()
        if not part:
            continue
        head, *tails = part.split("/")
        if not re.fullmatch(r"[A-Za-z_][\w$]*", head):
            continue
        out.append(head)
        for t in tails:
            if re.fullmatch(r"[0-9A-Fa-f]{1,7}", t) and re.fullmatch(r"fn_[0-9A-Fa-f]{8}", head):
                out.append(head[:len(head) - len(t)] + t.upper())
            elif re.fullmatch(r"[A-Za-z_][\w$]*", t):
                out.append(t)
    return out


def _outcome(text: str) -> str:
    low = text.lower()
    if "landed" in low:
        return "landed"
    if "partial" in low:
        return "partial"
    if "gave up" in low:
        return "gave-up"
    return "skipped"


_BULLET = re.compile(r"^- (?P<name>.+?) start [\d.]+%: (?P<rest>.*)$")
_PIPE = re.compile(r"^(?P<name>[^|]+) \| (?P<pct>\d+)% \| (?P<out>[^|]+) \| (?P<n>\d+) \| (?P<rest>.*)$")
_SKIP_WORDS = ("summary", "recurring")


def parse_log(text: str, version: str, source: str, day: str) -> list[dict]:
    """Rows for the attempt lines of a grind log (bullet form `- fn (unit, size) start P%: GAVE UP|LANDED …`
    or pipe form `fn | P% | outcome | minutes | note`). Lines it does not know are left out."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        m = _BULLET.match(line)
        if m:
            names, rest = names_of(m["name"]), m["rest"]
            outcome = "skipped" if re.search(r"looked only|not attempted", rest, re.I) and "gave up" not in rest.lower() \
                else _outcome(rest[:40])
            best = re.search(r"(?:GAVE UP|start)\s+(\d+\.\d+)%", rest)
            note = rest
        else:
            m = _PIPE.match(line)
            if not m or m["name"].strip().lower().startswith(_SKIP_WORDS):
                continue
            names, rest, outcome, best, note = names_of(m["name"]), m["rest"], _outcome(m["out"]), None, m["rest"]
        if outcome == "landed" and re.search(r"gave up", note[:40], re.I):
            outcome = "gave-up"
        for n in names:
            rows.append(make_row(n, classify_note(note), version, day, outcome,
                                 float(best.group(1)) if best else None, note, source))
    return rows


def backfill(ddir: Path, logs: list[Path], version: str, day: str) -> tuple[int, int]:
    """Append the rows of these logs not recorded yet (same function and source); (written, already there)."""
    have = {(r.get("function"), r.get("source")) for r in _rows(ddir)}
    fresh, skipped = [], 0
    for log in logs:
        source = f"{log.parent.name}/{log.name}"
        for r in parse_log(log.read_text(), version, source, day):
            if (r["function"], source) in have:
                skipped += 1
                continue
            have.add((r["function"], source))
            fresh.append(r)
    return append(ddir, fresh), skipped


# ---- the frozen sets and the fresh pool -------------------------------------------------------------

def frozen_functions(ddir: Path) -> set[str]:
    """Function names in the benchmark (bench.json) and the held-out set (heldout.json): never worked, never trained."""
    out: set[str] = set()
    for name in ("bench.json", "heldout.json"):
        try:
            out |= {r["function"] for r in json.loads((ddir / name).read_text())["functions"]}
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


FRESH_MAX_SIZE = 2048
FRESH_KINDS = ("register", "shape")
DEFAULT_PER_DAY = 300


def is_fresh(row: dict, version: str | None, frozen: set[str], min_near: float = 80.0, strict: bool = False) -> bool:
    """Open, at most 2 KB, a near register or shape miss, not frozen, and either never attempted or its blocker
    was recorded under another playbook version. "Attempted" is a blockers.jsonl row (a Claude or human grind);
    `strict` also counts the local-model grind's state files (weak attempts that say little about Claude)."""
    if row["size"] > FRESH_MAX_SIZE or row["fuzzy"] < min_near or row["function"] in frozen:
        return False
    if row.get("kind") not in FRESH_KINDS or row["class"] in ("sinit", "compiler-version", "unit-data", "fresh"):
        return False
    if row.get("blocker") is not None:
        return version is not None and row.get("playbook_version") != version
    return not strict or row.get("attempts", 0) == 0


def depletion(fresh: int, per_day: float) -> float:
    return fresh / per_day if per_day > 0 else float("inf")


def today() -> str:
    return date.today().isoformat()
