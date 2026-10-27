"""dk quality: measure source quality so refinement only ever moves it forward.

Two kinds of rules, both from dk.toml [quality]:
  forbidden  any hit fails (accept/save refuse a unit with one; CI fails the repo)
  debt       counted, never fails; placeholder density that refinement passes pay down.
             A debt rule with `duplicates = true` counts only repeats: a match whose text
             already matched earlier in the same file (a declaration pasted twice).
Vendored reference code (dk.toml `exclude`) is not judged.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

from .command import MISMATCH, OK, Command, hint
from .config import Config
from .doclint import GENERATED_MARK

SUFFIXES = (".c", ".cpp", ".cp", ".h", ".hpp", ".inc")


@dataclass
class Rule:
    name: str
    pattern: re.Pattern
    why: str
    duplicates: bool = False


@dataclass
class FileReport:
    path: str
    forbidden: list[tuple[int, str]] = field(default_factory=list)   # (line, rule)
    debt: dict[str, int] = field(default_factory=dict)


def _rules(raw: list[dict]) -> list[Rule]:
    return [Rule(r["name"], re.compile(r["pattern"], re.M), r.get("why", ""), bool(r.get("duplicates"))) for r in raw]


def settings(cfg: Config) -> tuple[list[Path], list[Path], list[Rule], list[Rule]]:
    q = cfg.quality
    roots = [cfg.root / p for p in q.get("paths", ["src", "include"])]
    exclude = [cfg.root / p for p in q.get("exclude", [])]
    return roots, exclude, _rules(q.get("forbidden", [])), _rules(q.get("debt", []))


def judged_files(cfg: Config, paths: list[Path] | None = None) -> list[Path]:
    roots, exclude, _, _ = settings(cfg)
    files = []
    for root in (paths or roots):
        candidates = [root] if root.is_file() else sorted(root.rglob("*")) if root.is_dir() else []
        for f in candidates:
            if f.suffix in SUFFIXES and not any(f.is_relative_to(e) for e in exclude):
                files.append(f)
    return sorted(set(files))


def scan(cfg: Config, files: list[Path]) -> list[FileReport]:
    _, _, forbidden, debt = settings(cfg)
    reports = []
    for f in files:
        text = f.read_text(errors="replace")
        r = FileReport(cfg.rel(f))
        for rule in forbidden:
            for m in rule.pattern.finditer(text):
                r.forbidden.append((text.count("\n", 0, m.start()) + 1, rule.name))
        for rule in debt:
            n = repeats(rule.pattern, text) if rule.duplicates else len(rule.pattern.findall(text))
            if n:
                r.debt[rule.name] = n
        reports.append(r)
    return reports


def repeats(pattern: re.Pattern, text: str) -> int:
    """Matches of `pattern` whose text (trailing blanks aside) already matched earlier in `text`."""
    seen, n = set(), 0
    for m in pattern.finditer(text):
        key = m.group(0).rstrip()
        n += key in seen
        seen.add(key)
    return n


def forbidden_hits(cfg: Config, files: list[Path]) -> list[str]:
    return [f"{r.path}:{line}: {rule}" for r in scan(cfg, files) for line, rule in r.forbidden]


def render(cfg: Config, reports: list[FileReport]) -> str:
    _, _, forbidden, debt = settings(cfg)
    names = [d.name for d in debt]
    totals = {n: sum(r.debt.get(n, 0) for r in reports) for n in names}
    out = [f"{GENERATED_MARK} quality; do not edit -->", "# Source quality", "",
           f"{len(reports)} judged files. Forbidden hits: {sum(len(r.forbidden) for r in reports)} (must stay 0).", "",
           "Debt (placeholders and leftovers that refinement passes pay down):", ""]
    out += [f"- `{d.name}`: {totals[d.name]} — {d.why}" for d in debt]
    out += ["", "| File | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for r in sorted(reports, key=lambda r: (-sum(r.debt.values()), r.path)):
        if r.debt:
            out.append(f"| `{r.path}` | " + " | ".join(str(r.debt.get(n, 0)) for n in names) + " |")
    return "\n".join(out) + "\n"


def run(cfg: Config, args: argparse.Namespace) -> int:
    files = judged_files(cfg, [cfg.root / p for p in args.paths] if args.paths else None)
    reports = scan(cfg, files)
    hits = [f"{r.path}:{line}: forbidden {rule}" for r in reports for line, rule in r.forbidden]
    for h in hits:
        print(h)
    if not args.paths:
        doc = cfg.docs.generated / "quality.md"
        text = render(cfg, reports)
        if args.check:
            if not doc.is_file() or doc.read_text() != text:
                print(f"{cfg.rel(doc)} is out of date")
                return MISMATCH
        else:
            doc.parent.mkdir(parents=True, exist_ok=True)
            doc.write_text(text)
    totals: dict[str, int] = {}
    for r in reports:
        for k, v in r.debt.items():
            totals[k] = totals.get(k, 0) + v
    print(f"{len(files)} files, {len(hits)} forbidden hit(s); debt: " + ", ".join(f"{k} {v}" for k, v in sorted(totals.items())))
    if hits:
        return MISMATCH
    hint("status")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("paths", nargs="*", help="files/dirs to judge (default: all judged code, and write docs/generated/quality.md)")
    p.add_argument("--check", action="store_true", help="exit 1 if docs/generated/quality.md is stale")


COMMAND = Command("quality", "docs", "forbidden patterns (must be 0) and placeholder debt per file", run, _configure)
