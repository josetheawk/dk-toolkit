"""dk brief <START-END | unit...> [--role sweeper|closer]: generate an agent's brief.

A brief is the rules template (tools/dk/templates/brief.md, or dk.toml [brief] template) filled
in from the last build's report and the tools, so it never needs hand-editing:
  {{scope}}       what the slice holds (a unit belongs to the slice its .text starts in)
  {{units}}       its units with open code, most open bytes first, each with its biggest open functions
  {{undeclared}}  code in the slice no unit declares yet, and the harvest command for it
  {{near}}        units at most 3 functions from exact and what blocks each (the platform's
                  brief_near scores them now; --no-score skips it)
  {{playbook}}    where to look: AGENTS.md, the playbook's and the project docs' sections
  {{notes}}       the agent's notes from earlier sessions: build/dk/notes/<slice>.md in the main
                  checkout (it outlives worktrees); {{notes_path}} says where to write them
  {{role}}, {{slice}}
Blocks between {{#sweeper}} and {{/sweeper}} (or closer) are kept only for that role.
Output is deterministic: the same report, notes and template give the same brief.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .. import platforms
from .command import OK, USAGE, Command, hint
from .config import Config
from .isolated import main_worktree
from .progress import load_report

ROLES = ("sweeper", "closer")
DEFAULT_TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "brief.md"
_RANGE = re.compile(r"^(?:0x)?([0-9A-Fa-f]+)-(?:0x)?([0-9A-Fa-f]+)$")
_BLOCK = re.compile(r"\{\{#(\w+)\}\}\n?(.*?)\{\{/\1\}\}\n?", re.S)
_FIELD = re.compile(r"\{\{(\w+)\}\}")


def _n(value) -> int:
    return int(value or 0)


def _kb(n: int) -> str:
    return f"{n / 1024:.1f} KB"


def _start(unit: dict) -> int | None:
    addrs = [_n(f.get("metadata", {}).get("virtual_address")) for f in unit.get("functions", [])]
    return min(addrs) if addrs else None


def label(cfg: Config, unit: dict) -> str:
    """What the unit commands accept: the source path under src/, else the report name."""
    src = unit.get("metadata", {}).get("source_path", "")
    root = cfg.platform_options.get("src", "src") + "/"
    return src.removeprefix(root) if src else unit["name"]


def open_bytes(unit: dict) -> int:
    m = unit.get("measures", {})
    return max(0, _n(m.get("total_code")) - _n(m.get("matched_code")))


def select(cfg: Config, report: dict, scope: list[str]) -> tuple[str, list[dict], list[dict], tuple | None, list[str]]:
    """(slice name, declared units, undeclared units, (start, end) or None, unknown queries)."""
    units = report.get("units", [])
    m = _RANGE.match(scope[0]) if len(scope) == 1 else None
    if m:
        lo, hi = int(m[1], 16), int(m[2], 16)
        inside = [u for u in units if (s := _start(u)) is not None and lo <= s < hi]
        declared = [u for u in inside if not u.get("metadata", {}).get("auto_generated")]
        undeclared = [u for u in inside if u.get("metadata", {}).get("auto_generated")]
        return f"{lo:08X}-{hi:08X}", declared, undeclared, (lo, hi), []
    found, unknown = [], []
    for q in scope:
        q = q.removeprefix(cfg.platform_options.get("src", "src") + "/")
        hit = next((u for u in units if q in (u["name"], u["name"].split("/", 1)[-1], label(cfg, u))
                    or label(cfg, u).rsplit(".", 1)[0] == q), None)
        if hit is None:
            unknown.append(q)
        elif hit not in found:
            found.append(hit)
    first = Path(label(cfg, found[0])).stem if found else "units"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", first) + (f"-and-{len(found) - 1}" if len(found) > 1 else "")
    return name, found, [], None, unknown


def scope_text(cfg: Config, name: str, declared: list[dict], undeclared: list[dict], span: tuple | None) -> str:
    linked = sum(1 for u in declared if u.get("metadata", {}).get("complete"))
    total = sum(_n(u.get("measures", {}).get("total_code")) for u in declared)
    left = sum(open_bytes(u) for u in declared)
    what = (f"Slice 0x{span[0]:08X}-0x{span[1]:08X} (a unit belongs to the slice its .text starts in)"
            if span else "Units " + ", ".join(label(cfg, u) for u in declared))
    out = f"{what}: {len(declared)} declared unit(s), {linked} linked; {_kb(left)} of {_kb(total)} code still open"
    if span:
        out += f"; {_kb(sum(_n(u.get('measures', {}).get('total_code')) for u in undeclared))} undeclared"
    return out + f". Numbers from the last build's report ({cfg.rel(cfg.report)})."


def units_text(cfg: Config, declared: list[dict], max_units: int, max_functions: int) -> str:
    rows = sorted((u for u in declared if open_bytes(u) or not u.get("metadata", {}).get("complete")),
                  key=lambda u: (-open_bytes(u), label(cfg, u)))
    if not rows:
        return "Every declared unit here is exact."
    shown = rows[:max_units] if max_units else rows
    out = ["Units with open code, most open bytes first (open / total code, exact functions):", ""]
    frozen = (getattr(platforms.load(cfg.platform), "frozen_functions", None) or (lambda c: {}))(cfg)
    if frozen:
        out += ["Functions marked FROZEN are in an evaluation set: never edit, rename inside or look up answers for them "
                "(save/accept/commit-tools refuse a change to one).", ""]
    for u in shown:
        m = u.get("measures", {})
        out.append(f"- `{label(cfg, u)}`: {open_bytes(u)} / {_n(m.get('total_code'))} bytes open, "
                   f"{_n(m.get('matched_functions'))}/{_n(m.get('total_functions'))} exact")
        fns = sorted((f for f in u.get("functions", []) if float(f.get("fuzzy_match_percent", 0.0)) < 100.0),
                     key=lambda f: (-_n(f.get("size")), f["name"]))
        for f in fns[:max_functions] if max_functions else fns:
            out.append(f"    - {float(f.get('fuzzy_match_percent', 0.0)):6.2f}%  0x{_n(f.get('size')):X} bytes  `{f['name']}`"
                       + (f"  **FROZEN ({frozen[(u['name'], f['name'])]}): skip**" if (u["name"], f["name"]) in frozen else ""))
        if max_functions and len(fns) > max_functions:
            out.append(f"    - ... {len(fns) - max_functions} more open (`./dk check {label(cfg, u)}`)")
            hidden = [f["name"] for f in fns[max_functions:] if (u["name"], f["name"]) in frozen]
            if hidden:
                out.append(f"    - FROZEN among those (skip): {', '.join(f'`{n}`' for n in hidden)}")
    if len(rows) > len(shown):
        out.append(f"- ... {len(rows) - len(shown)} more unit(s) with open code (`--max-units 0` lists all)")
    return "\n".join(out)


def undeclared_text(undeclared: list[dict], span: tuple | None) -> str:
    if span is None:
        return "Undeclared code: none in scope (this brief names its units)."
    code = sum(_n(u.get("measures", {}).get("total_code")) for u in undeclared)
    if not undeclared:
        return "Undeclared code: none, every function in the slice belongs to a declared unit."
    return (f"Undeclared code: {_kb(code)} in {len(undeclared)} stretch(es) no unit declares yet. Harvest first: "
            f"`./dk gen --harvest 0x{span[0]:08X} 0x{span[1]:08X}` ranks them by exact drafts; declare the best "
            "with `./dk unit propose` and `./dk unit add`, then `./dk gen --unit U --write`.")


def near_text(cfg: Config, declared: list[dict], score: bool) -> str:
    names = [u["name"] for u in declared if not u.get("metadata", {}).get("complete")]
    hook = getattr(platforms.load(cfg.platform), "brief_near", None)
    if not names:
        return "No WIP units here."
    if not score or hook is None:
        return "(not scored: run `./dk check --all --near`)"
    lines = hook(cfg, names)
    if len(lines) <= 1:
        return "None: no WIP unit here is within 3 functions of exact."
    return "\n".join(["```", *lines, "```"])


def playbook_text(cfg: Config) -> str:
    def heads(path: Path) -> str:
        found = [l[3:].strip() for l in path.read_text().splitlines() if l.startswith("## ")]
        return "; ".join(found)

    out = []
    if cfg.docs.agents_md.is_file():
        out.append(f"- `{cfg.rel(cfg.docs.agents_md)}`: start here (the task to command table and the rules)")
    for folder in (cfg.docs.playbook, cfg.docs.project):
        files = sorted(folder.glob("*.md"), key=lambda p: (p.name != "README.md", p.name)) if folder.is_dir() else []
        out += [f"- `{cfg.rel(p)}`" + (f": {h}" if (h := heads(p)) else "") for p in files]
    return "\n".join(out) or "(no playbook found)"


def notes_path(cfg: Config, name: str) -> Path:
    return main_worktree(cfg.root) / "build" / "dk" / "notes" / f"{name}.md"


def render(template: str, role: str, values: dict[str, str]) -> str:
    text = _BLOCK.sub(lambda m: m[2] if m[1] == role else "", template)
    missing = sorted({k for k in _FIELD.findall(text) if k not in values})
    if missing:
        raise KeyError(", ".join(missing))
    text = _FIELD.sub(lambda m: values[m[1]], text)
    return re.sub(r"\n{3,}", "\n\n", text).rstrip("\n") + "\n"


def run(cfg: Config, args: argparse.Namespace) -> int:
    report = load_report(cfg)
    if report is None:
        print("error: no report yet: run ./dk build")
        return USAGE
    template_path = cfg.brief_template or DEFAULT_TEMPLATE
    if not template_path.is_file():
        print(f"error: brief template {template_path} not found")
        return USAGE
    name, declared, undeclared, span, unknown = select(cfg, report, args.scope)
    if unknown:
        print(f"error: no unit named {', '.join(unknown)} in the report (give START-END or unit source paths)")
        return USAGE
    if not declared and not undeclared:
        print(f"error: nothing in {' '.join(args.scope)}")
        return USAGE
    name = args.slice or name
    template = template_path.read_text()
    kept = _BLOCK.sub(lambda m: m[2] if m[1] == args.role else "", template)
    notes = notes_path(cfg, name)
    values = {
        "role": args.role, "slice": name,
        "scope": scope_text(cfg, name, declared, undeclared, span),
        "units": units_text(cfg, declared, args.max_units, args.functions),
        "undeclared": undeclared_text(undeclared, span),
        "near": near_text(cfg, declared, not args.no_score) if "{{near}}" in kept else "",
        "playbook": playbook_text(cfg),
        "notes": notes.read_text().strip() if notes.is_file() and notes.read_text().strip() else "(none yet)",
        "notes_path": str(notes),
    }
    try:
        text = render(template, args.role, values)
    except KeyError as e:
        print(f"error: {cfg.rel(template_path)} uses unknown placeholder(s): {e.args[0]}")
        return USAGE
    if args.out:
        Path(args.out).write_text(text)
        print(f"brief written to {args.out} ({len(text.splitlines())} lines)")
        hint("status")
    else:
        print(text, end="")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("scope", nargs="+", metavar="START-END | unit",
                   help="an address slice (0x80010000-0x80020000: units whose .text starts in it) or units")
    p.add_argument("--role", choices=ROLES, default="sweeper", help="which job the brief describes (default sweeper)")
    p.add_argument("--slice", help="name for the slice (worktree and notes file); default from the scope")
    p.add_argument("--max-units", type=int, default=25, help="units to list (0 = all; default 25)")
    p.add_argument("--functions", type=int, default=6, help="open functions to list per unit (0 = all; default 6)")
    p.add_argument("--no-score", action="store_true", help="skip scoring the near-exact units (report numbers only)")
    p.add_argument("--out", help="write the brief to this file instead of printing it")


COMMAND = Command("brief", "docs", "an agent's brief for a slice or units: rules template + open work + notes",
                  run, _configure)
