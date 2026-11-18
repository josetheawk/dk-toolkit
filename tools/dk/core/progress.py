"""dk progress: summarise the objdiff report written by the build."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from typing import Any

from .command import OK, USAGE, Command, hint
from .config import Config


def _n(measures: dict[str, Any], key: str) -> int:
    # objdiff writes protobuf JSON: 64-bit ints are strings, zero values are omitted.
    return int(measures.get(key, 0) or 0)


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:6.2f}%" if whole else "   n/a "


def load_report(cfg: Config) -> dict[str, Any] | None:
    if not cfg.report.is_file():
        return None
    return json.loads(cfg.report.read_text())


def stale(cfg: Config) -> str | None:
    """Why the report may not describe HEAD: it was written before HEAD's commit (other agents'
    landings move the shared branch without rebuilding this checkout). None when it is newer."""
    if not cfg.report.is_file():
        return None
    proc = subprocess.run(["git", "log", "-1", "--format=%ct %h"], cwd=cfg.root, capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    when, rev = proc.stdout.split()
    written = cfg.report.stat().st_mtime
    if written >= int(when):
        return None
    ago = int(int(when) - written)
    return (f"{cfg.rel(cfg.report)} is older than HEAD ({rev}, committed {ago // 60} min after the report was "
            f"written, {time.strftime('%H:%M', time.localtime(written))}): the numbers may be stale")


def summary_lines(cfg: Config, categories: bool = False) -> list[str]:
    report = load_report(cfg)
    if report is None:
        return []
    rows = [("all", report.get("measures", {}))]
    if categories:
        rows += [(c.get("id", "?"), c.get("measures", {})) for c in report.get("categories", [])]
    lines = []
    for name, m in rows:
        code, data = _n(m, "total_code"), _n(m, "total_data")
        if name != "all" and not code and not data:
            continue  # category with no units assigned yet
        lines.append(
            f"{name:>8}: code {_pct(_n(m, 'matched_code'), code)} matched, {_pct(_n(m, 'complete_code'), code)} linked"
            f" | data {_pct(_n(m, 'matched_data'), data)}"
            f" | functions {_n(m, 'matched_functions')}/{_n(m, 'total_functions')}"
            f" | units linked {_n(m, 'complete_units')}/{_n(m, 'total_units')}"
        )
    return lines


def run(cfg: Config, args: argparse.Namespace) -> int:
    why = stale(cfg)
    if why and getattr(args, "build", False):
        from .build import run as build
        code = build(cfg, argparse.Namespace(verbose=False, limit=8, lines=40, ninja_args=[]))
        if code != OK:
            return code
        why = stale(cfg)
    lines = summary_lines(cfg, categories=True)
    if not lines:
        print(f"error: no report at {cfg.rel(cfg.report)}")
        hint("build")
        return USAGE
    for line in lines:
        print(line)
    if why:
        print(f"warning: {why}")
        hint("build", "progress --build")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("--build", action="store_true", help="rebuild first when the report is older than HEAD")


COMMAND = Command("progress", "build", "matched / linked percentages from the last build's objdiff report "
                  "(warns when the report is older than HEAD; --build rebuilds it)", run, _configure)
