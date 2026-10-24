"""dk progress: summarise the objdiff report written by the build."""

from __future__ import annotations

import argparse
import json
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
    lines = summary_lines(cfg, categories=True)
    if not lines:
        print(f"error: no report at {cfg.rel(cfg.report)}")
        hint("build")
        return USAGE
    for line in lines:
        print(line)
    return OK


COMMAND = Command("progress", "build", "matched / linked percentages from the last build's objdiff report", run)
