"""dk route --unlock: which open function or data fix links the most code.

A pure function of the objdiff report and `dk route`'s rows (no compiles). Per unlinked unit the blockers are its
open functions (route rows; an auto-generated unit's functions get `no-source`), its data sections below 100 %
(report `sections`, the same list route's `data_open` shows) and, when nothing else is open, `link` (layout or
emission order: `dk next --link`). Each blocker gets a worker, a tool and a cost from one table (COST, overridable in
dk.toml `[platform.unlock]`); a failed past attempt adds `per_try` (LoRA grind-state `idle` plus CPU attempts from the
close-queue logs and close memos, which route does not see). value = the unit's total_code / Σ cost. Units held by a
frozen function (the LoRA-2 gate) are listed apart, never valued or queued. THAW lists the frozen sets CPU tools may
work (owner decision 2026-11-07: the near-sample snapshots); those functions go to the cpu worker but stay frozen for
every dataset filter (gate.never, llmfixes), which this module never touches.
"""

from __future__ import annotations

import json
from pathlib import Path

COST = {"near-register": 1, "near-scheduling": 1, "near-shape": 3, "far": 6, "fresh": 8, "giant": 12, "giant_per_kb": 1,
        "unit-data": 2, "compiler-version": 1, "sinit": 6, "data": 2, "bss": 1, "link": 1, "no-source": 20, "thawed": 1,
        "per_try": 1}
THAW: tuple = ()   # 2026-11-18: every eval set is frozen again (decisions.md); `--thaw SET` opts in, the commit guard still refuses
TOOL = {"near-register": "close, regs", "near-scheduling": "close", "near-shape": "close, llm-grind", "far": "llm-grind",
        "fresh": "gen, llm-grind", "giant": "inlines, close", "unit-data": "diff --literals, gen --data",
        "compiler-version": "unit flags", "sinit": "unit naming and data", "data": "gen --data, diff --literals",
        "link": "next --link, reorder", "no-source": "unit propose, unit add", "thawed": "close --queue --thaw"}
ORDER = ("cpu", "data", "lora", "claude", "blocked")          # the linked_if_done curve, cheapest worker first
CODE_SECTIONS = (".text", ".init", "extab")


def _code(unit: dict) -> int:
    return int(unit.get("measures", {}).get("total_code", 0) or 0)


def data_open(unit: dict, census: dict | None = None) -> list[dict]:
    """The unit's data sections below 100 % (a section with no score counts as 0 %). With a data census
    (`dk close-data --census`: dk's scoring, weak copies and stripped objects allowed for) that has scored this
    unit, its sections replace the report's raw objdiff ones, which overstate data blockers."""
    sizes = {s["name"]: int(s.get("size", 0) or 0) for s in unit.get("sections", [])}
    row = (census or {}).get(unit.get("name"))
    if row and row.get("dk") is not None:
        return [{"name": sec, "size": sizes.get(sec, 0), "fuzzy": round(float(pct), 2)} for sec, pct in sorted(row["dk"].items())]
    return [{"name": s["name"], "size": sizes[s["name"]], "fuzzy": round(float(s.get("fuzzy_match_percent") or 0.0), 2)}
            for s in unit.get("sections", []) if not s["name"].startswith(CODE_SECTIONS)
            and float(s.get("fuzzy_match_percent") or 0.0) < 100.0]


def tried_counts(logs: list[Path], memo_dirs: list[Path]) -> dict[str, int]:
    """`unit::function` -> CPU close attempts that did not reach exact: close-queue log lines, or the close memo's
    `runs` (the larger of the two, since a queue run also writes the memo)."""
    log: dict[str, int] = {}
    for path in logs:
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("unit") and r.get("function") and not r.get("exact"):
                key = f"{r['unit']}::{r['function']}"
                log[key] = log.get(key, 0) + 1
    memo: dict[str, int] = {}
    for d in memo_dirs:
        for path in sorted(d.glob("*.json")) if d.is_dir() else []:
            try:
                m = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(m, dict) and m.get("unit") and m.get("function") and (m.get("best") or 0.0) < 100.0:
                key = f"{m['unit']}::{m['function']}"
                memo[key] = max(memo.get(key, 0), int(m.get("runs", 1) or 1))
    return {k: max(log.get(k, 0), memo.get(k, 0)) for k in sorted(set(log) | set(memo))}


def function_blocker(row: dict, cost: dict, thaw: tuple, tried: dict) -> dict:
    cls, held = row["class"], row.get("frozen") or ""
    cpu = tried.get(f"{row['unit']}::{row['function']}", 0)
    base = cost[cls] + (cost["giant_per_kb"] * row["size"] // 1024 if cls == "giant" else 0)
    worker, tool = row["worker"], TOOL[cls]
    if held and held in thaw:
        worker, tool, base = "cpu", TOOL["thawed"], cost["thawed"]
    elif held:
        worker = "frozen"
    return {"kind": "function", "name": row["function"], "class": cls, "worker": worker, "tool": tool,
            "size": row["size"], "fuzzy": row.get("fuzzy"), "held": held, "thawed": bool(held and held in thaw),
            "lora_tried": row.get("attempts", 0), "cpu_tried": cpu,
            "cost": None if worker == "frozen" else base + cost["per_try"] * (row.get("idle", 0) + cpu)}


def unit_blockers(unit: dict, rows: list[dict], cost: dict, thaw: tuple, tried: dict, census: dict | None = None) -> list[dict]:
    out = [function_blocker(r, cost, thaw, tried) for r in sorted(rows, key=lambda r: (r["address"], r["function"]))]
    if unit.get("metadata", {}).get("auto_generated"):
        out += [{"kind": "function", "name": f["name"], "class": "no-source", "worker": "blocked", "tool": TOOL["no-source"],
                 "size": int(f.get("size", 0) or 0), "fuzzy": round(float(f.get("fuzzy_match_percent") or 0.0), 2),
                 "held": "", "thawed": False, "lora_tried": 0, "cpu_tried": 0, "cost": cost["no-source"]}
                for f in unit.get("functions", []) if float(f.get("fuzzy_match_percent") or 0.0) < 100.0]
    for s in data_open(unit, census):
        out.append({"kind": "data", "name": s["name"], "class": "data", "worker": "data", "tool": TOOL["data"],
                    "size": s["size"], "fuzzy": s["fuzzy"], "held": "", "thawed": False, "lora_tried": 0, "cpu_tried": 0,
                    "cost": cost["bss"] if s["name"] in (".bss", ".sbss", ".sbss2") else cost["data"]})
    if not out:
        out.append({"kind": "link", "name": "layout", "class": "link", "worker": "cpu", "tool": TOOL["link"], "size": 0,
                    "fuzzy": None, "held": "", "thawed": False, "lora_tried": 0, "cpu_tried": 0, "cost": cost["link"]})
    return out


def rank(report: dict, rows: list[dict], cost: dict | None = None, thaw: tuple = THAW,
         tried: dict | None = None, census: dict | None = None) -> dict:
    """Every unlinked unit with its blockers, cost and value; queued units by (-value, -total_code, name), held
    units (a frozen blocker) apart; plus the linked % if every blocker of the workers up to each one were done."""
    cost = {**COST, **(cost or {})}
    total = int(report.get("measures", {}).get("total_code", 0) or 0) or 1
    linked = sum(_code(u) for u in report.get("units", []) if u.get("metadata", {}).get("complete"))
    by_unit: dict[str, list[dict]] = {}
    for r in rows:
        by_unit.setdefault(r["unit"], []).append(r)
    units, held = [], []
    for u in report.get("units", []):
        if u.get("metadata", {}).get("complete"):
            continue
        bs = unit_blockers(u, by_unit.get(u["name"], []), cost, thaw, tried or {}, census)
        fns = [b for b in bs if b["kind"] == "function"]
        row = {"unit": u["name"], "total_code": _code(u), "unlock_pct": round(100.0 * _code(u) / total, 4),
               "open_functions": len(fns), "open_bytes": sum(b["size"] for b in fns),
               "data_open": [b["name"] for b in bs if b["kind"] == "data"],
               "workers": sorted({b["worker"] for b in bs}), "blockers": bs,
               "last_blocker": bs[0]["name"] if len(bs) == 1 else None,
               "held": sorted({b["held"] for b in bs if b["worker"] == "frozen"}),
               "thawed": sum(b["thawed"] for b in bs)}
        if row["held"]:
            row["cost"] = row["value"] = None
            held.append(row)
        else:
            row["cost"] = sum(b["cost"] for b in bs)
            row["value"] = round(row["total_code"] / row["cost"], 2) if row["cost"] else float(row["total_code"])
            units.append(row)
    units.sort(key=lambda r: (-r["value"], -r["total_code"], r["unit"]))
    held.sort(key=lambda r: (-r["total_code"], r["unit"]))
    curve, alone = {}, {}
    for k, w in enumerate(ORDER):
        done = set(ORDER[:k + 1])
        curve[w] = round(100.0 * (linked + sum(r["total_code"] for r in units if set(r["workers"]) <= done)) / total, 2)
        alone[w] = round(100.0 * sum(r["total_code"] for r in units if set(r["workers"]) == {w}) / total, 2)
    return {"total_code": total, "linked_pct": round(100.0 * linked / total, 2), "units": units, "held": held,
            "linked_if_done": curve, "linked_alone": alone, "thaw": list(thaw), "cost": cost}


def annotate(rows: list[dict], ranking: dict) -> list[dict]:
    """Copies of route's function rows with the unit's `unlock_bytes`, `unlock_value` and `last_blocker` (this
    function is the unit's only blocker); a thawed function's copy gets worker cpu and no `frozen` mark, so the
    CPU queue may take it (the gate files keep it frozen for every dataset filter)."""
    units = {u["unit"]: u for u in ranking["units"]}
    out = []
    for r in rows:
        u = units.get(r["unit"])
        b = next((b for b in u["blockers"] if b["kind"] == "function" and b["name"] == r["function"]), None) if u else None
        extra = {"unlock_bytes": u["total_code"] if u else 0, "unlock_value": u["value"] if u else 0.0,
                 "last_blocker": bool(u and u["last_blocker"] == r["function"])}
        if b and b["thawed"]:
            extra.update(worker="cpu", frozen="", thawed=r.get("frozen"))
        out.append({**r, **extra})
    return out


def leverage(units: list[dict], top: int) -> float | None:
    """Σ total_code / Σ open-function bytes over the first `top` queued units (None when no function bytes)."""
    part = units[:top]
    cost = sum(u["open_bytes"] for u in part)
    return round(sum(u["total_code"] for u in part) / cost, 2) if cost else None


def render(ranking: dict, top: int, data: bool = False) -> list[str]:
    units = ranking["units"][:top] if top else ranking["units"]
    out = [f"{len(ranking['units'])} unlinked unit(s) queued, {len(ranking['held'])} held by a frozen function; "
           f"linked now {ranking['linked_pct']:.2f} %  (thawed for CPU: {', '.join(ranking['thaw']) or 'none'})", ""]
    for u in units:
        if data and not u["data_open"]:
            continue
        out.append(f"{u['unit']}  0x{u['total_code']:X} B (+{u['unlock_pct']:.2f} %)  cost {u['cost']}  value {u['value']:g}"
                   + ("  LAST BLOCKER" if u["last_blocker"] else ""))
        for b in u["blockers"]:
            if data and b["kind"] != "data":
                continue
            pct = f" {b['fuzzy']:.2f}%" if b["fuzzy"] is not None else ""
            tried = f" tried cpu {b['cpu_tried']}, lora {b['lora_tried']}" if b["cpu_tried"] or b["lora_tried"] else ""
            out.append(f"    {b['name']} 0x{b['size']:X}{pct} [{b['class']} {b['worker']}: {b['tool']}]{tried}"
                       + (f" (thawed from {b['held']})" if b["thawed"] else ""))
    if ranking["held"] and not data:
        out += ["", "held (a frozen LoRA-2 gate function blocks them; never queued):"]
        out += [f"  {u['unit']}  0x{u['total_code']:X} B (+{u['unlock_pct']:.2f} %)  {', '.join(u['held'])}"
                for u in ranking["held"][:top or None]]
    out += ["", "linked % if every blocker of these workers were done: "
            + "  ".join(f"{w} {p:.2f}" for w, p in ranking["linked_if_done"].items()),
            "units only one worker blocks (+points): " + "  ".join(f"{w} {p:.2f}" for w, p in ranking["linked_alone"].items())]
    return out
