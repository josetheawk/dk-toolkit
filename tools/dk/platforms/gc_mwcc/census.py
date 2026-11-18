"""T63 D5: data census (CPU, reads records only): real before -> exact pairs that need 2+ changes by fix kind and size band,
stacked fix-kind coverage of them, and stacked score paths whose step order improves at every step."""
from __future__ import annotations

import difflib
import json
from collections import Counter
from pathlib import Path


def hunks(a: str, b: str) -> int:
    sm = difflib.SequenceMatcher(None, a.split("\n"), b.split("\n"), autojunk=False)
    return sum(1 for o in sm.get_opcodes() if o[0] != "equal")


def band(size: int) -> str:
    return "<=0x80" if size <= 0x80 else "<=0x200" if size <= 0x200 else "<=0x400" if size <= 0x400 else ">0x400"


def pairs(paths: list[Path]) -> list[dict]:
    """Distinct (function, before) pairs from history records (wrong/exact) and shape-fix records (before/after)."""
    from .llmfixes import fix_kinds
    seen, out = set(), []
    for p in paths:
        for line in p.read_text(errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            w, e = r.get("wrong") or r.get("before"), r.get("exact") or r.get("after")
            if not (isinstance(w, str) and isinstance(e, str)) or (r.get("function"), hash(w)) in seen:
                continue
            seen.add((r.get("function"), hash(w)))
            out.append({"function": r.get("function"), "size": int(r.get("size") or 0), "changes": hunks(w, e),
                        "kinds": r.get("fix_kind") or fix_kinds(w, e)})
    return out


def monotone(path: list[float]) -> bool:
    """A score path (puzzle first, answer last) that never drops when walked from the puzzle to the answer."""
    return all(b >= a for a, b in zip(path, path[1:]))


def census(history: list[Path], stacked: list[Path]) -> dict:
    ps = pairs(history)
    multi = [p for p in ps if p["changes"] >= 2]
    real_kinds = Counter(k for p in multi for k in p["kinds"])
    puzzles = []
    for f in stacked:
        for line in f.read_text(errors="replace").splitlines():
            try:
                puzzles.append(json.loads(line))
            except ValueError:
                continue
    st_kinds = Counter(k for q in puzzles for k in q.get("fix_kind", []))
    covered = [k for k in real_kinds if st_kinds.get(k)]
    paths = [q["score_path"] for q in puzzles if isinstance(q.get("score_path"), list)]
    return {"pairs": len(ps), "multi_change_pairs": len(multi),
            "by_size_band": dict(Counter(band(p["size"]) for p in multi)),
            "by_fix_kind": dict(real_kinds.most_common()),
            "stacked_puzzles": len(puzzles), "stacked_kinds": dict(st_kinds),
            "kind_coverage": round(len(covered) / len(real_kinds), 3) if real_kinds else None,
            "stacked_paths": len(paths), "stacked_nondecreasing": sum(1 for x in paths if monotone(x)),
            "stacked_strictly_improving": sum(1 for x in paths if all(b > a for a, b in zip(x, x[1:])))}
