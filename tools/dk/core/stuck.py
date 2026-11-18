"""D4: the rule-based stuck detector over a function's attempt log (structured fields only).

A function is stuck after round r when its best score did not rise in the last PATIENCE rounds, at least
MIN_ATTEMPTS attempts have run, and the best is not exact. replay() measures the rule on finished logs:
precision = flagged functions whose best never rose afterwards; saved = attempts logged after the flag
on those functions."""
from __future__ import annotations

import json
from pathlib import Path

PATIENCE = 2
MIN_ATTEMPTS = 8
EPS = 1e-6


def stuck_at(rounds: list[tuple[int, float, int]]) -> int | None:
    """rounds: (round, best percent among that round's attempts, attempts). The round after which the rule fires."""
    best, last_gain, n = -1.0, None, 0
    for i, (r, pct, k) in enumerate(rounds):
        n += k
        if pct > best + EPS:
            best, last_gain = pct, i
        if best >= 100.0 - EPS:
            return None
        if n >= MIN_ATTEMPTS and last_gain is not None and i - last_gain >= PATIENCE:
            return r
    return None


def stuck_state(rounds: list[tuple[int, float, int]]) -> dict:
    r = stuck_at(rounds)
    return {"stuck": r is not None, "after_round": r, "rounds": len(rounds),
            "best_pct": round(max((p for _, p, _ in rounds), default=0.0), 3), "rule": f"patience={PATIENCE},min={MIN_ATTEMPTS}"}


def _by_function(paths: list[Path]) -> dict[str, dict[int, list[float]]]:
    out: dict[str, dict[int, list[float]]] = {}
    for p in paths:
        for line in Path(p).read_text(errors="replace").splitlines():
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if x.get("kind") != "attempt" or not x.get("function"):
                continue
            pct = x.get("percent") if x.get("compiled") else None
            out.setdefault(x["function"], {}).setdefault(int(x.get("round", 0)), []).append(-1.0 if pct is None else float(pct))
    return out


def replay(paths: list[Path]) -> dict:
    flagged = correct = saved = total_attempts = 0
    rows = []
    for fn, rs in _by_function(paths).items():
        seq = [(r, max(v), len(v)) for r, v in sorted(rs.items())]
        total_attempts += sum(k for _, _, k in seq)
        r = stuck_at(seq)
        if r is None:
            continue
        flagged += 1
        at = max(p for rr, p, _ in seq if rr <= r)
        later = [p for rr, p, _ in seq if rr > r]
        rose = bool(later) and max(later) > at + EPS
        after = sum(k for rr, _, k in seq if rr > r)
        if not rose:
            correct += 1
            saved += after
        rows.append({"function": fn, "after_round": r, "rose_later": rose, "attempts_after": after})
    return {"functions": len(_by_function(paths)), "attempts": total_attempts, "flagged": flagged,
            "precision": round(correct / flagged, 3) if flagged else None, "saved_attempts": saved,
            "saved_share": round(saved / total_attempts, 3) if total_attempts else None, "rows": rows[:50]}
