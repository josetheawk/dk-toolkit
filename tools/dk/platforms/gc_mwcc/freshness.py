"""Never prove the hash against a stale split.

ninja re-runs dtk's split only through the depfile dtk writes; after a failed split that
record can be empty, and later symbols.txt / splits.txt edits would not re-split. Every
platform build goes through build(), which first removes the split output when any of its
inputs is newer, so the split always reflects the current config.
"""

from __future__ import annotations

import re
import subprocess
import time

from ...core.config import Config
from ...core.util import NINJA_TIMEOUT, run_ninja

# Waiting for another agent's source while holding the project lock stalled every queued
# command behind it (16 agents, 25 commands, 20+ minutes): fail fast, naming the culprit.
WAIT_STEP, WAIT_TRIES = 20, 0
_FAILED_EDGE = re.compile(r"^FAILED: (?:\[code=\d+\] )?(?P<outputs>.+)$", re.M)


def ensure_split_fresh(cfg: Config) -> bool:
    out = cfg.root / "build" / cfg.game_id / "config.json"
    if not out.is_file():
        return False
    opts = cfg.platform_options
    inputs = [cfg.root / "config" / cfg.game_id / "config.yml", cfg.root / opts["symbols"], cfg.root / opts["splits"]]
    newest = max((p.stat().st_mtime_ns for p in inputs if p.is_file()), default=0)
    if newest > out.stat().st_mtime_ns:
        out.unlink()
        return True
    return False


def hash_target(cfg: Config) -> list[str]:
    """The ninja target that links the binary and checks its hash. It depends only on linked
    units' objects, so another agent's unfinished (unlinked) source never blocks a hash proof."""
    return [f"build/{cfg.game_id}/ok"]


def others_broken(output: str, own: set[str]) -> set[str]:
    """Objects whose compile failed that are not the caller's own: in a tree shared by parallel
    agents, another agent's source can be mid-edit for a moment."""
    failed = {path for m in _FAILED_EDGE.finditer(output) for path in m["outputs"].split()}
    return {p for p in failed if p.endswith(".o") and "/src/" in p} - own


def build(cfg: Config, targets: list[str] | None = None, timeout: int = NINJA_TIMEOUT,
          own: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    """ninja after a split freshness check. When only other units' sources fail to compile
    (another agent mid-edit), wait for them a little instead of failing every hash proof."""
    ensure_split_fresh(cfg)
    proc = run_ninja(cfg.root, targets, timeout=timeout)
    if proc.returncode not in (0, 124):
        broken = others_broken(proc.stdout + proc.stderr, set(targets or ()) | set(own))
        if broken:
            print(f"blocked: {', '.join(sorted(broken))} does not compile (another agent's edit in progress?); "
                  f"retry in a minute", flush=True)
    for attempt in range(WAIT_TRIES):
        if proc.returncode in (0, 124):
            return proc
        broken = others_broken(proc.stdout + proc.stderr, set(targets or ()) | set(own))
        if not broken:
            return proc
        print(f"waiting: {', '.join(sorted(broken))} does not compile (another agent's edit in progress?); "
              f"retry in {WAIT_STEP} s ({attempt + 1}/{WAIT_TRIES})", flush=True)
        time.sleep(WAIT_STEP)
        ensure_split_fresh(cfg)
        proc = run_ninja(cfg.root, targets, timeout=timeout)
    return proc
