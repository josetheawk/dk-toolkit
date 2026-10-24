"""Never prove the hash against a stale split.

ninja re-runs dtk's split only through the depfile dtk writes; after a failed split that
record can be empty, and later symbols.txt / splits.txt edits would not re-split. Every
platform build goes through build(), which first removes the split output when any of its
inputs is newer, so the split always reflects the current config.
"""

from __future__ import annotations

import subprocess

from ...core.config import Config
from ...core.util import NINJA_TIMEOUT, run_ninja


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


def build(cfg: Config, targets: list[str] | None = None, timeout: int = NINJA_TIMEOUT) -> subprocess.CompletedProcess:
    ensure_split_fresh(cfg)
    return run_ninja(cfg.root, targets, timeout=timeout)
