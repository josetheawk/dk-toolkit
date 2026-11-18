"""How many compiles to run at once when a command is not told (`--jobs`).

Agents share the machine: a command that takes every core by default, run by a dozen agents at
once, loads it far past its core count. The default is dk.toml's `[platform] jobs` when set (a
positive number), else the environment's DK_JOBS, else the cores the machine leaves idle (cores
minus the 1-minute load average), at least 1 and at most `cap`.
"""

from __future__ import annotations

import os


def default_jobs(cfg=None, cap: int = 16) -> int:
    configured = (getattr(cfg, "platform_options", None) or {}).get("jobs") if cfg is not None else None
    for value in (configured, os.environ.get("DK_JOBS")):
        try:
            if value is not None and int(value) > 0:
                return int(value)
        except ValueError:
            pass
    cores = os.cpu_count() or 4
    try:
        load = os.getloadavg()[0]
    except (OSError, AttributeError):
        load = 0.0
    return max(1, min(cap, cores, int(cores - load + 0.5)))


def resolve(jobs: int, cfg=None, cap: int = 16) -> int:
    """`--jobs` as given (a positive number), or the default."""
    return jobs if jobs and jobs > 0 else default_jobs(cfg, cap)
