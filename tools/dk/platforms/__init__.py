"""Platform packages. Each one exposes:

    COMMANDS: list[Command]               extra platform-specific commands
    first_diff(cfg, built, original, limit) -> list[str]
                                          human summary of where two binaries differ
    prebuild(cfg)                         optional: called before `dk build` runs ninja
"""

from __future__ import annotations

import importlib
from types import ModuleType


def load(name: str) -> ModuleType:
    return importlib.import_module(f"{__name__}.{name}")
