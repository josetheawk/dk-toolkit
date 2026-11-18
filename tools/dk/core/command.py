"""Command registry types and the shared exit-code contract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from .config import Config

# Exit codes. Agents branch on these, so never reuse one for another meaning.
OK = 0          # success / exact match
MISMATCH = 1    # ran fine, result differs from the target (hash, diff, lint)
USAGE = 2       # bad arguments or missing environment (files, tools, setup)
BUILD_ERROR = 3  # compiler / linker / ninja failed before anything could be compared

# Groups, in the order `./dk` lists them. Mirrors playbook/process.md phases.
GROUPS = (
    ("setup", "Setup"),
    ("build", "Build and verify"),
    ("analysis", "Analysis (splitting, naming)"),
    ("matching", "Matching loop"),
    ("docs", "Docs and retro"),
)


@dataclass(frozen=True)
class Command:
    name: str
    group: str
    summary: str
    run: Callable[[Config, argparse.Namespace], int]
    configure: Callable[[argparse.ArgumentParser], None] | None = None
    needs_config: bool = True
    locked: bool = False      # holds the project lock for the whole run (edits shared files / commits)


def hint(*commands: str) -> None:
    """Print the follow-up command(s). Every command ends with one."""
    print("next: " + "  |  ".join(f"./dk {c}" for c in commands))
