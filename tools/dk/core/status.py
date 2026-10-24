"""dk status: where the project stands and which command to run next."""

from __future__ import annotations

import argparse
import subprocess

from .command import OK, Command, hint
from .config import Config
from .progress import load_report, summary_lines
from .util import sha1_file


def _git_dirty(cfg: Config) -> int | None:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=cfg.root, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return len([line for line in out.stdout.splitlines() if line.strip()])


def checks(cfg: Config) -> list[tuple[bool, str, str]]:
    """(passed, description, next command if failed), in dependency order."""
    rows = []
    orig_ok = cfg.original.is_file() and sha1_file(cfg.original) == cfg.sha1
    rows.append((orig_ok, f"original binary present and verified ({cfg.rel(cfg.original)})", "setup"))
    rows.append(((cfg.root / "build.ninja").is_file(), "build files generated (build.ninja)", "setup"))
    built_ok = cfg.built.is_file() and sha1_file(cfg.built) == cfg.sha1
    rows.append((built_ok, f"last build matches SHA-1 ({cfg.rel(cfg.built)})", "build"))
    rows.append((load_report(cfg) is not None, f"progress report present ({cfg.rel(cfg.report)})", "build"))
    return rows


def run(cfg: Config, args: argparse.Namespace) -> int:
    print(f"{cfg.name} ({cfg.game_id}, platform {cfg.platform})")
    next_cmd = None
    for passed, text, fix in checks(cfg):
        print(f"  [{'x' if passed else ' '}] {text}")
        if not passed and next_cmd is None:
            next_cmd = fix
    for line in summary_lines(cfg, categories=True):
        print("  " + line)
    dirty = _git_dirty(cfg)
    if dirty:
        print(f"  git: {dirty} uncommitted change(s)")
    print("  phase guide: playbook/process.md")
    if next_cmd:
        hint(next_cmd)
    else:
        hint("next")
    return OK


COMMAND = Command("status", "setup", "show project state and the next command to run", run)
