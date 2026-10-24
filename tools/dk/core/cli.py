"""Entry point: `./dk <command>`. With no command, lists commands by group."""

from __future__ import annotations

import argparse
import sys

from .. import platforms
from . import build, doclint, floor, nextcmd, progress, quality, retro, rmcmd, setup, status, testcmd, toolkitcmd, verifyhead
from .command import GROUPS, USAGE, Command
from .config import ConfigError, load
from .util import project_lock

CORE_COMMANDS: list[Command] = [
    status.COMMAND,
    setup.COMMAND,
    build.COMMAND,
    verifyhead.COMMAND,
    progress.COMMAND,
    floor.COMMAND,
    nextcmd.COMMAND,
    rmcmd.COMMAND,
    doclint.COMMAND,
    quality.COMMAND,
    retro.COMMAND,
    testcmd.COMMAND,
    toolkitcmd.COMMAND,
]


def _commands(platform: str | None) -> list[Command]:
    cmds = list(CORE_COMMANDS)
    if platform:
        cmds += platforms.load(platform).COMMANDS
    return cmds


def _overview(cmds: list[Command]) -> str:
    out = ["usage: ./dk <command> [args]   (./dk <command> -h for details)", ""]
    for key, title in GROUPS:
        group = [c for c in cmds if c.group == key]
        if not group:
            continue
        out.append(f"{title}:")
        out += [f"  {c.name:<16} {c.summary}" for c in group]
        out.append("")
    out.append("Exit codes: 0 ok/match, 1 mismatch, 2 usage/environment, 3 build error.")
    out.append("Start with: ./dk status")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        cfg = load()
    except ConfigError as e:
        cfg = None
        config_error = str(e)
    cmds = _commands(cfg.platform if cfg else None)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_overview(cmds))
        return 0

    parser = argparse.ArgumentParser(prog="./dk", add_help=True)
    sub = parser.add_subparsers(dest="command", required=True)
    by_name = {}
    for c in cmds:
        p = sub.add_parser(c.name, help=c.summary, description=c.summary)
        if c.configure:
            c.configure(p)
        by_name[c.name] = c
    args = parser.parse_args(argv)
    cmd = by_name[args.command]
    if cmd.needs_config and cfg is None:
        print(f"error: {config_error}")
        return USAGE
    if cmd.locked and cfg is not None:
        with project_lock(cfg.root):
            return cmd.run(cfg, args)
    return cmd.run(cfg, args)
