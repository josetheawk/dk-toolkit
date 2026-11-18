"""dk map: regions, entry points, containers, files and names in a flat image."""

from __future__ import annotations

import argparse

from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from .analysis import analyze, map_lines
from .common import profile_of, read_image


def run(cfg: Config, args: argparse.Namespace) -> int:
    loaded = read_image(cfg)
    if loaded is None:
        hint("status")
        return USAGE
    data, digest = loaded
    profile = profile_of(cfg)
    if not profile.fill or any(not 0 <= b <= 255 for b in profile.fill) or profile.min_gap < 1:
        print("error: [platform] fill bytes must be 0..255 and the minimum gap >= 1")
        return USAGE
    report = analyze(data, profile, string_limit=args.limit)
    for line in map_lines(report, data, name=cfg.rel(cfg.original), sha1=digest, expected_sha1=cfg.sha1):
        print(line)
    hint("carve")
    return OK if digest == cfg.sha1 else MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("--limit", type=int, default=40, help="max strings kept per programmed region (default 40; 0 for none)")


COMMAND = Command("map", "analysis", "map an image: gaps, regions, entry points, containers, files, strings", run, _configure)
