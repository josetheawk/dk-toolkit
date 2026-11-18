"""dk setup: put the original binary in place and generate the build."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from .command import OK, USAGE, Command, hint
from .config import Config
from .util import find_disc_images, sha1_file


def _check_environment() -> list[str]:
    problems = []
    if sys.version_info < (3, 11):
        problems.append(f"python >= 3.11 required (have {sys.version.split()[0]})")
    if shutil.which("ninja") is None:
        problems.append("ninja not found on PATH")
    return problems


def _place_original(cfg: Config) -> tuple[bool, str]:
    """Returns (ready, message). ready=False with a message means stop."""
    if cfg.original.is_file():
        digest = sha1_file(cfg.original)
        if digest != cfg.sha1:
            return False, f"{cfg.rel(cfg.original)} has SHA-1 {digest}, expected {cfg.sha1} (wrong version or modified file)"
        return True, f"original OK: {cfg.rel(cfg.original)}"

    # No extracted binary: link a disc image next to it and let the build extract it.
    images = find_disc_images(cfg.disc_search)
    if not images:
        searched = ", ".join(cfg.rel(d) for d in cfg.disc_search) or "(no disc_search in dk.toml)"
        return False, f"{cfg.rel(cfg.original)} missing and no disc image found in: {searched}"
    image = images[0]
    base = cfg.disc_search[0] if cfg.disc_search else cfg.original.parent
    base.mkdir(parents=True, exist_ok=True)
    link = base / image.name
    if not link.exists():
        link.symlink_to(image)
    return True, f"disc image linked: {cfg.rel(link)} -> {image} (extracted on first build)"


def run(cfg: Config, args: argparse.Namespace) -> int:
    problems = _check_environment()
    if problems:
        for p in problems:
            print(f"error: {p}")
        return USAGE

    ready, message = _place_original(cfg)
    print(message)
    if not ready:
        return USAGE

    requirements = cfg.root / "requirements.txt"
    venv_python = cfg.root / ".venv" / "bin" / "python"
    if requirements.is_file() and not args.no_venv and not venv_python.is_file():
        print("creating .venv and installing requirements.txt")
        for step in ([sys.executable, "-m", "venv", ".venv"],
                     [str(venv_python), "-m", "pip", "install", "-q", "-r", "requirements.txt"]):
            if subprocess.run(step, cwd=cfg.root).returncode != 0:
                print(f"error: {' '.join(step)} failed")
                return USAGE

    configure = [sys.executable, *cfg.configure, *args.configure_args]
    proc = subprocess.run(configure, cwd=cfg.root)
    if proc.returncode != 0:
        print(f"error: {' '.join(cfg.configure)} failed ({proc.returncode})")
        return USAGE
    print("build files generated")
    hint("build")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-venv", action="store_true", help="skip creating .venv from requirements.txt")
    p.add_argument("configure_args", nargs=argparse.REMAINDER, help="extra arguments passed to configure.py")


COMMAND = Command("setup", "setup", "place the original binary, verify its SHA-1, generate build files", run, _configure)
