"""Helpers the raw commands share: the platform profile, the output folder, the verified image."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ... import platforms
from ...core.config import Config
from .analysis import Profile


def profile_of(cfg: Config) -> Profile:
    """The project's profile: the first `profile(cfg)` along its platform chain."""
    return platforms.hook(cfg.platform, "profile")(cfg)


def out_dir(cfg: Config) -> Path:
    return cfg.root / profile_of(cfg).out


def read_image(cfg: Config) -> tuple[bytes, str] | None:
    """(bytes, sha1) of the original, or None (after printing why) when it is missing."""
    if not cfg.original.is_file():
        print(f"error: image missing ({cfg.rel(cfg.original)})")
        return None
    data = cfg.original.read_bytes()
    return data, hashlib.sha1(data).hexdigest()
