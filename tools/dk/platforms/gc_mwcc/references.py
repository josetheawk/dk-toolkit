"""Reference decomp sources (SDK, runtime, ...) pinned in dk.toml and fetched on demand."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from ...core.config import Config


@dataclass(frozen=True)
class Reference:
    name: str
    url: str
    commit: str
    sources: tuple[str, ...]               # dirs (relative to the checkout) scanned for C/C++ files
    includes: tuple[tuple[str, str], ...]  # (flag, dir) as the reference build passes them
    compilers: tuple[str, ...]
    flags: tuple[str, ...]
    variants: tuple[tuple[str, ...], ...]  # extra flag sets tried in turn
    checkout: str = ""                     # shared checkout dir name (several entries, one repo)
    src_root: str = "src"                  # source paths are named relative to this dir
    unit_prefix: str = ""                  # prefix for our unit path (e.g. "dolphin/")
    vendor: tuple[tuple, ...] = ()   # (reference dir, our dir[, [subpaths]]) header trees copied once

    def unit_path(self, rel: str) -> str:
        return f"{self.unit_prefix}{rel}"

    def path(self, cfg: Config) -> Path:
        return cfg.root / "build" / "references" / (self.checkout or self.name)


def load_references(cfg: Config) -> list[Reference]:
    refs = []
    for raw in cfg.platform_options.get("references", []):
        refs.append(Reference(
            name=raw["name"],
            url=raw["url"],
            commit=raw["commit"],
            sources=tuple([raw["sources"]] if isinstance(raw.get("sources", "src"), str) else raw["sources"])
            if "sources" in raw else ("src",),
            checkout=raw.get("checkout", ""),
            src_root=raw.get("src_root", "src"),
            unit_prefix=raw.get("unit_prefix", ""),
            vendor=tuple(tuple(v) for v in raw.get("vendor", [["include", f"include/sdk/{raw['name']}"]])),
            includes=tuple((flag, d) for flag, d in raw.get("includes", [])),
            compilers=tuple(raw["compilers"]),
            flags=tuple(raw.get("flags", [])),
            variants=tuple(tuple(v) for v in raw.get("variants", [[]])),
        ))
    return refs


def ensure(cfg: Config, ref: Reference) -> Path:
    """Check out ref at its pinned commit (shallow). Never runs anything from the checkout."""
    dest = ref.path(cfg)
    head = dest / ".git" / "HEAD"
    if head.is_file():
        current = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True).stdout.strip()
        if current == ref.commit:
            return dest
    dest.mkdir(parents=True, exist_ok=True)
    for cmd in (["git", "init", "-q"],
                ["git", "fetch", "-q", "--depth", "1", ref.url, ref.commit],
                ["git", "checkout", "-q", "--force", "FETCH_HEAD"]):
        proc = subprocess.run(cmd, cwd=dest, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed for {ref.name}: {proc.stderr.strip()}")
    return dest
