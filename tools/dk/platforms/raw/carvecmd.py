"""dk carve: write programmed regions, containers and files, and check them back."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from .analysis import Analysis, analyze, labels
from .common import out_dir, profile_of, read_image
from .jffs2 import Jffs2File


@dataclass(frozen=True)
class Region:
    offset: int
    length: int
    kind: str
    note: str

    @property
    def name(self) -> str:
        return f"{self.offset:08x}.{self.kind}"

    def line(self, digest: str) -> str:
        return f"{self.offset:08x} {self.length:08x} {digest} {self.kind} {self.note.replace(' ', '_')}".rstrip()


def regions_of(report: Analysis) -> list[Region]:
    """Every island, plus each container whose extent is known and is not already an island."""
    regions = [Region(i.offset, i.length, "island", ",".join(labels(i, report))) for i in report.islands]
    spans = {(i.offset, i.length) for i in report.islands}
    extra = [(h.body, h.end - h.body, h.kind, h.note) for h in report.hits if h.end is not None and h.end > h.body]
    extra += [(fs.offset, fs.end - fs.offset, "jffs2", f"nodes {fs.nodes}") for fs in report.jffs2]
    regions += [Region(*item) for item in extra if item[:2] not in spans]
    return sorted(set(regions), key=lambda r: (r.offset, r.kind, r.note))


def _safe_rel(path: str) -> Path | None:
    parts = path.split("/")
    if not path or path.startswith("/") or "\\" in path or any(part in ("", ".", "..") for part in parts):
        return None
    return Path(*parts)


def render_manifest(image_sha1: str, data: bytes, regions: list[Region], files: tuple[Jffs2File, ...]) -> str:
    lines = [f"image_sha1 {image_sha1}", f"image_size {len(data)}"]
    lines += [r.line(hashlib.sha1(data[r.offset:r.offset + r.length]).hexdigest()) for r in regions]
    lines += [f"file {hashlib.sha1(f.data).hexdigest()} {len(f.data):08x} {f.path}" for f in files]
    return "\n".join(lines) + "\n"


def _check(cfg: Config, out: Path, text: str, data: bytes, regions: list[Region], files: tuple[Jffs2File, ...]) -> int:
    manifest = out / "manifest.txt"
    if not manifest.is_file():
        print(f"error: manifest missing ({cfg.rel(manifest)})")
        hint("carve")
        return MISMATCH
    actual = manifest.read_text()
    if actual != text:
        print(f"MISMATCH: {cfg.rel(manifest)} does not match the image")
        for got, want in zip(actual.splitlines(), text.splitlines()):
            if got != want:
                print(f"  manifest {got}\n  image    {want}")
                break
        else:
            print(f"  manifest lines {len(actual.splitlines())}, image lines {len(text.splitlines())}")
        hint("carve")
        return MISMATCH
    wanted = [(out / "regions" / r.name, data[r.offset:r.offset + r.length]) for r in regions]
    wanted += [(out / "files" / _safe_rel(f.path), f.data) for f in files]
    for path, blob in wanted:
        if not path.is_file() or path.read_bytes() != blob:
            print(f"MISMATCH: {cfg.rel(path)}")
            hint("carve")
            return MISMATCH
    print(f"OK: {len(regions)} regions, {len(files)} files match {cfg.rel(manifest)}")
    hint("dis")
    return OK


def _write_tree(folder: Path, blobs: dict[Path, bytes]) -> None:
    """Make `folder` hold exactly `blobs` (paths inside it)."""
    for path, blob in blobs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    if folder.is_dir():
        for stale in sorted(folder.rglob("*")):
            if stale.is_file() and stale not in blobs:
                stale.unlink()


def run(cfg: Config, args: argparse.Namespace) -> int:
    loaded = read_image(cfg)
    if loaded is None:
        hint("status")
        return USAGE
    data, digest = loaded
    if digest != cfg.sha1:
        print(f"error: sha1 {digest} mismatch dk.toml {cfg.sha1}")
        hint("map")
        return MISMATCH
    report = analyze(data, profile_of(cfg), string_limit=0)
    regions = regions_of(report)
    files = tuple(f for f in report.files if _safe_rel(f.path) is not None)
    text = render_manifest(digest, data, regions, files)
    out = out_dir(cfg)
    if args.check:
        return _check(cfg, out, text, data, regions, files)
    _write_tree(out / "regions", {out / "regions" / r.name: data[r.offset:r.offset + r.length] for r in regions})
    _write_tree(out / "files", {out / "files" / _safe_rel(f.path): f.data for f in files})
    (out / "manifest.txt").write_text(text)
    print(f"wrote {len(regions)} regions and {len(files)} files to {cfg.rel(out)}")
    hint("carve --check", "dis")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("--check", action="store_true", help="re-hash carved output against the image; write nothing")


COMMAND = Command("carve", "analysis", "carve regions, containers and files with a manifest (--check verifies it)", run, _configure)
