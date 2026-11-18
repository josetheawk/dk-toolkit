"""dk asset: embedded game data that a source includes at build time and that is never committed.

A unit whose data embeds a blob of game bytes (a texture inside .data) declares it in its
units.toml entry:

    assets = [{ name = "logo_shape", address = 0x80401000, size = 0x4050 }]

Before anything compiles, the build (a ninja pre-compile step that configure.py writes)
extracts each declared asset from the original binary into
build/<GAME>/include/assets/<name>.inc as a C initializer list, and the source writes

    unsigned char lbl_80401000[0x4050] = {
    #include "assets/logo_shape.inc"
    };

build/ is git-ignored, so the bytes never reach a commit; a fresh clone gets them from
`dk setup` + `dk build`, and save/accept's clean-tree verify extracts them like any build input.
The declaration rides in the unit's own units.toml entry, so it lands with the unit's save.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from ...core.util import project_lock, sha1_file
from .dol import Dol

KEY = "assets"                                   # the units.toml object key
WIDTHS = {1: "unsigned char", 2: "unsigned short", 4: "unsigned int"}
NO_BYTES = (".text", ".init", ".bss", ".sbss", ".sbss2")
_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_][A-Za-z0-9_.-]*)*$")
PER_LINE = 16                                    # bytes per line of the .inc


class AssetError(Exception):
    pass


@dataclass(frozen=True)
class Asset:
    unit: str            # source path relative to src/
    name: str            # include path below assets/, without .inc
    address: int
    size: int
    width: int = 1       # element size: 1 (bytes), 2 or 4 (big-endian halfwords / words)

    @property
    def include(self) -> str:
        return f"assets/{self.name}.inc"

    @property
    def end(self) -> int:
        return self.address + self.size


def out_dir(cfg: Config) -> Path:
    """Where the .inc files go: under the include dir the game's flags already search (-i build/<GAME>/include)."""
    return cfg.root / "build" / cfg.game_id / "include" / "assets"


def parse_entry(unit: str, raw: dict) -> Asset:
    """One units.toml asset table, validated."""
    unknown = sorted(set(raw) - {"name", "address", "size", "width"})
    if unknown:
        raise AssetError(f"{unit}: asset has unknown key(s) {', '.join(unknown)}")
    try:
        asset = Asset(unit, str(raw["name"]), int(raw["address"]), int(raw["size"]), int(raw.get("width", 1)))
    except KeyError as e:
        raise AssetError(f"{unit}: asset is missing {e.args[0]!r} (needs name, address, size)") from None
    if not _NAME.match(asset.name) or ".." in asset.name:
        raise AssetError(f"{unit}: asset name {asset.name!r} must be a relative path of [A-Za-z0-9_.-] parts")
    if asset.size <= 0:
        raise AssetError(f"{unit}: asset {asset.name} has size {asset.size}")
    if asset.width not in WIDTHS:
        raise AssetError(f"{unit}: asset {asset.name} width must be 1, 2 or 4")
    if asset.size % asset.width or asset.address % asset.width:
        raise AssetError(f"{unit}: asset {asset.name} address/size are not multiples of its width {asset.width}")
    return asset


def declared(units_text: str) -> list[Asset]:
    """Every asset the units file declares, in file order; names must be unique."""
    out: list[Asset] = []
    for lib in tomllib.loads(units_text).get("lib", []):
        for obj in lib.get("objects", []):
            for raw in obj.get(KEY, []) or []:
                out.append(parse_entry(obj["path"], raw))
    seen: dict[str, str] = {}
    for a in out:
        if a.name in seen:
            raise AssetError(f"asset name {a.name} is declared twice ({seen[a.name]}, {a.unit})")
        seen[a.name] = a.unit
    return out


def load(cfg: Config) -> list[Asset]:
    path = cfg.root / cfg.platform_options["units"]
    return declared(path.read_text()) if path.is_file() else []


def render(asset: Asset, data: bytes, origin: str) -> str:
    """The initializer list: a comment line, then the elements (hex, big-endian), a comma after each."""
    if len(data) != asset.size:
        raise AssetError(f"{asset.name}: read {len(data)} bytes, expected {asset.size}")
    w = asset.width
    values = [int.from_bytes(data[i:i + w], "big") for i in range(0, len(data), w)]
    per_line = max(1, PER_LINE // w)
    lines = [f"/* {asset.name}: 0x{asset.size:X} bytes at 0x{asset.address:08X} of {origin}, extracted at build "
             f"time by ./dk asset (build/ only: game data is never committed) */"]
    for i in range(0, len(values), per_line):
        lines.append(" ".join(f"0x{v:0{2 * w}X}," for v in values[i:i + per_line]))
    return "\n".join(lines) + "\n"


def read_original(cfg: Config) -> Dol:
    if not cfg.original.is_file():
        raise AssetError(f"{cfg.rel(cfg.original)} missing: run ./dk setup (the build extracts it from a disc image)")
    digest = sha1_file(cfg.original)
    if digest != cfg.sha1:
        raise AssetError(f"{cfg.rel(cfg.original)} has SHA-1 {digest}, expected {cfg.sha1}")
    return Dol.parse(cfg.original.read_bytes())


def extract(cfg: Config, assets: list[Asset] | None = None) -> tuple[list[str], list[str]]:
    """Write every declared asset's .inc (only those whose content changed, so ninja's restat
    keeps unrelated units from recompiling) and delete .inc files nothing declares any more.
    Returns (written, removed) as repo-relative paths."""
    assets = load(cfg) if assets is None else assets
    base = out_dir(cfg)
    wanted = {base / f"{a.name}.inc": a for a in assets}
    written: list[str] = []
    if assets:
        dol = read_original(cfg)
        for path, a in sorted(wanted.items()):
            try:
                data = dol.read(a.address, a.size)
            except ValueError as e:
                raise AssetError(f"{a.unit}: asset {a.name}: {e}") from None
            text = render(a, data, cfg.rel(cfg.original))
            if not path.is_file() or path.read_text() != text:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".inc.tmp")
                tmp.write_text(text)
                tmp.replace(path)
                written.append(cfg.rel(path))
    removed: list[str] = []
    if base.is_dir():
        for stale in sorted(base.rglob("*.inc")):
            if stale not in wanted:
                stale.unlink()
                removed.append(cfg.rel(stale))
    return written, removed


_ensured: set[Path] = set()


def ensure(cfg: Config) -> None:
    """Extract the declared assets if one is missing (private compiles run before any build).
    Once per process; never raises: a compile that needs a missing asset reports it anyway."""
    if cfg.root in _ensured:
        return
    _ensured.add(cfg.root)
    try:
        assets = load(cfg)
        if any(not (out_dir(cfg) / f"{a.name}.inc").is_file() for a in assets):
            extract(cfg, assets)
    except (AssetError, OSError, tomllib.TOMLDecodeError) as e:
        print(f"note: embedded assets not extracted: {e}")


# --- checks for `add` ------------------------------------------------------------------------

def unit_ranges(cfg: Config, unit: str) -> list[tuple[str, int, int]]:
    from .dtk_symbols import read_units
    for u in read_units(cfg.root / cfg.platform_options["splits"]):
        if u.name == unit:
            return list(u.splits)
    return []


def relocations_inside(cfg: Config, unit: str, section: str, section_start: int, asset: Asset) -> int | None:
    """Relocations the unit's split target object has inside the asset's range (pointers: not
    pure data). None when the target object is not there to check."""
    import json
    from .elf import Elf
    path = cfg.root / "objdiff.json"
    if not path.is_file():
        return None
    stem = unit.rsplit(".", 1)[0]
    entry = next((u for u in json.loads(path.read_text()).get("units", [])
                  if u.get("metadata", {}).get("source_path", "").endswith("/" + unit) or u["name"].endswith("/" + stem)), None)
    target = cfg.root / entry["target_path"] if entry and entry.get("target_path") else None
    if target is None or not target.is_file():
        return None
    elf = Elf.parse(target.read_bytes())
    sec = elf.section(section)
    if sec is None:
        return None
    lo, hi = asset.address - section_start, asset.end - section_start
    return sum(lo <= r.offset < hi for r in elf.relocations(sec.index))


def covering_symbols(cfg: Config, asset: Asset) -> list[str]:
    """Notes on how the range lines up with symbols.txt (one object exactly, or not)."""
    from .dtk_symbols import read_symbols
    inside = [s for s in read_symbols(cfg.root / cfg.platform_options["symbols"])
              if asset.address <= s.address < asset.end]
    exact = [s for s in inside if s.address == asset.address and s.size == asset.size]
    if exact:
        return [f"covers {exact[0].name} exactly"]
    names = ", ".join(f"{s.name} (0x{s.address:08X}, size 0x{s.size:X})" for s in inside[:4])
    return [f"note: the range is not exactly one symbol; symbols inside: {names or 'none'}"]


# --- the command -----------------------------------------------------------------------------

def _units(cfg: Config):
    from .units import read_units_file
    return read_units_file(cfg.root / cfg.platform_options["units"])


def run_add(cfg: Config, args: argparse.Namespace) -> int:
    from .units import render_units_file
    units_path = cfg.root / cfg.platform_options["units"]
    with project_lock(cfg.root):
        units = _units(cfg)
        found = units.find(args.unit)
        if found is None:
            print(f"error: {args.unit} not in {cfg.rel(units_path)} (./dk unit add first)")
            return USAGE
        raw = {"name": args.name, "address": args.address, "size": args.size}
        if args.width != 1:
            raw["width"] = args.width
        try:
            asset = parse_entry(args.unit, raw)
            others = [a for a in declared(units_path.read_text()) if not (a.unit == args.unit and a.name == args.name)]
        except AssetError as e:
            print(f"error: {e}")
            return USAGE
        if any(a.name == asset.name for a in others):
            print(f"error: asset name {asset.name} is already declared by another unit")
            return USAGE
        overlap = [a for a in others if a.address < asset.end and asset.address < a.end]
        if overlap:
            print(f"error: 0x{asset.address:08X}-0x{asset.end:08X} overlaps asset {overlap[0].name} ({overlap[0].unit})")
            return USAGE
        ranges = unit_ranges(cfg, args.unit)
        home = next(((sec, lo) for sec, lo, hi in ranges if lo <= asset.address and asset.end <= hi), None)
        if home is None:
            print(f"error: 0x{asset.address:08X}-0x{asset.end:08X} is not inside one of {args.unit}'s ranges")
            return USAGE
        if home[0] in NO_BYTES:
            print(f"error: {home[0]} holds code or zero-fill, not embedded data")
            return USAGE
        try:
            read_original(cfg).read(asset.address, asset.size)
        except (AssetError, ValueError) as e:
            print(f"error: {e}")
            return USAGE
        relocs = relocations_inside(cfg, args.unit, home[0], home[1], asset)
        if relocs:
            print(f"error: the target object has {relocs} relocation(s) inside the range: it holds pointers, so it "
                  "is not pure data (type it: ./dk gen --data)")
            return MISMATCH
        _, obj = found
        kept = [r for r in obj.get(KEY, []) if r.get("name") != asset.name]
        obj[KEY] = sorted(kept + [raw], key=lambda r: int(r["address"]))
        units_path.write_text(render_units_file(units))
        try:
            written, _ = extract(cfg)
        except AssetError as e:
            print(f"error: {e}")
            return USAGE
    print(f"declared: {args.unit} asset {asset.name} (0x{asset.size:X} bytes at 0x{asset.address:08X}, {home[0]})")
    for line in covering_symbols(cfg, asset):
        print(f"  {line}")
    if relocs is None:
        print("  note: no split target object to check for relocations (./dk build, then add again to check)")
    print(f"  extracted: {', '.join(written) or 'unchanged'}")
    print("  use it as (the element type is the original's; it may be a struct or have an alignment):")
    print(f"    {WIDTHS[asset.width]} NAME[0x{asset.size // asset.width:X}] = {{\n    #include \"{asset.include}\"\n    }};")
    hint(f"check {args.unit} --private")
    return OK


def run_remove(cfg: Config, args: argparse.Namespace) -> int:
    from .units import render_units_file
    units_path = cfg.root / cfg.platform_options["units"]
    with project_lock(cfg.root):
        units = _units(cfg)
        found = units.find(args.unit)
        rows = found[1].get(KEY, []) if found else []
        if not any(r.get("name") == args.name for r in rows):
            print(f"error: {args.unit} declares no asset {args.name}")
            return USAGE
        kept = [r for r in rows if r.get("name") != args.name]
        if kept:
            found[1][KEY] = kept
        else:
            del found[1][KEY]
        units_path.write_text(render_units_file(units))
        try:
            _, removed = extract(cfg)
        except AssetError as e:
            print(f"error: {e}")
            return USAGE
    print(f"removed: {args.unit} asset {args.name}" + (f" ({', '.join(removed)} deleted)" if removed else ""))
    hint(f"check {args.unit} --private")
    return OK


def run_list(cfg: Config, args: argparse.Namespace) -> int:
    try:
        assets = load(cfg)
    except AssetError as e:
        print(f"error: {e}")
        return USAGE
    if not assets:
        print("no assets declared")
        hint("asset add <unit> <name> <address> <size>")
        return OK
    for a in assets:
        there = "extracted" if (out_dir(cfg) / f"{a.name}.inc").is_file() else "not extracted"
        width = f", width {a.width}" if a.width != 1 else ""
        print(f"{a.unit}: #include \"{a.include}\"  0x{a.address:08X} size 0x{a.size:X}{width} ({there})")
    hint("build")
    return OK


def run_extract(cfg: Config, args: argparse.Namespace) -> int:
    try:
        written, removed = extract(cfg)
    except (AssetError, tomllib.TOMLDecodeError) as e:
        print(f"error: {e}")
        return USAGE
    if not args.quiet:
        for rel in written:
            print(f"wrote {rel}")
        for rel in removed:
            print(f"removed {rel} (no longer declared)")
        print(f"{len(load(cfg))} asset(s) up to date")
    return OK


def _number(text: str) -> int:
    return int(text, 0)


def _configure(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("add", help="declare a unit's embedded asset (in its units.toml entry) and extract it")
    a.add_argument("unit", help="source path below src/, e.g. game/gfx/80401000.cpp")
    a.add_argument("name", help="include name: the source writes #include \"assets/<name>.inc\"")
    a.add_argument("address", type=_number, help="start address (0x...)")
    a.add_argument("size", type=_number, help="size in bytes (0x...)")
    a.add_argument("--width", type=int, default=1, choices=sorted(WIDTHS),
                   help="element size: 1 bytes (default), 2 or 4 big-endian values")
    r = sub.add_parser("remove", help="drop a unit's asset declaration")
    r.add_argument("unit")
    r.add_argument("name")
    sub.add_parser("list", help="declared assets, their include lines and whether they are extracted")
    e = sub.add_parser("extract", help="write build/<GAME>/include/assets/*.inc from the original (the build runs this)")
    e.add_argument("--quiet", action="store_true", help="print nothing on success")


def run(cfg: Config, args: argparse.Namespace) -> int:
    if getattr(args, "unit", None):
        args.unit = args.unit.removeprefix("src/")
    return {"add": run_add, "remove": run_remove, "list": run_list, "extract": run_extract}[args.action](cfg, args)


# Not `locked`: ninja runs `asset extract` while `dk build`/`check` hold the project lock.
COMMAND = Command("asset", "matching",
                  "embedded game data a source #includes, extracted from the original at build time (add, remove, list, extract)",
                  run, _configure)
