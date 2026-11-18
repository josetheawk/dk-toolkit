"""One pass over a flat image: layout, entry points, containers, files and names.

`Profile` is what a platform knows about the medium (fill byte, gap size, size notes).
`raw` uses a generic profile; a specialised platform supplies its own through the
`profile(cfg)` hook.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import jffs2
from .arm import ArmVectors, CortexMVectors, find_arm_vectors, find_cortex_m
from .layout import AsciiMagic, Copy, Span, StringHit, entropy, fill_spans, find_ascii_magics, find_copies, find_strings, programmed_islands
from .signatures import Hit, scan


@dataclass(frozen=True)
class Profile:
    fill: tuple[int, ...] = (0x00, 0xFF)   # byte values a gap is made of
    min_gap: int = 4096                    # shortest run that separates two regions
    fill_name: str = "padding"             # what `map` calls the gaps ("erased" on flash)
    out: str = "build/raw"                 # where `carve` writes
    arch: str = "auto"                     # `dis` default when no vector table decides
    notes: tuple[str, ...] = ()            # extra `map` header lines (geometry, part label)


@dataclass(frozen=True)
class Analysis:
    size: int
    profile: Profile
    gaps: tuple[Span, ...]
    islands: tuple[Span, ...]
    arm: tuple[ArmVectors, ...]
    cortex_m: tuple[CortexMVectors, ...]
    hits: tuple[Hit, ...]
    jffs2: tuple[jffs2.Jffs2, ...]
    files: tuple[jffs2.Jffs2File, ...]
    magics: tuple[AsciiMagic, ...]
    copies: tuple[Copy, ...]
    strings: tuple[StringHit, ...]


def analyze(data: bytes, profile: Profile = Profile(), *, string_limit: int = 40) -> Analysis:
    gaps = fill_spans(data, profile.fill, profile.min_gap)
    islands = programmed_islands(len(data), gaps)
    filesystems = jffs2.find_jffs2(data)
    return Analysis(
        size=len(data),
        profile=profile,
        gaps=gaps,
        islands=islands,
        arm=find_arm_vectors(data),
        cortex_m=find_cortex_m(data, islands),
        hits=scan(data, islands),
        jffs2=filesystems,
        files=jffs2.files(data, filesystems),
        magics=find_ascii_magics(data, islands),
        copies=find_copies(data, islands),
        strings=find_strings(data, islands, limit=string_limit) if string_limit else (),
    )


def labels(island: Span, report: Analysis) -> list[str]:
    """What starts inside `island`, in a fixed order."""
    def inside(offset: int) -> bool:
        return island.offset <= offset < island.end

    out: list[str] = []
    if any(inside(t.offset) for t in report.arm):
        out.append("arm-vectors")
    if any(inside(t.offset) for t in report.cortex_m):
        out.append("cortex-m")
    out += sorted({hit.kind for hit in report.hits if inside(hit.offset)})
    if any(inside(fs.offset) for fs in report.jffs2):
        out.append("jffs2")
    out += [f"magic:{m.text}" for m in report.magics if inside(m.offset)]
    return out


def map_lines(report: Analysis, data: bytes, *, name: str, sha1: str, expected_sha1: str | None) -> list[str]:
    p = report.profile
    lines = [f"image: {name}", f"size: {report.size}", *p.notes]
    lines.append(f"sha1: {sha1} ok" if expected_sha1 in (None, sha1) else f"sha1: {sha1} mismatch dk.toml {expected_sha1}")
    fills = " ".join(f"0x{b:02x}" for b in sorted(set(p.fill)))
    lines.append(f"{p.fill_name}: byte {fills}, runs >= {p.min_gap}: {len(report.gaps)}, bytes {sum(s.length for s in report.gaps)}")
    lines.append(f"programmed: {len(report.islands)}")
    for island in report.islands:
        tags = ",".join(labels(island, report))
        ent = entropy(data[island.offset:island.end])
        lines.append(f"  0x{island.offset:08x}-0x{island.end:08x}  len 0x{island.length:08x} ({island.length})"
                     f"  entropy {ent:.2f}{'  ' + tags if tags else ''}")
    if report.arm:
        lines.append("arm-vectors:")
        for t in report.arm:
            reset = f"0x{t.reset_file:08x}" if t.reset_file is not None else "ldr"
            lines.append(f"  at 0x{t.offset:08x}  link 0x{t.link_base:08x}  reset {reset}  handlers "
                         + " ".join(f"0x{h:08x}" for h in t.handlers))
            lines.append(f"    va = link + file_offset - 0x{t.offset:08x}")
    if report.cortex_m:
        lines.append("cortex-m-vectors:")
        lines += [f"  at 0x{t.offset:08x}  sp 0x{t.stack:08x}  reset 0x{t.reset:08x}" for t in report.cortex_m]
    if report.hits:
        lines.append("signatures:")
        for hit in report.hits:
            span = f"body 0x{hit.body:08x}-0x{hit.end:08x}" if hit.end is not None else "end unknown"
            lines.append(f"  at 0x{hit.offset:08x}  {hit.kind:<9} {span}{'  ' + hit.note if hit.note else ''}")
    if report.jffs2:
        lines.append("jffs2:")
        for fs in report.jffs2:
            crc = "header crc ok" if not fs.bad_crc else f"header crc bad on {fs.bad_crc} of {fs.nodes} nodes"
            lines.append(f"  at 0x{fs.offset:08x}-0x{fs.end:08x}  {fs.endian}  nodes {fs.nodes}  dirents {len(fs.dirents)}  {crc}")
            lines += [f"    0x{d.offset:08x}  ino {d.ino}  parent {d.pino}  {d.kind}  {d.name}" for d in fs.dirents]
    if report.files:
        lines.append("jffs2-files:")
        lines += [f"  {f.path}  {len(f.data)} bytes" for f in report.files]
    if report.copies:
        lines.append("copies:")
        lines += [f"  0x{c.offset:08x}  same as 0x{c.same_as:08x}  len 0x{c.length:08x}" for c in report.copies]
    if report.strings:
        lines.append("strings:")
        lines += [f"  0x{s.offset:08x}  {s.text}" for s in report.strings]
    return lines
