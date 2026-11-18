"""Prove one compiled function byte-exact against the original binary without linking its unit.

Each relocation of our function is resolved the way the final link will resolve it once the
unit is complete: a name the original also has (symbols.txt) goes to that address, a name only
our object has (`@123` literals, section symbols) to the unit's split address plus our offset,
and the relocated instruction is compared with the original's bytes at the function's address.

A pooled literal (`@N`: a float constant, a string) sits where the whole unit's pool puts it, which in an
unfinished unit (functions still missing or in another order) is not yet the original's offset. When an
instruction differs only because of such a literal, it is accepted if the original instruction addresses a
place in the unit's own section that holds the same bytes as our literal: the code is the same, only the
pool's layout is the unit's open business. A literal of another value still differs.
"""

from __future__ import annotations

from dataclasses import dataclass

STB_LOCAL = 0
SMALL_DATA = (".sdata", ".sbss")
SMALL_DATA2 = (".sdata2", ".sbss2")


@dataclass
class Proof:
    ok: bool
    detail: str
    judged: bool = True       # False: nothing to prove here (a weak copy the link discards)


def sda_bases(read, init_registers: int) -> tuple[int, int] | None:
    """(_SDA_BASE_, _SDA2_BASE_) from `lis rN, hi; ori rN, rN, lo` for r13 and r2 in the
    runtime's register setup."""
    found: dict[int, int] = {}
    words = [int.from_bytes(read(init_registers + 4 * i, 4), "big") for i in range(16)]
    for a, b in zip(words, words[1:]):
        if a >> 26 == 15 and (a >> 16) & 31 == 0 and b >> 26 == 24:          # lis rD, hi; ori rA, rS, lo
            d = (a >> 21) & 31
            if (b >> 21) & 31 == d and (b >> 16) & 31 == d:
                found[d] = ((a & 0xFFFF) << 16) | (b & 0xFFFF)
    return (found[13], found[2]) if 13 in found and 2 in found else None


def link(function, address: int, ours_local: set[str], placed: dict[str, int], names: dict[str, tuple[int, str]],
         sda: tuple[int, int]) -> tuple[bytes | None, str]:
    """Our function's bytes as the final link would write them, or (None, why not).

    `placed`: our section -> the unit's split start for it; `names`: symbols.txt name ->
    (address, section); `ours_local`: names local to our object (never resolved by name)."""
    code = bytearray(function.code)
    for r in function.relocs:
        target, section, why = _target(r, ours_local, placed, names)
        if why:
            return None, why
        why = _apply(code, r, target, section, address, sda)
        if why:
            return None, why
    return bytes(code), ""


def _target(r, ours_local: set[str], placed: dict[str, int], names: dict[str, tuple[int, str]]):
    """(address, section, "") a relocation resolves to at the final link, or (0, "", why not)."""
    if r.symbol in names and r.symbol not in ours_local:
        target, section = names[r.symbol]
    elif r.target_section and r.target_section in placed:
        target, section = placed[r.target_section] + r.target_value, r.target_section
    elif r.target_section:
        return 0, "", f"+0x{r.offset:X}: {r.symbol} is in {r.target_section}, which the unit's split lacks"
    else:
        return 0, "", f"+0x{r.offset:X}: {r.symbol} is undefined at link (not in symbols.txt)"
    return target, section, ""


def _apply(code: bytearray, r, target: int, section: str, address: int, sda: tuple[int, int]) -> str:
    """Write one relocation resolved to `target` into `code`; "" or why it cannot be modelled."""
    value = (target + r.addend) & 0xFFFFFFFF
    at = r.offset
    if r.type == 1:                                    # ADDR32
        code[at:at + 4] = value.to_bytes(4, "big")
    elif r.type in (4, 5, 6):                          # ADDR16_LO / HI / HA
        half = value & 0xFFFF if r.type == 4 else (value >> 16 if r.type == 5 else ((value + 0x8000) >> 16) & 0xFFFF)
        code[at:at + 2] = half.to_bytes(2, "big")
    elif r.type == 10:                                 # REL24
        word = int.from_bytes(code[at:at + 4], "big")
        word = (word & ~0x03FFFFFC) | ((value - (address + at)) & 0x03FFFFFC)
        code[at:at + 4] = (word & 0xFFFFFFFF).to_bytes(4, "big")
    elif r.type == 109:                                # EMB_SDA21
        if section in SMALL_DATA:
            reg, base = 13, sda[0]
        elif section in SMALL_DATA2:
            reg, base = 2, sda[1]
        else:
            return f"+0x{at:X}: SDA21 relocation to {r.symbol} outside the small-data sections"
        word = int.from_bytes(code[at:at + 4], "big")
        word = (word & ~0x1FFFFF) | (reg << 16) | ((value - base) & 0xFFFF)
        code[at:at + 4] = word.to_bytes(4, "big")
    else:
        return f"+0x{at:X}: relocation type {r.type} is not modelled"
    return ""


def place_literals(function, code: bytes, original: bytes, address: int, literals, sda) -> tuple[bytes, list[str]]:
    """`code` with each instruction that differs only at a pooled-literal relocation resolved instead to a
    place where the original holds that literal's bytes, when that gives the original instruction.
    `literals(reloc)` -> [(address, section)...]: those places (the unit's own section, same bytes).
    Returns the code and the literals placed so."""
    out, moved = bytearray(code), []
    for r in function.relocs:
        at = r.offset & ~3
        if out[at:at + 4] == original[at:at + 4] or not r.symbol.startswith("@"):
            continue
        for place, section in literals(r) or []:
            trial = bytearray(out)
            if _apply(trial, r, place, section, address, sda) == "" and trial[at:at + 4] == original[at:at + 4]:
                out = trial
                if r.symbol not in moved:
                    moved.append(r.symbol)
                break
    return bytes(out), moved


def prove(function, address: int, ours_local: set[str], placed: dict[str, int],
          names: dict[str, tuple[int, str]], sda: tuple[int, int], read, literals=None) -> Proof:
    code, why = link(function, address, ours_local, placed, names, sda)
    if code is None:
        return Proof(False, f"cannot link: {why}")
    original = read(address, len(code))
    moved: list[str] = []
    if code != original and literals is not None:
        code, moved = place_literals(function, code, original, address, literals, sda)
    if code == original:
        note = (f"; pooled literal(s) {', '.join(moved)} sit elsewhere in our unit's pool than in the original "
                f"(same bytes at the original's address): the unit's literal order is still open") if moved else ""
        return Proof(True, f"links byte-identical to the original at 0x{address:08X} ({len(code)} bytes){note}")
    first = next(i for i in range(0, len(code), 4) if code[i:i + 4] != original[i:i + 4])
    near = [r for r in function.relocs if first <= r.offset < first + 4]
    cause = f" (relocation to {near[0].symbol})" if near else ""
    return Proof(False, f"differs at 0x{address + first:08X} (+0x{first:X}): ours {code[first:first + 4].hex()}, "
                        f"original {original[first:first + 4].hex()}{cause}")


def function_proof(cfg, unit: dict, obj_path, name: str) -> Proof:
    """Hand-link `name` from the compiled object at `obj_path` and compare it with the original."""
    from .dol import Dol
    from .dtk_symbols import read_symbols, read_units
    from .elf import Elf
    symbols = read_symbols(cfg.root / cfg.platform_options["symbols"])
    names = {s.name: (s.address, s.section) for s in symbols}
    if name not in names:
        return Proof(False, f"cannot link: {name} is not in symbols.txt (rename it first)")
    if "__init_registers" not in names:
        return Proof(False, "cannot link: no __init_registers in symbols.txt to read the small-data bases from")
    source = unit["metadata"]["source_path"].removeprefix(cfg.platform_options.get("src", "src") + "/")
    split = next((u for u in read_units(cfg.root / cfg.platform_options["splits"]) if u.name == source), None)
    if split is None:
        return Proof(False, f"cannot link: {source} is not in splits.txt")
    placed = {sec: start for sec, start, _ in split.splits}
    text = [(a, z) for sec, a, z in split.splits if sec == names[name][1]]
    elf = Elf.parse(obj_path.read_bytes())
    function = elf.functions().get(name)
    if function is None:
        return Proof(False, f"cannot link: our object does not define {name}")
    if not any(a <= names[name][0] < z for a, z in text):
        return Proof(False, f"{name} is a copy the link discards: the original keeps the one at "
                            f"0x{names[name][0]:08X}, outside this unit", judged=False)
    dol = Dol.parse(cfg.original.read_bytes())
    sda = sda_bases(dol.read, names["__init_registers"][0])
    if sda is None:
        return Proof(False, "cannot link: the small-data bases were not found in __init_registers")
    local = {x.name for x in elf.symbols if x.bind == STB_LOCAL and x.name}
    ranges = [(sec, a, z) for sec, a, z in split.splits]
    return prove(function, names[name][0], local, placed, names, sda, dol.read,
                 literals=lambda r: literal_places(elf, r, ranges, dol.read, local))


def literal_places(elf, r, ranges: list[tuple[str, int, int]], read, local: set[str]) -> list[tuple[int, str]]:
    """Addresses in the unit's own split of r's section where the original holds the bytes of our pooled
    literal r points at (its symbol's size; a string to its terminator)."""
    if r.symbol not in local or not r.symbol.startswith("@") or not r.target_section:
        return []
    sec = elf.section(r.target_section)
    sym = next((x for x in elf.symbols if x.name == r.symbol), None)
    if sec is None or sym is None or not sec.data:
        return []
    start = r.target_value
    end = start + sym.size if sym.size else (sec.data.find(b"\0", start) + 1 or start)
    lit = bytes(sec.data[start:end])
    if not lit:
        return []
    out = []
    for section, a, z in ranges:
        if section != r.target_section:
            continue
        blob = read(a, z - a)
        i = blob.find(lit)
        while i >= 0:
            out.append((a + i, section))
            i = blob.find(lit, i + 1)
    return out
