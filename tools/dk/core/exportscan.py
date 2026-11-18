"""The pre-export scan: nothing project-specific leaves for the public toolkit repo.

`dk toolkit export` copies this project's owned files (dk, tools/dk, tests, playbook) into the shared
toolkit repo, which is public. Three checks run over exactly those files first, and any hit refuses
the export (`dk toolkit export --check` runs them alone):

  names      words that name the game or the project: dk.toml [toolkit] export_banned, plus
             [docs] banned_in_playbook, [project] game_id and name, and the [docs] project folder
             (matched case-insensitively, as substrings)
  addresses  8-hex-digit tokens equal to a symbol address of the original binary (the [platform]
             symbols file): a real address in a fixture or comment says which game it came from
  code       3 consecutive 32-bit words that also follow each other in the original binary
             (dk.toml [target] original; byte order [toolkit] export_endian, default "big"): copied
             asm or data, however it is written (`/* addr word  AA BB CC DD */` listings, `AA BB CC DD`
             byte runs, 0x12345678 or bare 12345678 tokens on the same or the next line)

A code triple counts only when at least two of its words have 3 or more non-zero bytes (instruction-like):
small constants and float literals (0x20, 0x3F800000) occur in any binary. Runs of one repeated word are ignored. A hit that is not a leak goes into dk.toml
[toolkit] export_allow (exact strings: the word, the address or the three words as hex, space-separated).
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import Config


@dataclass(frozen=True)
class Hit:
    kind: str           # names | addresses | code
    path: str
    line: int
    what: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}: {self.what} :: {self.text[:100]}"


def _toolkit_table(cfg: Config) -> dict:
    try:
        with (cfg.root / "dk.toml").open("rb") as f:
            return dict(tomllib.load(f).get("toolkit", {}))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def banned_words(cfg: Config) -> list[str]:
    t = _toolkit_table(cfg)
    words = [*t.get("export_banned", []), *cfg.docs.banned_in_playbook, cfg.game_id, cfg.name]
    try:
        words.append(str(cfg.docs.project.relative_to(cfg.root)))
    except ValueError:
        pass
    out = []
    for w in words:
        w = str(w).strip()
        if len(w) >= 3 and not w.startswith("<") and w.lower() not in {x.lower() for x in out}:
            out.append(w)
    return out


_ADDR_IN_SYMBOLS = re.compile(r"0x([0-9A-Fa-f]{8})\b")


def real_addresses(path: Path) -> set[str]:
    """Upper-case hex of every address the symbols file names (the first 0x-word of each line)."""
    out: set[str] = set()
    if not path.is_file():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        m = _ADDR_IN_SYMBOLS.search(line)
        if m:
            out.add(m.group(1).upper())
    return out


def word_index(binary: Path, endian: str = "big") -> set[tuple[int, int, int]]:
    """Every 3 consecutive 4-byte-aligned words of the binary, runs of one repeated word left out."""
    if not binary.is_file():
        return set()
    data = binary.read_bytes()
    n = len(data) // 4
    words = [int.from_bytes(data[i * 4:i * 4 + 4], endian) for i in range(n)]
    return {(a, b, c) for a, b, c in zip(words, words[1:], words[2:]) if not a == b == c and informative(a, b, c)}


def _dense(w: int) -> bool:
    return sum(1 for k in range(4) if (w >> (8 * k)) & 0xFF) >= 3


def informative(a: int, b: int, c: int) -> bool:
    """Instruction-like: two of the three words have 3+ non-zero bytes."""
    return _dense(a) + _dense(b) + _dense(c) >= 2


_LISTING = re.compile(r"/\*\s*[0-9A-Fa-f]{8}\s+[0-9A-Fa-f]{8}\s+((?:[0-9A-Fa-f]{2}\s){3}[0-9A-Fa-f]{2})\s*\*/")
_TOKEN = re.compile(r"(?<![0-9A-Za-z_])((?:[0-9A-Fa-f]{2} ){3}[0-9A-Fa-f]{2})(?![0-9A-Za-z_])"
                    r"|(?<![0-9A-Za-z_])0x([0-9A-Fa-f]{8})(?![0-9A-Za-z_])"
                    r"|(?<![0-9A-Za-z_])([0-9A-Fa-f]{8})(?![0-9A-Za-z_])")
_ADDR = re.compile(r"(?<![0-9A-Za-z_])(?:0x)?(8[0-9A-Fa-f]{7})(?![0-9A-Za-z_])")


def words_of(text: str) -> list[tuple[int, int]]:
    """(word, line) for each 32-bit word written in the text, in order."""
    out: list[tuple[int, int]] = []
    for no, line in enumerate(text.splitlines(), 1):
        listing = _LISTING.findall(line)
        if listing:
            out.extend((int(x.replace(" ", ""), 16), no) for x in listing)
            continue
        for m in _TOKEN.finditer(line):
            out.append((int((m.group(1) or m.group(2) or m.group(3)).replace(" ", ""), 16), no))
    return out


def scan_files(files: dict[str, bytes], words: list[str], addresses: set[str],
               index: set[tuple[int, int, int]], allow: set[str] = frozenset()) -> list[Hit]:
    allow_l = {a.lower() for a in allow}
    pattern = re.compile("|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)), re.I) if words else None
    hits: list[Hit] = []
    for rel, data in sorted(files.items()):
        text = data.decode("utf-8", errors="ignore")
        lines = text.splitlines()
        if pattern is not None and rel.lower() not in allow_l:
            for no, line in enumerate(lines, 1):
                for m in pattern.finditer(line):
                    if m.group(0).lower() not in allow_l:
                        hits.append(Hit("names", rel, no, m.group(0), line.strip()))
            for m in pattern.finditer(rel):
                hits.append(Hit("names", rel, 0, m.group(0), "(the file's path)"))
        if addresses:
            for no, line in enumerate(lines, 1):
                for m in _ADDR.finditer(line):
                    a = m.group(1).upper()
                    if a in addresses and a.lower() not in allow_l:
                        hits.append(Hit("addresses", rel, no, a, line.strip()))
        if index:
            seq = words_of(text)
            for (a, la), (b, lb), (c, lc) in zip(seq, seq[1:], seq[2:]):
                if lb - la > 1 or lc - lb > 1 or (a, b, c) not in index:
                    continue
                key = f"{a:08X} {b:08X} {c:08X}"
                if key.lower() not in allow_l:
                    hits.append(Hit("code", rel, la, key, lines[la - 1].strip() if la <= len(lines) else ""))
    return hits


def scan(cfg: Config, files: dict[str, bytes]) -> tuple[list[Hit], dict]:
    """Hits in `files` (relative path -> bytes) and what was scanned for."""
    t = _toolkit_table(cfg)
    words = banned_words(cfg)
    sym = cfg.platform_options.get("symbols")
    addresses = real_addresses(cfg.root / sym) if sym else set()
    index = word_index(cfg.original, str(t.get("export_endian", "big")))
    allow = {str(a) for a in t.get("export_allow", [])}
    info = {"files": len(files), "words": words, "addresses": len(addresses), "code_triples": len(index)}
    return scan_files(files, words, addresses, index, allow), info
