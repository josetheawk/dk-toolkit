"""Batched candidate compiles: many versions of one function in one translation unit.

A search compiles hundreds of versions of one function, each a whole unit (compiler start-up, headers, every
other function). Here up to K versions go into one copy of the unit: each version's definition, renamed
`<name>_dkb<k>`, is placed right after the real definition, so every copy sees the same headers and earlier
definitions as the real one. One compile then gives each version's code. A control copy (the unchanged
definition, renamed) must come out identical to the real function; when it does not (a rename that breaks a
call cycle through an inlined function, say) the batch answers nothing and the caller compiles normally.

Only safe uses: the code hash of each version (same hash = same instructions and relocations), which lets a
search skip versions whose code a sibling or the parent already has. A version's score always comes from its
own normal compile, and nothing from a batch is ever kept. Only free functions with one definition qualify
(a member needs its class declaration changed); a version that edits anything outside the function body
(a new helper, a declaration above) is not batched."""

from __future__ import annotations

import hashlib
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...core import cbody
from .elf import Elf

SUFFIX = "_dkb"


def rename(text: str, name: str, new: str) -> str:
    """Identifier tokens `name` renamed to `new` (strings and comments untouched)."""
    out, last = [], 0
    for tk in cbody.tokens(text):
        if tk.kind == "id" and tk.text == name:
            out.append(text[last:tk.start])
            out.append(new)
            last = tk.start + len(tk.text)
    out.append(text[last:])
    return "".join(out)


def code_hash(fn, own: str) -> str:
    """Instructions plus (offset, type, symbol) of each relocation; calls to the copy itself count as `self`."""
    rel = [(r.offset, r.type, "self" if r.symbol == own else r.symbol) for r in fn.relocs]
    return hashlib.sha1(fn.code + repr(rel).encode()).hexdigest()


class Batcher:
    """Code hashes of versions of `symbol` (whole-source texts of the unit), from batched compiles."""

    def __init__(self, compile_text, base: str, symbol: str, c_source: bool, size: int | None = None,
                 limit: int = 64, jobs: int = 1, max_size: int = 2048):
        self.compile_text, self.symbol, self.c, self.limit, self.jobs = compile_text, symbol, c_source, limit, jobs
        self.base = base
        self.span = None
        self.reason = ""
        self.batches = 0
        self.disabled = False
        if size is not None and size > max_size:
            self.reason = f"function is {size} bytes (batching pays below {max_size})"
            return
        name = symbol.split("__")[0] if "__" in symbol and not symbol.startswith("__") else symbol
        if symbol.startswith("__") or not re.fullmatch(r"[A-Za-z_]\w*", name):
            self.reason = "not a free function"
            return
        spans = [f for f in cbody.functions(base) if f.name == name]
        if len(spans) != 1:
            self.reason = "not one definition by that name"
            return
        head = base[spans[0].head_start:spans[0].open]
        if re.search(r"\b(static|inline|template)\b", head) or "::" in head.split("(")[0]:
            self.reason = "static, inline, template or member definition"
            return
        self.span, self.name = spans[0], name
        self.rest = symbol[len(name):]
        self.wrap_c = not c_source and symbol == name           # C linkage in a C++ unit: keep it for the copies

    @property
    def ready(self) -> bool:
        return self.span is not None and not self.disabled

    def body_of(self, text: str) -> str | None:
        """The function's definition in `text` when the version differs from the base only inside it."""
        s = self.span
        tail = len(self.base) - (s.close + 1)
        if not text.startswith(self.base[:s.head_start]) or (tail and not text.endswith(self.base[-tail:])):
            return None
        body = text[s.head_start:len(text) - tail]
        return body if body.count("{") == body.count("}") else None

    def _copy(self, body: str, k: int) -> str:
        new = rename(body, self.name, f"{self.name}{SUFFIX}{k}")
        return f'extern "C" {{ {new} }}' if self.wrap_c else new

    def _one(self, bodies: list[str], slot: int) -> list[str | None] | None:
        s = self.span
        copies = [self._copy(self.base[s.head_start:s.close + 1], 0)] + [self._copy(b, k + 1) for k, b in enumerate(bodies)]
        text = self.base[:s.close + 1] + "\n" + "\n".join(copies) + "\n" + self.base[s.close + 1:]
        obj = self.compile_text(text, f"b{slot}")
        if obj is None:
            return None
        fns = Elf.parse(obj).functions()
        real = fns.get(self.symbol)
        ctrl = fns.get(f"{self.name}{SUFFIX}0{self.rest}")
        if real is None or ctrl is None or code_hash(real, self.symbol) != code_hash(ctrl, f"{self.name}{SUFFIX}0{self.rest}"):
            self.disabled = True                 # the copies do not compile like the real function: stop batching
            self.reason = "the control copy differs from the real function"
            return None
        out = []
        for k in range(1, len(bodies) + 1):
            own = f"{self.name}{SUFFIX}{k}{self.rest}"
            fn = fns.get(own)
            out.append(code_hash(fn, own) if fn is not None else None)
        return out

    def hashes(self, texts: list[str]) -> list[str | None]:
        """Per text: its function's code hash, or None (not batchable, or the batch failed)."""
        out: list[str | None] = [None] * len(texts)
        if not self.ready:
            return out
        todo = [(i, b) for i, b in ((i, self.body_of(t)) for i, t in enumerate(texts)) if b is not None]
        if not todo:
            return out
        n = max(1, min(self.limit, -(-len(todo) // max(1, self.jobs))))
        chunks = [todo[i:i + n] for i in range(0, len(todo), n)]
        with ThreadPoolExecutor(max_workers=max(1, self.jobs)) as pool:
            results = list(pool.map(lambda kc: self._one([b for _, b in kc[1]], kc[0]), enumerate(chunks)))
        self.batches += len(chunks)
        for chunk, res in zip(chunks, results):
            if res is None:
                continue
            for (i, _b), h in zip(chunk, res):
                out[i] = h
        return out


def compiler_for(cfg, unit: dict, work: Path):
    """compile_text(text, slot) -> object bytes or None: a private compile of the unit with `text` as its source."""
    from .reorder import compile_variant

    def go(text: str, slot: str) -> bytes | None:
        d = Path(work) / slot
        try:
            obj, _err = compile_variant(cfg, unit, text, d)
            return obj.read_bytes() if obj is not None else None
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return go
