"""Where two builds of an assembly differ: per-method IL compared by name, so token renumbering is not a difference."""

from __future__ import annotations

from pathlib import Path

from .assembly import Assembly
from .il import decode
from .listing import render
from .pe import FormatError


def method_texts(asm: Assembly) -> dict[str, list[str]]:
    """{method signature: rendered instructions without offsets}; duplicate signatures get #n."""
    out: dict[str, list[str]] = {}
    for method in range(1, len(asm.md.rows("MethodDef")) + 1):
        name = asm.method_signature(method)
        while name in out:
            name += "#"
        try:
            body = asm.body(method)
            out[name] = [render(asm, i).split(": ", 1)[1] for i in decode(body.code)] if body else []
        except FormatError as e:
            out[name] = [f"<{e}>"]
    return out


def first_diff(cfg, built: Path, original: Path, limit: int) -> list[str]:
    try:
        ours, target = method_texts(Assembly.load(built)), method_texts(Assembly.load(original))
    except (OSError, FormatError) as e:
        return [f"cannot compare as assemblies: {e}"]
    lines = [f"missing: {name}" for name in sorted(set(target) - set(ours))]
    lines += [f"extra: {name}" for name in sorted(set(ours) - set(target))]
    for name in sorted(set(ours) & set(target)):
        a, b = ours[name], target[name]
        if a != b:
            at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            ours_at = a[at] if at < len(a) else "<end>"
            target_at = b[at] if at < len(b) else "<end>"
            lines.append(f"differs: {name} at instruction {at}: ours {ours_at!r}, target {target_at!r}")
    if not lines:
        lines.append("IL identical in every method (the difference is in headers, metadata layout or resources)")
    return lines[:limit] + ([f"... {len(lines) - limit} more"] if len(lines) > limit else [])
