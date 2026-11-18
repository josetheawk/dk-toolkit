"""Frozen functions: the one union every reader uses, and the commit guard that keeps them unedited.

A project keeps evaluation sets (functions a model is scored on); a matching agent who solves or edits
one leaks its answer into the repo. The platform names the set files (`frozen_functions(cfg)`, for
gc_mwcc from dk.toml `[platform.frozen]`); this module reads them:

  union(data, sets, released)   {function: set label} of every set, minus the release file's functions
  FrozenMap                     that dict, also looked up by (unit, function): the unit is ignored, so a
                                stale unit name in a set file never hides a function
  guard(cfg, overlay, head)     refusals for a commit (save, accept, commit-tools, rm all land through
                                isolated.commit_overlay) that changes the source text of a frozen function

The release file is {function: {"date", "reason"}}: a released function leaves the union (and the guard);
the set files are never edited, so evaluations stay reproducible and must exclude released functions.
"""

from __future__ import annotations

import json
from pathlib import Path

SOURCE_SUFFIXES = (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".inc", ".inl")


class FrozenMap(dict):
    """{function: set label}; `(unit, function)` keys look up the function alone."""

    @staticmethod
    def _k(key):
        return key[1] if isinstance(key, tuple) else key

    def __contains__(self, key) -> bool:
        return dict.__contains__(self, self._k(key))

    def __getitem__(self, key):
        return dict.__getitem__(self, self._k(key))

    def get(self, key, default=None):
        return dict.get(self, self._k(key), default)


def _names(doc, keys: list[str] | None) -> list[str]:
    """Function names of one set file: `functions` rows (dicts with `function`, or plain names), or the
    named top-level lists (`file.json#dev,final`)."""
    rows: list = []
    if isinstance(doc, list):
        rows = doc
    elif isinstance(doc, dict):
        for k in keys or ["functions"]:
            v = doc.get(k)
            rows += v if isinstance(v, list) else []
    out = []
    for r in rows:
        name = r.get("function") if isinstance(r, dict) else r if isinstance(r, str) else None
        if name:
            out.append(name)
    return out


def set_files(data: Path, sets: dict[str, list[str]], released: Path | None = None) -> list[tuple[str, Path, list | None]]:
    """(label, file, keys) of every set file that exists, in the order given. A label with `*` takes each
    file's stem (`"eval:*" = ["eval/*.json"]`); the release file is never a set."""
    out = []
    for label, specs in sets.items():
        for spec in [specs] if isinstance(specs, str) else specs:
            pattern, _, keys = spec.partition("#")
            for p in sorted(data.glob(pattern)):
                if not p.is_file() or (released is not None and p.resolve() == released.resolve()):
                    continue
                out.append((label.replace("*", p.stem), p, keys.split(",") if keys else None))
    return out


def load_released(path: Path | None) -> dict[str, dict]:
    if path is None or not path.is_file():
        return {}
    doc = json.loads(path.read_text())
    return {k: v for k, v in doc.items() if isinstance(v, dict)} if isinstance(doc, dict) else {}   # a note: a string value


def union(data: Path, sets: dict[str, list[str]], released: Path | None = None) -> FrozenMap:
    """{function: first set label holding it} over every set file, minus the released functions."""
    gone = load_released(released)
    out = FrozenMap()
    for label, path, keys in set_files(data, sets, released):
        try:
            names = _names(json.loads(path.read_text()), keys)
        except (OSError, ValueError):
            continue
        for n in names:
            if n not in gone:
                out.setdefault(n, label)
    return out


# ---- the commit guard ------------------------------------------------------------------------------

def _param_count(head: str, name: str) -> int | None:
    """Parameters in the parenthesis right after the declared name (None: unreadable)."""
    at = head.rfind(name)
    start = head.find("(", at + len(name)) if at >= 0 else -1
    if start < 0:
        return None
    depth, i = 0, start
    while i < len(head):
        depth += head[i] == "("
        depth -= head[i] == ")"
        if depth == 0:
            break
        i += 1
    inner = head[start + 1:i].strip()
    if inner in ("", "void"):
        return 0
    depth, n = 0, 1
    for ch in inner:
        depth += ch in "(<["
        depth -= ch in ")>]"
        n += ch == "," and depth == 0
    return n


def definitions(text: str, wanted: dict[str, list[tuple[str, int | None]]]) -> dict[str, list[str]]:
    """{frozen function: [definition text...]} in one source file. `wanted` maps each frozen function
    to the (qualified source name, parameter count or None) its definition is written with."""
    from .cbody import functions
    by_qual: dict[str, list[tuple[str, int | None]]] = {}
    for fn, forms in wanted.items():
        for qual, n in forms:
            by_qual.setdefault(qual, []).append((fn, n))
    out: dict[str, list[str]] = {}
    if not any(q.rsplit("::", 1)[-1] in text for q in by_qual):
        return out
    try:
        spans = functions(text)
    except Exception:                       # an unreadable file: compared as a whole below
        return {"<unparsed>": [text]}
    for s in spans:
        hits = by_qual.get(s.name) or by_qual.get(s.name.lstrip(":"))
        if not hits:
            continue
        n = _param_count(text[s.head_start:s.open], s.name)
        body = text[s.head_start:s.close + 1]
        for fn, want in hits:
            if want is None or n is None or want == n:
                out.setdefault(fn, []).append(body)
    return out


def changed_frozen(before: dict[str, bytes | None], after: dict[str, bytes | None],
                   wanted: dict[str, list[tuple[str, int | None]]]) -> list[str]:
    """Frozen functions whose definitions differ between two versions of the same files (path -> bytes,
    None = absent). Compared as a sorted list of definition texts over all the files together: a body
    moved byte-identical to another file is no change; an added, removed or edited one is."""
    old: dict[str, list[str]] = {}
    new: dict[str, list[str]] = {}
    for side, files in ((old, before), (new, after)):
        for rel, data in files.items():
            if data is None or not rel.endswith(SOURCE_SUFFIXES):
                continue
            for fn, bodies in definitions(data.decode("utf-8", "replace"), wanted).items():
                side.setdefault(fn, []).extend(bodies)
    return sorted(fn for fn in set(old) | set(new) if fn != "<unparsed>" and sorted(old.get(fn, [])) != sorted(new.get(fn, [])))


def guard(cfg, overlay: dict[str, bytes | None], show) -> list[str]:
    """Refusal lines for an overlay that edits a frozen function (empty: fine). `show(rel)` returns the
    file at HEAD (bytes or None). The platform supplies the frozen union and how each is written."""
    from .. import platforms
    try:
        mod = platforms.load(cfg.platform)
    except Exception:
        return []
    forms = getattr(mod, "frozen_definitions", None)
    if forms is None:
        return []
    paths = [rel for rel in overlay if rel.endswith(SOURCE_SUFFIXES)]
    if not paths:
        return []
    wanted, released_file = forms(cfg)
    if not wanted:
        return []
    before = {rel: show(rel) for rel in paths}
    hit = changed_frozen(before, {rel: overlay[rel] for rel in paths}, wanted)
    return [f"{fn} is frozen (an evaluation set's function): this commit changes its source text. Undo that "
            f"edit; only the owner releases a frozen function, by adding it to {released_file} "
            f'as "{fn}": {{"date": "YYYY-MM-DD", "reason": "..."}}' for fn in hit]
