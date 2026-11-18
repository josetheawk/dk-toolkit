"""Content-keyed caches shared by every worktree of a project: compiled objects and objdiff results.

Search tools (close, llm-try, llm-agent, regs, reorder) compile the same candidate many times:
across rounds, across nights and across agents in separate worktrees. A compile is keyed by
everything that decides its bytes: the compile argv (paths made relative), the source text and the
content of every header it can include. A hit writes the stored object where the compile would
have, so callers can't tell the difference.

Safety:
- Verification paths (save, accept, check) never use it; only search compiles opt in.
- One hit in SPOT_CHECK is compiled anyway and compared byte for byte. A difference turns the
  cache off for the process and is logged to `<store>/MISMATCH.log`.
- `DK_NO_OBJ_MEMO=1` turns it off; `DK_OBJCACHE_STATS=1` prints the hit rate at exit.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

from ...core.config import Config

MAX_AGE_DAYS = 14                     # entries older than this go, checked about once per PRUNE_EVERY saves
MAX_BYTES = int(os.environ.get("DK_OBJCACHE_MB", "2048")) << 20   # then the oldest go until the store fits
PRUNE_EVERY = 2000
SPOT_CHECK = 50                       # one hit in this many is recompiled and compared
HEADER_SUFFIXES = {".h", ".hpp", ".hh", ".inc", ".inl"}
OBJDIFF_ARGS = ("-o", "-", "--format", "json", "-c", "functionRelocDiffs=data_value")

_lock = threading.Lock()
_disabled = bool(os.environ.get("DK_NO_OBJ_MEMO"))
_stats = {"hits": 0, "misses": 0, "checked": 0}
_stat_cache: dict[str, tuple[int, int, str]] = {}     # path -> (mtime_ns, size, sha1)


def main_root(cfg: Config) -> Path:
    """The main checkout (shared by all worktrees); cfg.root outside a git worktree."""
    from ...core.landing import common_dir
    common = common_dir(cfg.root)
    return common.parent if common.name == ".git" else Path(cfg.root)


def store(cfg: Config) -> Path:
    return main_root(cfg) / "build" / "dk" / "objcache"


def stats() -> dict:
    return dict(_stats, disabled=_disabled)


def _report() -> None:
    if _stats["hits"] or _stats["misses"]:
        print(f"objcache: {_stats['hits']} of {_stats['hits'] + _stats['misses']} compiles reused, "
              f"{_stats['checked']} spot-checked" + (" (turned off: a mismatch)" if _disabled else ""), file=sys.stderr)


if os.environ.get("DK_OBJCACHE_STATS"):
    atexit.register(_report)


def _sha_file(path: Path | str, st: os.stat_result | None = None) -> str:
    key = str(path)
    st = st or os.stat(key)
    hit = _stat_cache.get(key)
    if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
        return hit[2]
    with open(key, "rb") as f:
        sha = hashlib.sha1(f.read()).hexdigest()
    _stat_cache[key] = (st.st_mtime_ns, st.st_size, sha)
    return sha


FINGERPRINT_TTL = 5.0                 # seconds a header scan is reused (a search edits sources, not headers)
_fingerprints: dict = {}


def _scan(base: str, rel: str, out: list) -> None:
    try:
        entries = sorted(os.scandir(base), key=lambda e: e.name)
    except OSError:
        return
    for e in entries:
        if e.is_dir():
            _scan(e.path, f"{rel}{e.name}/", out)
        elif os.path.splitext(e.name)[1] in HEADER_SUFFIXES:
            try:
                out.append(f"{rel}{e.name}\0{_sha_file(e.path, e.stat())}\n")
            except OSError:
                continue


def header_fingerprint(root: Path, dirs: list[str]) -> str:
    """One hash over the content of every header under the include dirs and src/ (paths relative)."""
    key = (str(root), tuple(sorted(set(dirs) | {"src"})))
    now = time.monotonic()
    with _lock:
        hit = _fingerprints.get(key)
    if hit and now - hit[0] < FINGERPRINT_TTL:
        return hit[1]
    rows: list[str] = []
    for d in key[1]:
        _scan(os.path.join(key[0], d), f"{d}/", rows)
    fp = hashlib.sha1("".join(rows).encode()).hexdigest()
    with _lock:
        _fingerprints[key] = (now, fp)
    return fp


def compile_key(cfg: Config, argv: list[str], out_dir: Path, source_text: str) -> str:
    root = str(Path(cfg.root).resolve())
    norm, includes = [], []
    for i, a in enumerate(argv):
        if a == str(out_dir):
            a = "<OUT>"
        a = a.replace(root, "<ROOT>")
        norm.append(a)
        if i and argv[i - 1] in ("-i", "-I", "-ir"):
            includes.append(a)
    tool_ids = [_sha_file(Path(a)) for a in argv[:3] if os.path.isabs(a) and Path(a).is_file()]
    text = "\0".join(norm) + "\0" + "\0".join(tool_ids) + "\0" + header_fingerprint(Path(cfg.root), includes)
    return hashlib.sha1((text + "\0" + source_text).encode()).hexdigest()


def _entry(cfg: Config, key: str) -> Path:
    return store(cfg) / key[:2] / key


def lookup(cfg: Config, key: str, out_dir: Path) -> tuple[bool, str] | None:
    """A stored compile, its objects copied into out_dir; None on a miss or when the cache is off."""
    if _disabled:
        return None
    entry = _entry(cfg, key)
    try:
        meta = json.loads((entry / "meta.json").read_text())
        for name in meta["objects"]:
            shutil.copyfile(entry / name, out_dir / name)
    except (OSError, ValueError, KeyError):
        with _lock:
            _stats["misses"] += 1
        return None
    with _lock:
        _stats["hits"] += 1
    return bool(meta["ok"]), meta["output"]


def save(cfg: Config, key: str, out_dir: Path, ok: bool, output: str) -> None:
    if _disabled:
        return
    entry = _entry(cfg, key)
    tmp = entry.parent / f".{key}.{os.getpid()}.{threading.get_ident()}"
    try:
        tmp.mkdir(parents=True)
        objs = sorted(p.name for p in out_dir.glob("*.o")) if ok else []
        for name in objs:
            shutil.copyfile(out_dir / name, tmp / name)
        (tmp / "meta.json").write_text(json.dumps({"ok": ok, "output": output, "objects": objs}))
        os.rename(tmp, entry)                     # atomic; a concurrent writer of the same key wins
    except OSError:
        pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if random.randrange(PRUNE_EVERY) == 0:
        prune(cfg)


def _size(e: Path) -> int:
    if e.is_dir():
        return sum(f.stat().st_size for f in e.iterdir() if f.is_file())
    return e.stat().st_size


def prune(cfg: Config, max_age_days: float = MAX_AGE_DAYS, max_bytes: int | None = None) -> int:
    """Delete entries (objects and function pairs) not written for max_age_days, then the oldest until the store
    is under max_bytes (DK_OBJCACHE_MB, default 2048); how many went."""
    max_bytes = MAX_BYTES if max_bytes is None else max_bytes
    cutoff, gone, keep = time.time() - max_age_days * 86400, 0, []
    for top in (store(cfg), store(cfg) / "pairs"):
        for shard in top.glob("??"):
            for e in shard.iterdir():
                try:
                    st = e.stat()
                    if st.st_mtime < cutoff:
                        shutil.rmtree(e) if e.is_dir() else e.unlink()
                        gone += 1
                    else:
                        keep.append((st.st_mtime, _size(e), e))
                except OSError:
                    continue
    total = sum(s for _, s, _ in keep)
    for _, size, e in sorted(keep, key=lambda k: k[0]):
        if total <= max_bytes:
            break
        try:
            shutil.rmtree(e) if e.is_dir() else e.unlink()
            total -= size
            gone += 1
        except OSError:
            continue
    return gone


def should_check() -> bool:
    return random.randrange(SPOT_CHECK) == 0


def verify(cfg: Config, key: str, out_dir: Path, ok: bool, output: str) -> None:
    """After a spot-check compile: compare it with the stored entry; on a difference turn the cache off."""
    global _disabled
    entry = _entry(cfg, key)
    with _lock:
        _stats["checked"] += 1
    try:
        meta = json.loads((entry / "meta.json").read_text())
        same = bool(meta["ok"]) == ok and all((entry / n).read_bytes() == (out_dir / n).read_bytes()
                                              for n in meta["objects"])
    except (OSError, ValueError, KeyError):
        return
    if not same:
        _disabled = True
        shutil.rmtree(entry, ignore_errors=True)
        msg = f"objcache: stored compile {key} differs from a fresh one; cache off for this process"
        print(msg, file=sys.stderr)
        try:
            with open(store(cfg) / "MISMATCH.log", "a") as f:
                f.write(msg + "\n")
        except OSError:
            pass


# ---- objdiff ------------------------------------------------------------------------------------------

_DIFFS: OrderedDict = OrderedDict()
_DIFFS_MAX = 64


def objdiff_json(cfg: Config, target: Path | str, ours: Path | str) -> tuple[dict | None, str]:
    """objdiff-cli's JSON diff of two objects (data_value relocations), or (None, error).
    Memoised in-process by the two objects' content."""
    root = Path(cfg.root)
    target, ours = root / target, root / ours
    objdiff = root / "build" / "tools" / "objdiff-cli"
    try:
        key = (hashlib.sha1(target.read_bytes()).hexdigest(), hashlib.sha1(ours.read_bytes()).hexdigest())
    except OSError as e:
        return None, str(e)
    with _lock:
        if key in _DIFFS:
            _DIFFS.move_to_end(key)
            return _DIFFS[key], ""
    proc = subprocess.run([str(objdiff), "diff", "-1", str(target), "-2", str(ours), *OBJDIFF_ARGS],
                          cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, proc.stderr.strip()
    doc = json.loads(proc.stdout)
    with _lock:
        _DIFFS[key] = doc
        while len(_DIFFS) > _DIFFS_MAX:
            _DIFFS.popitem(last=False)
    return doc, ""


def symbol_pair(doc: dict, symbol: str) -> tuple[dict, dict] | None:
    """(target, ours) entries with instructions for one function of an objdiff document."""
    left = next((s for s in doc["left"].get("symbols", []) if s["name"] == symbol and "instructions" in s), None)
    right = next((s for s in doc.get("right", {}).get("symbols", [])
                  if s["name"] == symbol and "instructions" in s), None)
    return (left, right) if left and right else None


def function_pair(cfg: Config, target: Path | str, ours: Path | str, symbol: str) -> tuple[dict, dict] | None:
    """symbol_pair of objdiff_json, kept on disk by both objects' content, so a candidate compiled
    again (another round, night or worktree) needs no objdiff run."""
    root = Path(cfg.root)
    try:
        key = hashlib.sha1(b"\0".join([(root / target).read_bytes(), (root / ours).read_bytes(),
                                        symbol.encode(), " ".join(OBJDIFF_ARGS).encode()])).hexdigest()
    except OSError:
        return None
    path = store(cfg) / "pairs" / key[:2] / f"{key}.json"
    if not _disabled:
        try:
            pair = json.loads(path.read_text())
            return (pair[0], pair[1]) if pair else None
        except (OSError, ValueError):
            pass
    doc, _ = objdiff_json(cfg, target, ours)
    if doc is None:
        return None
    pair = symbol_pair(doc, symbol)
    if not _disabled:
        tmp = path.with_name(f".{key}.{os.getpid()}.{threading.get_ident()}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(pair))
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
    return pair

