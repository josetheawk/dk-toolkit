"""Usage log: every `./dk` run appends one JSON line, so the next toolkit change is chosen from evidence.

Where: $DK_USAGE_LOG (a path, or `off`), else dk.toml `[usage] log` (relative to the main checkout),
else `dk-usage.jsonl` beside the main checkout. One line per run, written after the command finishes
(one O_APPEND write, so concurrent agents never interleave); any logging error is swallowed.

A line: time, cmd, sub (a subcommand), flags (option names used), args (paths shortened, free text
replaced by its length, never source text), worktree, exit (0 ok, 1 mismatch, 2 usage, 3 build error;
null on a crash), error (the exception type on a crash), msg (the last error-like output line on a
failure), secs, and the outcome fields the command noted (`note(...)`: unit, function, score before
and after, refused_by, compiles, best...). The platform adds fields read from the parsed arguments
(`usage_fields(cfg, args)`) and names its search tools and landing commands for the report
(`USAGE_SEARCH`, `USAGE_LANDING`).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

OFF = ("off", "0", "no", "false", "")
DEFAULT_NAME = "dk-usage.jsonl"
TEXT_OPTIONS = ("-m", "--message", "--note", "--trailer")   # their values are free text: only the length is kept
MAX_ARG = 80
FAILURE_LINE = re.compile(r"^\s*(error|REJECTED|REFUSED|BUILD ERROR|usage:|\./dk: error|note: not committed|MISMATCH)")

_state: dict[str, Any] = {}


def begin(cfg, command: str, sub: str, args) -> None:
    """Called by the CLI once the arguments parse."""
    _state.update(cfg=cfg, cmd=command, sub=sub, args=args)


def note(**fields: Any) -> None:
    """A command's outcome fields for this run's log line (later calls add to and override earlier ones)."""
    _state.setdefault("fields", {}).update(fields)


def main_checkout(root: Path) -> Path:
    """The main checkout of `root` (itself, or the checkout a git worktree belongs to). No git call:
    a worktree's `.git` file names `<main>/.git/worktrees/<name>`."""
    dot = Path(root) / ".git"
    try:
        if dot.is_file():
            target = Path(dot.read_text().split("gitdir:", 1)[1].strip())
            if target.parent.name == "worktrees" and target.parent.parent.name == ".git":
                return target.parent.parent.parent
    except (OSError, IndexError):
        pass
    return Path(root)


def worktree_name(root: Path) -> str:
    root = Path(root)
    return "main" if main_checkout(root).resolve() == root.resolve() else root.name


def log_path(cfg=None, root: Path | None = None) -> Path | None:
    """The log file, or None when logging is off."""
    env = os.environ.get("DK_USAGE_LOG")
    if env is not None:
        return None if env.strip().lower() in OFF else Path(env).expanduser()
    root = Path(cfg.root) if cfg is not None else Path(root or ".")
    main = main_checkout(root)
    conf = getattr(cfg, "usage_log", "") if cfg is not None else ""
    if conf:
        return None if conf.strip().lower() in OFF else (main / Path(conf).expanduser()).resolve()
    return main.parent / DEFAULT_NAME


def shorten_args(argv: list[str], root: Path) -> list[str]:
    """Paths relative to the checkout (or ~), free text and long tokens replaced by their length."""
    home = str(Path.home())
    main = str(main_checkout(root))
    roots = sorted({str(Path(root)), main}, key=len, reverse=True)
    out, text_next = [], False
    for tok in argv:
        if text_next:
            out.append(f"<text {len(tok)}>")
            text_next = False
            continue
        if tok.startswith("-"):
            name, eq, value = tok.partition("=")
            if name in TEXT_OPTIONS:
                out.append(f"{name}=<text {len(value)}>" if eq else name)
                text_next = not eq
                continue
        if "\n" in tok:
            out.append(f"<text {len(tok)}>")
            continue
        for r in roots:
            if tok.startswith(r + "/"):
                tok = tok[len(r) + 1:]
                break
        else:
            if tok.startswith(home + "/"):
                tok = "~" + tok[len(home):]
        if len(tok) > MAX_ARG and "/" in tok and " " not in tok:
            tok = ".../" + "/".join(tok.split("/")[-2:])          # a long path: its last two parts
        out.append(tok if len(tok) <= MAX_ARG else f"<text {len(tok)}>")
    return out


class _Tee:
    """Passes writes through, keeping the last failure-like line (for the log's `msg`)."""

    def __init__(self, stream):
        self._stream = stream
        self.last = ""

    def write(self, text):
        n = self._stream.write(text)
        try:
            if text and ("error" in text or "RE" in text or "usage" in text or "MISMATCH" in text or "not committed" in text):
                for line in text.splitlines():
                    if FAILURE_LINE.match(line):
                        self.last = line.strip()[:200]
        except Exception:
            pass
        return n

    def __getattr__(self, name):
        return getattr(self._stream, name)


def invoke(main, argv: list[str] | None = None) -> int:
    """Run `main(argv)` and append its log line afterwards. Never changes what the command does."""
    argv = sys.argv[1:] if argv is None else argv
    _state.clear()
    start = time.monotonic()
    out, err = sys.stdout, sys.stderr
    tee_out, tee_err = _Tee(out), _Tee(err)
    sys.stdout, sys.stderr = tee_out, tee_err
    code: int | None = None
    error = ""
    message = ""
    try:
        code = main(argv)
        return code
    except SystemExit as e:                      # argparse's usage errors and -h
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 2)
        raise
    except BaseException as e:
        error, message = type(e).__name__, str(e).splitlines()[0][:200] if str(e) else ""
        raise
    finally:
        sys.stdout, sys.stderr = out, err
        try:
            if code not in (None, 0) and not message:
                message = tee_out.last or tee_err.last
            _write(argv, code, error, message, time.monotonic() - start)
        except Exception:
            pass


def _write(argv: list[str], code: int | None, error: str, message: str, secs: float) -> None:
    from .config import REPO
    cfg = _state.get("cfg")
    root = Path(cfg.root) if cfg is not None else REPO
    path = log_path(cfg, root)
    if path is None:
        return
    cmd = _state.get("cmd") or (argv[0] if argv and not argv[0].startswith("-") else "")
    rest = argv[1:] if argv and argv[0] == cmd else argv
    sub = _state.get("sub", "")
    row: dict[str, Any] = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cmd": cmd,
        "sub": sub,
        "flags": sorted({t.split("=", 1)[0] for t in rest if t.startswith("--")}),
        "args": shorten_args(rest, root),
        "worktree": worktree_name(root),
        "exit": code,
        "secs": round(secs, 3),
    }
    if error:
        row["error"] = error
    if message:
        row["msg"] = shorten_args([message], root)[0] if len(message) <= MAX_ARG else _short_message(message, root)
    args = _state.get("args")
    if cfg is not None and args is not None:
        try:
            from .. import platforms
            hook = platforms.hook(cfg.platform, "usage_fields")
            if hook:
                row.update({k: v for k, v in (hook(cfg, args) or {}).items() if v not in (None, "", [])})
        except Exception:
            pass
    row.update(_state.get("fields", {}))
    line = (json.dumps(row, separators=(",", ":"), default=str) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def _short_message(message: str, root: Path) -> str:
    home = str(Path.home())
    for r in (str(root), str(main_checkout(root))):
        message = message.replace(r + "/", "")
    return message.replace(home, "~")[:200]


def refusal_kind(detail: str) -> str:
    """Which guard a commit refusal came from: floor, frozen, quality, or verify (the clean build)."""
    if detail.startswith("below the progress floor"):
        return "floor"
    if "frozen" in detail:
        return "frozen"
    if detail.startswith("forbidden") or "forbidden" in detail[:40]:
        return "quality"
    return "verify"
