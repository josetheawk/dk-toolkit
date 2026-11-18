"""dk doctor: is everything an unattended run needs in place?

Each check is `ok`, `warn` (a feature is off or a cache is stale) or `FAIL` (a run would fail or,
worse, score everything 0 without saying why), with the command that fixes it. Run it before a
night queue; `--model M` also asks the model server for M.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from ...core.command import OK, USAGE, Command, hint
from ...core.config import Config

STALE_DAYS = 14


def checks(cfg: Config, model: str | None, url: str) -> list[tuple[str, str, str, str]]:
    """(status, what, detail, fix) rows."""
    root = Path(cfg.root)
    out = []

    def add(ok: bool, what: str, detail: str, fix: str, bad: str = "FAIL") -> None:
        out.append(("ok" if ok else bad, what, detail, "" if ok else fix))

    tools = root / "build" / "tools"
    for name in ("objdiff-cli", "dtk", "wibo"):
        add((tools / name).is_file(), f"tool {name}", str(tools / name), "./dk setup && ./dk build")
    add(any((root / "build" / "compilers").glob("GC/*/mwcceppc.exe")), "compilers", "build/compilers/GC/*",
        "./dk build (downloads them)")
    add(shutil.which("ninja") is not None, "ninja", shutil.which("ninja") or "not on PATH", "install ninja")
    add((root / "build.ninja").is_file(), "build.ninja", "", "./dk setup")
    orig = root / "orig"
    add(orig.is_dir() and any(orig.rglob("*.dol")), "original game files", "orig/*/sys/main.dol",
        "copy the game's files into orig/ (never committed)")
    m2c = root / cfg.platform_options.get("m2c", ".venv/bin/m2c")
    add(m2c.is_file(), "m2c", str(m2c), "./dk setup (installs requirements.txt)", bad="warn")

    rtti = root / "build" / "dk" / "rtti.json"
    add(rtti.is_file(), "RTTI cache", "build/dk/rtti.json", "./dk analyze rtti", bad="warn")
    from .route import sweep_path
    sweep = sweep_path(cfg)
    if not sweep.is_file():
        add(False, "compiler sweep", f"{sweep} missing: route can't tag compiler-version blockers",
            "./dk compiler-search --sweep", bad="warn")
    else:
        age = (time.time() - sweep.stat().st_mtime) / 86400
        add(age < STALE_DAYS, "compiler sweep", f"{age:.0f} days old", "./dk compiler-search --sweep", bad="warn")

    from .oracle import runner
    add(runner(cfg) is not None, "compiler oracle", "DK_ORACLE / [platform] oracle",
        "set DK_ORACLE (close --moves oracle|directed and regs --reach need it)", bad="warn")

    from .objcache import store
    mismatch = store(cfg) / "MISMATCH.log"
    add(not mismatch.is_file(), "compile cache", str(store(cfg)),
        f"a cached compile differed from a fresh one: read {mismatch}, then delete {store(cfg)}")

    if model:
        from .llmtry import Ollama
        why = Ollama(url).missing(model)
        add(why is None, f"model {model}", why or url, "./np pull the model, or start the model server")
    return out


def run(cfg: Config, args: argparse.Namespace) -> int:
    rows = checks(cfg, args.model, args.url)
    for status, what, detail, fix in rows:
        print(f"  {status:<4} {what:<20} {detail}" + (f"\n       fix: {fix}" if fix else ""))
    failed = [r for r in rows if r[0] == "FAIL"]
    warned = [r for r in rows if r[0] == "warn"]
    print(f"doctor: {len(failed)} failing, {len(warned)} warning(s)")
    hint("status")
    return USAGE if failed else OK


def _configure(p: argparse.ArgumentParser) -> None:
    from .llmtry import DEFAULT_URL
    p.add_argument("--model", help="also check that the model server has this model")
    p.add_argument("--url", default=DEFAULT_URL, help="model server (Ollama)")


COMMAND = Command("doctor", "docs", "check the tools, caches and model an unattended run needs", run, _configure)
