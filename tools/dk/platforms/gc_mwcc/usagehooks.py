"""The usage log's platform part (core/usage.py): fields read from the parsed arguments, and which
commands are search tools and which land work, for `dk usage report`'s hit rates."""

from __future__ import annotations

# tool label -> (command, flags of which one must be given; none: any run)
USAGE_SEARCH = {
    "close": ("close", ()),
    "try": ("try", ()),
    "regs --reach": ("regs", ("--reach",)),
    "regs --why-temp": ("regs", ("--why-temp",)),
    "inlines --scan": ("inlines", ("--scan",)),
}
# commands whose committed run lands functions (they note `landed` and `committed`)
USAGE_LANDING = ("save", "accept")          # llm-land runs them as their own dk runs


def usage_fields(cfg, args) -> dict:
    """unit and function as given on the command line (the commands note resolved names themselves)."""
    out = {}
    unit = getattr(args, "unit", None)
    fn = getattr(args, "function", None)
    target = getattr(args, "target", None)
    if isinstance(unit, str):
        out["unit"] = unit
    if isinstance(fn, str):
        out["function"] = fn
    if getattr(args, "scan", False) and isinstance(getattr(args, "only", None), str) and "," not in args.only:
        out["function"] = args.only
    if isinstance(target, str) and "unit" not in out:
        out["target"] = target
    return out
