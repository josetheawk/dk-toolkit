"""GameCube / Wii with Metrowerks CodeWarrior, built through decomp-toolkit."""

from . import acceptcmd, analyze, ctxcmd, diffcmd, gencmd, matchcmds, portcmd, regs, renamecmd, reorder, savecmd, stalecmd, symbolcmd, unitcmd
from .firstdiff import first_diff
from .freshness import ensure_split_fresh as prebuild
from .units import declared_sources

COMMANDS = [analyze.COMMAND, unitcmd.COMMAND, matchcmds.CHECK, diffcmd.COMMAND, matchcmds.SEARCH, acceptcmd.COMMAND, ctxcmd.COMMAND, gencmd.COMMAND, regs.COMMAND, reorder.COMMAND, portcmd.COMMAND, renamecmd.COMMAND, stalecmd.COMMAND, savecmd.COMMAND, symbolcmd.COMMAND]

__all__ = ["COMMANDS", "declared_sources", "first_diff", "prebuild"]
