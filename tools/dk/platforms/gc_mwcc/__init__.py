"""GameCube / Wii with Metrowerks CodeWarrior, built through decomp-toolkit."""

from . import acceptcmd, analyze, ctxcmd, diffcmd, gencmd, matchcmds, permute, portcmd, regs, renamecmd, savecmd, symbolcmd, unitcmd
from .firstdiff import first_diff
from .freshness import ensure_split_fresh as prebuild
from .units import declared_sources

COMMANDS = [analyze.COMMAND, unitcmd.COMMAND, matchcmds.CHECK, diffcmd.COMMAND, matchcmds.SEARCH, acceptcmd.COMMAND, ctxcmd.COMMAND, gencmd.COMMAND, permute.COMMAND, regs.COMMAND, portcmd.COMMAND, renamecmd.COMMAND, savecmd.COMMAND, symbolcmd.COMMAND]

__all__ = ["COMMANDS", "declared_sources", "first_diff", "prebuild"]
