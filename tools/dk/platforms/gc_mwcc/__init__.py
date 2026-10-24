"""GameCube / Wii with Metrowerks CodeWarrior, built through decomp-toolkit."""

from . import acceptcmd, analyze, ctxcmd, diffcmd, gencmd, matchcmds, portcmd, renamecmd, savecmd, symbolcmd, unitcmd
from .firstdiff import first_diff
from .freshness import ensure_split_fresh as prebuild

COMMANDS = [analyze.COMMAND, unitcmd.COMMAND, matchcmds.CHECK, diffcmd.COMMAND, matchcmds.SEARCH, acceptcmd.COMMAND, ctxcmd.COMMAND, gencmd.COMMAND, portcmd.COMMAND, renamecmd.COMMAND, savecmd.COMMAND, symbolcmd.COMMAND]

__all__ = ["COMMANDS", "first_diff", "prebuild"]
