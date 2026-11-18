"""GameCube / Wii with Metrowerks CodeWarrior, built through decomp-toolkit."""

from . import acceptcmd, analyze, assets, inlinecost, closecmd, closedata, cardcmd, ctxcmd, diffcmd, doctorcmd, nightcmd, systemscmd, gencmd, inlinescmd, llmagent, llmdata, llmgrind, llmtrain, llmtry, lorafix, matchcmds, portcmd, prunecmd, regions, regs, renamecmd, reorder, route, schedcmd, siblings, trycmd, savecmd, stalecmd, symbolcmd, unitcmd
from .firstdiff import first_diff
from .freshness import after_config_sync, ensure_split_fresh as prebuild
from .matchcmds import link_queue, brief_near
from .gate import frozen_definitions, frozen_keys as frozen_functions
from .units import declared_sources
from .usagehooks import USAGE_LANDING, USAGE_SEARCH, usage_fields

COMMANDS = [analyze.COMMAND, unitcmd.COMMAND, assets.COMMAND, matchcmds.CHECK, diffcmd.COMMAND, matchcmds.SEARCH, acceptcmd.COMMAND, ctxcmd.COMMAND, inlinescmd.COMMAND, siblings.COMMAND, inlinecost.COMMAND, gencmd.COMMAND, regs.COMMAND, regions.COMMAND, closecmd.COMMAND, closedata.COMMAND, reorder.COMMAND, prunecmd.COMMAND, trycmd.COMMAND, llmtry.TRY, lorafix.COMMAND, llmtry.BENCH, llmgrind.GRIND, llmgrind.REVIEW, llmgrind.LAND, llmagent.COMMAND, llmdata.COMMAND, llmtrain.COMMAND, route.COMMAND, schedcmd.COMMAND, portcmd.COMMAND, renamecmd.COMMAND, stalecmd.COMMAND, savecmd.COMMAND, symbolcmd.COMMAND, doctorcmd.COMMAND, nightcmd.COMMAND, cardcmd.COMMAND, systemscmd.COMMAND]

# `dk worktree add` runs these dk commands in a new worktree after it builds (caches other commands read)
WORKTREE_SETUP = (("analyze", "rtti"),)

__all__ = ["COMMANDS", "WORKTREE_SETUP", "after_config_sync", "brief_near", "declared_sources", "first_diff", "frozen_definitions", "frozen_functions", "link_queue", "prebuild", "USAGE_LANDING", "USAGE_SEARCH", "usage_fields"]
