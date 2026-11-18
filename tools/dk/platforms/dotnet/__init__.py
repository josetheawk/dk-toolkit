""".NET (CLI) assemblies: metadata, IL and resources, read statically. Nothing is ever executed."""

from . import cmds
from .firstdiff import first_diff

COMMANDS = [cmds.INFO, cmds.STRINGS, cmds.IL, cmds.RESOURCES]

__all__ = ["COMMANDS", "first_diff"]
